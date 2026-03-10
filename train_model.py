"""
PPO + Transformer RL for stock trading with peer attention.

Data CSV (feats.csv): symbol, date, price, <group>.<feature>, ...
Peers identified by running EWM correlation on 60-day price log ratio.

Continuous position sizing in [-1, 1] with squashed Gaussian policy (tanh).

Architecture:
  - Linear input projection → temporal transformer (shared across stocks)
  - Cross-sectional peer attention with correlation bias
  - Separate MLP heads for policy (mu) and value

Launch:
  python train_model.py
  Auto-detects GPUs. Uses DDP when multiple GPUs are available.
  Also works with: torchrun --nproc_per_node=N train_model.py
"""

import argparse
import os
import math
import time
import socket
import threading
import functools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_rank = 0
_world = 1


class Pruned(Exception):
    """Raised by epoch_callback to signal early termination."""
    pass


def _log(msg=""):
    if _rank == 0:
        print(msg, flush=True)


def timed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        _log(f"    {fn.__name__} ({time.perf_counter() - start:.1f}s)")
        return result
    return wrapper


class Config:
    # I/O
    data_path: str = "feats.csv"
    save_dir: str = "checkpoints"

    # Environment
    lookback: int = 60
    episode_length: int = 100
    transaction_cost: float = 0.001
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    n_peers: int = 16
    corr_halflife: int = 60
    corr_column: str = "close.lr_1"

    # Model
    d_model: int = 128
    d_ff: int = 512
    n_heads: int = 4
    n_layers: int = 2
    d_head_hidden: int = 64
    position_dim: int = 16
    dropout: float = 0.1
    peer_dropout: float = 0.5

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_ema_decay: float = 0.99
    policy_clip: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    log_std_init: float = -0.5

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.02
    grad_clip: float = 0.5

    # Training loop
    n_ppo_epochs: int = 2
    batch_size: int = 2048
    inference_batch_size: int = 4096
    n_epochs: int = 200
    warmup_epochs: int = 5
    patience: int = 20
    patience_smoothing: int = 10
    seed: int = 42

    # Runtime
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Diagnostics
    ablation: bool = True

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__class__.__annotations__}


# Vectorized Environment =======================================================

class VecEnv:
    """Batched single-stock trading environment with per-step peer context.

    Observations are 4D: (n_envs, K+1, lookback, n_features) where index 0 is
    the traded stock and indices 1..K are its peers (selected per step).
    """

    def __init__(self, episodes, lookback, transaction_cost,
                 global_features, zero_peers=False):
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        self.n_envs = len(episodes)
        self.stock_features = np.stack([ep["stock_features"] for ep in episodes])
        self.log_returns = np.stack([ep["log_returns"] for ep in episodes])
        self.susp = np.stack([ep["susp"] for ep in episodes])  # (n_envs, chunk_size)
        self.total_steps = self.stock_features.shape[1]

        # Global feature matrix reference (N, T, F) — no per-episode copy
        self.global_features = global_features
        self.zero_peers = zero_peers
        self.date_cols = np.stack(
            [ep["date_cols"] for ep in episodes],
        )  # (n_envs, chunk_size)

        # Per-step peer selection (global sym indices), corr scores, and masks
        self.step_peer_map = np.stack(
            [ep["step_peer_map"] for ep in episodes],
        )
        self.corr_scores = np.stack(
            [ep["corr_scores"] for ep in episodes],
        ).astype(np.float32)
        self.peer_mask = np.stack(
            [ep["peer_mask"] for ep in episodes],
        )

    def _build_observation(self):
        s = self.current_step
        t = min(s - self.lookback, self.step_peer_map.shape[1] - 1)
        lb = self.lookback
        self_window = self.stock_features[:, s - lb : s, :][:, None, :, :]

        if self.zero_peers:
            K = self.step_peer_map.shape[2]
            F = self_window.shape[3]
            peer_windows = np.zeros(
                (self.n_envs, K, lb, F), dtype=np.float32,
            )
        else:
            step_idx = self.step_peer_map[:, t, :]      # (n_envs, K)
            col_window = self.date_cols[:, s - lb : s]   # (n_envs, lb)
            # Gather from global features: (n_envs, K, lb, F)
            sym_exp = step_idx[:, :, None]               # (n_envs, K, 1)
            col_exp = col_window[:, None, :]             # (n_envs, 1, lb)
            peer_windows = self.global_features[sym_exp, col_exp]

        obs = np.concatenate(
            [self_window, peer_windows], axis=1,
        ).astype(np.float32)
        corr = self.corr_scores[:, t, :]
        mask = self.peer_mask[:, t, :]
        return obs, corr, mask

    def reset(self):
        self.current_step = self.lookback
        self.position = np.zeros(self.n_envs, dtype=np.float64)
        obs, corr, mask = self._build_observation()
        return obs, self.position.astype(np.float32), corr, mask

    def step(self, actions):
        new_position = np.clip(actions.astype(np.float64), -1.0, 1.0)
        # Force position hold on suspended dates (no trade possible)
        suspended = self.susp[:, self.current_step]
        new_position = np.where(suspended, self.position, new_position)
        rewards = (
            self.position * self.log_returns[:, self.current_step]
            + np.log1p(-self.transaction_cost * np.abs(new_position - self.position))
        ).astype(np.float32)
        self.position = new_position
        self.current_step += 1
        done = self.current_step >= self.total_steps
        if done:
            return None, rewards, done, self.position.astype(np.float32), None, None
        obs, corr, mask = self._build_observation()
        return obs, rewards, done, self.position.astype(np.float32), corr, mask

    def terminal_observation(self):
        return self._build_observation()


# Reward Normalizer ============================================================

class RewardNormalizer:
    """EMA reward normalizer. Divides by EMA std without mean-shifting."""

    def __init__(self, eps, decay):
        self.eps = eps
        self.decay = decay
        self.mean_sq = 1.0
        self.initialized = False

    def _apply_ema(self, batch_mean_sq):
        if not self.initialized:
            self.mean_sq = batch_mean_sq
            self.initialized = True
        else:
            self.mean_sq = self.decay * self.mean_sq + (1 - self.decay) * batch_mean_sq

    def update(self, rewards):
        flat = rewards.ravel().astype(np.float64)
        if len(flat) > 0:
            self._apply_ema(float((flat ** 2).mean()))

    def update_distributed(self, rewards, device):
        flat = rewards.ravel().astype(np.float64)
        local = torch.tensor(
            [float((flat ** 2).sum()), float(len(flat))],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(local)
        total_sq, total_n = local[0].item(), local[1].item()
        if total_n > 0:
            self._apply_ema(total_sq / total_n)

    def normalize(self, rewards):
        if not self.initialized:
            return rewards
        std = np.sqrt(self.mean_sq + self.eps)
        return (rewards / std).astype(np.float32)

    def state_dict(self):
        return {"mean_sq": self.mean_sq, "initialized": self.initialized}

    def load_state_dict(self, state):
        self.mean_sq = state.get("mean_sq", 1.0)
        self.initialized = state.get("initialized", False)


# Feature Normalization ========================================================

@timed
def compute_feature_stats(train_data):
    """Compute per-feature mean and std from training data using Welford's
    batch-merge algorithm. Iterates one symbol at a time to avoid
    concatenating all rows into a single array.

    Returns (mean, std) as float32 numpy arrays of shape (n_features,).
    """
    count = 0
    mean = None
    m2 = None

    for sd in train_data.values():
        feats = sd["stock_features"]  # (n_rows, n_features), float32
        n = feats.shape[0]
        if n == 0:
            continue
        batch_mean = feats.mean(axis=0).astype(np.float64)
        batch_var = feats.var(axis=0).astype(np.float64)
        batch_m2 = batch_var * n

        if mean is None:
            count = n
            mean = batch_mean
            m2 = batch_m2
        else:
            new_count = count + n
            delta = batch_mean - mean
            mean = mean + delta * (n / new_count)
            m2 = m2 + batch_m2 + delta ** 2 * (count * n / new_count)
            count = new_count

    if mean is None:
        raise ValueError("No training data to compute feature statistics")

    std = np.sqrt(m2 / count)

    zero_std = np.where(std < 1e-8)[0]
    if len(zero_std) > 0:
        raise ValueError(
            f"{len(zero_std)} features have zero variance (indices: "
            f"{zero_std.tolist()}). Check data pipeline."
        )

    return mean.astype(np.float32), std.astype(np.float32)


# Model ========================================================================

class PeerAttention(nn.Module):
    """Cross-sectional attention across peer stocks.

    Uses correlation scores as additive attention bias so that the
    attention weights shift continuously as peer sets evolve.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, corr_scores, peer_mask, need_weights=False):
        """
        x:           (B, K+1, d_model)
        corr_scores: (B, K+1)  — correlation of each slot with query stock
        peer_mask:   (B, K+1)  bool — True for padded/invalid positions
        """
        K1 = x.shape[1]
        bias = self.bias_scale * corr_scores.unsqueeze(1).expand(-1, K1, -1)
        bias = bias.repeat_interleave(self.n_heads, dim=0)

        if peer_mask is not None and peer_mask.any():
            pad_expand = (
                peer_mask
                .unsqueeze(1)
                .expand(-1, K1, -1)
                .repeat_interleave(self.n_heads, dim=0)
            )
            bias = bias.masked_fill(pad_expand, float("-inf"))

        attn_out, attn_weights = self.attn(
            x, x, x, attn_mask=bias, need_weights=need_weights,
            average_attn_weights=True,
        )
        out = self.norm(x + attn_out)
        if need_weights:
            return out, attn_weights
        return out


class PolicyNetwork(nn.Module):
    """Shared-trunk policy network.

    Temporal transformer per stock → peer cross-attention → separate heads.
    """

    def __init__(self, n_features, cfg, feat_mean, feat_std):
        super().__init__()
        self.cfg = cfg

        # Feature normalization (frozen Welford stats from training data)
        self.register_buffer("feat_mean", torch.from_numpy(feat_mean))
        self.register_buffer("feat_std", torch.from_numpy(feat_std))

        # Input projection: simple linear
        self.input_proj = nn.Linear(n_features, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.pos_emb = nn.Parameter(
            torch.randn(1, cfg.lookback, cfg.d_model) * 0.02,
        )

        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, cfg.n_layers)

        # Peer cross-attention
        self.peer_attn = PeerAttention(
            cfg.d_model, cfg.n_heads, cfg.dropout,
        )

        # Position embedding
        self.position_proj = nn.Linear(1, cfg.position_dim)

        # Heads
        head_input_dim = cfg.d_model + cfg.position_dim
        self.policy_head = nn.Sequential(
            nn.Linear(head_input_dim, cfg.d_head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.d_head_hidden, 1),
        )
        self.log_std = nn.Parameter(torch.tensor(cfg.log_std_init))
        self.value_head = nn.Sequential(
            nn.Linear(head_input_dim, cfg.d_head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.d_head_hidden, 1),
        )

    def forward(self, obs, positions, corr_scores, peer_mask):
        """
        obs:         (B, K+1, lookback, F)
        positions:   (B,)
        corr_scores: (B, K+1)
        peer_mask:   (B, K+1)
        """
        B, K1, T, F = obs.shape

        # Normalize input features
        obs = (obs - self.feat_mean) / self.feat_std

        # Stage 1: per-stock temporal encoding (shared transformer)
        flat = obs.reshape(B * K1, T, F)
        hidden = self.input_norm(self.input_proj(flat)) + self.pos_emb
        # Chunk to stay within CUDA kernel grid-dimension limits
        _CHUNK = 32768
        n_seq = hidden.shape[0]
        if n_seq <= _CHUNK:
            pooled = self.transformer(hidden).mean(dim=1)
        else:
            parts = []
            for i in range(0, n_seq, _CHUNK):
                parts.append(self.transformer(hidden[i:i + _CHUNK]).mean(dim=1))
            pooled = torch.cat(parts, dim=0)
        pooled = pooled.view(B, K1, -1)                # (B, K+1, d_model)

        # Stage 2: peer cross-attention with dropout
        if self.training and self.cfg.peer_dropout > 0:
            keep = torch.bernoulli(torch.full(
                (B, 1), 1 - self.cfg.peer_dropout,
                device=obs.device, dtype=obs.dtype,
            ))
            drop_mask = peer_mask.clone()
            drop_mask[:, 1:] = drop_mask[:, 1:] | (keep == 0)
            pooled = self.peer_attn(pooled, corr_scores, drop_mask)
        else:
            pooled = self.peer_attn(pooled, corr_scores, peer_mask)

        # Extract traded stock representation (index 0)
        self_repr = pooled[:, 0, :]  # (B, d_model)

        # Stage 3: position embedding + heads
        pos_emb = self.position_proj(positions.unsqueeze(-1))
        combined = torch.cat([self_repr, pos_emb], dim=-1)

        mu = self.policy_head(combined).squeeze(-1)
        values = self.value_head(combined).squeeze(-1)
        return mu, self.log_std.expand_as(mu), values


# Policy =======================================================================

def _squashed_gaussian_log_prob(u, mu, log_std):
    """Numerically stable log-prob of tanh-squashed Gaussian.

    Uses identity: log(1 - tanh²(u)) = 2(log(2) - u - softplus(-2u))
    """
    gauss_log_prob = (
        -0.5 * (u - mu) ** 2 * torch.exp(-2.0 * log_std)
        - log_std
        - 0.5 * math.log(2 * math.pi)
    )
    squash_correction = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
    return gauss_log_prob - squash_correction


def select_stochastic(model, obs, positions, corr_scores, peer_mask):
    mu, log_std, values = model(obs, positions, corr_scores, peer_mask)
    std = log_std.exp()
    u = mu + std * torch.randn_like(std)
    log_probs = _squashed_gaussian_log_prob(u.float(), mu.float(), log_std.float())
    actions = u.tanh()
    return actions, u, log_probs, values


def select_greedy(model, obs, positions, corr_scores, peer_mask):
    mu, _, values = model(obs, positions, corr_scores, peer_mask)
    return mu.tanh(), values


def evaluate_actions(model, obs, raw_actions, positions, corr_scores, peer_mask):
    mu, log_std, values = model(obs, positions, corr_scores, peer_mask)
    mu, log_std = mu.float(), log_std.float()
    u = raw_actions.float()
    log_probs = _squashed_gaussian_log_prob(u, mu, log_std)
    entropy = -log_probs  # sample-based entropy for squashed distribution
    return log_probs, entropy, values


# Batched GPU Inference ========================================================

def _batched_stochastic(model, obs, pos, corr_scores, peer_mask, cfg):
    n = obs.shape[0]
    actions = np.empty(n, dtype=np.float32)
    raw = np.empty(n, dtype=np.float32)
    logp = np.empty(n, dtype=np.float32)
    vals = np.empty(n, dtype=np.float32)
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
        for i in range(0, n, cfg.inference_batch_size):
            j = min(i + cfg.inference_batch_size, n)
            s = torch.from_numpy(obs[i:j]).to(cfg.device)
            p = torch.from_numpy(pos[i:j]).to(cfg.device)
            c = torch.from_numpy(corr_scores[i:j]).to(cfg.device)
            m = torch.from_numpy(peer_mask[i:j]).to(cfg.device)
            a, r, lp, v = select_stochastic(model, s, p, c, m)
            actions[i:j] = a.cpu().numpy()
            raw[i:j] = r.cpu().numpy()
            logp[i:j] = lp.cpu().numpy()
            vals[i:j] = v.cpu().numpy()
    return actions, raw, logp, vals


def _batched_greedy(model, obs, pos, corr_scores, peer_mask, cfg):
    n = obs.shape[0]
    actions = np.empty(n, dtype=np.float32)
    vals = np.empty(n, dtype=np.float32)
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
        for i in range(0, n, cfg.inference_batch_size):
            j = min(i + cfg.inference_batch_size, n)
            s = torch.from_numpy(obs[i:j]).to(cfg.device)
            p = torch.from_numpy(pos[i:j]).to(cfg.device)
            c = torch.from_numpy(corr_scores[i:j]).to(cfg.device)
            m = torch.from_numpy(peer_mask[i:j]).to(cfg.device)
            a, v = select_greedy(model, s, p, c, m)
            actions[i:j] = a.cpu().numpy()
            vals[i:j] = v.cpu().numpy()
    return actions, vals


# Rollout Buffer ===============================================================

class RolloutBuffer:
    """Pre-allocated PPO rollout buffer with lazy observation reconstruction."""

    def __init__(self, n_episodes, episode_length, lookback):
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.lookback = lookback

        self.raw_actions = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.rewards = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.log_probs = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.values = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.positions = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.final_values = np.empty(n_episodes, dtype=np.float32)

        self.all_stock_features = None
        self.all_date_cols = None
        self.global_features = None
        self.all_step_peer_map = None
        self.all_corr_scores = None
        self.all_peer_masks = None

    def register_episodes(self, episodes, global_features):
        self.all_stock_features = np.stack(
            [ep["stock_features"] for ep in episodes],
        )
        self.global_features = global_features
        self.all_date_cols = np.stack(
            [ep["date_cols"] for ep in episodes],
        )
        self.all_step_peer_map = np.stack(
            [ep["step_peer_map"] for ep in episodes],
        )
        self.all_corr_scores = np.stack(
            [ep["corr_scores"] for ep in episodes],
        ).astype(np.float32)
        self.all_peer_masks = np.stack(
            [ep["peer_mask"] for ep in episodes],
        )

    def compute_gae(self, gamma, gae_lambda, eps, distributed=False, device=None):
        advantages = np.zeros_like(self.rewards)
        running_gae = np.zeros(self.n_episodes, dtype=np.float32)

        for step in reversed(range(self.episode_length)):
            if step == self.episode_length - 1:
                next_value = self.final_values
            else:
                next_value = self.values[:, step + 1]
            td_error = (
                self.rewards[:, step] + gamma * next_value - self.values[:, step]
            )
            running_gae = td_error + gamma * gae_lambda * running_gae
            advantages[:, step] = running_gae

        returns = advantages + self.values

        flat_advantages = advantages.ravel()
        if distributed and device is not None:
            stats = torch.tensor(
                [float(flat_advantages.size),
                 float(flat_advantages.sum()),
                 float((flat_advantages ** 2).sum())],
                device=device, dtype=torch.float64,
            )
            dist.all_reduce(stats)
            n, s, sq = stats[0].item(), stats[1].item(), stats[2].item()
            global_mean = s / n
            global_std = math.sqrt(sq / n - global_mean * global_mean + eps)
            flat_advantages = (flat_advantages - global_mean) / global_std
        else:
            flat_advantages = (
                (flat_advantages - flat_advantages.mean())
                / (flat_advantages.std() + eps)
            )

        self._flat_advantages = flat_advantages
        self._flat_returns = returns.ravel()
        self._flat_raw_actions = self.raw_actions.ravel()
        self._flat_log_probs = self.log_probs.ravel()
        self._flat_positions = self.positions.ravel()

        return self.n_episodes * self.episode_length

    def get_batch(self, flat_indices, device):
        cpu_indices = (
            flat_indices if isinstance(flat_indices, np.ndarray)
            else flat_indices.cpu().numpy()
        )
        episode_ids = cpu_indices // self.episode_length
        step_offsets = cpu_indices % self.episode_length

        lookback = self.lookback
        window_range = np.arange(lookback)

        stock_row_indices = step_offsets[:, None] + window_range[None, :]
        self_batch = self.all_stock_features[
            episode_ids[:, None], stock_row_indices, :
        ]

        # Select per-step peers from global features via step_peer_map
        step_map = self.all_step_peer_map[episode_ids, step_offsets]  # (batch, K)
        # Date columns for the lookback window
        date_col_indices = self.all_date_cols[
            episode_ids[:, None], stock_row_indices,
        ]  # (batch, lb)
        # Gather from global features: (batch, K, lb, F)
        sym_exp = step_map[:, :, None]          # (batch, K, 1)
        col_exp = date_col_indices[:, None, :]  # (batch, 1, lb)
        peer_batch = self.global_features[sym_exp, col_exp]

        observation_batch = np.concatenate(
            [self_batch[:, None, :, :], peer_batch], axis=1,
        )

        return {
            "states": torch.from_numpy(observation_batch).to(
                device, non_blocking=True,
            ),
            "corr_scores": torch.from_numpy(
                self.all_corr_scores[episode_ids, step_offsets],
            ).to(device, non_blocking=True),
            "peer_mask": torch.from_numpy(
                self.all_peer_masks[episode_ids, step_offsets],
            ).to(device, non_blocking=True),
            "raw_actions": torch.tensor(
                self._flat_raw_actions[cpu_indices], dtype=torch.float32, device=device,
            ),
            "log_probs": torch.tensor(
                self._flat_log_probs[cpu_indices], dtype=torch.float32, device=device,
            ),
            "advantages": torch.tensor(
                self._flat_advantages[cpu_indices], dtype=torch.float32, device=device,
            ),
            "returns": torch.tensor(
                self._flat_returns[cpu_indices], dtype=torch.float32, device=device,
            ),
            "positions": torch.tensor(
                self._flat_positions[cpu_indices], dtype=torch.float32, device=device,
            ),
        }


# Data Loading =================================================================

def _read_csv(path, **kwargs):
    return pd.read_csv(path, engine="pyarrow", **kwargs)


@timed
def load_stock_data(path):
    df = _read_csv(path, dtype={"symbol": str})
    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    meta_cols = {"symbol", "date", "price", "delist", "susp"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    symbols = df["symbol"].values
    prices = df["price"].values.astype(np.float64)
    features = df[feature_cols].values.astype(np.float32)
    dates = df["date"].values

    # R writes TRUE/FALSE as strings; map to bool explicitly
    _bool_map = {"TRUE": True, "FALSE": False, True: True, False: False}

    has_delist = "delist" in df.columns
    if has_delist:
        delist_vals = df["delist"].map(_bool_map).fillna(False).values.astype(bool)
    else:
        delist_vals = np.zeros(len(df), dtype=bool)

    has_susp = "susp" in df.columns
    if has_susp:
        susp_vals = df["susp"].map(_bool_map).fillna(False).values.astype(bool)
    else:
        susp_vals = np.zeros(len(df), dtype=bool)

    symbol_breaks = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1

    log_returns = np.zeros(len(prices), dtype=np.float32)
    log_returns[1:] = np.log(prices[1:] / prices[:-1]).astype(np.float32)
    log_returns[symbol_breaks] = 0.0

    starts = np.empty(len(symbol_breaks) + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = symbol_breaks
    ends = np.empty_like(starts)
    ends[:-1] = symbol_breaks
    ends[-1] = len(df)

    data = {}
    for symbol_idx in range(len(starts)):
        start_row, end_row = int(starts[symbol_idx]), int(ends[symbol_idx])
        data[str(symbols[start_row])] = {
            "stock_features": features[start_row:end_row],
            "log_returns": log_returns[start_row:end_row],
            "dates": dates[start_row:end_row].tolist(),
            "prices": prices[start_row:end_row],
            "delist": bool(delist_vals[start_row]),
            "susp": susp_vals[start_row:end_row],
        }
    return data, feature_cols


def _split_stock_data(all_data, train_ratio, val_ratio):
    """Split data into train / val / test by global date cutoffs."""
    all_dates = sorted({d for sd in all_data.values() for d in sd["dates"]})
    n_dates = len(all_dates)
    date_train_end = all_dates[int(n_dates * train_ratio) - 1]
    date_val_end = all_dates[int(n_dates * (train_ratio + val_ratio)) - 1]

    train, val, test = {}, {}, {}
    for sym, sd in all_data.items():
        dates = sd["dates"]
        sliceable_keys = ("stock_features", "log_returns", "dates", "prices", "susp")

        train_mask = [d <= date_train_end for d in dates]
        val_mask = [date_train_end < d <= date_val_end for d in dates]
        test_mask = [d > date_val_end for d in dates]

        for split_dict, mask in [(train, train_mask), (val, val_mask), (test, test_mask)]:
            idx = [i for i, m in enumerate(mask) if m]
            if not idx:
                continue
            s, e = idx[0], idx[-1] + 1
            entry = {k: sd[k][s:e] for k in sliceable_keys}
            entry["delist"] = sd.get("delist", False)
            split_dict[sym] = entry

    return train, val, test


@timed
def build_global_features(all_data, symbols, symbol_to_idx, date_to_col, T):
    """Build dense feature matrix (N, T, F) from all_data.

    Each symbol's features occupy a contiguous slice [first_col, last_col]
    in the time axis.  Columns outside a symbol's range remain zero.

    Also returns per-symbol first/last column arrays (N,) and delist
    flags for peer validity checks during episode generation.
    """
    F = next(iter(all_data.values()))["stock_features"].shape[1]
    N = len(symbols)
    gf = np.zeros((N, T, F), dtype=np.float32)
    first_col = np.full(N, -1, dtype=np.int32)
    last_col = np.full(N, -1, dtype=np.int32)
    delist_flags = np.zeros(N, dtype=bool)

    for sym in symbols:
        sd = all_data.get(sym)
        if sd is None:
            continue
        i = symbol_to_idx[sym]
        fc = date_to_col[sd["dates"][0]]
        lc = date_to_col[sd["dates"][-1]]
        gf[i, fc:lc + 1, :] = sd["stock_features"]
        first_col[i] = fc
        last_col[i] = lc
        delist_flags[i] = sd.get("delist", False)

    return gf, first_col, last_col, delist_flags


# Peer Map =====================================================================

def _extract_top_k(corr, N, K):
    """Extract top-K peer indices and clipped correlation scores per stock."""
    np.fill_diagonal(corr, -2.0)
    np.nan_to_num(corr, nan=-2.0, copy=False)
    if K >= N - 1:
        top_k_idx = np.argsort(-corr, axis=1)[:, :K].astype(np.int32)
    else:
        top_k_idx = np.argpartition(corr, -K, axis=1)[:, -K:]
        row_idx = np.arange(N)[:, None]
        top_k_corr_vals = corr[row_idx, top_k_idx]
        sort_order = np.argsort(-top_k_corr_vals, axis=1)
        top_k_idx = np.take_along_axis(
            top_k_idx, sort_order, axis=1,
        ).astype(np.int32)
    top_k_corr = np.clip(
        corr[np.arange(N)[:, None], top_k_idx], 0.0, 1.0,
    ).astype(np.float32)
    return top_k_idx, top_k_corr


@timed
def compute_peer_maps(all_data, feature_cols, cfg):
    """Precompute daily top-K peer indices using incremental EMA correlation
    on the close.lr_1 feature column.

    Phase 1: Sequential EMA accumulation (inherently sequential).
    Phase 2: Parallel correlation -> top-K extraction via thread pool.
    """
    corr_col_name = cfg.corr_column
    if corr_col_name not in feature_cols:
        raise ValueError(
            f"Correlation column '{corr_col_name}' not found in features. "
            f"Available: {feature_cols[:10]}..."
        )
    corr_col_idx = feature_cols.index(corr_col_name)

    symbols = sorted(all_data.keys())
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    N = len(symbols)

    all_dates = sorted({d for sd in all_data.values() for d in sd["dates"]})
    date_to_col = {d: i for i, d in enumerate(all_dates)}
    T = len(all_dates)

    # Build return matrix (N x T), 0 for suspended/missing dates
    returns = np.zeros((N, T), dtype=np.float64)
    for i, sym in enumerate(symbols):
        sd = all_data[sym]
        for d_idx, d in enumerate(sd["dates"]):
            col = date_to_col[d]
            returns[i, col] = float(sd["stock_features"][d_idx, corr_col_idx])

    halflife = cfg.corr_halflife
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    decay = 1.0 - alpha
    min_history = 2 * halflife
    K = min(cfg.n_peers, N - 1)

    # Running EMA accumulators
    mu = np.zeros(N, dtype=np.float64)       # EMA of returns
    S = np.zeros((N, N), dtype=np.float64)   # EMA of return outer products

    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 1

    n_workers = max(1, n_cores)

    def _corr_top_k_worker(t, mu_t, S_t):
        # Lazily allocate one reusable buffer per thread
        tname = threading.current_thread().name
        if tname not in _thread_bufs:
            _thread_bufs[tname] = np.empty((N, N), dtype=np.float64)
        outer_buf = _thread_bufs[tname]
        np.outer(mu_t, mu_t, out=outer_buf)
        S_t -= outer_buf

        var = np.diag(S_t).copy()
        dead = var < 1e-8
        S_t[dead, :] = 0.0
        S_t[:, dead] = 0.0
        std = np.sqrt(np.maximum(var, 1e-8))
        np.divide(S_t, std[:, None], out=S_t)
        np.divide(S_t, std[None, :], out=S_t)

        top_k_idx, top_k_corr = _extract_top_k(S_t, N, K)
        return t, (top_k_idx, top_k_corr)

    _thread_bufs = {}

    snapshots = {}
    pending = []

    # Pre-allocate buffer for EMA outer product
    ema_outer_buf = np.empty((N, N), dtype=np.float64)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for t in range(T):
            x = returns[:, t]
            mu *= decay
            mu += alpha * x
            S *= decay
            np.outer(x, x, out=ema_outer_buf)
            ema_outer_buf *= alpha
            S += ema_outer_buf

            if t < min_history:
                continue

            # Collect completed futures to free memory
            still_pending = []
            for f in pending:
                if f.done():
                    t_idx, result = f.result()
                    snapshots[t_idx] = result
                else:
                    still_pending.append(f)
            pending = still_pending

            mu_copy, S_copy = mu.copy(), S.copy()
            pending.append(
                pool.submit(_corr_top_k_worker, t, mu_copy, S_copy),
            )

        # Drain remaining futures
        for f in pending:
            t_idx, result = f.result()
            snapshots[t_idx] = result

    checkpoint_arr = np.array(sorted(snapshots.keys()), dtype=np.int64)

    # Pack snapshots into dense arrays for vectorized episode generation
    n_cp = len(checkpoint_arr)
    all_top_k_idx = np.empty((n_cp, N, K), dtype=np.int32)
    all_top_k_corr = np.empty((n_cp, N, K), dtype=np.float32)
    for i, cp in enumerate(checkpoint_arr):
        all_top_k_idx[i], all_top_k_corr[i] = snapshots[cp]
    del snapshots

    return {
        "all_top_k_idx": all_top_k_idx,
        "all_top_k_corr": all_top_k_corr,
        "checkpoint_cols": checkpoint_arr,
        "symbols": symbols,
        "symbol_to_idx": symbol_to_idx,
        "all_dates": all_dates,
        "date_to_col": date_to_col,
        "n_peers": K,
    }


# Episode Generation ===========================================================

def _generate_symbol_episodes(symbol, symbol_data, peer_info, cfg, anchor):
    chunk_size = cfg.lookback + cfg.episode_length
    K = peer_info["n_peers"]
    checkpoint_cols = peer_info["checkpoint_cols"]
    all_top_k_idx = peer_info["all_top_k_idx"]
    all_top_k_corr = peer_info["all_top_k_corr"]
    global_first_col = peer_info["global_first_col"]
    global_last_col = peer_info["global_last_col"]
    global_delist = peer_info["global_delist"]
    lookback = cfg.lookback
    ep_len = cfg.episode_length

    n_rows = len(symbol_data["stock_features"])
    if n_rows < chunk_size:
        return []
    sym_idx = peer_info["symbol_to_idx"].get(symbol)
    if sym_idx is None:
        return []

    # Pre-compute all date column indices for this symbol (once)
    date_to_col = peer_info["date_to_col"]
    all_sym_date_cols = np.array(
        [date_to_col[d] for d in symbol_data["dates"]], dtype=np.int32,
    )

    max_start = n_rows - chunk_size
    episodes = []

    start = anchor
    while start <= max_start:
        date_cols = all_sym_date_cols[start : start + chunk_size]
        episode_first_col = int(date_cols[0])

        # Observation columns for all steps (vectorized)
        obs_cols = date_cols[lookback - 1 : lookback - 1 + ep_len]
        cp_indices = np.searchsorted(
            checkpoint_cols, obs_cols, side="right",
        ).astype(np.intp) - 1

        if np.any(cp_indices < 0):
            start += ep_len
            continue

        # Direct array indexing — no dict lookups or Python loops
        step_peer_idx = all_top_k_idx[cp_indices, sym_idx]    # (ep_len, K)
        step_peer_corr = all_top_k_corr[cp_indices, sym_idx]  # (ep_len, K)

        # --- Vectorized peer validity (ep_len, K) ---
        peer_first = global_first_col[step_peer_idx]
        peer_last = global_last_col[step_peer_idx]
        peer_del = global_delist[step_peer_idx]
        obs_2d = obs_cols[:, None]  # (ep_len, 1) for broadcast

        valid = (
            (step_peer_corr > 0)
            & (peer_first >= 0)
            & (peer_first <= episode_first_col)
            & ~(peer_del & (peer_last < obs_2d))
            & ~(~peer_del & (peer_last < obs_2d)
                & ((obs_2d - peer_last) > lookback))
        )

        step_corr_scores = np.zeros((ep_len, K + 1), dtype=np.float32)
        step_corr_scores[:, 0] = 1.0
        step_corr_scores[:, 1:] = np.where(valid, step_peer_corr, 0.0)

        step_peer_mask = np.ones((ep_len, K + 1), dtype=bool)
        step_peer_mask[:, 0] = False
        step_peer_mask[:, 1:] = ~valid

        episodes.append({
            "symbol": symbol,
            "stock_features": symbol_data["stock_features"][start : start + chunk_size],
            "log_returns": symbol_data["log_returns"][start : start + chunk_size],
            "dates": symbol_data["dates"][start : start + chunk_size],
            "prices": symbol_data["prices"][start : start + chunk_size],
            "susp": symbol_data["susp"][start : start + chunk_size],
            "date_cols": date_cols,
            "step_peer_map": step_peer_idx,
            "corr_scores": step_corr_scores,
            "peer_mask": step_peer_mask,
        })
        start += ep_len

    return episodes


@timed
def generate_episodes(stock_data, peer_info, cfg, epoch_rng):
    """Generate non-overlapping episodes with peer context."""
    chunk_size = cfg.lookback + cfg.episode_length
    episodes = []

    for symbol, symbol_data in stock_data.items():
        n_rows = len(symbol_data["stock_features"])
        if n_rows < chunk_size:
            continue
        max_start = n_rows - chunk_size
        anchor = int(
            epoch_rng.integers(0, min(cfg.episode_length, max_start + 1)),
        )
        episodes.extend(
            _generate_symbol_episodes(symbol, symbol_data, peer_info, cfg, anchor),
        )

    epoch_rng.shuffle(episodes)
    return episodes


# Metrics ======================================================================

def compute_episode_returns(rewards):
    return np.expm1(rewards.sum(axis=1))


def compute_max_drawdown(rewards):
    cum_returns = np.cumsum(rewards, axis=1)
    running_max = np.maximum.accumulate(cum_returns, axis=1)
    drawdowns = 1.0 - np.exp(cum_returns - running_max)
    return float(drawdowns.max(axis=1).mean())


def compute_baseline_rewards(episodes, cfg):
    """Buy-and-hold baseline."""
    log_returns = np.stack([ep["log_returns"] for ep in episodes])
    trading_returns = log_returns[:, cfg.lookback :]
    rewards = np.empty_like(trading_returns)
    rewards[:, 0] = np.log1p(-cfg.transaction_cost)
    rewards[:, 1:] = trading_returns[:, 1:]
    return rewards


# Rollout Collection ===========================================================

@timed
def collect_rollout(model, episodes, cfg, reward_normalizer, distributed,
                             global_features):
    n_episodes = len(episodes)
    episode_length = cfg.episode_length

    rollout = RolloutBuffer(n_episodes, episode_length, cfg.lookback)
    rollout.register_episodes(episodes, global_features)

    env = VecEnv(episodes, cfg.lookback, cfg.transaction_cost, global_features)
    obs, pos, corr, mask = env.reset()
    model.train()
    for step in range(episode_length):
        actions, raw_actions, log_probs, values = _batched_stochastic(
            model, obs, pos, corr, mask, cfg,
        )
        rollout.positions[:, step] = pos
        rollout.raw_actions[:, step] = raw_actions
        obs, rewards, _, pos, corr, mask = env.step(actions)
        rollout.log_probs[:, step] = log_probs
        rollout.values[:, step] = values
        rollout.rewards[:, step] = rewards

    terminal_obs, terminal_corr, terminal_mask = env.terminal_observation()
    _, terminal_values = _batched_greedy(
        model, terminal_obs, pos, terminal_corr, terminal_mask, cfg,
    )
    rollout.final_values[:] = terminal_values

    baseline_rewards = compute_baseline_rewards(episodes, cfg)
    model_return = float(compute_episode_returns(rollout.rewards).mean())
    baseline_return = float(compute_episode_returns(baseline_rewards).mean())

    if distributed:
        reward_normalizer.update_distributed(rollout.rewards, cfg.device)
    else:
        reward_normalizer.update(rollout.rewards)
    rollout.rewards = reward_normalizer.normalize(rollout.rewards)

    return rollout, model_return, baseline_return


# Evaluation ===================================================================

def evaluate_deterministic(model, episodes, cfg, global_features,
                           zero_peers=False):
    n_episodes = len(episodes)
    episode_length = cfg.episode_length
    all_rewards = np.empty((n_episodes, episode_length), dtype=np.float32)
    all_positions = np.empty((n_episodes, episode_length), dtype=np.float32)

    env = VecEnv(episodes, cfg.lookback, cfg.transaction_cost,
                 global_features, zero_peers=zero_peers)
    obs, pos, corr, mask = env.reset()
    model.eval()
    for step in range(episode_length):
        actions, _ = _batched_greedy(model, obs, pos, corr, mask, cfg)
        obs, rewards, _, pos, corr, mask = env.step(actions)
        all_rewards[:, step] = rewards
        all_positions[:, step] = pos

    return all_rewards, all_positions


@timed
def evaluate_episodes(model, episodes, cfg, global_features):
    if len(episodes) == 0:
        return {
            "model_return": 0.0, "baseline_return": 0.0, "beat_rate": 0.0,
            "max_drawdown": 0.0, "mean_turnover": 0.0,
            "position_histogram": np.zeros(50, dtype=np.float64),
            "pos_sum": 0.0, "pos_sum_sq": 0.0, "n_actions": 0,
            "model_rewards": np.empty((0, cfg.episode_length)),
            "baseline_rewards": np.empty((0, cfg.episode_length)),
            "model_positions": np.empty((0, cfg.episode_length), dtype=np.float32),
        }

    model_rewards, model_positions = evaluate_deterministic(
        model, episodes, cfg, global_features,
    )
    baseline_rewards = compute_baseline_rewards(episodes, cfg)
    model_returns = compute_episode_returns(model_rewards)
    baseline_returns = compute_episode_returns(baseline_rewards)

    pos_flat = model_positions.ravel().astype(np.float64)
    pos_hist, _ = np.histogram(
        pos_flat, bins=50, range=(-1.0, 1.0),
    )

    turnover = float(np.abs(np.diff(model_positions, axis=1)).sum(axis=1).mean())

    return {
        "model_return": float(model_returns.mean()),
        "baseline_return": float(baseline_returns.mean()),
        "beat_rate": float(np.mean(model_returns > baseline_returns)),
        "max_drawdown": compute_max_drawdown(model_rewards),
        "mean_turnover": turnover,
        "position_histogram": pos_hist.astype(np.float64),
        "pos_sum": float(pos_flat.sum()),
        "pos_sum_sq": float((pos_flat ** 2).sum()),
        "n_actions": int(model_positions.size),
        "model_rewards": model_rewards,
        "baseline_rewards": baseline_rewards,
        "model_positions": model_positions,
    }


@timed
def evaluate_ablated(model, episodes, cfg, global_features):
    """Evaluate with peers or stock features zeroed."""
    if len(episodes) == 0:
        return {"no_peers": 0.0, "no_stock": 0.0}

    no_peer_episodes = []
    for ep in episodes:
        ablated_corr = np.zeros_like(ep["corr_scores"])
        ablated_corr[:, 0] = 1.0
        ablated_mask = np.ones_like(ep["peer_mask"])
        ablated_mask[:, 0] = False
        no_peer_episodes.append({
            **ep,
            "corr_scores": ablated_corr,
            "peer_mask": ablated_mask,
        })
    no_peer_rewards, _ = evaluate_deterministic(
        model, no_peer_episodes, cfg, global_features, zero_peers=True,
    )

    no_stock_episodes = [
        {**ep, "stock_features": np.zeros_like(ep["stock_features"])}
        for ep in episodes
    ]
    no_stock_rewards, _ = evaluate_deterministic(
        model, no_stock_episodes, cfg, global_features,
    )

    return {
        "no_peers": float(compute_episode_returns(no_peer_rewards).mean()),
        "no_stock": float(compute_episode_returns(no_stock_rewards).mean()),
    }


def diagnose_peer_attention(model, batch):
    """Extract peer attention self-weight and entropy from a small batch."""
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        states = batch["states"].float()
        corr_scores = batch["corr_scores"]
        peer_mask = batch["peer_mask"]

        B, K1, T, F = states.shape
        flat = states.reshape(B * K1, T, F)
        hidden = base_model.input_norm(base_model.input_proj(flat)) + base_model.pos_emb
        pooled = base_model.transformer(hidden).mean(dim=1).view(B, K1, -1)

        _, attn_weights = base_model.peer_attn(
            pooled, corr_scores, peer_mask, need_weights=True,
        )

        self_weight = float(attn_weights[:, 0, 0].mean().item())
        self_query_weights = attn_weights[:, 0, :]
        eps = 1e-8
        log_w = torch.log(self_query_weights.clamp(min=eps))
        attn_entropy = float(-(self_query_weights * log_w).sum(dim=-1).mean().item())

    return {"self_weight": self_weight, "attn_entropy": attn_entropy}


# PPO Update ===================================================================

@timed
def ppo_update(model, optimizer, scaler, rollout, cfg,
               n_local_transitions, n_padded_transitions, entropy_coeff):
    model.train()
    base_model = model.module if hasattr(model, "module") else model
    policy_loss_sum = value_loss_sum = 0.0
    clip_fraction_sum = approx_kl_sum = 0.0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    skipped_batches = 0
    n_batches = 0
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")

    permutation = torch.randperm(n_padded_transitions, device=cfg.device)
    if n_local_transitions < n_padded_transitions:
        valid_mask = permutation < n_local_transitions
        permutation = permutation % n_local_transitions
    else:
        valid_mask = None

    for batch_start in range(0, n_padded_transitions, cfg.batch_size):
        batch_end = min(batch_start + cfg.batch_size, n_padded_transitions)
        batch_indices = permutation[batch_start:batch_end]
        batch = rollout.get_batch(batch_indices, cfg.device)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            new_log_probs, entropy, new_values = evaluate_actions(
                model, batch["states"], batch["raw_actions"], batch["positions"],
                batch["corr_scores"], batch["peer_mask"],
            )
            log_ratio = new_log_probs - batch["log_probs"]
            ratio = torch.exp(log_ratio)
            advantages = batch["advantages"]

            surrogate = torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - cfg.policy_clip, 1 + cfg.policy_clip) * advantages,
            )

            if valid_mask is not None:
                batch_weight = valid_mask[batch_start:batch_end].float()
                n_valid = batch_weight.sum().clamp(min=1)
                policy_loss = -(surrogate * batch_weight).sum() / n_valid
                value_loss = (
                    F.mse_loss(new_values, batch["returns"], reduction="none")
                    * batch_weight
                ).sum() / n_valid
                entropy_term = (entropy * batch_weight).sum() / n_valid
                clipped = ((ratio - 1.0).abs() > cfg.policy_clip).float()
                clip_frac = (clipped * batch_weight).sum() / n_valid
                approx_kl_batch = ((log_ratio.exp() - 1.0 - log_ratio) * batch_weight).sum() / n_valid
            else:
                policy_loss = -surrogate.mean()
                value_loss = F.mse_loss(new_values, batch["returns"])
                entropy_term = entropy.mean()
                clipped = ((ratio - 1.0).abs() > cfg.policy_clip).float()
                clip_frac = clipped.mean()
                approx_kl_batch = (log_ratio.exp() - 1.0 - log_ratio).mean()

            loss = policy_loss + cfg.value_coeff * value_loss - entropy_coeff * entropy_term

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        combined_norm = nn.utils.clip_grad_norm_(
            base_model.parameters(), cfg.grad_clip,
        ).item()

        if not math.isfinite(combined_norm):
            skipped_batches += 1

        scaler.step(optimizer)
        scaler.update()

        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        clip_fraction_sum += clip_frac.item()
        approx_kl_sum += approx_kl_batch.item()
        if math.isfinite(combined_norm):
            grad_norm_sum += combined_norm
            grad_norm_count += 1
        n_batches += 1

    if _world > 1:
        sync_tensor = torch.tensor(
            [policy_loss_sum, value_loss_sum,
             clip_fraction_sum, approx_kl_sum, grad_norm_sum,
             float(n_batches), float(skipped_batches), float(grad_norm_count)],
            device=cfg.device,
        )
        dist.all_reduce(sync_tensor)
        policy_loss_sum = sync_tensor[0].item()
        value_loss_sum = sync_tensor[1].item()
        clip_fraction_sum = sync_tensor[2].item()
        approx_kl_sum = sync_tensor[3].item()
        grad_norm_sum = sync_tensor[4].item()
        n_batches = int(sync_tensor[5].item())
        skipped_batches = int(sync_tensor[6].item())
        grad_norm_count = int(sync_tensor[7].item())

    n_batches = max(n_batches, 1)
    return {
        "policy_loss": policy_loss_sum / n_batches,
        "value_loss": value_loss_sum / n_batches,
        "clip_fraction": clip_fraction_sum / n_batches,
        "approx_kl": approx_kl_sum / n_batches,
        "grad_norm": grad_norm_sum / max(grad_norm_count, 1),
        "skip_rate": skipped_batches / n_batches,
    }


# Test Results =================================================================

def build_test_results(episodes, eval_result, cfg):
    rows = []
    for episode_idx, episode in enumerate(episodes):
        dates = episode["dates"][cfg.lookback :]
        prices = episode["prices"][cfg.lookback :]
        for step in range(cfg.episode_length):
            rows.append({
                "symbol": episode["symbol"],
                "date": dates[step],
                "price": prices[step],
                "position": float(eval_result["model_positions"][episode_idx, step]),
                "model_reward": float(eval_result["model_rewards"][episode_idx, step]),
                "baseline_reward": float(eval_result["baseline_rewards"][episode_idx, step]),
            })
    return rows


# DDP Helpers ==================================================================

def _allreduce_means(*means, local_count, device):
    tensor = torch.tensor(
        [m * local_count for m in means] + [float(local_count)],
        device=device, dtype=torch.float64,
    )
    dist.all_reduce(tensor)
    total = tensor[-1].item()
    if total < 1:
        return means
    return tuple(float(tensor[i].item() / total) for i in range(len(means)))


def _allreduce_eval(result, local_count, device):
    scalar_fields = [
        result["model_return"] * local_count,
        result["baseline_return"] * local_count,
        result["beat_rate"] * local_count,
        result["max_drawdown"] * local_count,
        result["mean_turnover"] * local_count,
        float(local_count),
        result["pos_sum"],
        result["pos_sum_sq"],
        float(result["n_actions"]),
    ]
    tensor = torch.tensor(scalar_fields, device=device, dtype=torch.float64)
    dist.all_reduce(tensor)
    total = tensor[5].item()
    if total < 1:
        return result

    hist_tensor = torch.tensor(
        result["position_histogram"], device=device, dtype=torch.float64,
    )
    dist.all_reduce(hist_tensor)

    return {
        "model_return": float(tensor[0].item() / total),
        "baseline_return": float(tensor[1].item() / total),
        "beat_rate": float(tensor[2].item() / total),
        "max_drawdown": float(tensor[3].item() / total),
        "mean_turnover": float(tensor[4].item() / total),
        "position_histogram": hist_tensor.cpu().numpy(),
        "pos_sum": tensor[6].item(),
        "pos_sum_sq": tensor[7].item(),
        "n_actions": int(tensor[8].item()),
    }


def _allreduce_ablation(result, local_count, device):
    tensor = torch.tensor([
        result["no_peers"] * local_count,
        result["no_stock"] * local_count,
        float(local_count),
    ], device=device, dtype=torch.float64)
    dist.all_reduce(tensor)
    total = tensor[2].item()
    if total < 1:
        return result
    return {
        "no_peers": float(tensor[0].item() / total),
        "no_stock": float(tensor[1].item() / total),
    }


def _allreduce_max_int(value, device):
    tensor = torch.tensor([value], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _gather_test_arrays(test_result, n_local, n_total, episode_length, device):
    local = torch.zeros(n_local, episode_length, 3, device=device)
    local[..., 0] = torch.from_numpy(test_result["model_rewards"]).to(device)
    local[..., 1] = torch.from_numpy(test_result["baseline_rewards"]).to(device)
    local[..., 2] = torch.from_numpy(test_result["model_positions"]).to(device)

    max_local = (n_total + _world - 1) // _world
    padded = torch.zeros(max_local, episode_length, 3, device=device)
    padded[:n_local] = local

    gathered = [torch.zeros_like(padded) for _ in range(_world)]
    dist.all_gather(gathered, padded)

    if _rank != 0:
        return None

    result = torch.zeros(n_total, episode_length, 3, device=device)
    for rank_idx in range(_world):
        rank_episodes = list(range(rank_idx, n_total, _world))
        result[rank_episodes] = gathered[rank_idx][:len(rank_episodes)]

    result_np = result.cpu().numpy()
    return {
        "model_rewards": result_np[..., 0],
        "baseline_rewards": result_np[..., 1],
        "model_positions": result_np[..., 2],
    }


# Plotting =====================================================================

def plot_training(history, path, cfg):
    """Plot training curves on a 4x3 grid."""
    from matplotlib.ticker import MaxNLocator

    epochs = range(1, len(history["policy_loss"]) + 1)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    # (0,0) Episode returns + max drawdown
    ablation_epochs = history.get("ablation_epochs", [])
    axes[0, 0].plot(epochs, history["train_return"], color="b", label="Train")
    axes[0, 0].plot(epochs, history["train_baseline"], color="b", linestyle="--", alpha=0.5, label="Train base")
    axes[0, 0].plot(epochs, history["val_return"], color="r", label="Val")
    axes[0, 0].plot(epochs, history["val_baseline"], color="r", linestyle="--", alpha=0.5, label="Val base")
    if ablation_epochs:
        axes[0, 0].plot(ablation_epochs, history["val_no_peers"], color="#ffaa00", label="No peers")
        axes[0, 0].plot(ablation_epochs, history["val_no_stock"], color="#ff00aa", label="No stock")
    if "val_max_drawdown" in history:
        axes[0, 0].plot(epochs, history["val_max_drawdown"], color="k", label="Max DD")
    axes[0, 0].set_title("Episode Return")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) Beat rate
    axes[0, 1].plot(epochs, history["val_beat_rate"])
    axes[0, 1].set_title("Val Beat Rate")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)

    # (0,2) Val turnover
    if "val_turnover" in history:
        axes[0, 2].plot(epochs, history["val_turnover"])
    axes[0, 2].set_title("Val Turnover")
    axes[0, 2].grid(True, alpha=0.3)

    # (1,0) Position heatmap
    pos_hists = history.get("position_histogram", [])
    if pos_hists:
        heatmap_data = np.array(pos_hists).T
        col_sums = heatmap_data.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums > 0, col_sums, 1.0)
        heatmap_data = heatmap_data / col_sums
        im = axes[1, 0].imshow(
            heatmap_data, aspect="auto", origin="lower",
            extent=[1, len(pos_hists), -1, 1], interpolation="nearest",
        )
        fig.colorbar(im, ax=axes[1, 0], label="Frequency")
    axes[1, 0].set_title("Val Position")

    # (1,1) Policy loss
    axes[1, 1].plot(epochs, history["policy_loss"])
    axes[1, 1].set_title("Policy Loss")
    axes[1, 1].grid(True, alpha=0.3)

    # (1,2) Value loss
    axes[1, 2].plot(epochs, history["value_loss"])
    axes[1, 2].set_title("Value Loss")
    axes[1, 2].grid(True, alpha=0.3)

    # (2,0) Clip fraction / Approx KL (dual axis)
    ax_clip = axes[2, 0]
    ax_clip.plot(epochs, history["clip_fraction"], color="b", label="Clip frac")
    ax_clip.set_ylabel("Clip Fraction", color="b")
    ax_clip.tick_params(axis="y", labelcolor="b")
    ax_kl = ax_clip.twinx()
    ax_kl.plot(epochs, history["approx_kl"], color="r", label="Approx KL")
    ax_kl.set_ylabel("Approx KL", color="r")
    ax_kl.tick_params(axis="y", labelcolor="r")
    ax_clip.set_title("Clip Fraction / Approx KL")
    ax_clip.grid(True, alpha=0.3)

    # (2,1) Grad norm
    axes[2, 1].plot(epochs, history["grad_norm"])
    axes[2, 1].set_title("Grad Norm")
    axes[2, 1].grid(True, alpha=0.3)

    # (2,2) Skip rate
    if "skip_rate" in history:
        axes[2, 2].plot(epochs, history["skip_rate"])
    axes[2, 2].set_title("Skip Rate")
    axes[2, 2].grid(True, alpha=0.3)

    # (3,0) Log std
    axes[3, 0].plot(epochs, history["log_std"])
    axes[3, 0].set_title("Log Std")
    axes[3, 0].grid(True, alpha=0.3)

    # (3,1) Bias scale
    if "bias_scale" in history:
        axes[3, 1].plot(epochs, history["bias_scale"])
    axes[3, 1].set_title("Bias Scale")
    axes[3, 1].grid(True, alpha=0.3)

    # (3,2) Peer attention diagnostics (dual axis)
    has_peer = len(history.get("self_weight", [])) > 0
    if has_peer:
        peer_epochs = range(1, len(history["self_weight"]) + 1)
        ax_peer = axes[3, 2]
        ax_peer.plot(peer_epochs, history["self_weight"], color="b", label="Self-weight")
        ax_peer.set_ylabel("Self-weight", color="b")
        ax_peer.tick_params(axis="y", labelcolor="b")
        ax_twin = ax_peer.twinx()
        ax_twin.plot(peer_epochs, history["attn_entropy"], color="r", label="Entropy")
        ax_twin.set_ylabel("Attn Entropy", color="r")
        ax_twin.tick_params(axis="y", labelcolor="r")
        ax_peer.set_title("Peer Attention")
        ax_peer.grid(True, alpha=0.3)
    else:
        axes[3, 2].set_visible(False)

    for ax in axes.flat:
        if ax.get_visible():
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Training Loop ================================================================


def warmup_cosine_lr(base_lr, epoch, n_epochs, warmup_epochs):
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

def _load_all_data(cfg):
    _log()
    _log("[Data]")
    all_data, feature_cols = load_stock_data(cfg.data_path)
    train_data, val_data, test_data = _split_stock_data(
        all_data, cfg.train_ratio, cfg.val_ratio,
    )

    def _date_range(d):
        dates = sorted({dt for sd in d.values() for dt in sd["dates"]})
        return (dates[0], dates[-1]) if dates else (None, None)

    train_range = _date_range(train_data)
    val_range = _date_range(val_data)
    test_range = _date_range(test_data)

    peer_info = compute_peer_maps(all_data, feature_cols, cfg)

    # Build global forward-filled feature matrix for peer lookups
    gf, gf_first_col, gf_last_col, gf_delist = build_global_features(
        all_data, peer_info["symbols"], peer_info["symbol_to_idx"],
        peer_info["date_to_col"], len(peer_info["all_dates"]),
    )
    peer_info["global_features"] = gf
    peer_info["global_first_col"] = gf_first_col
    peer_info["global_last_col"] = gf_last_col
    peer_info["global_delist"] = gf_delist

    peer_info_path = os.path.join(cfg.save_dir, "peer_info.pt")
    if not os.path.exists(peer_info_path):
        torch.save(peer_info, peer_info_path)
        _log(f"    {'Saved file':<20s}: {peer_info_path}")

    _log(f"    {'Symbols':<20s}: {len(all_data)}")
    _log(f"    {'Train symbols':<20s}: {len(train_data)}")
    _log(f"    {'Val symbols':<20s}: {len(val_data)}")
    _log(f"    {'Test symbols':<20s}: {len(test_data)}")
    _log(f"    {'Train dates':<20s}: {train_range[0]} .. {train_range[1]}")
    _log(f"    {'Val dates':<20s}: {val_range[0]} .. {val_range[1]}")
    _log(f"    {'Test dates':<20s}: {test_range[0]} .. {test_range[1]}")
    _log(f"    {'Stock features':<20s}: {len(feature_cols)}")
    _log(f"    {'Peer snapshots':<20s}: {len(peer_info['checkpoint_cols'])}")
    _log(f"    {'Global features':<20s}: {gf.shape}  ({gf.nbytes / 1e9:.2f} GB)")
    _log(f"    {'Train observations':<20s}: {sum(len(d['stock_features']) for d in train_data.values()):,}")
    _log(f"    {'Val observations':<20s}: {sum(len(d['stock_features']) for d in val_data.values()):,}")
    _log(f"    {'Test observations':<20s}: {sum(len(d['stock_features']) for d in test_data.values()):,}")

    return peer_info, train_data, val_data, test_data, feature_cols


def train(cfg, preprocess_path=None, epoch_callback=None):
    global _rank, _world

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        _rank = dist.get_rank()
        _world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        cfg.device = f"cuda:{local_rank}"
    is_main = _rank == 0

    os.makedirs(cfg.save_dir, exist_ok=True)

    _log()
    _log("[Configuration]")
    for key, value in cfg.to_dict().items():
        _log(f"    {key:<20s}: {value}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if preprocess_path:
        _log()
        _log("[Data]")
        _log("    Loading preprocessed data...")
        preprocess = torch.load(preprocess_path, map_location="cpu", weights_only=False)
        peer_info = preprocess["peer_info"]
        train_data = preprocess["train_data"]
        val_data = preprocess["val_data"]
        test_data = preprocess["test_data"]
        feature_cols = preprocess["feature_cols"]
        feat_mean = preprocess["feat_mean"]
        feat_std = preprocess["feat_std"]
        del preprocess
    else:
        peer_info, train_data, val_data, test_data, feature_cols = _load_all_data(cfg)
        feat_mean, feat_std = compute_feature_stats(train_data)
    n_features = len(feature_cols)
    global_features = peer_info["global_features"]

    _log()
    _log("[Model]")
    base_model = PolicyNetwork(n_features, cfg, feat_mean, feat_std).to(cfg.device)
    model = base_model
    if distributed:
        model = DDP(base_model, device_ids=[int(os.environ["LOCAL_RANK"])])
        torch.manual_seed(cfg.seed + _rank)

    n_params = sum(p.numel() for p in base_model.parameters())
    _log(f"    {'Parameters':<20s}: {n_params:,}")

    effective_lr = cfg.lr * math.sqrt(_world)

    decay_params, no_decay_params = [], []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        is_norm_or_bias = param.dim() < 2 or "norm" in name
        if is_norm_or_bias:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=effective_lr, eps=1e-5,
    )
    reward_normalizer = RewardNormalizer(1e-8, cfg.reward_ema_decay)
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    device_label = f"{_world} GPUs (DDP)" if distributed else cfg.device
    _log(f"    {'Device':<20s}: {device_label}")
    _log(f"    {'Mixed precision':<20s}: {'float16' if amp_enabled else 'disabled'}")
    if _world > 1:
        _log(f"    {'Scaled LR':<20s}: {effective_lr:.1e}")

    history = defaultdict(list)
    best_score = -float("inf")
    patience_counter = 0
    start_epoch = 1

    latest_path = os.path.join(cfg.save_dir, "model_latest.pt")
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=cfg.device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])
        saved_groups = ckpt["optimizer_state_dict"].get("param_groups", [])
        if len(saved_groups) == len(optimizer.param_groups):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            _log("    Optimizer param groups changed, resetting optimizer state")
        start_epoch = ckpt["epoch"] + 1
        patience_counter = ckpt.get("patience_counter", 0)
        if "reward_normalizer" in ckpt:
            reward_normalizer.load_state_dict(ckpt["reward_normalizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        for key, vals in ckpt.get("history", {}).items():
            history[key] = list(vals)

        val_returns = history.get("val_return", [])
        if val_returns:
            window = cfg.patience_smoothing
            best_score = max(
                float(np.mean(val_returns[max(0, i + 1 - window) : i + 1]))
                for i in range(len(val_returns))
            )
        else:
            best_score = ckpt.get("best_score", -float("inf"))

        _log()
        _log(
            f"Resumed from epoch {ckpt['epoch']}"
            f" (best score {best_score:.4f},"
            f" patience {patience_counter})"
        )

    def _build_checkpoint():
        return {
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "feature_cols": feature_cols,
            "n_features": n_features,
            "epoch": epoch,
            "best_score": best_score,
            "patience_counter": patience_counter,
            "reward_normalizer": reward_normalizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": dict(history),
        }

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        current_lr = warmup_cosine_lr(
            effective_lr, epoch, cfg.n_epochs, cfg.warmup_epochs,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        _log()
        _log(f"[Epoch {epoch}/{cfg.n_epochs}]")

        epoch_rng = np.random.default_rng(cfg.seed + epoch * 7919)
        train_episodes = generate_episodes(train_data, peer_info, cfg, epoch_rng)
        rank_train_episodes = train_episodes[_rank::_world]

        val_rng = np.random.default_rng(cfg.seed + epoch * 7919 + 1)
        val_episodes = generate_episodes(val_data, peer_info, cfg, val_rng)
        rank_val_episodes = val_episodes[_rank::_world]

        rollout, model_return, baseline_return = collect_rollout(
            model, rank_train_episodes, cfg, reward_normalizer, distributed,
            global_features,
        )

        if distributed:
            model_return, baseline_return = _allreduce_means(
                model_return, baseline_return,
                local_count=len(rank_train_episodes), device=cfg.device,
            )

        n_local_transitions = rollout.compute_gae(
            cfg.gamma, cfg.gae_lambda, 1e-8, distributed, cfg.device,
        )
        if _world > 1:
            max_transitions = _allreduce_max_int(n_local_transitions, cfg.device)
        else:
            max_transitions = n_local_transitions
        n_padded_transitions = (
            math.ceil(max_transitions / cfg.batch_size) * cfg.batch_size
        )

        # Peer attention diagnostics (cheap, run every epoch)
        peer_diag = None
        diag_n = min(cfg.batch_size, n_local_transitions)
        diag_indices = np.random.choice(n_local_transitions, diag_n, replace=False)
        diag_batch = rollout.get_batch(diag_indices, cfg.device)
        peer_diag = diagnose_peer_attention(model, diag_batch)

        epoch_losses = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "clip_fraction": 0.0, "approx_kl": 0.0, "grad_norm": 0.0,
            "skip_rate": 0.0,
        }
        for _ in range(cfg.n_ppo_epochs):
            losses = ppo_update(
                model, optimizer, scaler, rollout, cfg,
                n_local_transitions, n_padded_transitions, cfg.entropy_coeff,
            )
            for k in epoch_losses:
                epoch_losses[k] += losses[k]
        for k in epoch_losses:
            epoch_losses[k] /= cfg.n_ppo_epochs

        current_log_std = base_model.log_std.item()
        bias_scale = base_model.peer_attn.bias_scale.item()

        del rollout
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

        val_result = evaluate_episodes(model, rank_val_episodes, cfg, global_features)
        if distributed:
            val_aggregated = _allreduce_eval(
                val_result, len(rank_val_episodes), cfg.device,
            )
        else:
            val_aggregated = val_result

        if cfg.ablation:
            ablation_result = evaluate_ablated(model, rank_val_episodes, cfg, global_features)
            if distributed:
                ablation_aggregated = _allreduce_ablation(
                    ablation_result, len(rank_val_episodes), cfg.device,
                )
            else:
                ablation_aggregated = ablation_result
            history["ablation_epochs"].append(epoch)
            history["val_no_peers"].append(ablation_aggregated["no_peers"])
            history["val_no_stock"].append(ablation_aggregated["no_stock"])

        # Compute position stats
        n_actions = val_aggregated.get("n_actions", 0)
        if n_actions > 0:
            pos_mean = val_aggregated["pos_sum"] / n_actions
            pos_std = math.sqrt(max(0, val_aggregated["pos_sum_sq"] / n_actions - pos_mean ** 2))
        else:
            pos_mean, pos_std = 0.0, 0.0

        for key, value in [
            ("train_return", model_return),
            ("train_baseline", baseline_return),
            ("val_return", val_aggregated["model_return"]),
            ("val_baseline", val_aggregated["baseline_return"]),
            ("val_beat_rate", val_aggregated["beat_rate"]),
            ("val_max_drawdown", val_aggregated["max_drawdown"]),
            ("val_turnover", val_aggregated["mean_turnover"]),
            ("policy_loss", epoch_losses["policy_loss"]),
            ("value_loss", epoch_losses["value_loss"]),
            ("clip_fraction", epoch_losses["clip_fraction"]),
            ("approx_kl", epoch_losses["approx_kl"]),
            ("grad_norm", epoch_losses["grad_norm"]),
            ("skip_rate", epoch_losses["skip_rate"]),
            ("log_std", current_log_std),
            ("bias_scale", bias_scale),
        ]:
            history[key].append(value)

        history["position_histogram"].append(
            val_aggregated["position_histogram"]
            if "position_histogram" in val_aggregated
            else np.zeros(50, dtype=np.float64)
        )

        if peer_diag is not None:
            history["self_weight"].append(peer_diag["self_weight"])
            history["attn_entropy"].append(peer_diag["attn_entropy"])

        window = cfg.patience_smoothing
        recent_returns = history["val_return"][-window:]
        current_score = float(np.mean(recent_returns))

        improved = current_score > best_score
        if improved:
            best_score = current_score
            patience_counter = 0
        elif epoch > cfg.warmup_epochs:
            patience_counter += 1

        _log(f"    {'Train episodes':<20s}: {len(train_episodes)}")
        _log(f"    {'Val episodes':<20s}: {len(val_episodes)}")
        _log(f"    {'Train return':<20s}: {model_return:.4f}")
        _log(f"    {'Train baseline':<20s}: {baseline_return:.4f}")
        _log(f"    {'Val return':<20s}: {val_aggregated['model_return']:.4f}")
        _log(f"    {'Val baseline':<20s}: {val_aggregated['baseline_return']:.4f}")
        if cfg.ablation:
            _log(f"    {'No-peers return':<20s}: {ablation_aggregated['no_peers']:.4f}")
            _log(f"    {'No-stock return':<20s}: {ablation_aggregated['no_stock']:.4f}")
        _log(f"    {'Val max drawdown':<20s}: {val_aggregated['max_drawdown']:.4f}")
        _log(f"    {'Val beat rate':<20s}: {val_aggregated['beat_rate']:.4f}")
        _log(f"    {'Val turnover':<20s}: {val_aggregated['mean_turnover']:.4f}")
        _log(f"    {'Val position mean':<20s}: {pos_mean:.4f}")
        _log(f"    {'Val position std':<20s}: {pos_std:.4f}")
        _log(f"    {'Policy loss':<20s}: {epoch_losses['policy_loss']:.4f}")
        _log(f"    {'Value loss':<20s}: {epoch_losses['value_loss']:.4f}")
        _log(f"    {'Clip fraction':<20s}: {epoch_losses['clip_fraction']:.4f}")
        _log(f"    {'Approx KL':<20s}: {epoch_losses['approx_kl']:.4f}")
        _log(f"    {'Grad norm':<20s}: {epoch_losses['grad_norm']:.4f}")
        _log(f"    {'Skip rate':<20s}: {epoch_losses['skip_rate']:.4f}")
        _log(f"    {'Log std':<20s}: {current_log_std:.4f}")
        _log(f"    {'Bias scale':<20s}: {bias_scale:.4f}")
        if peer_diag is not None:
            _log(f"    {'Self-weight':<20s}: {peer_diag['self_weight']:.4f}")
            _log(f"    {'Attn entropy':<20s}: {peer_diag['attn_entropy']:.4f}")
        _log(f"    {'Current score':<20s}: {current_score:.4f}")
        _log(f"    {'Best score':<20s}: {best_score:.4f}")
        _log(f"    {'LR':<20s}: {current_lr:.2e}")
        _log(f"    {'Patience':<20s}: {patience_counter}/{cfg.patience}")

        if is_main:
            ckpt = _build_checkpoint()
            if improved:
                torch.save(ckpt, os.path.join(cfg.save_dir, "model_best.pt"))
            torch.save(ckpt, os.path.join(cfg.save_dir, "model_latest.pt"))
            plot_training(
                dict(history),
                os.path.join(cfg.save_dir, "training_curves.png"),
                cfg,
            )

        if patience_counter >= cfg.patience:
            _log(f"    Early stopping at epoch {epoch}")
            break

        if epoch_callback is not None:
            epoch_callback(epoch, current_score)

    _log()
    _log(f"Training complete, best score: {best_score:.4f}")

    # Final evaluation on test set
    best_path = os.path.join(cfg.save_dir, "model_best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])

    test_rng = np.random.default_rng(cfg.seed + 1000)
    test_episodes = generate_episodes(test_data, peer_info, cfg, test_rng)
    rank_test_episodes = test_episodes[_rank::_world]

    _log()
    _log("[Test]")
    test_result = evaluate_episodes(model, rank_test_episodes, cfg, global_features)
    if distributed:
        test_aggregated = _allreduce_eval(
            test_result, len(rank_test_episodes), cfg.device,
        )
    else:
        test_aggregated = test_result

    _log(f"    {'Test episodes':<20s}: {len(test_episodes)}")
    _log(f"    {'Test return':<20s}: {test_aggregated['model_return']:.4f}")
    _log(f"    {'Test baseline':<20s}: {test_aggregated['baseline_return']:.4f}")
    _log(f"    {'Test beat rate':<20s}: {test_aggregated['beat_rate']:.4f}")
    _log(f"    {'Test max drawdown':<20s}: {test_aggregated['max_drawdown']:.4f}")
    _log(f"    {'Test turnover':<20s}: {test_aggregated['mean_turnover']:.4f}")

    if distributed:
        full_result = _gather_test_arrays(
            test_result, len(rank_test_episodes), len(test_episodes),
            cfg.episode_length, cfg.device,
        )
    else:
        full_result = test_result
    if is_main:
        all_rows = build_test_results(test_episodes, full_result, cfg)
        results_path = os.path.join(cfg.save_dir, "test_results.csv")
        pd.DataFrame(all_rows).to_csv(results_path, index=False)
        _log(f"    {'Saved file':<20s}: {results_path}")

    if distributed:
        dist.destroy_process_group()

    return best_score


# Launch =======================================================================

def _preprocess_data(cfg):
    """Load and preprocess data, or reuse cached result."""
    os.makedirs(cfg.save_dir, exist_ok=True)
    preprocess_path = os.path.join(cfg.save_dir, "_preprocess.pt")
    if os.path.exists(preprocess_path):
        _log()
        _log(f"Reusing cached preprocessed data: {preprocess_path}")
        return preprocess_path
    peer_info, train_data, val_data, test_data, feature_cols = _load_all_data(cfg)
    feat_mean, feat_std = compute_feature_stats(train_data)
    torch.save({
        "peer_info": peer_info,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "feature_cols": feature_cols,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }, preprocess_path)
    _log(f"    {'Saved file':<20s}: {preprocess_path}")
    return preprocess_path


def _spawn_ddp_worker(local_rank, cfg_dict, n_gpus, port, preprocess_path):
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(n_gpus)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    cfg = Config()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)
    train(cfg, preprocess_path=preprocess_path)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="PPO + Transformer RL for stock trading")

    for name, ann_type in Config.__annotations__.items():
        default = getattr(cfg, name)
        flag = f"--{name}"
        if ann_type is bool:
            parser.add_argument(flag, type=lambda v: v.lower() in ("true", "1", "yes"),
                                default=default, metavar="BOOL",
                                help=f"(default: {default})")
        else:
            parser.add_argument(flag, type=ann_type, default=default,
                                help=f"(default: {default})")

    args = parser.parse_args()
    for name in Config.__annotations__:
        setattr(cfg, name, getattr(args, name))

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and "RANK" not in os.environ:
        preprocess_path = _preprocess_data(cfg)
        port = _find_free_port()
        _log()
        _log(f"Auto launching DDP on {n_gpus} GPUs (port {port})")
        mp.spawn(
            _spawn_ddp_worker,
            args=(cfg.to_dict(), n_gpus, port, preprocess_path),
            nprocs=n_gpus,
            join=True,
        )
        if os.path.exists(preprocess_path):
            os.remove(preprocess_path)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
