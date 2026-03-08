"""
PPO + Transformer RL for stock trading with cross-sectional peer attention.

Stock CSV: symbol, date, price, <stock_feat_1>, ...

Instead of explicit market features, stocks attend to historically similar
peers via rolling EWM correlation.  Each stock's top-K correlated peers form
the cross-sectional context at each decision point.  Peer sets evolve smoothly
(~1 member change per week) with correlation scores used as attention bias,
eliminating the edge effects of periodic re-clustering.

Continuous position sizing in [-1, 1] with squashed Gaussian policy (tanh).

Architecture: shared-trunk PolicyNetwork (SB3-style shared feature extractor)
with temporal transformer per stock, cross-sectional peer attention across
the K nearest peers, then separate MLP heads for policy (mu) and value.

Launch:
  python train_model.py
  Auto-detects GPUs. Uses DDP when multiple GPUs are available.
  Also works with: torchrun --nproc_per_node=N train_model.py
  Configure all parameters in the Config class.

Hyperparameter sweep:
  python train_model.py --sweep true
  Parallel workers: run multiple processes pointing at the same --sweep_db.
  Monitor with: optuna-dashboard <sweep_db>
"""

import argparse
import bisect
import os
import math
import time
import socket
import functools
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

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
    peer_halflife: int = 60

    # Model
    d_model: int = 128
    d_ff: int = 512
    n_heads: int = 2
    n_layers: int = 2
    peer_n_heads: int = 2
    head_hidden_dim: int = 64
    position_dim: int = 16
    dropout: float = 0.1
    peer_dropout: float = 0.5

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_ema_decay: float = 0.99
    policy_clip: float = 0.2
    value_coeff: float = 0.01
    entropy_coeff: float = 0.01
    log_std_init: float = -0.5

    # Optimizer
    lr: float = 1e-4
    adam_eps: float = 1e-5
    weight_decay: float = 0.02
    grad_clip: float = 0.5

    # Training loop
    ppo_epochs: int = 2
    batch_size: int = 256
    inference_batch: int = 1024
    n_epochs: int = 200
    warmup_epochs: int = 5
    patience: int = 20
    patience_smoothing: int = 10
    seed: int = 42

    # Runtime
    env_workers: int = 0
    use_amp: bool = True
    pytorch_eps: float = 1e-8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Diagnostics
    ablation: bool = True
    grad_diagnostic: bool = True
    sweep: bool = False
    sweep_trials: int = 100
    sweep_db: str = "sqlite:///ppo_sweep.db"

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__class__.__annotations__}


# Feature grouping

def build_feature_groups(column_names):
    """Group feature columns by dot-separated prefix."""
    groups = defaultdict(list)
    for col_idx, col_name in enumerate(column_names):
        prefix = col_name.split(".")[0]
        groups[prefix].append(col_idx)
    return dict(groups)


# Vectorized environment

class _VecEnv:
    """Internal: batched single-stock trading environment with peer context.

    Observations are 4D: (n_envs, K+1, lookback, F_stock) where index 0 is
    the traded stock and indices 1..K are its peers.  corr_scores and
    peer_mask are constant per episode set and stored as attributes.

    Execution model:
        - Agent observes features on day d and trades at price[d].
        - Reward = position * log(price[d+1] / price[d])
          minus transaction cost on position changes.
    """

    def __init__(self, episodes, lookback, transaction_cost):
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        self.n_envs = len(episodes)
        self.stock_features = np.stack([ep["stock_features"] for ep in episodes])
        self.peer_features = np.stack([ep["peer_features"] for ep in episodes])
        self.log_returns = np.stack([ep["log_returns"] for ep in episodes])
        self._corr_scores = np.stack(
            [ep["corr_scores"] for ep in episodes],
        ).astype(np.float32)
        self._peer_mask = np.stack([ep["peer_mask"] for ep in episodes])
        self.total_steps = self.stock_features.shape[1]

    @property
    def corr_scores(self):
        return self._corr_scores

    @property
    def peer_mask(self):
        return self._peer_mask

    def _build_observation(self):
        s = self.current_step
        lb = self.lookback
        self_window = self.stock_features[:, s - lb : s, :][:, None, :, :]
        peer_windows = self.peer_features[:, :, s - lb : s, :]
        return np.concatenate(
            [self_window, peer_windows], axis=1,
        ).astype(np.float32)

    def reset(self):
        self.current_step = self.lookback
        self.position = np.zeros(self.n_envs, dtype=np.float64)
        return self._build_observation(), self.position.astype(np.float32)

    def step(self, actions):
        new_position = np.clip(actions.astype(np.float64), -1.0, 1.0)
        rewards = (
            self.position * self.log_returns[:, self.current_step]
            + np.log1p(-self.transaction_cost * np.abs(new_position - self.position))
        ).astype(np.float32)
        self.position = new_position
        self.current_step += 1
        done = self.current_step >= self.total_steps
        obs = None if done else self._build_observation()
        return obs, rewards, done, self.position.astype(np.float32)

    def terminal_observation(self):
        """Build observation at the truncation boundary (current_step == total_steps)."""
        return self._build_observation()


# Parallel vectorized environment


class VecEnv:
    """Batched trading environment with thread parallelism and peer context.

    Numpy operations release the GIL, so threads achieve true multi-core
    parallelism for the observation building and environment stepping.
    """

    def __init__(self, episodes, lookback, transaction_cost, n_workers):
        n_workers = max(1, min(n_workers, len(episodes)))
        self._pool = ThreadPoolExecutor(max_workers=n_workers)

        shard_size = math.ceil(len(episodes) / n_workers)
        shard_ranges = []
        for i in range(n_workers):
            start = i * shard_size
            end = min(start + shard_size, len(episodes))
            if start >= end:
                break
            shard_ranges.append((start, end))

        def _make_shard(start, end):
            return _VecEnv(episodes[start:end], lookback, transaction_cost)

        futures = [
            self._pool.submit(_make_shard, s, e) for s, e in shard_ranges
        ]
        self._shards = [f.result() for f in futures]
        self._shard_slices = []
        offset = 0
        for s, e in shard_ranges:
            n = e - s
            self._shard_slices.append((offset, offset + n))
            offset += n

        self.corr_scores = np.concatenate(
            [s.corr_scores for s in self._shards], axis=0,
        )
        self.peer_mask = np.concatenate(
            [s.peer_mask for s in self._shards], axis=0,
        )

    def reset(self):
        futures = [self._pool.submit(s.reset) for s in self._shards]
        results = [f.result() for f in futures]
        obs = np.concatenate([r[0] for r in results], axis=0)
        pos = np.concatenate([r[1] for r in results], axis=0)
        return obs, pos

    def step(self, actions):
        futures = [
            self._pool.submit(shard.step, actions[s:e])
            for shard, (s, e) in zip(self._shards, self._shard_slices)
        ]
        results = [f.result() for f in futures]
        obs = (np.concatenate([r[0] for r in results], axis=0)
               if results[0][0] is not None else None)
        rewards = np.concatenate([r[1] for r in results], axis=0)
        done = results[0][2]
        pos = np.concatenate([r[3] for r in results], axis=0)
        return obs, rewards, done, pos

    def terminal_observation(self):
        futures = [self._pool.submit(s.terminal_observation) for s in self._shards]
        return np.concatenate([f.result() for f in futures], axis=0)

    def close(self):
        self._pool.shutdown(wait=False)


# Reward normalizer


class RewardNormalizer:
    """EMA reward normalizer for non-stationary reward scaling.

    Tracks an exponential moving average of mean squared reward. Divides
    rewards by the EMA standard deviation without mean-shifting, so the
    optimal policy is unchanged. Adapts to distribution shift as the
    policy improves, unlike a monotonically accumulating estimator.
    """

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
            self.mean_sq = (
                self.decay * self.mean_sq
                + (1 - self.decay) * batch_mean_sq
            )

    def update(self, rewards):
        flat = rewards.ravel().astype(np.float64)
        if len(flat) > 0:
            self._apply_ema(float((flat ** 2).mean()))

    def update_distributed(self, rewards, device):
        """All-reduce local squared sums so every rank applies identical EMA."""
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


# Model

class GroupProjection(nn.Module):
    def __init__(self, group_indices, d_model):
        super().__init__()
        self.group_names = sorted(group_indices)
        n_groups = len(self.group_names)
        base_dim = d_model // n_groups
        remainder = d_model % n_groups
        self.proj = nn.ModuleDict()
        for group_rank, name in enumerate(self.group_names):
            out_dim = base_dim + (1 if group_rank < remainder else 0)
            self.proj[name] = nn.Linear(len(group_indices[name]), out_dim)
            self.register_buffer(
                f"idx_{name}", torch.tensor(group_indices[name], dtype=torch.long),
            )

    def forward(self, x):
        return torch.cat(
            [
                self.proj[name](x[..., getattr(self, f"idx_{name}")])
                for name in self.group_names
            ],
            dim=-1,
        )


class PeerCrossSectionalAttention(nn.Module):
    """Cross-sectional attention across peer stocks at the same timestep.

    Uses correlation scores as additive attention bias so that even when
    the top-K peer set changes by one member, attention weights shift
    continuously (the swapped peers have nearly equal correlation).
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
        corr_scores: (B, K+1)  — correlation of each slot with the query stock
                     (index 0 = self, set to 1.0)
        peer_mask:   (B, K+1)  bool — True for padded positions
        need_weights: if True, return (output, attn_weights) instead of output
        """
        K1 = x.shape[1]
        # Additive bias: each key position's relevance = its correlation
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


class Trunk(nn.Module):
    """Encoder backbone: group projection, temporal transformer, peer
    cross-sectional attention, pooling.

    Shared between policy and value heads (SB3-style shared feature extractor).
    """

    def __init__(self, feature_groups, cfg):
        super().__init__()
        self.cfg = cfg
        self.stock_proj = GroupProjection(feature_groups, cfg.d_model)
        self.stock_norm = nn.LayerNorm(cfg.d_model)
        self.pos_emb = nn.Parameter(
            torch.randn(1, cfg.lookback, cfg.d_model) * 0.02,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, cfg.n_layers)

        self.peer_attn = PeerCrossSectionalAttention(
            cfg.d_model, cfg.peer_n_heads, cfg.dropout,
        )

        self.position_proj = nn.Linear(1, cfg.position_dim)

    @property
    def output_dim(self):
        return self.cfg.d_model + self.cfg.position_dim

    def forward(self, obs, positions, corr_scores, peer_mask):
        """
        obs:         (B, K+1, lookback, F)
        positions:   (B,)
        corr_scores: (B, K+1)
        peer_mask:   (B, K+1)
        """
        B, K1, T, F = obs.shape

        # Stage 1: per-stock temporal encoding (shared transformer)
        flat = obs.reshape(B * K1, T, F)
        hidden = self.stock_norm(self.stock_proj(flat)) + self.pos_emb
        pooled = self.transformer(hidden).mean(dim=1)   # (B*K1, d_model)
        pooled = pooled.view(B, K1, -1)                 # (B, K+1, d_model)

        # Stage 2: cross-sectional peer attention
        # Peer dropout: during training, with probability peer_dropout,
        # mask ALL peers for a sample so the model learns to function
        # without peer context (analogous to old market_dropout).
        if self.training and self.cfg.peer_dropout > 0:
            keep = torch.bernoulli(torch.full(
                (B, 1), 1 - self.cfg.peer_dropout,
                device=obs.device, dtype=obs.dtype,
            ))
            # keep=0 → mask all peers (indices 1:); self (index 0) stays
            drop_mask = peer_mask.clone()
            drop_mask[:, 1:] = drop_mask[:, 1:] | (keep == 0)
            pooled = self.peer_attn(pooled, corr_scores, drop_mask)
        else:
            pooled = self.peer_attn(pooled, corr_scores, peer_mask)

        # Extract the traded stock's representation (index 0)
        self_repr = pooled[:, 0, :]                     # (B, d_model)

        # Stage 3: position embedding
        pos_emb = self.position_proj(positions.unsqueeze(-1))
        return torch.cat([self_repr, pos_emb], dim=-1)


class PolicyNetwork(nn.Module):
    """Shared-trunk policy network (SB3-style shared feature extractor).

    One transformer trunk (feature extractor) shared between policy and value.
    Separate MLP heads branch from the pooled trunk output.
    """

    def __init__(self, feature_groups, cfg):
        super().__init__()
        self.trunk = Trunk(feature_groups, cfg)
        head_input_dim = self.trunk.output_dim
        self.policy_head = nn.Sequential(
            nn.Linear(head_input_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.tensor(cfg.log_std_init))
        self.value_head = nn.Sequential(
            nn.Linear(head_input_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def forward(self, obs, positions, corr_scores, peer_mask):
        pooled = self.trunk(obs, positions, corr_scores, peer_mask)
        mu = self.policy_head(pooled).squeeze(-1)
        values = self.value_head(pooled).squeeze(-1)
        return mu, self.log_std.expand_as(mu), values



def _squashed_gaussian_log_prob(u, mu, log_std):
    """Log-prob of a tanh-squashed Gaussian, numerically stable.

    u:       pre-tanh sample (the raw Gaussian draw)
    mu:      mean of the Gaussian
    log_std: log standard deviation

    Gaussian log-prob is computed entirely in log-space to avoid
    exponentiating log_std.
    Uses the identity log(1 - tanh²(u)) = 2(log(2) - u - softplus(-2u))
    to avoid the catastrophic cancellation in 1 - tanh²(u) when |u| > 5.
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
    """Greedy action with mu for evaluation."""
    mu, _, values = model(obs, positions, corr_scores, peer_mask)
    actions = mu.tanh()
    return actions, values, mu


def compute_action_log_probs(model, obs, raw_actions, positions,
                             corr_scores, peer_mask):
    """raw_actions stores the pre-tanh Gaussian sample u."""
    mu, log_std, values = model(obs, positions, corr_scores, peer_mask)
    mu, log_std = mu.float(), log_std.float()
    u = raw_actions.float()
    log_probs = _squashed_gaussian_log_prob(u, mu, log_std)
    entropy = -log_probs
    return log_probs, entropy, values



# Batched inference (GPU chunking for large episode counts)

def _batched_stochastic(model, obs, pos, corr_scores, peer_mask, cfg):
    """Run select_stochastic in inference_batch chunks."""
    n = obs.shape[0]
    actions = np.empty(n, dtype=np.float32)
    raw = np.empty(n, dtype=np.float32)
    logp = np.empty(n, dtype=np.float32)
    vals = np.empty(n, dtype=np.float32)
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
        for i in range(0, n, cfg.inference_batch):
            j = min(i + cfg.inference_batch, n)
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
    """Run select_greedy in inference_batch chunks."""
    n = obs.shape[0]
    actions = np.empty(n, dtype=np.float32)
    vals = np.empty(n, dtype=np.float32)
    mus = np.empty(n, dtype=np.float32)
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
        for i in range(0, n, cfg.inference_batch):
            j = min(i + cfg.inference_batch, n)
            s = torch.from_numpy(obs[i:j]).to(cfg.device)
            p = torch.from_numpy(pos[i:j]).to(cfg.device)
            c = torch.from_numpy(corr_scores[i:j]).to(cfg.device)
            m = torch.from_numpy(peer_mask[i:j]).to(cfg.device)
            a, v, mu = select_greedy(model, s, p, c, m)
            actions[i:j] = a.cpu().numpy()
            vals[i:j] = v.cpu().numpy()
            mus[i:j] = mu.cpu().numpy()
    return actions, vals, mus


# Rollout buffer

class RolloutBuffer:
    """Pre-allocated buffer for PPO rollout data.

    All arrays are shaped (n_episodes, episode_length), enabling vectorized
    GAE computation and batch construction via advanced numpy indexing.
    Peer features, correlation scores, and masks are stored for lazy
    observation reconstruction in get_batch.
    """

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
        self.all_peer_features = None
        self.all_corr_scores = None
        self.all_peer_masks = None

    def register_episodes(self, episodes):
        self.all_stock_features = np.stack(
            [ep["stock_features"] for ep in episodes],
        )
        self.all_peer_features = np.stack(
            [ep["peer_features"] for ep in episodes],
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
        # Self-stock window: (batch, lookback, F)
        self_batch = self.all_stock_features[
            episode_ids[:, None], stock_row_indices, :
        ]

        # Peer windows: (batch, K, lookback, F)
        K = self.all_peer_features.shape[1]
        batch_size = len(episode_ids)
        ep_peers = self.all_peer_features[episode_ids]  # (batch, K, T_full, F)
        b_idx = np.arange(batch_size)[:, None, None]
        k_idx = np.arange(K)[None, :, None]
        t_idx = np.broadcast_to(
            stock_row_indices[:, None, :], (batch_size, K, lookback),
        )
        peer_batch = ep_peers[b_idx, k_idx, t_idx]  # (batch, K, lookback, F)

        # Pack: (batch, K+1, lookback, F)
        observation_batch = np.concatenate(
            [self_batch[:, None, :, :], peer_batch], axis=1,
        )

        return {
            "states": torch.from_numpy(observation_batch).to(
                device, non_blocking=True,
            ),
            "corr_scores": torch.from_numpy(
                self.all_corr_scores[episode_ids],
            ).to(device, non_blocking=True),
            "peer_mask": torch.from_numpy(
                self.all_peer_masks[episode_ids],
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


# Data loading

def _read_csv(path, **kwargs):
    return pd.read_csv(path, engine="pyarrow", **kwargs)


@timed
def load_stock_data(path):
    df = _read_csv(path, dtype={"symbol": str})
    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    feature_cols = [c for c in df.columns if c not in ("symbol", "date", "price")]
    symbols = df["symbol"].values
    prices = df["price"].values.astype(np.float64)
    features = df[feature_cols].values.astype(np.float32)
    dates = df["date"].values

    symbol_breaks = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1

    ratio = prices[1:] / prices[:-1]
    log_returns = np.zeros(len(prices), dtype=np.float32)
    log_returns[1:] = np.log(ratio).astype(np.float32)
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
        }
    return data, feature_cols


# Peer map computation

def _split_stock_data(all_data, train_ratio, val_ratio):
    """Split data into train / val / test by global date cutoffs.

    Collects all unique dates across every symbol, sorts them, and picks
    two cutoff dates at the 80% and 90% marks.  Every symbol is then
    sliced by these shared dates so that no test date appears in any
    symbol's training window.
    """
    all_dates = sorted({d for sd in all_data.values() for d in sd["dates"]})
    n_dates = len(all_dates)
    date_train_end = all_dates[int(n_dates * train_ratio) - 1]
    date_val_end = all_dates[int(n_dates * (train_ratio + val_ratio)) - 1]

    train, val, test = {}, {}, {}
    for sym, sd in all_data.items():
        dates = sd["dates"]
        keys = ("stock_features", "log_returns", "dates", "prices")

        # Find split indices in this symbol's sorted date array
        train_mask = [d <= date_train_end for d in dates]
        val_mask = [date_train_end < d <= date_val_end for d in dates]
        test_mask = [d > date_val_end for d in dates]

        for split_dict, mask in [(train, train_mask), (val, val_mask), (test, test_mask)]:
            idx = [i for i, m in enumerate(mask) if m]
            if not idx:
                continue
            s, e = idx[0], idx[-1] + 1
            split_dict[sym] = {
                k: (sd[k][s:e] if k != "dates" else sd[k][s:e])
                for k in keys
            }

    return train, val, test


@timed
def _compute_snapshot(t_idx, ret_matrix, ewm_weights, lookback, N, K):
    """Compute a single EWM correlation snapshot and extract top-K peers."""
    max_window = len(ewm_weights)
    corr_end = t_idx - lookback
    corr_start = max(0, corr_end - max_window)
    window = ret_matrix[:, corr_start:corr_end]
    W = window.shape[1]
    if W < 2:
        return None
    weights = ewm_weights[:W][::-1].copy()

    valid = ~np.isnan(window)
    window_filled = np.nan_to_num(window)

    w_sum = valid * weights[None, :]
    denom = np.maximum(w_sum.sum(axis=1, keepdims=True), 1e-12)
    mu = (window_filled * w_sum).sum(axis=1, keepdims=True) / denom

    centered = (window_filled - mu) * np.sqrt(w_sum)
    cov = centered @ centered.T
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / (std[:, None] * std[None, :])
    np.fill_diagonal(corr, -2.0)

    top_k_idx = np.zeros((N, K), dtype=np.int32)
    top_k_corr = np.zeros((N, K), dtype=np.float32)

    for i in range(N):
        row = np.nan_to_num(corr[i], nan=-2.0)
        if K >= N - 1:
            idx = np.argsort(-row)[:K]
        else:
            idx = np.argpartition(row, -K)[-K:]
            idx = idx[np.argsort(-row[idx])]
        top_k_idx[i] = idx
        top_k_corr[i] = np.clip(row[idx], 0.0, 1.0)

    return t_idx, (top_k_idx, top_k_corr)


def precompute_peer_maps(merged_data, n_peers, halflife, lookback):
    """Precompute top-K peer indices at regular intervals using EWM correlation.

    The correlation window ends at (date_col - lookback) so it never overlaps
    with the observation window that the model sees.  Snapshots are taken every
    ``halflife`` trading days.  At query time the most recent earlier snapshot
    is used.

    Returns a dict with everything needed to look up peers for any date.
    """
    symbols = sorted(merged_data.keys())
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    N = len(symbols)

    all_dates = sorted({d for sd in merged_data.values() for d in sd["dates"]})
    date_to_col = {d: i for i, d in enumerate(all_dates)}
    T = len(all_dates)

    ret_matrix = np.full((N, T), np.nan, dtype=np.float64)
    for i, sym in enumerate(symbols):
        sd = merged_data[sym]
        for d, r in zip(sd["dates"], sd["log_returns"]):
            ret_matrix[i, date_to_col[d]] = float(r)

    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    max_window = min(4 * halflife, T)
    ewm_weights = alpha * ((1 - alpha) ** np.arange(max_window))
    ewm_weights /= ewm_weights.sum()

    min_history = 2 * halflife + lookback
    checkpoint_cols = list(range(min_history, T, halflife))
    K = min(n_peers, N - 1)

    n_workers = max(1, len(os.sched_getaffinity(0)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _compute_snapshot, t_idx, ret_matrix, ewm_weights,
                lookback, N, K,
            )
            for t_idx in checkpoint_cols
        ]
        snapshots = {}
        for f in futures:
            result = f.result()
            if result is not None:
                snapshots[result[0]] = result[1]

    checkpoint_arr = np.array(sorted(snapshots.keys()), dtype=np.int64)

    return {
        "snapshots": snapshots,
        "checkpoint_cols": checkpoint_arr,
        "symbols": symbols,
        "symbol_to_idx": symbol_to_idx,
        "all_dates": all_dates,
        "date_to_col": date_to_col,
        "n_peers": K,
    }


def _load_all_data(cfg, log=True):
    if log:
        _log()
        _log("[Data]")
    all_data, stock_feature_cols = load_stock_data(cfg.data_path)
    train_data, val_data, test_data = _split_stock_data(
        all_data, cfg.train_ratio, cfg.val_ratio,
    )

    # Determine date boundaries for logging
    def _date_range(d):
        dates = sorted({dt for sd in d.values() for dt in sd["dates"]})
        return (dates[0], dates[-1]) if dates else (None, None)
    train_range = _date_range(train_data)
    val_range = _date_range(val_data)
    test_range = _date_range(test_data)

    # Peer map uses the full (unsplit) data — no merge boundary artefacts.
    peer_info = precompute_peer_maps(
        all_data, cfg.n_peers, cfg.peer_halflife, cfg.lookback,
    )

    if log:
        _log(f"    {'Symbols':<20s}: {len(all_data)}")
        _log(f"    {'Train symbols':<20s}: {len(train_data)}")
        _log(f"    {'Val symbols':<20s}: {len(val_data)}")
        _log(f"    {'Test symbols':<20s}: {len(test_data)}")
        _log(f"    {'Train dates':<20s}: {train_range[0]} .. {train_range[1]}")
        _log(f"    {'Val dates':<20s}: {val_range[0]} .. {val_range[1]}")
        _log(f"    {'Test dates':<20s}: {test_range[0]} .. {test_range[1]}")
        _log(f"    {'Train observations':<20s}: {sum(len(d['stock_features']) for d in train_data.values()):,}")
        _log(f"    {'Val observations':<20s}: {sum(len(d['stock_features']) for d in val_data.values()):,}")
        _log(f"    {'Test observations':<20s}: {sum(len(d['stock_features']) for d in test_data.values()):,}")
        _log(f"    {'Stock features':<20s}: {len(stock_feature_cols)}")
        _log(f"    {'Peer snapshots':<20s}: {len(peer_info['snapshots'])}")
        _log(f"    {'Effective K':<20s}: {peer_info['n_peers']}")
    return peer_info, train_data, val_data, test_data, stock_feature_cols


# Episode generation

def _generate_symbol_episodes(
    symbol, symbol_data, stock_data, peer_info, cfg, anchor,
    date_maps, sorted_dates,
):
    """Generate all episodes for a single symbol."""
    chunk_size = cfg.lookback + cfg.episode_length
    symbols = peer_info["symbols"]
    symbol_to_idx = peer_info["symbol_to_idx"]
    date_to_col = peer_info["date_to_col"]
    checkpoint_cols = peer_info["checkpoint_cols"]
    snapshots = peer_info["snapshots"]

    n_rows = len(symbol_data["stock_features"])
    if n_rows < chunk_size:
        return []
    sym_idx = symbol_to_idx.get(symbol)
    if sym_idx is None:
        return []

    F_dim = symbol_data["stock_features"].shape[1]
    max_start = n_rows - chunk_size
    episodes = []

    start = anchor
    while start <= max_start:
        obs_date = symbol_data["dates"][start + cfg.lookback - 1]
        col = date_to_col.get(obs_date)
        if col is None:
            start += cfg.episode_length
            continue

        cp_idx = int(np.searchsorted(checkpoint_cols, col, side="right")) - 1
        if cp_idx < 0 or len(checkpoint_cols) == 0:
            start += cfg.episode_length
            continue

        cp_col = int(checkpoint_cols[cp_idx])
        top_k_idx, top_k_corr = snapshots[cp_col]
        peer_indices = top_k_idx[sym_idx]
        peer_corrs = top_k_corr[sym_idx]

        episode_dates = symbol_data["dates"][start : start + chunk_size]

        peer_feature_list = []
        valid_corrs = []
        valid_mask = [False]  # self always valid

        for pidx, pcorr in zip(peer_indices, peer_corrs):
            peer_sym = symbols[pidx]
            p_map = date_maps.get(peer_sym)
            if p_map is None or pcorr <= 0:
                peer_feature_list.append(
                    np.zeros((chunk_size, F_dim), dtype=np.float32),
                )
                valid_corrs.append(0.0)
                valid_mask.append(True)
                continue

            p_sorted = sorted_dates[peer_sym]
            aligned = np.empty(chunk_size, dtype=np.intp)
            usable = True
            for t, d in enumerate(episode_dates):
                row = p_map.get(d)
                if row is not None:
                    aligned[t] = row
                else:
                    pos = bisect.bisect_right(p_sorted, d) - 1
                    if pos < 0:
                        usable = False
                        break
                    aligned[t] = p_map[p_sorted[pos]]

            if not usable:
                peer_feature_list.append(
                    np.zeros((chunk_size, F_dim), dtype=np.float32),
                )
                valid_corrs.append(0.0)
                valid_mask.append(True)
                continue

            peer_feature_list.append(
                stock_data[peer_sym]["stock_features"][aligned],
            )
            valid_corrs.append(float(pcorr))
            valid_mask.append(False)

        episodes.append({
            "symbol": symbol,
            "stock_features": symbol_data["stock_features"][start : start + chunk_size],
            "log_returns": symbol_data["log_returns"][start : start + chunk_size],
            "dates": episode_dates,
            "prices": symbol_data["prices"][start : start + chunk_size],
            "peer_features": np.stack(peer_feature_list),
            "corr_scores": np.array([1.0] + valid_corrs, dtype=np.float32),
            "peer_mask": np.array(valid_mask, dtype=bool),
        })
        start += cfg.episode_length

    return episodes


def generate_episodes(stock_data, peer_info, cfg, epoch_rng):
    """Generate non-overlapping episodes with peer context.

    For each episode, peers are looked up from the nearest earlier EWM
    correlation snapshot.  Peer features are date-aligned from stock_data.
    Symbols are processed in parallel across CPU threads.
    """
    chunk_size = cfg.lookback + cfg.episode_length

    date_maps = {}
    sorted_dates = {}
    for sym, sd in stock_data.items():
        date_maps[sym] = {d: i for i, d in enumerate(sd["dates"])}
        sorted_dates[sym] = sorted(sd["dates"])

    # Pre-generate per-symbol anchors (epoch_rng is not thread-safe)
    anchors = {}
    for symbol, symbol_data in stock_data.items():
        n_rows = len(symbol_data["stock_features"])
        if n_rows < chunk_size:
            continue
        max_start = n_rows - chunk_size
        anchors[symbol] = int(
            epoch_rng.integers(0, min(cfg.episode_length, max_start + 1)),
        )

    n_workers = max(1, len(os.sched_getaffinity(0)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _generate_symbol_episodes,
                symbol, stock_data[symbol], stock_data, peer_info, cfg,
                anchors[symbol], date_maps, sorted_dates,
            )
            for symbol in anchors
        ]
        episodes = []
        for f in futures:
            episodes.extend(f.result())

    epoch_rng.shuffle(episodes)
    return episodes


# Metrics

def compute_episode_returns(rewards):
    """Per-episode percentage return from additive log rewards."""
    return np.expm1(rewards.sum(axis=1))


def compute_max_drawdown(rewards):
    """Mean per-episode max drawdown as percentage decline from peak."""
    cum_returns = np.cumsum(rewards, axis=1)
    running_max = np.maximum.accumulate(cum_returns, axis=1)
    drawdowns = 1.0 - np.exp(cum_returns - running_max)
    return float(drawdowns.max(axis=1).mean())


# Baseline

def compute_baseline_rewards(episodes, cfg):
    """Buy-and-hold baseline: enter long at open of first trading day, hold."""
    log_returns = np.stack([ep["log_returns"] for ep in episodes])
    trading_returns = log_returns[:, cfg.lookback :]
    rewards = np.empty_like(trading_returns)
    rewards[:, 0] = np.log1p(-cfg.transaction_cost)
    rewards[:, 1:] = trading_returns[:, 1:]
    return rewards


# Rollout collection


def _env_n_workers(cfg):
    """Compute per-rank CPU worker count for VecEnv."""
    n_workers = cfg.env_workers
    if n_workers == 0:
        n_cores = len(os.sched_getaffinity(0))
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", _world))
        n_workers = max(1, n_cores // local_world)
    return n_workers


@timed
def collect_training_rollout(model, episodes, cfg,
                             reward_normalizer, distributed):
    n_episodes = len(episodes)
    episode_length = cfg.episode_length

    rollout = RolloutBuffer(n_episodes, episode_length, cfg.lookback)
    rollout.register_episodes(episodes)

    env = VecEnv(
        episodes, cfg.lookback, cfg.transaction_cost, _env_n_workers(cfg),
    )
    obs, pos = env.reset()
    corr = env.corr_scores
    mask = env.peer_mask
    model.train()
    for step in range(episode_length):
        actions, raw_actions, log_probs, values = _batched_stochastic(
            model, obs, pos, corr, mask, cfg,
        )
        rollout.positions[:, step] = pos
        rollout.raw_actions[:, step] = raw_actions
        obs, rewards, _, pos = env.step(actions)
        rollout.log_probs[:, step] = log_probs
        rollout.values[:, step] = values
        rollout.rewards[:, step] = rewards

    terminal_obs = env.terminal_observation()
    _, terminal_values, _ = _batched_greedy(
        model, terminal_obs, pos, corr, mask, cfg,
    )
    rollout.final_values[:] = terminal_values
    env.close()

    baseline_rewards = compute_baseline_rewards(episodes, cfg)
    model_return = float(compute_episode_returns(rollout.rewards).mean())
    baseline_return = float(compute_episode_returns(baseline_rewards).mean())

    if distributed:
        reward_normalizer.update_distributed(rollout.rewards, cfg.device)
    else:
        reward_normalizer.update(rollout.rewards)
    rollout.rewards = reward_normalizer.normalize(rollout.rewards)

    return rollout, model_return, baseline_return


# Deterministic evaluation

def evaluate_deterministic(model, episodes, cfg):
    n_episodes = len(episodes)
    episode_length = cfg.episode_length
    all_rewards = np.empty((n_episodes, episode_length), dtype=np.float32)
    all_positions = np.empty((n_episodes, episode_length), dtype=np.float32)
    all_mus = np.empty((n_episodes, episode_length), dtype=np.float32)

    env = VecEnv(
        episodes, cfg.lookback, cfg.transaction_cost, _env_n_workers(cfg),
    )
    obs, pos = env.reset()
    corr = env.corr_scores
    mask = env.peer_mask
    model.eval()
    for step in range(episode_length):
        actions, _, mus = _batched_greedy(model, obs, pos, corr, mask, cfg)
        obs, rewards, _, pos = env.step(actions)
        all_rewards[:, step] = rewards
        all_positions[:, step] = pos
        all_mus[:, step] = mus
    env.close()

    return all_rewards, all_positions, all_mus


@timed
def evaluate_episodes(model, episodes, cfg):
    if len(episodes) == 0:
        return {
            "model_return": 0.0, "baseline_return": 0.0, "beat_rate": 0.0,
            "max_drawdown": 0.0, "mean_turnover": 0.0,
            "position_histogram": np.zeros(_POSITION_BINS, dtype=np.float64),
            "mu_histogram": np.zeros(_MU_BINS, dtype=np.float64),
            "pos_sum": 0.0, "pos_sum_sq": 0.0, "n_actions": 0,
            "mu_sum": 0.0, "mu_sum_sq": 0.0,
            "model_rewards": np.empty((0, cfg.episode_length)),
            "baseline_rewards": np.empty((0, cfg.episode_length)),
            "model_positions": np.empty((0, cfg.episode_length), dtype=np.float32),
            "model_mus": np.empty((0, cfg.episode_length)),
        }

    model_rewards, model_positions, model_mus = evaluate_deterministic(
        model, episodes, cfg,
    )
    baseline_rewards = compute_baseline_rewards(episodes, cfg)
    model_returns = compute_episode_returns(model_rewards)
    baseline_returns = compute_episode_returns(baseline_rewards)

    pos_flat = model_positions.ravel().astype(np.float64)
    pos_hist, _ = np.histogram(
        pos_flat, bins=_POSITION_BINS, range=(-1.0, 1.0),
    )
    pos_hist = pos_hist.astype(np.float64)

    mu_flat = model_mus.ravel().astype(np.float64)
    mu_flat = np.clip(mu_flat, _MU_RANGE[0], _MU_RANGE[1])
    mu_hist, _ = np.histogram(
        mu_flat, bins=_MU_BINS, range=_MU_RANGE,
    )
    mu_hist = mu_hist.astype(np.float64)

    n_actions = int(model_positions.size)
    turnover = float(np.abs(np.diff(model_positions, axis=1)).sum(axis=1).mean())

    return {
        "model_return": float(model_returns.mean()),
        "baseline_return": float(baseline_returns.mean()),
        "beat_rate": float(np.mean(model_returns > baseline_returns)),
        "max_drawdown": compute_max_drawdown(model_rewards),
        "mean_turnover": turnover,
        "position_histogram": pos_hist,
        "mu_histogram": mu_hist,
        "pos_sum": float(pos_flat.sum()),
        "pos_sum_sq": float((pos_flat ** 2).sum()),
        "n_actions": n_actions,
        "mu_sum": float(mu_flat.sum()),
        "mu_sum_sq": float((mu_flat ** 2).sum()),
        "model_rewards": model_rewards,
        "baseline_rewards": baseline_rewards,
        "model_positions": model_positions,
        "model_mus": model_mus,
    }


@timed
def evaluate_ablated(model, episodes, cfg):
    """Evaluate with peers or stock features zeroed to measure feature dependency."""
    if len(episodes) == 0:
        return {"no_peers": 0.0, "no_stock": 0.0}

    no_peer_episodes = [
        {**ep, "peer_features": np.zeros_like(ep["peer_features"]),
         "corr_scores": np.array(
             [1.0] + [0.0] * (len(ep["corr_scores"]) - 1), dtype=np.float32,
         ),
         "peer_mask": np.array(
             [False] + [True] * (len(ep["peer_mask"]) - 1), dtype=bool,
         )}
        for ep in episodes
    ]
    no_peer_rewards, _, _ = evaluate_deterministic(
        model, no_peer_episodes, cfg,
    )

    no_stock_episodes = [
        {**ep, "stock_features": np.zeros_like(ep["stock_features"])}
        for ep in episodes
    ]
    no_stock_rewards, _, _ = evaluate_deterministic(
        model, no_stock_episodes, cfg,
    )

    return {
        "no_peers": float(compute_episode_returns(no_peer_rewards).mean()),
        "no_stock": float(compute_episode_returns(no_stock_rewards).mean()),
    }


def diagnose_peer_attention(model, batch):
    """Extract peer attention statistics from a small batch.

    Returns mean self-weight (attention on index 0 averaged across query
    positions and batch) and mean attention entropy.
    """
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        states = batch["states"].float()
        corr_scores = batch["corr_scores"]
        peer_mask = batch["peer_mask"]

        B, K1, T, F = states.shape
        flat = states.reshape(B * K1, T, F)
        trunk = base_model.trunk
        hidden = trunk.stock_norm(trunk.stock_proj(flat)) + trunk.pos_emb
        pooled = trunk.transformer(hidden).mean(dim=1).view(B, K1, -1)

        # Run peer attention with weights
        _, attn_weights = trunk.peer_attn(
            pooled, corr_scores, peer_mask, need_weights=True,
        )
        # attn_weights: (B, K+1, K+1) — [query, key] averaged over heads

        # Self-weight: attention from the self query (row 0) to self key (col 0)
        self_weight = float(attn_weights[:, 0, 0].mean().item())

        # Attention entropy of the self query (row 0) over all keys
        self_query_weights = attn_weights[:, 0, :]          # (B, K+1)
        eps = 1e-8
        log_w = torch.log(self_query_weights.clamp(min=eps))
        attn_entropy = float(-(self_query_weights * log_w).sum(dim=-1).mean().item())

    return {
        "self_weight": self_weight,
        "attn_entropy": attn_entropy,
    }


# Gradient diagnostics

def diagnose_gradient_contributions(model, batch, cfg):
    """Run separate backward passes to measure each loss term's gradient norm."""
    base_model = model.module if hasattr(model, "module") else model
    base_model.zero_grad()

    with torch.amp.autocast("cuda", enabled=False):
        states = batch["states"].float()
        raw_actions = batch["raw_actions"].float()
        positions = batch["positions"].float()
        log_probs_old = batch["log_probs"].float()
        advantages = batch["advantages"].float()
        returns = batch["returns"].float()
        corr_scores = batch["corr_scores"]
        peer_mask = batch["peer_mask"]

        new_log_probs, _, new_values = compute_action_log_probs(
            base_model, states, raw_actions, positions,
            corr_scores, peer_mask,
        )
        ratio = torch.exp(new_log_probs - log_probs_old)
        surrogate = torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - cfg.policy_clip, 1 + cfg.policy_clip)
            * advantages,
        )
        policy_loss = -surrogate.mean()
        value_loss = cfg.value_coeff * F.mse_loss(new_values, returns)

    norms = {}
    all_params = list(base_model.parameters())

    base_model.zero_grad()
    policy_loss.backward(retain_graph=True)
    policy_grads = torch.cat([
        p.grad.data.flatten() if p.grad is not None
        else torch.zeros_like(p.data.flatten())
        for p in all_params
    ])
    norms["policy"] = policy_grads.norm(2).item()

    base_model.zero_grad()
    value_loss.backward()
    value_grads = torch.cat([
        p.grad.data.flatten() if p.grad is not None
        else torch.zeros_like(p.data.flatten())
        for p in all_params
    ])
    norms["value"] = value_grads.norm(2).item()

    norms["cosine_sim"] = F.cosine_similarity(
        policy_grads.unsqueeze(0), value_grads.unsqueeze(0),
    ).item()

    base_model.zero_grad()
    return norms


# PPO update

_VALUE_PRED_BINS = 50
_VALUE_PRED_RANGE = (-10.0, 10.0)
_POSITION_BINS = 50
_MU_BINS = 50
_MU_RANGE = (-1.0, 1.0)


def ppo_update(model, optimizer, scaler,
                   rollout, cfg, n_local_transitions,
                   n_padded_transitions, entropy_coeff):
    """One PPO pass over the rollout buffer."""
    model.train()
    base_model = model.module if hasattr(model, "module") else model
    policy_loss_sum = value_loss_sum = entropy_sum = 0.0
    clip_fraction_sum = approx_kl_sum = 0.0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    skipped_batches = 0
    value_pred_hist = np.zeros(_VALUE_PRED_BINS, dtype=np.float64)
    value_target_hist = np.zeros(_VALUE_PRED_BINS, dtype=np.float64)
    vp_sum = 0.0
    vp_sum_sq = 0.0
    vt_sum = 0.0
    vt_sum_sq = 0.0
    v_count = 0
    n_batches = 0
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")

    permutation = torch.randperm(n_padded_transitions, device=cfg.device)
    if n_local_transitions < n_padded_transitions:
        valid_mask = permutation < n_local_transitions
        permutation = permutation % n_local_transitions
    else:
        valid_mask = None

    for batch_start in range(0, n_padded_transitions, cfg.batch_size):
        batch_end = batch_start + cfg.batch_size
        batch_indices = permutation[batch_start:batch_end]
        batch = rollout.get_batch(batch_indices, cfg.device)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            new_log_probs, entropy, new_values = compute_action_log_probs(
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

        with torch.no_grad():
            values_np = new_values.float().cpu().numpy()
            targets_np = batch["returns"].float().cpu().numpy()
            if valid_mask is not None:
                bw_np = valid_mask[batch_start:batch_end].cpu().numpy().astype(bool)
                values_np = values_np[bw_np]
                targets_np = targets_np[bw_np]
            vp_hist, _ = np.histogram(
                np.clip(values_np, *_VALUE_PRED_RANGE),
                bins=_VALUE_PRED_BINS, range=_VALUE_PRED_RANGE,
            )
            value_pred_hist += vp_hist.astype(np.float64)
            vt_hist, _ = np.histogram(
                np.clip(targets_np, *_VALUE_PRED_RANGE),
                bins=_VALUE_PRED_BINS, range=_VALUE_PRED_RANGE,
            )
            value_target_hist += vt_hist.astype(np.float64)
            vp_sum += values_np.sum()
            vp_sum_sq += (values_np ** 2).sum()
            vt_sum += targets_np.sum()
            vt_sum_sq += (targets_np ** 2).sum()
            v_count += len(values_np)

        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        entropy_sum += entropy_term.item()
        clip_fraction_sum += clip_frac.item()
        approx_kl_sum += approx_kl_batch.item()
        if math.isfinite(combined_norm):
            grad_norm_sum += combined_norm
            grad_norm_count += 1
        n_batches += 1

    if _world > 1:
        sync_tensor = torch.tensor(
            [policy_loss_sum, value_loss_sum, entropy_sum,
             clip_fraction_sum, approx_kl_sum, grad_norm_sum,
             float(n_batches), float(skipped_batches),
             float(grad_norm_count),
             vp_sum, vp_sum_sq, vt_sum, vt_sum_sq, float(v_count)],
            device=cfg.device,
        )
        dist.all_reduce(sync_tensor)
        policy_loss_sum = sync_tensor[0].item()
        value_loss_sum = sync_tensor[1].item()
        entropy_sum = sync_tensor[2].item()
        clip_fraction_sum = sync_tensor[3].item()
        approx_kl_sum = sync_tensor[4].item()
        grad_norm_sum = sync_tensor[5].item()
        n_batches = int(sync_tensor[6].item())
        skipped_batches = int(sync_tensor[7].item())
        grad_norm_count = int(sync_tensor[8].item())
        vp_sum = sync_tensor[9].item()
        vp_sum_sq = sync_tensor[10].item()
        vt_sum = sync_tensor[11].item()
        vt_sum_sq = sync_tensor[12].item()
        v_count = int(sync_tensor[13].item())

        vp_tensor = torch.tensor(value_pred_hist, device=cfg.device, dtype=torch.float64)
        dist.all_reduce(vp_tensor)
        value_pred_hist = vp_tensor.cpu().numpy()

        vt_tensor = torch.tensor(value_target_hist, device=cfg.device, dtype=torch.float64)
        dist.all_reduce(vt_tensor)
        value_target_hist = vt_tensor.cpu().numpy()

    n_batches = max(n_batches, 1)
    return {
        "policy_loss": policy_loss_sum / n_batches,
        "value_loss": value_loss_sum / n_batches,
        "entropy": entropy_sum / n_batches,
        "clip_fraction": clip_fraction_sum / n_batches,
        "approx_kl": approx_kl_sum / n_batches,
        "grad_norm": grad_norm_sum / max(grad_norm_count, 1),
        "value_pred_histogram": value_pred_hist,
        "value_target_histogram": value_target_hist,
        "skipped_batches": skipped_batches,
        "total_batches": n_batches,
        "vp_sum": vp_sum,
        "vp_sum_sq": vp_sum_sq,
        "vt_sum": vt_sum,
        "vt_sum_sq": vt_sum_sq,
        "v_count": v_count,
    }


# Test results

def build_test_results(episodes, eval_result, cfg):
    rows = []
    for episode_idx, episode in enumerate(episodes):
        dates = episode["dates"][cfg.lookback :]
        prices = episode["prices"][cfg.lookback :]
        for step in range(cfg.episode_length):
            mu = float(eval_result["model_mus"][episode_idx, step])
            rows.append({
                "symbol": episode["symbol"],
                "date": dates[step],
                "price": prices[step],
                "position": float(eval_result["model_positions"][episode_idx, step]),
                "mu": mu,
                "model_reward": float(
                    eval_result["model_rewards"][episode_idx, step],
                ),
                "baseline_reward": float(
                    eval_result["baseline_rewards"][episode_idx, step],
                ),
            })
    return rows


# DDP helpers

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
        result["mu_sum"],
        result["mu_sum_sq"],
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

    mu_hist_tensor = torch.tensor(
        result["mu_histogram"], device=device, dtype=torch.float64,
    )
    dist.all_reduce(mu_hist_tensor)

    return {
        "model_return": float(tensor[0].item() / total),
        "baseline_return": float(tensor[1].item() / total),
        "beat_rate": float(tensor[2].item() / total),
        "max_drawdown": float(tensor[3].item() / total),
        "mean_turnover": float(tensor[4].item() / total),
        "position_histogram": hist_tensor.cpu().numpy(),
        "mu_histogram": mu_hist_tensor.cpu().numpy(),
        "pos_sum": tensor[6].item(),
        "pos_sum_sq": tensor[7].item(),
        "n_actions": int(tensor[8].item()),
        "mu_sum": tensor[9].item(),
        "mu_sum_sq": tensor[10].item(),
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
    """Gather numeric test result arrays to rank 0 via tensor all_gather."""
    local = torch.zeros(n_local, episode_length, 4, device=device)
    local[..., 0] = torch.from_numpy(test_result["model_rewards"]).to(device)
    local[..., 1] = torch.from_numpy(test_result["baseline_rewards"]).to(device)
    local[..., 2] = torch.from_numpy(
        test_result["model_positions"],
    ).to(device)
    local[..., 3] = torch.from_numpy(test_result["model_mus"]).to(device)

    max_local = (n_total + _world - 1) // _world
    padded = torch.zeros(max_local, episode_length, 4, device=device)
    padded[:n_local] = local

    gathered = [torch.zeros_like(padded) for _ in range(_world)]
    dist.all_gather(gathered, padded)

    if _rank != 0:
        return None

    result = torch.zeros(n_total, episode_length, 4, device=device)
    for rank_idx in range(_world):
        rank_episodes = list(range(rank_idx, n_total, _world))
        result[rank_episodes] = gathered[rank_idx][:len(rank_episodes)]

    result_np = result.cpu().numpy()
    return {
        "model_rewards": result_np[..., 0],
        "baseline_rewards": result_np[..., 1],
        "model_positions": result_np[..., 2],
        "model_mus": result_np[..., 3],
    }


# Plotting and checkpointing

def _plot_heatmap(ax, fig, hist_list, n_bins, y_range, xlabel, ylabel, title, cmap="viridis"):
    """Plot a frequency heatmap from a list of per-epoch histograms."""
    if not hist_list:
        ax.set_visible(False)
        return
    heatmap_data = np.array(hist_list).T
    col_sums = heatmap_data.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    heatmap_data = heatmap_data / col_sums
    heatmap_data = np.log1p(heatmap_data * 100) / np.log1p(100)
    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        origin="lower",
        extent=[1, len(hist_list), y_range[0], y_range[1]],
        cmap=cmap,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Frequency (log scale)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


@timed
def plot_training(history, path, cfg):
    """Plot training curves on a 5x4 grid."""
    from matplotlib.ticker import MaxNLocator

    epochs = range(1, len(history["policy_loss"]) + 1)
    fig, axes = plt.subplots(5, 4, figsize=(24, 25))

    # Row 1 — Returns
    ablation_epochs = history.get("ablation_epochs", [])
    axes[0, 0].plot(epochs, history["train_return"], color="b", label="Train return")
    axes[0, 0].plot(epochs, history["train_baseline"], color="b", linestyle="--", label="Train baseline")
    axes[0, 0].plot(epochs, history["val_return"], color="r", label="Val return")
    axes[0, 0].plot(epochs, history["val_baseline"], color="r", linestyle="--", label="Val baseline")
    if ablation_epochs:
        axes[0, 0].plot(
            ablation_epochs, history["val_no_peers"], color="orange",
            label="Val no-peers",
        )
        axes[0, 0].plot(
            ablation_epochs, history["val_no_stock"], color="purple",
            label="Val no-stock",
        )
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].set_title("Episode Return")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["val_beat_rate"])
    axes[0, 1].set_ylabel("Beat Rate")
    axes[0, 1].set_title("Val Beat Rate")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history["val_max_drawdown"])
    axes[0, 2].set_ylabel("Max Drawdown")
    axes[0, 2].set_title("Val Max Drawdown")
    axes[0, 2].grid(True, alpha=0.3)

    axes[0, 3].plot(epochs, history["val_turnover"])
    axes[0, 3].set_ylabel("Turnover")
    axes[0, 3].set_title("Val Turnover")
    axes[0, 3].grid(True, alpha=0.3)

    # Row 2 — Policy behavior
    _plot_heatmap(
        axes[1, 0], fig,
        history.get("position_histogram", []),
        _POSITION_BINS, (-1.0, 1.0),
        "Epoch", "Position", "Val Position",
    )

    _plot_heatmap(
        axes[1, 1], fig,
        history.get("mu_histogram", []),
        _MU_BINS, _MU_RANGE,
        "Epoch", "Mu", "Val Mu",
    )

    axes[1, 2].plot(epochs, history["policy_loss"])
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].set_title("Policy Loss")
    axes[1, 2].grid(True, alpha=0.3)

    axes[1, 3].plot(epochs, history["value_loss"])
    axes[1, 3].set_ylabel("Loss")
    axes[1, 3].set_title("Value Loss")
    axes[1, 3].grid(True, alpha=0.3)

    # Row 3 — Exploration + PPO
    axes[2, 0].plot(epochs, history["entropy"])
    axes[2, 0].set_ylabel("Entropy")
    axes[2, 0].set_title("Entropy")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(epochs, history["log_std"])
    axes[2, 1].set_ylabel("Log Std")
    axes[2, 1].set_title("Policy Log Std")
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(epochs, history["clip_fraction"])
    axes[2, 2].set_ylabel("Clip Fraction")
    axes[2, 2].set_title("Clip Fraction")
    axes[2, 2].grid(True, alpha=0.3)

    axes[2, 3].plot(epochs, history["approx_kl"])
    axes[2, 3].set_ylabel("Approx KL")
    axes[2, 3].set_title("Approx KL")
    axes[2, 3].grid(True, alpha=0.3)

    # Row 4 — Value + AMP
    _plot_heatmap(
        axes[3, 0], fig,
        history.get("value_target_histogram", []),
        _VALUE_PRED_BINS, _VALUE_PRED_RANGE,
        "Epoch", "Value", "Value Target",
    )

    _plot_heatmap(
        axes[3, 1], fig,
        history.get("value_pred_histogram", []),
        _VALUE_PRED_BINS, _VALUE_PRED_RANGE,
        "Epoch", "Value", "Value Pred",
    )

    axes[3, 2].bar(list(epochs), history["skip_rate"])
    axes[3, 2].set_xlabel("Epoch")
    axes[3, 2].set_ylabel("Skip Rate")
    axes[3, 2].set_title("Skip Rate")
    axes[3, 2].grid(True, alpha=0.3)

    axes[3, 3].plot(epochs, history["loss_scale"])
    axes[3, 3].set_xlabel("Epoch")
    axes[3, 3].set_ylabel("Loss Scale")
    axes[3, 3].set_title("Loss Scale")
    axes[3, 3].set_yscale("log")
    axes[3, 3].grid(True, alpha=0.3)

    # Row 5 — Gradients
    axes[4, 0].plot(epochs, history["grad_norm"])
    axes[4, 0].set_xlabel("Epoch")
    axes[4, 0].set_ylabel("Grad Norm")
    axes[4, 0].set_title("Grad Norm")
    axes[4, 0].set_yscale("log")
    axes[4, 0].grid(True, alpha=0.3)

    has_grad_diag = (
        len(history.get("grad_norm_policy", [])) > 0
        and len(history["grad_norm_policy"]) == len(history["policy_loss"])
    )
    if has_grad_diag:
        axes[4, 1].plot(epochs, history["grad_norm_policy"], color="r", label="Policy")
        axes[4, 1].plot(epochs, history["grad_norm_value"], color="b", label="Value")
        axes[4, 1].legend(fontsize=7)
    axes[4, 1].set_xlabel("Epoch")
    axes[4, 1].set_ylabel("Grad Norm")
    axes[4, 1].set_title("Per-Term Grad Norm")
    axes[4, 1].set_yscale("log")
    axes[4, 1].grid(True, alpha=0.3)

    if has_grad_diag and len(history.get("grad_cosine_sim", [])) == len(history["grad_norm_policy"]):
        axes[4, 2].plot(epochs, history["grad_cosine_sim"])
        axes[4, 2].axhline(0, linestyle=":")
    axes[4, 2].set_xlabel("Epoch")
    axes[4, 2].set_ylabel("Cosine Sim")
    axes[4, 2].set_title("Grad Cosine Sim")
    axes[4, 2].grid(True, alpha=0.3)

    has_peer_diag = len(history.get("self_weight", [])) > 0
    if has_peer_diag:
        peer_epochs = range(1, len(history["self_weight"]) + 1)
        ax_peer = axes[4, 3]
        ax_peer.plot(peer_epochs, history["self_weight"], color="b", label="Self-weight")
        ax_peer.set_ylabel("Self-weight", color="b")
        ax_peer.tick_params(axis="y", labelcolor="b")
        ax_twin = ax_peer.twinx()
        ax_twin.plot(peer_epochs, history["attn_entropy"], color="r", label="Entropy")
        ax_twin.set_ylabel("Attn Entropy", color="r")
        ax_twin.tick_params(axis="y", labelcolor="r")
        ax_peer.set_xlabel("Epoch")
        ax_peer.set_title("Peer Attention")
        ax_peer.grid(True, alpha=0.3)
    else:
        axes[4, 3].set_visible(False)

    for ax in axes.flat:
        if ax.get_visible():
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def warmup_cosine_lr(base_lr, epoch, n_epochs, warmup_epochs):
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# Training loop

def train(cfg, trial=None, data=None, sweep_mode=False):
    global _rank, _world

    distributed = "RANK" in os.environ
    if distributed and not sweep_mode:
        dist.init_process_group("nccl")
        _rank = dist.get_rank()
        _world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        cfg.device = f"cuda:{local_rank}"
    is_main = _rank == 0

    os.makedirs(cfg.save_dir, exist_ok=True)

    _log()
    if not sweep_mode:
        _log("[Configuration]")
        for key, value in cfg.to_dict().items():
            _log(f"    {key:<20s}: {value}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if data is not None:
        peer_info, train_data, val_data, test_data, stock_feature_cols = data
    else:
        peer_info, train_data, val_data, test_data, stock_feature_cols = (
            _load_all_data(cfg, log=not sweep_mode)
        )

    feature_groups = build_feature_groups(stock_feature_cols)

    if not sweep_mode:
        _log()
        _log("[Model]")
    base_model = PolicyNetwork(feature_groups, cfg).to(cfg.device)

    model = base_model
    if distributed:
        model = DDP(base_model, device_ids=[local_rank])
        torch.manual_seed(cfg.seed + _rank)

    n_params = sum(p.numel() for p in base_model.parameters())
    if not sweep_mode:
        _log(f"    {'Feature groups':<20s}: {len(feature_groups)}")
        _log(f"    {'Peers (K)':<20s}: {peer_info['n_peers']}")
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
        lr=effective_lr, eps=cfg.adam_eps,
    )
    reward_normalizer = RewardNormalizer(cfg.pytorch_eps, cfg.reward_ema_decay)
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    device_label = f"{_world} GPUs" if distributed else cfg.device
    if not sweep_mode:
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

        if sweep_mode and (
            start_epoch > cfg.n_epochs or patience_counter >= cfg.patience
        ):
            _log(f"    Already complete, returning best score {best_score:.4f}")
            return best_score

    def _build_checkpoint():
        return {
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "feature_cols": stock_feature_cols,
            "feature_groups": feature_groups,
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
        current_entropy_coeff = cfg.entropy_coeff

        _log()
        _log(f"[Epoch {epoch}/{cfg.n_epochs}]")

        epoch_rng = np.random.default_rng(cfg.seed + epoch * 7919)
        train_episodes = generate_episodes(train_data, peer_info, cfg, epoch_rng)
        rank_train_episodes = train_episodes[_rank::_world]

        val_rng = np.random.default_rng(cfg.seed + epoch * 7919 + 1)
        val_episodes = generate_episodes(val_data, peer_info, cfg, val_rng)
        rank_val_episodes = val_episodes[_rank::_world]

        rollout, model_return, baseline_return = collect_training_rollout(
            model, rank_train_episodes, cfg,
            reward_normalizer, distributed,
        )

        if distributed:
            model_return, baseline_return = _allreduce_means(
                model_return, baseline_return,
                local_count=len(rank_train_episodes), device=cfg.device,
            )

        _gae_t = time.perf_counter()
        n_local_transitions = rollout.compute_gae(
            cfg.gamma, cfg.gae_lambda, cfg.pytorch_eps,
            distributed, cfg.device,
        )
        _log(f"    compute_gae ({time.perf_counter() - _gae_t:.1f}s)")
        if _world > 1:
            max_transitions = _allreduce_max_int(
                n_local_transitions, cfg.device,
            )
        else:
            max_transitions = n_local_transitions
        n_padded_transitions = (
            math.ceil(max_transitions / cfg.batch_size) * cfg.batch_size
        )

        grad_contrib = None
        peer_diag = None
        if cfg.grad_diagnostic:
            _diag_t = time.perf_counter()
            diag_n = min(cfg.batch_size, n_local_transitions)
            diag_indices = np.random.choice(
                n_local_transitions, diag_n, replace=False,
            )
            diag_batch = rollout.get_batch(diag_indices, cfg.device)
            grad_contrib = diagnose_gradient_contributions(
                model, diag_batch, cfg,
            )
            peer_diag = diagnose_peer_attention(model, diag_batch)
            _log(f"    diagnose_gradients ({time.perf_counter() - _diag_t:.1f}s)")

        epoch_losses = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "clip_fraction": 0.0, "approx_kl": 0.0, "grad_norm": 0.0,
        }
        epoch_value_pred_hist = np.zeros(_VALUE_PRED_BINS, dtype=np.float64)
        epoch_value_target_hist = np.zeros(_VALUE_PRED_BINS, dtype=np.float64)
        epoch_vp_sum = 0.0
        epoch_vp_sum_sq = 0.0
        epoch_vt_sum = 0.0
        epoch_vt_sum_sq = 0.0
        epoch_v_count = 0
        epoch_skipped = 0
        epoch_total = 0
        _ppo_t = time.perf_counter()
        for _ in range(cfg.ppo_epochs):
            losses = ppo_update(
                model, optimizer, scaler,
                rollout, cfg,
                n_local_transitions, n_padded_transitions,
                current_entropy_coeff,
            )
            for k in epoch_losses:
                epoch_losses[k] += losses[k]
            epoch_value_pred_hist += losses["value_pred_histogram"]
            epoch_value_target_hist += losses["value_target_histogram"]
            epoch_vp_sum += losses["vp_sum"]
            epoch_vp_sum_sq += losses["vp_sum_sq"]
            epoch_vt_sum += losses["vt_sum"]
            epoch_vt_sum_sq += losses["vt_sum_sq"]
            epoch_v_count += losses["v_count"]
            epoch_skipped += losses["skipped_batches"]
            epoch_total += losses["total_batches"]
        _log(f"    ppo_update ({time.perf_counter() - _ppo_t:.1f}s)")
        for k in epoch_losses:
            epoch_losses[k] /= cfg.ppo_epochs
        skip_rate = epoch_skipped / max(epoch_total, 1)
        loss_scale = scaler.get_scale()
        current_log_std = base_model.log_std.item()
        bias_scale = base_model.trunk.peer_attn.bias_scale.item()
        mean_valid_peers = np.mean([
            (~ep["peer_mask"]).sum() for ep in train_episodes
        ])

        del rollout
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

        val_result = evaluate_episodes(model, rank_val_episodes, cfg)
        if distributed:
            val_aggregated = _allreduce_eval(
                val_result, len(rank_val_episodes), cfg.device,
            )
        else:
            val_aggregated = val_result

        if cfg.ablation:
            ablation_result = evaluate_ablated(
                model, rank_val_episodes, cfg,
            )
            if distributed:
                ablation_aggregated = _allreduce_ablation(
                    ablation_result, len(rank_val_episodes), cfg.device,
                )
            else:
                ablation_aggregated = ablation_result
            history["ablation_epochs"].append(epoch)
            history["val_no_peers"].append(ablation_aggregated["no_peers"])
            history["val_no_stock"].append(ablation_aggregated["no_stock"])

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
            ("entropy", epoch_losses["entropy"]),
            ("clip_fraction", epoch_losses["clip_fraction"]),
            ("approx_kl", epoch_losses["approx_kl"]),
            ("grad_norm", epoch_losses["grad_norm"]),
            ("loss_scale", loss_scale),
            ("skip_rate", skip_rate),
            ("log_std", current_log_std),
            ("entropy_coeff", current_entropy_coeff),
        ]:
            history[key].append(value)

        if grad_contrib is not None:
            history["grad_norm_policy"].append(grad_contrib["policy"])
            history["grad_norm_value"].append(grad_contrib["value"])
            history["grad_cosine_sim"].append(grad_contrib["cosine_sim"])

        history["bias_scale"].append(bias_scale)
        history["mean_valid_peers"].append(mean_valid_peers)
        if peer_diag is not None:
            history["self_weight"].append(peer_diag["self_weight"])
            history["attn_entropy"].append(peer_diag["attn_entropy"])

        pos_hist = (
            val_aggregated["position_histogram"]
            if "position_histogram" in val_aggregated
            else np.zeros(_POSITION_BINS, dtype=np.float64)
        )
        history["position_histogram"].append(pos_hist)
        history["value_pred_histogram"].append(epoch_value_pred_hist)
        history["value_target_histogram"].append(epoch_value_target_hist)

        mu_hist = (
            val_aggregated["mu_histogram"]
            if "mu_histogram" in val_aggregated
            else np.zeros(_MU_BINS, dtype=np.float64)
        )
        history["mu_histogram"].append(mu_hist)

        n_actions = val_aggregated.get("n_actions", 0)
        if n_actions > 0:
            pos_mean = val_aggregated["pos_sum"] / n_actions
            pos_std = math.sqrt(max(0, val_aggregated["pos_sum_sq"] / n_actions - pos_mean ** 2))
            mu_mean = val_aggregated["mu_sum"] / n_actions
            mu_std = math.sqrt(max(0, val_aggregated["mu_sum_sq"] / n_actions - mu_mean ** 2))
        else:
            pos_mean, pos_std = 0.0, 0.0
            mu_mean, mu_std = 0.0, 0.0
        if epoch_v_count > 0:
            vp_mean = epoch_vp_sum / epoch_v_count
            vp_std = math.sqrt(max(0, epoch_vp_sum_sq / epoch_v_count - vp_mean ** 2))
            vt_mean = epoch_vt_sum / epoch_v_count
            vt_std = math.sqrt(max(0, epoch_vt_sum_sq / epoch_v_count - vt_mean ** 2))
        else:
            vp_mean, vp_std = 0.0, 0.0
            vt_mean, vt_std = 0.0, 0.0

        window = cfg.patience_smoothing
        recent_returns = history["val_return"][-window:]
        current_score = float(np.mean(recent_returns))

        improved = current_score > best_score
        if improved:
            best_score = current_score
            patience_counter = 0
        elif epoch > cfg.warmup_epochs:
            patience_counter += 1

        if is_main:
            ckpt = _build_checkpoint()
            if improved:
                torch.save(ckpt, os.path.join(cfg.save_dir, "model_best.pt"))
            torch.save(ckpt, os.path.join(cfg.save_dir, "model_latest.pt"))
            plot_training(
                dict(history),
                os.path.join(cfg.save_dir, "training_plots.png"),
                cfg,
            )

        _log(f"    {'Train episodes':<20s}: {len(train_episodes)}")
        _log(f"    {'Val episodes':<20s}: {len(val_episodes)}")
        _log(f"    {'Train return':<20s}: {model_return:.4f}")
        _log(f"    {'Train baseline':<20s}: {baseline_return:.4f}")
        _log(f"    {'Val return':<20s}: {val_aggregated['model_return']:.4f}")
        _log(f"    {'Val baseline':<20s}: {val_aggregated['baseline_return']:.4f}")
        if cfg.ablation:
            _log(f"    {'Val no-peers':<20s}: {ablation_aggregated['no_peers']:.4f}")
            _log(f"    {'Val no-stock':<20s}: {ablation_aggregated['no_stock']:.4f}")
        _log(f"    {'Val beat rate':<20s}: {val_aggregated['beat_rate']:.4f}")
        _log(f"    {'Val max drawdown':<20s}: {val_aggregated['max_drawdown']:.4f}")
        _log(f"    {'Val turnover':<20s}: {val_aggregated['mean_turnover']:.4f}")
        _log(f"    {'Val position mean':<20s}: {pos_mean:.4f}")
        _log(f"    {'Val position std':<20s}: {pos_std:.4f}")
        _log(f"    {'Val mu mean':<20s}: {mu_mean:.4f}")
        _log(f"    {'Val mu std':<20s}: {mu_std:.4f}")
        _log(f"    {'Policy loss':<20s}: {epoch_losses['policy_loss']:.4f}")
        _log(f"    {'Value loss':<20s}: {epoch_losses['value_loss']:.4f}")
        _log(f"    {'Entropy':<20s}: {epoch_losses['entropy']:.4f}")
        _log(f"    {'Log std':<20s}: {current_log_std:.4f}")
        _log(f"    {'Clip fraction':<20s}: {epoch_losses['clip_fraction']:.4f}")
        _log(f"    {'Approx KL':<20s}: {epoch_losses['approx_kl']:.4f}")
        _log(f"    {'Value target mean':<20s}: {vt_mean:.4f}")
        _log(f"    {'Value target std':<20s}: {vt_std:.4f}")
        _log(f"    {'Value pred mean':<20s}: {vp_mean:.4f}")
        _log(f"    {'Value pred std':<20s}: {vp_std:.4f}")
        _log(f"    {'Skip rate':<20s}: {skip_rate:.4f} ({epoch_skipped}/{epoch_total})")
        _log(f"    {'Loss scale':<20s}: {loss_scale:.1f}")
        _log(f"    {'Grad norm':<20s}: {epoch_losses['grad_norm']:.4f}")
        if grad_contrib is not None:
            _log(f"    {'Policy grad norm':<20s}: {grad_contrib['policy']:.4f}")
            _log(f"    {'Value grad norm':<20s}: {grad_contrib['value']:.4f}")
            _log(f"    {'Grad cosine sim':<20s}: {grad_contrib['cosine_sim']:.4f}")
        _log(f"    {'Bias scale':<20s}: {bias_scale:.4f}")
        _log(f"    {'Mean valid peers':<20s}: {mean_valid_peers:.1f}")
        if peer_diag is not None:
            _log(f"    {'Self-weight':<20s}: {peer_diag['self_weight']:.4f}")
            _log(f"    {'Attn entropy':<20s}: {peer_diag['attn_entropy']:.4f}")
        _log(f"    {'Current score':<20s}: {current_score:.4f}")
        _log(f"    {'Best score':<20s}: {best_score:.4f}")
        _log(f"    {'Patience':<20s}: {patience_counter}/{cfg.patience}")
        _log(f"    {'LR':<20s}: {current_lr:.2e}")
        _log(f"    {'Entropy coeff':<20s}: {current_entropy_coeff:.4f}")

        if patience_counter >= cfg.patience:
            _log(f"    Early stopping at epoch {epoch} (patience {cfg.patience})")
            break

        if trial is not None:
            trial.report(current_score, epoch)
            if trial.should_prune():
                _log(f"    Optuna pruned at epoch {epoch}")
                if distributed and not sweep_mode:
                    dist.destroy_process_group()
                import optuna
                raise optuna.TrialPruned()

    _log()
    _log(f"    Training complete, best score: {best_score:.4f}")

    best_path = os.path.join(cfg.save_dir, "model_best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])

    test_episode_rng = np.random.default_rng(cfg.seed + 1000)
    test_episodes = generate_episodes(test_data, peer_info, cfg, test_episode_rng)
    rank_test_episodes = test_episodes[_rank::_world]

    _log()
    _log("[Final Evaluation]")
    test_result = evaluate_episodes(model, rank_test_episodes, cfg)
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
        _log(f"    {'Saved':<20s}: {results_path}")

    if distributed and not sweep_mode:
        dist.destroy_process_group()

    return best_score


# Optuna sweep

def sweep(cfg):
    """Run Optuna hyperparameter sweep with TPE sampler and median pruning."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

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

    if is_main:
        _log()
        _log("[Configuration]")
        for key, value in cfg.to_dict().items():
            _log(f"    {key:<20s}: {value}")

    peer_info, train_data, val_data, test_data, stock_feature_cols = (
        _load_all_data(cfg, log=is_main)
    )
    data = (peer_info, train_data, val_data, test_data, stock_feature_cols)

    if is_main:
        feature_groups = build_feature_groups(stock_feature_cols)
        _log()
        _log("[Model]")
        _log(f"    {'Feature groups':<20s}: {len(feature_groups)}")
        _log(f"    {'Peers (K)':<20s}: {peer_info['n_peers']}")
        tmp_model = PolicyNetwork(feature_groups, cfg)
        n_params = sum(p.numel() for p in tmp_model.parameters())
        _log(f"    {'Parameters':<20s}: {n_params:,}")
        del tmp_model

    _SWEEP_KEYS = [
        "dropout", "value_coeff", "entropy_coeff", "lr",
    ]

    def _param_hash(params):
        return hashlib.md5(
            json.dumps({k: params[k] for k in sorted(_SWEEP_KEYS)}).encode(),
        ).hexdigest()[:12]

    study = None
    if is_main:
        study = optuna.create_study(
            study_name="ppo_stock_sweep",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=10),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=20,
            ),
            storage=cfg.sweep_db,
            load_if_exists=True,
        )

        retryable = (optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING)
        for t in study.trials:
            if t.state in retryable and t.params:
                h = _param_hash(t.params)
                if os.path.exists(os.path.join("sweep_runs", h, "model_latest.pt")):
                    study.enqueue_trial(t.params)
                    _log(f"    Re-enqueued trial {t.number} ({h})")

    results_path = os.path.join("sweep_runs", "sweep_results.csv")
    plot_path = os.path.join("sweep_runs", "sweep_best_scores.png")

    def _save_results():
        rows = []
        for t in study.trials:
            row = {"trial": t.number, "state": t.state.name, "score": t.value}
            row.update(t.params)
            rows.append(row)
        pd.DataFrame(rows).to_csv(results_path, index=False)

        complete = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not complete:
            return
        scores = [t.value for t in complete]
        best_so_far = np.maximum.accumulate(scores)
        trials_x = range(1, len(scores) + 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(trials_x, scores, color="b", s=15, label="Trial score")
        ax.plot(trials_x, best_so_far, color="r", linewidth=2, label="Best score")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.set_title("Sweep Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    while True:
        params = None
        trial_number = -1
        if is_main:
            n_complete = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            if n_complete >= cfg.sweep_trials:
                params = None
            else:
                trial = study.ask()
                trial.suggest_float("dropout", 0.0, 0.2)
                trial.suggest_float("value_coeff", 0.01, 1.0, log=True)
                trial.suggest_float("entropy_coeff", 0.001, 0.1, log=True)
                trial.suggest_float("lr", 1e-5, 1e-3, log=True)
                params = trial.params
                trial_number = trial.number

        if distributed:
            broadcast_list = [params]
            dist.broadcast_object_list(broadcast_list, src=0)
            params = broadcast_list[0]

        if params is None:
            break

        for k, v in params.items():
            setattr(cfg, k, v)
        param_hash = _param_hash(params)
        cfg.save_dir = os.path.join("sweep_runs", param_hash)

        if is_main:
            _log()
            _log(f"[Optuna Trial {trial_number}]")
            _log(f"    {'hash':<20s}: {param_hash}")
            for key in _SWEEP_KEYS:
                _log(f"    {key:<20s}: {getattr(cfg, key)}")

        score = train(cfg, data=data, sweep_mode=True)

        if is_main:
            study.tell(trial, score)
            _save_results()

    if is_main:
        _log()
        _log("[Sweep Complete]")
        _log(f"    {'Best trial':<20s}: {study.best_trial.number}")
        _log(f"    {'Best score':<20s}: {study.best_value:.4f}")
        for key, value in study.best_params.items():
            _log(f"    {key:<20s}: {value}")
        _log(f"    {'Saved':<20s}: {results_path}")

    if distributed:
        dist.destroy_process_group()


# Launch

def _spawn_ddp_worker(local_rank, cfg_dict, n_gpus, port):
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(n_gpus)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    cfg = Config()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)
    if cfg.sweep:
        sweep(cfg)
    else:
        train(cfg)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="PPO + Transformer RL for stock trading")

    def _parse_bool(v):
        if v.lower() not in ("true", "false"):
            raise argparse.ArgumentTypeError(
                f"expected 'true' or 'false', got '{v}'")
        return v.lower() == "true"

    for name, ann_type in Config.__annotations__.items():
        default = getattr(cfg, name)
        flag = f"--{name}"
        if ann_type is bool:
            parser.add_argument(flag, type=_parse_bool,
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
        port = _find_free_port()
        _log(f"Auto launching DDP on {n_gpus} GPUs (port {port})")
        mp.spawn(
            _spawn_ddp_worker,
            args=(cfg.to_dict(), n_gpus, port),
            nprocs=n_gpus,
            join=True,
        )
    elif cfg.sweep:
        sweep(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
