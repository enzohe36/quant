"""
PPO + Transformer RL for stock trading.

Stock CSV: symbol, date, open, close, <stock_feat_1>, ...
Market CSV: date, <mkt_feat_1>, ...  (columns except date start with "mkt_")

Market features loaded separately, aligned to stock data at each lookback
window's last date. Features with "mkt_" prefix route through cross-attention.

Continuous position sizing in [-1, 1] with squashed Gaussian policy (tanh).

Architecture: shared-trunk PolicyNetwork (SB3-style shared feature extractor) with
transformer trunk. Orthogonal initialization on MLP heads (gain sqrt(2)
hidden, 0.01 policy output, 1.0 value output).

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
    # Data
    train_path: str = "train.csv"
    val_path: str = "val.csv"
    test_path: str = "test.csv"
    market_path: str = "mkt_feats.csv"
    save_dir: str = "checkpoints"

    # Environment
    lookback: int = 60
    episode_length: int = 100
    transaction_cost: float = 0.001

    # Architecture
    d_model: int = 256
    d_ff: int = 512
    n_heads: int = 8
    n_layers: int = 3
    head_hidden_dim: int = 128
    position_dim: int = 16
    dropout: float = 0.0
    market_dropout: float = 0.5
    log_std_init: float = -0.5
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    # RL algorithm
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_ema_decay: float = 0.99

    # PPO
    ppo_epochs: int = 4
    batch_size: int = 4096
    policy_clip: float = 0.2

    # Loss
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01

    # Optimizer
    lr: float = 1e-4
    adam_eps: float = 1e-5
    weight_decay: float = 0.02
    xattn_weight_decay: float = 0.05
    grad_clip: float = 0.5

    # Training schedule
    n_epochs: int = 200
    warmup_epochs: int = 5
    patience: int = 20
    patience_smoothing: int = 10
    seed: int = 42

    # Runtime
    inference_batch: int = 16384
    env_workers: int = 0
    use_amp: bool = True
    pytorch_eps: float = 1e-8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Diagnostics
    ablation: bool = True
    diagnostic: bool = True

    # Sweep
    sweep: bool = False
    sweep_trials: int = 100
    sweep_db: str = "sqlite:///ppo_sweep.db"

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__class__.__annotations__}


# Feature grouping

def _remap_groups(groups, feature_indices):
    """Remap group indices from global (all_feature_cols) to local (0-based within slice)."""
    global_to_local = {g: l for l, g in enumerate(feature_indices)}
    return {name: [global_to_local[i] for i in idxs] for name, idxs in groups.items()}


def split_feature_groups(column_names):
    groups = defaultdict(list)
    for col_idx, col_name in enumerate(column_names):
        prefix = col_name.split(".")[0]
        groups[prefix].append(col_idx)
    stock_groups, market_groups = {}, {}
    for group_name, group_indices in groups.items():
        target = market_groups if group_name.startswith("mkt") else stock_groups
        target[group_name] = group_indices
    stock_feature_indices = sorted(idx for vals in stock_groups.values() for idx in vals)
    market_feature_indices = sorted(idx for vals in market_groups.values() for idx in vals)
    return stock_groups, market_groups, stock_feature_indices, market_feature_indices


# Vectorized environment

class VecEnv:
    """Batched single-stock trading environment with continuous positions in [-1, 1].

    Observations concatenate [stock_window, market_window] along the feature
    axis. For each lookback window the stock slice is ``lookback`` consecutive
    rows from the episode's stock_features. The market slice is ``lookback``
    consecutive rows from the global market array, ending at the market row
    whose date matches the stock window's last date.

    Execution model (open-to-open):
        - Agent observes features through day d's close.
        - Position executes at day d+1's open.
        - Reward = log(open[d+2]/open[d+1]), decomposed as
          log(close[d+1]/open[d+1]) + log(open[d+2]/close[d+1]),
          where the overnight gap is clamped per A-share price limits.
    """

    def __init__(self, episodes, lookback, transaction_cost, global_market_features):
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        self.n_envs = len(episodes)
        self.stock_features = np.stack([ep["stock_features"] for ep in episodes])
        self.log_returns = np.stack([ep["log_returns"] for ep in episodes])
        self.market_date_indices = np.stack([ep["market_indices"] for ep in episodes])
        self.global_market_features = global_market_features
        self.total_steps = self.stock_features.shape[1]
        self._lookback_offsets = np.arange(-lookback + 1, 1)

    def _build_observation(self):
        stock_window = self.stock_features[
            :, self.current_step - self.lookback : self.current_step
        ]
        market_end_idx = self.market_date_indices[:, self.current_step - 1]
        market_lookup = market_end_idx[:, None] + self._lookback_offsets[None, :]
        market_window = self.global_market_features[market_lookup]
        return np.concatenate([stock_window, market_window], axis=-1).astype(np.float32)

    def reset(self):
        self.current_step = self.lookback
        self.position = np.zeros(self.n_envs, dtype=np.float64)
        return self._build_observation(), self.position.astype(np.float32)

    def step(self, actions):
        new_position = np.clip(actions.astype(np.float64), -1.0, 1.0)
        reward = (
            self.position * self.log_returns[:, self.current_step]
            + np.log1p(-self.transaction_cost * np.abs(new_position - self.position))
        ).astype(np.float32)
        self.position = new_position
        self.current_step += 1
        done = self.current_step >= self.total_steps
        obs = None if done else self._build_observation()
        return obs, reward, done, self.position.astype(np.float32)

    def terminal_observation(self):
        """Build observation at the truncation boundary (current_step == total_steps)."""
        return self._build_observation()


# Parallel vectorized environment


class ParallelVecEnv:
    """VecEnv that distributes episodes across worker threads.

    Numpy operations release the GIL, so threads achieve true multi-core
    parallelism for the observation building (fancy indexing into the global
    market array) and environment stepping that dominate CPU time during
    rollout collection. Zero IPC overhead compared to multiprocessing.
    """

    def __init__(self, episodes, lookback, transaction_cost,
                 global_market_features, n_workers):
        n_workers = max(1, min(n_workers, len(episodes)))
        self.n_envs = len(episodes)
        self.n_workers = n_workers
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
            return VecEnv(
                episodes[start:end], lookback, transaction_cost,
                global_market_features,
            )

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
        self.n_workers = len(self._shards)

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


class Trunk(nn.Module):
    """Encoder backbone: group projections, cross-attention, transformer, pooling.

    Shared between policy and value heads (SB3-style shared feature extractor).
    """

    def __init__(self, stock_groups, market_groups, stock_feature_indices,
                 market_feature_indices, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "stock_idx", torch.tensor(stock_feature_indices, dtype=torch.long),
        )
        self.register_buffer(
            "market_idx", torch.tensor(market_feature_indices, dtype=torch.long),
        )
        self.has_market = len(market_feature_indices) > 0

        local_stock_groups = _remap_groups(stock_groups, stock_feature_indices)
        self.stock_proj = GroupProjection(local_stock_groups, cfg.d_model)
        self.stock_norm = nn.LayerNorm(cfg.d_model)
        self.pos_emb = nn.Parameter(
            torch.randn(1, cfg.lookback, cfg.d_model) * 0.02,
        )

        if self.has_market:
            local_market_groups = _remap_groups(market_groups, market_feature_indices)
            self.market_proj = GroupProjection(local_market_groups, cfg.d_model)
            self.market_norm = nn.LayerNorm(cfg.d_model)
            self.xattn = nn.MultiheadAttention(
                cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
            )
            self.xattn_norm = nn.LayerNorm(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, cfg.n_layers)

        self.position_proj = nn.Linear(1, cfg.position_dim)

    @property
    def output_dim(self):
        return self.cfg.d_model + self.cfg.position_dim

    def forward(self, x, position):
        hidden = self.stock_norm(self.stock_proj(x[..., self.stock_idx])) + self.pos_emb
        if self.has_market:
            market_emb = self.market_norm(self.market_proj(x[..., self.market_idx]))
            xattn_out, _ = self.xattn(
                query=hidden, key=market_emb, value=market_emb,
            )
            if self.training and self.cfg.market_dropout > 0:
                keep_mask = torch.bernoulli(torch.full(
                    (xattn_out.shape[0], 1, 1),
                    1 - self.cfg.market_dropout,
                    device=xattn_out.device, dtype=xattn_out.dtype,
                ))
                xattn_out = xattn_out * keep_mask / (1 - self.cfg.market_dropout)
            hidden = self.xattn_norm(hidden + xattn_out)
        pooled = self.transformer(hidden).mean(dim=1)
        pos_emb = self.position_proj(position.unsqueeze(-1))
        return torch.cat([pooled, pos_emb], dim=-1)


class PolicyNetwork(nn.Module):
    """Shared-trunk policy network (SB3-style shared feature extractor).

    One transformer trunk (feature extractor) shared between policy and value.
    Separate MLP heads branch from the pooled trunk output.
    """

    def __init__(self, stock_groups, market_groups, stock_feature_indices,
                 market_feature_indices, cfg):
        super().__init__()
        self.log_std_min = cfg.log_std_min
        self.log_std_max = cfg.log_std_max
        self.trunk = Trunk(
            stock_groups, market_groups,
            stock_feature_indices, market_feature_indices, cfg,
        )
        head_input_dim = self.trunk.output_dim
        self.policy_head = nn.Sequential(
            nn.Linear(head_input_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((), cfg.log_std_init))
        self.value_head = nn.Sequential(
            nn.Linear(head_input_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def forward(self, x, position):
        pooled = self.trunk(x, position)
        mu = self.policy_head(pooled).squeeze(-1)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        values = self.value_head(pooled).squeeze(-1)
        return mu, log_std.expand_as(mu), values


def _ortho_init(module):
    """SB3-style orthogonal initialization.

    Hidden linear layers get gain sqrt(2), output layers are identified as the
    last Linear in each Sequential head and get a role-specific gain:
      - policy_head output → 0.01 (small initial actions)
      - value_head output  → 1.0  (neutral value scale)
    All biases are zeroed. Trunk layers (projections, transformer) keep their
    default PyTorch init, which is already reasonable for attention/FFN.
    """
    for name, sub in module.named_modules():
        if not isinstance(sub, nn.Sequential):
            continue
        # Determine output gain from head name
        if "policy_head" in name:
            output_gain = 0.01
        elif "value_head" in name:
            output_gain = 1.0
        else:
            continue
        linears = [m for m in sub if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linears):
            gain = output_gain if i == len(linears) - 1 else math.sqrt(2)
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0.0)


def _squashed_gaussian_log_prob(u, mu, log_std):
    """Log-prob of a tanh-squashed Gaussian, numerically stable.

    u:       pre-tanh sample (the raw Gaussian draw)
    mu:      mean of the Gaussian
    log_std: log standard deviation

    Gaussian log-prob is computed entirely in log-space to avoid
    exponentiating log_std (cfg.log_std_min guarantees std >= exp(-5)).
    Uses the identity log(1 - tanh²(u)) = 2(log(2) - u - softplus(-2u))
    to avoid the catastrophic cancellation in 1 - tanh²(u) when |u| > 5.
    """
    # Gaussian log-prob in log-space: -0.5*(u-mu)²/σ² = -0.5*(u-mu)²*exp(-2*log_std)
    gauss_log_prob = (
        -0.5 * (u - mu) ** 2 * torch.exp(-2.0 * log_std)
        - log_std
        - 0.5 * math.log(2 * math.pi)
    )
    # Stable squash correction: log(1 - tanh²(u)) = 2(log(2) - u - softplus(-2u))
    squash_correction = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
    return gauss_log_prob + squash_correction


def sample_stochastic(model, states, position):
    mu, log_std, values = model(states, position)
    std = log_std.exp()
    u = mu + std * torch.randn_like(std)        # reparameterized sample
    log_prob = _squashed_gaussian_log_prob(u.float(), mu.float(), log_std.float())
    action = u.tanh()                            # squashed into (-1, 1)
    return action, u, log_prob, values           # store u (pre-tanh) as raw


def select_greedy(model, states, position):
    """Greedy action with mu for diagnostics."""
    mu, log_std, values = model(states, position)
    action = mu.tanh()
    return action, values, mu


def evaluate_actions(model, states, raw_actions, position):
    """raw_actions stores the pre-tanh Gaussian sample u."""
    mu, log_std, values = model(states, position)
    mu, log_std = mu.float(), log_std.float()
    u = raw_actions.float()

    log_prob = _squashed_gaussian_log_prob(u, mu, log_std)

    # No closed-form entropy for squashed Gaussian (matching SB3).
    # Use sample-based estimate: H ≈ -E[log π(a)]
    entropy = -log_prob

    return log_prob, entropy, values



# Batched inference (GPU chunking for large episode counts)

def _batched_stochastic(model, obs, pos, cfg):
    """Run sample_stochastic in inference_batch chunks."""
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
            a, r, lp, v = sample_stochastic(model, s, p)
            actions[i:j] = a.cpu().numpy()
            raw[i:j] = r.cpu().numpy()
            logp[i:j] = lp.cpu().numpy()
            vals[i:j] = v.cpu().numpy()
    return actions, raw, logp, vals


def _batched_greedy(model, obs, pos, cfg):
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
            a, v, mu = select_greedy(model, s, p)
            actions[i:j] = a.cpu().numpy()
            vals[i:j] = v.cpu().numpy()
            mus[i:j] = mu.cpu().numpy()
    return actions, vals, mus


# Rollout buffer

class RolloutBuffer:
    """Pre-allocated buffer for PPO rollout data.

    All arrays are shaped (n_episodes, episode_length), enabling vectorized
    GAE computation and batch construction via advanced numpy indexing.
    """

    def __init__(self, n_episodes, episode_length, lookback, global_market_features):
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.lookback = lookback
        self.global_market_features = global_market_features

        self.raw_actions = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.rewards = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.log_probs = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.values = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.positions = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.final_values = np.empty(n_episodes, dtype=np.float32)

        self.all_stock_features = None
        self.all_market_indices = None

    def register_episodes(self, episodes):
        self.all_stock_features = np.stack(
            [ep["stock_features"] for ep in episodes],
        )
        self.all_market_indices = np.stack(
            [ep["market_indices"] for ep in episodes],
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
        stock_batch = self.all_stock_features[
            episode_ids[:, None], stock_row_indices, :
        ]

        market_end = self.all_market_indices[
            episode_ids, step_offsets + lookback - 1
        ]
        market_row_indices = (
            market_end[:, None] - lookback + 1 + window_range[None, :]
        )
        market_batch = self.global_market_features[market_row_indices]

        observation_batch = np.concatenate([stock_batch, market_batch], axis=-1)

        return {
            "states": torch.from_numpy(observation_batch).to(
                device, non_blocking=True,
            ),
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
def load_stock_data(path, market_date_to_idx):
    df = _read_csv(path, dtype={"symbol": str})
    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    feature_cols = [c for c in df.columns if c not in ("symbol", "date", "open", "close")]
    symbols = df["symbol"].values
    opens = df["open"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    features = df[feature_cols].values.astype(np.float32)
    dates = df["date"].values

    symbol_breaks = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1

    # Price limit bands (Chinese A-share rules).
    # STAR (68xxxx) and ChiNext (300xxx post 2020-08-24): ±20%.
    # Mainboard: ±10%. Strict limits applied to overnight gap.
    sym_str = symbols.astype(str)
    starts_with_68 = np.char.startswith(sym_str, "68")
    starts_with_30 = np.char.startswith(sym_str, "30")
    import datetime as _dt
    after_cutoff = dates >= _dt.date(2020, 8, 24)
    wide = starts_with_68 | (starts_with_30 & after_cutoff)

    lo = np.where(wide, 0.80, 0.90)
    hi = np.where(wide, 1.20, 1.10)

    # Open-to-open return decomposition:
    #   ratio[t] = open[t] / open[t-1]
    #            = (close[t-1] / open[t-1]) * (open[t] / close[t-1])
    #            = intraday[t-1]             * overnight[t]
    #
    # Intraday (same-day close/open): unclamped — both prices lie within
    # the same limit band relative to the previous close.
    # Overnight gap (next open / this close): clamped using next day's
    # price limit band, since open[t] is constrained by close[t-1] per
    # day t's rules.
    intraday_prev = closes[:-1] / opens[:-1]       # close[t-1]/open[t-1]
    overnight_gap = opens[1:] / closes[:-1]         # open[t]/close[t-1]
    overnight_gap = np.clip(overnight_gap, lo[1:], hi[1:])  # day t's limits

    ratio = np.ones(len(opens), dtype=np.float64)
    ratio[1:] = intraday_prev * overnight_gap
    ratio[symbol_breaks] = 1.0
    ratio[0] = 1.0
    log_returns = np.log(ratio).astype(np.float32)

    starts = np.empty(len(symbol_breaks) + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = symbol_breaks
    ends = np.empty_like(starts)
    ends[:-1] = symbol_breaks
    ends[-1] = len(df)

    data = {}
    for symbol_idx in range(len(starts)):
        start_row, end_row = int(starts[symbol_idx]), int(ends[symbol_idx])
        symbol_dates = dates[start_row:end_row]
        market_indices = np.array(
            [market_date_to_idx.get(d, -1) for d in symbol_dates], dtype=np.int32,
        )
        data[str(symbols[start_row])] = {
            "features": features[start_row:end_row],
            "log_returns": log_returns[start_row:end_row],
            "dates": symbol_dates.tolist(),
            "opens": opens[start_row:end_row],
            "closes": closes[start_row:end_row],
            "market_indices": market_indices,
        }
    return data, feature_cols


@timed
def load_market_data(path):
    df = _read_csv(path)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    columns = [c for c in df.columns if c != "date"]
    return {
        "features": df[columns].values.astype(np.float32),
        "date_to_idx": dict(zip(df["date"], range(len(df)))),
        "columns": columns,
    }


# Episode generation

def generate_episodes(stock_data, cfg, epoch_rng):
    """Generate non-overlapping episodes with a random per-symbol anchor.

    Stride equals episode_length so trading portions never overlap. A fresh
    random anchor per symbol per call gives different coverage each epoch.
    """
    chunk_size = cfg.lookback + cfg.episode_length
    episodes = []

    for symbol, symbol_data in stock_data.items():
        n_rows = len(symbol_data["features"])
        if n_rows < chunk_size:
            continue

        max_start = n_rows - chunk_size
        anchor = int(epoch_rng.integers(0, min(cfg.episode_length, max_start + 1)))
        precomputed_market = symbol_data["market_indices"]

        start = anchor
        while start <= max_start:
            market_slice = precomputed_market[start : start + chunk_size]
            required_region = market_slice[cfg.lookback - 1 : chunk_size]
            if np.all(required_region >= cfg.lookback - 1):
                episodes.append({
                    "symbol": symbol,
                    "stock_features": symbol_data["features"][start : start + chunk_size],
                    "market_indices": market_slice,
                    "log_returns": symbol_data["log_returns"][start : start + chunk_size],
                    "dates": symbol_data["dates"][start : start + chunk_size],
                    "opens": symbol_data["opens"][start : start + chunk_size],
                    "closes": symbol_data["closes"][start : start + chunk_size],
                })
            start += cfg.episode_length

    epoch_rng.shuffle(episodes)
    return episodes


# Metrics

def compute_episode_returns(reward_matrix):
    """Per-episode percentage return from additive log rewards."""
    return np.expm1(reward_matrix.sum(axis=1))


def compute_max_drawdown(reward_matrix):
    """Mean per-episode max drawdown as percentage decline from peak."""
    cum_returns = np.cumsum(reward_matrix, axis=1)
    running_max = np.maximum.accumulate(cum_returns, axis=1)
    drawdowns = 1.0 - np.exp(cum_returns - running_max)
    return float(drawdowns.max(axis=1).mean())


# Baseline

def compute_baseline_rewards(episodes, cfg):
    """Buy-and-hold baseline: enter long at open of day lookback, hold."""
    log_returns = np.stack([ep["log_returns"] for ep in episodes])
    trading_returns = log_returns[:, cfg.lookback :]
    rewards = np.empty_like(trading_returns)
    rewards[:, 0] = np.log1p(-cfg.transaction_cost)
    rewards[:, 1:] = trading_returns[:, 1:]
    return rewards


# Rollout collection


def _env_n_workers(cfg):
    """Compute per-rank CPU worker count for ParallelVecEnv."""
    n_workers = cfg.env_workers
    if n_workers == 0:
        n_cores = len(os.sched_getaffinity(0))
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", _world))
        n_workers = max(1, n_cores // local_world)
    return n_workers


@timed
def collect_training_rollout(model, episodes, cfg, global_market_features,
                             reward_normalizer, distributed):
    n_episodes = len(episodes)
    episode_length = cfg.episode_length

    rollout = RolloutBuffer(
        n_episodes, episode_length, cfg.lookback, global_market_features,
    )
    rollout.register_episodes(episodes)

    env = ParallelVecEnv(
        episodes, cfg.lookback, cfg.transaction_cost,
        global_market_features, _env_n_workers(cfg),
    )
    obs, pos = env.reset()
    model.train()
    for step in range(episode_length):
        actions, raw_actions, log_probs, values = _batched_stochastic(
            model, obs, pos, cfg,
        )
        rollout.positions[:, step] = pos
        rollout.raw_actions[:, step] = raw_actions
        obs, rewards, _, pos = env.step(actions)
        rollout.log_probs[:, step] = log_probs
        rollout.values[:, step] = values
        rollout.rewards[:, step] = rewards

    terminal_obs = env.terminal_observation()
    _, terminal_values, _ = _batched_greedy(model, terminal_obs, pos, cfg)
    rollout.final_values[:] = terminal_values
    env.close()

    baseline_rewards = compute_baseline_rewards(episodes, cfg)
    model_return = float(compute_episode_returns(rollout.rewards).mean())
    baseline_return = float(compute_episode_returns(baseline_rewards).mean())

    # EMA update and normalize
    if distributed:
        reward_normalizer.update_distributed(rollout.rewards, cfg.device)
    else:
        reward_normalizer.update(rollout.rewards)
    rollout.rewards = reward_normalizer.normalize(rollout.rewards)

    return rollout, model_return, baseline_return


# Deterministic evaluation

def evaluate_deterministic(model, episodes, cfg, global_market_features):
    n_episodes = len(episodes)
    episode_length = cfg.episode_length
    all_rewards = np.empty((n_episodes, episode_length), dtype=np.float32)
    all_positions = np.empty((n_episodes, episode_length), dtype=np.float32)
    all_mus = np.empty((n_episodes, episode_length), dtype=np.float32)

    env = ParallelVecEnv(
        episodes, cfg.lookback, cfg.transaction_cost,
        global_market_features, _env_n_workers(cfg),
    )
    obs, pos = env.reset()
    model.eval()
    for step in range(episode_length):
        actions, _, mus = _batched_greedy(model, obs, pos, cfg)
        obs, rewards, _, pos = env.step(actions)
        all_rewards[:, step] = rewards
        all_positions[:, step] = actions
        all_mus[:, step] = mus
    env.close()

    return all_rewards, all_positions, all_mus


@timed
def evaluate_episodes(model, episodes, cfg, global_market_features):
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
        model, episodes, cfg, global_market_features,
    )
    baseline_rewards = compute_baseline_rewards(episodes, cfg)
    model_returns = compute_episode_returns(model_rewards)
    baseline_returns = compute_episode_returns(baseline_rewards)

    # Position histogram
    pos_flat = model_positions.ravel().astype(np.float64)
    pos_hist, _ = np.histogram(
        pos_flat, bins=_POSITION_BINS, range=(-1.0, 1.0),
    )
    pos_hist = pos_hist.astype(np.float64)

    # Mu histogram
    mu_flat = model_mus.ravel().astype(np.float64)
    mu_hist, _ = np.histogram(
        mu_flat, bins=_MU_BINS, range=_MU_RANGE,
    )
    mu_hist = mu_hist.astype(np.float64)

    n_actions = int(model_positions.size)

    # Per-episode total turnover, averaged across episodes
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
def evaluate_ablated(model, episodes, cfg, global_market_features):
    """Evaluate with market or stock features zeroed to measure feature dependency."""
    if len(episodes) == 0:
        return {"no_market": 0.0, "no_stock": 0.0}

    zero_market = np.zeros_like(global_market_features)
    no_market_rewards, _, _ = evaluate_deterministic(
        model, episodes, cfg, zero_market,
    )

    zero_episodes = [
        {**ep, "stock_features": np.zeros_like(ep["stock_features"])}
        for ep in episodes
    ]
    no_stock_rewards, _, _ = evaluate_deterministic(
        model, zero_episodes, cfg, global_market_features,
    )

    return {
        "no_market": float(compute_episode_returns(no_market_rewards).mean()),
        "no_stock": float(compute_episode_returns(no_stock_rewards).mean()),
    }


# Gradient diagnostics

def diagnose_gradient_contributions(model, batch, cfg):
    """Run separate backward passes to measure each loss term's gradient norm.

    Uses the base model (unwrapped from DDP) in full precision to avoid
    DDP synchronisation and AMP scaling artifacts.
    """
    base_model = model.module if hasattr(model, "module") else model
    base_model.zero_grad()

    with torch.amp.autocast("cuda", enabled=False):
        states = batch["states"].float()
        raw_actions = batch["raw_actions"].float()
        positions = batch["positions"].float()
        log_probs_old = batch["log_probs"].float()
        advantages = batch["advantages"].float()
        returns = batch["returns"].float()

        new_log_probs, _, new_values = evaluate_actions(
            base_model, states, raw_actions, positions,
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

    # Policy gradient norm
    base_model.zero_grad()
    policy_loss.backward(retain_graph=True)
    policy_grads = torch.cat([
        p.grad.data.flatten() if p.grad is not None
        else torch.zeros_like(p.data.flatten())
        for p in all_params
    ])
    norms["policy"] = policy_grads.norm(2).item()

    # Value gradient norm
    base_model.zero_grad()
    value_loss.backward()
    value_grads = torch.cat([
        p.grad.data.flatten() if p.grad is not None
        else torch.zeros_like(p.data.flatten())
        for p in all_params
    ])
    norms["value"] = value_grads.norm(2).item()

    # Cosine similarity between policy and value gradients
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
_MU_RANGE = (-3.0, 3.0)


def run_ppo_update(model, optimizer, scaler,
                   rollout, cfg, n_local_transitions,
                   n_padded_transitions, entropy_coeff):
    """One PPO pass over the rollout buffer.

    All ranks iterate over the same number of mini-batches (determined by
    n_padded_transitions) so DDP gradient synchronisation stays aligned.
    Ranks with fewer local transitions pad with masked transitions that
    contribute zero loss, avoiding gradient bias from duplicated samples.
    """
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

    for batch_start in range(0, n_padded_transitions - cfg.batch_size + 1, cfg.batch_size):
        batch_end = batch_start + cfg.batch_size
        batch_indices = permutation[batch_start:batch_end]
        batch = rollout.get_batch(batch_indices, cfg.device)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            new_log_probs, entropy, new_values = evaluate_actions(
                model, batch["states"], batch["raw_actions"], batch["positions"],
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

            # Combined loss (SB3-style single backward)
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

        # Value prediction and value target histograms
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

        # All-reduce value prediction histogram
        vp_tensor = torch.tensor(value_pred_hist, device=cfg.device, dtype=torch.float64)
        dist.all_reduce(vp_tensor)
        value_pred_hist = vp_tensor.cpu().numpy()

        # All-reduce value target histogram
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
        opens = episode["opens"][cfg.lookback :]
        closes = episode["closes"][cfg.lookback :]
        for step in range(cfg.episode_length):
            mu = float(eval_result["model_mus"][episode_idx, step])
            rows.append({
                "symbol": episode["symbol"],
                "date": dates[step],
                "open": opens[step],
                "close": closes[step],
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

    # All-reduce position histogram
    hist_tensor = torch.tensor(
        result["position_histogram"], device=device, dtype=torch.float64,
    )
    dist.all_reduce(hist_tensor)

    # All-reduce mu histogram
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
        result["no_market"] * local_count,
        result["no_stock"] * local_count,
        float(local_count),
    ], device=device, dtype=torch.float64)
    dist.all_reduce(tensor)
    total = tensor[2].item()
    if total < 1:
        return result
    return {
        "no_market": float(tensor[0].item() / total),
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
    heatmap_data = np.array(hist_list).T  # (n_bins, n_epochs)
    col_sums = heatmap_data.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    heatmap_data = heatmap_data / col_sums
    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        origin="lower",
        extent=[1, len(hist_list), y_range[0], y_range[1]],
        cmap=cmap,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Frequency")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


@timed
def plot_training(history, path, cfg):
    """Plot training curves on a 5x4 grid.

    Row 1 (returns + behavior): episode return, val beat rate, val max drawdown, val turnover
    Row 2 (distributions): val position, val mu, log std, entropy
    Row 3 (losses + PPO stability): policy loss, value loss, clip fraction, approx KL
    Row 4 (value diagnostics + AMP): value target, value pred, loss scale, skip rate
    Row 5 (gradient health): grad norm, per-term grad norm, grad cosine sim, empty
    """
    from matplotlib.ticker import MaxNLocator

    epochs = range(1, len(history["policy_loss"]) + 1)
    fig, axes = plt.subplots(5, 4, figsize=(24, 25))

    # Row 1 — Returns + Behavior
    ablation_epochs = history.get("ablation_epochs", [])
    axes[0, 0].plot(epochs, history["train_return"], color="b", label="Train return")
    axes[0, 0].plot(epochs, history["train_baseline"], color="b", linestyle="--", label="Train baseline")
    axes[0, 0].plot(epochs, history["val_return"], color="r", label="Val return")
    axes[0, 0].plot(epochs, history["val_baseline"], color="r", linestyle="--", label="Val baseline")
    if ablation_epochs:
        axes[0, 0].plot(
            ablation_epochs, history["val_no_market"], color="orange",
            label="Val no-market",
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

    # Row 2 — Distributions
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
        "Epoch", "\u03bc", "Val Mu",
    )

    axes[1, 2].plot(epochs, history["log_std"])
    axes[1, 2].set_ylabel("log \u03c3")
    axes[1, 2].set_title("Policy Log Std")
    axes[1, 2].grid(True, alpha=0.3)

    axes[1, 3].plot(epochs, history["entropy"])
    axes[1, 3].set_ylabel("Entropy")
    axes[1, 3].set_title("Entropy")
    axes[1, 3].grid(True, alpha=0.3)

    # Row 3 — Losses + PPO stability
    axes[2, 0].plot(epochs, history["policy_loss"])
    axes[2, 0].set_ylabel("Loss")
    axes[2, 0].set_title("Policy Loss")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(epochs, history["value_loss"])
    axes[2, 1].set_ylabel("Loss")
    axes[2, 1].set_title("Value Loss")
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(epochs, history["clip_fraction"])
    axes[2, 2].set_ylabel("Clip Fraction")
    axes[2, 2].set_title("Clip Fraction")
    axes[2, 2].grid(True, alpha=0.3)

    axes[2, 3].plot(epochs, history["approx_kl"])
    axes[2, 3].set_ylabel("Approx KL")
    axes[2, 3].set_title("Approx KL")
    axes[2, 3].grid(True, alpha=0.3)

    # Row 4 — Value diagnostics + AMP
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

    axes[3, 2].plot(epochs, history["loss_scale"])
    axes[3, 2].set_xlabel("Epoch")
    axes[3, 2].set_ylabel("Loss Scale")
    axes[3, 2].set_title("Loss Scale")
    axes[3, 2].set_yscale("log")
    axes[3, 2].grid(True, alpha=0.3)

    axes[3, 3].bar(list(epochs), history["skip_rate"], width=0.8)
    axes[3, 3].set_xlabel("Epoch")
    axes[3, 3].set_ylabel("Skip Rate")
    axes[3, 3].set_title("Skip Rate")
    axes[3, 3].grid(True, alpha=0.3)

    # Row 5 — Gradient health
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

    axes[4, 3].set_visible(False)

    for ax in axes.flat:
        if ax.get_visible():
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def warmup_lr(base_lr, epoch, warmup_epochs):
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    return base_lr



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
        market, train_data, val_data, test_data, stock_feature_cols = data
    else:
        _log()
        if not sweep_mode:
            _log("[Data]")
        market = load_market_data(cfg.market_path)
        train_data, stock_feature_cols = load_stock_data(
            cfg.train_path, market["date_to_idx"],
        )
        val_data, _ = load_stock_data(cfg.val_path, market["date_to_idx"])
        test_data, _ = load_stock_data(cfg.test_path, market["date_to_idx"])
        if not sweep_mode:
            _log(f"    {'Train symbols':<20s}: {len(train_data)}")
            _log(f"    {'Val symbols':<20s}: {len(val_data)}")
            _log(f"    {'Test symbols':<20s}: {len(test_data)}")
            _log(f"    {'Train observations':<20s}: {sum(len(d['features']) for d in train_data.values()):,}")
            _log(f"    {'Val observations':<20s}: {sum(len(d['features']) for d in val_data.values()):,}")
            _log(f"    {'Test observations':<20s}: {sum(len(d['features']) for d in test_data.values()):,}")
            _log(f"    {'Stock features':<20s}: {len(stock_feature_cols)}")
            _log(f"    {'Market features':<20s}: {len(market['columns'])}")
    global_market_features = market["features"]
    all_feature_cols = stock_feature_cols + market["columns"]

    stock_groups, market_groups, stock_feature_indices, market_feature_indices = (
        split_feature_groups(all_feature_cols)
    )

    if not sweep_mode:
        _log()
        _log("[Model]")
    model_args = (
        stock_groups, market_groups,
        stock_feature_indices, market_feature_indices, cfg,
    )
    base_model = PolicyNetwork(*model_args).to(cfg.device)

    # SB3-style orthogonal init for MLP heads
    _ortho_init(base_model)

    model = base_model
    if distributed:
        local_rank_id = int(os.environ["LOCAL_RANK"])
        model = DDP(base_model, device_ids=[local_rank_id])
        torch.manual_seed(cfg.seed + _rank)

    n_params = sum(p.numel() for p in base_model.parameters())
    if not sweep_mode:
        _log(f"    {'Stock groups':<20s}: {len(stock_groups)}")
        _log(f"    {'Market groups':<20s}: {len(market_groups)}")
        _log(f"    {'Parameters':<20s}: {n_params:,}")

    effective_lr = cfg.lr * math.sqrt(_world)

    decay_params, xattn_decay_params, no_decay_params = [], [], []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        is_norm_or_bias = param.dim() < 2 or "norm" in name
        if is_norm_or_bias:
            no_decay_params.append(param)
        elif "xattn." in name:
            xattn_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": xattn_decay_params, "weight_decay": cfg.xattn_weight_decay},
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
        # Support loading from either old split-model or new shared checkpoints
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

        # Already completed sweep trial — skip re-training
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
            "feature_cols": all_feature_cols,
            "stock_groups": stock_groups,
            "market_groups": market_groups,
            "stock_indices": stock_feature_indices,
            "market_indices": market_feature_indices,
            "epoch": epoch,
            "best_score": best_score,
            "patience_counter": patience_counter,
            "reward_normalizer": reward_normalizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": dict(history),
        }

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        current_lr = warmup_lr(
            effective_lr, epoch, cfg.warmup_epochs,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        _log()
        _log(f"[Epoch {epoch}/{cfg.n_epochs}]")

        epoch_rng = np.random.default_rng(cfg.seed + epoch * 7919)
        train_episodes = generate_episodes(train_data, cfg, epoch_rng)
        rank_train_episodes = train_episodes[_rank::_world]

        val_rng = np.random.default_rng(cfg.seed + epoch * 7919 + 1)
        val_episodes = generate_episodes(val_data, cfg, val_rng)
        rank_val_episodes = val_episodes[_rank::_world]

        rollout, model_return, baseline_return = collect_training_rollout(
            model, rank_train_episodes, cfg, global_market_features,
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
            n_padded_transitions = _allreduce_max_int(
                n_local_transitions, cfg.device,
            )
        else:
            n_padded_transitions = n_local_transitions

        # Gradient diagnostics (before PPO updates, on a fresh batch)
        grad_contrib = None
        if cfg.diagnostic:
            _diag_t = time.perf_counter()
            diag_n = min(cfg.batch_size, n_local_transitions)
            diag_indices = np.random.choice(
                n_local_transitions, diag_n, replace=False,
            )
            diag_batch = rollout.get_batch(diag_indices, cfg.device)
            grad_contrib = diagnose_gradient_contributions(
                model, diag_batch, cfg,
            )
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
            losses = run_ppo_update(
                model, optimizer, scaler,
                rollout, cfg,
                n_local_transitions, n_padded_transitions,
                cfg.entropy_coeff,
            )
            for k in epoch_losses:
                epoch_losses[k] += losses[k]
            # Accumulate across PPO epochs for consistency with scalar value_loss
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
        bm = model.module if hasattr(model, "module") else model
        current_log_std = bm.log_std.item()

        del rollout
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

        val_result = evaluate_episodes(
            model, rank_val_episodes, cfg, global_market_features,
        )
        if distributed:
            val_aggregated = _allreduce_eval(
                val_result, len(rank_val_episodes), cfg.device,
            )
        else:
            val_aggregated = val_result

        if cfg.ablation:
            ablation_result = evaluate_ablated(
                model, rank_val_episodes, cfg, global_market_features,
            )
            if distributed:
                ablation_aggregated = _allreduce_ablation(
                    ablation_result, len(rank_val_episodes), cfg.device,
                )
            else:
                ablation_aggregated = ablation_result
            history["ablation_epochs"].append(epoch)
            history["val_no_market"].append(ablation_aggregated["no_market"])
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
        ]:
            history[key].append(value)

        if grad_contrib is not None:
            history["grad_norm_policy"].append(grad_contrib["policy"])
            history["grad_norm_value"].append(grad_contrib["value"])
            history["grad_cosine_sim"].append(grad_contrib["cosine_sim"])

        # Histograms
        pos_hist = (
            val_aggregated["position_histogram"]
            if "position_histogram" in val_aggregated
            else np.zeros(_POSITION_BINS, dtype=np.float64)
        )
        history["position_histogram"].append(pos_hist)
        history["value_pred_histogram"].append(epoch_value_pred_hist)
        history["value_target_histogram"].append(epoch_value_target_hist)

        # Mu histogram
        mu_hist = (
            val_aggregated["mu_histogram"]
            if "mu_histogram" in val_aggregated
            else np.zeros(_MU_BINS, dtype=np.float64)
        )
        history["mu_histogram"].append(mu_hist)

        # Exact summary stats from running sums
        n_actions = val_aggregated.get("n_actions", 0)
        if n_actions > 0:
            pos_mean = val_aggregated["pos_sum"] / n_actions
            pos_std = math.sqrt(val_aggregated["pos_sum_sq"] / n_actions - pos_mean ** 2)
            mu_mean = val_aggregated["mu_sum"] / n_actions
            mu_std = math.sqrt(val_aggregated["mu_sum_sq"] / n_actions - mu_mean ** 2)
        else:
            pos_mean, pos_std = 0.0, 0.0
            mu_mean, mu_std = 0.0, 0.0
        if epoch_v_count > 0:
            vp_mean = epoch_vp_sum / epoch_v_count
            vp_std = math.sqrt(epoch_vp_sum_sq / epoch_v_count - vp_mean ** 2)
            vt_mean = epoch_vt_sum / epoch_v_count
            vt_std = math.sqrt(epoch_vt_sum_sq / epoch_v_count - vt_mean ** 2)
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

        # Metrics (ordered to match plot rows)
        # Row 1 — Returns + behavior
        _log(f"    {'Train episodes':<20s}: {len(train_episodes)}")
        _log(f"    {'Val episodes':<20s}: {len(val_episodes)}")
        _log(f"    {'Train return':<20s}: {model_return:.4f}")
        _log(f"    {'Train baseline':<20s}: {baseline_return:.4f}")
        _log(f"    {'Val return':<20s}: {val_aggregated['model_return']:.4f}")
        _log(f"    {'Val baseline':<20s}: {val_aggregated['baseline_return']:.4f}")
        if cfg.ablation:
            _log(f"    {'Val no-market':<20s}: {ablation_aggregated['no_market']:.4f}")
            _log(f"    {'Val no-stock':<20s}: {ablation_aggregated['no_stock']:.4f}")
        _log(f"    {'Val beat rate':<20s}: {val_aggregated['beat_rate']:.4f}")
        _log(f"    {'Val max drawdown':<20s}: {val_aggregated['max_drawdown']:.4f}")
        _log(f"    {'Val turnover':<20s}: {val_aggregated['mean_turnover']:.4f}")
        # Row 2 — Distributions
        _log(f"    {'Val position mean':<20s}: {pos_mean:.4f}")
        _log(f"    {'Val position std':<20s}: {pos_std:.4f}")
        _log(f"    {'Val mu mean':<20s}: {mu_mean:.4f}")
        _log(f"    {'Val mu std':<20s}: {mu_std:.4f}")
        _log(f"    {'Log std':<20s}: {current_log_std:.4f}")
        _log(f"    {'Entropy':<20s}: {epoch_losses['entropy']:.4f}")
        # Row 3 — Losses + PPO stability
        _log(f"    {'Policy loss':<20s}: {epoch_losses['policy_loss']:.4f}")
        _log(f"    {'Value loss':<20s}: {epoch_losses['value_loss']:.4f}")
        _log(f"    {'Clip fraction':<20s}: {epoch_losses['clip_fraction']:.4f}")
        _log(f"    {'Approx KL':<20s}: {epoch_losses['approx_kl']:.4f}")
        # Row 4 — Value diagnostics + AMP
        _log(f"    {'Value target mean':<20s}: {vt_mean:.4f}")
        _log(f"    {'Value target std':<20s}: {vt_std:.4f}")
        _log(f"    {'Value pred mean':<20s}: {vp_mean:.4f}")
        _log(f"    {'Value pred std':<20s}: {vp_std:.4f}")
        _log(f"    {'Loss scale':<20s}: {loss_scale:.1f}")
        _log(f"    {'Skip rate':<20s}: {skip_rate:.4f} ({epoch_skipped}/{epoch_total})")
        # Row 5 — Gradient health
        _log(f"    {'Grad norm':<20s}: {epoch_losses['grad_norm']:.4f}")
        if grad_contrib is not None:
            _log(f"    {'Policy grad norm':<20s}: {grad_contrib['policy']:.4f}")
            _log(f"    {'Value grad norm':<20s}: {grad_contrib['value']:.4f}")
            _log(f"    {'Grad cosine sim':<20s}: {grad_contrib['cosine_sim']:.4f}")
        # Score + patience
        _log(f"    {'Current score':<20s}: {current_score:.4f}")
        _log(f"    {'Best score':<20s}: {best_score:.4f}")
        _log(f"    {'Patience':<20s}: {patience_counter}/{cfg.patience}")

        if patience_counter >= cfg.patience:
            _log(f"    Early stopping at epoch {epoch} (patience {cfg.patience})")
            break

        # Optuna pruning
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
    test_episodes = generate_episodes(test_data, cfg, test_episode_rng)
    rank_test_episodes = test_episodes[_rank::_world]

    _log()
    _log("[Final Evaluation]")
    test_result = evaluate_episodes(
        model, rank_test_episodes, cfg, global_market_features,
    )
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
    """Run Optuna hyperparameter sweep with TPE sampler and median pruning.

    Each trial overrides Config fields and calls train(). The objective is
    the best smoothed validation return. Trials are persisted to SQLite so
    multiple processes can run in parallel against the same database.

    Automatically uses DDP when multiple GPUs are available.  Rank 0 drives
    Optuna and broadcasts trial parameters to all ranks.

    Monitor with: optuna-dashboard sqlite:///ppo_sweep.db
    """
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

    # Load data once for all trials (each rank loads independently)
    if is_main:
        _log()
        _log("[Data]")
    market = load_market_data(cfg.market_path)
    train_data, stock_feature_cols = load_stock_data(
        cfg.train_path, market["date_to_idx"],
    )
    val_data, _ = load_stock_data(cfg.val_path, market["date_to_idx"])
    test_data, _ = load_stock_data(cfg.test_path, market["date_to_idx"])
    if is_main:
        _log(f"    {'Train symbols':<20s}: {len(train_data)}")
        _log(f"    {'Val symbols':<20s}: {len(val_data)}")
        _log(f"    {'Test symbols':<20s}: {len(test_data)}")
        _log(f"    {'Train observations':<20s}: {sum(len(d['features']) for d in train_data.values()):,}")
        _log(f"    {'Val observations':<20s}: {sum(len(d['features']) for d in val_data.values()):,}")
        _log(f"    {'Test observations':<20s}: {sum(len(d['features']) for d in test_data.values()):,}")
        _log(f"    {'Stock features':<20s}: {len(stock_feature_cols)}")
        _log(f"    {'Market features':<20s}: {len(market['columns'])}")
    data = (market, train_data, val_data, test_data, stock_feature_cols)

    if is_main:
        all_feature_cols = stock_feature_cols + market["columns"]
        stock_groups, market_groups, stock_feature_indices, market_feature_indices = (
            split_feature_groups(all_feature_cols)
        )
        _log()
        _log("[Model]")
        _log(f"    {'Stock groups':<20s}: {len(stock_groups)}")
        _log(f"    {'Market groups':<20s}: {len(market_groups)}")
        tmp_model = PolicyNetwork(
            stock_groups, market_groups,
            stock_feature_indices, market_feature_indices, cfg,
        )
        n_params = sum(p.numel() for p in tmp_model.parameters())
        _log(f"    {'Parameters':<20s}: {n_params:,}")
        del tmp_model

    _SWEEP_KEYS = [
        "dropout", "market_dropout", "value_coeff", "entropy_coeff",
        "lr", "weight_decay", "xattn_weight_decay",
    ]

    def _param_hash(params):
        return hashlib.md5(
            json.dumps({k: params[k] for k in sorted(_SWEEP_KEYS)}).encode(),
        ).hexdigest()[:12]

    # Only rank 0 manages Optuna
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

        # Re-enqueue crashed trials that left checkpoints on disk
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

    # Manual trial loop so we can broadcast params across ranks
    while True:
        # Rank 0 picks params and checks stopping condition
        params = None
        trial_number = -1
        if is_main:
            n_complete = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            if n_complete >= cfg.sweep_trials:
                params = None  # signal to stop
            else:
                trial = study.ask()
                trial.suggest_float("dropout", 0.0, 0.2)
                trial.suggest_float("market_dropout", 0.2, 0.6)
                trial.suggest_float("value_coeff", 0.01, 1.0, log=True)
                trial.suggest_float("entropy_coeff", 0.0001, 0.01, log=True)
                trial.suggest_float("lr", 1e-5, 1e-3, log=True)
                params = trial.params
                trial_number = trial.number

        # Broadcast params (or None to signal stop)
        if distributed:
            broadcast_list = [params]
            dist.broadcast_object_list(broadcast_list, src=0)
            params = broadcast_list[0]

        if params is None:
            break

        # All ranks apply params
        for k, v in params.items():
            setattr(cfg, k, v)
        param_hash = _param_hash(params)
        cfg.save_dir = os.path.join("sweep_runs", param_hash)
        cfg.ablation = False
        cfg.diagnostic = False

        if is_main:
            _log()
            _log(f"[Optuna Trial {trial_number}]")
            _log(f"    {'hash':<20s}: {param_hash}")
            for key in _SWEEP_KEYS:
                _log(f"    {key:<20s}: {getattr(cfg, key)}")

        # All ranks train together (DDP sync happens inside train)
        score = train(cfg, data=data, sweep_mode=True)

        # Rank 0 reports result
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
