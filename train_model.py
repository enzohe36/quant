"""
PPO + Transformer RL for stock trading.

Stock CSV: symbol, date, price, <stock_feat_1>, ...
Market CSV: date, <mkt_feat_1>, ...  (columns except date start with "mkt_")

Market features loaded separately, aligned to stock data at each lookback
window's last date. Features with "mkt_" prefix route through cross-attention.

Launch:
  python train_model.py
  Auto-detects GPUs. Uses DDP when multiple GPUs are available.
  Also works with: torchrun --nproc_per_node=N train_model.py
  Configure all parameters in the Config class.
"""

import argparse
import os
import math
import time
import socket
import functools
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
    # Paths
    train_path: str = "train.csv"
    test_path: str = "test.csv"
    market_path: str = "mkt_feats.csv"
    save_dir: str = "checkpoints"

    # Environment
    lookback: int = 60
    episode_length: int = 100
    transaction_cost: float = 0.001
    n_actions: int = 3

    # Model
    d_model: int = 256
    d_ff: int = 512
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.0
    market_dropout: float = 0.0

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff_start: float = 0.05
    entropy_coeff_end: float = 0.005
    grad_clip: float = 0.5

    # Regularization
    weight_decay: float = 0.02
    xattn_weight_decay: float = 0.05

    # Training
    n_epochs: int = 200
    warmup_epochs: int = 5
    ppo_epochs: int = 4
    batch_size: int = 4096
    inference_batch: int = 65536
    lr: float = 1e-4
    patience: int = 40
    patience_smoothing: int = 10
    ablation_interval: int = 1
    seed: int = 42

    # Numerical
    reward_clip: float = 10.0
    reward_ema_decay: float = 0.99
    sortino_clip: float = 10.0
    pytorch_eps: float = 1e-8
    use_amp: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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
    """Batched single-stock trading environment.

    Observations concatenate [stock_window, market_window] along the feature
    axis. For each lookback window the stock slice is ``lookback`` consecutive
    rows from the episode's stock_features. The market slice is ``lookback``
    consecutive rows from the global market array, ending at the market row
    whose date matches the stock window's last date.
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
        position_idx = (self.position + 1).astype(np.int64)
        return self._build_observation(), position_idx

    def step(self, actions):
        new_position = actions.astype(np.float64) - 1.0
        reward = (
            self.position * self.log_returns[:, self.current_step]
            - self.transaction_cost * np.abs(new_position - self.position)
        ).astype(np.float32)
        self.position = new_position
        self.current_step += 1
        done = self.current_step >= self.total_steps
        position_idx = (self.position + 1).astype(np.int64)
        obs = None if done else self._build_observation()
        return obs, reward, done, position_idx

    def terminal_observation(self):
        """Build observation at the truncation boundary (current_step == total_steps)."""
        return self._build_observation()


# Reward normalizer

class RewardNormalizer:
    """EMA reward normalizer for non-stationary reward scaling.

    Tracks an exponential moving average of mean squared reward. Divides
    rewards by the EMA standard deviation without mean-shifting, so the
    optimal policy is unchanged. Adapts to distribution shift as the
    policy improves, unlike a monotonically accumulating estimator.
    """

    def __init__(self, clip, eps, decay):
        self.clip = clip
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
        return np.clip(
            rewards / std, -self.clip, self.clip,
        ).astype(np.float32)

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


class PolicyNetwork(nn.Module):
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

        position_dim = 16
        self.position_emb = nn.Embedding(cfg.n_actions, position_dim)
        head_input_dim = cfg.d_model + position_dim

        self.policy_head = nn.Sequential(
            nn.Linear(head_input_dim, 128), nn.ReLU(), nn.Linear(128, cfg.n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(head_input_dim, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, x, position_idx):
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
        pos_emb = self.position_emb(position_idx)
        pooled = torch.cat([pooled, pos_emb], dim=-1)
        return self.policy_head(pooled), self.value_head(pooled).squeeze(-1)


def sample_stochastic(model, states, position_idx):
    logits, values = model(states, position_idx)
    policy = torch.distributions.Categorical(logits=logits)
    actions = policy.sample()
    return actions, policy.log_prob(actions), values


def select_greedy(model, states, position_idx):
    logits, values = model(states, position_idx)
    return logits.argmax(-1), values


def evaluate_actions(model, states, actions, position_idx):
    logits, values = model(states, position_idx)
    policy = torch.distributions.Categorical(logits=logits)
    return policy.log_prob(actions), policy.entropy(), values


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

        self.actions = np.empty((n_episodes, episode_length), dtype=np.int64)
        self.rewards = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.log_probs = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.values = np.empty((n_episodes, episode_length), dtype=np.float32)
        self.positions = np.empty((n_episodes, episode_length), dtype=np.int64)
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
        self._flat_actions = self.actions.ravel()
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
            "actions": torch.tensor(
                self._flat_actions[cpu_indices], dtype=torch.long, device=device,
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
                self._flat_positions[cpu_indices], dtype=torch.long, device=device,
            ),
        }


# Data loading

def _read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, engine="pyarrow", **kwargs)
    except ImportError:
        return pd.read_csv(path, **kwargs)


@timed
def load_stock_data(path, market_date_to_idx):
    df = _read_csv(path, dtype={"symbol": str})
    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    feature_cols = [c for c in df.columns if c not in ("symbol", "date", "price")]
    symbols = df["symbol"].values
    prices = df["price"].values.astype(np.float64)
    features = df[feature_cols].values.astype(np.float32)
    dates = df["date"].values

    log_returns = np.empty(len(prices), dtype=np.float32)
    log_returns[0] = 0.0
    log_returns[1:] = np.log(prices[1:] / prices[:-1]).astype(np.float32)
    symbol_breaks = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1
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
        symbol_dates = dates[start_row:end_row]
        market_indices = np.array(
            [market_date_to_idx.get(d, -1) for d in symbol_dates], dtype=np.int32,
        )
        data[str(symbols[start_row])] = {
            "features": features[start_row:end_row],
            "log_returns": log_returns[start_row:end_row],
            "dates": symbol_dates.tolist(),
            "prices": prices[start_row:end_row],
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
                    "prices": symbol_data["prices"][start : start + chunk_size],
                })
            start += cfg.episode_length

    epoch_rng.shuffle(episodes)
    return episodes


# Metrics

def compute_sortino(rewards, clip, eps):
    mean_reward = rewards.mean()
    downside_deviation = np.sqrt((np.minimum(rewards, 0.0) ** 2).mean())
    if downside_deviation < eps:
        return float(np.clip(np.sign(mean_reward) * clip, -clip, clip))
    return float(np.clip(mean_reward / downside_deviation, -clip, clip))


def compute_episode_sortinos(reward_matrix, clip, eps):
    return np.array(
        [compute_sortino(reward_matrix[i], clip, eps)
         for i in range(len(reward_matrix))],
    )


# Baseline

def compute_baseline_rewards(episodes, cfg):
    """Buy-and-hold baseline: enter long at step 0, hold for remaining steps."""
    log_returns = np.stack([ep["log_returns"] for ep in episodes])
    trading_returns = log_returns[:, cfg.lookback :]
    rewards = np.empty_like(trading_returns)
    rewards[:, 0] = -cfg.transaction_cost
    rewards[:, 1:] = trading_returns[:, 1:]
    return rewards


# Rollout collection

@timed
def collect_training_rollout(model, episodes, cfg, global_market_features,
                             reward_normalizer, distributed):
    n_episodes = len(episodes)
    episode_length = cfg.episode_length

    rollout = RolloutBuffer(
        n_episodes, episode_length, cfg.lookback, global_market_features,
    )
    rollout.register_episodes(episodes)

    model.train()
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
        for batch_start in range(0, n_episodes, cfg.inference_batch):
            batch_end = min(batch_start + cfg.inference_batch, n_episodes)
            batch_episodes = episodes[batch_start:batch_end]
            env = VecEnv(
                batch_episodes, cfg.lookback, cfg.transaction_cost,
                global_market_features,
            )
            obs, pos_idx = env.reset()

            for step in range(episode_length):
                states = torch.from_numpy(obs).to(cfg.device)
                pos_tensor = torch.from_numpy(pos_idx).to(cfg.device)
                actions, log_probs, values = sample_stochastic(
                    model, states, pos_tensor,
                )
                actions_np = actions.cpu().numpy()

                rollout.positions[batch_start:batch_end, step] = pos_idx
                rollout.actions[batch_start:batch_end, step] = actions_np

                obs, step_rewards, _, pos_idx = env.step(actions_np)
                rollout.log_probs[batch_start:batch_end, step] = (
                    log_probs.cpu().numpy()
                )
                rollout.values[batch_start:batch_end, step] = (
                    values.cpu().numpy()
                )
                rollout.rewards[batch_start:batch_end, step] = step_rewards

            # Bootstrap V(s_T) at the truncation boundary
            terminal_obs = env.terminal_observation()
            terminal_states = torch.from_numpy(terminal_obs).to(cfg.device)
            terminal_pos = torch.from_numpy(pos_idx).to(cfg.device)
            _, terminal_values = select_greedy(model, terminal_states, terminal_pos)
            rollout.final_values[batch_start:batch_end] = (
                terminal_values.cpu().numpy()
            )

    baseline_rewards = compute_baseline_rewards(episodes, cfg)
    model_sortino = float(compute_episode_sortinos(
        rollout.rewards, cfg.sortino_clip, cfg.pytorch_eps,
    ).mean())
    baseline_sortino = float(compute_episode_sortinos(
        baseline_rewards, cfg.sortino_clip, cfg.pytorch_eps,
    ).mean())

    if distributed:
        reward_normalizer.update_distributed(rollout.rewards, cfg.device)
    else:
        reward_normalizer.update(rollout.rewards)
    rollout.rewards = reward_normalizer.normalize(rollout.rewards)

    return rollout, model_sortino, baseline_sortino


# Deterministic evaluation

def evaluate_deterministic(model, episodes, cfg, global_market_features):
    n_episodes = len(episodes)
    episode_length = cfg.episode_length
    all_rewards = np.empty((n_episodes, episode_length), dtype=np.float32)
    all_positions = np.empty((n_episodes, episode_length), dtype=np.int64)

    model.eval()
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
        for batch_start in range(0, n_episodes, cfg.inference_batch):
            batch_end = min(batch_start + cfg.inference_batch, n_episodes)
            batch_episodes = episodes[batch_start:batch_end]
            env = VecEnv(
                batch_episodes, cfg.lookback, cfg.transaction_cost,
                global_market_features,
            )
            obs, pos_idx = env.reset()

            for step in range(episode_length):
                states = torch.from_numpy(obs).to(cfg.device)
                pos_tensor = torch.from_numpy(pos_idx).to(cfg.device)
                actions, _ = select_greedy(model, states, pos_tensor)
                actions_np = actions.cpu().numpy()
                obs, step_rewards, _, pos_idx = env.step(actions_np)
                all_rewards[batch_start:batch_end, step] = step_rewards
                all_positions[batch_start:batch_end, step] = actions_np - 1

    return all_rewards, all_positions


@timed
def evaluate_episodes(model, episodes, cfg, global_market_features):
    if len(episodes) == 0:
        return {
            "model_sortino": 0.0, "baseline_sortino": 0.0, "beat_rate": 0.0,
            "model_rewards": np.empty((0, cfg.episode_length)),
            "baseline_rewards": np.empty((0, cfg.episode_length)),
            "model_positions": np.empty((0, cfg.episode_length), dtype=np.int64),
        }

    model_rewards, model_positions = evaluate_deterministic(
        model, episodes, cfg, global_market_features,
    )
    baseline_rewards = compute_baseline_rewards(episodes, cfg)
    model_sortinos = compute_episode_sortinos(
        model_rewards, cfg.sortino_clip, cfg.pytorch_eps,
    )
    baseline_sortinos = compute_episode_sortinos(
        baseline_rewards, cfg.sortino_clip, cfg.pytorch_eps,
    )

    return {
        "model_sortino": float(model_sortinos.mean()),
        "baseline_sortino": float(baseline_sortinos.mean()),
        "beat_rate": float(np.mean(model_sortinos > baseline_sortinos)),
        "model_rewards": model_rewards,
        "baseline_rewards": baseline_rewards,
        "model_positions": model_positions,
    }


@timed
def evaluate_ablated(model, episodes, cfg, global_market_features):
    """Evaluate with market or stock features zeroed to measure feature dependency."""
    if len(episodes) == 0:
        return {"no_market": 0.0, "no_stock": 0.0}

    zero_market = np.zeros_like(global_market_features)
    no_market_rewards, _ = evaluate_deterministic(
        model, episodes, cfg, zero_market,
    )

    zero_episodes = [
        {**ep, "stock_features": np.zeros_like(ep["stock_features"])}
        for ep in episodes
    ]
    no_stock_rewards, _ = evaluate_deterministic(
        model, zero_episodes, cfg, global_market_features,
    )

    return {
        "no_market": float(compute_episode_sortinos(
            no_market_rewards, cfg.sortino_clip, cfg.pytorch_eps,
        ).mean()),
        "no_stock": float(compute_episode_sortinos(
            no_stock_rewards, cfg.sortino_clip, cfg.pytorch_eps,
        ).mean()),
    }


# PPO update

def run_ppo_update(model, optimizer, scaler, rollout, cfg, n_local_transitions,
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
                model, batch["states"], batch["actions"], batch["positions"],
            )
            ratio = torch.exp(new_log_probs - batch["log_probs"])
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
            else:
                policy_loss = -surrogate.mean()
                value_loss = F.mse_loss(new_values, batch["returns"])
                entropy_term = entropy.mean()

            total_loss = (
                policy_loss
                + cfg.value_coeff * value_loss
                - entropy_coeff * entropy_term
            )

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(base_model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        entropy_sum += entropy_term.item()
        n_batches += 1

    if _world > 1:
        sync_tensor = torch.tensor(
            [policy_loss_sum, value_loss_sum, entropy_sum, float(n_batches)],
            device=cfg.device,
        )
        dist.all_reduce(sync_tensor)
        policy_loss_sum = sync_tensor[0].item()
        value_loss_sum = sync_tensor[1].item()
        entropy_sum = sync_tensor[2].item()
        n_batches = int(sync_tensor[3].item())

    n_batches = max(n_batches, 1)
    return {
        "policy_loss": policy_loss_sum / n_batches,
        "value_loss": value_loss_sum / n_batches,
        "entropy": entropy_sum / n_batches,
    }


# Test results

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
                "position": int(eval_result["model_positions"][episode_idx, step]),
                "model_reward": float(
                    eval_result["model_rewards"][episode_idx, step],
                ),
                "baseline_reward": float(
                    eval_result["baseline_rewards"][episode_idx, step],
                ),
            })
    return rows


# DDP helpers

def _allreduce_means(mean_a, mean_b, local_count, device):
    tensor = torch.tensor(
        [mean_a * local_count, mean_b * local_count, float(local_count)],
        device=device, dtype=torch.float64,
    )
    dist.all_reduce(tensor)
    total = tensor[2].item()
    if total < 1:
        return mean_a, mean_b
    return float(tensor[0].item() / total), float(tensor[1].item() / total)


def _allreduce_eval(result, local_count, device):
    tensor = torch.tensor([
        result["model_sortino"] * local_count,
        result["baseline_sortino"] * local_count,
        result["beat_rate"] * local_count,
        float(local_count),
    ], device=device, dtype=torch.float64)
    dist.all_reduce(tensor)
    total = tensor[3].item()
    if total < 1:
        return result
    return {
        "model_sortino": float(tensor[0].item() / total),
        "baseline_sortino": float(tensor[1].item() / total),
        "beat_rate": float(tensor[2].item() / total),
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
    local = torch.zeros(n_local, episode_length, 3, device=device)
    local[..., 0] = torch.from_numpy(test_result["model_rewards"]).to(device)
    local[..., 1] = torch.from_numpy(test_result["baseline_rewards"]).to(device)
    local[..., 2] = torch.from_numpy(
        test_result["model_positions"].astype(np.float32),
    ).to(device)

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
        "model_positions": result_np[..., 2].astype(np.int64),
    }


# Plotting and checkpointing

@timed
def plot_training(history, path):
    """Plot training curves on a 2x3 grid.

    Top row: policy loss, value loss, entropy.
    Bottom row: train/val sortino, ablation sortino, val beat rate.
    """
    from matplotlib.ticker import MaxNLocator

    C_TRAIN = "b"
    C_VAL = "r"
    C_NO_MARKET = "darkorange"
    C_NO_STOCK = "purple"

    epochs = range(1, len(history["train_model"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(epochs, history["policy_loss"], "b-", alpha=0.8)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Policy Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["value_loss"], "b-", alpha=0.8)
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history["entropy"], "b-", alpha=0.8)
    axes[0, 2].set_ylabel("Entropy")
    axes[0, 2].set_title("Entropy")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["train_model"], color=C_TRAIN, label="Train model", alpha=0.8)
    axes[1, 0].plot(epochs, history["val_model"], color=C_VAL, label="Val model", alpha=0.8)
    axes[1, 0].plot(epochs, history["train_baseline"], color=C_TRAIN, linestyle="--", label="Train baseline", alpha=0.5)
    axes[1, 0].plot(epochs, history["val_baseline"], color=C_VAL, linestyle="--", label="Val baseline", alpha=0.5)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Sortino")
    axes[1, 0].set_title("Train vs Val")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    ablation_epochs = history.get("ablation_epochs", [])
    axes[1, 1].plot(epochs, history["val_model"], color=C_VAL, label="Val model", alpha=0.8)
    axes[1, 1].plot(epochs, history["val_baseline"], color=C_VAL, linestyle="--", label="Val baseline", alpha=0.5)
    if ablation_epochs:
        axes[1, 1].plot(
            ablation_epochs, history["val_no_market"], color=C_NO_MARKET,
            label="Val no-market", alpha=0.8,
        )
        axes[1, 1].plot(
            ablation_epochs, history["val_no_stock"], color=C_NO_STOCK,
            label="Val no-stock", alpha=0.8,
        )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Sortino")
    axes[1, 1].set_title("Feature Ablation")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, history["beat_rate"], color=C_VAL, alpha=0.8)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Beat Rate")
    axes[1, 2].set_title("Val Beat Rate")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@timed
def save_checkpoint(state, path):
    torch.save(state, path)


def warmup_cosine_lr(base_lr, epoch, n_epochs, warmup_epochs):
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def cosine_entropy_coeff(start, end, epoch, n_epochs):
    progress = min(epoch / n_epochs, 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


# Training loop

def train(cfg):
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

    _log()
    _log("[Data]")
    market = load_market_data(cfg.market_path)
    global_market_features = market["features"]
    train_data, stock_feature_cols = load_stock_data(
        cfg.train_path, market["date_to_idx"],
    )
    test_data, _ = load_stock_data(cfg.test_path, market["date_to_idx"])
    all_feature_cols = stock_feature_cols + market["columns"]
    _log(f"    {'Train symbols':<20s}: {len(train_data)}")
    _log(f"    {'Test symbols':<20s}: {len(test_data)}")
    _log(f"    {'Stock features':<20s}: {len(stock_feature_cols)}")
    _log(f"    {'Market features':<20s}: {len(market['columns'])}")

    _log()
    _log("[Episodes]")
    test_episode_rng = np.random.default_rng(cfg.seed + 1000)
    all_test_episodes = generate_episodes(test_data, cfg, test_episode_rng)

    test_split_rng = np.random.default_rng(cfg.seed + 2000)
    test_symbols = sorted(set(ep["symbol"] for ep in all_test_episodes))
    test_split_rng.shuffle(test_symbols)
    n_val_symbols = len(test_symbols) // 2
    val_symbol_set = set(test_symbols[:n_val_symbols])
    val_episodes = [ep for ep in all_test_episodes if ep["symbol"] in val_symbol_set]
    test_episodes = [ep for ep in all_test_episodes if ep["symbol"] not in val_symbol_set]
    del all_test_episodes
    _log(f"    {'Val symbols':<20s}: {n_val_symbols}")
    _log(f"    {'Test symbols':<20s}: {len(test_symbols) - n_val_symbols}")
    _log(f"    {'Val episodes':<20s}: {len(val_episodes)}")
    _log(f"    {'Test episodes':<20s}: {len(test_episodes)}")

    stock_groups, market_groups, stock_feature_indices, market_feature_indices = (
        split_feature_groups(all_feature_cols)
    )

    _log()
    _log("[Model]")
    base_model = PolicyNetwork(
        stock_groups, market_groups,
        stock_feature_indices, market_feature_indices, cfg,
    ).to(cfg.device)
    model = base_model
    if distributed:
        model = DDP(base_model, device_ids=[int(os.environ["LOCAL_RANK"])])
        torch.manual_seed(cfg.seed + _rank)
    n_params = sum(p.numel() for p in base_model.parameters())
    _log(f"    {'Stock groups':<20s}: {len(stock_groups)} ({len(stock_feature_indices)} features)")
    _log(f"    {'Market groups':<20s}: {len(market_groups)} ({len(market_feature_indices)} features)")
    _log(f"    {'Parameters':<20s}: {n_params:,}")
    device_label = f"{_world} GPUs (DDP)" if distributed else cfg.device
    _log(f"    {'Device':<20s}: {device_label}")

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
        lr=effective_lr, eps=1e-5
    )
    reward_normalizer = RewardNormalizer(
        cfg.reward_clip, cfg.pytorch_eps, cfg.reward_ema_decay,
    )
    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    _log(f"    {'Decay params':<20s}: {sum(p.numel() for p in decay_params):,} (wd={cfg.weight_decay})")
    _log(f"    {'Xattn params':<20s}: {sum(p.numel() for p in xattn_decay_params):,} (wd={cfg.xattn_weight_decay})")
    _log(f"    {'No decay params':<20s}: {sum(p.numel() for p in no_decay_params):,} (wd=0)")
    _log(f"    {'Mixed precision':<20s}: {'float16' if amp_enabled else 'disabled'}")
    if _world > 1:
        _log(f"    {'LR (sqrt scaled)':<20s}: {cfg.lr} x sqrt({_world}) = {effective_lr:.6f}")

    history = defaultdict(list)
    best_smoothed_sortino = -float("inf")
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

        val_series = history.get("val_model", [])
        if val_series:
            window = cfg.patience_smoothing
            best_smoothed_sortino = max(
                float(np.mean(val_series[max(0, i + 1 - window) : i + 1]))
                for i in range(len(val_series))
            )
        else:
            best_smoothed_sortino = ckpt.get("best_smoothed_sortino", -float("inf"))

        _log()
        _log(
            f"Resumed from epoch {ckpt['epoch']}"
            f" (best smoothed {best_smoothed_sortino:.4f},"
            f" patience {patience_counter})"
        )

    rank_val_episodes = val_episodes[_rank::_world]

    _log()
    _log(
        f"[Training] {cfg.n_epochs} epochs, patience {cfg.patience}"
        f", smoothing window {cfg.patience_smoothing}"
    )

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
            "best_smoothed_sortino": best_smoothed_sortino,
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
        current_entropy_coeff = cosine_entropy_coeff(
            cfg.entropy_coeff_start, cfg.entropy_coeff_end, epoch, cfg.n_epochs,
        )

        epoch_start = time.perf_counter()

        epoch_rng = np.random.default_rng(cfg.seed + epoch * 7919)
        train_episodes = generate_episodes(train_data, cfg, epoch_rng)
        rank_train_episodes = train_episodes[_rank::_world]

        if epoch == start_epoch:
            _log(f"    {'Training episodes':<20s}: {len(train_episodes)} (epoch {epoch})")

        rollout, model_sortino, baseline_sortino = collect_training_rollout(
            model, rank_train_episodes, cfg, global_market_features,
            reward_normalizer, distributed,
        )

        if distributed:
            model_sortino, baseline_sortino = _allreduce_means(
                model_sortino, baseline_sortino,
                len(rank_train_episodes), cfg.device,
            )

        n_local_transitions = rollout.compute_gae(
            cfg.gamma, cfg.gae_lambda, cfg.pytorch_eps,
            distributed, cfg.device,
        )
        if _world > 1:
            n_padded_transitions = _allreduce_max_int(
                n_local_transitions, cfg.device,
            )
        else:
            n_padded_transitions = n_local_transitions

        epoch_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(cfg.ppo_epochs):
            losses = run_ppo_update(
                model, optimizer, scaler, rollout, cfg,
                n_local_transitions, n_padded_transitions,
                current_entropy_coeff,
            )
            for k in epoch_losses:
                epoch_losses[k] += losses[k]
        for k in epoch_losses:
            epoch_losses[k] /= cfg.ppo_epochs

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

        run_ablation = (
            cfg.ablation_interval > 0 and epoch % cfg.ablation_interval == 0
        )
        if run_ablation:
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
            ("train_model", model_sortino),
            ("train_baseline", baseline_sortino),
            ("val_model", val_aggregated["model_sortino"]),
            ("val_baseline", val_aggregated["baseline_sortino"]),
            ("beat_rate", val_aggregated["beat_rate"]),
            ("policy_loss", epoch_losses["policy_loss"]),
            ("value_loss", epoch_losses["value_loss"]),
            ("entropy", epoch_losses["entropy"]),
        ]:
            history[key].append(value)

        window = cfg.patience_smoothing
        recent_vals = history["val_model"][-window:]
        smoothed_val_sortino = float(np.mean(recent_vals))

        improved = smoothed_val_sortino > best_smoothed_sortino
        if improved:
            best_smoothed_sortino = smoothed_val_sortino
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed = time.perf_counter() - epoch_start
        _log()
        _log(f"[Epoch {epoch}/{cfg.n_epochs}] ({elapsed:.1f}s)")
        _log(f"    {'Train sortino':<20s}: {model_sortino:.4f}")
        _log(f"    {'Train baseline':<20s}: {baseline_sortino:.4f}")
        _log(f"    {'Val sortino':<20s}: {val_aggregated['model_sortino']:.4f}")
        _log(f"    {'Val baseline':<20s}: {val_aggregated['baseline_sortino']:.4f}")
        _log(f"    {'Val smoothed':<20s}: {smoothed_val_sortino:.4f}")
        _log(f"    {'Beat rate':<20s}: {val_aggregated['beat_rate']:.1%}")
        _log(f"    {'Policy loss':<20s}: {epoch_losses['policy_loss']:.4f}")
        _log(f"    {'Value loss':<20s}: {epoch_losses['value_loss']:.4f}")
        _log(f"    {'Entropy':<20s}: {epoch_losses['entropy']:.4f}")
        _log(f"    {'LR':<20s}: {current_lr:.2e}")
        _log(f"    {'Entropy coeff':<20s}: {current_entropy_coeff:.4f}")
        _log(f"    {'Patience':<20s}: {patience_counter}/{cfg.patience}")
        if run_ablation:
            _log(f"    {'Val no-market':<20s}: {ablation_aggregated['no_market']:.4f}")
            _log(f"    {'Val no-stock':<20s}: {ablation_aggregated['no_stock']:.4f}")
        if improved:
            _log(f"    {'New best smoothed':<20s}: {best_smoothed_sortino:.4f}")

        if is_main:
            ckpt = _build_checkpoint()
            if improved:
                save_checkpoint(ckpt, os.path.join(cfg.save_dir, "model_best.pt"))
            save_checkpoint(
                ckpt,
                os.path.join(cfg.save_dir, "model_latest.pt"),
            )
            plot_training(
                dict(history),
                os.path.join(cfg.save_dir, "training_curves.png"),
            )

        if patience_counter >= cfg.patience:
            _log(f"    Early stopping at epoch {epoch} (patience {cfg.patience})")
            break

    _log()
    _log(f"Training complete, best smoothed validation: {best_smoothed_sortino:.4f}")

    best_path = os.path.join(cfg.save_dir, "model_best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])

    rank_test_episodes = test_episodes[_rank::_world]
    test_result = evaluate_episodes(
        model, rank_test_episodes, cfg, global_market_features,
    )
    if distributed:
        test_aggregated = _allreduce_eval(
            test_result, len(rank_test_episodes), cfg.device,
        )
    else:
        test_aggregated = test_result

    _log()
    _log(f"[Final Evaluation] {len(test_episodes)} episodes")
    _log(f"    {'Model sortino':<20s}: {test_aggregated['model_sortino']:.4f}")
    _log(f"    {'Baseline sortino':<20s}: {test_aggregated['baseline_sortino']:.4f}")
    _log(f"    {'Beat rate':<20s}: {test_aggregated['beat_rate']:.1%}")

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
    train(cfg)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description="PPO + Transformer RL for stock trading")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Transformer dropout rate (overrides Config.dropout)")
    parser.add_argument("--market_dropout", type=float, default=None,
                        help="Market cross-attention dropout rate (overrides Config.market_dropout)")
    args = parser.parse_args()

    cfg = Config()
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.market_dropout is not None:
        cfg.market_dropout = args.market_dropout

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
    else:
        train(cfg)


if __name__ == "__main__":
    main()
