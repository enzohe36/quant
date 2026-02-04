"""
Stock Trading RL Model Training Script
Simplified implementation
"""

import os
import sys
import time
import random
import math
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Get machine epsilon once for use throughout the script
EPS = float(np.finfo(np.float32).eps)

# Configuration
CONFIG = {
    'train_path': "data/train.csv",
    'test_path': "data/test.csv",
    'output_dir': "models/",
    'sim_length': 240,
    'seq_length': 60,
    'gamma': 0.99,
    'lambda_': 0.95,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    'episodes_per_update': 1024,
    'batch_size': 8192,
    'num_epochs': 10,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_rate': 0.8,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.05,
    'initial_cash': 100.0,
    'test_groups': 1000,
    'sims_per_group': 30,
}
CONFIG['required_length'] = CONFIG['sim_length'] + CONFIG['seq_length']

NUM_CPUS = max(1, cpu_count() - 1)
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Global state
_EPISODE_DATA = {}
_TRAIN_LOGGER = None

# Log format header (shared between file and console)
LOG_HEADER = f"{'Epoch':>5} {'Update':>11} | {'Eps':>7} | {'PolL':>7} | {'ValL':>7} | {'Ent':>7} | {'Grad':>7} | {'Ret':>7}"


@dataclass
class Experience:
    states: np.ndarray
    state_infos: np.ndarray
    actions: np.ndarray
    advantages: np.ndarray
    value_targets: np.ndarray
    holdings: np.ndarray
    is_random: np.ndarray  # track which actions were random exploration


class TradingModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=0.1 if num_layers > 1 else 0)
        self.state_fc = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32))
        combined_dim = hidden_dim + 32
        self.policy_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.certainty_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.value_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq: torch.Tensor, state: torch.Tensor):
        lstm_out, _ = self.lstm(seq)
        combined = torch.cat([lstm_out[:, -1, :], self.state_fc(state)], dim=1)
        return self.policy_head(combined), self.certainty_head(combined), self.value_head(combined)


def setup_logger(output_dir: str) -> logging.Logger:
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(output_dir, 'training.log'), mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['symbol', 'date']).reset_index(drop=True)


def get_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c not in ['symbol', 'date']]
    price_cols = [c for c in df.columns if c.startswith('p_')]
    return feature_cols, price_cols


def init_episode_data(df: pd.DataFrame, feature_cols: List[str], price_col_indices: List[int]):
    global _EPISODE_DATA
    df = df.reset_index(drop=True)

    symbol_indices = {}
    current_symbol, start_idx = None, 0
    symbols = df['symbol'].values

    for i, symbol in enumerate(symbols):
        if symbol != current_symbol:
            if current_symbol is not None:
                symbol_indices[current_symbol] = (start_idx, i)
            current_symbol, start_idx = symbol, i
    if current_symbol is not None:
        symbol_indices[current_symbol] = (start_idx, len(symbols))

    _EPISODE_DATA = {
        'features': df[feature_cols].values.astype(np.float32),
        'prices': df[['p_open', 'p_close']].values.astype(np.float32),
        'symbol_indices': symbol_indices,
        'price_col_indices': price_col_indices,
    }


def _build_pool_worker(args) -> List[Tuple[str, int]]:
    symbol, n_rows, required_length = args
    if n_rows >= required_length:
        return [(symbol, pos) for pos in range(required_length - 1, n_rows)]
    return []


def build_pool(df: pd.DataFrame, required_length: int) -> List[Tuple[str, int]]:
    start = time.time()
    symbols = df['symbol'].unique()
    args_list = [(s, len(df[df['symbol'] == s]), required_length) for s in symbols]

    pool = []
    with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        for result in executor.map(_build_pool_worker, args_list):
            pool.extend(result)

    print(f"  [build_pool] {time.time() - start:.2f}s - {len(pool)} samples")
    return pool


def prepare_episodes(samples: List[Tuple[str, int]]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    features_arr = _EPISODE_DATA['features']
    prices_arr = _EPISODE_DATA['prices']
    symbol_indices = _EPISODE_DATA['symbol_indices']
    price_col_indices = _EPISODE_DATA['price_col_indices']
    req_len = CONFIG['required_length']
    seq_len = CONFIG['seq_length']

    all_features, all_prices, all_open_1 = [], [], []

    for symbol, end_pos in samples:
        sym_start, _ = symbol_indices[symbol]
        global_start = sym_start + end_pos - req_len + 1
        global_end = sym_start + end_pos + 1

        features = features_arr[global_start:global_end].copy()
        prices = prices_arr[global_start:global_end].copy()

        p_close_0 = prices[seq_len, 1]
        features[:, price_col_indices] /= p_close_0

        open_1 = prices[seq_len + 1, 0] if seq_len + 1 < len(prices) else prices[seq_len, 1]

        all_features.append(features)
        all_prices.append(prices)
        all_open_1.append(open_1)

    return all_features, all_prices, all_open_1


def get_epsilon(update: int, total_updates: int) -> float:
    decay_updates = int(total_updates * CONFIG['epsilon_decay_rate'])
    if update >= decay_updates:
        return CONFIG['epsilon_end']
    ratio = update / decay_updates
    return CONFIG['epsilon_start'] * (CONFIG['epsilon_end'] / CONFIG['epsilon_start']) ** ratio


def compute_trades(actions: np.ndarray, certainties: np.ndarray, cash: np.ndarray,
                   holding: np.ndarray, open_next: np.ndarray) -> np.ndarray:
    trades = np.zeros_like(cash, dtype=np.float32)

    for mask, sign in [(actions == 1, 1), (actions == -1, -1)]:
        if mask.any():
            open_safe = np.maximum(open_next[mask], EPS)
            potential = cash[mask] / open_safe + holding[mask]
            desired = certainties[mask] * potential
            max_trade = (cash[mask] / open_safe) if sign == 1 else holding[mask]
            trades[mask] = sign * np.floor(np.minimum(desired, max_trade))

    return trades


def run_simulations(samples: List[Tuple[str, int]], epsilon: float, n_features: int,
                    price_col_indices: List[int], model: Optional[nn.Module],
                    device: torch.device) -> List[Tuple]:
    all_features, all_prices, all_open_1 = prepare_episodes(samples)

    n_episodes = len(all_features)
    sim_len, seq_len = CONFIG['sim_length'], CONFIG['seq_length']
    initial_cash = CONFIG['initial_cash']

    features_stacked = np.stack(all_features)
    prices_stacked = np.stack(all_prices)
    open_1_arr = np.array(all_open_1, dtype=np.float32)

    cash = np.full(n_episodes, initial_cash, dtype=np.float32)
    holding = np.zeros(n_episodes, dtype=np.float32)

    all_states = np.zeros((n_episodes, sim_len, seq_len, n_features), dtype=np.float32)
    all_state_infos = np.zeros((n_episodes, sim_len, 3), dtype=np.float32)
    all_actions = np.zeros((n_episodes, sim_len), dtype=np.int64)
    all_is_random = np.zeros((n_episodes, sim_len), dtype=np.bool_)  # track random actions
    all_portfolios = np.zeros((n_episodes, sim_len + 1), dtype=np.float32)

    if model is not None:
        model.eval()

    for step in range(sim_len):
        day = seq_len + step
        p_close = prices_stacked[:, day, 1]
        portfolio = cash + holding * p_close
        all_portfolios[:, step] = portfolio

        all_states[:, step] = features_stacked[:, day - seq_len + 1:day + 1]
        all_state_infos[:, step] = np.stack([cash, holding, portfolio], axis=1) / initial_cash

        is_random = np.random.random(n_episodes) < epsilon
        all_is_random[:, step] = is_random  # store random flags
        actions = np.zeros(n_episodes, dtype=np.int64)
        certainties = np.random.random(n_episodes).astype(np.float32)

        # Random actions
        if is_random.any():
            rand_idx = np.where(is_random)[0]
            has_holding = holding[rand_idx] > 0
            rand_vals = np.random.random(len(rand_idx))
            actions[rand_idx] = np.where(has_holding, np.floor(rand_vals * 3).astype(np.int64) - 1,
                                         np.floor(rand_vals * 2).astype(np.int64))

        # Model actions
        non_rand_idx = np.where(~is_random)[0]
        if len(non_rand_idx) > 0 and model is not None:
            batch_seqs = all_states[non_rand_idx, step].copy()
            batch_seqs[:, :, price_col_indices] -= 1.0
            batch_infos = all_state_infos[non_rand_idx, step] - 1.0

            with torch.no_grad():
                seqs_t = torch.from_numpy(batch_seqs).to(device)
                infos_t = torch.from_numpy(batch_infos.astype(np.float32)).to(device)
                logits, cert_preds, _ = model(seqs_t, infos_t)
                logits[holding[non_rand_idx] <= 0, 0] = -1e9
                probs = F.softmax(logits, dim=1)
                actions[non_rand_idx] = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy() - 1
                certainties[non_rand_idx] = cert_preds.squeeze(-1).cpu().numpy()

        all_actions[:, step] = actions + 1

        open_next = prices_stacked[:, day + 1, 0] if day + 1 < prices_stacked.shape[1] else p_close
        trades = compute_trades(actions, certainties, cash, holding, open_next)
        holding += trades
        cash -= trades * open_next

    final_p_close = prices_stacked[:, seq_len + sim_len - 1, 1]
    final_portfolio = cash + holding * final_p_close
    all_portfolios[:, sim_len] = final_portfolio

    if model is not None:
        model.train()

    results = []
    for i in range(n_episodes):
        rewards = np.diff(all_portfolios[i]) / initial_cash
        baseline = (final_p_close[i] - open_1_arr[i]) / open_1_arr[i] if open_1_arr[i] > 0 else 0.0
        results.append((all_states[i], all_state_infos[i], all_actions[i], rewards,
                       final_portfolio[i], baseline, all_is_random[i]))  # include is_random

    return results


def compute_gae(rewards: np.ndarray, values: np.ndarray, final_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE with proper bootstrapping.

    Args:
        rewards: Array of rewards of length T
        values: Array of value estimates of length T
        final_value: Bootstrap value for the terminal state (V(s_T+1))
    """
    gamma, lambda_ = CONFIG['gamma'], CONFIG['lambda_']
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        # Use final_value for bootstrapping at the last step
        next_val = values[t + 1] if t + 1 < len(values) else final_value
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages[t] = gae

    return advantages, advantages + values[:T]


def collect_experiences(results: List[Tuple], price_col_indices: List[int],
                        model: nn.Module, device: torch.device, batch_size: int = 4096) -> Experience:
    all_states, all_infos, all_actions, all_rewards, all_holdings, all_is_random, lengths = [], [], [], [], [], [], []
    final_states_list, final_infos_list = [], []  # For computing bootstrap values

    for states, state_infos, actions, rewards, final_portfolio, _, is_random in results:
        model_states = states.copy()
        model_states[:, :, price_col_indices] -= 1.0
        all_states.append(model_states)
        all_infos.append((state_infos - 1.0).astype(np.float32))
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_holdings.append(state_infos[:, 1])
        all_is_random.append(is_random)
        lengths.append(len(states))

        # Store final state info for bootstrap value computation
        # Use the last state as approximation for terminal state
        final_states_list.append(model_states[-1:])
        final_infos_list.append(all_infos[-1][-1:])

    states_concat = np.concatenate(all_states)
    infos_concat = np.concatenate(all_infos)

    model.eval()
    all_values = []
    with torch.no_grad():
        for start in range(0, len(states_concat), batch_size):
            end = min(start + batch_size, len(states_concat))
            states_t = torch.from_numpy(states_concat[start:end]).to(device)
            infos_t = torch.from_numpy(infos_concat[start:end]).to(device)
            _, _, values = model(states_t, infos_t)
            all_values.append(values.squeeze(-1).cpu().numpy())

    # Compute bootstrap values for terminal states
    final_states_concat = np.concatenate(final_states_list)
    final_infos_concat = np.concatenate(final_infos_list)
    final_values = []
    with torch.no_grad():
        for start in range(0, len(final_states_concat), batch_size):
            end = min(start + batch_size, len(final_states_concat))
            states_t = torch.from_numpy(final_states_concat[start:end]).to(device)
            infos_t = torch.from_numpy(final_infos_concat[start:end]).to(device)
            _, _, values = model(states_t, infos_t)
            final_values.append(values.squeeze(-1).cpu().numpy())
    final_values = np.concatenate(final_values)

    model.train()

    values_concat = np.concatenate(all_values)

    all_advantages, all_targets = [], []
    idx = 0
    for i, length in enumerate(lengths):
        # Pass the bootstrap value for proper GAE computation
        adv, targets = compute_gae(all_rewards[i], values_concat[idx:idx + length], final_values[i])
        all_advantages.append(adv)
        all_targets.append(targets)
        idx += length

    return Experience(
        states=states_concat, state_infos=infos_concat,
        actions=np.concatenate(all_actions), advantages=np.concatenate(all_advantages),
        value_targets=np.concatenate(all_targets), holdings=np.concatenate(all_holdings),
        is_random=np.concatenate(all_is_random)
    )


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, exp: Experience,
               device: torch.device, batch_size: int, epsilon: float) -> Dict[str, float]:
    T = len(exp.states)
    if T == 0:
        return {'loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'entropy_unmasked': 0, 'grad_norm': 0}

    states_t = torch.from_numpy(exp.states).to(device)
    infos_t = torch.from_numpy(exp.state_infos).to(device)
    actions_t = torch.from_numpy(exp.actions).to(device)
    advantages_t = torch.from_numpy(exp.advantages).to(device)
    targets_t = torch.from_numpy(exp.value_targets).to(device)
    holdings_t = torch.from_numpy(exp.holdings).to(device)
    is_random_t = torch.from_numpy(exp.is_random).to(device)

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + EPS)

    totals = {'policy': 0, 'value': 0, 'entropy': 0, 'entropy_u': 0, 'grad': 0}
    indices = np.random.permutation(T)
    num_batches = 0

    for start in range(0, T, batch_size):
        idx = indices[start:min(start + batch_size, T)]
        logits, _, values = model(states_t[idx], infos_t[idx])
        values = values.squeeze(-1)

        probs_u = F.softmax(logits, dim=1)
        log_probs_u = F.log_softmax(logits, dim=1)
        entropy_u = -(probs_u * log_probs_u).sum(dim=1).mean()

        mask = torch.ones_like(logits)
        mask[:, 0] = (holdings_t[idx] > 0).float()
        logits_masked = logits + (1 - mask) * (-1e9)

        log_probs = F.log_softmax(logits_masked, dim=1)
        probs = F.softmax(logits_masked, dim=1)
        action_log_probs = log_probs.gather(1, actions_t[idx].unsqueeze(1)).squeeze(1)
        action_probs = probs.gather(1, actions_t[idx].unsqueeze(1)).squeeze(1)

        # Importance sampling weights
        # π_behavior = ε * uniform + (1-ε) * π_model
        # For 3 actions with masking, uniform prob depends on whether sell is available
        n_valid_actions = mask.sum(dim=1)  # 2 or 3 depending on holdings
        uniform_prob = 1.0 / n_valid_actions
        pi_behavior = epsilon * uniform_prob + (1 - epsilon) * action_probs
        importance_weights = action_probs / (pi_behavior + EPS)

        # Clip weights to prevent extreme values (PPO-style)
        importance_weights = torch.clamp(importance_weights, 0.1, 10.0)

        # Only apply weights to random actions; model actions have weight ≈ 1
        weights = torch.where(is_random_t[idx], importance_weights, torch.ones_like(importance_weights))

        policy_loss = -(action_log_probs * advantages_t[idx] * weights).mean()
        value_loss = F.mse_loss(values, targets_t[idx])

        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = policy_loss + CONFIG['value_loss_coef'] * value_loss - CONFIG['entropy_coef'] * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()

        totals['policy'] += policy_loss.item()
        totals['value'] += value_loss.item()
        totals['entropy'] += entropy.item()
        totals['entropy_u'] += entropy_u.item()
        totals['grad'] += grad_norm.item()
        num_batches += 1

    del states_t, infos_t, actions_t, advantages_t, targets_t, holdings_t, is_random_t
    torch.cuda.empty_cache()

    n = max(num_batches, 1)
    return {
        'loss': (totals['policy'] + CONFIG['value_loss_coef'] * totals['value']) / n,
        'policy_loss': totals['policy'] / n, 'value_loss': totals['value'] / n,
        'entropy': totals['entropy'] / n, 'entropy_unmasked': totals['entropy_u'] / n,
        'grad_norm': totals['grad'] / n
    }


def run_test(pool: List[Tuple[str, int]], n_features: int, price_col_indices: List[int],
             model: nn.Module, device: torch.device, batch_size: int) -> Tuple[List[float], List[float]]:
    start = time.time()
    test_groups, sims_per_group = CONFIG['test_groups'], CONFIG['sims_per_group']
    print(f"  Running {test_groups * sims_per_group} test simulations...")

    all_samples = []
    group_indices = []
    for g in range(test_groups):
        samples = random.sample(pool, k=sims_per_group)
        all_samples.extend(samples)
        group_indices.extend([g] * sims_per_group)
    group_indices = np.array(group_indices)

    all_features, all_prices, all_open_1 = prepare_episodes(all_samples)

    n_episodes = len(all_features)
    sim_len, seq_len = CONFIG['sim_length'], CONFIG['seq_length']
    initial_cash = CONFIG['initial_cash']

    features_stacked = np.stack(all_features)
    prices_stacked = np.stack(all_prices)
    open_1_arr = np.array(all_open_1, dtype=np.float32)

    cash = np.full(n_episodes, initial_cash, dtype=np.float32)
    holding = np.zeros(n_episodes, dtype=np.float32)

    model.eval()
    for step in range(sim_len):
        day = seq_len + step
        p_close = prices_stacked[:, day, 1]
        portfolio = cash + holding * p_close

        batch_seqs = features_stacked[:, day - seq_len + 1:day + 1].copy()
        batch_seqs[:, :, price_col_indices] -= 1.0
        batch_infos = (np.stack([cash, holding, portfolio], axis=1) / initial_cash - 1.0).astype(np.float32)

        actions = np.zeros(n_episodes, dtype=np.int64)
        certainties = np.zeros(n_episodes, dtype=np.float32)

        for bs in range(0, n_episodes, batch_size):
            be = min(bs + batch_size, n_episodes)
            with torch.no_grad():
                seqs_t = torch.from_numpy(batch_seqs[bs:be]).to(device)
                infos_t = torch.from_numpy(batch_infos[bs:be]).to(device)
                logits, cert, _ = model(seqs_t, infos_t)
                logits[holding[bs:be] <= 0, 0] = -1e9
                probs = F.softmax(logits, dim=1)
                actions[bs:be] = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy() - 1
                certainties[bs:be] = cert.squeeze(-1).cpu().numpy()

        open_next = prices_stacked[:, day + 1, 0] if day + 1 < prices_stacked.shape[1] else p_close
        trades = compute_trades(actions, certainties, cash, holding, open_next)
        holding += trades
        cash -= trades * open_next

    model.train()

    final_p_close = prices_stacked[:, seq_len + sim_len - 1, 1]
    final_portfolio = cash + holding * final_p_close

    port_changes = (final_portfolio - initial_cash) / initial_cash
    base_changes = (final_p_close - open_1_arr) / np.where(open_1_arr > 0, open_1_arr, 1.0)

    port_means = [port_changes[group_indices == g].mean() for g in range(test_groups)]
    base_means = [base_changes[group_indices == g].mean() for g in range(test_groups)]

    print(f"  [run_test] {time.time() - start:.2f}s")
    print(f"  Portfolio: mean={np.mean(port_means):.4f}, std={np.std(port_means):.4f}")
    print(f"  Baseline:  mean={np.mean(base_means):.4f}, std={np.std(base_means):.4f}")

    return port_means, base_means


def plot_results(losses: List[float], returns: List[float],
                 test_port: List[float], test_base: List[float], output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(losses, 'b-', alpha=0.7)
    axes[0, 0].set(xlabel='Update', ylabel='Loss', title='Training Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(returns, 'g-', alpha=0.7)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set(xlabel='Update', ylabel='Return', title='Training Returns')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(test_port, bins=50, alpha=0.7, label='Model', color='blue')
    axes[1, 0].hist(test_base, bins=50, alpha=0.7, label='Baseline', color='orange')
    axes[1, 0].legend()
    axes[1, 0].set(xlabel='Return', ylabel='Frequency', title='Test Returns Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(test_base, test_port, alpha=0.3, s=10)
    min_val, max_val = min(min(test_base), min(test_port)), max(max(test_base), max(test_port))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    axes[1, 1].set(xlabel='Baseline Return', ylabel='Model Return', title='Model vs Baseline')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'), dpi=150)
    plt.close()


def format_log_line(epoch: int, update: int, total_updates: int, epsilon: float,
                    metrics: Dict[str, float], mean_return: float) -> str:
    """Format a log line matching the header format."""
    return f"{epoch:>5} {update:>5}/{total_updates:<5} | {epsilon:>7.3f} | {metrics['policy_loss']:>7.3f} | {metrics['value_loss']:>7.3f} | {metrics['entropy']:>7.3f} | {metrics['grad_norm']:>7.3f} | {mean_return:>7.3f}"


def main():
    global _TRAIN_LOGGER

    print("\nSTOCK TRADING RL TRAINING\n")
    print(f"  CPUs: {NUM_CPUS}, GPUs: {NUM_GPUS if NUM_GPUS else 'None'}")
    print(f"  Machine epsilon: {EPS}")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    _TRAIN_LOGGER = setup_logger(CONFIG['output_dir'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Load data
    print("\nLOADING DATA\n")
    start = time.time()
    train_df = load_data(CONFIG['train_path'])
    print(f"  [load train] {time.time() - start:.2f}s - {len(train_df)} rows, {train_df['symbol'].nunique()} symbols")

    start = time.time()
    test_df = load_data(CONFIG['test_path'])
    print(f"  [load test] {time.time() - start:.2f}s - {len(test_df)} rows, {test_df['symbol'].nunique()} symbols")

    feature_cols, price_cols = get_columns(train_df)
    price_col_indices = [feature_cols.index(c) for c in price_cols]
    n_features = len(feature_cols)
    print(f"  Features: {n_features}")

    # Initialize
    print("\nINITIALIZING\n")
    start = time.time()
    init_episode_data(train_df, feature_cols, price_col_indices)
    print(f"  [init_episode_data] {time.time() - start:.2f}s")

    train_pool = build_pool(train_df, CONFIG['required_length'])
    test_pool = build_pool(test_df, CONFIG['required_length'])

    if not train_pool:
        print("ERROR: No valid training samples!")
        return

    # Model
    print("\nMODEL\n")
    start = time.time()
    model = TradingModel(n_features).to(device)
    if NUM_GPUS > 1:
        model = DataParallel(model)
        print(f"  Using {NUM_GPUS} GPUs")
    base_model = model.module if isinstance(model, DataParallel) else model
    print(f"  [create_model] {time.time() - start:.2f}s - {sum(p.numel() for p in model.parameters()):,} params")

    effective_batch = CONFIG['batch_size'] * max(1, NUM_GPUS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

    updates_per_epoch = math.ceil(len(train_pool) / CONFIG['episodes_per_update'])
    total_updates = CONFIG['num_epochs'] * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=CONFIG['lr']/10)

    # Training
    print("\nTRAINING\n")
    print(f"  {len(train_pool)} samples, {updates_per_epoch} updates/epoch, {total_updates} total")
    print(f"  Log: {CONFIG['output_dir']}training.log")

    # Log header to file
    _TRAIN_LOGGER.info(LOG_HEADER)

    # Print header to console (will stay at top while values update below)
    print(f"\n  {LOG_HEADER}")

    train_losses, train_returns = [], []
    global_update = 0

    for epoch in range(CONFIG['num_epochs']):
        epoch_pool = train_pool.copy()
        random.shuffle(epoch_pool)
        pool_idx = 0

        while pool_idx < len(epoch_pool):
            end_idx = min(pool_idx + CONFIG['episodes_per_update'], len(epoch_pool))
            sampled = epoch_pool[pool_idx:end_idx]
            pool_idx = end_idx

            epsilon = get_epsilon(global_update, total_updates)

            results = run_simulations(sampled, epsilon, n_features, price_col_indices, base_model, device)
            experiences = collect_experiences(results, price_col_indices, model, device)
            metrics = train_step(model, optimizer, experiences, device, effective_batch, epsilon)
            scheduler.step()

            mean_return = np.mean([(r[4] - CONFIG['initial_cash']) / CONFIG['initial_cash'] for r in results])
            train_losses.append(metrics['loss'])
            train_returns.append(mean_return)

            del results, experiences

            # Format the log line (same format for file and console)
            log_line = format_log_line(epoch + 1, global_update + 1, total_updates,
                                       epsilon, metrics, mean_return)

            # Log to file
            _TRAIN_LOGGER.info(log_line)

            # Console: update value line in place (same format as log)
            sys.stdout.write(f"\r  {log_line}")
            sys.stdout.flush()

            global_update += 1

        print(f"\n  Epoch {epoch+1} complete")

    # Save
    print("\nSAVING\n")
    start = time.time()
    state_dict = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
    torch.save(state_dict, os.path.join(CONFIG['output_dir'], 'trading_model.pt'))
    print(f"  [save_model] {time.time() - start:.2f}s")

    # Test
    print("\nTESTING\n")
    start = time.time()
    init_episode_data(test_df, feature_cols, price_col_indices)
    print(f"  [init_episode_data] {time.time() - start:.2f}s")

    test_port, test_base = run_test(test_pool, n_features, price_col_indices, model, device, effective_batch)

    # Plot
    print("\nPLOTTING\n")
    start = time.time()
    plot_results(train_losses, train_returns, test_port, test_base, CONFIG['output_dir'])
    print(f"  [plot] {time.time() - start:.2f}s")
    print(f"  Saved: {CONFIG['output_dir']}training_results.png")

    print("\nCOMPLETE\n")


if __name__ == "__main__":
    main()
