"""
Stock Trading with PPO and Beta Policy
Multi-GPU distributed training for portfolio optimization
"""

import os
import sys
import time
import math
import random
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions import Beta
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

DATA_PATH = "data_train.csv"
OUTPUT_DIR = "models/"

SEQ_LEN = 60
SIM_LEN = 240
REQ_LEN = SEQ_LEN + SIM_LEN
INIT_CASH = 1e6
TRAIN_SPLIT = 0.8

HIDDEN_SIZE = 128
POSITION_EMBED_SIZE = 32

GAMMA = 0.99
GAE_LAMBDA = 0.95
POLICY_CLIP = 0.2
VALUE_COEF = 0.5
PPO_EPOCHS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 0.5
ENTROPY_COEF_START = 0.05
ENTROPY_COEF_END = 0.001
ENTROPY_DECAY_FRAC = 0.8

MIN_UPDATES = 100
EVAL_INTERVAL = 50
PATIENCE_FRAC = 0.2
MAX_COVERAGE = 0.999

EPISODES_PER_UPDATE = 1024
BATCH_SIZE = 4096
TEST_GROUPS = 1000
SIMS_PER_GROUP = 30

SEED = 42
MACHINE_EPS = float(np.finfo(np.float64).eps)
NUM_CPUS = max(1, cpu_count() - 1)
NUM_GPUS = max(1, torch.cuda.device_count())

_GLOBAL_DATA = {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_base_model(model):
    return model.module if isinstance(model, DDP) else model


class TradingPolicyNetwork(nn.Module):
    def __init__(self, num_features, hidden_size=HIDDEN_SIZE, position_embed_size=POSITION_EMBED_SIZE):
        super().__init__()
        self.feature_encoder = nn.LSTM(num_features, hidden_size, num_layers=2, batch_first=True)
        self.position_encoder = nn.Sequential(nn.Linear(1, position_embed_size), nn.ReLU())
        combined_size = hidden_size + position_embed_size
        self.policy_head = nn.Sequential(nn.Linear(combined_size, 64), nn.ReLU(), nn.Linear(64, 2))
        self.value_head = nn.Sequential(nn.Linear(combined_size, 64), nn.ReLU(), nn.Linear(64, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, feature_sequence, position):
        lstm_out, _ = self.feature_encoder(feature_sequence)
        encoded = torch.cat([lstm_out[:, -1], self.position_encoder(position)], dim=1)
        policy_params = self.policy_head(encoded)
        alpha = F.softplus(policy_params[:, 0]) + 1.0
        beta = F.softplus(policy_params[:, 1]) + 1.0
        value = self.value_head(encoded).squeeze(-1)
        return alpha, beta, value


def _build_segments_worker(args):
    symbols, symbol_ranges, required_length = args
    segments = []
    for symbol in symbols:
        start_idx, end_idx = symbol_ranges[symbol]
        num_rows = end_idx - start_idx
        for i in range(num_rows - required_length + 1):
            segments.append((symbol, i, i + required_length - 1))
    return segments


def initialize_global_data(dataframe, feature_columns):
    global _GLOBAL_DATA
    dataframe = dataframe.sort_values(['symbol', 'date']).reset_index(drop=True)
    symbols = dataframe['symbol'].values
    symbol_ranges = {}
    start_idx, prev_symbol = 0, symbols[0]
    for i in range(1, len(symbols)):
        if symbols[i] != prev_symbol:
            symbol_ranges[prev_symbol] = (start_idx, i)
            prev_symbol, start_idx = symbols[i], i
    symbol_ranges[prev_symbol] = (start_idx, len(symbols))
    _GLOBAL_DATA = {
        'features': dataframe[feature_columns].values.astype(np.float32),
        'open_prices': dataframe['open'].values.astype(np.float32),
        'close_prices': dataframe['close'].values.astype(np.float32),
        'symbol_ranges': symbol_ranges,
    }


def build_segment_pool(symbols):
    symbol_ranges = _GLOBAL_DATA['symbol_ranges']
    valid_symbols = [s for s in symbols if s in symbol_ranges]
    chunk_size = max(1, len(valid_symbols) // (NUM_CPUS * 4) + 1)
    worker_args = [
        (valid_symbols[i:i + chunk_size], symbol_ranges, REQ_LEN)
        for i in range(0, len(valid_symbols), chunk_size)
    ]
    pool = []
    with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        for chunk in executor.map(_build_segments_worker, worker_args):
            pool.extend(chunk)
    return pool


def build_symbol_stratified_pools(segment_pool):
    stratified_pools = {}
    for symbol, start, end in segment_pool:
        stratified_pools.setdefault(symbol, []).append((symbol, start, end))
    return stratified_pools


def sample_stratified_segments(stratified_pools, num_samples, rng):
    symbols = list(stratified_pools.keys())
    working_pools = {s: list(p) for s, p in stratified_pools.items()}
    for s in working_pools:
        rng.shuffle(working_pools[s])
    samples = []
    symbol_idx = 0
    for _ in range(num_samples):
        for _ in range(len(symbols)):
            symbol = symbols[symbol_idx]
            symbol_idx = (symbol_idx + 1) % len(symbols)
            if working_pools[symbol]:
                samples.append(working_pools[symbol].pop())
                break
        else:
            break
    return samples


def extract_episode_data(samples):
    features = _GLOBAL_DATA['features']
    open_prices = _GLOBAL_DATA['open_prices']
    close_prices = _GLOBAL_DATA['close_prices']
    symbol_ranges = _GLOBAL_DATA['symbol_ranges']
    global_starts = np.array([symbol_ranges[s][0] + i for s, i, _ in samples], dtype=np.int64)
    row_indices = np.arange(REQ_LEN)[None, :] + global_starts[:, None]
    return features[row_indices], open_prices[row_indices], close_prices[row_indices]


def simulate_trading(samples, model, device, is_training=True):
    position_mean = 0.5
    position_alpha = F.softplus(torch.tensor(0.0)).item() + 1.0
    position_std = 1.0 / (4.0 * (2.0 * position_alpha + 1.0))

    features, open_prices, close_prices = extract_episode_data(samples)
    num_episodes = len(samples)

    cash = np.full(num_episodes, INIT_CASH, np.float32)
    holdings = np.zeros(num_episodes, np.float32)

    stored_states = np.zeros((num_episodes, SIM_LEN, SEQ_LEN, features.shape[-1]), np.float32)
    stored_positions = np.zeros((num_episodes, SIM_LEN), np.float32)
    stored_targets = np.zeros((num_episodes, SIM_LEN), np.float32)
    stored_logprobs = np.zeros((num_episodes, SIM_LEN), np.float32)
    stored_values = np.zeros((num_episodes, SIM_LEN), np.float32)
    stored_entropies = np.zeros((num_episodes, SIM_LEN), np.float32)
    portfolio_values = np.zeros((num_episodes, SIM_LEN + 1), np.float32)

    base_model = get_base_model(model)
    base_model.eval()

    with torch.no_grad():
        for step in range(SIM_LEN):
            day_idx = SEQ_LEN - 1 + step
            current_close = close_prices[:, day_idx]
            current_portfolio = cash + holdings * current_close
            portfolio_values[:, step] = current_portfolio

            current_position = holdings * current_close / np.maximum(current_portfolio, MACHINE_EPS)
            normalized_position = (current_position - position_mean) / position_std

            feature_window = features[:, step:step + SEQ_LEN]
            stored_states[:, step] = feature_window
            stored_positions[:, step] = normalized_position

            step_targets = np.zeros(num_episodes, np.float32)
            step_logprobs = np.zeros(num_episodes, np.float32)
            step_values = np.zeros(num_episodes, np.float32)
            step_entropies = np.zeros(num_episodes, np.float32)

            for batch_start in range(0, num_episodes, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_episodes)
                features_tensor = torch.from_numpy(feature_window[batch_start:batch_end]).to(device)
                position_tensor = torch.from_numpy(
                    normalized_position[batch_start:batch_end, None].astype(np.float32)
                ).to(device)

                alpha, beta, value = base_model(features_tensor, position_tensor)

                if is_training:
                    distribution = Beta(alpha, beta)
                    sampled_target = distribution.sample()
                    step_targets[batch_start:batch_end] = sampled_target.cpu().numpy()
                    step_logprobs[batch_start:batch_end] = distribution.log_prob(sampled_target).cpu().numpy()
                    step_entropies[batch_start:batch_end] = distribution.entropy().cpu().numpy()
                else:
                    step_targets[batch_start:batch_end] = (alpha / (alpha + beta)).cpu().numpy()
                step_values[batch_start:batch_end] = value.cpu().numpy()

            stored_targets[:, step] = step_targets
            stored_logprobs[:, step] = step_logprobs
            stored_values[:, step] = step_values
            stored_entropies[:, step] = step_entropies

            next_open = open_prices[:, day_idx + 1]
            max_total_shares = cash / next_open + holdings
            desired_shares = step_targets * max_total_shares
            share_delta = desired_shares - holdings
            max_buyable = cash / next_open
            trade_quantity = np.floor(np.clip(share_delta, -holdings, max_buyable)).astype(np.float32)
            cash = cash - trade_quantity * next_open
            holdings = holdings + trade_quantity

        final_close = close_prices[:, SEQ_LEN - 1 + SIM_LEN]
        portfolio_values[:, SIM_LEN] = cash + holdings * final_close
        rewards = portfolio_values[:, 1:] / np.maximum(portfolio_values[:, :-1], MACHINE_EPS) - 1

    return {
        'states': stored_states,
        'positions': stored_positions,
        'targets': stored_targets,
        'logprobs': stored_logprobs,
        'values': stored_values,
        'entropies': stored_entropies,
        'rewards': rewards,
        'final_portfolio': portfolio_values[:, -1],
        'first_open': open_prices[:, SEQ_LEN],
        'final_close': final_close,
    }


def compute_advantages_and_targets(simulation_results, model, device):
    base_model = get_base_model(model)
    with torch.no_grad():
        values = torch.from_numpy(simulation_results['values']).to(device)
        rewards = torch.from_numpy(simulation_results['rewards']).to(device)
        num_episodes, num_steps = values.shape

        bootstrap_values = []
        for batch_start in range(0, num_episodes, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, num_episodes)
            states_tensor = torch.from_numpy(simulation_results['states'][batch_start:batch_end, -1]).to(device)
            positions_tensor = torch.from_numpy(
                simulation_results['positions'][batch_start:batch_end, -1, None].astype(np.float32)
            ).to(device)
            _, _, value = base_model(states_tensor, positions_tensor)
            bootstrap_values.append(value)
        bootstrap = torch.cat(bootstrap_values)

        advantages = torch.zeros_like(values)
        gae = torch.zeros(num_episodes, device=device)
        for t in reversed(range(num_steps)):
            next_value = bootstrap if t == num_steps - 1 else values[:, t + 1]
            td_error = rewards[:, t] + GAMMA * next_value - values[:, t]
            gae = td_error + GAMMA * GAE_LAMBDA * gae
            advantages[:, t] = gae

        value_targets = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + MACHINE_EPS)

    return advantages, value_targets


def prepare_training_batch(simulation_results, advantages, value_targets, device):
    num_features = simulation_results['states'].shape[-1]
    return {
        'states': torch.from_numpy(simulation_results['states'].reshape(-1, SEQ_LEN, num_features)).to(device),
        'positions': torch.from_numpy(simulation_results['positions'].reshape(-1, 1).astype(np.float32)).to(device),
        'targets': torch.from_numpy(simulation_results['targets'].reshape(-1)).to(device),
        'old_logprobs': torch.from_numpy(simulation_results['logprobs'].reshape(-1)).to(device),
        'advantages': advantages.reshape(-1),
        'value_targets': value_targets.reshape(-1),
    }


def compute_entropy_coefficient(current_update, total_updates):
    progress = current_update / total_updates
    if progress < ENTROPY_DECAY_FRAC:
        return ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress / ENTROPY_DECAY_FRAC
    return ENTROPY_COEF_END


def execute_ppo_update(model, optimizer, batch, entropy_coef):
    model.train()
    device = next(model.parameters()).device
    num_samples = len(batch['states'])
    metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'clip_frac': 0.0}
    num_minibatches = 0

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(num_samples, device=device)
        for minibatch_indices in indices.split(BATCH_SIZE):
            if len(minibatch_indices) < BATCH_SIZE // 4:
                continue

            alpha, beta, value = model(
                batch['states'][minibatch_indices],
                batch['positions'][minibatch_indices]
            )
            distribution = Beta(alpha, beta)
            log_probs = distribution.log_prob(batch['targets'][minibatch_indices])
            entropy = distribution.entropy().mean()

            ratio = torch.exp(log_probs - batch['old_logprobs'][minibatch_indices])
            advantages = batch['advantages'][minibatch_indices]
            clipped_ratio = torch.clamp(ratio, 1.0 - POLICY_CLIP, 1.0 + POLICY_CLIP)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            value_loss = F.mse_loss(value, batch['value_targets'][minibatch_indices])
            total_loss = policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            with torch.no_grad():
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['clip_frac'] += ((ratio - 1).abs() > POLICY_CLIP).float().mean().item()
                num_minibatches += 1

    return {k: v / max(num_minibatches, 1) for k, v in metrics.items()}


def _sample_test_groups_worker(args):
    pool, num_sims, seeds = args
    return [random.Random(seed).sample(pool, k=min(num_sims, len(pool))) for seed in seeds]


def evaluate_model(segment_pool, model, device, rank, world_size):
    rng = random.Random(SEED)
    group_seeds = [rng.randint(0, 2**31) for _ in range(TEST_GROUPS)]

    chunk_size = max(1, TEST_GROUPS // NUM_CPUS + 1)
    worker_args = [
        (segment_pool, SIMS_PER_GROUP, group_seeds[i:i + chunk_size])
        for i in range(0, TEST_GROUPS, chunk_size)
    ]

    all_samples = []
    with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        for groups in executor.map(_sample_test_groups_worker, worker_args):
            for group in groups:
                all_samples.extend(group)

    local_samples = all_samples[rank::world_size]
    results = simulate_trading(local_samples, model, device, is_training=False)

    final_portfolio = torch.from_numpy(results['final_portfolio']).to(device)
    first_open = torch.from_numpy(results['first_open']).to(device)
    final_close = torch.from_numpy(results['final_close']).to(device)

    if world_size > 1:
        gathered_portfolio = [torch.zeros_like(final_portfolio) for _ in range(world_size)]
        gathered_open = [torch.zeros_like(first_open) for _ in range(world_size)]
        gathered_close = [torch.zeros_like(final_close) for _ in range(world_size)]
        dist.all_gather(gathered_portfolio, final_portfolio)
        dist.all_gather(gathered_open, first_open)
        dist.all_gather(gathered_close, final_close)
        final_portfolio = torch.cat(gathered_portfolio).cpu().numpy()
        first_open = torch.cat(gathered_open).cpu().numpy()
        final_close = torch.cat(gathered_close).cpu().numpy()
    else:
        final_portfolio = final_portfolio.cpu().numpy()
        first_open = first_open.cpu().numpy()
        final_close = final_close.cpu().numpy()

    model_returns = final_portfolio / INIT_CASH - 1
    baseline_returns = final_close / first_open - 1

    model_group_means = [
        model_returns[i * SIMS_PER_GROUP:(i + 1) * SIMS_PER_GROUP].mean()
        for i in range(TEST_GROUPS)
    ]
    baseline_group_means = [
        baseline_returns[i * SIMS_PER_GROUP:(i + 1) * SIMS_PER_GROUP].mean()
        for i in range(TEST_GROUPS)
    ]
    return model_group_means, baseline_group_means


def create_training_plot(history, model_returns, baseline_returns, filepath):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].plot([h['policy_loss'] for h in history], 'b-', alpha=0.7)
    axes[0, 0].set(xlabel='Update', ylabel='Loss', title='Policy Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot([h['value_loss'] for h in history], 'r-', alpha=0.7)
    axes[0, 1].set(xlabel='Update', ylabel='Loss', title='Value Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot([h['entropy'] for h in history], 'm-', alpha=0.7)
    axes[0, 2].set(xlabel='Update', ylabel='Entropy', title='Entropy')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot([h['mean_return'] for h in history], 'g-', alpha=0.7, label='Model')
    axes[1, 0].plot([h['mean_baseline'] for h in history], 'orange', alpha=0.7, label='Baseline')
    axes[1, 0].axhline(0, color='k', ls='--', alpha=0.3)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set(xlabel='Update', ylabel='Return', title='Training')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(model_returns, bins=50, alpha=0.6, label='Model', color='blue')
    axes[1, 1].hist(baseline_returns, bins=50, alpha=0.6, label='Baseline', color='orange')
    axes[1, 1].axvline(np.mean(model_returns), color='blue', ls='--', lw=2,
                       label=f'Model μ={np.mean(model_returns):.3f}')
    axes[1, 1].axvline(np.mean(baseline_returns), color='orange', ls='--', lw=2,
                       label=f'Base μ={np.mean(baseline_returns):.3f}')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set(xlabel='Return', ylabel='Freq', title='Test')
    axes[1, 1].grid(True, alpha=0.3)

    min_val = min(min(baseline_returns), min(model_returns))
    max_val = max(max(baseline_returns), max(model_returns))
    axes[1, 2].scatter(baseline_returns, model_returns, alpha=0.3, s=10, color='green')
    axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, lw=2, label='y=x')
    axes[1, 2].legend()
    axes[1, 2].set(xlabel='Baseline', ylabel='Model', title='Comparison')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def training_worker(rank, world_size, train_stratified_pools, test_pool, total_updates, num_features, data):
    global _GLOBAL_DATA
    _GLOBAL_DATA = data

    set_seed(SEED + rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = "gloo" if world_size == 1 else "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    logger = None
    if rank == 0:
        logger = logging.getLogger('training')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        handler = logging.FileHandler(os.path.join(OUTPUT_DIR, 'training.log'), mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

    model = TradingPolicyNetwork(num_features).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_updates, LEARNING_RATE / 10)

    patience_limit = int(total_updates * PATIENCE_FRAC) + 1
    best_validation_return = float('-inf')
    best_model_state = None
    updates_without_improvement = 0

    if rank == 0:
        num_params = sum(p.numel() for p in get_base_model(model).parameters() if p.requires_grad)
        print(f"Model: {num_params:,} params, {world_size} GPUs")
        print(f"Training: {total_updates} max updates, patience={patience_limit}")
        header = (f"{'update':>11} | {'pol_loss':>8} | {'val_loss':>8} | {'entropy':>8} | "
                  f"{'ent_coef':>8} | {'pol_clip':>8} | {'return':>8} | {'baseline':>8}")
        print(header)
        if logger:
            logger.info(header)

    training_history = []
    start_time = time.time()
    final_update_count = total_updates
    stopped_early = False

    for update_idx in range(total_updates):
        sample_rng = random.Random(update_idx * world_size + rank)
        samples = sample_stratified_segments(train_stratified_pools, EPISODES_PER_UPDATE, sample_rng)

        sim_results = simulate_trading(samples, model, device, is_training=True)
        advantages, value_targets = compute_advantages_and_targets(sim_results, model, device)
        batch = prepare_training_batch(sim_results, advantages, value_targets, device)

        entropy_coef = compute_entropy_coefficient(update_idx, total_updates)
        metrics = execute_ppo_update(model, optimizer, batch, entropy_coef)
        scheduler.step()

        mean_return = sim_results['final_portfolio'].mean() / INIT_CASH - 1
        mean_baseline = (sim_results['final_close'] / sim_results['first_open'] - 1).mean()

        sync_tensor = torch.tensor([
            metrics['policy_loss'], metrics['value_loss'], metrics['entropy'],
            metrics['clip_frac'], mean_return, mean_baseline
        ], device=device)
        if world_size > 1:
            dist.all_reduce(sync_tensor, op=dist.ReduceOp.AVG)
        pol_loss, val_loss, ent, clip_frac, mean_ret, mean_base = sync_tensor.tolist()

        metrics = {
            'policy_loss': pol_loss, 'value_loss': val_loss, 'entropy': ent,
            'clip_frac': clip_frac, 'mean_return': mean_ret, 'mean_baseline': mean_base
        }

        if rank == 0:
            training_history.append(metrics)
            line = (f"{update_idx + 1:>5}/{total_updates:<5} | {pol_loss:>8.3f} | {val_loss:>8.3f} | "
                    f"{ent:>8.3f} | {entropy_coef:>8.3f} | {clip_frac:>8.3f} | "
                    f"{mean_ret:>8.3f} | {mean_base:>8.3f}")
            if logger:
                logger.info(line)
            sys.stdout.write(f"\r{line}")
            sys.stdout.flush()

        if (update_idx + 1) >= MIN_UPDATES and (update_idx + 1) % EVAL_INTERVAL == 0:
            val_start = time.time()
            val_model_ret, val_base_ret = evaluate_model(test_pool, model, device, rank, world_size)
            val_mean = np.mean(val_model_ret)

            if rank == 0:
                base_model = get_base_model(model)
                torch.save(base_model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{update_idx + 1}.pt'))
                create_training_plot(training_history, val_model_ret, val_base_ret,
                                     os.path.join(OUTPUT_DIR, f'plot_{update_idx + 1}.png'))
                print(f"\n  [Val] mean={val_mean:.4f}, std={np.std(val_model_ret):.4f}, "
                      f"base={np.mean(val_base_ret):.4f} [{time.time() - val_start:.1f}s]")
                if logger:
                    logger.info(f"  [Val] mean={val_mean:.4f}")

            if val_mean > best_validation_return:
                best_validation_return = val_mean
                updates_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in get_base_model(model).state_dict().items()}
                if rank == 0:
                    torch.save(best_model_state, os.path.join(OUTPUT_DIR, 'model_best.pt'))
                    create_training_plot(training_history, val_model_ret, val_base_ret,
                                         os.path.join(OUTPUT_DIR, 'plot_best.png'))
                    print(f"  [Val] New best: {val_mean:.4f}")
                    if logger:
                        logger.info(f"  [Val] New best: {val_mean:.4f}")
            else:
                updates_without_improvement += EVAL_INTERVAL
                if rank == 0:
                    print(f"  [Val] No improvement, patience: {patience_limit - updates_without_improvement}")
                    if logger:
                        logger.info(f"  [Val] patience: {patience_limit - updates_without_improvement}")

            if updates_without_improvement >= patience_limit:
                if rank == 0:
                    print(f"\n  Early stop at update {update_idx + 1}")
                    if logger:
                        logger.info(f"  Early stop at update {update_idx + 1}")
                final_update_count = update_idx + 1
                stopped_early = True
                break

    if rank == 0:
        elapsed = time.time() - start_time
        status = 'Early stopped' if stopped_early else 'Completed'
        print(f"\n\n{status} at update {final_update_count} [{elapsed:.1f}s]")

        base_model = get_base_model(model)
        if best_model_state:
            base_model.load_state_dict(best_model_state)
        torch.save(base_model.state_dict(), os.path.join(OUTPUT_DIR, 'model_final.pt'))
        print("Models saved")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print("\nTesting")
    test_start = time.time()
    test_model_ret, test_base_ret = evaluate_model(test_pool, model, device, rank, world_size)

    if rank == 0:
        print(f"  Model:    mean={np.mean(test_model_ret):.4f}, std={np.std(test_model_ret):.4f}")
        print(f"  Baseline: mean={np.mean(test_base_ret):.4f}, std={np.std(test_base_ret):.4f}")
        print(f"  [{time.time() - test_start:.1f}s]")
        create_training_plot(training_history, test_model_ret, test_base_ret,
                             os.path.join(OUTPUT_DIR, 'plot_final.png'))
        print("Plots saved")

    dist.destroy_process_group()


def main():
    global _GLOBAL_DATA

    set_seed(SEED)

    print("\nStock Trading with PPO")
    print(f"System: {NUM_CPUS} CPUs, {NUM_GPUS} GPUs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading data")
    t0 = time.time()
    dataframe = pd.read_csv(DATA_PATH)
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    print(f"  {len(dataframe):,} rows, {dataframe['symbol'].nunique():,} symbols [{time.time() - t0:.1f}s]")

    feature_columns = [c for c in dataframe.columns if c not in ['symbol', 'date', 'open', 'close']]
    print(f"  Features: {len(feature_columns)}")

    print("\nFiltering symbols")
    t0 = time.time()
    symbol_counts = dataframe.groupby('symbol').size()
    valid_symbols = symbol_counts[symbol_counts >= REQ_LEN].index.tolist()
    print(f"  Valid: {len(valid_symbols)} (excluded {len(symbol_counts) - len(valid_symbols)}) [{time.time() - t0:.1f}s]")

    print("\nSplitting train/test")
    t0 = time.time()
    split_rng = random.Random(SEED)
    split_rng.shuffle(valid_symbols)
    num_train = int(len(valid_symbols) * TRAIN_SPLIT)
    train_symbols, test_symbols = valid_symbols[:num_train], valid_symbols[num_train:]
    print(f"  Train: {len(train_symbols)}, Test: {len(test_symbols)} [{time.time() - t0:.1f}s]")

    print("\nInitializing data")
    t0 = time.time()
    initialize_global_data(dataframe[dataframe['symbol'].isin(valid_symbols)], feature_columns)
    shared_data = _GLOBAL_DATA.copy()
    print(f"  [{time.time() - t0:.1f}s]")

    print("\nBuilding pools")
    t0 = time.time()
    train_pool = build_segment_pool(train_symbols)
    test_pool = build_segment_pool(test_symbols)
    train_stratified_pools = build_symbol_stratified_pools(train_pool)
    print(f"  Train: {len(train_pool):,}, Test: {len(test_pool):,} [{time.time() - t0:.1f}s]")

    avg_segments_per_symbol = len(train_pool) / max(len(train_symbols), 1)
    total_updates = math.ceil(math.log(1 - MAX_COVERAGE) / math.log(1 - 1 / max(avg_segments_per_symbol, 1)))
    print(f"  Avg segments/symbol: {avg_segments_per_symbol:.1f}")
    print(f"  Total updates: {total_updates}")

    print(f"\nLaunching {NUM_GPUS} processes")
    if NUM_GPUS == 1:
        training_worker(0, 1, train_stratified_pools, test_pool, total_updates, len(feature_columns), shared_data)
    else:
        torch.multiprocessing.spawn(
            training_worker,
            args=(NUM_GPUS, train_stratified_pools, test_pool, total_updates, len(feature_columns), shared_data),
            nprocs=NUM_GPUS,
            join=True
        )
    print("\nDone")


if __name__ == "__main__":
    main()
