"""
Stock Trading with PPO and Beta Policy
Multi-GPU distributed training for portfolio optimization
"""

import os
import sys
import time
import copy
import random
import math
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
import logging
import warnings
warnings.filterwarnings('ignore')


# Paths and directories
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
OUTPUT_DIR = "models/"

# Simulation parameters
SIM_LEN = 240
SEQ_LEN = 60
REQ_LEN = SIM_LEN + SEQ_LEN
INIT_CASH = 100.0

# Training hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 4
LR = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 0.5
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
TAU = 0.005

EPISODES_PER_UPDATE = 1024
BATCH_SIZE = 4096
NUM_EPOCHS = 10
TEST_GROUPS = 1000
SIMS_PER_GROUP = 30

# System resources
EPS = float(np.finfo(np.float32).eps)
NUM_CPUS = max(1, cpu_count() - 1)

# Global data storage
_DATA = {}


# Model definition
class PolicyModel(nn.Module):
    """LSTM-based policy with Beta distribution output for continuous [0,1] targets."""

    def __init__(self, n_features, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.state_fc = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        combined = hidden + 32
        self.policy = nn.Sequential(nn.Linear(combined, 64), nn.ReLU(), nn.Linear(64, 2))
        self.value = nn.Sequential(nn.Linear(combined, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value_target = copy.deepcopy(self.value)
        for p in self.value_target.parameters():
            p.requires_grad = False
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _encode(self, seq, info):
        out, _ = self.lstm(seq)
        return torch.cat([out[:, -1], self.state_fc(info)], dim=1)

    def forward(self, seq, info):
        enc = self._encode(seq, info)
        raw = self.policy(enc)
        alpha = F.softplus(raw[:, 0]) + 1.0
        beta = F.softplus(raw[:, 1]) + 1.0
        value = self.value(enc).squeeze(-1)
        return alpha, beta, value

    def get_bootstrap_value(self, seq, info):
        return self.value_target(self._encode(seq, info)).squeeze(-1)

    def update_target(self, tau=TAU):
        for p, tp in zip(self.value.parameters(), self.value_target.parameters()):
            tp.data.lerp_(p.data, tau)


# Data preparation functions
def _build_pool_worker(args):
    """CPU worker to build sample pool for a chunk of symbols."""
    symbols, ranges, req_len = args
    samples = []
    for sym in symbols:
        start, end = ranges[sym]
        n_rows = end - start
        if n_rows >= req_len:
            for idx_end in range(req_len - 1, n_rows):
                idx_start = idx_end - req_len + 1
                samples.append((sym, idx_start, idx_end))
    return samples


def init_data(df, feat_cols, price_col_idx):
    """Initialize global data structure for fast episode extraction."""
    global _DATA

    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    symbols = df['symbol'].values

    # Build symbol index ranges
    ranges = {}
    start = 0
    prev = symbols[0]
    for i in range(1, len(symbols)):
        if symbols[i] != prev:
            ranges[prev] = (start, i)
            prev, start = symbols[i], i
    ranges[prev] = (start, len(symbols))

    _DATA = {
        'features': df[feat_cols].values.astype(np.float32),
        'prices': df[['p_open', 'p_close']].values.astype(np.float32),
        'ranges': ranges,
        'price_idx': np.array(price_col_idx, dtype=np.int64),
    }


def build_pool():
    """Build sample pool using CPU parallelization."""
    ranges = _DATA['ranges']
    symbols = list(ranges.keys())

    n_chunks = min(NUM_CPUS * 4, len(symbols))
    chunk_sz = max(1, len(symbols) // n_chunks + 1)
    args = [(symbols[i:i+chunk_sz], ranges, REQ_LEN) for i in range(0, len(symbols), chunk_sz)]

    pool = []
    with ProcessPoolExecutor(max_workers=NUM_CPUS) as ex:
        for chunk in ex.map(_build_pool_worker, args):
            pool.extend(chunk)

    return pool, len(ranges)


def extract_episodes(samples):
    """Extract episode data from samples with vectorized operations."""
    features = _DATA['features']
    prices = _DATA['prices']
    ranges = _DATA['ranges']
    price_idx = _DATA['price_idx']

    n = len(samples)
    global_starts = np.empty(n, dtype=np.int64)
    for i, (sym, s_start, _) in enumerate(samples):
        global_starts[i] = ranges[sym][0] + s_start

    # Vectorized extraction
    row_idx = np.arange(REQ_LEN)[None, :] + global_starts[:, None]
    feat_batch = features[row_idx]
    price_batch = prices[row_idx]

    # Scale price features by p_close at index_0
    ref_price = price_batch[:, SEQ_LEN - 1, 1]  # p_close at day 0
    ref_price = np.maximum(ref_price, EPS)[:, None, None]
    feat_batch[:, :, price_idx] /= ref_price

    return feat_batch, price_batch


# Trade simulation functions
def compute_trades(targets, cash, holding, open_price):
    """
    Compute trade quantities from target positions.
    Formula: trade = floor(clip(target * (cash/open + holding) - holding, -holding, cash/open))
    """
    safe_price = np.maximum(open_price, EPS)
    max_position = cash / safe_price + holding
    desired = targets * max_position
    delta = desired - holding
    max_buy = cash / safe_price
    return np.floor(np.clip(delta, -holding, max_buy)).astype(np.float32)


def simulate_episodes(samples, model, device, training=True, batch_sz=4096):
    """Run trading simulation with batched GPU inference."""

    # Extract episode data
    feat_batch, price_batch = extract_episodes(samples)
    n_ep = len(samples)
    price_idx = _DATA['price_idx']

    # Initialize trading state
    cash = np.full(n_ep, INIT_CASH, np.float32)
    holding = np.zeros(n_ep, np.float32)

    # Preallocate result arrays
    states = np.zeros((n_ep, SIM_LEN, SEQ_LEN, feat_batch.shape[-1]), np.float32)
    info = np.zeros((n_ep, SIM_LEN, 3), np.float32)
    targets = np.zeros((n_ep, SIM_LEN), np.float32)
    logprobs = np.zeros((n_ep, SIM_LEN), np.float32)
    values = np.zeros((n_ep, SIM_LEN), np.float32)
    entropy = np.zeros((n_ep, SIM_LEN), np.float32)
    portfolios = np.zeros((n_ep, SIM_LEN + 1), np.float32)

    model.eval()

    with torch.no_grad():
        for step in range(SIM_LEN):
            day = SEQ_LEN - 1 + step
            close = price_batch[:, day, 1]
            portfolio = cash + holding * close
            portfolios[:, step] = portfolio

            # Prepare model input
            # Sequence: price features are already scaled, now center by -1
            seq = feat_batch[:, step:step + SEQ_LEN].copy()
            seq[:, :, price_idx] -= 1.0  # Center price features

            # Info state: [cash, holding, portfolio] scaled by cash_0 and centered by -1
            info_vec = np.stack([
                cash / INIT_CASH - 1.0,
                holding / INIT_CASH - 1.0,
                portfolio / INIT_CASH - 1.0
            ], axis=1)

            states[:, step] = seq
            info[:, step] = info_vec

            # Batched inference
            n_batches = math.ceil(n_ep / batch_sz)
            step_targets = np.zeros(n_ep, np.float32)
            step_logprobs = np.zeros(n_ep, np.float32)
            step_values = np.zeros(n_ep, np.float32)
            step_entropy = np.zeros(n_ep, np.float32)

            for b in range(n_batches):
                start_idx = b * batch_sz
                end_idx = min(start_idx + batch_sz, n_ep)

                seq_t = torch.from_numpy(seq[start_idx:end_idx]).to(device)
                info_t = torch.from_numpy(info_vec[start_idx:end_idx]).to(device)

                alpha, beta, value = model(seq_t, info_t)

                if training:
                    dist = Beta(alpha, beta)
                    target = dist.sample()
                    logprob = dist.log_prob(target)
                    ent = dist.entropy()
                else:
                    target = alpha / (alpha + beta)  # Mean of Beta
                    logprob = torch.zeros_like(target)
                    ent = torch.zeros_like(target)

                step_targets[start_idx:end_idx] = target.cpu().numpy()
                step_logprobs[start_idx:end_idx] = logprob.cpu().numpy()
                step_values[start_idx:end_idx] = value.cpu().numpy()
                step_entropy[start_idx:end_idx] = ent.cpu().numpy()

            targets[:, step] = step_targets
            logprobs[:, step] = step_logprobs
            values[:, step] = step_values
            entropy[:, step] = step_entropy

            # Execute trades
            if step < SIM_LEN - 1:
                next_open = price_batch[:, day + 1, 0]
                qty = compute_trades(step_targets, cash, holding, next_open)
                cost = qty * next_open
                cash -= cost
                holding += qty

        # Final portfolio value
        final_close = price_batch[:, SEQ_LEN - 1 + SIM_LEN, 1]
        portfolios[:, SIM_LEN] = cash + holding * final_close

        # Compute rewards
        portfolio_changes = np.diff(portfolios, axis=1)
        rewards = portfolio_changes / np.maximum(portfolios[:, :-1], EPS)

    return {
        'states': states,
        'info': info,
        'targets': targets,
        'logprobs': logprobs,
        'values': values,
        'entropy': entropy,
        'rewards': rewards,
        'final_portfolio': portfolios[:, -1],
        'first_open': price_batch[:, SEQ_LEN - 1, 0],
        'final_close': final_close,
    }


# GAE computation (GPU-accelerated)
def compute_gae(results, model, device, batch_sz=4096):
    """Compute Generalized Advantage Estimation on GPU."""

    with torch.no_grad():
        values = torch.from_numpy(results['values']).to(device)
        rewards = torch.from_numpy(results['rewards']).to(device)
        n_ep, n_steps = values.shape

        # Compute bootstrap values for final states
        all_next_values = []
        for i in range(0, n_ep, batch_sz):
            end_idx = min(i + batch_sz, n_ep)
            states_t = torch.from_numpy(results['states'][i:end_idx, -1]).to(device)
            info_t = torch.from_numpy(results['info'][i:end_idx, -1]).to(device)
            next_v = model.module.get_bootstrap_value(states_t, info_t)
            all_next_values.append(next_v)
        next_values = torch.cat(all_next_values, dim=0)

        # Compute GAE
        advantages = torch.zeros_like(values)
        gae = torch.zeros(n_ep, device=device)

        for t in reversed(range(n_steps)):
            next_v = next_values if t == n_steps - 1 else values[:, t + 1]
            delta = rewards[:, t] + GAMMA * next_v - values[:, t]
            gae = delta + GAMMA * GAE_LAMBDA * gae
            advantages[:, t] = gae

        value_targets = advantages + values

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

    return advantages, value_targets


def prepare_training_batch(results, advantages, value_targets, device):
    """Flatten and prepare training tensors on GPU."""

    n_ep, n_steps = results['states'].shape[:2]

    states = torch.from_numpy(results['states'].reshape(-1, SEQ_LEN, results['states'].shape[-1])).to(device)
    info = torch.from_numpy(results['info'].reshape(-1, 3)).to(device)
    targets = torch.from_numpy(results['targets'].reshape(-1)).to(device)
    old_logprobs = torch.from_numpy(results['logprobs'].reshape(-1)).to(device)
    advantages_flat = advantages.reshape(-1)
    value_targets_flat = value_targets.reshape(-1)

    return {
        'states': states,
        'info': info,
        'targets': targets,
        'old_logprobs': old_logprobs,
        'advantages': advantages_flat,
        'value_targets': value_targets_flat,
    }


# PPO training
def train_step(model, optimizer, batch, batch_sz):
    """Perform PPO update with mini-batch training."""

    model.train()
    device = next(model.parameters()).device
    n_samples = len(batch['states'])

    metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'clip_frac': 0.0}
    n_batches = 0

    for epoch in range(PPO_EPOCHS):
        indices = torch.randperm(n_samples, device=device)

        for i in range(0, n_samples, batch_sz):
            idx = indices[i:i + batch_sz]

            states = batch['states'][idx]
            info = batch['info'][idx]
            targets = batch['targets'][idx]
            old_logprobs = batch['old_logprobs'][idx]
            advantages = batch['advantages'][idx]
            value_targets = batch['value_targets'][idx]

            # Forward pass
            alpha, beta, values = model(states, info)
            dist = Beta(alpha, beta)
            logprobs = dist.log_prob(targets)
            ent = dist.entropy().mean()

            # Policy loss (PPO clipped objective)
            ratio = torch.exp(logprobs - old_logprobs)
            clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # Value loss
            value_loss = F.mse_loss(values, value_targets)

            # Total loss
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * ent

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # Track metrics
            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > CLIP_EPS).float().mean()
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += ent.item()
                metrics['clip_frac'] += clip_frac.item()
                n_batches += 1

    # Average metrics
    for k in metrics:
        metrics[k] /= max(n_batches, 1)

    # Update target network
    base = model.module if isinstance(model, DDP) else model
    base.update_target()

    return metrics


# Testing functions
def _sample_test_groups(args):
    """CPU worker to sample test groups."""
    pool, n_sims, seeds = args
    groups = []
    for seed in seeds:
        rng = random.Random(seed)
        groups.append(rng.sample(pool, k=n_sims))
    return groups


def evaluate_model(pool, model, device, batch_sz, rank, world_size):
    """Evaluate model on test data with multi-GPU parallelization."""

    # Sample test groups (CPU parallelized, same across all GPUs)
    seeds = [random.randint(0, 2**31) for _ in range(TEST_GROUPS)]
    chunk_sz = max(1, TEST_GROUPS // NUM_CPUS + 1)
    args = [(pool, SIMS_PER_GROUP, seeds[i:i+chunk_sz]) for i in range(0, TEST_GROUPS, chunk_sz)]

    all_samples = []
    with ProcessPoolExecutor(max_workers=NUM_CPUS) as ex:
        for groups in ex.map(_sample_test_groups, args):
            for g in groups:
                all_samples.extend(g)

    # Split samples across GPUs for parallel simulation
    local_samples = all_samples[rank::world_size]

    # Run simulations on this GPU's subset
    results = simulate_episodes(local_samples, model, device, training=False, batch_sz=batch_sz)

    # Gather results from all GPUs
    if world_size > 1:
        # Convert results to tensors for gathering
        final_portfolio = torch.from_numpy(results['final_portfolio']).to(device)
        first_open = torch.from_numpy(results['first_open']).to(device)
        final_close = torch.from_numpy(results['final_close']).to(device)

        # Gather from all GPUs
        gathered_portfolio = [torch.zeros_like(final_portfolio) for _ in range(world_size)]
        gathered_open = [torch.zeros_like(first_open) for _ in range(world_size)]
        gathered_close = [torch.zeros_like(final_close) for _ in range(world_size)]

        dist.all_gather(gathered_portfolio, final_portfolio)
        dist.all_gather(gathered_open, first_open)
        dist.all_gather(gathered_close, final_close)

        # Reconstruct full results on all ranks (for consistency)
        full_portfolio = torch.cat(gathered_portfolio).cpu().numpy()
        full_open = torch.cat(gathered_open).cpu().numpy()
        full_close = torch.cat(gathered_close).cpu().numpy()
    else:
        full_portfolio = results['final_portfolio']
        full_open = results['first_open']
        full_close = results['final_close']

    # Compute returns
    model_ret = (full_portfolio - INIT_CASH) / INIT_CASH
    base_ret = full_close / np.maximum(full_open, EPS) - 1

    # Aggregate by group
    model_means = [model_ret[g * SIMS_PER_GROUP:(g + 1) * SIMS_PER_GROUP].mean() for g in range(TEST_GROUPS)]
    base_means = [base_ret[g * SIMS_PER_GROUP:(g + 1) * SIMS_PER_GROUP].mean() for g in range(TEST_GROUPS)]

    return model_means, base_means


# Plotting
def create_plots(history, model_ret, base_ret, out_dir):
    """Generate training and testing result plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # First row: Training metrics
    axes[0, 0].plot([h['policy_loss'] for h in history], 'b-', alpha=0.7, linewidth=1.5)
    axes[0, 0].set(xlabel='Update', ylabel='Loss', title='Policy Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot([h['value_loss'] for h in history], 'r-', alpha=0.7, linewidth=1.5)
    axes[0, 1].set(xlabel='Update', ylabel='Loss', title='Value Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot([h['entropy'] for h in history], 'm-', alpha=0.7, linewidth=1.5)
    axes[0, 2].set(xlabel='Update', ylabel='Entropy', title='Policy Entropy')
    axes[0, 2].grid(True, alpha=0.3)

    # Second row: Returns and comparisons
    axes[1, 0].plot([h['mean_return'] for h in history], 'g-', alpha=0.7, linewidth=1.5)
    axes[1, 0].axhline(0, color='k', ls='--', alpha=0.3)
    axes[1, 0].set(xlabel='Update', ylabel='Return', title='Training Returns')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(model_ret, bins=50, alpha=0.6, label='Model', color='blue')
    axes[1, 1].hist(base_ret, bins=50, alpha=0.6, label='Baseline', color='orange')
    axes[1, 1].axvline(np.mean(model_ret), color='blue', ls='--', lw=2, label=f'Model μ={np.mean(model_ret):.3f}')
    axes[1, 1].axvline(np.mean(base_ret), color='orange', ls='--', lw=2, label=f'Base μ={np.mean(base_ret):.3f}')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set(xlabel='Return', ylabel='Frequency', title='Test Returns Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    min_v = min(min(base_ret), min(model_ret))
    max_v = max(max(base_ret), max(model_ret))
    axes[1, 2].scatter(base_ret, model_ret, alpha=0.3, s=10, color='green')
    axes[1, 2].plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.5, lw=2, label='y=x')
    axes[1, 2].legend()
    axes[1, 2].set(xlabel='Baseline Return', ylabel='Model Return', title='Model vs Baseline')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_results.png'), dpi=150, bbox_inches='tight')
    plt.close()


# Distributed training
def setup_distributed(rank, world_size):
    """Initialize distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed resources."""
    dist.destroy_process_group()


def train_worker(rank, world_size, train_pool, test_pool, n_features, train_data):
    """Main training loop for each GPU."""
    global _DATA

    _DATA = train_data
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Setup logging (rank 0 only)
    logger = None
    if rank == 0:
        logger = logging.getLogger('training')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(OUTPUT_DIR, 'training.log'), mode='w')
        fh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)

    # Create model
    model = PolicyModel(n_features).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized: {n_params:,} parameters, {world_size} GPUs")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Calculate actual updates per epoch (accounting for skipped small batches)
    local_pool_size = len(train_pool) // world_size
    updates_per_epoch = local_pool_size // EPISODES_PER_UPDATE  # Floor division (skips partial batch)
    total_updates = NUM_EPOCHS * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_updates, LR / 10)

    # Training loop
    if rank == 0:
        samples_per_gpu = len(train_pool) // world_size
        samples_used = updates_per_epoch * EPISODES_PER_UPDATE
        samples_skipped = samples_per_gpu - samples_used

        header = f"{'Epoch':>5} | {'Update':>10} | {'PolLoss':>7} | {'ValLoss':>7} | {'Entropy':>7} | {'Clip':>6} | {'Return':>7}"
        print(f"\nTraining: {updates_per_epoch} updates/epoch, {total_updates} total")
        print(f"  Per GPU: {samples_used:,}/{samples_per_gpu:,} samples used ({samples_skipped:,} skipped in final batch)")
        print(header)
        if logger:
            logger.info(header)

    history = []
    update_n = 0
    t_start = time.time()

    try:
        for epoch in range(NUM_EPOCHS):
            # Each GPU shuffles differently
            epoch_pool = train_pool.copy()
            random.seed(epoch * world_size + rank)
            random.shuffle(epoch_pool)
            local_pool = epoch_pool[rank::world_size]

            for i in range(0, len(local_pool), EPISODES_PER_UPDATE):
                samples = local_pool[i:i + EPISODES_PER_UPDATE]

                # Skip last batch if it's too small (causes instability)
                if len(samples) < EPISODES_PER_UPDATE // 2:
                    if rank == 0:
                        print(f"\n  Skipping final batch of {len(samples)} samples (< {EPISODES_PER_UPDATE//2} threshold)")
                    break

                # Simulate episodes
                results = simulate_episodes(samples, model, device, training=True, batch_sz=BATCH_SIZE)

                # Compute GAE
                advantages, value_targets = compute_gae(results, model, device, BATCH_SIZE)
                batch = prepare_training_batch(results, advantages, value_targets, device)

                # Training step
                metrics = train_step(model, optimizer, batch, BATCH_SIZE)
                scheduler.step()

                # Compute mean return
                mean_return = (results['final_portfolio'].mean() - INIT_CASH) / INIT_CASH

                # Synchronize metrics across GPUs
                if world_size > 1:
                    metrics_tensor = torch.tensor([
                        metrics['policy_loss'],
                        metrics['value_loss'],
                        metrics['entropy'],
                        metrics['clip_frac'],
                        mean_return
                    ], device=device)
                    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
                    metrics['policy_loss'] = metrics_tensor[0].item()
                    metrics['value_loss'] = metrics_tensor[1].item()
                    metrics['entropy'] = metrics_tensor[2].item()
                    metrics['clip_frac'] = metrics_tensor[3].item()
                    mean_return = metrics_tensor[4].item()

                metrics['mean_return'] = mean_return

                # Log results (rank 0 only)
                if rank == 0:
                    history.append(metrics)
                    log_line = (f"{epoch+1:>5} | {update_n+1:>4}/{total_updates:<5} | "
                               f"{metrics['policy_loss']:>7.4f} | {metrics['value_loss']:>7.4f} | "
                               f"{metrics['entropy']:>7.4f} | {metrics['clip_frac']:>6.3f} | "
                               f"{mean_return:>7.4f}")
                    if logger:
                        logger.info(log_line)
                    sys.stdout.write(f"\r{log_line}")
                    sys.stdout.flush()

                update_n += 1

            if rank == 0:
                print()  # New line after epoch

        t_elapsed = time.time() - t_start
        if rank == 0:
            print(f"\nTraining completed [{t_elapsed:.1f}s]")

        # Save model (rank 0 only)
        if rank == 0:
            base = model.module if isinstance(model, DDP) else model
            torch.save(base.state_dict(), os.path.join(OUTPUT_DIR, 'trading_model.pt'))
            print(f"Model saved: {OUTPUT_DIR}trading_model.pt")

        # Synchronize before testing
        if world_size > 1:
            dist.barrier()

        # Testing (all GPUs participate)
        if rank == 0:
            print(f"\nTesting")

        _DATA = test_pool['data']

        t_test = time.time()
        model_ret, base_ret = evaluate_model(test_pool['pool'], model, device, BATCH_SIZE, rank, world_size)
        t_test = time.time() - t_test

        # Only rank 0 prints and plots
        if rank == 0:
            print(f"  Model:    mean={np.mean(model_ret):.4f}, std={np.std(model_ret):.4f}")
            print(f"  Baseline: mean={np.mean(base_ret):.4f}, std={np.std(base_ret):.4f}")
            print(f"Testing completed [{t_test:.1f}s]")

            # Plot results
            t_plot = time.time()
            create_plots(history, model_ret, base_ret, OUTPUT_DIR)
            print(f"Plots saved: {OUTPUT_DIR}training_results.png [{time.time()-t_plot:.1f}s]")

    finally:
        cleanup_distributed()


# Main entry point
def main():
    """Main training script."""
    global _DATA

    print("\nStock Trading with PPO and Beta Policy")
    print(f"System: {NUM_CPUS} CPUs, {torch.cuda.device_count()} GPUs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\nLoading data")
    t0 = time.time()
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['date'] = pd.to_datetime(train_df['date'])
    print(f"  Train: {len(train_df):,} rows, {train_df['symbol'].nunique():,} symbols [{time.time()-t0:.1f}s]")

    t0 = time.time()
    test_df = pd.read_csv(TEST_PATH)
    test_df['date'] = pd.to_datetime(test_df['date'])
    print(f"  Test: {len(test_df):,} rows, {test_df['symbol'].nunique():,} symbols [{time.time()-t0:.1f}s]")

    # Prepare feature columns
    feat_cols = [c for c in train_df.columns if c not in ['symbol', 'date']]
    price_cols = [c for c in feat_cols if c.startswith('p_')]
    price_col_idx = [feat_cols.index(c) for c in price_cols]
    print(f"  Features: {len(feat_cols)} ({len(price_cols)} price features)")

    # Build sample pools
    print("\nBuilding sample pools")
    t0 = time.time()
    init_data(train_df, feat_cols, price_col_idx)
    train_pool, train_n_sym = build_pool()
    print(f"  Train: {len(train_pool):,} samples, {train_n_sym:,} symbols [{time.time()-t0:.1f}s]")
    train_data = _DATA.copy()

    t0 = time.time()
    init_data(test_df, feat_cols, price_col_idx)
    test_pool_data, test_n_sym = build_pool()
    print(f"  Test: {len(test_pool_data):,} samples, {test_n_sym:,} symbols [{time.time()-t0:.1f}s]")
    test_data = _DATA.copy()

    _DATA = train_data

    if not train_pool:
        print("ERROR: No valid training samples")
        return

    test_pool = {'pool': test_pool_data, 'data': test_data}

    # Launch distributed training
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("ERROR: No CUDA GPUs available")
        return

    print(f"\nLaunching {world_size} training processes")
    torch.multiprocessing.spawn(
        train_worker,
        args=(world_size, train_pool, test_pool, len(feat_cols), train_data),
        nprocs=world_size,
        join=True
    )

    print("\nTraining pipeline completed")


if __name__ == "__main__":
    main()
