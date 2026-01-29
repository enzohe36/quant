"""
LSTM + DQN Trading Model
Predicts: action (1) vs wait (0)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple
import time

# =============================================================================
# PARAMETERS
# =============================================================================

SEQUENCE_LENGTH = 60
NUM_EPISODES = 100
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 5
TRAIN_FREQ = 4

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
MODEL_PATH = 'model.pth'

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# MODEL
# =============================================================================

class LSTMDQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=0.2 if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        return self.fc3(out)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv(path: str) -> Tuple[Dict, int]:
    """Load CSV and return data grouped by symbol"""
    df = pd.read_csv(path)
    
    exclude = {'symbol', 'date', 'reward'}
    feature_cols = [c for c in df.columns if c not in exclude]
    
    data = {}
    for symbol, group in df.groupby('symbol'):
        group = group.sort_values('date') if 'date' in df.columns else group
        data[symbol] = {
            'features': group[feature_cols].values.astype(np.float32),
            'rewards': group['reward'].values.astype(np.float32)
        }
    
    return data, len(feature_cols)


def _create_sequences(args) -> Dict:
    """Worker function for parallel sequence creation"""
    symbol, features, rewards, seq_len = args
    n = len(features)
    if n < seq_len + 1:
        return None
    
    n_seq = n - seq_len
    seqs = np.zeros((n_seq, seq_len, features.shape[1]), dtype=np.float32)
    rews = np.zeros(n_seq, dtype=np.float32)
    
    for i in range(n_seq):
        seqs[i] = features[i:i + seq_len]
        rews[i] = rewards[i + seq_len]
    
    return {'sequences': seqs, 'rewards': rews}


def preprocess(data: Dict, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Convert raw data to GPU tensors with parallel sequence creation"""
    args = [(sym, d['features'], d['rewards'], seq_len) for sym, d in data.items()]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = [r for r in ex.map(_create_sequences, args) if r]
    
    total = sum(len(r['sequences']) for r in results)
    n_feat = results[0]['sequences'].shape[2]
    
    all_seq = np.zeros((total, seq_len, n_feat), dtype=np.float32)
    all_rew = np.zeros(total, dtype=np.float32)
    all_done = np.zeros(total, dtype=np.float32)
    
    indices = []
    pos = 0
    for r in results:
        n = len(r['sequences'])
        all_seq[pos:pos+n] = r['sequences']
        all_rew[pos:pos+n] = r['rewards']
        indices.extend(range(pos, pos + n - 1))
        if n >= 2:
            all_done[pos + n - 2] = 1.0
        pos += n
    
    return (
        torch.tensor(all_seq, device=device),
        torch.tensor(all_rew, device=device),
        torch.tensor(all_done, device=device),
        torch.tensor(indices, dtype=torch.long, device=device)
    )


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.idx = torch.zeros(capacity, dtype=torch.long, device=device)
        self.act = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rew = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.done = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.pos = 0
        self.size = 0

    def push(self, idx, act, rew, done):
        n = len(idx)
        end = self.pos + n
        if end <= self.capacity:
            self.idx[self.pos:end] = idx
            self.act[self.pos:end] = act
            self.rew[self.pos:end] = rew
            self.done[self.pos:end] = done
        else:
            first = self.capacity - self.pos
            self.idx[self.pos:] = idx[:first]
            self.act[self.pos:] = act[:first]
            self.rew[self.pos:] = rew[:first]
            self.done[self.pos:] = done[:first]
            rest = n - first
            self.idx[:rest] = idx[first:]
            self.act[:rest] = act[first:]
            self.rew[:rest] = rew[first:]
            self.done[:rest] = done[first:]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, all_seq):
        i = torch.randint(0, self.size, (batch_size,), device=self.device)
        state_idx = self.idx[i]
        next_idx = torch.clamp(state_idx + 1, max=len(all_seq) - 1)
        return (all_seq[state_idx], self.act[i], self.rew[i], 
                all_seq[next_idx], self.done[i])

    def __len__(self):
        return self.size


# =============================================================================
# TRAINING
# =============================================================================

def select_actions(model, states, epsilon, device):
    """Epsilon-greedy action selection"""
    n = states.shape[0]
    with torch.no_grad():
        q = model(states)
        greedy = q.argmax(dim=1)
    rand_mask = torch.rand(n, device=device) < epsilon
    rand_act = torch.randint(0, 2, (n,), device=device)
    return torch.where(rand_mask, rand_act, greedy)


def train_worker(rank: int, world_size: int, data: Dict, n_feat: int, result: Dict):
    """DDP training worker"""
    # Setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    
    # Model
    policy = DDP(LSTMDQN(n_feat).to(device), device_ids=[rank])
    target = LSTMDQN(n_feat).to(device)
    target.load_state_dict(policy.module.state_dict())
    target.eval()
    
    opt = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    # Data
    seqs, rews, dones, trans_idx = preprocess(data, SEQUENCE_LENGTH, device)
    n_trans = len(trans_idx)
    
    buf_size = ((n_trans // 100000) + 1) * 100000
    buffer = ReplayBuffer(buf_size, device)
    
    # Epsilon schedule
    decay_eps = int(0.8 * NUM_EPISODES)
    eps_decay = (EPSILON_MIN / EPSILON_MAX) ** (1.0 / decay_eps)
    epsilon = EPSILON_MAX
    
    # Split work
    per_gpu = n_trans // world_size
    start = rank * per_gpu
    end = n_trans if rank == world_size - 1 else start + per_gpu
    
    metrics = {'rewards': [], 'losses': [], 'action_rates': []}
    
    if rank == 0:
        print(f"\nTraining: {NUM_EPISODES} eps, {n_trans:,} transitions, {world_size} GPUs")
        print(f"{'Ep':>4} | {'Reward':>12} | {'Loss':>10} | {'ActRate':>8} | {'Eps':>6}")
        print("-" * 55)
    
    for ep in range(NUM_EPISODES):
        # Shuffle (sync across GPUs)
        torch.manual_seed(SEED + ep)
        perm = torch.randperm(n_trans, device=device)
        shuffled = trans_idx[perm]
        torch.manual_seed(SEED + rank + ep * world_size)
        
        my_idx = shuffled[start:end]
        ep_rew, ep_loss, n_act, n_upd, step = 0.0, 0.0, 0, 0, 0
        
        policy.train()
        for i in range(0, len(my_idx), 256):
            batch_idx = my_idx[i:i+256]
            states = seqs[batch_idx]
            batch_rew = rews[batch_idx]
            batch_done = dones[batch_idx]
            
            policy.eval()
            actions = select_actions(policy, states, epsilon, device)
            policy.train()
            
            rewards = torch.where(actions == 1, batch_rew, torch.zeros_like(batch_rew))
            ep_rew += rewards.sum().item()
            n_act += actions.sum().item()
            
            buffer.push(batch_idx, actions, rewards, batch_done)
            step += len(batch_idx)
            
            # Train
            if step >= TRAIN_FREQ and len(buffer) >= BATCH_SIZE:
                for _ in range(step // TRAIN_FREQ):
                    s, a, r, s_next, d = buffer.sample(BATCH_SIZE, seqs)
                    
                    with autocast():
                        q = policy(s)
                        q_cur = q.gather(1, a.unsqueeze(1)).squeeze()
                        with torch.no_grad():
                            q_next = target(s_next).max(1)[0]
                            q_tgt = r + (1 - d) * GAMMA * q_next
                        loss = criterion(q_cur, q_tgt)
                    
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    
                    ep_loss += loss.item()
                    n_upd += 1
                step %= TRAIN_FREQ
        
        # Sync metrics
        t = torch.tensor([ep_rew, ep_loss, n_act, n_upd], device=device)
        dist.all_reduce(t)
        ep_rew, ep_loss, n_act, n_upd = t[0].item(), t[1].item(), t[2].item(), int(t[3].item())
        
        # Target update
        if (ep + 1) % TARGET_UPDATE_FREQ == 0:
            target.load_state_dict(policy.module.state_dict())
        
        # Epsilon decay
        if ep < decay_eps:
            epsilon = max(EPSILON_MIN, epsilon * eps_decay)
        else:
            epsilon = EPSILON_MIN
        
        if rank == 0:
            act_rate = n_act / n_trans
            avg_loss = ep_loss / max(n_upd, 1)
            metrics['rewards'].append(ep_rew)
            metrics['losses'].append(avg_loss)
            metrics['action_rates'].append(act_rate)
            print(f"{ep+1:>4} | {ep_rew:>12.2f} | {avg_loss:>10.6f} | {act_rate:>8.4f} | {epsilon:>6.4f}")
        
        dist.barrier()
    
    if rank == 0:
        result['model'] = policy.module.state_dict()
        result['metrics'] = metrics
        result['final_action_rate'] = metrics['action_rates'][-1]
    
    dist.destroy_process_group()


def train_single(data: Dict, n_feat: int, device: torch.device) -> Tuple[Dict, Dict, float]:
    """Single GPU/CPU training"""
    policy = LSTMDQN(n_feat).to(device)
    target = LSTMDQN(n_feat).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    
    opt = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    seqs, rews, dones, trans_idx = preprocess(data, SEQUENCE_LENGTH, device)
    n_trans = len(trans_idx)
    
    buf_size = ((n_trans // 100000) + 1) * 100000
    buffer = ReplayBuffer(buf_size, device)
    
    decay_eps = int(0.8 * NUM_EPISODES)
    eps_decay = (EPSILON_MIN / EPSILON_MAX) ** (1.0 / decay_eps)
    epsilon = EPSILON_MAX
    
    metrics = {'rewards': [], 'losses': [], 'action_rates': []}
    
    print(f"\nTraining: {NUM_EPISODES} eps, {n_trans:,} transitions")
    print(f"{'Ep':>4} | {'Reward':>12} | {'Loss':>10} | {'ActRate':>8} | {'Eps':>6}")
    print("-" * 55)
    
    for ep in range(NUM_EPISODES):
        perm = torch.randperm(n_trans, device=device)
        shuffled = trans_idx[perm]
        
        ep_rew, ep_loss, n_act, n_upd, step = 0.0, 0.0, 0, 0, 0
        policy.train()
        
        for i in range(0, n_trans, 256):
            batch_idx = shuffled[i:i+256]
            states = seqs[batch_idx]
            batch_rew = rews[batch_idx]
            batch_done = dones[batch_idx]
            
            policy.eval()
            actions = select_actions(policy, states, epsilon, device)
            policy.train()
            
            rewards = torch.where(actions == 1, batch_rew, torch.zeros_like(batch_rew))
            ep_rew += rewards.sum().item()
            n_act += actions.sum().item()
            
            buffer.push(batch_idx, actions, rewards, batch_done)
            step += len(batch_idx)
            
            if step >= TRAIN_FREQ and len(buffer) >= BATCH_SIZE:
                for _ in range(step // TRAIN_FREQ):
                    s, a, r, s_next, d = buffer.sample(BATCH_SIZE, seqs)
                    
                    if use_amp:
                        with autocast():
                            q = policy(s)
                            q_cur = q.gather(1, a.unsqueeze(1)).squeeze()
                            with torch.no_grad():
                                q_next = target(s_next).max(1)[0]
                                q_tgt = r + (1 - d) * GAMMA * q_next
                            loss = criterion(q_cur, q_tgt)
                        opt.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        q = policy(s)
                        q_cur = q.gather(1, a.unsqueeze(1)).squeeze()
                        with torch.no_grad():
                            q_next = target(s_next).max(1)[0]
                            q_tgt = r + (1 - d) * GAMMA * q_next
                        loss = criterion(q_cur, q_tgt)
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                        opt.step()
                    
                    ep_loss += loss.item()
                    n_upd += 1
                step %= TRAIN_FREQ
        
        if (ep + 1) % TARGET_UPDATE_FREQ == 0:
            target.load_state_dict(policy.state_dict())
        
        if ep < decay_eps:
            epsilon = max(EPSILON_MIN, epsilon * eps_decay)
        else:
            epsilon = EPSILON_MIN
        
        act_rate = n_act / n_trans
        avg_loss = ep_loss / max(n_upd, 1)
        metrics['rewards'].append(ep_rew)
        metrics['losses'].append(avg_loss)
        metrics['action_rates'].append(act_rate)
        print(f"{ep+1:>4} | {ep_rew:>12.2f} | {avg_loss:>10.6f} | {act_rate:>8.4f} | {epsilon:>6.4f}")
    
    return policy.state_dict(), metrics, metrics['action_rates'][-1]


def train(data: Dict, n_feat: int) -> Tuple[Dict, Dict, float]:
    """Main training entry point"""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        print(f"Using {world_size} GPUs")
        
        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        result = manager.dict()
        
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=train_worker, args=(rank, world_size, data, n_feat, result))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        return dict(result['model']), dict(result['metrics']), result['final_action_rate']
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {device}")
        return train_single(data, n_feat, device)


# =============================================================================
# TESTING
# =============================================================================

def test_random_baseline(test_data: Dict, action_prob: float, n_rounds: int = 1000) -> float:
    """
    Test using random actions with given action probability.
    Runs resampling on GPU for speed.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Collect all rewards from test data
    all_rewards = []
    for d in test_data.values():
        all_rewards.extend(d['rewards'].tolist())
    
    rewards = torch.tensor(all_rewards, device=device)
    n = len(rewards)
    
    print(f"\nTesting with action_prob={action_prob:.4f}, {n:,} samples, {n_rounds} rounds")
    
    # Generate all random actions at once on GPU
    rand = torch.rand(n_rounds, n, device=device)
    actions = (rand < action_prob).float()
    
    # Compute rewards for each round
    round_rewards = (actions * rewards).sum(dim=1)
    
    avg_reward = round_rewards.mean().item()
    std_reward = round_rewards.std().item()
    
    print(f"Avg total reward: {avg_reward:.2f} (±{std_reward:.2f})")
    print(f"Avg reward per sample: {avg_reward / n:.6f}")
    
    return avg_reward / n


# =============================================================================
# MAIN
# =============================================================================

def main():
    start = time.time()
    
    print("=" * 60)
    print("LSTM + DQN Trading Model")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_data, n_feat = load_csv(TRAIN_PATH)
    test_data, _ = load_csv(TEST_PATH)
    print(f"Features: {n_feat}, Train symbols: {len(train_data)}, Test symbols: {len(test_data)}")
    
    # Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    model_state, metrics, final_action_rate = train(train_data, n_feat)
    
    # Save model
    torch.save(model_state, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Test
    print("\n" + "=" * 60)
    print("Testing")
    print("=" * 60)
    
    avg_reward = test_random_baseline(test_data, final_action_rate)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Final action rate: {final_action_rate:.4f}")
    print(f"Test avg reward/sample: {avg_reward:.6f}")
    print(f"Total time: {time.time() - start:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
