"""
LSTM + DQN Trading Model with Dual Buy/Sell Heads
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# =============================================================================
# PARAMETERS - Modify these as needed
# =============================================================================

# Training parameters
NUM_EPISODES = 100
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 5
TRAIN_EVERY_N_STEPS = 4
ACTION_BATCH_SIZE = 256

# Random baseline parameters
MIN_BASELINE_ROUNDS = 30

# Data paths
TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'

# Model output
MODEL_SAVE_PATH = 'lstm_dqn_dual_model.pth'
PLOT_SAVE_PATH = 'training_results.png'

# Random seed
SEED = 42

# =============================================================================
# SETUP
# =============================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Neural Engine (MPS)")
else:
    device = torch.device('cpu')
    print("Using CPU")

print(f"Using {NUM_WORKERS} workers for parallel processing")


# =============================================================================
# MODEL
# =============================================================================

class DualHeadLSTMDQN(nn.Module):
    """LSTM + DQN with shared backbone and separate buy/sell heads"""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.shared_fc = nn.Linear(hidden_size, 128)
        self.buy_fc1 = nn.Linear(128, 64)
        self.buy_fc2 = nn.Linear(64, 2)
        self.sell_fc1 = nn.Linear(128, 64)
        self.sell_fc2 = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        shared = self.dropout(self.relu(self.shared_fc(lstm_out)))

        buy_x = self.dropout(self.relu(self.buy_fc1(shared)))
        buy_q = self.buy_fc2(buy_x)

        sell_x = self.dropout(self.relu(self.sell_fc1(shared)))
        sell_q = self.sell_fc2(sell_x)

        return buy_q, sell_q


# =============================================================================
# REPLAY BUFFER (Index-based to save memory)
# =============================================================================

class IndexReplayBuffer:
    """Stores indices into shared data arrays instead of copying data"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros((capacity, 4), dtype=np.int64)  # idx, buy_act, sell_act, done
        self.rewards_buy = np.zeros(capacity, dtype=np.float32)
        self.rewards_sell = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push_batch(self, indices: np.ndarray, buy_actions: np.ndarray, sell_actions: np.ndarray,
                   buy_rewards: np.ndarray, sell_rewards: np.ndarray, dones: np.ndarray):
        batch_size = len(indices)

        for i in range(batch_size):
            pos = self.position
            self.buffer[pos, 0] = indices[i]
            self.buffer[pos, 1] = buy_actions[i]
            self.buffer[pos, 2] = sell_actions[i]
            self.buffer[pos, 3] = dones[i]
            self.rewards_buy[pos] = buy_rewards[i]
            self.rewards_sell[pos] = sell_rewards[i]

            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, all_states: np.ndarray) -> Tuple[torch.Tensor, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)

        state_indices = self.buffer[idxs, 0]
        buy_actions = self.buffer[idxs, 1]
        sell_actions = self.buffer[idxs, 2]
        dones = self.buffer[idxs, 3].astype(np.float32)
        buy_rewards = self.rewards_buy[idxs]
        sell_rewards = self.rewards_sell[idxs]

        # Fetch actual states and next_states from shared array
        states = all_states[state_indices]
        next_states = all_states[state_indices + 1]  # next_state = states[idx + 1]

        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(buy_actions).to(device),
            torch.LongTensor(sell_actions).to(device),
            torch.FloatTensor(buy_rewards).to(device),
            torch.FloatTensor(sell_rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device)
        )

    def __len__(self) -> int:
        return self.size


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def create_sequences_for_stock(args: Tuple) -> Optional[Dict]:
    """Create sequences for a single stock (for parallel processing)"""
    symbol, features, rewards_buy, rewards_sell, sequence_length = args

    if len(features) <= sequence_length + 1:
        return None

    n_sequences = len(features) - sequence_length
    sequences = np.zeros((n_sequences, sequence_length, features.shape[1]), dtype=np.float32)

    for i in range(n_sequences):
        sequences[i] = features[i:i + sequence_length]

    if n_sequences < 2:
        return None

    return {
        'symbol': symbol,
        'sequences': sequences,
        'rewards_buy': rewards_buy[sequence_length:].astype(np.float32),
        'rewards_sell': rewards_sell[sequence_length:].astype(np.float32),
        'n_transitions': n_sequences - 1  # Can form n_sequences - 1 (state, next_state) pairs
    }


class PreprocessedData:
    """Container for preprocessed data - stores sequences contiguously"""

    def __init__(self, data_by_stock: Dict, sequence_length: int):
        args_list = [
            (symbol, stock_data['features'], stock_data['rewards_buy'],
             stock_data['rewards_sell'], sequence_length)
            for symbol, stock_data in data_by_stock.items()
        ]

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = [r for r in executor.map(create_sequences_for_stock, args_list) if r]

        # Count total sequences (not transitions)
        total_sequences = sum(len(r['sequences']) for r in results)
        num_features = results[0]['sequences'].shape[2] if results else 0

        # Allocate contiguous arrays
        self.all_sequences = np.zeros((total_sequences, sequence_length, num_features), dtype=np.float32)
        self.all_rewards_buy = np.zeros(total_sequences, dtype=np.float32)
        self.all_rewards_sell = np.zeros(total_sequences, dtype=np.float32)
        self.all_dones = np.zeros(total_sequences, dtype=np.float32)

        # Fill arrays and track transition indices
        self.transition_indices = []  # Valid indices where (i, i+1) forms a transition
        pos = 0

        for r in results:
            n_seq = len(r['sequences'])
            self.all_sequences[pos:pos + n_seq] = r['sequences']
            self.all_rewards_buy[pos:pos + n_seq] = r['rewards_buy']
            self.all_rewards_sell[pos:pos + n_seq] = r['rewards_sell']

            # Transitions: indices 0 to n_seq-2 (so next_state = idx+1 is valid)
            for i in range(n_seq - 1):
                self.transition_indices.append(pos + i)

            # Mark last sequence of this stock as "done" for the transition ending there
            # The transition at index (pos + n_seq - 2) ends at state (pos + n_seq - 1)
            if n_seq >= 2:
                self.all_dones[pos + n_seq - 2] = 1.0

            pos += n_seq

        self.transition_indices = np.array(self.transition_indices, dtype=np.int64)
        self.total_transitions = len(self.transition_indices)
        self.n_stocks = len(results)

        print(f"Preprocessed {self.n_stocks} stocks with {self.total_transitions:,} transitions")


def load_data(filepath: str) -> Tuple[Dict, int, int]:
    """Load trading data. Features = all columns except symbol, date, reward_buy, reward_sell"""
    df = pd.read_csv(filepath)

    feature_cols = [c for c in df.columns if c not in ['symbol', 'date', 'reward_buy', 'reward_sell']]

    data_by_stock = {}
    for symbol in df['symbol'].unique():
        stock_df = df[df['symbol'] == symbol]
        data_by_stock[symbol] = {
            'features': stock_df[feature_cols].values.astype(np.float32),
            'rewards_buy': stock_df['reward_buy'].values.astype(np.float32),
            'rewards_sell': stock_df['reward_sell'].values.astype(np.float32),
        }

    return data_by_stock, len(feature_cols), len(df)


# =============================================================================
# TRAINING
# =============================================================================

def select_actions_batch(model: nn.Module, states: torch.Tensor, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Epsilon-greedy action selection for both heads"""
    batch_size = states.shape[0]

    with torch.no_grad():
        buy_q, sell_q = model(states)
        buy_actions = buy_q.argmax(dim=1).cpu().numpy()
        sell_actions = sell_q.argmax(dim=1).cpu().numpy()

    # Apply random actions
    random_mask_buy = np.random.random(batch_size) < epsilon
    random_mask_sell = np.random.random(batch_size) < epsilon
    buy_actions[random_mask_buy] = np.random.randint(0, 2, size=random_mask_buy.sum())
    sell_actions[random_mask_sell] = np.random.randint(0, 2, size=random_mask_sell.sum())

    return buy_actions, sell_actions


def train_dqn(train_data: PreprocessedData, num_features: int) -> Tuple[nn.Module, Dict]:
    """Train dual-head LSTM-DQN"""

    policy_net = DualHeadLSTMDQN(num_features).to(device)
    target_net = DualHeadLSTMDQN(num_features).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    buffer_size = math.ceil(train_data.total_transitions / 100000) * 100000
    replay_buffer = IndexReplayBuffer(buffer_size)

    # Epsilon decay: drops from max to min over 80% of episodes
    decay_episodes = int(0.8 * NUM_EPISODES)
    epsilon_decay = (EPSILON_MIN / EPSILON_MAX) ** (1.0 / decay_episodes)
    epsilon = EPSILON_MAX

    metrics = {
        'episode_rewards_buy': [], 'episode_rewards_sell': [], 'episode_rewards_total': [],
        'episode_losses': [], 'episode_buy_rates': [], 'episode_sell_rates': [], 'epsilons': []
    }

    n_transitions = train_data.total_transitions
    all_sequences = train_data.all_sequences
    all_rewards_buy = train_data.all_rewards_buy
    all_rewards_sell = train_data.all_rewards_sell
    all_dones = train_data.all_dones
    transition_indices = train_data.transition_indices

    print(f"\nTraining: {NUM_EPISODES} episodes, {n_transitions:,} transitions/ep, buffer={buffer_size:,}")
    print(f"{'Ep':>4} | {'BuyRwd':>10} | {'SellRwd':>10} | {'Total':>10} | {'Loss':>10} | {'BuyRt':>6} | {'SellRt':>6} | {'Eps':>6}")
    print("-" * 82)

    for episode in range(NUM_EPISODES):
        shuffled_indices = np.random.permutation(transition_indices)

        ep_reward_buy, ep_reward_sell, ep_loss = 0.0, 0.0, 0.0
        num_updates, total_buys, total_sells, step_count = 0, 0, 0, 0

        policy_net.train()

        for batch_start in range(0, n_transitions, ACTION_BATCH_SIZE):
            batch_idx = shuffled_indices[batch_start:batch_start + ACTION_BATCH_SIZE]
            batch_size = len(batch_idx)

            # Get states for action selection
            batch_states = all_sequences[batch_idx]
            states_tensor = torch.FloatTensor(batch_states).to(device)

            # Select actions
            policy_net.eval()
            buy_actions, sell_actions = select_actions_batch(policy_net, states_tensor, epsilon)
            policy_net.train()

            # Get rewards based on actions
            batch_rewards_buy_data = all_rewards_buy[batch_idx]
            batch_rewards_sell_data = all_rewards_sell[batch_idx]
            batch_dones = all_dones[batch_idx]

            buy_rewards = np.where(buy_actions == 1, batch_rewards_buy_data, 0.0).astype(np.float32)
            sell_rewards = np.where(sell_actions == 1, batch_rewards_sell_data, 0.0).astype(np.float32)

            ep_reward_buy += buy_rewards.sum()
            ep_reward_sell += sell_rewards.sum()
            total_buys += buy_actions.sum()
            total_sells += sell_actions.sum()

            # Store in replay buffer
            replay_buffer.push_batch(batch_idx, buy_actions, sell_actions,
                                     buy_rewards, sell_rewards, batch_dones)

            step_count += batch_size

            # Train
            if step_count >= TRAIN_EVERY_N_STEPS and len(replay_buffer) >= BATCH_SIZE:
                n_updates = step_count // TRAIN_EVERY_N_STEPS

                for _ in range(n_updates):
                    (b_states, b_buy_act, b_sell_act, b_buy_rew, b_sell_rew,
                     b_next_states, b_dones) = replay_buffer.sample(BATCH_SIZE, all_sequences)

                    buy_q, sell_q = policy_net(b_states)
                    current_buy_q = buy_q.gather(1, b_buy_act.unsqueeze(1))
                    current_sell_q = sell_q.gather(1, b_sell_act.unsqueeze(1))

                    with torch.no_grad():
                        next_buy_q, next_sell_q = target_net(b_next_states)
                        target_buy_q = b_buy_rew + (1 - b_dones) * GAMMA * next_buy_q.max(1)[0]
                        target_sell_q = b_sell_rew + (1 - b_dones) * GAMMA * next_sell_q.max(1)[0]

                    loss = criterion(current_buy_q.squeeze(), target_buy_q) + \
                           criterion(current_sell_q.squeeze(), target_sell_q)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    optimizer.step()

                    ep_loss += loss.item()
                    num_updates += 1

                step_count %= TRAIN_EVERY_N_STEPS

        # Update target network
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Update epsilon
        epsilon = max(EPSILON_MIN, epsilon * epsilon_decay) if episode < decay_episodes else EPSILON_MIN

        # Record metrics
        avg_loss = ep_loss / max(num_updates, 1)
        buy_rate = total_buys / n_transitions
        sell_rate = total_sells / n_transitions

        metrics['episode_rewards_buy'].append(ep_reward_buy)
        metrics['episode_rewards_sell'].append(ep_reward_sell)
        metrics['episode_rewards_total'].append(ep_reward_buy + ep_reward_sell)
        metrics['episode_losses'].append(avg_loss)
        metrics['episode_buy_rates'].append(buy_rate)
        metrics['episode_sell_rates'].append(sell_rate)
        metrics['epsilons'].append(epsilon)

        print(f"{episode+1:>4} | {ep_reward_buy:>10.2f} | {ep_reward_sell:>10.2f} | "
              f"{ep_reward_buy + ep_reward_sell:>10.2f} | {avg_loss:>10.6f} | "
              f"{buy_rate:>6.3f} | {sell_rate:>6.3f} | {epsilon:>6.4f}")

    return policy_net, metrics


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model: nn.Module, test_data: PreprocessedData) -> Dict:
    """Evaluate model on test data (buy and sell tested separately)"""
    model.eval()

    indices = test_data.transition_indices
    n_samples = len(indices)

    all_buy_actions, all_sell_actions = [], []
    all_buy_rewards, all_sell_rewards = [], []

    with torch.no_grad():
        for i in range(0, n_samples, 1024):
            batch_idx = indices[i:i + 1024]
            states = torch.FloatTensor(test_data.all_sequences[batch_idx]).to(device)

            buy_q, sell_q = model(states)
            buy_actions = buy_q.argmax(dim=1).cpu().numpy()
            sell_actions = sell_q.argmax(dim=1).cpu().numpy()

            buy_rewards = np.where(buy_actions == 1, test_data.all_rewards_buy[batch_idx], 0.0)
            sell_rewards = np.where(sell_actions == 1, test_data.all_rewards_sell[batch_idx], 0.0)

            all_buy_actions.extend(buy_actions)
            all_sell_actions.extend(sell_actions)
            all_buy_rewards.extend(buy_rewards)
            all_sell_rewards.extend(sell_rewards)

    all_buy_actions = np.array(all_buy_actions)
    all_sell_actions = np.array(all_sell_actions)
    all_buy_rewards = np.array(all_buy_rewards)
    all_sell_rewards = np.array(all_sell_rewards)

    results = {
        'n_samples': n_samples,
        'buy_rate': all_buy_actions.mean(),
        'sell_rate': all_sell_actions.mean(),
        'avg_buy_reward': all_buy_rewards.sum() / n_samples,
        'avg_sell_reward': all_sell_rewards.sum() / n_samples,
        'avg_total_reward': (all_buy_rewards.sum() + all_sell_rewards.sum()) / n_samples,
        'rewards_buy_data': test_data.all_rewards_buy[indices],
        'rewards_sell_data': test_data.all_rewards_sell[indices],
    }

    print(f"\n=== Model Evaluation ===")
    print(f"  Observations: {n_samples:,}")
    print(f"  Buy:  avg_reward={results['avg_buy_reward']:.6f}, rate={results['buy_rate']:.4f}")
    print(f"  Sell: avg_reward={results['avg_sell_reward']:.6f}, rate={results['sell_rate']:.4f}")
    print(f"  Total: avg_reward={results['avg_total_reward']:.6f}")

    return results


def generate_random_baseline(model_results: Dict) -> Dict:
    """Random baseline with probability matched to model's action rate"""
    rng = np.random.RandomState(SEED + 1)

    n_samples = model_results['n_samples']
    buy_rate = model_results['buy_rate']
    sell_rate = model_results['sell_rate']
    rewards_buy = model_results['rewards_buy_data']
    rewards_sell = model_results['rewards_sell_data']

    buy_rounds = max(MIN_BASELINE_ROUNDS, math.ceil(1.0 / buy_rate)) if buy_rate > 0 else MIN_BASELINE_ROUNDS
    sell_rounds = max(MIN_BASELINE_ROUNDS, math.ceil(1.0 / sell_rate)) if sell_rate > 0 else MIN_BASELINE_ROUNDS

    # Accumulate rewards directly (no array storage)
    buy_total_reward, buy_total_actions = 0.0, 0
    for _ in range(buy_rounds):
        actions = (rng.random(n_samples) < buy_rate).astype(np.float32)
        buy_total_reward += (actions * rewards_buy).sum()
        buy_total_actions += actions.sum()

    sell_total_reward, sell_total_actions = 0.0, 0
    for _ in range(sell_rounds):
        actions = (rng.random(n_samples) < sell_rate).astype(np.float32)
        sell_total_reward += (actions * rewards_sell).sum()
        sell_total_actions += actions.sum()

    results = {
        'buy_rounds': buy_rounds,
        'sell_rounds': sell_rounds,
        'avg_buy_reward_per_obs': buy_total_reward / buy_rounds / n_samples,
        'avg_sell_reward_per_obs': sell_total_reward / sell_rounds / n_samples,
        'avg_total_reward_per_obs': (buy_total_reward / buy_rounds + sell_total_reward / sell_rounds) / n_samples,
        'buy_rate': buy_total_actions / (buy_rounds * n_samples),
        'sell_rate': sell_total_actions / (sell_rounds * n_samples),
    }

    print(f"\n=== Random Baseline (min {MIN_BASELINE_ROUNDS} rounds, with replacement) ===")
    print(f"  Buy:  {buy_rounds} rounds, avg_reward/obs={results['avg_buy_reward_per_obs']:.6f}")
    print(f"  Sell: {sell_rounds} rounds, avg_reward/obs={results['avg_sell_reward_per_obs']:.6f}")
    print(f"  Total: avg_reward/obs={results['avg_total_reward_per_obs']:.6f}")

    return results


def always_trade_baseline(test_data: PreprocessedData) -> Dict:
    """Baseline that always buys and always sells"""
    indices = test_data.transition_indices
    n = len(indices)

    buy_reward = test_data.all_rewards_buy[indices].sum() / n
    sell_reward = test_data.all_rewards_sell[indices].sum() / n

    print(f"\n=== Always Trade Baseline ===")
    print(f"  Buy avg_reward/obs:  {buy_reward:.6f}")
    print(f"  Sell avg_reward/obs: {sell_reward:.6f}")
    print(f"  Total avg_reward/obs: {buy_reward + sell_reward:.6f}")

    return {
        'avg_buy_reward': buy_reward,
        'avg_sell_reward': sell_reward,
        'avg_total_reward': buy_reward + sell_reward,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(metrics: Dict, test_results: Dict, random_results: Dict,
                 always_results: Dict, save_path: str):
    """Plot training and testing results"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Dual-Head LSTM + DQN Trading Model Results', fontsize=14, fontweight='bold')

    episodes = range(1, len(metrics['episode_rewards_total']) + 1)
    window = min(10, len(episodes))

    # Row 1: Training rewards
    for i, (key, color, title) in enumerate([
        ('episode_rewards_buy', 'green', 'Buy'),
        ('episode_rewards_sell', 'red', 'Sell'),
        ('episode_rewards_total', 'blue', 'Total')
    ]):
        ax = axes[0, i]
        ax.plot(episodes, metrics[key], alpha=0.6, color=color)
        ma = np.convolve(metrics[key], np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(episodes) + 1), ma, color='dark' + color if color != 'blue' else 'darkblue', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'Training {title} Rewards')
        ax.grid(True, alpha=0.3)

    # Row 2: Loss, Action rates, Epsilon
    axes[1, 0].plot(episodes, metrics['episode_losses'], 'purple', alpha=0.7)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(episodes, metrics['episode_buy_rates'], 'green', alpha=0.7, label='Buy')
    axes[1, 1].plot(episodes, metrics['episode_sell_rates'], 'red', alpha=0.7, label='Sell')
    axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Rate')
    axes[1, 1].set_title('Action Rates')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(episodes, metrics['epsilons'], 'orange', linewidth=2)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Epsilon')
    axes[1, 2].set_title('Exploration Rate')
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3: Test comparisons
    methods = ['Model', 'Random', 'Always']
    for i, (model_key, rand_key, always_key, title, color) in enumerate([
        ('avg_buy_reward', 'avg_buy_reward_per_obs', 'avg_buy_reward', 'Buy', 'green'),
        ('avg_sell_reward', 'avg_sell_reward_per_obs', 'avg_sell_reward', 'Sell', 'red'),
        ('avg_total_reward', 'avg_total_reward_per_obs', 'avg_total_reward', 'Total', 'steelblue'),
    ]):
        ax = axes[2, i]
        vals = [test_results[model_key], random_results[rand_key], always_results[always_key]]
        bars = ax.bar(methods, vals, color=[color, 'gray', 'lightgray'], edgecolor='black')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('Avg Reward/Obs')
        ax.set_title(f'Test {title} Performance')
        for bar, v in zip(bars, vals):
            ax.annotate(f'{v:.5f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to '{save_path}'")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Dual-Head LSTM + DQN Trading Model")
    print("=" * 70)

    # Load data
    print(f"\nLoading data...")
    train_raw, num_features, train_rows = load_data(TRAIN_DATA_PATH)
    test_raw, _, test_rows = load_data(TEST_DATA_PATH)

    print(f"\nPreprocessing (sequence_length={SEQUENCE_LENGTH})...")
    train_data = PreprocessedData(train_raw, SEQUENCE_LENGTH)
    test_data = PreprocessedData(test_raw, SEQUENCE_LENGTH)

    print(f"\nData: train={train_rows:,} rows, test={test_rows:,} rows, features={num_features}")

    # Train
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    model, metrics = train_dqn(train_data, num_features)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to '{MODEL_SAVE_PATH}'")

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)

    test_results = evaluate_model(model, test_data)
    random_results = generate_random_baseline(test_results)
    always_results = always_trade_baseline(test_data)

    # Plot
    plot_results(metrics, test_results, random_results, always_results, PLOT_SAVE_PATH)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<12} {'Buy':>12} {'Sell':>12} {'Total':>12}")
    print("-" * 50)
    print(f"{'Model':<12} {test_results['avg_buy_reward']:>12.6f} {test_results['avg_sell_reward']:>12.6f} {test_results['avg_total_reward']:>12.6f}")
    print(f"{'Random':<12} {random_results['avg_buy_reward_per_obs']:>12.6f} {random_results['avg_sell_reward_per_obs']:>12.6f} {random_results['avg_total_reward_per_obs']:>12.6f}")
    print(f"{'Always':<12} {always_results['avg_buy_reward']:>12.6f} {always_results['avg_sell_reward']:>12.6f} {always_results['avg_total_reward']:>12.6f}")

    if random_results['avg_total_reward_per_obs'] != 0:
        improvement = (test_results['avg_total_reward'] - random_results['avg_total_reward_per_obs']) / \
                      abs(random_results['avg_total_reward_per_obs']) * 100
        print(f"\nModel vs Random: {improvement:+.2f}%")

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
