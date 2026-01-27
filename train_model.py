"""
Stock Trading RL Model - LSTM + DQN Architecture (OPTIMIZED VERSION)

FIXES APPLIED:
1. Epsilon decay only after RANDOM_ONLY_EPISODES
2. Reward assignment only at actual buy decision points
3. Sell experience includes hold_length_actual and days_held
4. Batched model predictions O(max_hold)
5. Race condition in ReplayBuffer.sample() fixed
6. ProcessPoolExecutor created once outside loop
7. Vectorized hold_length clamping and reward computation
8. Optimized tensor conversions with np.stack + torch.from_numpy
9. Batched experience buffer operations

MODIFICATIONS:
- Use "price" column as buy/sell price instead of "next_open"
- Removed transaction cost and holding cost
- Use mean instead of mode for calculating p_hold_length
- Added ANE (Apple Silicon MPS) fallback when GPU not available
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import threading
from scipy.optimize import brentq
from scipy.stats import lognorm
import queue
warnings.filterwarnings('ignore')

NUM_WORKERS = max(1, mp.cpu_count() - 1)

# ============================================================================
# Configuration
# ============================================================================

# Device selection: CUDA > MPS (Apple Silicon ANE) > CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon Neural Engine via Metal Performance Shaders
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

INITIAL_PORTFOLIO = 1_000_000
MAX_POSITIONS = 4

# Transaction and holding costs removed for simplified calculation

LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100
TAU = 0.005

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
SEQUENCE_LENGTH = 20
FC_HIDDEN_SIZE = 128

HOLDING_PERIOD_MEAN = 20  # Changed from MODE to MEAN
HOLDING_PERIOD_SD = 10
BUY_PROBABILITY = 0.5
EPSILON_TARGET_PERCENT = 0.8

RANDOM_ONLY_EPISODES = 2

BATCH_CHUNK_SIZE = 4096
MAX_HOLD_LENGTH = 1000
MAX_CACHE_SIZE = 100_000  # Limit entry state cache


# ============================================================================
# Experience Replay (FIXED: Race condition in sample())
# ============================================================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

DualExperience = namedtuple('DualExperience', [
    'entry_state', 'state', 'action', 'reward',
    'hold_length_actual', 'days_held',
    'next_entry_state', 'next_state', 'done'
])


class ReplayBuffer:
    """Thread-safe replay buffer with fixed race condition."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, state, action, reward, next_state, done):
        with self._lock:
            self.buffer.append(Experience(state, action, reward, next_state, done))

    def push_batch(self, states, actions, rewards, next_states, dones):
        """Batch push to reduce lock contention."""
        with self._lock:
            for i in range(len(states)):
                self.buffer.append(Experience(
                    states[i], actions[i], rewards[i], next_states[i], dones[i]))

    # FIX: Convert to list inside lock to prevent race condition
    def sample(self, batch_size: int) -> List[Experience]:
        with self._lock:
            buffer_list = list(self.buffer)
        return random.sample(buffer_list, min(len(buffer_list), batch_size))

    def __len__(self):
        with self._lock:
            return len(self.buffer)


class DualReplayBuffer:
    """Thread-safe dual replay buffer with fixed race condition."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, entry_state, state, action, reward, hold_length_actual, days_held,
             next_entry_state, next_state, done):
        with self._lock:
            self.buffer.append(DualExperience(
                entry_state, state, action, reward, hold_length_actual, days_held,
                next_entry_state, next_state, done))

    def push_batch(self, entry_states, states, actions, rewards, hold_lengths, days_helds,
                   next_entry_states, next_states, dones):
        """Batch push to reduce lock contention."""
        with self._lock:
            for i in range(len(states)):
                self.buffer.append(DualExperience(
                    entry_states[i], states[i], actions[i], rewards[i],
                    hold_lengths[i], days_helds[i],
                    next_entry_states[i], next_states[i], dones[i]))

    # FIX: Convert to list inside lock
    def sample(self, batch_size: int) -> List[DualExperience]:
        with self._lock:
            buffer_list = list(self.buffer)
        return random.sample(buffer_list, min(len(buffer_list), batch_size))

    def __len__(self):
        with self._lock:
            return len(self.buffer)


# ============================================================================
# LSTM + DQN Network Architecture
# ============================================================================

class LSTM_DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = LSTM_HIDDEN_SIZE,
                 num_layers: int = LSTM_NUM_LAYERS, num_actions: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           dropout=0.2 if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, FC_HIDDEN_SIZE)
        self.fc2 = nn.Linear(FC_HIDDEN_SIZE, FC_HIDDEN_SIZE // 2)
        self.fc3 = nn.Linear(FC_HIDDEN_SIZE // 2, num_actions)
        self.ln1 = nn.LayerNorm(FC_HIDDEN_SIZE)
        self.ln2 = nn.LayerNorm(FC_HIDDEN_SIZE // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        x = F.relu(self.ln1(self.fc1(lstm_out[:, -1, :])))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class DualLSTM_DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = LSTM_HIDDEN_SIZE,
                 num_layers: int = LSTM_NUM_LAYERS, num_actions: int = 2):
        super().__init__()
        self.lstm_entry = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True,
                                  dropout=0.2 if num_layers > 1 else 0)
        self.lstm_current = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                    num_layers=num_layers, batch_first=True,
                                    dropout=0.2 if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size * 2, FC_HIDDEN_SIZE)
        self.fc2 = nn.Linear(FC_HIDDEN_SIZE, FC_HIDDEN_SIZE // 2)
        self.fc3 = nn.Linear(FC_HIDDEN_SIZE // 2, num_actions)
        self.ln1 = nn.LayerNorm(FC_HIDDEN_SIZE)
        self.ln2 = nn.LayerNorm(FC_HIDDEN_SIZE // 2)

    def forward(self, entry_state: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        entry_out, _ = self.lstm_entry(entry_state)
        current_out, _ = self.lstm_current(current_state)
        combined = torch.cat([entry_out[:, -1, :], current_out[:, -1, :]], dim=-1)
        x = F.relu(self.ln1(self.fc1(combined)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# ============================================================================
# DQN Agents (OPTIMIZED: tensor conversions)
# ============================================================================

class DQNAgent:
    def __init__(self, state_size: int, num_actions: int = 2, name: str = "agent",
                 replay_buffer_size: int = 100_000):
        self.state_size = state_size
        self.num_actions = num_actions
        self.name = name

        self.policy_net = LSTM_DQN(state_size, num_actions=num_actions).to(DEVICE)
        self.target_net = LSTM_DQN(state_size, num_actions=num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.epsilon = EPSILON_START
        self.steps = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            return self.policy_net(state_tensor).argmax(dim=1).item()

    # OPTIMIZED: Use np.stack + torch.from_numpy
    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        experiences = self.replay_buffer.sample(BATCH_SIZE)

        states = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(DEVICE)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=DEVICE)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=DEVICE)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(DEVICE)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32, device=DEVICE)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self._soft_update()

        return loss.item()

    def _soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)

    def decay_epsilon(self, epsilon_decay: float):
        self.epsilon = max(EPSILON_END, self.epsilon * epsilon_decay)

    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


class DualDQNAgent:
    def __init__(self, state_size: int, num_actions: int = 2, name: str = "sell_agent",
                 replay_buffer_size: int = 100_000):
        self.state_size = state_size
        self.num_actions = num_actions
        self.name = name

        self.policy_net = DualLSTM_DQN(state_size, num_actions=num_actions).to(DEVICE)
        self.target_net = DualLSTM_DQN(state_size, num_actions=num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = DualReplayBuffer(replay_buffer_size)
        self.epsilon = EPSILON_START
        self.steps = 0

    def select_action(self, entry_state: np.ndarray, current_state: np.ndarray,
                      explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            entry_t = torch.from_numpy(entry_state).float().unsqueeze(0).to(DEVICE)
            current_t = torch.from_numpy(current_state).float().unsqueeze(0).to(DEVICE)
            return self.policy_net(entry_t, current_t).argmax(dim=1).item()

    # OPTIMIZED: Use np.stack + torch.from_numpy
    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        experiences = self.replay_buffer.sample(BATCH_SIZE)

        entry_states = torch.from_numpy(np.stack([e.entry_state for e in experiences])).float().to(DEVICE)
        states = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(DEVICE)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=DEVICE)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=DEVICE)
        next_entry_states = torch.from_numpy(np.stack([e.next_entry_state for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(DEVICE)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32, device=DEVICE)

        current_q = self.policy_net(entry_states, states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_entry_states, next_states).argmax(dim=1)
            next_q = self.target_net(next_entry_states, next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self._soft_update()

        return loss.item()

    def _soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)

    def decay_epsilon(self, epsilon_decay: float):
        self.epsilon = max(EPSILON_END, self.epsilon * epsilon_decay)

    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


# ============================================================================
# Data Loading
# ============================================================================

def load_data(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    # Exclude price and delist from features
    feature_cols = [col for col in df.columns if col not in ['symbol', 'date', 'price', 'delist']]
    print(f"  Loaded {len(df):,} rows, {df['symbol'].nunique()} stocks")
    return df, feature_cols


def get_stock_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return df[df['symbol'] == symbol].reset_index(drop=True)


def calculate_training_params(train_df: pd.DataFrame, num_stocks: int) -> dict:
    total_rows = len(train_df)
    avg_rows_per_stock = total_rows / train_df['symbol'].nunique()
    steps_per_stock = avg_rows_per_stock - SEQUENCE_LENGTH - 1

    # Total steps across all stocks
    total_steps = num_stocks * steps_per_stock

    # Buy buffer: total steps, rounded up to 100k's
    buy_buffer_size = int(np.ceil(total_steps / 100_000) * 100_000)

    # Sell buffer: total_steps * buy_probability * mean_hold_length, rounded up to 100k's
    sell_buffer_raw = total_steps * BUY_PROBABILITY * HOLDING_PERIOD_MEAN
    sell_buffer_size = int(np.ceil(sell_buffer_raw / 100_000) * 100_000)

    num_episodes = max(50, min(200, int(np.sqrt(num_stocks * steps_per_stock / 100))))

    effective_episodes = num_episodes - RANDOM_ONLY_EPISODES
    target_episode = int(effective_episodes * EPSILON_TARGET_PERCENT)
    epsilon_decay = np.exp(np.log(EPSILON_END / EPSILON_START) / max(1, target_episode))

    return {
        'num_episodes': num_episodes,
        'buy_buffer_size': buy_buffer_size,
        'sell_buffer_size': sell_buffer_size,
        'epsilon_decay': epsilon_decay
    }


# ============================================================================
# Training Functions
# ============================================================================

def calculate_hold_length_distribution(mean: float, sd: float, nrow_train: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate hold length distribution using MEAN (not mode) as input.
    Adapted from R code:
        meanlog = log(mean) - 0.5 * sdlog^2
        variance = (exp(sdlog^2) - 1) * exp(2 * meanlog + sdlog^2)
    """
    def find_sdlog(sdlog, mean, sd):
        # For lognormal: mean = exp(meanlog + sdlog^2/2)
        # So: meanlog = log(mean) - 0.5 * sdlog^2
        meanlog = np.log(mean) - 0.5 * sdlog**2
        # variance = (exp(sdlog^2) - 1) * exp(2*meanlog + sdlog^2)
        variance = (np.exp(sdlog**2) - 1) * np.exp(2 * meanlog + sdlog**2)
        return np.sqrt(variance) - sd

    sdlog = brentq(find_sdlog, 0.01, 3, args=(mean, sd))
    meanlog = np.log(mean) - 0.5 * sdlog**2

    x = np.arange(1, 1001)
    pdf = lognorm.pdf(x, s=sdlog, scale=np.exp(meanlog))
    cdf = lognorm.cdf(x, s=sdlog, scale=np.exp(meanlog))

    # Filter based on CDF threshold (from R code)
    valid_mask = cdf <= 1 - 1 / (2 * mean * nrow_train)
    x, pdf = x[valid_mask], pdf[valid_mask]
    pdf = pdf / pdf.sum()

    return x, pdf


def simulate_stock_trades_worker(args: Tuple) -> Dict:
    (symbol, features, prices, nrow, epsilon, p_buy, hold_x, hold_pdf) = args

    rand_buy = (np.random.random(nrow) < epsilon).astype(np.int32)
    buy_rand = (np.random.random(nrow) < p_buy).astype(np.int32)
    rand_sell = (np.random.random(nrow) < epsilon).astype(np.int32)
    hold_length_rand = np.random.choice(hold_x, size=nrow, p=hold_pdf).astype(np.int32)

    return {
        'symbol': symbol, 'nrow': nrow,
        'rand_buy': rand_buy, 'buy_rand': buy_rand,
        'rand_sell': rand_sell, 'hold_length_rand': hold_length_rand
    }


def compute_model_predictions_batched(
    stock_results: Dict, features: np.ndarray,
    buy_agent, sell_agent, device, use_model: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched GPU inference: O(max_hold) instead of O(n²)."""
    nrow = stock_results['nrow']
    rand_buy, buy_rand, rand_sell = stock_results['rand_buy'], stock_results['buy_rand'], stock_results['rand_sell']

    buy_model = np.zeros(nrow, dtype=np.int32)
    hold_length_model = np.zeros(nrow, dtype=np.int32)

    if not use_model:
        return buy_model, hold_length_model

    # Batched buy predictions
    valid_indices = np.arange(SEQUENCE_LENGTH, nrow)
    for chunk_start in range(0, len(valid_indices), BATCH_CHUNK_SIZE):
        chunk_indices = valid_indices[chunk_start:chunk_start + BATCH_CHUNK_SIZE]
        states = np.stack([features[i - SEQUENCE_LENGTH:i] for i in chunk_indices])
        with torch.no_grad():
            preds = buy_agent.policy_net(torch.from_numpy(states).float().to(device)).argmax(dim=1).cpu().numpy()
        buy_model[chunk_indices] = preds

    # Identify model-sell positions
    buy_actual = ((rand_buy == 1) & (buy_rand == 1)) | ((rand_buy == 0) & (buy_model == 1))
    model_sell_indices = np.where((buy_actual == 1) & (rand_sell == 0) & (np.arange(nrow) >= SEQUENCE_LENGTH))[0]

    if len(model_sell_indices) == 0:
        return buy_model, hold_length_model

    # Batched sell predictions
    active = {int(idx): 1 for idx in model_sell_indices}
    entry_cache = {idx: features[idx - SEQUENCE_LENGTH:idx].copy() for idx in active}

    for _ in range(MAX_HOLD_LENGTH):
        if not active:
            break

        batch_items = [(buy_idx, offset, buy_idx + offset)
                       for buy_idx, offset in active.items()
                       if buy_idx + offset < nrow and buy_idx + offset >= SEQUENCE_LENGTH]
        if not batch_items:
            break

        all_actions = {}
        for chunk_start in range(0, len(batch_items), BATCH_CHUNK_SIZE):
            chunk = batch_items[chunk_start:chunk_start + BATCH_CHUNK_SIZE]
            entry_states = np.stack([entry_cache[b[0]] for b in chunk])
            current_states = np.stack([features[b[2] - SEQUENCE_LENGTH:b[2]] for b in chunk])

            with torch.no_grad():
                actions = sell_agent.policy_net(
                    torch.from_numpy(entry_states).float().to(device),
                    torch.from_numpy(current_states).float().to(device)
                ).argmax(dim=1).cpu().numpy()

            for i, (buy_idx, _, _) in enumerate(chunk):
                all_actions[buy_idx] = actions[i]

        to_remove = []
        for buy_idx, offset in list(active.items()):
            action = all_actions.get(buy_idx, 0)
            if action == 1 or offset >= MAX_HOLD_LENGTH - 1:
                hold_length_model[buy_idx] = offset
                to_remove.append(buy_idx)
            elif buy_idx + offset + 1 >= nrow:
                hold_length_model[buy_idx] = nrow - buy_idx - 1
                to_remove.append(buy_idx)
            else:
                active[buy_idx] = offset + 1

        for idx in to_remove:
            del active[idx]

    for buy_idx, offset in active.items():
        hold_length_model[buy_idx] = min(offset, nrow - buy_idx - 1)

    return buy_model, hold_length_model


# ============================================================================
# OPTIMIZED: Vectorized Experience Streaming
# ============================================================================

def stream_experiences_to_buffers_vectorized(
    stock_results: Dict, features: np.ndarray, prices: np.ndarray,
    delist: np.ndarray, buy_model: np.ndarray, hold_length_model: np.ndarray,
    buy_agent: DQNAgent, sell_agent: DualDQNAgent,
    entry_state_cache: Dict
) -> Tuple[int, int, float]:
    """Vectorized experience streaming with batched buffer operations."""
    nrow = stock_results['nrow']
    rand_buy, buy_rand = stock_results['rand_buy'], stock_results['buy_rand']
    rand_sell, hold_length_rand = stock_results['rand_sell'], stock_results['hold_length_rand']
    symbol = stock_results['symbol']

    # Compute buy_actual
    buy_actual = (((rand_buy == 1) & (buy_rand == 1)) | ((rand_buy == 0) & (buy_model == 1))).astype(np.int32)

    # Compute hold_length_actual (VECTORIZED)
    hold_length_actual = np.zeros(nrow, dtype=np.int32)
    mask_rand = (buy_actual == 1) & (rand_sell == 1)
    mask_model = (buy_actual == 1) & (rand_sell == 0)
    hold_length_actual[mask_rand] = hold_length_rand[mask_rand]
    hold_length_actual[mask_model] = hold_length_model[mask_model]

    # Track original hold lengths before clamping
    original_hold_length = hold_length_actual.copy()

    # VECTORIZED: Clamp to valid range
    max_allowed = nrow - np.arange(nrow) - 1
    hold_length_actual = np.where(hold_length_actual > 0,
                                   np.minimum(hold_length_actual, max_allowed),
                                   hold_length_actual)

    # Identify trades that were clamped (would extend past data)
    was_clamped = (original_hold_length > max_allowed) & (buy_actual == 1)

    # Check if stock is delisted (last row has delist=1)
    is_delisted = delist[-1] == 1 if len(delist) > 0 else False

    # For clamped trades: if delisted, keep trade with reward=-1; if not delisted, discard
    if not is_delisted:
        # Discard clamped trades by setting hold_length to 0
        hold_length_actual[was_clamped] = 0

    # VECTORIZED: Compute trade rewards (simplified - no transaction/holding costs)
    buy_indices = np.where((buy_actual == 1) & (hold_length_actual > 0))[0]
    trade_rewards = np.zeros(nrow, dtype=np.float32)

    if len(buy_indices) > 0:
        sell_indices = buy_indices + hold_length_actual[buy_indices]
        valid = sell_indices < nrow
        buy_prices = prices[buy_indices[valid]]
        sell_prices = prices[sell_indices[valid]]
        nonzero_mask = buy_prices > 0
        trade_rewards[buy_indices[valid][nonzero_mask]] = (
            (sell_prices[nonzero_mask] - buy_prices[nonzero_mask]) / buy_prices[nonzero_mask]
        )

        # Override reward to -1 for clamped trades on delisted stocks
        if is_delisted:
            clamped_buy_indices = buy_indices[was_clamped[buy_indices]]
            trade_rewards[clamped_buy_indices] = -1.0

    # --- Buy experiences (batched) ---
    valid_range = np.arange(SEQUENCE_LENGTH, nrow - 1)
    n_buy_exp = len(valid_range)

    if n_buy_exp > 0:
        # Pre-allocate arrays
        states_list = []
        next_states_list = []

        # Build states in chunks to manage memory
        chunk_size = 10000
        for chunk_start in range(0, n_buy_exp, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_buy_exp)
            chunk_indices = valid_range[chunk_start:chunk_end]

            for i in chunk_indices:
                states_list.append(features[i - SEQUENCE_LENGTH:i])
                next_states_list.append(features[i - SEQUENCE_LENGTH + 1:i + 1])

        actions_arr = buy_actual[valid_range]

        # Rewards: only at buy decisions with valid trades
        rewards_arr = np.zeros(n_buy_exp, dtype=np.float32)
        buy_mask = (actions_arr == 1) & (hold_length_actual[valid_range] > 0)
        rewards_arr[buy_mask] = trade_rewards[valid_range[buy_mask]]

        dones_arr = np.zeros(n_buy_exp, dtype=bool)
        dones_arr[-1] = True

        # Batch push
        buy_agent.replay_buffer.push_batch(
            states_list, actions_arr.tolist(), rewards_arr.tolist(),
            next_states_list, dones_arr.tolist()
        )

        total_reward = float(rewards_arr[actions_arr == 1].sum())
    else:
        total_reward = 0.0

    # --- Sell experiences (batched per trade) ---
    num_sell_exp = 0

    # Limit cache size
    if len(entry_state_cache) > MAX_CACHE_SIZE:
        entry_state_cache.clear()

    # Process in batches of trades
    if len(buy_indices) > 0:
        sell_entry_states = []
        sell_states = []
        sell_actions = []
        sell_rewards = []
        sell_hold_lengths = []
        sell_days_held = []
        sell_next_entry_states = []
        sell_next_states = []
        sell_dones = []

        for buy_idx in buy_indices:
            if buy_idx < SEQUENCE_LENGTH:
                continue

            cache_key = (symbol, buy_idx)
            if cache_key not in entry_state_cache:
                entry_state_cache[cache_key] = features[buy_idx - SEQUENCE_LENGTH:buy_idx].copy()
            entry_state = entry_state_cache[cache_key]

            hold_len = hold_length_actual[buy_idx]
            trade_reward = trade_rewards[buy_idx]

            for day in range(1, hold_len + 1):
                sell_idx = buy_idx + day
                if sell_idx >= nrow or sell_idx < SEQUENCE_LENGTH:
                    continue

                current_state = features[sell_idx - SEQUENCE_LENGTH:sell_idx]
                next_state = (features[sell_idx - SEQUENCE_LENGTH + 1:sell_idx + 1]
                             if sell_idx + 1 < nrow and sell_idx + 1 >= SEQUENCE_LENGTH
                             else current_state)

                sell_entry_states.append(entry_state)
                sell_states.append(current_state)
                sell_actions.append(1 if day == hold_len else 0)
                sell_rewards.append(trade_reward)
                sell_hold_lengths.append(hold_len)
                sell_days_held.append(day)
                sell_next_entry_states.append(entry_state)
                sell_next_states.append(next_state)
                sell_dones.append(sell_idx == nrow - 2)
                num_sell_exp += 1

                # Batch push periodically
                if len(sell_states) >= 5000:
                    sell_agent.replay_buffer.push_batch(
                        sell_entry_states, sell_states, sell_actions, sell_rewards,
                        sell_hold_lengths, sell_days_held,
                        sell_next_entry_states, sell_next_states, sell_dones
                    )
                    sell_entry_states, sell_states, sell_actions = [], [], []
                    sell_rewards, sell_hold_lengths, sell_days_held = [], [], []
                    sell_next_entry_states, sell_next_states, sell_dones = [], [], []

        # Push remaining
        if sell_states:
            sell_agent.replay_buffer.push_batch(
                sell_entry_states, sell_states, sell_actions, sell_rewards,
                sell_hold_lengths, sell_days_held,
                sell_next_entry_states, sell_next_states, sell_dones
            )

    return n_buy_exp, num_sell_exp, total_reward


# ============================================================================
# Training Loop (FIXED: ProcessPool outside loop)
# ============================================================================

def train_agents(train_df: pd.DataFrame, feature_cols: List[str],
                 buy_agent: DQNAgent, sell_agent: DualDQNAgent,
                 training_params: dict) -> Dict:
    """Train with ProcessPool created once and all optimizations applied."""
    num_episodes = training_params['num_episodes']
    epsilon_decay = training_params['epsilon_decay']

    # Prepare stock data
    print("Preparing stock data...")
    stock_data = {}
    total_rows = 0
    for symbol in train_df['symbol'].unique():
        data = get_stock_data(train_df, symbol)
        if len(data) >= SEQUENCE_LENGTH + 10:
            feature_df = data[feature_cols].copy().ffill().bfill()
            stock_data[symbol] = {
                'features': feature_df.values.astype(np.float32),
                'prices': data['price'].values.astype(np.float32),
                'delist': data['delist'].values.astype(np.int32),
                'nrow': len(data)
            }
            total_rows += len(data)

    valid_symbols = list(stock_data.keys())
    num_stocks = len(valid_symbols)

    # Changed: use HOLDING_PERIOD_MEAN instead of HOLDING_PERIOD_MODE
    hold_x, hold_pdf = calculate_hold_length_distribution(
        HOLDING_PERIOD_MEAN, HOLDING_PERIOD_SD, total_rows)

    # Metrics
    episode_rewards, portfolio_values, buy_losses, sell_losses = [], [], [], []

    # Background training
    training_queue = queue.Queue()
    training_done = threading.Event()

    def background_trainer():
        while not training_done.is_set() or not training_queue.empty():
            try:
                task = training_queue.get(timeout=0.1)
                if task is None:
                    break
                agent, is_buy = task
                if len(agent.replay_buffer) >= BATCH_SIZE:
                    loss = agent.train_step()
                    if loss:
                        (buy_losses if is_buy else sell_losses).append(loss)
                training_queue.task_done()
            except queue.Empty:
                continue

    trainer_thread = threading.Thread(target=background_trainer, daemon=True)
    trainer_thread.start()

    print(f"\nTraining: {num_stocks} stocks, {num_episodes} episodes")
    print(f"Random episodes: {RANDOM_ONLY_EPISODES}, Model starts: episode {RANDOM_ONLY_EPISODES + 1}")

    # FIX: Create ProcessPoolExecutor ONCE outside the loop
    executor = ProcessPoolExecutor(max_workers=NUM_WORKERS)

    try:
        for episode in range(num_episodes):
            epsilon = buy_agent.epsilon
            use_model = (episode >= RANDOM_ONLY_EPISODES)
            entry_state_cache = {}

            # Parallel random vector generation
            worker_args = [
                (sym, stock_data[sym]['features'], stock_data[sym]['prices'],
                 stock_data[sym]['nrow'], epsilon, BUY_PROBABILITY, hold_x, hold_pdf)
                for sym in valid_symbols
            ]

            partial_results = {r['symbol']: r for r in executor.map(simulate_stock_trades_worker, worker_args)}

            # Model predictions + stream experiences
            total_buy_exp, total_sell_exp, total_reward = 0, 0, 0.0

            for symbol in valid_symbols:
                result = partial_results[symbol]
                features = stock_data[symbol]['features']
                prices = stock_data[symbol]['prices']
                delist = stock_data[symbol]['delist']

                buy_model, hold_length_model = compute_model_predictions_batched(
                    result, features, buy_agent, sell_agent, DEVICE, use_model)

                num_buy, num_sell, reward = stream_experiences_to_buffers_vectorized(
                    result, features, prices, delist, buy_model, hold_length_model,
                    buy_agent, sell_agent, entry_state_cache)

                total_buy_exp += num_buy
                total_sell_exp += num_sell
                total_reward += reward

                # Queue background training
                if episode >= 1:
                    for _ in range(min(5, num_buy // BATCH_SIZE)):
                        training_queue.put((buy_agent, True))
                        training_queue.put((sell_agent, False))

            entry_state_cache.clear()

            # Metrics
            avg_reward = total_reward / max(1, total_buy_exp)
            episode_rewards.append(avg_reward)
            portfolio_values.append(INITIAL_PORTFOLIO * (1 + avg_reward))

            print(f"Ep {episode+1}/{num_episodes} | Buy: {total_buy_exp:,} | "
                  f"Sell: {total_sell_exp:,} | Reward: {avg_reward:.4f} | "
                  f"ε: {epsilon:.3f} | Model: {'Y' if use_model else 'N'}")

            # Decay epsilon only after RANDOM_ONLY_EPISODES
            if episode >= RANDOM_ONLY_EPISODES:
                buy_agent.decay_epsilon(epsilon_decay)
                sell_agent.decay_epsilon(epsilon_decay)

    finally:
        # FIX: Properly shutdown executor
        executor.shutdown(wait=False)

    # Cleanup training thread
    training_done.set()
    training_queue.put(None)
    trainer_thread.join(timeout=10)

    return {
        'episode_rewards': episode_rewards,
        'portfolio_values': portfolio_values,
        'buy_losses': buy_losses,
        'sell_losses': sell_losses
    }


# ============================================================================
# Testing Functions
# ============================================================================

class PortfolioManager:
    def __init__(self, initial_cash: float, max_positions: int):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_positions = max_positions
        self.positions = {}
        self.trades = []

    def get_available_slots(self) -> int:
        return self.max_positions - len(self.positions)

    def get_buy_amount(self) -> float:
        slots = self.get_available_slots()
        return self.cash / slots if slots > 0 else 0

    def buy(self, symbol: str, price: float, date: str, step: int, date_dt,
            entry_state: np.ndarray = None) -> bool:
        if symbol in self.positions or self.get_available_slots() <= 0:
            return False

        buy_amount = self.get_buy_amount()
        if buy_amount < 100:
            return False

        # Simplified: no transaction cost
        shares = buy_amount / price

        self.cash -= buy_amount
        self.positions[symbol] = {
            'shares': shares, 'entry_price': price, 'entry_date': date,
            'entry_step': step, 'entry_date_dt': date_dt, 'entry_state': entry_state
        }
        self.trades.append({
            'type': 'BUY', 'symbol': symbol, 'date': date, 'price': price,
            'shares': shares, 'amount': buy_amount
        })
        return True

    def sell(self, symbol: str, price: float, date: str, current_step: int, current_date_dt) -> bool:
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        # Simplified: no transaction cost or holding cost
        sale_value = pos['shares'] * price
        net_proceeds = sale_value
        self.cash += net_proceeds

        calendar_days = (current_date_dt - pos['entry_date_dt']).days
        profit = (price - pos['entry_price']) / pos['entry_price']
        self.trades.append({
            'type': 'SELL', 'symbol': symbol, 'date': date, 'price': price,
            'shares': pos['shares'], 'amount': net_proceeds,
            'entry_price': pos['entry_price'], 'holding_days': calendar_days, 'profit_pct': profit
        })
        del self.positions[symbol]
        return True

    def get_portfolio_value(self, prices: Dict[str, float], current_step: int = None,
                           dates_dt_map: Dict = None) -> float:
        # Simplified: no holding cost or transaction cost deductions
        position_value = 0
        for symbol, pos in self.positions.items():
            current_price = prices.get(symbol, pos['entry_price'])
            gross_value = pos['shares'] * current_price
            position_value += gross_value
        return self.cash + position_value

    def save_trades(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write("Trade Log\n" + "=" * 80 + "\n\n")
            for trade in self.trades:
                if trade['type'] == 'BUY':
                    f.write(f"BUY  | {trade['symbol']} | {trade['date']} | "
                           f"${trade['price']:.2f} | {trade['shares']:.2f} shares\n")
                else:
                    f.write(f"SELL | {trade['symbol']} | {trade['date']} | "
                           f"${trade['price']:.2f} | {trade['holding_days']}d | "
                           f"{trade['profit_pct']:.2%}\n")
            f.write(f"\nTotal: {len(self.trades)} | Cash: ${self.cash:,.2f}\n")


def _preprocess_stock_worker(args: Tuple) -> Optional[Tuple]:
    symbol, stock_data_dict, feature_cols = args
    stock_data = pd.DataFrame(stock_data_dict)
    if len(stock_data) >= SEQUENCE_LENGTH + 10:
        feature_df = stock_data[feature_cols].copy().ffill().bfill()
        return symbol, {
            'features': feature_df.values.astype(np.float32),
            'prices': stock_data['price'].values,  # Changed: use 'price' instead of 'next_open'
            'dates': stock_data['date'].values,
            'dates_dt': pd.to_datetime(stock_data['date'].values),
            'step': SEQUENCE_LENGTH
        }
    return None


def test_agents(test_df: pd.DataFrame, feature_cols: List[str],
                buy_agent: DQNAgent, sell_agent: DualDQNAgent,
                max_positions: int = MAX_POSITIONS) -> Dict:
    symbols = test_df['symbol'].unique()
    portfolio = PortfolioManager(INITIAL_PORTFOLIO, max_positions)

    # Parallel preprocessing
    worker_args = [(s, get_stock_data(test_df, s).to_dict('list'), feature_cols) for s in symbols]
    stock_envs = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for result in executor.map(_preprocess_stock_worker, worker_args):
            if result:
                stock_envs[result[0]] = result[1]

    portfolio_history = [INITIAL_PORTFOLIO]
    buy_hold_history = [INITIAL_PORTFOLIO]
    bh_symbols, bh_shares, bh_entry_dates, bh_entry_prices = [], {}, {}, {}
    bh_amounts_spent, bh_total_spent = {}, 0

    print(f"\nTesting on {len(stock_envs)} stocks...")
    max_steps = min(len(env['prices']) - SEQUENCE_LENGTH - 1 for env in stock_envs.values())

    for step in range(max_steps):
        current_prices, stock_states, active_symbols, dates_dt_map = {}, {}, [], {}

        for symbol, env in stock_envs.items():
            if env['step'] >= len(env['prices']) - 1:
                continue
            current_prices[symbol] = env['prices'][env['step']]
            dates_dt_map[symbol] = env['dates_dt'][env['step']]

            start_idx = max(0, env['step'] - SEQUENCE_LENGTH)
            state = env['features'][start_idx:env['step']]
            if len(state) < SEQUENCE_LENGTH:
                state = np.vstack([np.zeros((SEQUENCE_LENGTH - len(state), len(feature_cols))), state])
            stock_states[symbol] = state
            active_symbols.append(symbol)

        # Batch sell decisions
        held = [s for s in active_symbols if s in portfolio.positions]
        if held:
            with torch.no_grad():
                entry_t = torch.from_numpy(np.stack([portfolio.positions[s]['entry_state'] for s in held])).float().to(DEVICE)
                current_t = torch.from_numpy(np.stack([stock_states[s] for s in held])).float().to(DEVICE)
                q_values = sell_agent.policy_net(entry_t, current_t)
                sell_actions = (q_values[:, 1] > q_values[:, 0]).cpu().numpy()
            for symbol, should_sell in zip(held, sell_actions):
                if should_sell:
                    env = stock_envs[symbol]
                    portfolio.sell(symbol, current_prices[symbol], env['dates'][env['step']],
                                 env['step'], env['dates_dt'][env['step']])

        # Batch buy decisions
        n = portfolio.get_available_slots()
        if n > 0:
            non_held = [s for s in active_symbols if s not in portfolio.positions]
            if non_held:
                with torch.no_grad():
                    states_t = torch.from_numpy(np.stack([stock_states[s] for s in non_held])).float().to(DEVICE)
                    q_values = buy_agent.policy_net(states_t)
                    buy_q, hold_q = q_values[:, 1].cpu().numpy(), q_values[:, 0].cpu().numpy()

                candidates = [(s, bq) for s, bq, hq in zip(non_held, buy_q, hold_q) if bq > hq]
                candidates.sort(key=lambda x: x[1], reverse=True)

                for symbol, _ in candidates[:n]:
                    env = stock_envs[symbol]
                    success = portfolio.buy(symbol, current_prices[symbol], env['dates'][env['step']],
                                          env['step'], env['dates_dt'][env['step']], stock_states[symbol].copy())

                    if success and len(bh_symbols) < max_positions and symbol not in bh_symbols:
                        bh_symbols.append(symbol)
                        price = current_prices[symbol]
                        remaining_slots = max_positions - len(bh_symbols) + 1
                        amount = (INITIAL_PORTFOLIO - bh_total_spent) / remaining_slots
                        # Simplified B&H: no transaction cost
                        shares = amount / price
                        bh_shares[symbol], bh_entry_dates[symbol] = shares, env['dates_dt'][env['step']]
                        bh_entry_prices[symbol], bh_amounts_spent[symbol] = price, amount
                        bh_total_spent += amount

        for env in stock_envs.values():
            if env['step'] < len(env['prices']) - 1:
                env['step'] += 1

        portfolio_history.append(portfolio.get_portfolio_value(current_prices, step, dates_dt_map))

        # B&H value (simplified: no holding cost or transaction cost)
        bh_value = INITIAL_PORTFOLIO - bh_total_spent
        for symbol in bh_symbols:
            if symbol in current_prices:
                bh_value += bh_shares[symbol] * current_prices[symbol]
        buy_hold_history.append(bh_value)

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{max_steps} | Portfolio: ${portfolio_history[-1]:,.0f} | B&H: ${bh_value:,.0f}")

    # Close positions
    for symbol in list(portfolio.positions.keys()):
        env = stock_envs[symbol]
        idx = min(env['step'], len(env['prices']) - 1)
        portfolio.sell(symbol, env['prices'][idx], env['dates'][idx], max_steps, env['dates_dt'][idx])

    return {
        'portfolio_history': portfolio_history,
        'buy_hold_history': buy_hold_history,
        'final_value': portfolio.cash,
        'trades': portfolio.trades,
        'portfolio_manager': portfolio,
        'bh_symbols': bh_symbols
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_results(train_metrics: Dict, test_metrics: Dict, output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training reward
    rewards = train_metrics['episode_rewards']
    window = min(20, len(rewards) // 5) if len(rewards) >= 5 else 1
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid') if window > 0 and len(rewards) > window else rewards
    axes[0, 0].plot(smoothed, color='blue')
    axes[0, 0].set_title('Training: Avg Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # Training portfolio
    axes[0, 1].plot(train_metrics['portfolio_values'], color='green')
    axes[0, 1].axhline(y=INITIAL_PORTFOLIO, color='red', linestyle='--')
    axes[0, 1].set_title('Training: Portfolio Value')
    axes[0, 1].grid(True, alpha=0.3)

    # Test performance
    axes[1, 0].plot(test_metrics['portfolio_history'], label='DQN', color='blue')
    axes[1, 0].plot(test_metrics['buy_hold_history'], label='B&H', color='orange')
    axes[1, 0].legend()
    axes[1, 0].set_title('Test: DQN vs Buy-and-Hold')
    axes[1, 0].grid(True, alpha=0.3)

    # Training loss
    for losses, label in [(train_metrics['buy_losses'], 'Buy'), (train_metrics['sell_losses'], 'Sell')]:
        if losses:
            window = min(100, len(losses) // 10) if len(losses) >= 10 else 1
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid') if window > 0 and len(losses) > window else losses
            axes[1, 1].plot(smoothed, label=label, alpha=0.7)
    axes[1, 1].legend()
    axes[1, 1].set_title('Training: Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'), dpi=150)
    plt.close()
    print(f"\nPlots saved to {output_dir}/training_results.png")


# ============================================================================
# Main
# ============================================================================

def main():
    device_name = str(DEVICE)
    if device_name == "mps":
        device_name = "mps (Apple Silicon ANE)"
    print(f"Using device: {device_name}")
    os.makedirs('models', exist_ok=True)

    train_df, feature_cols = load_data('data/train.csv')
    test_df, _ = load_data('data/test.csv')

    num_valid = sum(1 for s in train_df['symbol'].unique()
                    if len(get_stock_data(train_df, s)) >= SEQUENCE_LENGTH + 10)
    training_params = calculate_training_params(train_df, num_valid)

    print(f"\nTraining params: {training_params['num_episodes']} episodes, "
          f"buy_buffer={training_params['buy_buffer_size']:,}, "
          f"sell_buffer={training_params['sell_buffer_size']:,}, "
          f"decay={training_params['epsilon_decay']:.6f}")

    buy_agent = DQNAgent(len(feature_cols), replay_buffer_size=training_params['buy_buffer_size'])
    sell_agent = DualDQNAgent(len(feature_cols), replay_buffer_size=training_params['sell_buffer_size'])

    print(f"Buy params: {sum(p.numel() for p in buy_agent.policy_net.parameters()):,}")
    print(f"Sell params: {sum(p.numel() for p in sell_agent.policy_net.parameters()):,}")

    print("\n" + "="*60 + "\nTRAINING\n" + "="*60)
    train_metrics = train_agents(train_df, feature_cols, buy_agent, sell_agent, training_params)

    print("\n" + "="*60 + "\nTESTING\n" + "="*60)
    test_metrics = test_agents(test_df, feature_cols, buy_agent, sell_agent, MAX_POSITIONS)

    # Save
    k = int(round(test_metrics['final_value'] / 1000))
    buy_agent.save(f'models/buy_{k}.pt')
    sell_agent.save(f'models/sell_{k}.pt')
    test_metrics['portfolio_manager'].save_trades(f'models/trades_{k}.txt')

    plot_results(train_metrics, test_metrics, 'models')

    # Summary
    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    print(f"Final DQN: ${test_metrics['final_value']:,.0f} ({(test_metrics['final_value']/INITIAL_PORTFOLIO-1)*100:.2f}%)")
    print(f"Final B&H: ${test_metrics['buy_hold_history'][-1]:,.0f} ({(test_metrics['buy_hold_history'][-1]/INITIAL_PORTFOLIO-1)*100:.2f}%)")

    sells = [t for t in test_metrics['trades'] if t['type'] == 'SELL']
    if sells:
        profits = [t['profit_pct'] for t in sells]
        print(f"Trades: {len(sells)} | Win rate: {sum(1 for p in profits if p > 0)/len(profits)*100:.1f}% | "
              f"Avg profit: {np.mean(profits)*100:.2f}%")


if __name__ == "__main__":
    main()
