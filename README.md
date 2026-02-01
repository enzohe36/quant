# Stock Trading RL Model Training Script

## Overview

This script trains a reinforcement learning model to make stock trading decisions. The model uses an LSTM network to process historical price sequences and outputs trading actions (buy, hold, sell) along with position sizing certainty.

---

## 1. Data Preparation

### 1.1 Loading Raw Data

The script loads CSV files containing stock data with columns for symbol, date, and features (including price columns prefixed with `p_`).

```
train_df = load_data(CONFIG['train_path'])
test_df = load_data(CONFIG['test_path'])
```

Data is sorted by symbol and date to ensure chronological order within each stock.

### 1.2 Building the Sample Pool

The `build_pool()` function identifies valid starting points for simulations. Each sample requires `required_length` consecutive days (300 days = 60 sequence + 240 simulation).

For each symbol with sufficient history, the pool contains tuples of `(symbol, end_position)` representing valid simulation windows.

### 1.3 Episode Data Initialization

`init_episode_data()` creates a global data structure for efficient episode generation:

| Field | Description |
|-------|-------------|
| `features` | All feature values as a contiguous float32 array |
| `prices` | Open and close prices for trade execution |
| `symbol_indices` | Mapping of symbol → (start_idx, end_idx) in the arrays |
| `price_col_indices` | Column indices for price features (used for normalization) |

### 1.4 Episode Preparation

`prepare_episodes()` extracts data windows for a batch of samples:

1. For each `(symbol, end_pos)` sample, extract the full window of features and prices
2. Normalize price features by dividing by the closing price at day 0 of simulation (`p_close_0`)
3. Record `open_1` (next day's open price) for baseline return calculation

---

## 2. Trade Simulation

### 2.1 Simulation Loop

`run_simulations()` executes trading episodes over 240 time steps:

**State Tracking:**
- `cash`: Available cash (starts at 100.0)
- `holding`: Number of shares held
- `portfolio`: Total value (cash + holding × price)

**Per-Step Process:**

1. **Calculate current portfolio value** using closing price
2. **Build state representations:**
   - Sequence state: 60-day feature window (price features shifted by -1.0)
   - Info state: [cash, holding, portfolio] normalized by initial cash, then shifted by -1.0
3. **Select actions** using ε-greedy exploration:
   - Random actions with probability ε
   - Model actions otherwise (via softmax sampling)
4. **Compute trades** via `compute_trades()`:
   - Action 1 (buy): Purchase shares up to certainty × potential position
   - Action -1 (sell): Sell shares up to certainty × current holding
   - Action 0 (hold): No trade
5. **Execute trades** at next day's open price

### 2.2 Action Masking

The model outputs logits for three actions: [sell, hold, buy]. When `holding ≤ 0`, the sell action is masked by setting its logit to -1e9, preventing invalid sells.

### 2.3 Trade Computation

```python
def compute_trades(actions, certainties, cash, holding, open_next):
```

For buy actions:
- `potential = cash / open_price + holding` (max possible position)
- `desired = certainty × potential`
- `max_trade = cash / open_price` (can only buy what cash allows)
- `trade = floor(min(desired, max_trade))`

For sell actions:
- `trade = -floor(min(certainty × potential, holding))`

---

## 3. Model Architecture

### 3.1 Network Structure

```
TradingModel
├── LSTM (input_dim → 128, 2 layers, dropout=0.1)
├── State FC (3 → 32 → 32)
└── Combined (160) →
    ├── Policy Head (160 → 64 → 3)      # action logits
    ├── Certainty Head (160 → 64 → 1)   # sigmoid output
    └── Value Head (160 → 64 → 1)       # state value
```

### 3.2 Forward Pass

1. Process 60-step sequence through LSTM, take final hidden state
2. Process state info [cash, holding, portfolio] through FC layers
3. Concatenate LSTM output and state encoding
4. Output policy logits, certainty (0-1), and value estimate

---

## 4. Model Evaluation (GAE)

### 4.1 Value Estimation

`collect_experiences()` computes value estimates for all states in completed episodes:

1. Concatenate all states from all episodes
2. Run batched inference to get V(s) for each state
3. Compute bootstrap values V(s_terminal) for GAE computation

### 4.2 Generalized Advantage Estimation

```python
def compute_gae(rewards, values, final_value=0.0):
```

For each timestep t (iterating backwards):
```
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
A_t = δ_t + γ × λ × A_{t+1}
```

Where:
- `γ = 0.99` (discount factor)
- `λ = 0.95` (GAE parameter)
- `final_value` = V(s_terminal) for proper bootstrapping

Returns:
- `advantages`: A_t values for policy gradient
- `value_targets`: A_t + V(s_t) for value function training

---

## 5. Training

### 5.1 Training Loop Structure

```
For each epoch:
    Shuffle training pool
    For each batch of episodes:
        1. Run simulations with ε-greedy exploration
        2. Collect experiences and compute GAE
        3. Perform training step
        4. Update learning rate (cosine annealing)
```

### 5.2 Training Step

`train_step()` performs mini-batch PPO-style updates:

**Loss Components:**

| Component | Formula | Coefficient |
|-----------|---------|-------------|
| Policy Loss | -E[log π(a|s) × A × w] | 1.0 |
| Value Loss | MSE(V(s), targets) | 0.5 |
| Entropy Bonus | -E[Σ π log π] | -0.05 |

**Importance Sampling:**

For off-policy correction of random exploration actions:
```
π_behavior = ε × uniform + (1-ε) × π_model
w = π_model(a|s) / π_behavior(a|s)
w = clamp(w, 0.1, 10.0)
```

Weights only applied to actions taken randomly; model actions use w=1.

### 5.3 Exploration Schedule

Epsilon decays exponentially from 1.0 to 0.05 over 80% of training:
```
ε = ε_start × (ε_end / ε_start)^(update / decay_updates)
```

---

## 6. Testing

### 6.1 Test Procedure

`run_test()` evaluates the trained model:

1. Sample 1000 groups × 30 simulations = 30,000 test episodes
2. Run simulations with ε=0 (pure model policy)
3. Calculate returns for each group

### 6.2 Metrics

**Portfolio Return:**
```
(final_portfolio - initial_cash) / initial_cash
```

**Baseline Return (Buy-and-Hold):**
```
(final_close - open_1) / open_1
```

### 6.3 Output Statistics

- Mean and std of portfolio returns across groups
- Mean and std of baseline returns across groups
- Distribution plots comparing model vs baseline

---

## 7. Configuration Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sim_length` | 240 | Trading days per episode |
| `seq_length` | 60 | LSTM input sequence length |
| `gamma` | 0.99 | Discount factor |
| `lambda_` | 0.95 | GAE parameter |
| `lr` | 1e-4 | Learning rate |
| `batch_size` | 8192 | Mini-batch size |
| `episodes_per_update` | 1024 | Episodes between updates |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.05 | Final exploration rate |
| `initial_cash` | 100.0 | Starting portfolio value |
