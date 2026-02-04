# Stock Trading with PPO: Technical Summary

## Model Design

### Architecture
The `TradingPolicyNetwork` is an LSTM-based actor-critic network with Beta distribution output for continuous position targeting.

**Components:**
- **Feature Encoder**: 2-layer LSTM (hidden_size=128) processes SEQ_LEN=60 timesteps of market features
- **Position Encoder**: Linear layer (1→32) with ReLU encodes current position
- **Combined Representation**: LSTM final hidden state concatenated with position embedding (160 dimensions)
- **Policy Head**: MLP (160→64→2) outputs Beta distribution parameters (α, β)
- **Value Head**: MLP (160→64→1) outputs state value estimate

**Output Distribution:**
- Alpha and beta parameters: `softplus(raw) + 1.0` ensures α,β > 1 (unimodal distribution)
- Target position sampled from Beta(α, β) ∈ [0, 1]

**Initialization:**
- Orthogonal weight initialization with gain √2
- Zero bias initialization

### Input Processing
**Features:** All columns except symbol, date, open, close (pre-normalized)

**Position Normalization:**
- Raw position: `holdings × close / portfolio`
- Normalized: `(position - 0.5) / s` where `s = 1 / (4 × (2α₀ + 1))` and `α₀ = softplus(0) + 1`

## Trade Simulation

Function: `simulate_trading()`

Simulation runs for SIM_LEN=240 trading days per episode.

### Indexing Convention
- Simulation data segment: rows [0, SEQ_LEN + SIM_LEN - 1]
- day_0 = row index SEQ_LEN - 1 (60th row, 0-indexed as 59)
- Simulation loop: step ∈ [0, SIM_LEN - 1]
- day_idx = SEQ_LEN - 1 + step

### Step-by-Step Simulation (for each step)

**Step 1: Calculate Current State**
```
day_idx = SEQ_LEN - 1 + step
current_close = close_prices[day_idx]
current_portfolio = cash + holdings × current_close
current_position = holdings × current_close / current_portfolio
```

**Step 2: Prepare Model Input**
```
feature_window = features[step : step + SEQ_LEN]  # 60-day window
normalized_position = (current_position - 0.5) / position_std
```

**Step 3: Model Inference (batched)**
```
α, β, value = model(feature_window, normalized_position)
if is_training:
    target = sample from Beta(α, β)
    logprob = log_prob(target)
else:
    target = α / (α + β)  # distribution mean
```

**Step 4: Execute Trade**
```
next_open = open_prices[day_idx + 1]
max_total_shares = cash / next_open + holdings
desired_shares = target × max_total_shares
share_delta = desired_shares - holdings
max_buyable = cash / next_open
trade_quantity = floor(clip(share_delta, -holdings, max_buyable))
holdings = holdings + trade_quantity
cash = cash - trade_quantity × next_open
```

**Step 5: Final Portfolio**
```
final_close = close_prices[SEQ_LEN - 1 + SIM_LEN]
final_portfolio = cash + holdings × final_close
```

**Step 6: Calculate Rewards**
```
rewards = portfolio_values[1:] / portfolio_values[:-1] - 1
```

## Generalized Advantage Estimation (GAE)

Function: `compute_advantages_and_targets()`

**Bootstrap Value:**
- Compute value estimate for terminal state using final observation

**GAE Computation (backward pass):**
```
for t = num_steps-1 down to 0:
    next_value = bootstrap (if t = num_steps-1) else values[t+1]
    td_error = rewards[t] + γ × next_value - values[t]
    gae = td_error + γ × λ × gae
    advantages[t] = gae
```

**Value Targets:**
```
value_targets = advantages + values  # before normalization
advantages = (advantages - mean) / (std + ε)
```

Parameters: γ=0.99, λ=0.95

## Training

Function: `execute_ppo_update()`

### PPO Update

**Per Update:**
1. Sample EPISODES_PER_UPDATE=1024 episodes stratified by symbol
2. Run simulation to collect experiences
3. Compute GAE advantages and value targets
4. Run PPO_EPOCHS=4 epochs of mini-batch updates

**Mini-batch Training:**
```
for each mini-batch (size=BATCH_SIZE=4096):
    α, β, value = model(states, positions)
    log_probs = Beta(α, β).log_prob(target)
    ratio = exp(log_probs - old_logprobs)

    # Clipped policy loss
    clipped_ratio = clip(ratio, 1-ε, 1+ε)
    policy_loss = -min(ratio × advantages, clipped_ratio × advantages)

    # Value loss
    value_loss = MSE(value, value_targets)

    # Entropy bonus (decaying)
    entropy = Beta(α, β).entropy()

    # Total loss
    total_loss = policy_loss + 0.5 × value_loss - entropy_coef × entropy
```

**Entropy Coefficient Schedule:**
- Start: 0.05
- End: 0.001
- Linear decay over first 80% of training

**Optimizer:** AdamW (lr=3e-4, weight_decay=1e-5)
**Learning Rate Schedule:** Cosine annealing to lr/10
**Gradient Clipping:** max_norm=0.5

### Total Updates Calculation
```
n = average segments per symbol
k = ceil(log(1 - MAX_COVERAGE) / log(1 - 1/n))
```
Where MAX_COVERAGE=0.999

## Validation

Function: `evaluate_model()` called during training

**Trigger:** Every EVAL_INTERVAL=50 updates, starting from update MIN_UPDATES=100

**Process:**
1. Run evaluation procedure (same as testing)
2. Compare mean model return against best recorded
3. If improved: save model as best, reset patience counter
4. If not improved: increment stale counter by EVAL_INTERVAL

**Early Stopping:**
- Patience = PATIENCE_FRAC × total_updates = 0.2 × total_updates
- Stop if no improvement for patience updates

**Outputs:**
- Intermediate models: `model_{update}.pt`
- Best model: `model_best.pt`
- Plots: `plot_{update}.png`, `plot_best.png`

## Testing / Evaluation

Function: `evaluate_model()`

**Sampling:**
1. Sample TEST_GROUPS=1000 groups with deterministic seeds
2. Each group: SIMS_PER_GROUP=30 segments sampled without replacement from test pool
3. Groups sampled with replacement across the 1000 groups

**Simulation:**
- Run trade simulation with is_training=False (use Beta mean instead of sampling)
- Compute model_returns = final_portfolio / INIT_CASH - 1
- Compute baseline_returns = final_close / first_open - 1

**Aggregation:**
1. Average returns within each group (30 simulations → 1 observation)
2. Report mean and std across 1000 group means

**Multi-GPU:**
- Samples distributed across GPUs (rank-interleaved)
- Results gathered via all_gather

## Reproducibility

**Seeding:**
- Global seed (SEED=42) set via `set_seed()` for random, numpy, torch
- Worker seeds: SEED + rank
- Stratified sampling: deterministic seed per update (update_idx × world_size + rank)
- Evaluation: deterministic seeds from `random.Random(SEED)`

## Output Files

| File | Description |
|------|-------------|
| training.log | Per-update metrics |
| model_{n}.pt | Checkpoint at update n |
| model_best.pt | Best validation model |
| model_final.pt | Final model (best weights) |
| plot_{n}.png | Training/validation plots |
| plot_final.png | Final test results |
