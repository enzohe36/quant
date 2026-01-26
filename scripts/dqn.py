"""
Dual DQN for Stock Trading - Separate Buy and Sell Models
Buy DQN learns: when to enter positions (BUY vs WAIT)
Sell DQN learns: when to exit positions (SELL vs HOLD)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)


class DualDQNAgent:
    """Dual DQN Agent with separate models for buy and sell decisions"""
    
    def __init__(self, state_size, learning_rate=0.001, buffer_size=10000, 
                 buy_probability=0.3, num_episodes=50):
        self.state_size = state_size
        self.gamma = 0.95  # Discount factor
        self.buy_probability = buy_probability
        self.epsilon = 1.0
        self.epsilon_min = 0.10  # Keep 10% exploration to discover diverse patterns
        
        # Linear decay: reaches epsilon_min at 80% of episodes
        target_episode = int(0.8 * num_episodes)
        self.epsilon_decay_per_step = (self.epsilon - self.epsilon_min) / target_episode
        
        print(f"Linear epsilon decay: {self.epsilon_decay_per_step:.6f} per episode "
              f"(reaches {self.epsilon_min} at episode {target_episode})")
        
        # Separate replay buffers for buy and sell experiences
        self.buy_memory = deque(maxlen=buffer_size)
        self.sell_memory = deque(maxlen=buffer_size)
        
        # Buy DQN: decides BUY (0) or WAIT (1)
        self.buy_model = self._build_model(learning_rate, name="BuyModel")
        self.buy_target_model = self._build_model(learning_rate, name="BuyTargetModel")
        
        # Sell DQN: decides SELL (0) or HOLD (1)
        self.sell_model = self._build_model(learning_rate, name="SellModel")
        self.sell_target_model = self._build_model(learning_rate, name="SellTargetModel")
        
        self.update_target_models()
        
    def _build_model(self, learning_rate, name="DQN"):
        """Build neural network for Q-value approximation"""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2, activation='linear')
        ], name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='huber'
        )
        
        return model
    
    def update_target_models(self):
        """Copy weights to target networks"""
        self.buy_target_model.set_weights(self.buy_model.get_weights())
        self.sell_target_model.set_weights(self.sell_model.get_weights())
    
    def remember_buy_experience(self, state, action, reward, next_state, done):
        """Store buy decision experience"""
        self.buy_memory.append((state, action, reward, next_state, done))
    
    def remember_sell_experience(self, state, action, reward, next_state, done):
        """Store sell decision experience"""
        self.sell_memory.append((state, action, reward, next_state, done))
    
    def _train_step(self, states, actions, rewards, next_states, dones, model, target_model):
        """Training step using Bellman equation"""
        next_q = target_model(next_states, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1.0 - tf.cast(dones, tf.float32))
        
        actions_one_hot = tf.one_hot(actions, 2, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            q_values = model(states, training=True)
            q_selected = tf.reduce_sum(q_values * actions_one_hot, axis=1)
            loss = tf.reduce_mean(tf.keras.losses.huber(targets, q_selected))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    def train_buy_model(self, batch_size=2048, num_batches=50):
        """Train buy model"""
        if len(self.buy_memory) < batch_size:
            return None
        
        # Limit batches to available samples
        max_batches = len(self.buy_memory) // batch_size
        actual_batches = min(num_batches, max_batches)
        
        if actual_batches == 0:
            return None
        
        total_loss = 0
        all_samples = random.sample(self.buy_memory, batch_size * actual_batches)
        
        for i in range(actual_batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            minibatch = all_samples[batch_start:batch_end]
            
            states = tf.constant([t[0] for t in minibatch], dtype=tf.float32)
            actions = tf.constant([t[1] for t in minibatch], dtype=tf.int32)
            rewards = tf.constant([t[2] for t in minibatch], dtype=tf.float32)
            next_states = tf.constant([t[3] for t in minibatch], dtype=tf.float32)
            dones = tf.constant([t[4] for t in minibatch], dtype=tf.bool)
            
            loss = self._train_step(states, actions, rewards, next_states, dones, 
                                   self.buy_model, self.buy_target_model)
            total_loss += float(loss)
        
        return total_loss / actual_batches
    
    def train_sell_model(self, batch_size=2048, num_batches=50):
        """Train sell model"""
        if len(self.sell_memory) < batch_size:
            return None
        
        # Limit batches to available samples
        max_batches = len(self.sell_memory) // batch_size
        actual_batches = min(num_batches, max_batches)
        
        if actual_batches == 0:
            return None
        
        total_loss = 0
        all_samples = random.sample(self.sell_memory, batch_size * actual_batches)
        
        for i in range(actual_batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            minibatch = all_samples[batch_start:batch_end]
            
            states = tf.constant([t[0] for t in minibatch], dtype=tf.float32)
            actions = tf.constant([t[1] for t in minibatch], dtype=tf.int32)
            rewards = tf.constant([t[2] for t in minibatch], dtype=tf.float32)
            next_states = tf.constant([t[3] for t in minibatch], dtype=tf.float32)
            dones = tf.constant([t[4] for t in minibatch], dtype=tf.bool)
            
            loss = self._train_step(states, actions, rewards, next_states, dones,
                                   self.sell_model, self.sell_target_model)
            total_loss += float(loss)
        
        return total_loss / actual_batches
    
    def should_buy(self, state):
        """Decide whether to buy"""
        if np.random.rand() <= self.epsilon:
            return np.random.rand() < self.buy_probability
        
        state_tensor = tf.constant([state], dtype=tf.float32)
        q_values = self.buy_model(state_tensor, training=False)
        return q_values[0, 0] > q_values[0, 1]
    
    def should_sell(self, state):
        """Decide whether to sell"""
        if np.random.rand() <= self.epsilon:
            return np.random.rand() < 0.5
        
        state_tensor = tf.constant([state], dtype=tf.float32)
        q_values = self.sell_model(state_tensor, training=False)
        return q_values[0, 0] > q_values[0, 1]
    
    def save(self, filepath_prefix):
        """Save both models"""
        self.buy_model.save(f"{filepath_prefix}_buy.keras")
        self.sell_model.save(f"{filepath_prefix}_sell.keras")
        print(f"Saved: {filepath_prefix}_buy.keras and {filepath_prefix}_sell.keras")
    
    def load(self, filepath_prefix):
        """Load both models"""
        self.buy_model = keras.models.load_model(f"{filepath_prefix}_buy.keras")
        self.sell_model = keras.models.load_model(f"{filepath_prefix}_sell.keras")
        self.update_target_models()
        print(f"Loaded: {filepath_prefix}_buy.keras and {filepath_prefix}_sell.keras")


def train_dual_dqn(train_data_list, state_size, episodes=50, num_envs=16, agent=None, 
                   buffer_size=None, buy_probability=0.3, median_hold=20, 
                   initial_balance=1000000.0, short_wait_penalty=0.01):
    """
    Train dual DQN agent on multiple stocks
    train_data_list: list of (stock_name, data_array) tuples
    short_wait_penalty: penalty per step for buying within median_hold after selling
    """
    
    print("Dual DQN Training | Episodes:", episodes, "| Envs:", num_envs)
    print(f"Initial balance: ${initial_balance:,.0f}")
    print(f"Training on {len(train_data_list)} stocks")
    print(f"Short wait penalty: {short_wait_penalty} per step")
    
    if agent is None:
        if buffer_size is None:
            buffer_size = 10000
        agent = DualDQNAgent(state_size, buffer_size=buffer_size, 
                            buy_probability=buy_probability, num_episodes=episodes)
        print(f"New agent | Buffer size: {buffer_size}")
    else:
        print(f"Continuing | Buy exp: {len(agent.buy_memory)} | Sell exp: {len(agent.sell_memory)}")
    
    transaction_cost = 0.00009
    holding_cost_per_day = 0.0001875
    
    episode_rewards = []
    episode_values = []
    
    for episode in range(episodes):
        episode_trades = 0
        episode_reward = 0
        episode_balances = []
        buy_loss_total = 0
        sell_loss_total = 0
        training_count = 0
        stock_counter = 0
        
        # Train on each stock sequentially
        for stock_name, train_data in train_data_list:
            balances = np.full(num_envs, initial_balance, dtype=np.float32)
            shares = np.zeros(num_envs, dtype=np.int32)
            positions = np.zeros(num_envs, dtype=np.int32)
            steps = np.zeros(num_envs, dtype=np.int32)
            active = np.ones(num_envs, dtype=bool)
            
            buy_states = [None] * num_envs
            buy_prices = np.zeros(num_envs, dtype=np.float32)
            buy_values = np.zeros(num_envs, dtype=np.float32)
            buy_steps = np.zeros(num_envs, dtype=np.int32)
            last_sell_steps = np.full(num_envs, -median_hold, dtype=np.int32)  # Track last sell
            
            # Run environments through this stock's data
            while np.any(active):
                n_active = np.sum(active)
                if n_active == 0:
                    break
                
                states = np.zeros((n_active, state_size), dtype=np.float32)
                active_idx = np.where(active)[0]
                
                # Vectorized state building
                valid_mask = steps[active_idx] < len(train_data) - 1
                
                # Deactivate environments that reached end
                for i, env_idx in enumerate(active_idx):
                    if not valid_mask[i]:
                        active[env_idx] = False
                
                # Get only still-valid environments
                active_idx = np.where(active)[0]
                if len(active_idx) == 0:
                    break
                
                # Vectorized feature extraction
                steps_array = steps[active_idx]
                features_batch = train_data[steps_array, 1:]  # All features at once
                
                # Vectorized portfolio state
                positions_batch = positions[active_idx].reshape(-1, 1)
                prices_batch = train_data[steps_array, 0].reshape(-1, 1)
                shares_batch = shares[active_idx].reshape(-1, 1)
                balances_batch = balances[active_idx].reshape(-1, 1)
                
                portfolio_batch = np.hstack([
                    positions_batch,
                    (shares_batch * prices_batch) / initial_balance,
                    balances_batch / initial_balance
                ])
                
                # Combine features and portfolio
                states = np.hstack([features_batch, portfolio_batch]).astype(np.float32)
                
                active_idx = np.where(active)[0]
                for i, env_idx in enumerate(active_idx):
                    if not active[env_idx]:
                        continue
                        
                    step = steps[env_idx]
                    state = states[i]
                    current_price = train_data[step, 0]
                    
                    if step + 1 < len(train_data):
                        execution_price = train_data[step + 1, 0]
                    else:
                        execution_price = current_price
                    
                    if positions[env_idx] == 0:
                        should_buy = agent.should_buy(state)
                        
                        if should_buy and step + 1 < len(train_data):
                            # Calculate short wait penalty
                            wait_time = step - last_sell_steps[env_idx]
                            if wait_time < median_hold:
                                # Penalty for buying too soon after selling
                                wait_penalty = (median_hold - wait_time) * short_wait_penalty
                            else:
                                wait_penalty = 0
                            
                            # Execute buy at next open price
                            shares[env_idx] = int(balances[env_idx] / (execution_price * (1 + transaction_cost)))
                            cost = shares[env_idx] * execution_price * (1 + transaction_cost)
                            balances[env_idx] -= cost
                            positions[env_idx] = 1
                            
                            buy_states[env_idx] = state.copy()
                            buy_prices[env_idx] = execution_price
                            buy_values[env_idx] = balances[env_idx] + shares[env_idx] * execution_price
                            buy_steps[env_idx] = step
                            
                            # Store wait penalty for this buy decision
                            buy_states[env_idx] = (state.copy(), wait_penalty)
                    
                    else:
                        should_sell = agent.should_sell(state)
                        
                        if should_sell and step + 1 < len(train_data):
                            proceeds = shares[env_idx] * execution_price * (1 - transaction_cost)
                            balances[env_idx] += proceeds
                            
                            sell_value = balances[env_idx]
                            buy_value = buy_values[env_idx]
                            base_reward = (sell_value - buy_value) / buy_value
                            
                            hold_duration = step - buy_steps[env_idx]
                            holding_cost = hold_duration * holding_cost_per_day
                            
                            # Get wait penalty from buy state
                            buy_state, wait_penalty = buy_states[env_idx]
                            
                            # Total reward includes wait penalty
                            reward = base_reward - holding_cost - wait_penalty
                            
                            next_features = train_data[min(step + 1, len(train_data) - 1), 1:]
                            next_portfolio = np.array([0, 0, balances[env_idx] / initial_balance], 
                                                     dtype=np.float32)
                            next_state = np.concatenate([next_features, next_portfolio])
                            
                            agent.remember_buy_experience(buy_state, 0, reward, next_state, False)
                            agent.remember_sell_experience(state, 0, reward, next_state, False)
                            
                            episode_trades += 1
                            episode_reward += reward
                            
                            # Track when we sold
                            last_sell_steps[env_idx] = step
                            
                            shares[env_idx] = 0
                            positions[env_idx] = 0
                            buy_states[env_idx] = None
                        
                        else:
                            hold_reward = -holding_cost_per_day
                            next_features = train_data[min(step + 1, len(train_data) - 1), 1:]
                            next_portfolio = np.array([
                                1,
                                shares[env_idx] * train_data[min(step + 1, len(train_data) - 1), 0] / initial_balance,
                                balances[env_idx] / initial_balance
                            ], dtype=np.float32)
                            next_state = np.concatenate([next_features, next_portfolio])
                            
                            agent.remember_sell_experience(state, 1, hold_reward, next_state, False)
                    
                    steps[env_idx] += 1
                    if steps[env_idx] >= len(train_data) - 1:
                        active[env_idx] = False
            
            # Incomplete trades at stock boundary are discarded (not force-closed)
            # Record final balances after this stock
            final_vals = balances + shares * train_data[-1, 0]
            episode_balances.extend(final_vals)
            
            stock_counter += 1
            
            # Train after every stock with larger batches
            if len(agent.buy_memory) >= 2048:
                loss = agent.train_buy_model(batch_size=2048, num_batches=50)
                if loss is not None:
                    buy_loss_total += loss
                    training_count += 1
            
            if len(agent.sell_memory) >= 2048:
                loss = agent.train_sell_model(batch_size=2048, num_batches=50)
                if loss is not None:
                    sell_loss_total += loss
        
        # Update target networks every 5 episodes (reduced overhead)
        if episode % 5 == 0:
            agent.update_target_models()
        
        # Linear epsilon decay
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon -= agent.epsilon_decay_per_step
            agent.epsilon = max(agent.epsilon_min, agent.epsilon)
        
        avg_val = np.mean(episode_balances)
        
        episode_rewards.append(episode_reward / max(episode_trades, 1))
        episode_values.append(avg_val)
        
        # Average losses across all training runs
        loss_str = ""
        if training_count > 0:
            avg_buy_loss = buy_loss_total / training_count
            avg_sell_loss = sell_loss_total / training_count
            loss_str = f"BuyLoss: {avg_buy_loss:.4f} | SellLoss: {avg_sell_loss:.4f} | "
        
        print(f"Ep {episode+1}/{episodes} | Trades: {episode_trades} | "
              f"AvgReward: {episode_rewards[-1]:+.3f} | {loss_str}"
              f"Value: ${avg_val:,.0f} | ε: {agent.epsilon:.3f}")
    
    return {
        'agent': agent,
        'rewards': episode_rewards,
        'values': episode_values
    }


def test_dual_dqn_portfolio(test_data_dict, agent, initial_balance=1000000.0):
    """
    Test dual DQN on portfolio of multiple stocks
    test_data_dict: {ticker: data_array}
    Allocates up to 1/N of cash per stock when buying (N = stocks available to buy)
    """
    agent.epsilon = 0
    
    transaction_cost = 0.00009
    holding_cost_per_day = 0.0001875
    
    # Portfolio state
    cash = initial_balance
    positions = {ticker: {'shares': 0, 'buy_step': None} for ticker in test_data_dict.keys()}
    
    # Find max length for iteration
    max_len = max(len(data) for data in test_data_dict.values())
    
    portfolio_values = [initial_balance]
    
    for step in range(max_len - 1):
        # Evaluate each stock
        for ticker, test_data in test_data_dict.items():
            if step >= len(test_data) - 1:
                continue
                
            features = test_data[step, 1:]
            current_price = test_data[step, 0]
            
            if step + 1 < len(test_data):
                execution_price = test_data[step + 1, 0]
            else:
                execution_price = current_price
            
            # Build state
            position = 1 if positions[ticker]['shares'] > 0 else 0
            shares_value = positions[ticker]['shares'] * current_price
            portfolio = np.array([position, shares_value / initial_balance, cash / initial_balance], 
                                dtype=np.float32)
            state = np.concatenate([features, portfolio])
            
            if positions[ticker]['shares'] == 0:  # Not holding this stock
                should_buy = agent.should_buy(state)
                
                if should_buy and step + 1 < len(test_data):
                    # Count stocks available to buy (not currently holding)
                    available_to_buy = sum(1 for p in positions.values() if p['shares'] == 0)
                    
                    # Allocate up to 1/N of cash
                    allocation = cash / available_to_buy
                    shares = int(allocation / (execution_price * (1 + transaction_cost)))
                    
                    if shares > 0:
                        cost = shares * execution_price * (1 + transaction_cost)
                        cash -= cost
                        positions[ticker]['shares'] = shares
                        positions[ticker]['buy_step'] = step
            
            else:  # Holding this stock
                should_sell = agent.should_sell(state)
                
                if should_sell and step + 1 < len(test_data):
                    shares = positions[ticker]['shares']
                    hold_duration = step - positions[ticker]['buy_step']
                    daily_holding_cost = shares * execution_price * holding_cost_per_day * hold_duration
                    
                    proceeds = shares * execution_price * (1 - transaction_cost) - daily_holding_cost
                    cash += proceeds
                    
                    positions[ticker]['shares'] = 0
                    positions[ticker]['buy_step'] = None
        
        # Calculate total portfolio value
        total_value = cash
        for ticker, pos in positions.items():
            if pos['shares'] > 0 and step + 1 < len(test_data_dict[ticker]):
                next_price = test_data_dict[ticker][step + 1, 0]
                
                # Subtract holding costs
                if pos['buy_step'] is not None:
                    hold_duration = step + 1 - pos['buy_step']
                    holding_cost = pos['shares'] * next_price * holding_cost_per_day * hold_duration
                    total_value += pos['shares'] * next_price - holding_cost
                else:
                    total_value += pos['shares'] * next_price
        
        portfolio_values.append(total_value)
    
    return {
        'portfolio_values': portfolio_values,
        'final_cash': cash,
        'positions': positions
    }


def calculate_buyhold_portfolio(test_data_dict, initial_balance=1000000.0):
    """Calculate buy-and-hold returns for equal-weighted portfolio"""
    transaction_cost = 0.00009
    n_stocks = len(test_data_dict)
    allocation_per_stock = initial_balance / n_stocks
    
    holdings = {}
    for ticker, test_data in test_data_dict.items():
        initial_price = test_data[0, 0]
        shares = int(allocation_per_stock / (initial_price * (1 + transaction_cost)))
        cost = shares * initial_price * (1 + transaction_cost)
        holdings[ticker] = {'shares': shares, 'cost': cost}
    
    # Remaining cash after buying all
    cash = initial_balance - sum(h['cost'] for h in holdings.values())
    
    # Calculate portfolio value over time
    max_len = max(len(data) for data in test_data_dict.values())
    portfolio_values = []
    
    for step in range(max_len):
        value = cash
        for ticker, data in test_data_dict.items():
            if step < len(data):
                value += holdings[ticker]['shares'] * data[step, 0]
        portfolio_values.append(value)
    
    return {
        'portfolio_values': portfolio_values,
        'final_value': portfolio_values[-1]
    }


def test_dual_dqn(test_data, agent, initial_balance=1000000.0):
    """Test dual DQN agent"""
    agent.epsilon = 0
    
    transaction_cost = 0.00009
    holding_cost_per_day = 0.0001875
    balance = initial_balance
    shares = 0
    position = 0
    
    portfolio_values = [initial_balance]
    actions_taken = []
    
    buy_step = None
    
    for step in range(len(test_data) - 1):
        features = test_data[step, 1:]
        portfolio = np.array([
            position,
            shares * test_data[step, 0] / initial_balance,
            balance / initial_balance
        ], dtype=np.float32)
        state = np.concatenate([features, portfolio])
        
        current_price = test_data[step, 0]
        
        if step + 1 < len(test_data):
            execution_price = test_data[step + 1, 0]
        else:
            execution_price = current_price
        
        if position == 0:
            should_buy = agent.should_buy(state)
            
            if should_buy and step + 1 < len(test_data):
                shares = int(balance / (execution_price * (1 + transaction_cost)))
                balance -= shares * execution_price * (1 + transaction_cost)
                position = 1
                buy_step = step
                actions_taken.append(0)
            else:
                actions_taken.append(2)
                
        else:
            should_sell = agent.should_sell(state)
            
            if should_sell and step + 1 < len(test_data):
                hold_duration = step - buy_step
                daily_holding_cost = shares * execution_price * holding_cost_per_day * hold_duration
                
                proceeds = shares * execution_price * (1 - transaction_cost) - daily_holding_cost
                balance += proceeds
                shares = 0
                position = 0
                actions_taken.append(1)
            else:
                actions_taken.append(2)
        
        if position == 1:
            current_holding_cost = shares * test_data[step + 1, 0] * holding_cost_per_day * (step + 1 - buy_step)
            next_price = test_data[step + 1, 0]
            portfolio_value = balance + shares * next_price - current_holding_cost
        else:
            next_price = test_data[step + 1, 0]
            portfolio_value = balance + shares * next_price
        portfolio_values.append(portfolio_value)
    
    return {
        'portfolio_values': portfolio_values,
        'actions': actions_taken
    }


def analyze_buy_patterns(agent, test_data, initial_balance=1000000.0):
    """Analyze buy patterns"""
    import pandas as pd
    
    agent.epsilon = 0
    buy_decisions = []
    
    for step in range(len(test_data) - 1):
        features = test_data[step, 1:]
        portfolio = np.array([0, 0, 1.0], dtype=np.float32)
        state = np.concatenate([features, portfolio])
        
        if agent.should_buy(state):
            buy_decisions.append({
                'step': step,
                'price': test_data[step, 0],
                'returns_1': features[0],
                'returns_5': features[1],
                'returns_20': features[2],
                'sma_5_ratio': features[3],
                'sma_20_ratio': features[4],
                'sma_50_ratio': features[5],
                'rsi': features[6],
                'macd': features[7],
                'macd_signal': features[8],
                'bb_upper': features[9],
                'bb_lower': features[10],
                'volatility': features[11]
            })
    
    if len(buy_decisions) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(buy_decisions)


def analyze_sell_patterns(agent, test_data, initial_balance=1000000.0):
    """Analyze sell patterns"""
    import pandas as pd
    
    agent.epsilon = 0
    sell_decisions = []
    position = 0
    buy_step = None
    buy_price = None
    
    for step in range(len(test_data) - 1):
        features = test_data[step, 1:]
        portfolio = np.array([position, 0, 1.0], dtype=np.float32)
        state = np.concatenate([features, portfolio])
        
        current_price = test_data[step, 0]
        
        if position == 0:
            if agent.should_buy(state):
                position = 1
                buy_step = step
                buy_price = current_price
        else:
            if agent.should_sell(state):
                hold_duration = step - buy_step
                price_change = (current_price - buy_price) / buy_price
                
                sell_decisions.append({
                    'step': step,
                    'price': current_price,
                    'hold_duration': hold_duration,
                    'price_change_pct': price_change * 100,
                    'returns_1': features[0],
                    'returns_5': features[1],
                    'returns_20': features[2],
                    'sma_5_ratio': features[3],
                    'sma_20_ratio': features[4],
                    'sma_50_ratio': features[5],
                    'rsi': features[6],
                    'macd': features[7],
                    'macd_signal': features[8],
                    'bb_upper': features[9],
                    'bb_lower': features[10],
                    'volatility': features[11]
                })
                
                position = 0
    
    if len(sell_decisions) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(sell_decisions)


def print_pattern_analysis(buy_df, sell_df):
    """Print pattern analysis"""
    print("\n" + "="*80)
    print("PATTERN ANALYSIS: Buy Model")
    print("="*80)
    
    if len(buy_df) > 0:
        print(f"\nTotal Buy Signals: {len(buy_df)}")
        print("\nFeature Statistics at Buy Signals:")
        print("-" * 80)
        
        features = ['rsi', 'macd', 'sma_20_ratio', 'returns_5', 'volatility']
        print(buy_df[features].describe().to_string())
        
        print("\n" + "-" * 80)
        print("Pattern Diversity:")
        print("-" * 80)
        
        rsi_std = buy_df['rsi'].std()
        print(f"RSI range: {buy_df['rsi'].min():.2f} to {buy_df['rsi'].max():.2f}")
        print(f"RSI std dev: {rsi_std:.2f} {'(LOW diversity)' if rsi_std < 5 else '(GOOD diversity)'}")
        
        print("\n" + "-" * 80)
        print("Pattern Clusters:")
        print("-" * 80)
        
        oversold = buy_df[buy_df['rsi'] < 35]
        print(f"Oversold (RSI<35): {len(oversold)} ({len(oversold)/len(buy_df)*100:.1f}%)")
        
        overbought = buy_df[buy_df['rsi'] > 65]
        print(f"Overbought (RSI>65): {len(overbought)} ({len(overbought)/len(buy_df)*100:.1f}%)")
        
        bullish_macd = buy_df[buy_df['macd'] > buy_df['macd_signal']]
        print(f"Bullish MACD: {len(bullish_macd)} ({len(bullish_macd)/len(buy_df)*100:.1f}%)")
        
        below_sma = buy_df[buy_df['sma_20_ratio'] < 0.98]
        print(f"Below SMA20: {len(below_sma)} ({len(below_sma)/len(buy_df)*100:.1f}%)")
        
        above_sma = buy_df[buy_df['sma_20_ratio'] > 1.02]
        print(f"Above SMA20: {len(above_sma)} ({len(above_sma)/len(buy_df)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("PATTERN ANALYSIS: Sell Model")
    print("="*80)
    
    if len(sell_df) > 0:
        print(f"\nTotal Sell Signals: {len(sell_df)}")
        print("\nFeature Statistics:")
        print("-" * 80)
        
        features = ['hold_duration', 'price_change_pct', 'rsi', 'macd', 'volatility']
        print(sell_df[features].describe().to_string())
        
        print("\n" + "-" * 80)
        print("Trade Performance:")
        print("-" * 80)
        
        profitable = sell_df[sell_df['price_change_pct'] > 0]
        losing = sell_df[sell_df['price_change_pct'] < 0]
        
        print(f"Profitable: {len(profitable)} ({len(profitable)/len(sell_df)*100:.1f}%)")
        print(f"Losing: {len(losing)} ({len(losing)/len(sell_df)*100:.1f}%)")
        
        if len(profitable) > 0:
            print(f"\nWinners - Avg hold: {profitable['hold_duration'].mean():.1f} days, "
                  f"Avg gain: {profitable['price_change_pct'].mean():.2f}%")
        
        if len(losing) > 0:
            print(f"Losers - Avg hold: {losing['hold_duration'].mean():.1f} days, "
                  f"Avg loss: {losing['price_change_pct'].mean():.2f}%")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("Dual DQN Module")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {len(tf.config.list_physical_devices('GPU'))}")
