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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        self.gamma = 0.95
        self.buy_probability = buy_probability
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        
        target_episode = int(0.8 * num_episodes)
        self.epsilon_decay = (self.epsilon_min / self.epsilon) ** (1.0 / target_episode)
        
        print(f"Epsilon decay: {self.epsilon_decay:.6f} (reaches {self.epsilon_min} at episode {target_episode})")
        
        self.buy_memory = deque(maxlen=buffer_size)
        self.sell_memory = deque(maxlen=buffer_size)
        
        self.buy_model = self._build_model(learning_rate, name="BuyModel")
        self.buy_target_model = self._build_model(learning_rate, name="BuyTargetModel")
        
        self.sell_model = self._build_model(learning_rate, name="SellModel")
        self.sell_target_model = self._build_model(learning_rate, name="SellTargetModel")
        
        self.update_target_models()
        
    def _build_model(self, learning_rate, name="DQN"):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2, activation='linear')
        ], name=name)
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='huber')
        return model
    
    def update_target_models(self):
        self.buy_target_model.set_weights(self.buy_model.get_weights())
        self.sell_target_model.set_weights(self.sell_model.get_weights())
    
    def remember_buy_experience(self, state, action, reward, next_state, done):
        self.buy_memory.append((state, action, reward, next_state, done))
    
    def remember_sell_experience(self, state, action, reward, next_state, done):
        self.sell_memory.append((state, action, reward, next_state, done))
    
    def _train_step(self, states, actions, rewards, next_states, dones, model, target_model):
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
    
    def train_buy_model(self, batch_size=256, num_batches=10):
        if len(self.buy_memory) < batch_size:
            return None
        
        total_loss = 0
        for _ in range(num_batches):
            minibatch = random.sample(self.buy_memory, batch_size)
            
            states = tf.constant([t[0] for t in minibatch], dtype=tf.float32)
            actions = tf.constant([t[1] for t in minibatch], dtype=tf.int32)
            rewards = tf.constant([t[2] for t in minibatch], dtype=tf.float32)
            next_states = tf.constant([t[3] for t in minibatch], dtype=tf.float32)
            dones = tf.constant([t[4] for t in minibatch], dtype=tf.bool)
            
            loss = self._train_step(states, actions, rewards, next_states, dones, 
                                   self.buy_model, self.buy_target_model)
            total_loss += float(loss)
        
        return total_loss / num_batches
    
    def train_sell_model(self, batch_size=256, num_batches=10):
        if len(self.sell_memory) < batch_size:
            return None
        
        total_loss = 0
        for _ in range(num_batches):
            minibatch = random.sample(self.sell_memory, batch_size)
            
            states = tf.constant([t[0] for t in minibatch], dtype=tf.float32)
            actions = tf.constant([t[1] for t in minibatch], dtype=tf.int32)
            rewards = tf.constant([t[2] for t in minibatch], dtype=tf.float32)
            next_states = tf.constant([t[3] for t in minibatch], dtype=tf.float32)
            dones = tf.constant([t[4] for t in minibatch], dtype=tf.bool)
            
            loss = self._train_step(states, actions, rewards, next_states, dones,
                                   self.sell_model, self.sell_target_model)
            total_loss += float(loss)
        
        return total_loss / num_batches
    
    def should_buy(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.rand() < self.buy_probability
        
        state_tensor = tf.constant([state], dtype=tf.float32)
        q_values = self.buy_model(state_tensor, training=False)
        return q_values[0, 0] > q_values[0, 1]
    
    def should_sell(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.rand() < 0.5
        
        state_tensor = tf.constant([state], dtype=tf.float32)
        q_values = self.sell_model(state_tensor, training=False)
        return q_values[0, 0] > q_values[0, 1]
    
    def save(self, filepath_prefix):
        self.buy_model.save(f"{filepath_prefix}_buy.keras")
        self.sell_model.save(f"{filepath_prefix}_sell.keras")
        print(f"Saved: {filepath_prefix}_buy.keras and {filepath_prefix}_sell.keras")
    
    def load(self, filepath_prefix):
        self.buy_model = keras.models.load_model(f"{filepath_prefix}_buy.keras")
        self.sell_model = keras.models.load_model(f"{filepath_prefix}_sell.keras")
        self.update_target_models()
        print(f"Loaded: {filepath_prefix}_buy.keras and {filepath_prefix}_sell.keras")


def train_dual_dqn(stock_data_dict, state_size, episodes=50, num_envs=16, agent=None, 
                   buffer_size=None, buy_probability=0.3, batch_size=256, num_batches=10,
                   initial_balance=1000000.0, median_hold=10):
    """Train dual DQN agent on multiple stocks"""
    
    print("Dual DQN Training | Episodes:", episodes, "| Envs:", num_envs)
    print(f"Initial balance: ${initial_balance:,.0f}")
    print(f"Stocks: {len(stock_data_dict)}")
    print(f"Median hold (random trades): {median_hold} days")
    
    if agent is None:
        if buffer_size is None:
            buffer_size = 10000
        agent = DualDQNAgent(state_size, buffer_size=buffer_size, 
                            buy_probability=buy_probability, num_episodes=episodes)
        print(f"New agent | Buffer size: {buffer_size}")
    else:
        print(f"Continuing | Buy exp: {len(agent.buy_memory)} | Sell exp: {len(agent.sell_memory)}")
    
    transaction_cost = 0.00009  # 0.009%
    holding_cost_per_day = 0.0001875  # 0.01875%
    
    # Log-normal parameters for holding duration
    mu = np.log(median_hold)
    sigma = 0.5
    
    episode_rewards = []
    episode_values = []
    
    stock_symbols = list(stock_data_dict.keys())
    
    for episode in range(episodes):
        episode_reward = 0
        trades_collected = 0
        all_final_values = []
        
        # Train on each stock separately
        for symbol in stock_symbols:
            train_data = stock_data_dict[symbol]
            
            balances = np.full(num_envs, initial_balance, dtype=np.float32)
            shares = np.zeros(num_envs, dtype=np.int32)
            positions = np.zeros(num_envs, dtype=np.int32)
            steps = np.zeros(num_envs, dtype=np.int32)
            active = np.ones(num_envs, dtype=bool)
            
            buy_states = [None] * num_envs
            buy_prices = np.zeros(num_envs, dtype=np.float32)
            buy_values = np.zeros(num_envs, dtype=np.float32)
            buy_steps = np.zeros(num_envs, dtype=np.int32)
            target_hold_durations = np.zeros(num_envs, dtype=np.int32)  # For random trades
            
            while active.any():
                states = np.zeros((active.sum(), state_size), dtype=np.float32)
                
                for i, env_idx in enumerate(np.where(active)[0]):
                    step = steps[env_idx]
                    features = train_data[step, 1:]
                    portfolio = np.array([
                        positions[env_idx],
                        shares[env_idx] * train_data[step, 0] / initial_balance if positions[env_idx] else 0,
                        balances[env_idx] / initial_balance
                    ], dtype=np.float32)
                    states[i] = np.concatenate([features, portfolio])
                
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
                        # Determine if buy (model or random)
                        is_random_action = np.random.rand() <= agent.epsilon
                        if is_random_action:
                            should_buy = np.random.rand() < agent.buy_probability
                        else:
                            should_buy = agent.should_buy(state)
                        
                        if should_buy and step + 1 < len(train_data):
                            shares[env_idx] = int(balances[env_idx] / (execution_price * (1 + transaction_cost)))
                            cost = shares[env_idx] * execution_price * (1 + transaction_cost)
                            balances[env_idx] -= cost
                            positions[env_idx] = 1
                            
                            buy_states[env_idx] = state.copy()
                            buy_prices[env_idx] = execution_price
                            buy_values[env_idx] = balances[env_idx] + shares[env_idx] * execution_price
                            buy_steps[env_idx] = step
                            
                            # Sample holding duration for random trades
                            if is_random_action:
                                target_hold_durations[env_idx] = max(1, int(np.random.lognormal(mu, sigma)))
                            else:
                                target_hold_durations[env_idx] = 0  # Model decides when to sell
                    
                    else:
                        # Currently holding - decide whether to sell
                        hold_duration = step - buy_steps[env_idx]
                        
                        # For random trades, check if holding duration reached
                        if target_hold_durations[env_idx] > 0:
                            # This was a random buy, use log-normal duration
                            should_sell = hold_duration >= target_hold_durations[env_idx]
                        else:
                            # Model-based decision
                            should_sell = agent.should_sell(state)
                        
                        if should_sell and step + 1 < len(train_data):
                            proceeds = shares[env_idx] * execution_price * (1 - transaction_cost)
                            balances[env_idx] += proceeds
                            
                            sell_value = balances[env_idx]
                            buy_value = buy_values[env_idx]
                            base_reward = (sell_value - buy_value) / buy_value
                            
                            holding_cost = hold_duration * holding_cost_per_day
                            reward = base_reward - holding_cost
                            
                            next_features = train_data[min(step + 1, len(train_data) - 1), 1:]
                            next_portfolio = np.array([0, 0, balances[env_idx] / initial_balance], 
                                                     dtype=np.float32)
                            next_state = np.concatenate([next_features, next_portfolio])
                            
                            agent.remember_buy_experience(buy_states[env_idx], 0, reward, next_state, False)
                            agent.remember_sell_experience(state, 0, reward, next_state, False)
                            
                            trades_collected += 1
                            episode_reward += reward
                            
                            shares[env_idx] = 0
                            positions[env_idx] = 0
                            buy_states[env_idx] = None
                            target_hold_durations[env_idx] = 0
                        
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
            
            # Calculate final values for this stock
            final_vals = balances + shares * train_data[-1, 0]
            all_final_values.extend(final_vals.tolist())
        
        # Train models after each episode
        buy_loss = None
        sell_loss = None
        
        if len(agent.buy_memory) >= batch_size:
            buy_loss = agent.train_buy_model(batch_size=batch_size, num_batches=num_batches)
        
        if len(agent.sell_memory) >= batch_size:
            sell_loss = agent.train_sell_model(batch_size=batch_size, num_batches=num_batches)
        
        if episode % 3 == 0:
            agent.update_target_models()
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Calculate average final value across all stocks and envs
        avg_val = np.mean(all_final_values) if all_final_values else initial_balance
        
        episode_rewards.append(episode_reward / max(trades_collected, 1))
        episode_values.append(avg_val)
        
        loss_str = ""
        if buy_loss is not None:
            loss_str += f"BuyLoss: {buy_loss:.4f} | "
        if sell_loss is not None:
            loss_str += f"SellLoss: {sell_loss:.4f} | "
        
        print(f"Ep {episode+1}/{episodes} | Trades: {trades_collected} | "
              f"AvgReward: {episode_rewards[-1]:+.3f} | {loss_str}"
              f"Value: ${avg_val:,.0f} | ε: {agent.epsilon:.3f}")
    
    return {
        'agent': agent,
        'rewards': episode_rewards,
        'values': episode_values
    }


def test_dual_dqn(stock_data_dict, agent, initial_balance=1000000.0, max_stocks=10):
    """Test dual DQN agent with multi-stock portfolio"""
    agent.epsilon = 0
    
    transaction_cost = 0.00009  # 0.009%
    holding_cost_per_day = 0.0001875  # 0.01875%
    
    # Get all stock symbols and their data
    stock_symbols = list(stock_data_dict.keys())
    stock_data = stock_data_dict
    
    # Find the maximum length among all stocks
    max_len = max(len(data) for data in stock_data.values())
    
    # Portfolio state
    balance = initial_balance
    positions = {}  # symbol -> (shares, buy_price, buy_step)
    
    portfolio_values = [initial_balance]
    trades = []
    
    for step in range(max_len - 1):
        # Current portfolio value
        portfolio_value = balance
        for symbol, (shares, buy_price, buy_step) in positions.items():
            if step < len(stock_data[symbol]):
                current_price = stock_data[symbol][step, 0]
                hold_duration = step - buy_step
                holding_cost = shares * current_price * holding_cost_per_day * hold_duration
                portfolio_value += shares * current_price - holding_cost
        
        # Process sell decisions first
        symbols_to_sell = []
        for symbol in list(positions.keys()):
            if step >= len(stock_data[symbol]):
                continue
                
            shares, buy_price, buy_step = positions[symbol]
            features = stock_data[symbol][step, 1:]
            current_price = stock_data[symbol][step, 0]
            
            portfolio_state = np.array([
                1,
                shares * current_price / initial_balance,
                balance / initial_balance
            ], dtype=np.float32)
            state = np.concatenate([features, portfolio_state])
            
            should_sell = agent.should_sell(state)
            
            if should_sell and step + 1 < len(stock_data[symbol]):
                execution_price = stock_data[symbol][step + 1, 0]
                hold_duration = step - buy_step
                holding_cost_total = shares * execution_price * holding_cost_per_day * hold_duration
                proceeds = shares * execution_price * (1 - transaction_cost) - holding_cost_total
                balance += proceeds
                
                trades.append(f"SELL {symbol} at step {step}: {shares} shares @ ${execution_price:.2f}, "
                            f"held {hold_duration} days, proceeds ${proceeds:.2f}")
                symbols_to_sell.append(symbol)
        
        for symbol in symbols_to_sell:
            del positions[symbol]
        
        # Process buy decisions
        num_stocks_held = len(positions)
        if num_stocks_held < max_stocks:
            for symbol in stock_symbols:
                if symbol in positions or step >= len(stock_data[symbol]):
                    continue
                
                features = stock_data[symbol][step, 1:]
                portfolio_state = np.array([
                    0,
                    0,
                    balance / initial_balance
                ], dtype=np.float32)
                state = np.concatenate([features, portfolio_state])
                
                should_buy = agent.should_buy(state)
                
                if should_buy and step + 1 < len(stock_data[symbol]):
                    num_not_bought = max_stocks - num_stocks_held
                    allocation = balance / num_not_bought if num_not_bought > 0 else balance
                    
                    execution_price = stock_data[symbol][step + 1, 0]
                    shares = int(allocation / (execution_price * (1 + transaction_cost)))
                    
                    if shares > 0:
                        cost = shares * execution_price * (1 + transaction_cost)
                        balance -= cost
                        positions[symbol] = (shares, execution_price, step)
                        
                        trades.append(f"BUY {symbol} at step {step}: {shares} shares @ ${execution_price:.2f}, "
                                    f"cost ${cost:.2f}")
                        num_stocks_held += 1
                        
                        if num_stocks_held >= max_stocks:
                            break
        
        # Update portfolio value for next step
        next_value = balance
        for symbol, (shares, buy_price, buy_step) in positions.items():
            if step + 1 < len(stock_data[symbol]):
                next_price = stock_data[symbol][step + 1, 0]
                hold_duration = step + 1 - buy_step
                holding_cost = shares * next_price * holding_cost_per_day * hold_duration
                next_value += shares * next_price - holding_cost
        
        portfolio_values.append(next_value)
    
    return {
        'portfolio_values': portfolio_values,
        'trades': trades
    }


if __name__ == "__main__":
    print("Dual DQN Module (Separate Buy and Sell Models)")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {len(tf.config.list_physical_devices('GPU'))}")
