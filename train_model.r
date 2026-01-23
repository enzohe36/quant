library(keras)
library(tensorflow)
library(quantmod)
library(TTR)
library(R6)

# Download and prepare data
getSymbols("AAPL", from = "2018-01-01", to = "2023-12-31")
prices <- Cl(AAPL)

# Create features
create_features <- function(prices) {
  returns_1 <- ROC(prices, n = 1)
  returns_5 <- ROC(prices, n = 5)
  returns_20 <- ROC(prices, n = 20)

  sma_5 <- SMA(prices, n = 5)
  sma_20 <- SMA(prices, n = 20)
  sma_50 <- SMA(prices, n = 50)

  rsi <- RSI(prices, n = 14)
  macd_obj <- MACD(prices, nFast = 12, nSlow = 26, nSig = 9)
  bb <- BBands(prices, n = 20, sd = 2)

  volatility <- runSD(returns_1, n = 20)

  features <- data.frame(
    price = as.numeric(prices),
    returns_1 = as.numeric(returns_1),
    returns_5 = as.numeric(returns_5),
    returns_20 = as.numeric(returns_20),
    sma_5_ratio = as.numeric(prices / sma_5),
    sma_20_ratio = as.numeric(prices / sma_20),
    sma_50_ratio = as.numeric(prices / sma_50),
    rsi = as.numeric(rsi),
    macd = as.numeric(macd_obj[, 1]),
    macd_signal = as.numeric(macd_obj[, 2]),
    bb_upper = as.numeric(bb[, "up"] / prices),
    bb_lower = as.numeric(bb[, "dn"] / prices),
    volatility = as.numeric(volatility)
  )

  return(na.omit(features))
}

features <- create_features(prices)

# Normalize features
normalize <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

for (col in names(features)) {
  if (col != "price") {
    features[[col]] <- normalize(features[[col]])
  }
}

# DQN Model Creation
create_dqn_model <- function(state_size, action_size, learning_rate = 0.001) {
  inputs <- layer_input(shape = c(state_size), name = "state_input")

  x <- inputs %>%
    layer_dense(units = 128, activation = 'relu', name = "dense_1") %>%
    layer_dropout(rate = 0.2, name = "dropout_1") %>%
    layer_dense(units = 64, activation = 'relu', name = "dense_2") %>%
    layer_dropout(rate = 0.2, name = "dropout_2") %>%
    layer_dense(units = 32, activation = 'relu', name = "dense_3")

  outputs <- x %>%
    layer_dense(units = action_size, activation = 'linear', name = "q_values")

  model <- keras_model(inputs = inputs, outputs = outputs)

  model$compile(
    optimizer = optimizer_adam(learning_rate = learning_rate),
    loss = 'mse'
  )

  return(model)
}

# Experience Replay Buffer
ReplayBuffer <- R6Class(
  "ReplayBuffer",
  public = list(
    capacity = NULL,
    buffer = NULL,
    position = 0,

    initialize = function(capacity = 10000) {
      self$capacity <- as.integer(capacity)
      self$buffer <- list()
    },

    push = function(state, action, reward, next_state, done) {
      experience <- list(
        state = state,
        action = action,
        reward = reward,
        next_state = next_state,
        done = done
      )

      if (length(self$buffer) < self$capacity) {
        self$buffer <- c(self$buffer, list(experience))
      } else {
        self$position <- (self$position %% self$capacity) + 1
        self$buffer[[self$position]] <- experience
      }
    },

    sample = function(batch_size) {
      batch_size <- as.integer(batch_size)
      indices <- sample(1:length(self$buffer), min(batch_size, length(self$buffer)))
      self$buffer[indices]
    },

    size = function() {
      length(self$buffer)
    }
  )
)

# CORRECTED: DQN Agent with fixed history access
DQNAgent <- R6Class(
  "DQNAgent",
  public = list(
    state_size = NULL,
    action_size = NULL,
    model = NULL,
    target_model = NULL,
    memory = NULL,
    gamma = 0.95,
    epsilon = 1.0,
    epsilon_min = 0.01,
    epsilon_decay = 0.995,
    learning_rate = 0.001,
    update_target_freq = 10,
    train_step = 0,

    initialize = function(state_size, action_size) {
      self$state_size <- as.integer(state_size)
      self$action_size <- as.integer(action_size)
      self$memory <- ReplayBuffer$new(capacity = 10000)
      self$model <- create_dqn_model(state_size, action_size, self$learning_rate)
      self$target_model <- create_dqn_model(state_size, action_size, self$learning_rate)
      self$update_target_network()
    },

    update_target_network = function() {
      self$target_model$set_weights(self$model$get_weights())
    },

    remember = function(state, action, reward, next_state, done) {
      self$memory$push(state, action, reward, next_state, done)
    },

    act = function(state) {
      if (runif(1) <= self$epsilon) {
        return(as.integer(sample(1:self$action_size, 1)))
      }

      state_matrix <- array(as.numeric(state), dim = c(1, length(state)))
      q_values <- self$model$predict(state_matrix, verbose = 0)
      return(as.integer(which.max(q_values[1, ])))
    },

    replay = function(batch_size = 32) {
      batch_size <- as.integer(batch_size)

      if (self$memory$size() < batch_size) {
        return(NULL)
      }

      minibatch <- self$memory$sample(batch_size)

      states <- do.call(rbind, lapply(minibatch, function(x) as.numeric(x$state)))
      next_states <- do.call(rbind, lapply(minibatch, function(x) as.numeric(x$next_state)))

      states <- array(as.numeric(states), dim = dim(states))
      next_states <- array(as.numeric(next_states), dim = dim(next_states))

      q_values <- self$model$predict(states, verbose = 0)
      next_q_values <- self$target_model$predict(next_states, verbose = 0)

      for (i in 1:length(minibatch)) {
        exp <- minibatch[[i]]
        target <- as.numeric(exp$reward)

        if (!exp$done) {
          target <- as.numeric(exp$reward) + self$gamma * max(next_q_values[i, ])
        }

        q_values[i, exp$action] <- target
      }

      q_values <- array(as.numeric(q_values), dim = dim(q_values))

      # CORRECTED: Access history correctly
      history <- self$model$fit(
        x = states,
        y = q_values,
        epochs = as.integer(1),
        verbose = 0,
        batch_size = batch_size
      )

      # CORRECTED: Access loss from history$history$loss
      loss <- NULL
      tryCatch({
        # Try different ways to access loss based on Keras version
        if (!is.null(history$history)) {
          loss <- tail(history$history$loss, 1)
        } else if (!is.null(history$metrics)) {
          loss <- history$metrics$loss
        }
      }, error = function(e) {
        loss <- NA
      })

      # Decay epsilon
      if (self$epsilon > self$epsilon_min) {
        self$epsilon <- self$epsilon * self$epsilon_decay
      }

      # Update target network periodically
      self$train_step <- self$train_step + 1
      if (self$train_step %% self$update_target_freq == 0) {
        self$update_target_network()
      }

      return(loss)
    },

    save_model = function(filepath) {
      self$model$save(filepath)
    },

    load_model = function(filepath) {
      self$model <- load_model_tf(filepath)
      self$update_target_network()
    }
  )
)

# Trading Environment
TradingEnvironment <- R6Class(
  "TradingEnvironment",
  public = list(
    data = NULL,
    current_step = 0,
    initial_balance = 10000,
    balance = 10000,
    shares = 0,
    position = 0,
    transaction_cost = 0.001,
    max_steps = NULL,

    initialize = function(data, initial_balance = 10000) {
      self$data <- data
      self$initial_balance <- initial_balance
      self$balance <- initial_balance
      self$max_steps <- as.integer(nrow(data) - 1)
    },

    reset = function() {
      self$current_step <- as.integer(1)
      self$balance <- self$initial_balance
      self$shares <- 0
      self$position <- 0
      return(self$get_state())
    },

    get_state = function() {
      current_features <- as.numeric(self$data[self$current_step, -1])
      portfolio_features <- c(
        self$position,
        self$shares * self$data$price[self$current_step] / self$initial_balance,
        self$balance / self$initial_balance
      )
      return(as.numeric(c(current_features, portfolio_features)))
    },

    step = function(action) {
      current_price <- self$data$price[self$current_step]
      next_price <- self$data$price[self$current_step + 1]

      reward <- 0

      if (action == 1 && self$position == 0) {
        # Buy
        self$shares <- floor(self$balance / (current_price * (1 + self$transaction_cost)))
        cost <- self$shares * current_price * (1 + self$transaction_cost)
        self$balance <- self$balance - cost
        self$position <- 1

      } else if (action == 2 && self$position == 1) {
        # Sell
        proceeds <- self$shares * current_price * (1 - self$transaction_cost)
        self$balance <- self$balance + proceeds
        self$shares <- 0
        self$position <- 0
      }

      self$current_step <- self$current_step + 1

      current_portfolio_value <- self$balance + self$shares * next_price
      previous_portfolio_value <- self$initial_balance

      if (self$current_step > 1) {
        previous_portfolio_value <- self$balance + self$shares * current_price
      }

      reward <- (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value

      done <- self$current_step >= self$max_steps
      next_state <- self$get_state()

      return(list(
        next_state = next_state,
        reward = as.numeric(reward),
        done = done,
        portfolio_value = current_portfolio_value
      ))
    },

    get_portfolio_value = function() {
      current_price <- self$data$price[self$current_step]
      return(self$balance + self$shares * current_price)
    }
  )
)

# Training function
train_dqn <- function(env, agent, episodes = 100, batch_size = 32) {
  episodes <- as.integer(episodes)
  batch_size <- as.integer(batch_size)
  portfolio_values <- list()

  for (episode in 1:episodes) {
    state <- env$reset()
    total_reward <- 0
    episode_values <- c()
    step_count <- 0

    while (TRUE) {
      action <- agent$act(state)
      result <- env$step(action)

      agent$remember(state, action, result$reward, result$next_state, result$done)

      state <- result$next_state
      total_reward <- total_reward + result$reward
      episode_values <- c(episode_values, result$portfolio_value)
      step_count <- step_count + 1

      if (result$done) {
        break
      }

      if (agent$memory$size() > batch_size) {
        agent$replay(batch_size)
      }
    }

    portfolio_values[[episode]] <- episode_values
    final_value <- tail(episode_values, 1)

    cat(sprintf("Episode: %d/%d, Steps: %d, Final Portfolio: $%.2f, Return: %.2f%%, Epsilon: %.3f\n",
                episode, episodes, step_count, final_value,
                (final_value - env$initial_balance) / env$initial_balance * 100,
                agent$epsilon))
  }

  return(portfolio_values)
}

# Test function
test_agent <- function(env, agent) {
  agent$epsilon <- 0
  state <- env$reset()
  portfolio_values <- c(env$initial_balance)
  actions_taken <- c()

  while (TRUE) {
    action <- agent$act(state)
    result <- env$step(action)

    actions_taken <- c(actions_taken, action)
    portfolio_values <- c(portfolio_values, result$portfolio_value)

    state <- result$next_state

    if (result$done) {
      break
    }
  }

  return(list(
    portfolio_values = portfolio_values,
    actions = actions_taken
  ))
}

# Main execution
cat("Preparing data...\n")
train_size <- floor(0.8 * nrow(features))
train_data <- features[1:train_size, ]
test_data <- features[(train_size + 1):nrow(features), ]

cat(sprintf("Training data: %d samples\n", nrow(train_data)))
cat(sprintf("Test data: %d samples\n", nrow(test_data)))

cat("\nInitializing environment and agent...\n")
state_size <- ncol(train_data) - 1 + 3
action_size <- 3

cat(sprintf("State size: %d\n", state_size))
cat(sprintf("Action size: %d\n", action_size))

env <- TradingEnvironment$new(train_data)
agent <- DQNAgent$new(state_size, action_size)

cat("\nStarting training...\n")
training_history <- train_dqn(env, agent, episodes = 50, batch_size = 32)

cat("\nSaving model...\n")
agent$save_model("dqn_trading_model")

cat("\nTesting on unseen data...\n")
test_env <- TradingEnvironment$new(test_data)
test_results <- test_agent(test_env, agent)

# Results
initial_value <- test_env$initial_balance
final_value <- tail(test_results$portfolio_values, 1)
total_return <- (final_value - initial_value) / initial_value

cat(sprintf("\n=== Test Results ===\n"))
cat(sprintf("Initial Portfolio Value: $%.2f\n", initial_value))
cat(sprintf("Final Portfolio Value: $%.2f\n", final_value))
cat(sprintf("Total Return: %.2f%%\n", total_return * 100))

buy_hold_return <- (test_data$price[nrow(test_data)] - test_data$price[1]) / test_data$price[1]
cat(sprintf("Buy-and-Hold Return: %.2f%%\n", buy_hold_return * 100))

# Action statistics
action_counts <- table(test_results$actions)
cat(sprintf("\nAction Distribution:\n"))
cat(sprintf("  Buy:  %d (%.1f%%)\n",
            ifelse(is.na(action_counts["1"]), 0, action_counts["1"]),
            ifelse(is.na(action_counts["1"]), 0, action_counts["1"]/length(test_results$actions)*100)))
cat(sprintf("  Sell: %d (%.1f%%)\n",
            ifelse(is.na(action_counts["2"]), 0, action_counts["2"]),
            ifelse(is.na(action_counts["2"]), 0, action_counts["2"]/length(test_results$actions)*100)))
cat(sprintf("  Hold: %d (%.1f%%)\n",
            ifelse(is.na(action_counts["3"]), 0, action_counts["3"]),
            ifelse(is.na(action_counts["3"]), 0, action_counts["3"]/length(test_results$actions)*100)))

# Plot results
par(mfrow = c(2, 1))

# Portfolio value over time
plot(test_results$portfolio_values, type = 'l', col = 'blue', lwd = 2,
     main = "DQN Trading Strategy - Portfolio Value",
     xlab = "Time Steps", ylab = "Portfolio Value ($)")
abline(h = initial_value, col = 'red', lty = 2)
legend("topleft", legend = c("DQN Strategy", "Initial Value"),
       col = c("blue", "red"), lty = c(1, 2), lwd = c(2, 1))

# Actions over time
plot(test_results$actions, type = 'p', pch = 20, col = 'darkgreen',
     main = "Trading Actions Over Time",
     xlab = "Time Steps", ylab = "Action (1=Buy, 2=Sell, 3=Hold)",
     ylim = c(0.5, 3.5))
abline(h = c(1, 2, 3), col = 'gray', lty = 3)

par(mfrow = c(1, 1))