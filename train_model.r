# Dual DQN Trading Agent Training
# Trains on multiple stocks with pattern analysis

library(tensorflow)
library(reticulate)
library(quantmod)
library(TTR)

cat("Dual DQN Training (Multiple Stocks)\n")
cat(strrep("=", 80), "\n\n")

# Load Python DQN module
cat("Loading DQN module...\n")
tf_config()
py_run_file("scripts/dqn.py")

# Create models directory
if (!dir.exists("models")) {
  dir.create("models")
  cat("Created models/ directory\n")
}

# Configuration
tickers <- c("AAPL", "GOOGL", "MSFT", "TSLA")
initial_balance <- 1000000
buy_probability <- 0.01
median_hold <- 20
short_wait_penalty <- 0.01

cat(sprintf("\nTraining configuration:\n"))
cat(sprintf("  Stocks: %s\n", paste(tickers, collapse=", ")))
cat(sprintf("  Initial balance: $%s\n", format(initial_balance, big.mark=",", scientific=FALSE)))
cat(sprintf("  Buy probability: %.1f%%\n", buy_probability * 100))
cat(sprintf("  Target hold: %d days\n", median_hold))
cat(sprintf("  Short wait penalty: %.2f per step\n\n", short_wait_penalty))

# Feature creation function
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
  
  data.frame(
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
  ) |> na.omit()
}

normalize <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

# Download and prepare all stocks
cat("Downloading stock data (2018-2023)...\n")
all_train_data <- list()
all_test_data <- list()

for (ticker in tickers) {
  cat(sprintf("  %s...", ticker))
  getSymbols(ticker, from = "2018-01-01", to = "2023-12-31", auto.assign = TRUE)
  prices <- Cl(get(ticker))
  
  features <- create_features(prices)
  
  for (col in names(features)) {
    if (col != "price") features[[col]] <- normalize(features[[col]])
  }
  
  train_size <- floor(0.8 * nrow(features))
  all_train_data[[ticker]] <- as.matrix(features[1:train_size, ])
  all_test_data[[ticker]] <- as.matrix(features[(train_size + 1):nrow(features), ])
  
  cat(sprintf(" %d train / %d test days\n", nrow(all_train_data[[ticker]]), nrow(all_test_data[[ticker]])))
}

cat("\n")

# Calculate total training days across all stocks
total_train_days <- sum(sapply(all_train_data, nrow))
cat(sprintf("Total training days: %d\n\n", total_train_days))

# State size
state_size <- ncol(all_train_data[[1]]) - 1 + 3

# Calculate hyperparameters
train_days <- total_train_days
calculated_envs <- 2 * sqrt(train_days / 100)
num_envs <- 2^round(log2(calculated_envs))
num_envs <- max(32, min(64, num_envs))  # Increased from 8-64 to 32-64

cat(sprintf("Environments: %d\n", num_envs))

mu <- log(median_hold)
avg_hold_duration <- exp(mu + 0.5^2 / 2)
estimated_trades_per_env <- (train_days * buy_probability) / avg_hold_duration
estimated_trades_per_episode <- estimated_trades_per_env * num_envs

min_episodes <- ceiling((3 * train_days) / (num_envs * estimated_trades_per_env))
max_episodes <- ceiling((5 * train_days) / (num_envs * estimated_trades_per_env))
optimal_episodes <- ceiling((min_episodes + max_episodes) / 2)
optimal_episodes <- max(30, min(100, optimal_episodes))

buffer_size <- 25 * estimated_trades_per_episode
buffer_size <- max(10000, min(50000, buffer_size))

cat(sprintf("Episodes: %d\n", optimal_episodes))
cat(sprintf("Buffer size: %d\n\n", buffer_size))

# Prepare training data as list of (name, data) tuples for Python
train_data_list <- list()
for (ticker in tickers) {
  train_data_list[[length(train_data_list) + 1]] <- tuple(ticker, all_train_data[[ticker]])
}

# Train
cat("Starting training...\n")
cat("Note: Incomplete trades at stock boundaries are discarded\n")
cat("Note: Buying within median_hold after selling incurs penalty\n")
cat(strrep("-", 80), "\n")

t0 <- Sys.time()
results <- py$train_dual_dqn(
  train_data_list = train_data_list,
  state_size = as.integer(state_size),
  episodes = as.integer(optimal_episodes),
  num_envs = as.integer(num_envs),
  buffer_size = as.integer(buffer_size),
  buy_probability = buy_probability,
  median_hold = as.integer(median_hold),
  initial_balance = initial_balance,
  short_wait_penalty = short_wait_penalty
)
t1 <- Sys.time()

cat(strrep("-", 80), "\n")
training_time <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("Training complete (%.1f sec)\n", training_time))
cat(sprintf("Buy exp: %d | Sell exp: %d\n\n", 
            length(results$agent$buy_memory), 
            length(results$agent$sell_memory)))

# Test on portfolio of all stocks
cat("Testing on multi-stock portfolio...\n")
cat(strrep("=", 80), "\n\n")

# Prepare test data dict
test_data_dict <- list()
for (ticker in tickers) {
  test_data_dict[[ticker]] <- all_test_data[[ticker]]
}

# Run portfolio test
test_results <- py$test_dual_dqn_portfolio(
  test_data_dict = test_data_dict,
  agent = results$agent,
  initial_balance = initial_balance
)

# Calculate buy-and-hold benchmark
buyhold_results <- py$calculate_buyhold_portfolio(
  test_data_dict = test_data_dict,
  initial_balance = initial_balance
)

# Calculate returns
portfolio_vals <- unlist(test_results$portfolio_values)
buyhold_vals <- unlist(buyhold_results$portfolio_values)

final_dqn <- as.numeric(tail(portfolio_vals, 1))
final_buyhold <- as.numeric(tail(buyhold_vals, 1))

dqn_return <- (final_dqn - initial_balance) / initial_balance
buyhold_return <- (final_buyhold - initial_balance) / initial_balance

cat("PORTFOLIO RESULTS:\n")
cat(strrep("-", 80), "\n")
cat(sprintf("Initial:     $%s\n", format(initial_balance, big.mark=",", scientific=FALSE)))
cat(sprintf("DQN Final:   $%s (%+.2f%%)\n", format(round(final_dqn), big.mark=",", scientific=FALSE), 
            dqn_return * 100))
cat(sprintf("B&H Final:   $%s (%+.2f%%)\n", format(round(final_buyhold), big.mark=",", scientific=FALSE),
            buyhold_return * 100))
cat(sprintf("Outperformance: %+.2f%%\n", (dqn_return - buyhold_return) * 100))
cat(strrep("-", 80), "\n\n")

# Pattern analysis on AAPL
cat("Running pattern analysis on AAPL...\n")
buy_patterns <- py$analyze_buy_patterns(results$agent, all_test_data[["AAPL"]], initial_balance)
sell_patterns <- py$analyze_sell_patterns(results$agent, all_test_data[["AAPL"]], initial_balance)
py$print_pattern_analysis(buy_patterns, sell_patterns)

# Plots
cat("Generating plots...\n\n")

# Plot 1: Training progress
training_values <- unlist(results$values)
plot(1:length(training_values), training_values, type = 'l', col = 'darkgreen', lwd = 2,
     main = "Training Progress: Portfolio Value",
     xlab = "Episode", ylab = "Portfolio Value ($)")
abline(h = initial_balance, col = 'gray', lty = 2)

# Plot 2: Portfolio performance comparison
test_steps <- 1:length(portfolio_vals)
plot(test_steps, portfolio_vals, type = 'l', col = 'blue', lwd = 2,
     main = "Portfolio Performance: DQN vs Buy-and-Hold",
     xlab = "Time Steps", ylab = "Portfolio Value ($)",
     ylim = c(min(portfolio_vals, buyhold_vals) * 0.95,
              max(portfolio_vals, buyhold_vals) * 1.05))
lines(test_steps, buyhold_vals, col = 'red', lwd = 2)
abline(h = initial_balance, col = 'gray', lty = 2)

legend("topleft",
       legend = c(
         sprintf("DQN: %+.2f%%", dqn_return * 100),
         sprintf("Buy & Hold: %+.2f%%", buyhold_return * 100),
         "Initial"
       ),
       col = c("blue", "red", "gray"),
       lty = c(1, 1, 2),
       lwd = c(2, 2, 1),
       bg = "white")

# Plot 3: Normalized comparison
portfolio_norm <- (portfolio_vals / portfolio_vals[1]) * 100
buyhold_norm <- (buyhold_vals / buyhold_vals[1]) * 100

plot(test_steps, portfolio_norm, type = 'l', col = 'blue', lwd = 2,
     main = "Portfolio Performance (Normalized to 100)",
     xlab = "Time Steps", ylab = "Normalized Value",
     ylim = c(min(portfolio_norm, buyhold_norm) * 0.98,
              max(portfolio_norm, buyhold_norm) * 1.02))
lines(test_steps, buyhold_norm, col = 'red', lwd = 2)
abline(h = 100, col = 'gray', lty = 2)

legend("topleft",
       legend = c("DQN Strategy", "Buy & Hold", "Starting Point"),
       col = c("blue", "red", "gray"),
       lty = c(1, 1, 2),
       lwd = c(2, 2, 1),
       bg = "white")

cat("Plots displayed\n\n")

# Save models
portfolio_k <- round(final_dqn / 1000)

buy_path <- sprintf("models/buy_%dk.keras", portfolio_k)
sell_path <- sprintf("models/sell_%dk.keras", portfolio_k)

cat(sprintf("Saving models:\n"))
cat(sprintf("  %s\n", buy_path))
cat(sprintf("  %s\n", sell_path))

results$agent$buy_model$save(buy_path)
results$agent$sell_model$save(sell_path)

cat("\nDone!\n")
