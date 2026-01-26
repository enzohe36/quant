# Dual DQN Trading Agent Training

library(tensorflow)
library(reticulate)
library(data.table)

cat("Dual DQN Training\n")
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
initial_balance <- 1e6
max_test_stocks <- 10  # N parameter - adjustable
buy_probability <- 0.01
median_hold <- 10  # Median holding period for random trades (days)

cat(sprintf("\nTraining configuration:\n"))
cat(sprintf("  Initial balance: $%s\n", format(initial_balance, big.mark=",", scientific=FALSE)))
cat(sprintf("  Max test stocks: %d\n", max_test_stocks))
cat(sprintf("  Buy probability: %.1f%%\n", buy_probability * 100))
cat(sprintf("  Median hold (random): %d days\n\n", median_hold))

# Load data
cat("Loading training data from data/train.rds...\n")
train_dt <- readRDS("data/train.rds")
cat(sprintf("  Rows: %s\n", format(nrow(train_dt), big.mark=",")))
cat(sprintf("  Columns: %d\n", ncol(train_dt)))

cat("Loading test data from data/test.rds...\n")
test_dt <- readRDS("data/test.rds")
cat(sprintf("  Rows: %s\n", format(nrow(test_dt), big.mark=",")))
cat(sprintf("  Columns: %d\n\n", ncol(test_dt)))

# Get feature columns (all except symbol, date, price)
feature_cols <- setdiff(names(train_dt), c("symbol", "date", "price"))
cat(sprintf("Features: %d\n", length(feature_cols)))

# Convert to data.table if not already
setDT(train_dt)
setDT(test_dt)

# Get unique symbols
train_symbols <- unique(train_dt$symbol)
test_symbols <- unique(test_dt$symbol)
cat(sprintf("Training stocks: %d\n", length(train_symbols)))
cat(sprintf("Test stocks: %d\n\n", length(test_symbols)))

# Prepare data by stock
cat("Preparing stock data...\n")

prepare_stock_data <- function(dt, symbols) {
  stock_data_dict <- list()
  for (sym in symbols) {
    stock_dt <- dt[symbol == sym]
    if (nrow(stock_dt) > 0) {
      # Create matrix: first column is price, rest are features
      mat <- as.matrix(stock_dt[, c("price", ..feature_cols)])
      stock_data_dict[[sym]] <- mat
    }
  }
  return(stock_data_dict)
}

train_stock_dict <- prepare_stock_data(train_dt, train_symbols)
test_stock_dict <- prepare_stock_data(test_dt, test_symbols)

cat(sprintf("Training stocks prepared: %d\n", length(train_stock_dict)))
cat(sprintf("Test stocks prepared: %d\n\n", length(test_stock_dict)))

# State size: features + 3 portfolio features (position, shares_value_ratio, cash_ratio)
state_size <- length(feature_cols) + 3
cat(sprintf("State size: %d\n\n", state_size))

# Calculate hyperparameters
total_train_rows <- nrow(train_dt)
num_stocks <- length(train_stock_dict)

# Determine number of environments
num_envs <- 2^round(log2(2 * sqrt(total_train_rows / (100 * num_stocks))))
num_envs <- max(8, min(64, num_envs))

# Estimate trades per episode
avg_rows_per_stock <- total_train_rows / num_stocks
trades_per_env_per_stock <- avg_rows_per_stock * buy_probability / 10
trades_per_episode <- trades_per_env_per_stock * num_envs * num_stocks

# Episodes
episodes <- 10 * ceiling((4 * total_train_rows) / (num_envs * trades_per_env_per_stock * num_stocks) / 10)
episodes <- max(30, min(100, episodes))

# Buffer size
buffer_size <- 1000 * ceiling(25 * trades_per_episode / 1000)
buffer_size <- max(10000, min(100000, buffer_size))

# Batch parameters
batch_size <- 512
num_batches <- 10

cat(sprintf("Hyperparameters:\n"))
cat(sprintf("  Environments: %d\n", num_envs))
cat(sprintf("  Episodes: %d\n", episodes))
cat(sprintf("  Buffer size: %s\n", format(buffer_size, big.mark=",")))
cat(sprintf("  Batch size: %d\n", batch_size))
cat(sprintf("  Num batches: %d\n\n", num_batches))

# Train
cat("Starting training...\n")
cat(strrep("-", 80), "\n")

t0 <- Sys.time()
results <- py$train_dual_dqn(
  stock_data_dict = train_stock_dict,
  state_size = as.integer(state_size),
  episodes = as.integer(episodes),
  num_envs = as.integer(num_envs),
  buffer_size = as.integer(buffer_size),
  buy_probability = buy_probability,
  batch_size = as.integer(batch_size),
  num_batches = as.integer(num_batches),
  initial_balance = initial_balance,
  median_hold = as.integer(median_hold)
)
t1 <- Sys.time()

cat(strrep("-", 80), "\n")
training_time <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("Training complete (%.1f sec)\n", training_time))
cat(sprintf("Buy exp: %s | Sell exp: %s\n\n", 
            format(length(results$agent$buy_memory), big.mark=","),
            format(length(results$agent$sell_memory), big.mark=",")))

# Test
cat("Testing on unseen data...\n")
test_results <- py$test_dual_dqn(
  stock_data_dict = test_stock_dict,
  agent = results$agent,
  initial_balance = initial_balance,
  max_stocks = as.integer(max_test_stocks)
)

portfolio_vals <- unlist(test_results$portfolio_values)
final <- as.numeric(tail(portfolio_vals, 1))
dqn_return <- (final - initial_balance) / initial_balance

# Calculate buy-and-hold across all test stocks
buyhold_vals <- rep(initial_balance, length(portfolio_vals))
if (length(test_stock_dict) > 0) {
  # Equal weight buy-and-hold across all stocks
  allocation_per_stock <- initial_balance / length(test_stock_dict)
  buyhold_vals <- numeric(length(portfolio_vals))
  
  for (i in 1:length(portfolio_vals)) {
    total_val <- 0
    for (sym in names(test_stock_dict)) {
      stock_data <- test_stock_dict[[sym]]
      if (i <= nrow(stock_data)) {
        price_ratio <- stock_data[i, 1] / stock_data[1, 1]
        total_val <- total_val + allocation_per_stock * price_ratio
      } else {
        price_ratio <- stock_data[nrow(stock_data), 1] / stock_data[1, 1]
        total_val <- total_val + allocation_per_stock * price_ratio
      }
    }
    buyhold_vals[i] <- total_val
  }
}

buyhold_final <- tail(buyhold_vals, 1)
buyhold_return <- (buyhold_final - initial_balance) / initial_balance

trades <- unlist(test_results$trades)

cat(sprintf("Initial: $%s\n", format(initial_balance, big.mark=",", scientific=FALSE)))
cat(sprintf("Final:   $%s\n", format(round(final), big.mark=",", scientific=FALSE)))
cat(sprintf("DQN Return:      %+.2f%%\n", dqn_return * 100))
cat(sprintf("Buy-Hold Return: %+.2f%%\n", buyhold_return * 100))
cat(sprintf("Outperformance:  %+.2f%%\n", (dqn_return - buyhold_return) * 100))
cat(sprintf("Total trades: %d\n\n", length(trades)))

# Save trades
portfolio_k <- round(initial_balance / 1000)
trades_file <- sprintf("models/trades_%d.txt", portfolio_k)
cat(sprintf("Saving trades to %s...\n", trades_file))
writeLines(trades, trades_file)

# Plots
cat("Generating plots...\n")

par(mfrow = c(1, 3), mar = c(4, 4, 3, 2))

# Plot 1: Training rewards
training_rewards <- unlist(results$rewards)
plot(1:length(training_rewards), training_rewards, type = 'l', col = 'darkblue', lwd = 2,
     main = "Training: Avg Reward per Trade",
     xlab = "Episode", ylab = "Avg Reward")
abline(h = 0, col = 'gray', lty = 2)
grid()

# Plot 2: Training portfolio value
training_values <- unlist(results$values)
plot(1:length(training_values), training_values, type = 'l', col = 'darkgreen', lwd = 2,
     main = "Training: Portfolio Value",
     xlab = "Episode", ylab = "Value ($)")
abline(h = initial_balance, col = 'gray', lty = 2)
grid()

# Plot 3: Test performance
plot(1:length(portfolio_vals), portfolio_vals, type = 'l', col = 'blue', lwd = 2,
     main = "Test: DQN vs Buy-and-Hold",
     xlab = "Time Steps", ylab = "Portfolio Value ($)",
     ylim = c(min(c(portfolio_vals, buyhold_vals)) * 0.95, 
              max(c(portfolio_vals, buyhold_vals)) * 1.05))
lines(1:length(buyhold_vals), buyhold_vals, col = 'red', lwd = 2)
legend("topleft", legend = c("DQN", "Buy & Hold"), 
       col = c("blue", "red"), lty = 1, lwd = 2, bg = "white")
grid()

par(mfrow = c(1, 1))

# Save models
buy_path <- sprintf("models/buy_%d.keras", portfolio_k)
sell_path <- sprintf("models/sell_%d.keras", portfolio_k)

cat(sprintf("\nSaving models:\n"))
cat(sprintf("  %s\n", buy_path))
cat(sprintf("  %s\n", sell_path))

results$agent$buy_model$save(buy_path)
results$agent$sell_model$save(sell_path)

cat("\nDone!\n")
