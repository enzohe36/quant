# Backtesting Script: Support + Volume Entry with Multi-Indicator Exit
# Using foreach and doFuture for Parallel Processing
# ======================================================================

library(TTR)        # For technical indicators
library(foreach)    # For parallel loops
library(doFuture)   # For parallel backend
library(future)     # For parallel processing setup
library(data.table) # For rbindlist
library(tidyverse)

set.seed(123)  # For reproducibility

# Helper Functions
# ================

#' Calculate Technical Indicators
calculate_indicators <- function(df) {
  df %>%
    arrange(date) %>%
    mutate(
      # Moving Averages
      ema_20 = EMA(close, n = 20),
      ema_50 = EMA(close, n = 50),
      ema_200 = EMA(close, n = 200),

      # RSI
      rsi = RSI(close, n = 14),

      # MACD
      macd_obj = MACD(close, nFast = 12, nSlow = 26, nSig = 9),
      macd = macd_obj[, 1],
      macd_signal = macd_obj[, 2],

      # ATR for volatility
      atr = ATR(cbind(high, low, close), n = 14)[, "atr"],

      # Volume indicators
      avg_volume_20 = SMA(volume, n = 20),
      volume_ratio = volume / avg_volume_20,

      # Support level (20-day low)
      support_20 = runMin(low, n = 20),

      # Rolling high for trailing stop
      rolling_high_20 = runMax(high, n = 20),

      # Bullish candle
      is_bullish_candle = close > open,

      # Price distance from support
      dist_from_support = (close - support_20) / support_20,

      # MACD momentum
      macd_rising = macd > lag(macd, 1),
      macd_positive = macd > 0
    ) %>%
    select(-macd_obj)  # Remove the MACD object column
}


#' Identify Entry Signals (Strategy 2: Support + Volume)
identify_entry_signals <- function(df) {
  df %>%
    mutate(
      # Entry conditions:
      # 1. Price near support (within 2% of 20-day low)
      near_support = dist_from_support < 0.02 & dist_from_support >= 0,

      # 2. Volume spike (1.5x average or more)
      volume_spike = volume_ratio >= 1.5,

      # 3. Bullish candlestick
      bullish_pattern = is_bullish_candle,

      # 4. Not already overbought
      not_overbought = rsi < 70 | is.na(rsi),

      # Combined entry signal
      entry_signal = near_support &
                     volume_spike &
                     bullish_pattern &
                     not_overbought &
                     !is.na(ema_50)  # Ensure indicators are calculated
    )
}


#' Identify Exit Signals (Multi-Indicator Confirmation)
identify_exit_signals <- function(df) {
  df %>%
    mutate(
      # Exit conditions:
      # 1. Price closes below 50 EMA
      below_ema_50 = close < ema_50,

      # 2. RSI overbought (>70)
      rsi_overbought = rsi > 70,

      # 3. MACD turns negative or crosses below signal
      macd_bearish = macd < 0 | macd < macd_signal,

      # 4. ATR-based trailing stop (2x ATR below 20-period high)
      atr_stop = rolling_high_20 - (2 * atr),
      price_below_atr_stop = close < atr_stop,

      # Combined exit signal (any condition triggers exit)
      exit_signal = below_ema_50 |
                    rsi_overbought |
                    macd_bearish |
                    price_below_atr_stop
    )
}


#' Simulate Trades
simulate_trades <- function(df, initial_capital = 10000, commission = 0.001) {

  # Initialize tracking variables
  in_position <- FALSE
  entry_price <- 0
  entry_date <- as.Date(NA)
  trades <- list()

  for (i in 1:nrow(df)) {
    row <- df[i, ]

    # Skip if indicators not ready
    if (is.na(row$ema_50) || is.na(row$ema_200)) next

    # Entry logic
    if (!in_position && row$entry_signal) {
      in_position <- TRUE
      entry_price <- row$close
      entry_date <- row$date
    }

    # Exit logic
    if (in_position && row$exit_signal) {
      exit_price <- row$close
      exit_date <- row$date

      # Calculate returns
      gross_return <- (exit_price - entry_price) / entry_price
      net_return <- gross_return - (2 * commission)  # Entry + exit commission

      trades[[length(trades) + 1]] <- list(
        entry_date = entry_date,
        entry_price = entry_price,
        exit_date = exit_date,
        exit_price = exit_price,
        gross_return = gross_return,
        net_return = net_return,
        holding_days = as.numeric(exit_date - entry_date)
      )

      in_position <- FALSE
    }
  }

  # Convert trades to data frame
  if (length(trades) == 0) {
    return(tibble(
      entry_date = as.Date(character()),
      entry_price = numeric(),
      exit_date = as.Date(character()),
      exit_price = numeric(),
      gross_return = numeric(),
      net_return = numeric(),
      holding_days = numeric()
    ))
  }

  bind_rows(trades)
}




#' Calculate Performance Metrics
calculate_performance <- function(trades_df) {

  if (nrow(trades_df) == 0) {
    return(tibble(
      total_trades = 0,
      winning_trades = 0,
      losing_trades = 0,
      win_rate = NA,
      avg_return = NA,
      avg_winning_return = NA,
      avg_losing_return = NA,
      total_return = NA,
      max_return = NA,
      min_return = NA,
      avg_holding_days = NA,
      profit_factor = NA
    ))
  }

  winning_trades <- trades_df %>% filter(net_return > 0)
  losing_trades <- trades_df %>% filter(net_return <= 0)

  tibble(
    total_trades = nrow(trades_df),
    winning_trades = nrow(winning_trades),
    losing_trades = nrow(losing_trades),
    win_rate = nrow(winning_trades) / nrow(trades_df),
    avg_return = mean(trades_df$net_return),
    avg_winning_return = ifelse(nrow(winning_trades) > 0,
                                 mean(winning_trades$net_return), NA),
    avg_losing_return = ifelse(nrow(losing_trades) > 0,
                                mean(losing_trades$net_return), NA),
    total_return = sum(trades_df$net_return),
    max_return = max(trades_df$net_return),
    min_return = min(trades_df$net_return),
    avg_holding_days = mean(trades_df$holding_days),
    profit_factor = ifelse(nrow(losing_trades) > 0,
                           sum(winning_trades$net_return) / abs(sum(losing_trades$net_return)),
                           NA)
  )
}


# Main Backtesting Function
# ==========================
stock_list_path <- "models/data_combined.rds"  # Path to your RDS file
initial_capital <- 10000
n_workers <- NULL  # Auto-detect
future_strategy <- "multisession"  # or "multicore" on Unix/Linux

# Read RDS file
cat("Reading stock data...\n")
stock_list <- readRDS(stock_list_path)[1:10]

# Get stock symbols
stock_symbols <- names(stock_list)
cat(sprintf("Loaded %d stocks\n", length(stock_symbols)))

# Set up parallel processing
if (is.null(n_workers)) {
  n_workers <- max(1, availableCores() - 1)
}
cat(sprintf("Setting up parallel processing with %d workers\n", n_workers))

# Configure future plan
# Options: "multisession", "multicore" (Unix only), "cluster"
plan(future_strategy, workers = n_workers)

cat("Running backtests in parallel...\n")

# Run backtests using foreach with %dofuture%
all_trades <- foreach(
  symbol = stock_symbols,
  .combine = "c"
) %dofuture% {
  df <- stock_list[[symbol]]

  # Calculate indicators
  df_with_indicators <- df %>%
    calculate_indicators() %>%
    identify_entry_signals() %>%
    identify_exit_signals()

  # Simulate trades
  trades <- simulate_trades(df_with_indicators, initial_capital)

  # Add stock symbol
  if (nrow(trades) > 0) {
    trades <- trades %>%
      mutate(symbol = symbol, .before = 1)
  } else {
    trades <- trades %>%
      mutate(symbol = character(), .before = 1)
  }

  return(trades)
} %>%
  rbindlist()

# Calculate overall performance
cat("\nCalculating performance metrics...\n")
overall_performance <- calculate_performance(all_trades)

# Calculate per-stock performance
per_stock_performance <- all_trades %>%
  group_by(symbol) %>%
  summarise(
    trades = n(),
    win_rate = mean(net_return > 0),
    avg_return = mean(net_return),
    total_return = sum(net_return),
    avg_holding_days = mean(holding_days),
    .groups = "drop"
  ) %>%
  arrange(desc(total_return))

# Return results
list(
  all_trades = all_trades,
  overall_performance = overall_performance,
  per_stock_performance = per_stock_performance
)

# View results
cat("\n=== OVERALL PERFORMANCE ===\n")
print(results$overall_performance)

cat("\n=== TOP 10 STOCKS BY TOTAL RETURN ===\n")
print(head(results$per_stock_performance, 10))

cat("\n=== BOTTOM 10 STOCKS BY TOTAL RETURN ===\n")
print(tail(results$per_stock_performance, 10))

# Sample of individual trades
cat("\n=== SAMPLE TRADES ===\n")
print(head(results$all_trades, 10))

# Save results
saveRDS(results, "backtest_results.rds")
write.csv(results$all_trades, "all_trades.csv", row.names = FALSE)
write.csv(results$per_stock_performance, "per_stock_performance.csv", row.names = FALSE)

cat("\nBacktest complete! Results saved.\n")

# Optional: Generate summary statistics by year
if (nrow(results$all_trades) > 0) {
  yearly_performance <- results$all_trades %>%
    mutate(year = format(entry_date, "%Y")) %>%
    group_by(year) %>%
    summarise(
      trades = n(),
      win_rate = mean(net_return > 0),
      avg_return = mean(net_return),
      total_return = sum(net_return),
      .groups = "drop"
    )

  cat("\n=== YEARLY PERFORMANCE ===\n")
  print(yearly_performance)
}

# Optional: Reset to sequential processing when done
# plan(sequential)