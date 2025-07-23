# Hybrid CNN-LSTM Stock Prediction Model
# Load required libraries
library(keras3)
library(tensorflow)
library(tidyverse)
library(TTR)
library(lubridate)
library(ggplot2)
library(caret)
library(zoo)

# Set random seed for reproducibility
set.seed(42)
tf$random$set_seed(42)

# 1. SIMULATE DATA
simulate_stock_data <- function(n_days = 1000, n_stocks = 20, n_industries = 5) {
  dates <- seq(as.Date("2020-01-01"), length.out = n_days, by = "day")

  # Industry assignments
  industries <- rep(paste0("INDUSTRY_", 1:n_industries), length.out = n_stocks)
  stock_names <- paste0("STOCK_", 1:n_stocks)

  # Initialize data list
  all_data <- list()

  # Simulate index data first
  index_name <- "INDEX_1"
  base_price <- 1000
  returns <- rnorm(n_days, mean = 0.0003, sd = 0.015)
  prices <- base_price * cumprod(1 + returns)

  all_data[[index_name]] <- data.frame(
    date = dates,
    symbol = index_name,
    type = "index",
    industry = NA,
    open = prices * runif(n_days, 0.99, 1.01),
    high = prices * runif(n_days, 1.01, 1.03),
    low = prices * runif(n_days, 0.97, 0.99),
    close = prices,
    vol = round(runif(n_days, 1e8, 5e8)),
    mktcap = NA,
    pe = NA
  )

  # Simulate stock data
  for (i in 1:n_stocks) {
    # Industry-specific parameters
    industry_idx <- ceiling(i / (n_stocks / n_industries))
    industry_beta <- 0.8 + (industry_idx - 1) * 0.1

    # Correlate with first index
    index_returns <- diff(log(all_data[["INDEX_1"]]$close))
    stock_returns <- industry_beta * index_returns + rnorm(n_days - 1, 0, 0.02)
    stock_returns <- c(0, stock_returns)  # Add initial return

    base_price <- runif(1, 20, 200)
    prices <- base_price * cumprod(1 + stock_returns)

    # Generate OHLCV data
    opens <- prices * runif(n_days, 0.99, 1.01)
    highs <- pmax(opens, prices) * runif(n_days, 1.0, 1.02)
    lows <- pmin(opens, prices) * runif(n_days, 0.98, 1.0)
    vols <- round(runif(n_days, 1e6, 1e7) * (1 + sin(1:n_days / 20) * 0.3))

    # Market cap and P/E ratio
    shares_outstanding <- runif(1, 1e8, 1e9)
    mktcaps <- prices * shares_outstanding
    pes <- runif(n_days, 10, 30) + industry_idx * 2 + sin(1:n_days / 100) * 5

    all_data[[stock_names[i]]] <- data.frame(
      date = dates,
      symbol = stock_names[i],
      type = "stock",
      industry = industries[i],
      open = opens,
      high = highs,
      low = lows,
      close = prices,
      vol = vols,
      mktcap = mktcaps,
      pe = pes
    )
  }

  # Combine all data
  combined_data <- bind_rows(all_data)
  return(combined_data)
}

# 2. FEATURE ENGINEERING
calculate_technical_indicators <- function(data) {
  data <- data %>%
    arrange(symbol, date) %>%
    group_by(symbol) %>%
    mutate(
      # Price-based features
      returns = (close - lag(close)) / lag(close),
      intraday_range = (high - low) / close,
      close_position = (close - low) / (high - low + 1e-8),

      # Volume features
      vol_ma20 = rollapply(vol, 20, mean, fill = NA, align = "right"),
      vol_ratio = vol / (vol_ma20 + 1e-8),

      # Price momentum
      returns_5d = (close - lag(close, 5)) / lag(close, 5),
      returns_20d = (close - lag(close, 20)) / lag(close, 20),

      # Technical indicators
      rsi = RSI(close, n = 14),

      # ATR
      atr = ATR(HLC = cbind(high, low, close), n = 14)[, "atr"] / close,

      # Bollinger Bands
      bb = BBands(close, n = 20, sd = 2),
      bb_position = (close - bb[, "mavg"]) / (bb[, "up"] - bb[, "dn"] + 1e-8),

      # Target: 5-day forward return
      target = (lead(close, 5) - close) / close
    ) %>%
    ungroup()

  # MACD
  for (sym in unique(data$symbol)) {
    idx <- data$symbol == sym
    if (sum(idx) > 26) {
      macd_result <- MACD(data$close[idx], nFast = 12, nSlow = 26, nSig = 9)
      data$macd[idx] <- macd_result[, "macd"]
      data$macd_signal[idx] <- macd_result[, "signal"]
    }
  }

  data$macd_diff <- data$macd - data$macd_signal

  return(data)
}

# 3. CALCULATE SECTOR AND INDEX METRICS
calculate_relative_metrics <- function(data) {
  # Calculate daily sector medians
  sector_daily <- data %>%
    filter(type == "stock") %>%
    group_by(date, industry) %>%
    summarise(
      sector_return = median(returns, na.rm = TRUE),
      sector_pe_median = median(pe, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    # Handle cases where all values are NA
    mutate(
      sector_return = ifelse(is.na(sector_return), 0, sector_return),
      sector_pe_median = ifelse(is.na(sector_pe_median), 20, sector_pe_median)  # Default P/E
    )

  # Calculate index returns
  index_daily <- data %>%
    filter(type == "index") %>%
    select(date, symbol, returns) %>%
    pivot_wider(names_from = symbol, values_from = returns, names_prefix = "return_") %>%
    # Fill NA values with 0
    mutate(across(starts_with("return_"), ~ifelse(is.na(.x), 0, .x)))

  # Merge back with main data
  data <- data %>%
    left_join(sector_daily, by = c("date", "industry")) %>%
    left_join(index_daily, by = "date")

  # Calculate relative metrics
  data <- data %>%
    mutate(
      # P/E relative to sector
      pe_relative = ifelse(sector_pe_median == 0, 1, pe / sector_pe_median),

      # Market cap features
      mktcap_log = log(pmax(mktcap, 1)),  # Ensure positive values

      # Relative strength - handle division by zero
      rel_strength_index = ifelse(
        is.na(return_INDEX_1) | return_INDEX_1 == 0,
        1,
        ifelse(is.na(returns_5d), 0, returns_5d / return_INDEX_1)
      ),
      rel_strength_sector = ifelse(
        is.na(sector_return) | sector_return == 0,
        1,
        ifelse(is.na(returns_5d), 0, returns_5d / (sector_return * 5))
      )
    ) %>%
    group_by(symbol) %>%
    mutate(
      # Rolling beta and correlation (20-day)
      beta_index = rollapply(
        cbind(returns, return_INDEX_1), 20,
        function(x) {
          if (all(is.na(x)) || nrow(x) < 2) return(NA)
          x_clean <- na.omit(x)
          if (nrow(x_clean) < 2) return(NA)
          if (var(x_clean[,2]) == 0) return(NA)
          tryCatch({
            cov(x_clean[,1], x_clean[,2]) / var(x_clean[,2])
          }, error = function(e) NA)
        },
        by.column = FALSE, fill = NA, align = "right"
      ),

      corr_sector = rollapply(
        cbind(returns, sector_return), 20,
        function(x) {
          if (all(is.na(x)) || nrow(x) < 2) return(NA)
          x_clean <- na.omit(x)
          if (nrow(x_clean) < 2) return(NA)
          if (sd(x_clean[,1]) == 0 || sd(x_clean[,2]) == 0) return(NA)
          tryCatch({
            cor(x_clean[,1], x_clean[,2])
          }, error = function(e) NA)
        },
        by.column = FALSE, fill = NA, align = "right"
      ),

      # Market cap percentile within sector
      mktcap_pct = percent_rank(mktcap)
    ) %>%
    ungroup()

  return(data)
}

# 4. PREPARE DATA FOR NEURAL NETWORK
prepare_nn_data <- function(data, lookback = 30) {
  # Select features for modeling
  feature_cols <- c(
    "returns", "intraday_range", "close_position", "vol_ratio",
    "returns_5d", "returns_20d", "rsi", "atr", "bb_position",
    "macd_diff", "pe_relative", "mktcap_log", "mktcap_pct",
    "rel_strength_index", "rel_strength_sector", "beta_index", "corr_sector"
  )

  # Normalize features using rolling z-score
  data_norm <- data %>%
    filter(type == "stock") %>%
    arrange(symbol, date) %>%
    group_by(symbol) %>%
    mutate(across(all_of(feature_cols),
                  ~{
                    rolling_mean <- rollapply(.x, 20, mean, na.rm = TRUE, fill = NA, align = "right")
                    rolling_sd <- rollapply(.x, 20, sd, na.rm = TRUE, fill = NA, align = "right")
                    # Handle zero standard deviation
                    rolling_sd <- ifelse(rolling_sd == 0 | is.na(rolling_sd), 1, rolling_sd)
                    (.x - rolling_mean) / rolling_sd
                  },
                  .names = "{col}_norm")) %>%
    ungroup()

  # Winsorize at 3 standard deviations and handle NA values
  feature_cols_norm <- paste0(feature_cols, "_norm")
  data_norm[feature_cols_norm] <- lapply(data_norm[feature_cols_norm],
                                          function(x) {
                                            x[is.na(x)] <- 0  # Replace NA with 0
                                            pmax(pmin(x, 3), -3)
                                          })

  # Create sequences
  X_list <- list()
  y_list <- list()

  symbols <- unique(data_norm$symbol)

  for (sym in symbols) {
    sym_data <- data_norm %>% filter(symbol == sym)

    if (nrow(sym_data) < lookback + 5) next

    features <- as.matrix(sym_data[, feature_cols_norm])
    targets <- sym_data$target

    # Create sequences
    for (i in (lookback + 1):(nrow(sym_data) - 5)) {
      if (!any(is.na(features[(i - lookback + 1):i, ])) && !is.na(targets[i])) {
        X_list[[length(X_list) + 1]] <- features[(i - lookback + 1):i, ]
        y_list[[length(y_list) + 1]] <- targets[i]
      }
    }
  }

  # Convert to arrays
  X <- array(unlist(X_list), dim = c(length(X_list), lookback, length(feature_cols)))
  y <- unlist(y_list)

  return(list(X = X, y = y, feature_names = feature_cols))
}

# 5. BUILD HYBRID CNN-LSTM MODEL
build_hybrid_model <- function(input_shape) {
  inputs <- layer_input(shape = input_shape)

  # Alternative 1: 1D CNN approach (treating time series as 1D)
  cnn_features <- inputs %>%
    layer_conv_1d(filters = 64, kernel_size = 3, activation = "relu", padding = "same") %>%
    layer_batch_normalization() %>%
    layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu", padding = "same") %>%
    layer_batch_normalization()

  # LSTM layers
  lstm_features <- cnn_features %>%
    layer_lstm(units = 128, return_sequences = TRUE, dropout = 0.2) %>%
    layer_lstm(units = 64, return_sequences = TRUE, dropout = 0.2) %>%
    layer_lstm(units = 32, dropout = 0.2)

  # Output layer
  outputs <- lstm_features %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 1)

  model <- keras_model(inputs = inputs, outputs = outputs)

  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "mse",
    metrics = list("mae")
  )

  return(model)
}

# MAIN EXECUTION
cat("1. Simulating stock data...\n")
raw_data <- simulate_stock_data(n_days = 1000, n_stocks = 20, n_industries = 5)

cat("2. Calculating technical indicators...\n")
data_with_indicators <- calculate_technical_indicators(raw_data)

cat("3. Calculating relative metrics...\n")
data_complete <- calculate_relative_metrics(data_with_indicators)

cat("4. Preparing data for neural network...\n")
nn_data <- prepare_nn_data(data_complete, lookback = 30)

# Split data
n_samples <- dim(nn_data$X)[1]
train_size <- floor(0.7 * n_samples)
val_size <- floor(0.15 * n_samples)

train_idx <- 1:train_size
val_idx <- (train_size + 1):(train_size + val_size)
test_idx <- (train_size + val_size + 1):n_samples

X_train <- nn_data$X[train_idx, , ]
y_train <- nn_data$y[train_idx]
X_val <- nn_data$X[val_idx, , ]
y_val <- nn_data$y[val_idx]
X_test <- nn_data$X[test_idx, , ]
y_test <- nn_data$y[test_idx]

cat("\nData shapes:\n")
cat("X_train:", dim(X_train), "\n")
cat("X_val:", dim(X_val), "\n")
cat("X_test:", dim(X_test), "\n")

# 5. BUILD AND TRAIN MODEL
cat("\n5. Building hybrid CNN-LSTM model...\n")
model <- build_hybrid_model(c(dim(X_train)[2], dim(X_train)[3]))

# Display model summary
summary(model)

# Define callbacks
checkpoint_callback <- callback_model_checkpoint(
  filepath = "model_checkpoint.keras",
  save_best_only = TRUE,
  monitor = "val_loss",
  mode = "min",
  verbose = 1
)

early_stopping <- callback_early_stopping(
  monitor = "val_loss",
  patience = 20,
  restore_best_weights = TRUE
)

reduce_lr <- callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.5,
  patience = 10,
  min_lr = 1e-6
)

# Train model
cat("\nTraining model...\n")
history <- model %>% fit(
  X_train, y_train,
  epochs = 1,
  batch_size = 32,
  validation_data = list(X_val, y_val),
  callbacks = list(checkpoint_callback, early_stopping, reduce_lr),
  verbose = 1
)

# 6. DISPLAY TRAINING HISTORY
plot(history)

# Additional custom plot
history_df <- data.frame(
  epoch = 1:length(history$metrics$loss),
  train_loss = history$metrics$loss,
  val_loss = history$metrics$val_loss,
  train_mae = history$metrics$mae,
  val_mae = history$metrics$val_mae
)

p1 <- ggplot(history_df, aes(x = epoch)) +
  geom_line(aes(y = train_loss, color = "Training Loss")) +
  geom_line(aes(y = val_loss, color = "Validation Loss")) +
  scale_color_manual(values = c("Training Loss" = "blue", "Validation Loss" = "red")) +
  labs(title = "Model Loss Over Epochs", y = "Loss", color = "Metric") +
  theme_minimal()

print(p1)

# Make predictions for confusion matrix
y_pred <- model %>% predict(X_test)
y_pred <- as.vector(y_pred)

# Create classification bins for confusion matrix
create_return_bins <- function(returns) {
  breaks <- c(-Inf, -0.02, -0.005, 0.005, 0.02, Inf)
  labels <- c("Large Loss", "Small Loss", "Neutral", "Small Gain", "Large Gain")
  cut(returns, breaks = breaks, labels = labels)
}

y_test_binned <- create_return_bins(y_test)
y_pred_binned <- create_return_bins(y_pred)

# Display confusion matrix
conf_matrix <- confusionMatrix(y_pred_binned, y_test_binned)
print(conf_matrix)

# Visualize confusion matrix
conf_matrix_df <- as.data.frame(conf_matrix$table)
p2 <- ggplot(conf_matrix_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "darkblue") +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p2)

# 7. SAVE TRAINED MODEL
cat("\n7. Saving trained model...\n")
save_model(model, "final_stock_model.keras")
cat("Model saved to 'final_stock_model.keras'\n")

# EXAMPLE: HOW TO RESTART TRAINING IF TERMINATED
cat("\n--- Example: Restarting training from checkpoint ---\n")
cat("
# Load the checkpoint
restored_model <- load_model_keras('model_checkpoint.keras')

# Continue training
history_continued <- restored_model %>% fit(
  X_train, y_train,
  epochs = 20,  # Additional epochs
  batch_size = 32,
  validation_data = list(X_val, y_val),
  initial_epoch = length(history$metrics$loss),  # Start from where we left off
  callbacks = list(checkpoint_callback, early_stopping, reduce_lr)
)
")

# 8. EXAMPLE: PREDICTING NEW DATA
cat("\n8. Example: Predicting new data\n")

# Function to prepare new data for prediction
predict_new_data <- function(new_raw_data, model, lookback = 30) {
  # Process the new data through the same pipeline
  new_data <- new_raw_data %>%
    calculate_technical_indicators() %>%
    calculate_relative_metrics()

  # Prepare for neural network
  nn_new <- prepare_nn_data(new_data, lookback = lookback)

  # Make predictions
  predictions <- model %>% predict(nn_new$X)

  # Create results dataframe
  results <- data.frame(
    predicted_5d_return = as.vector(predictions),
    return_category = create_return_bins(as.vector(predictions))
  )

  return(results)
}

# Example usage
cat("
# Load saved model
loaded_model <- load_model_keras('final_stock_model.keras')

# Prepare new data (same format as training data)
new_data <- simulate_stock_data(n_days = 100, n_stocks = 20, n_industries = 5)

# Make predictions
predictions <- predict_new_data(new_data, loaded_model)

# View predictions
head(predictions, 10)
")

# Create a small example
cat("\nDemonstrating prediction on last few samples:\n")
demo_predictions <- model %>% predict(X_test[1:5, , ])
demo_df <- data.frame(
  actual_return = y_test[1:5],
  predicted_return = as.vector(demo_predictions),
  return_category = create_return_bins(as.vector(demo_predictions))
)
print(demo_df)

cat("\n✓ Training complete! Check saved files:\n")
cat("  - model_checkpoint.keras (best model during training)\n")
cat("  - final_stock_model.keras (final trained model)\n")