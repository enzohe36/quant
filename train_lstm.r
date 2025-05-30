# Multi-Layer LSTM Stock Prediction Model
# Predicts if max 10-day gain exceeds historical volatility

library(keras3)
library(tensorflow)
library(tidyverse)
library(gridExtra)
library(caret)

# Set random seed for reproducibility
set.seed(42)
tf$random$set_seed(42L)

# Parameters
n_stocks <- 2
n_days_sim <- 1200
n_days <- 240
seq_length <- 20
forecast_horizon <- 10
lookback_period <- 240

# 1. Simulate stock data
simulate_stock_data <- function(n_stocks, n_days_sim) {
  stocks_data <- map_dfr(1:n_stocks, function(stock_id) {
    # Generate realistic stock price movements
    initial_price <- runif(1, 20, 200)
    price_returns <- rnorm(n_days_sim, mean = 0.0005, sd = 0.02)

    # Add some autocorrelation to make it more realistic
    for(i in 2:length(price_returns)) {
      price_returns[i] <- 0.1 * price_returns[i-1] + 0.9 * price_returns[i]
    }

    prices <- initial_price * cumprod(1 + price_returns)

    # Volume correlated with price volatility
    base_volume <- runif(1, 100000, 1000000)
    volume_multiplier <- 1 + abs(price_returns) * 5
    volumes <- base_volume * volume_multiplier * (1 + rnorm(n_days_sim, 0, 0.3))
    volumes <- pmax(volumes, 1000)  # Minimum volume

    tibble(
      stock_id = stock_id,
      day = 1:n_days_sim,
      price = prices,
      volume = volumes,
      price_change_pct = c(0, diff(log(prices)) * 100),
      volume_change_pct = c(0, diff(log(volumes)) * 100)
    )
  })

  return(stocks_data)
}

cat("Simulating stock data...\n")
stock_data <- simulate_stock_data(n_stocks, n_days_sim)

# 2. Calculate target variable and sample weights
calculate_targets_and_weights <- function(data) {
  data %>%
    group_by(stock_id) %>%
    arrange(day) %>%
    mutate(
      # Calculate 10-day forward max gain
      future_max_gain = map_dbl(1:n(), function(i) {
        if(i > n() - forecast_horizon) return(NA_real_)
        future_prices <- price[(i+1):min(i+forecast_horizon, n())]
        current_price <- price[i]
        max_gain <- (max(future_prices) - current_price) / current_price * 100
        return(max_gain)
      }),

      # Calculate rolling 240-day std of 10-day gains for threshold
      rolling_std = map_dbl(1:n(), function(i) {
        if(i < lookback_period) return(NA_real_)

        # Calculate historical 10-day gains
        hist_gains <- map_dbl(max(1, i-lookback_period+1):(i-forecast_horizon), function(j) {
          if(j > n() - forecast_horizon) return(NA_real_)
          future_prices <- price[(j+1):min(j+forecast_horizon, n())]
          current_price <- price[j]
          gain <- (max(future_prices) - current_price) / current_price * 100
          return(gain)
        })

        hist_gains <- hist_gains[!is.na(hist_gains)]
        if(length(hist_gains) < 10) return(NA_real_)
        return(sd(hist_gains))
      }),

      # Binary target: does future max gain exceed historical std?
      target = ifelse(!is.na(future_max_gain) & !is.na(rolling_std),
                     as.numeric(future_max_gain > rolling_std),
                     NA_real_),

      # Sample weights based on max gain magnitude
      sample_weight = ifelse(!is.na(future_max_gain),
                            1 + pmax(future_max_gain, 0) / 10,
                            1)
    ) %>%
    ungroup()
}

cat("Calculating targets and weights...\n")
stock_data <- calculate_targets_and_weights(stock_data)

# Filter to keep only the last n days of data
cat("Filtering to last n days...\n")
stock_data <- stock_data %>%
  group_by(stock_id) %>%
  arrange(day) %>%
  slice_tail(n = n_days) %>%
  ungroup()

# 3. Create sequences for LSTM
create_sequences <- function(data, seq_length) {
  sequences <- data %>%
    dplyr::filter(!is.na(target)) %>%
    group_by(stock_id) %>%
    arrange(day) %>%
    dplyr::filter(n() >= seq_length + forecast_horizon) %>%
    do({
      stock_data <- .
      n_obs <- nrow(stock_data)

      # Create sequences
      seq_indices <- map(seq_length:(n_obs - forecast_horizon), function(i) {
        start_idx <- i - seq_length + 1
        list(
          price_seq = stock_data$price_change_pct[start_idx:i],
          volume_seq = stock_data$volume_change_pct[start_idx:i],
          target = stock_data$target[i],
          weight = stock_data$sample_weight[i],
          stock_id = stock_data$stock_id[1],
          day = stock_data$day[i]
        )
      })

      tibble(sequences = seq_indices)
    }) %>%
    ungroup()

  return(sequences$sequences)
}

cat("Creating sequences...\n")
all_sequences <- create_sequences(stock_data, seq_length)

# Remove sequences with NA values
valid_sequences <- keep(all_sequences, function(seq) {
  !any(is.na(c(seq$price_seq, seq$volume_seq, seq$target)))
})

cat(sprintf("Created %d valid sequences\n", length(valid_sequences)))

# 4. Split into train and test sets
split_point <- floor(length(valid_sequences) * 0.8)
train_sequences <- valid_sequences[1:split_point]
test_sequences <- valid_sequences[(split_point + 1):length(valid_sequences)]

# Prepare arrays for Keras
prepare_arrays <- function(sequences) {
  n_seq <- length(sequences)

  # Initialize arrays
  X_price <- array(0, dim = c(n_seq, seq_length, 1))
  X_volume <- array(0, dim = c(n_seq, seq_length, 1))
  y <- numeric(n_seq)
  weights <- numeric(n_seq)

  for(i in 1:n_seq) {
    X_price[i, , 1] <- sequences[[i]]$price_seq
    X_volume[i, , 1] <- sequences[[i]]$volume_seq
    y[i] <- sequences[[i]]$target
    weights[i] <- sequences[[i]]$weight
  }

  list(X_price = X_price, X_volume = X_volume, y = y, weights = weights)
}

cat("Preparing training arrays...\n")
train_data <- prepare_arrays(train_sequences)
cat("Preparing test arrays...\n")
test_data <- prepare_arrays(test_sequences)

# Normalize features
normalize_features <- function(train_X, test_X) {
  train_mean <- mean(train_X)
  train_sd <- sd(train_X)

  train_X_norm <- (train_X - train_mean) / train_sd
  test_X_norm <- (test_X - train_mean) / train_sd

  list(train = train_X_norm, test = test_X_norm, mean = train_mean, sd = train_sd)
}

price_norm <- normalize_features(train_data$X_price, test_data$X_price)
volume_norm <- normalize_features(train_data$X_volume, test_data$X_volume)

train_data$X_price <- price_norm$train
train_data$X_volume <- volume_norm$train
test_data$X_price <- price_norm$test
test_data$X_volume <- volume_norm$test

# 5. Build multi-layer LSTM model
build_lstm_model <- function(seq_length) {
  # Price input branch
  price_input <- layer_input(shape = c(seq_length, 1), name = "price_input")
  price_lstm1 <- price_input %>%
    layer_lstm(units = 64, return_sequences = TRUE, dropout = 0.2, recurrent_dropout = 0.2) %>%
    layer_lstm(units = 32, return_sequences = FALSE, dropout = 0.2, recurrent_dropout = 0.2)

  # Volume input branch
  volume_input <- layer_input(shape = c(seq_length, 1), name = "volume_input")
  volume_lstm1 <- volume_input %>%
    layer_lstm(units = 64, return_sequences = TRUE, dropout = 0.2, recurrent_dropout = 0.2) %>%
    layer_lstm(units = 32, return_sequences = FALSE, dropout = 0.2, recurrent_dropout = 0.2)

  # Combine branches
  combined <- layer_concatenate(list(price_lstm1, volume_lstm1))

  # Additional dense layers
  output <- combined %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 1, activation = "sigmoid")

  # Create model
  model <- keras_model(inputs = list(price_input, volume_input), outputs = output)

  # Compile model
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "binary_crossentropy",
    metrics = c("accuracy", "precision", "recall")
  )

  return(model)
}

cat("Building LSTM model...\n")
model <- build_lstm_model(seq_length)
summary(model)

# 6. Train the model with sample weights
cat("Training model...\n")
history <- model %>% fit(
  x = list(train_data$X_price, train_data$X_volume),
  y = train_data$y,
  sample_weight = train_data$weights,
  epochs = 50,
  batch_size = 64,
  validation_split = 0.2,
  callbacks = list(
    callback_early_stopping(patience = 10, restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(patience = 5, factor = 0.7)
  ),
  verbose = 1
)

# 7. Evaluate on test set
cat("Evaluating on test set...\n")
test_predictions <- model %>% predict(list(test_data$X_price, test_data$X_volume))
test_pred_binary <- as.numeric(test_predictions > 0.5)

# Calculate performance metrics
test_accuracy <- mean(test_pred_binary == test_data$y)
test_precision <- sum(test_pred_binary == 1 & test_data$y == 1) / sum(test_pred_binary == 1)
test_recall <- sum(test_pred_binary == 1 & test_data$y == 1) / sum(test_data$y == 1)
test_f1 <- 2 * test_precision * test_recall / (test_precision + test_recall)

# Results summary
cat("\n=== MODEL PERFORMANCE SUMMARY ===\n")
cat(sprintf("Training samples: %d\n", length(train_sequences)))
cat(sprintf("Test samples: %d\n", length(test_sequences)))
cat(sprintf("Sequence length: %d days\n", seq_length))
cat(sprintf("Forecast horizon: %d days\n", forecast_horizon))
cat(sprintf("\nTest Accuracy: %.4f\n", test_accuracy))
cat(sprintf("Test Precision: %.4f\n", test_precision))
cat(sprintf("Test Recall: %.4f\n", test_recall))
cat(sprintf("Test F1-Score: %.4f\n", test_f1))

# Class distribution
pos_rate_train <- mean(train_data$y)
pos_rate_test <- mean(test_data$y)
cat(sprintf("\nPositive class rate (train): %.4f\n", pos_rate_train))
cat(sprintf("Positive class rate (test): %.4f\n", pos_rate_test))

# Create confusion matrix
confusion_matrix <- table(Predicted = test_pred_binary, Actual = test_data$y)
cat("\n=== CONFUSION MATRIX ===\n")
print(confusion_matrix)

# Plot training history
plot_training_history <- function(history) {
  # Extract history data
  hist_data <- data.frame(
    epoch = 1:length(history$metrics$loss),
    train_loss = history$metrics$loss,
    val_loss = history$metrics$val_loss,
    train_acc = history$metrics$accuracy,
    val_acc = history$metrics$val_accuracy,
    train_precision = history$metrics$precision,
    val_precision = history$metrics$val_precision,
    train_recall = history$metrics$recall,
    val_recall = history$metrics$val_recall
  )

  # Loss plot
  p1 <- ggplot(hist_data, aes(x = epoch)) +
    geom_line(aes(y = train_loss, color = "Training"), size = 1) +
    geom_line(aes(y = val_loss, color = "Validation"), size = 1) +
    labs(title = "Model Loss", x = "Epoch", y = "Loss", color = "Dataset") +
    theme_minimal() +
    scale_color_manual(values = c("Training" = "blue", "Validation" = "red"))

  # Accuracy plot
  p2 <- ggplot(hist_data, aes(x = epoch)) +
    geom_line(aes(y = train_acc, color = "Training"), size = 1) +
    geom_line(aes(y = val_acc, color = "Validation"), size = 1) +
    labs(title = "Model Accuracy", x = "Epoch", y = "Accuracy", color = "Dataset") +
    theme_minimal() +
    scale_color_manual(values = c("Training" = "blue", "Validation" = "red"))

  # Precision plot
  p3 <- ggplot(hist_data, aes(x = epoch)) +
    geom_line(aes(y = train_precision, color = "Training"), size = 1) +
    geom_line(aes(y = val_precision, color = "Validation"), size = 1) +
    labs(title = "Model Precision", x = "Epoch", y = "Precision", color = "Dataset") +
    theme_minimal() +
    scale_color_manual(values = c("Training" = "blue", "Validation" = "red"))

  # Recall plot
  p4 <- ggplot(hist_data, aes(x = epoch)) +
    geom_line(aes(y = train_recall, color = "Training"), size = 1) +
    geom_line(aes(y = val_recall, color = "Validation"), size = 1) +
    labs(title = "Model Recall", x = "Epoch", y = "Recall", color = "Dataset") +
    theme_minimal() +
    scale_color_manual(values = c("Training" = "blue", "Validation" = "red"))

  # Combine plots
  grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2,
               top = "Training History - Multi-Layer LSTM Stock Prediction Model")
}

# Plot confusion matrix heatmap
plot_confusion_matrix <- function(confusion_matrix) {
  # Convert to data frame for plotting
  cm_df <- as.data.frame.table(confusion_matrix)
  names(cm_df) <- c("Predicted", "Actual", "Freq")

  # Calculate percentages
  cm_df$Percentage <- round(cm_df$Freq / sum(cm_df$Freq) * 100, 1)

  # Create heatmap
  p <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = paste0(Freq, "\n(", Percentage, "%)")),
              color = "white", size = 4, fontface = "bold") +
    scale_fill_gradient(low = "lightblue", high = "darkblue", name = "Count") +
    labs(title = "Confusion Matrix - Test Set",
         x = "Actual Class", y = "Predicted Class") +
    theme_minimal() +
    theme(axis.text = element_text(size = 12),
          axis.title = element_text(size = 12),
          plot.title = element_text(size = 14, hjust = 0.5))

  return(p)
}

cat("\n=== PLOTTING TRAINING HISTORY ===\n")
plot_training_history(history)

cat("\n=== PLOTTING CONFUSION MATRIX ===\n")
cm_plot <- plot_confusion_matrix(confusion_matrix)
print(cm_plot)

cat("\n=== MODEL ARCHITECTURE ===\n")
cat("- Multi-layer LSTM with separate branches for price and volume\n")
cat("- Price branch: 64 -> 32 LSTM units\n")
cat("- Volume branch: 64 -> 32 LSTM units\n")
cat("- Combined dense layers: 32 -> 16 -> 1\n")
cat("- Dropout and recurrent dropout for regularization\n")
cat("- Sample weighting based on max gain magnitude\n")
cat("- Early stopping and learning rate reduction callbacks\n")