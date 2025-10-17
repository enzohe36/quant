library(lightgbm)
library(caret)
library(parallel)
library(tidyverse)

# Set seed for reproducibility
set.seed(42)

# Define gap width
width <- 20

# ===== 1. SETUP DIRECTORIES AND LOAD DATA =====
model_dir <- "models/"

# Load features and labels
features <- readRDS(paste0(model_dir, "features.rds"))
labels <- readRDS(paste0(model_dir, "labels.rds"))
# sample_idx <- sample(1:nrow(features), 10000)  # For quick testing
# features <- features[sample_idx, ]
# labels <- labels[sample_idx]

# Sort by date to ensure temporal order
features <- features %>% arrange(date)
labels <- labels[order(features$date)]

cat("Data loaded successfully\n")
cat(sprintf("Features shape: %d rows, %d columns\n", nrow(features), ncol(features)))
cat(sprintf("Labels shape: %d rows\n", length(labels)))

# ===== 2. TIME SERIES CROSS-VALIDATION FUNCTION (REDESIGNED) =====
time_series_cv <- function(features, n_splits = 5, width = 20) {
  # Get unique dates and number of days in training set
  unique_dates <- unique(features$date)
  n_days <- length(unique_dates)

  cat(sprintf("Total unique trading days: %d\n", n_days))

  # Calculate validation size: min of (n_days / (n_splits + 2)) or (n_days * 0.1)
  val_size_option1 <- floor(n_days / (n_splits + 2))
  val_size_option2 <- floor(n_days * 0.1)
  val_size <- min(val_size_option1, val_size_option2)

  cat(sprintf("Validation size (days): %d\n", val_size))

  # Calculate minimum training size
  min_train_size <- n_days - width - val_size * n_splits

  if (min_train_size <= 0) {
    stop("Not enough data for the specified n_splits and width. Reduce n_splits or width.")
  }

  cat(sprintf("Minimum training size (days): %d\n", min_train_size))
  cat(sprintf("Gap width (days): %d\n\n", width))

  splits <- list()

  for (i in 1:n_splits) {
    # Calculate training end date index (in days)
    train_end_day_idx <- min_train_size + (i - 1) * val_size

    # Add gap
    val_start_day_idx <- train_end_day_idx + width + 1
    val_end_day_idx <- val_start_day_idx + val_size - 1

    # Check validity
    if (val_end_day_idx <= n_days) {
      # Convert day indices to observation indices
      train_end_date <- unique_dates[train_end_day_idx]
      val_start_date <- unique_dates[val_start_day_idx]
      val_end_date <- unique_dates[val_end_day_idx]

      train_idx <- which(features$date <= train_end_date)
      val_idx <- which(features$date >= val_start_date & features$date <= val_end_date)

      splits[[i]] <- list(
        train_idx = train_idx,
        val_idx = val_idx,
        train_end_date = train_end_date,
        val_start_date = val_start_date,
        val_end_date = val_end_date
      )
    }
  }

  return(splits)
}

# Create CV splits
cv_splits <- time_series_cv(features, n_splits = 5, width = width)

cat("Time Series Cross-Validation Splits:\n")
for (i in 1:length(cv_splits)) {
  cat(sprintf("Fold %d:\n", i))
  cat(sprintf("  Train: %s to %s (%d obs)\n",
              min(features$date[cv_splits[[i]]$train_idx]),
              cv_splits[[i]]$train_end_date,
              length(cv_splits[[i]]$train_idx)))
  cat(sprintf("  Gap: %d days\n", width))
  cat(sprintf("  Val: %s to %s (%d obs)\n",
              cv_splits[[i]]$val_start_date,
              cv_splits[[i]]$val_end_date,
              length(cv_splits[[i]]$val_idx)))
}

# Detect number of physical cores
n_cores <- detectCores(logical = FALSE)
cat(sprintf("\nUsing %d physical cores for parallel processing\n", n_cores))

# ===== 3. EVALUATION FUNCTION WITH CV (REDESIGNED) =====
evaluate_params <- function(params, cv_splits, features, labels) {
  cv_scores <- numeric(length(cv_splits))

  for (fold in 1:length(cv_splits)) {
    # Get train and validation indices
    train_idx <- cv_splits[[fold]]$train_idx
    val_idx <- cv_splits[[fold]]$val_idx

    X_train <- features %>% slice(train_idx) %>% select(-date)
    y_train <- labels[train_idx]
    X_val <- features %>% slice(val_idx) %>% select(-date)
    y_val <- labels[val_idx]

    # Equalize classes in training set (undersample to smallest class)
    train_data <- X_train %>% mutate(label = y_train)
    class_counts_train <- table(y_train)
    min_class_size_train <- min(class_counts_train)

    train_balanced <- train_data %>%
      group_by(label) %>%
      slice_sample(n = min_class_size_train) %>%
      ungroup()

    X_train_balanced <- train_balanced %>% select(-label)
    y_train_balanced <- train_balanced$label

    # Equalize classes in validation set (undersample to smallest class)
    val_data <- X_val %>% mutate(label = y_val)
    class_counts_val <- table(y_val)
    min_class_size_val <- min(class_counts_val)

    val_balanced <- val_data %>%
      group_by(label) %>%
      slice_sample(n = min_class_size_val) %>%
      ungroup()

    X_val_balanced <- val_balanced %>% select(-label)
    y_val_balanced <- val_balanced$label

    # Create LightGBM datasets (no weights)
    dtrain <- lgb.Dataset(
      data = as.matrix(X_train_balanced),
      label = y_train_balanced
    )

    dval <- lgb.Dataset(
      data = as.matrix(X_val_balanced),
      label = y_val_balanced,
      reference = dtrain
    )

    # Train model
    model <- lgb.train(
      params = params,
      data = dtrain,
      nrounds = 1000,
      valids = list(validation = dval),
      early_stopping_rounds = 50
    )

    cv_scores[fold] <- model$best_score
  }

  return(list(
    mean_score = mean(cv_scores),
    std_score = sd(cv_scores),
    scores = cv_scores
  ))
}

# ===== 4. INITIALIZE BASE PARAMETERS =====
base_params <- list(
  objective = "multiclass",
  num_class = 3,
  metric = "multi_logloss",
  max_bin = 255,             # Default - will tune first
  num_leaves = 31,           # Default
  max_depth = -1,            # No limit (default)
  min_data_in_leaf = 20,     # Default
  feature_fraction = 1.0,    # Default
  bagging_fraction = 1.0,    # Default
  bagging_freq = 0,          # Default (no bagging)
  learning_rate = 0.1,       # Default
  verbosity = 1,
  num_threads = n_cores      # Use all physical cores
)

results_all <- data.frame()

# ===== 5. STEP 1: TUNE MAX_BIN =====
cat("\n===== STEP 1: TUNING MAX_BIN =====\n")
max_bin_values <- c(63, 127, 255)
best_max_bin <- NULL
best_score_step1 <- Inf

step1_results <- data.frame()

for (mb in max_bin_values) {
  cat(sprintf("Testing max_bin = %d\n", mb))

  params <- base_params
  params$max_bin <- mb

  cv_result <- evaluate_params(params, cv_splits, features, labels)

  step1_results <- rbind(step1_results, data.frame(
    max_bin = mb,
    mean_score = cv_result$mean_score,
    std_score = cv_result$std_score
  ))

  cat(sprintf("  Mean score: %.4f (+/- %.4f)\n",
              cv_result$mean_score, cv_result$std_score))

  if (cv_result$mean_score < best_score_step1) {
    best_score_step1 <- cv_result$mean_score
    best_max_bin <- mb
    cat("  -> New best!\n")
  }
}

cat(sprintf("\nBest max_bin: %d (score: %.4f)\n",
            best_max_bin, best_score_step1))
base_params$max_bin <- best_max_bin

# ===== 6. STEP 2: TUNE NUM_LEAVES =====
cat("\n===== STEP 2: TUNING NUM_LEAVES =====\n")
num_leaves_values <- c(15, 31, 63, 127, 255)
best_num_leaves <- NULL
best_score_step2 <- Inf

step2_results <- data.frame()

for (nl in num_leaves_values) {
  cat(sprintf("Testing num_leaves = %d\n", nl))

  params <- base_params
  params$num_leaves <- nl

  cv_result <- evaluate_params(params, cv_splits, features, labels)

  step2_results <- rbind(step2_results, data.frame(
    num_leaves = nl,
    mean_score = cv_result$mean_score,
    std_score = cv_result$std_score
  ))

  cat(sprintf("  Mean score: %.4f (+/- %.4f)\n",
              cv_result$mean_score, cv_result$std_score))

  if (cv_result$mean_score < best_score_step2) {
    best_score_step2 <- cv_result$mean_score
    best_num_leaves <- nl
    cat("  -> New best!\n")
  }
}

cat(sprintf("\nBest num_leaves: %d (score: %.4f)\n",
            best_num_leaves, best_score_step2))
base_params$num_leaves <- best_num_leaves

# ===== 7. STEP 3: TUNE MAX_DEPTH =====
cat("\n===== STEP 3: TUNING MAX_DEPTH =====\n")
max_depth_values <- c(3, 5, 7, 10, 15, -1)  # -1 means no limit
best_max_depth <- NULL
best_score_step3 <- Inf

step3_results <- data.frame()

for (md in max_depth_values) {
  cat(sprintf("Testing max_depth = %d\n", md))

  params <- base_params
  params$max_depth <- md

  cv_result <- evaluate_params(params, cv_splits, features, labels)

  step3_results <- rbind(step3_results, data.frame(
    max_depth = md,
    mean_score = cv_result$mean_score,
    std_score = cv_result$std_score
  ))

  cat(sprintf("  Mean score: %.4f (+/- %.4f)\n",
              cv_result$mean_score, cv_result$std_score))

  if (cv_result$mean_score < best_score_step3) {
    best_score_step3 <- cv_result$mean_score
    best_max_depth <- md
    cat("  -> New best!\n")
  }
}

cat(sprintf("\nBest max_depth: %d (score: %.4f)\n",
            best_max_depth, best_score_step3))
base_params$max_depth <- best_max_depth

# ===== 8. STEP 4: TUNE MIN_DATA_IN_LEAF =====
cat("\n===== STEP 4: TUNING MIN_DATA_IN_LEAF =====\n")
min_data_values <- c(200, 100, 50, 20, 10)  # Decreasing order
best_min_data <- NULL
best_score_step4 <- Inf

step4_results <- data.frame()

for (md in min_data_values) {
  cat(sprintf("Testing min_data_in_leaf = %d\n", md))

  params <- base_params
  params$min_data_in_leaf <- md

  cv_result <- evaluate_params(params, cv_splits, features, labels)

  step4_results <- rbind(step4_results, data.frame(
    min_data_in_leaf = md,
    mean_score = cv_result$mean_score,
    std_score = cv_result$std_score
  ))

  cat(sprintf("  Mean score: %.4f (+/- %.4f)\n",
              cv_result$mean_score, cv_result$std_score))

  if (cv_result$mean_score < best_score_step4) {
    best_score_step4 <- cv_result$mean_score
    best_min_data <- md
    cat("  -> New best!\n")
  }
}

cat(sprintf("\nBest min_data_in_leaf: %d (score: %.4f)\n",
            best_min_data, best_score_step4))
base_params$min_data_in_leaf <- best_min_data

# ===== 9. STEP 5: TUNE FEATURE_FRACTION AND BAGGING_FRACTION =====
cat("\n===== STEP 5: TUNING FEATURE_FRACTION AND BAGGING_FRACTION =====\n")
feature_fraction_values <- c(0.6, 0.7, 0.8, 0.9, 1.0)
bagging_fraction_values <- c(0.6, 0.7, 0.8, 0.9, 1.0)
best_feature_fraction <- NULL
best_bagging_fraction <- NULL
best_score_step5 <- Inf

step5_results <- data.frame()

for (ff in feature_fraction_values) {
  for (bf in bagging_fraction_values) {
    cat(sprintf("Testing feature_fraction = %.1f, bagging_fraction = %.1f\n", ff, bf))

    params <- base_params
    params$feature_fraction <- ff
    params$bagging_fraction <- bf
    params$bagging_freq <- ifelse(bf < 1.0, 5, 0)  # Enable bagging if bf < 1

    cv_result <- evaluate_params(params, cv_splits, features, labels)

    step5_results <- rbind(step5_results, data.frame(
      feature_fraction = ff,
      bagging_fraction = bf,
      mean_score = cv_result$mean_score,
      std_score = cv_result$std_score
    ))

    cat(sprintf("  Mean score: %.4f (+/- %.4f)\n",
                cv_result$mean_score, cv_result$std_score))

    if (cv_result$mean_score < best_score_step5) {
      best_score_step5 <- cv_result$mean_score
      best_feature_fraction <- ff
      best_bagging_fraction <- bf
      cat("  -> New best!\n")
    }
  }
}

cat(sprintf("\nBest feature_fraction: %.1f, bagging_fraction: %.1f (score: %.4f)\n",
            best_feature_fraction, best_bagging_fraction, best_score_step5))
base_params$feature_fraction <- best_feature_fraction
base_params$bagging_fraction <- best_bagging_fraction
base_params$bagging_freq <- ifelse(best_bagging_fraction < 1.0, 5, 0)

# ===== 10. STEP 6: TUNE LEARNING_RATE =====
cat("\n===== STEP 6: TUNING LEARNING_RATE =====\n")
learning_rate_values <- c(0.005, 0.01, 0.02, 0.05, 0.1)
best_learning_rate <- NULL
best_score_step6 <- Inf

step6_results <- data.frame()

for (lr in learning_rate_values) {
  cat(sprintf("Testing learning_rate = %.3f\n", lr))

  params <- base_params
  params$learning_rate <- lr

  cv_result <- evaluate_params(params, cv_splits, features, labels)

  step6_results <- rbind(step6_results, data.frame(
    learning_rate = lr,
    mean_score = cv_result$mean_score,
    std_score = cv_result$std_score
  ))

  cat(sprintf("  Mean score: %.4f (+/- %.4f)\n",
              cv_result$mean_score, cv_result$std_score))

  if (cv_result$mean_score < best_score_step6) {
    best_score_step6 <- cv_result$mean_score
    best_learning_rate <- lr
    cat("  -> New best!\n")
  }
}

cat(sprintf("\nBest learning_rate: %.3f (score: %.4f)\n",
            best_learning_rate, best_score_step6))
base_params$learning_rate <- best_learning_rate

# ===== 11. TRAIN FINAL MODEL WITH BEST PARAMETERS =====
cat("\n===== FINAL BEST PARAMETERS =====\n")
print(base_params)

# Define train/test split dates
end_date <- max(features$date)
train_end <- end_date %m-% years(2)

cat(sprintf("\nEnd date: %s\n", end_date))
cat(sprintf("Train end date: %s\n", train_end))

# Get final training and test sets
X_train_final <- features %>% filter(date < train_end) %>% select(-date)
y_train_final <- labels[features$date < train_end]

X_test_final <- features %>% filter(date >= train_end) %>% select(-date)
y_test_final <- labels[features$date >= train_end]

cat(sprintf("\nOriginal training set: %d observations\n", nrow(X_train_final)))
cat(sprintf("Original test set: %d observations\n", nrow(X_test_final)))

# Equalize classes in training set (undersample to smallest class)
train_final_data <- X_train_final %>% mutate(label = y_train_final)
class_counts_train_final <- table(y_train_final)
cat("\nOriginal training class distribution:\n")
print(class_counts_train_final)

min_class_size_train_final <- min(class_counts_train_final)

train_final_balanced <- train_final_data %>%
  group_by(label) %>%
  slice_sample(n = min_class_size_train_final) %>%
  ungroup()

X_train_final_balanced <- train_final_balanced %>% select(-label)
y_train_final_balanced <- train_final_balanced$label

cat(sprintf("\nBalanced training set: %d observations\n", nrow(X_train_final_balanced)))
cat("Balanced training class distribution:\n")
print(table(y_train_final_balanced))

# Equalize classes in test set (undersample to smallest class)
test_final_data <- X_test_final %>% mutate(label = y_test_final)
class_counts_test_final <- table(y_test_final)
cat("\nOriginal test class distribution:\n")
print(class_counts_test_final)

min_class_size_test_final <- min(class_counts_test_final)

test_final_balanced <- test_final_data %>%
  group_by(label) %>%
  slice_sample(n = min_class_size_test_final) %>%
  ungroup()

X_test_final_balanced <- test_final_balanced %>% select(-label)
y_test_final_balanced <- test_final_balanced$label

cat(sprintf("\nBalanced test set: %d observations\n", nrow(X_test_final_balanced)))
cat("Balanced test class distribution:\n")
print(table(y_test_final_balanced))

# Train final model on balanced data
dtrain_final <- lgb.Dataset(
  data = as.matrix(X_train_final_balanced),
  label = y_train_final_balanced
)

final_model <- lgb.train(
  params = base_params,
  data = dtrain_final,
  nrounds = 1000,
  verbose = 1
)

# ===== 12. EVALUATE ON BALANCED TEST SET =====
test_pred_probs <- predict(final_model, as.matrix(X_test_final_balanced))
test_pred_probs <- matrix(test_pred_probs, ncol = 3, byrow = TRUE)
test_pred_class <- apply(test_pred_probs, 1, which.max) - 1

conf_matrix <- confusionMatrix(
  as.factor(test_pred_class),
  as.factor(y_test_final_balanced)
)

cat("\n===== TEST SET PERFORMANCE =====\n")
print(conf_matrix)
cat("\nBalanced Accuracy:", mean(conf_matrix$byClass[, "Balanced Accuracy"]), "\n")

# Feature importance
importance <- lgb.importance(final_model, percentage = TRUE)
cat("\nTop 10 most important features:\n")
print(head(importance, 10))

# ===== 13. SAVE RESULTS TO MODEL_DIR =====
# Save final model as RDS
saveRDS(final_model, paste0(model_dir, "final_model_sequential_tuning.rds"))

# Save tuning results for each step
write.csv(step1_results, paste0(model_dir, "step1_max_bin_results.csv"), row.names = FALSE)
write.csv(step2_results, paste0(model_dir, "step2_num_leaves_results.csv"), row.names = FALSE)
write.csv(step3_results, paste0(model_dir, "step3_max_depth_results.csv"), row.names = FALSE)
write.csv(step4_results, paste0(model_dir, "step4_min_data_results.csv"), row.names = FALSE)
write.csv(step5_results, paste0(model_dir, "step5_fraction_results.csv"), row.names = FALSE)
write.csv(step6_results, paste0(model_dir, "step6_learning_rate_results.csv"), row.names = FALSE)

# Save final parameters
final_params_df <- data.frame(
  parameter = names(base_params),
  value = sapply(base_params, as.character)
)
write.csv(final_params_df, paste0(model_dir, "final_best_parameters.csv"), row.names = FALSE)

# Save predictions with dates
test_dates <- features %>%
  filter(date >= train_end) %>%
  pull(date)

# Match dates with balanced test set
test_final_data_with_date <- features %>%
  filter(date >= train_end) %>%
  mutate(label = y_test_final) %>%
  semi_join(test_final_balanced %>% select(-label), by = names(X_test_final))

predictions_df <- data.frame(
  date = test_final_data_with_date$date,
  actual = y_test_final_balanced,
  predicted = test_pred_class,
  prob_class0 = test_pred_probs[, 1],
  prob_class1 = test_pred_probs[, 2],
  prob_class2 = test_pred_probs[, 3]
)
write.csv(predictions_df, paste0(model_dir, "test_predictions_3class.csv"), row.names = FALSE)

cat("\n===== ALL RESULTS SAVED TO", model_dir, "=====\n")
cat("Model: final_model_sequential_tuning.rds\n")
cat("Step results: step1-6_*.csv\n")
cat("Final parameters: final_best_parameters.csv\n")
cat("Predictions: test_predictions_3class.csv\n")