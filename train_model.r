library(lightgbm)
library(caret)
library(parallel)
library(tidyverse)

# ===== 1. SETUP DIRECTORIES AND LOAD DATA =====
model_dir <- "models/"

# Load features and labels
features <- readRDS(paste0(model_dir, "features.rds"))
labels <- readRDS(paste0(model_dir, "labels.rds"))

# features <- readRDS(paste0(model_dir, "features.rds"))
# sample_idx <- sample(1:nrow(features), 10000)  # For quick testing
# features <- features[sample_idx, ]
# labels <- readRDS(paste0(model_dir, "labels.rds")) %>% .[sample_idx]

# Sort by date to ensure temporal order
features <- features %>% arrange(date)
labels <- labels[order(features$date)]

cat("Data loaded successfully\n")
cat(sprintf("Features shape: %d rows, %d columns\n", nrow(features), ncol(features)))
cat(sprintf("Labels shape: %d rows\n", length(labels)))

# ===== 2. TIME SERIES CROSS-VALIDATION FUNCTION (REDESIGNED) =====
time_series_cv <- function(features, n_splits = 5) {
  n_obs <- nrow(features)

  # Define validation size
  val_size <- floor(n_obs / (n_splits + 2))

  # Minimum training size
  min_train_size <- val_size * 2

  splits <- list()

  for (i in 1:n_splits) {
    # Each split adds val_size observations to training
    train_end_idx <- min_train_size + (i - 1) * val_size
    val_start_idx <- train_end_idx + 1
    val_end_idx <- val_start_idx + val_size - 1

    # Only add split if indices are valid
    if (val_end_idx <= n_obs) {
      splits[[i]] <- list(
        train_idx = 1:train_end_idx,
        val_idx = val_start_idx:val_end_idx
      )
    }
  }

  return(splits)
}

# Create CV splits
cv_splits <- time_series_cv(features, n_splits = 5)

cat("\nTime Series Cross-Validation Splits:\n")
for (i in 1:length(cv_splits)) {
  cat(sprintf("Fold %d: Train=%d-%d (%d obs), Val=%d-%d (%d obs)\n",
              i,
              min(cv_splits[[i]]$train_idx),
              max(cv_splits[[i]]$train_idx),
              length(cv_splits[[i]]$train_idx),
              min(cv_splits[[i]]$val_idx),
              max(cv_splits[[i]]$val_idx),
              length(cv_splits[[i]]$val_idx)))
}

# Detect number of physical cores
n_cores <- detectCores(logical = FALSE)
cat(sprintf("\nUsing %d physical cores for parallel processing\n", n_cores))

# ===== 3. EVALUATION FUNCTION WITH CV =====
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

    # Calculate class weights for training set
    class_counts_train <- table(y_train)
    total_train <- sum(class_counts_train)
    n_classes <- length(class_counts_train)
    class_weights_train <- total_train / (n_classes * class_counts_train)
    sample_weights_train <- class_weights_train[as.character(y_train)]

    # Calculate class weights for validation set
    class_counts_val <- table(y_val)
    total_val <- sum(class_counts_val)
    class_weights_val <- total_val / (n_classes * class_counts_val)
    sample_weights_val <- class_weights_val[as.character(y_val)]

    # Create LightGBM datasets
    dtrain <- lgb.Dataset(
      data = as.matrix(X_train),
      label = y_train,
      weight = sample_weights_train
    )

    dval <- lgb.Dataset(
      data = as.matrix(X_val),
      label = y_val,
      weight = sample_weights_val,
      reference = dtrain
    )

    # Train model
    model <- lgb.train(
      params = params,
      data = dtrain,
      nrounds = 1000,
      valids = list(validation = dval),
      early_stopping_rounds = 50,
      verbose = -1
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
  verbosity = -1,
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
min_data_values <- c(10, 20, 50, 100, 200)
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

# Train final model on data up to train_end
X_train_final <- features %>% filter(date < train_end) %>% select(-date)
y_train_final <- labels[features$date < train_end]

X_test_final <- features %>% filter(date >= train_end) %>% select(-date)
y_test_final <- labels[features$date >= train_end]

cat(sprintf("\nFinal training set: %d observations\n", nrow(X_train_final)))
cat(sprintf("Final test set: %d observations\n", nrow(X_test_final)))

# Calculate weights
class_counts_train <- table(y_train_final)
total_train <- sum(class_counts_train)
n_classes <- length(class_counts_train)
class_weights_train <- total_train / (n_classes * class_counts_train)
sample_weights_train <- class_weights_train[as.character(y_train_final)]

# Train final model
dtrain_final <- lgb.Dataset(
  data = as.matrix(X_train_final),
  label = y_train_final,
  weight = sample_weights_train
)

final_model <- lgb.train(
  params = base_params,
  data = dtrain_final,
  nrounds = 1000,
  verbose = 1
)

# ===== 12. EVALUATE ON TEST SET =====
test_pred_probs <- predict(final_model, as.matrix(X_test_final))
test_pred_probs <- matrix(test_pred_probs, ncol = 3, byrow = TRUE)
test_pred_class <- apply(test_pred_probs, 1, which.max) - 1

conf_matrix <- confusionMatrix(
  as.factor(test_pred_class),
  as.factor(y_test_final)
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

# Save predictions
predictions_df <- data.frame(
  date = features %>% filter(date >= train_end) %>% pull(date),
  actual = y_test_final,
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