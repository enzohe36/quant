rm(list = ls())

gc()

library(lightgbm)
library(caret)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

################################################################################

model_dir <- "models/"
features_path <- paste0(model_dir, "features.rds")

features <- readRDS(features_path)

end_date <- as_tradedate(now() - hours(16))
train_start <- end_date %m-% years(10)
train_end <- end_date %m-% years(2)
val_end <- end_date %m-% years(1)

features_ds <- downSample(
  x = filter(features, date >= train_start & date < train_end) %>%
    select(-date),
  y = filter(features, date >= train_start & date < train_end) %>%
    pull(label),
  list = TRUE
)
X_train <- features_ds$x
y_train <- features_ds$y

X_val <- filter(features, date >= train_end & date < val_end) %>%
  select(-date)
y_val <- filter(features, date >= train_end & date < val_end) %>%
  pull(label)

X_test <- filter(features, date >= val_end) %>%
  select(-date)
y_test <- filter(features, date >= val_end) %>%
  pull(label)

# Prepare data in LightGBM format
dtrain <- lgb.Dataset(
  data = as.matrix(X_train),
  label = y_train
)

dval <- lgb.Dataset(
  data = as.matrix(X_val),
  label = y_val,
  reference = dtrain
)

# Set parameters
params <- list(
  objective = "multiclass",
  num_class = 3,
  metric = "multi_logloss",
  learning_rate = 0.01,
  max_depth = 5,
  num_leaves = 31,
  num_threads = parallel::detectCores() - 1,
  verbosity = 1
)

# Train with early stopping
model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,
  valids = list(validation = dval),
  early_stopping_rounds = 50
)

# Predict on test set
predictions <- predict(model, as.matrix(X_test))
predicted_class <- ifelse(predictions > 0.5, 1, 0)
