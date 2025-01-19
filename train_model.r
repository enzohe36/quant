rm(list = ls())

library(doFuture)
library(caret)
library(ranger)
library(tidyverse)

# ------------------------------------------------------------------------------

index_list <- c("000300", "000905", "000852", "932000")
model_dir <- "models/"

for (index in index_list) {
  data_train_path <- paste0(model_dir, "data_train_", index, ".rds")
  data_train <- readRDS(data_train_path)

  for (rf_type in c("class", "regr")) {
    model_path <- paste0(model_dir, "rf_", rf_type, "_", index, ".rds")

    x <- data_train %>%
      select(-c(symbol, date, target_class, target_regr, weight))
    y <- data_train %>% pull(paste0("target_", rf_type))

    rf_ctrl <- trainControl(
      method = "cv",
      number = 5,
      p = 2 / 3
    )

    rf_grid <- expand.grid(
      mtry = round(seq(1, ncol(x), length.out = 10)),
      splitrule = if (is.factor(y)) "gini" else "variance",
      min.node.size = if (is.factor(y)) 1 else 5
    )

    rf_tune <- train(
      x = x, y = y,
      method = "ranger",
      num.trees = 50,
      num.threads = availableCores(omit = 1),
      trControl = rf_ctrl,
      tuneGrid = rf_grid
    )

    mtry_opt <- rf_tune$bestTune[, "mtry"]
    rf <- ranger(
      x = x, y = y,
      num.trees = 500,
      mtry = mtry_opt,
      importance = "permutation",
      probability = if (is.factor(y)) TRUE else FALSE,
      num.threads = availableCores(omit = 1)
    )

    saveRDS(rf, file = model_path)
    print(paste0("Wrote to ", model_path, "."))
  }
}
