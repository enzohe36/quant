rm(list = ls())

library(foreach)
library(doFuture)
library(data.table)
library(glue)
library(doFuture)
library(caret)
library(ranger)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

model_dir <- "models/"

data_list_path <- paste0(model_dir, "data_list.rds")
data_cat <- readRDS(data_list_path) %>% rbindlist() %>% na.omit()

for (i in 1:5) {
  data_test_path <- paste0(model_dir, "data_test_", i, ".rds")
  model_path <- paste0(model_dir, "rf_", i, ".rds")

  data_train_ind <- createDataPartition(
    as.factor(data_cat$date), p = 4 / 5, list = FALSE
  )
  data_train <- data_cat[data_train_ind, ] %>% slice_sample(n = 100)
  data_test <- data_cat[-data_train_ind, ]
  saveRDS(data_test, data_test_path)

  x <- data_train %>% select(-c(symbol, date, target))
  y <- data_train %>% pull(target)

  rf_ctrl <- trainControl(
    method = "cv",
    number = 5,
    p = 3 / 4
  )

  rf_grid <- expand.grid(
    mtry = floor(seq(1, 2 * sqrt(ncol(x)), length.out = 5)),
    splitrule = "gini",
    min.node.size = 1
  )

  rf_tune <- train(
    x = x, y = y,
    method = "ranger",
    num.threads = availableCores(omit = 1),
    trControl = rf_ctrl,
    tuneGrid = rf_grid
  )

  mtry_opt <- rf_tune$bestTune[, "mtry"]

  rf <- ranger(
    x = x, y = y,
    mtry = mtry_opt,
    importance = "permutation",
    probability = TRUE,
    num.threads = availableCores(omit = 1)
  )
  saveRDS(rf, model_path)

  class <- c("a", "b", "c", "d", "e")
  compar <- predict(rf, data_test)[["predictions"]] %>%
    as_data_frame() %>%
    mutate(
      prob_max = apply(., 1, function(v) max(v)),
      pred = apply(., 1, function(v) class[match(max(v), v)]) %>%
        as.factor(),
      target = data_test$target
    ) %>%
    filter(prob_max > 0.5)
  cm <- confusionMatrix(compar$pred, compar$target)
  acc <- cm$overall["Accuracy"]
  tsprint(glue("Model {i}: accuracy = {acc}."))
  print(cm$table)

  par(mar = c(5.1, 7, 4.1, 2.1))
  barplot(
    rf[["variable.importance"]] %>% sort(),
    horiz = TRUE, las = 1, cex.names = 0.9,
    xlab = "Variable importance",
    main = glue("Model {i}")
  )
}
