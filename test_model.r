rm(list = ls())

library(ranger)
library(caret)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

index_list <- c("000300", "000905", "000852", "932000")
model_dir <- "models/"

for (index in index_list) {
  data_test_path <- paste0(model_dir, "data_test_", index, ".rds")
  model_class_path <- paste0(model_dir, "rf_class_", index, ".rds")
  model_regr_path <- paste0(model_dir, "rf_regr_", index, ".rds")

  data_test <- readRDS(data_test_path)
  rf_class <- readRDS(model_class_path)
  rf_regr <- readRDS(model_regr_path)

  comparison <- predict(rf_class, data_test)[["predictions"]] %>%
    as_data_frame() %>%
    mutate(
      symbol = data_test$symbol,
      date = data_test$date,
      target_class = data_test$target_class,
      target_regr = data_test$target_regr,
      .before = 1
    ) %>%
    rename(
      pred_prob_buy = buy,
      pred_prob_sell = sell
    ) %>%
    mutate(
      pred_class = ifelse(pred_prob_buy > pred_prob_sell, "buy", "sell") %>%
        as.factor(),
      pred_regr = predict(rf_regr, data_test)[["predictions"]]
    )

  par(mfrow = c(1,2))

  prob_rmse <- data_frame(
    prob = seq(0, 1, 0.01),
    rmse = sapply(
      prob,
      function(x) {
        get_rmse(
          comparison$pred_regr[comparison$pred_prob_buy >= x],
          comparison$target_regr[comparison$pred_prob_buy >= x]
        )
      }
    )
  )
  thr_prob <- quantile(comparison$pred_prob_buy, 0.95) %>% unname()
  plot(prob_rmse$prob, prob_rmse$rmse, type = "l")
  abline(v = thr_prob, col = "red")

  comparison <- mutate(
    comparison, col = ifelse(pred_prob_buy >= thr_prob, "red", NA)
  )
  plot_range <- c(
    min(comparison$target_regr, comparison$pred_regr),
    max(comparison$target_regr, comparison$pred_regr)
  )
  plot(
    comparison$target_regr, comparison$pred_regr,
    xlim = plot_range,
    ylim = plot_range,
    col = comparison$col,
    pch = 20,
    cex = 0.3
  )
  abline(a = 0, b = 1, col = "blue")

  cm <- confusionMatrix(comparison$pred_class, comparison$target_class)
  acc <- cm$overall["Accuracy"] %>% unname()
  acc_p <- cm$overall["AccuracyPValue"] %>% unname()
  fdr <- cm$table[2, 1] / (cm$table[1, 1] + cm$table[2, 1])
  rmse <- get_rmse(comparison$pred_regr, comparison$target_regr)
  thr_rmse <- get_rmse(
    comparison$pred_regr[comparison$pred_prob_buy >= thr_prob],
    comparison$target_regr[comparison$pred_prob_buy >= thr_prob]
  )
  v <- c(
    acc = acc, acc_p = acc_p, fdr = fdr, rmse = rmse,
    thr_prob = thr_prob, thr_rmse = thr_rmse
  ) %>%
    round(3)
  tsprint(glue("Evaluated {index}."))
  print(v)

  print(
    rf_regr$variable.importance %>%
      sort(decreasing = TRUE) %>%
      names() %>%
      setNames(seq_len(length(.)))
  )
}
