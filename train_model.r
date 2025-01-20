rm(list = ls())

library(data.table)
library(glue)
library(doFuture)
library(caret)
library(ranger)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

index_list <- c("000300", "000905", "000852", "932000")
model_dir <- "models/"

for (index in index_list) {
  data_list_path <- paste0(model_dir, "data_list_", index, ".rds")
  data_cat <- readRDS(data_list_path) %>% rbindlist() %>% na.omit()

  for (i in 1:5) {
    model_path <- paste0(model_dir, "rf_list_", index, "_", i, ".rds")
    plot_path <- paste0(model_dir, "rf_result_", index, "_", i, ".pdf")

    data_train_ind <- createDataPartition(
      as.factor(data_cat$date), p = 3 / 4, list = FALSE
    )
    data_train <- data_cat[data_train_ind, ]
    data_test <- data_cat[-data_train_ind, ]

    for (rf_type in c("class", "regr")) {
      x <- data_train %>% select(-c(symbol, date, target_class, target_regr))
      y <- data_train %>% pull(paste0("target_", rf_type))

      rf_ctrl <- trainControl(method = "cv", number = 5, p = 2 / 3)

      rf_grid <- expand.grid(
        mtry = floor(seq(1, ncol(x), length.out = 10)),
        splitrule = ifelse(is.factor(y), "gini", "variance"),
        min.node.size = ifelse(is.factor(y), 1, 5)
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

      assign(
        paste0("rf_", rf_type),
        ranger(
          x = x, y = y,
          num.trees = 500,
          mtry = mtry_opt,
          importance = "permutation",
          probability = ifelse(is.factor(y), TRUE, FALSE),
          num.threads = availableCores(omit = 1)
        )
      )
    }

    pdf(file = plot_path)

    par(mar = c(5.1, 7, 4.1, 2.1))
    barplot(
      rf_class[["variable.importance"]] %>% sort(),
      horiz = TRUE, las = 1, cex.names = 0.9,
      xlab = paste0("Importance: ", rf_type),
      main = glue("Index {index}, model {i}")
    )
    barplot(
      rf_regr[["variable.importance"]] %>% sort(),
      horiz = TRUE, las = 1, cex.names = 0.9,
      xlab = paste0("Importance: ", rf_type),
      main = glue("Index {index}, model {i}")
    )

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
    plot(
      prob_rmse$prob, prob_rmse$rmse, type = "l",
      xlab = "Probability threshold", ylab = "RMSE",
      main = glue("Index {index}, model {i}")
    )

    thr_prob <- quantile(comparison$pred_prob_buy, 0.95) %>% unname()
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
      xlim = plot_range, ylim = plot_range,
      col = comparison$col, pch = 20, cex = 0.3,
      xlab = "Target", ylab = "Prediction",
      main = glue("Index {index}, model {i}")
    )
    abline(a = 0, b = 1, col = "blue")

    dev.off()

    cm <- confusionMatrix(comparison$pred_class, comparison$target_class)

    eval <- c(
      acc = cm$overall["Accuracy"] %>% unname(),
      acc_p = cm$overall["AccuracyPValue"] %>% unname(),
      fdr = cm$table[2, 1] / (cm$table[1, 1] + cm$table[2, 1]),
      rmse = get_rmse(comparison$pred_regr, comparison$target_regr),
      thr_prob = thr_prob,
      thr_rmse = get_rmse(
        comparison$pred_regr[comparison$pred_prob_buy >= thr_prob],
        comparison$target_regr[comparison$pred_prob_buy >= thr_prob]
      )
    )
    saveRDS(
      list(rf_class = rf_class, rf_regr = rf_regr, eval = eval), model_path
    )
    tsprint(glue("Wrote to {model_path}."))
    print(round(eval, 3))
  }
}
