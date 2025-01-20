# python -m aktools

rm(list = ls())

library(doFuture)
library(foreach)
library(glue)
library(data.table)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

plan(multisession, workers = availableCores() - 1)

index_list <- c("000300", "000905", "000852", "932000")
model_dir <- "models/"
data_dir <- "data/"

symbol_dict <- foreach(
  index = index_list,
  .combine = "append"
) %dofuture% {
  symbol_dict_path <- paste0(data_dir, "symbol_dict_", index, ".csv")
  read_csv(symbol_dict_path, show_col_types = FALSE) %>% list
} %>%
  rbindlist()

result_buy <- c()

for (index in index_list) {
  data_list_path <- paste0(model_dir, "data_list_", index, ".rds")
  data_test_path <- paste0(model_dir, "data_test_", index, ".rds")
  model_class_path <- paste0(model_dir, "rf_class_", index, ".rds")
  model_regr_path <- paste0(model_dir, "rf_regr_", index, ".rds")

  data_list <- readRDS(data_list_path)
  data_test <- readRDS(data_test_path)
  rf_class <- readRDS(model_class_path)
  rf_regr <- readRDS(model_regr_path)

  data_new <- foreach(
    data = data_list,
    .combine = "append"
  ) %dofuture% {
    date <- as_tradedate(now() - hours(16))
    list(filter(data, date == !!date))
  } %>%
    rbindlist()

  thr_prob <- predict(rf_class, data_test)[["predictions"]][, "buy"] %>%
    quantile(0.95)

  result <- data_frame(symbol = data_new$symbol) %>%
    bind_cols(predict(rf_class, data_new)[["predictions"]]) %>%
    rename(prob_buy = buy, prob_sell = sell) %>%
    mutate(
      class = ifelse(prob_buy >= thr_prob, "buy", NA) %>%
        coalesce(ifelse(prob_sell >= 0.5, "sell", NA)),
      regr = predict(rf_regr, data_new)[["predictions"]]
    )

  result_buy <- c(result_buy, filter(result, class == "buy") %>% list())
}

out <- reduce(
  list(symbol_dict, rbindlist(result_buy)), right_join, by = "symbol"
) %>%
  arrange(desc(regr)) %>%
  mutate(across(where(is.numeric), round, 3))
print(out)

plan(sequential)
