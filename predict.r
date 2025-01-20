rm(list = ls())

library(doFuture)
library(foreach)
library(glue)
library(data.table)
library(ranger)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

plan(multisession, workers = availableCores() - 1)

index_list <- c("000300", "000905", "000852", "932000")
data_dir <- "data/"
model_dir <- "models/"

symbol_dict <- foreach(
  index = index_list,
  .combine = "append"
) %dofuture% {
  symbol_dict_path <- paste0(data_dir, "symbol_dict_", index, ".csv")
  read_csv(symbol_dict_path, show_col_types = FALSE) %>% list
} %>%
  rbindlist()

out <- c()

for (index in index_list) {
  data_list_path <- paste0(model_dir, "data_list_", index, ".rds")
  data_list <- readRDS(data_list_path)

  data_new <- foreach(
    data = data_list,
    .combine = "append"
  ) %dofuture% {
    date <- as_tradedate(ymd(20250117))
    list(filter(data, date == !!date))
  } %>%
    rbindlist()

  for (i in 1:5) {
    model_path <- paste0(model_dir, "rf_list_", index, "_", i, ".rds")
    rf_list <- readRDS(model_path)
    rf_class <- rf_list[["rf_class"]]
    rf_regr <- rf_list[["rf_regr"]]
    thr_prob <- rf_list[["eval"]]["thr_prob"]

    pred <- select(data_new, symbol, date) %>%
      left_join(symbol_dict %>% select(-matches("^mkt")), by = "symbol") %>%
      bind_cols(predict(rf_class, data_new)[["predictions"]]) %>%
      rename(prob_buy = buy, prob_sell = sell) %>%
      mutate(
        class = ifelse(prob_buy >= thr_prob, "buy", NA) %>%
          coalesce(ifelse(prob_sell >= 0.5, "sell", NA)),
        regr = predict(rf_regr, data_new)[["predictions"]]
      )

    regr_name <- paste0("regr", "_", index, "_", i)
    pred_buy <- filter(pred, class == "buy") %>%
      select(symbol, date, name, regr) %>%
      rename(!!regr_name := regr)
    out <- c(out, list(pred_buy))

    pred_regr_index <- sum(pred$weight * pred$regr) %>% round(3)
    tsprint(glue("Index {index}, model {i}: return = {pred_regr_index}"))
  }
}

out <- reduce(
  out, full_join, by = names(out[[1]]) %>% .[!grepl("^regr", .)]
) %>%
  mutate(
    regr = select(., matches("^regr_")) %>%
      apply(1, function(v) mean(v, na.rm = TRUE)),
    vote = select(., matches("^regr_")) %>%
      apply(1, function(v) length(v[complete.cases(v)])),
    across(matches("^regr_"), ~ NULL)
  )
print(out)

plan(sequential)
