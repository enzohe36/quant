rm(list = ls())

library(doFuture)
library(foreach)
library(combinat)
library(glue)
library(data.table)
library(caret)
library(TTR)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

plan(multisession, workers = availableCores() - 1)

index_list <- c("000300", "000905", "000852", "932000")
data_dir <- "data/"
model_dir <- "models/"

dir.create(model_dir)

for (index in index_list) {
  symbol_dict_path <- paste0(data_dir, "symbol_dict_", index, ".csv")
  data_list_path <- paste0(model_dir, "data_list_", index, ".rds")
  data_train_path <- paste0(model_dir, "data_train_", index, ".rds")
  data_test_path <- paste0(model_dir, "data_test_", index, ".rds")

  symbol_list <- read_csv(symbol_dict_path, show_col_types = FALSE) %>%
    pull(symbol)

  data_list <- foreach(
    symbol = symbol_list,
    .combine = "append"
  ) %dofuture% {
    data_path <- paste0(data_dir, symbol, ".csv")
    if (!file.exists(data_path)) return(NULL)

    data <- read_csv(data_path, show_col_types = FALSE)
    if (nrow(data) <= 240) return(NULL)

    hold_period <- 5

    data <- data %>%
      mutate(
        data, symbol = symbol, .before = date
      ) %>%
      mutate(
        target_class = ifelse(
          lead(open) * 1.01 < lead(runMax(high, n = 4), 5), "buy", "sell"
        ) %>%
          as.factor(),
        target_regr = get_roc(
          lead(open), lead(runMax(high, n = 4), 5)
        ),
        weight = close * sapply(
          as.list(mktcap_float / mktcap),
          function(x) {
            if (x <= 0.1) x else if (x > 0.8) 1 else ceiling(x / 0.1) * 0.1
          }
        ),
        .after = date
      ) %>%
      mutate(
        adx = get_adx(cbind(high, low, close))[, "adx"],
        turnover = val / mktcap_float
      ) %>%
      add_roc("close", c(5, 10, 20, 40, 60, 120, 240)) %>%
      add_ma("turnover", c(20, 60, 120, 240)) %>%
      select(
        -c(
          open, high, low, close, vol, matches("^val"),
          mktcap, pe, peg, pc, turnover
        )
      ) %>%
      tail(60)

    lst <- list()
    lst[[symbol]] <- data
    return(lst)
  }

  data_cat <- rbindlist(data_list) %>% na.omit()
  data_train_ind <- createDataPartition(
    as.factor(data_cat$date), p = 3 / 4, list = FALSE
  )
  data_train <- data_cat[data_train_ind, ]
  data_test <- data_cat[-data_train_ind, ]

  saveRDS(data_list, file = data_list_path)
  saveRDS(data_train, file = data_train_path)
  saveRDS(data_test, file = data_test_path)
  tsprint(glue("Generated features for constituents of {index}."))
}

plan(sequential)
