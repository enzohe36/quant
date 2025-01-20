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

  symbol_dict <- read.csv(
    symbol_dict_path, colClasses = c(symbol = "character")
  )

  data_list <- foreach(
    symbol = symbol_dict$symbol,
    .combine = "append"
  ) %dofuture% {
    data_path <- paste0(data_dir, symbol, ".csv")
    if (!file.exists(data_path)) return(NULL)

    data <- read_csv(data_path, show_col_types = FALSE)
    if (nrow(data) <= 240) return(NULL)

    hold_period <- 5

    data <- data %>%
      mutate(
        symbol = !!symbol %>% as.factor(),
        mkt = symbol_dict$mkt[symbol_dict$symbol == !!symbol] %>%
          as.factor(),
        target_class = ifelse(
          lead(open) * 1.01 <
            lead(runMax(high, n = hold_period - 1), hold_period),
          "buy",
          "sell"
        ) %>%
          as.factor(),
        target_regr = get_ror(
          lead(open),
          lead(runMax(high, n = hold_period - 1), hold_period)
        ),
        adx = get_adx(cbind(high, low, close))[, "adx"],
        turnover = val / mktcap_float,
        mktcost_ror = get_ror(mktcost, close)
      ) %>%
      add_roc("close", c(5, 10, 20, 40, 60, 120, 240)) %>%
      add_ma("turnover", c(20, 60, 120, 240)) %>%
      select(
        -c(
          open, high, low, close, vol, matches("^val"),
          mkt, mktcap, pe, peg, pc, turnover
        )
      ) %>%
      relocate(symbol, .before = date) %>%
      relocate(target_class, target_regr, .after = date) %>%
      tail(60)

    lst <- list()
    lst[[symbol]] <- data
    return(lst)
  }

  saveRDS(data_list, file = data_list_path)
  tsprint(glue("Generated features for constituents of {index}."))
}

plan(sequential)
