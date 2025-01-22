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

data_dir <- "data/"
model_dir <- "models/"

dir.create(model_dir)

symbol_dict_path <- paste0(data_dir, "symbol_dict.csv")
data_list_path <- paste0(model_dir, "data_list.rds")

symbol_dict <- read.csv(symbol_dict_path, colClasses = c(symbol = "character"))

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
    mutate(symbol = !!symbol %>% as.factor(), .before = date) %>%
    mutate(
      sd21 = runSD(ROC(close, hold_period), 21),
      r = get_pctr(
        lead(data$open),
        lead(runMax(high, hold_period - 1), hold_period)
      ),
      target = ifelse(r >= 2 * sd21, "a", NA) %>%
        coalesce(ifelse(r >= sd21 & r < 2 * sd21, "b", NA)) %>%
        coalesce(ifelse(r >= 0 & r < sd21, "c", NA)) %>%
        coalesce(ifelse(r < 0, "d", NA)) %>%
        as.factor(),
      .after = date
    ) %>%
    mutate(
      adx = get_adx(cbind(high, low, close))[, "adx"],
      turnover = val / mktcap_float,
      mktcost_ror = get_pctr(mktcost, close)
    ) %>%
    add_roc("close", c(5, 10, 20, 40, 60, 120, 240)) %>%
    add_mom("adx", c(5, 10)) %>%
    add_ma("turnover", c(20, 60, 120, 240)) %>%
    select(
      -c(
        open, high, low, close, vol, matches("^val"), mktcost,
        sd21, r, adx, turnover
      )
    ) %>%
    tail(60)

  lst <- list()
  lst[[symbol]] <- data
  return(lst)
}

saveRDS(data_list, file = data_list_path)
tsprint(glue("Generated features for {length(data_list)} stocks."))

plan(sequential)
