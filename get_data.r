# python -m aktools

rm(list = ls())

library(doFuture)
library(foreach)
library(RCurl)
library(jsonlite)
library(glue)
library(data.table)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

plan(multisession, workers = availableCores() - 1)

index_list <- c("000300", "000905", "000852", "932000")
data_dir <- "data/"

dir.create(data_dir)

symbol_dict <- foreach(
  index = index_list,
  .combine = "append"
) %dofuture% {
  symbol_dict_path <- paste0(data_dir, "symbol_dict_", index, ".csv")
  symbol_dict <- get_index_comp(index)
  write.csv(symbol_dict, symbol_dict_path, quote = FALSE, row.names = FALSE)
  return(list(symbol_dict))
} %>%
  rbindlist()

tsprint(glue("Found {nrow(symbol_dict)} stocks."))

period <- "daily"
end_date <- as_tradedate(now() - hours(16))
adjust <- "qfq"

count <- foreach(
  symbol = symbol_dict$symbol,
  .combine = "c"
) %dofuture% {
  data_path <- paste0(data_dir, symbol, ".csv")
  if (file.exists(data_path)) {
    hist_old <- read_csv(data_path, show_col_types = FALSE) %>%
      filter(row_number() == n()) %>%
      select(date, open, high, low, close, vol, val)
  }

  for (i in 1:2) {
    if (exists("hist_old")) {
      if (end_date == hist_old$date) return(1)

      start_date <- hist_old$date
      hist <- get_hist(symbol, period, start_date, end_date, adjust)

      if (all(hist[1, ] == hist_old)) {
        hist <- hist[-1, ]
        append_tf <- TRUE
        break
      } else {
        rm("hist_old")
      }
    } else {
      start_date <- end_date %m-% months(15)
      hist <- get_hist(symbol, period, start_date, end_date, adjust)
      append_tf <- FALSE
      break
    }
  }

  hist_valuation <- get_hist_valuation(symbol)

  hist_fundflow <- get_hist_fundflow(
    symbol, symbol_dict$mkt_abbr[symbol_dict$symbol == symbol]
  )

  hist_cost <- get_hist_cost(symbol, adjust)

  data <- reduce(
    list(hist, hist_valuation, hist_fundflow, hist_cost), left_join, by = "date"
  )
  if (append_tf) {
    write.table(
      data, data_path,
      append = TRUE,
      quote = FALSE,
      sep = ",",
      row.names = FALSE,
      col.names = FALSE
    )
  } else {
    write.csv(data, data_path, quote = FALSE, row.names = FALSE)
  }

  return(1)
} %>%
  sum()

tsprint(glue("Retrieved historical data of {count} stocks."))

plan(sequential)
