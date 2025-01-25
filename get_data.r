# python -m aktools

rm(list = ls())

gc()

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
index_comp_path <- paste0(data_dir, "index_comp.csv")

dir.create(data_dir)

# Find constituents of CSI 300, 500, 1000 & 2000
index_comp <- foreach(
  index = index_list,
  .combine = "append"
) %dofuture% {
  list(get_index_comp(index))
} %>%
  rbindlist()
write.csv(index_comp, index_comp_path, quote = FALSE, row.names = FALSE)
tsprint(glue("Found {nrow(index_comp)} stocks."))

period <- "daily"
end_date <- as_tradedate(now() - hours(16))
adjust <- "qfq"

# Download historical data for listed stocks
count <- foreach(
  symbol = index_comp$symbol,
  .combine = "c"
) %dofuture% {
  rm(
    list = c(
      "append_existing", "data", "data_path", "hist", "hist_fundflow",
      "hist_mktcost", "hist_valuation", "i", "start_date"
    )
  )

  # If file exists, read last entry
  data_path <- paste0(data_dir, symbol, ".csv")
  if (file.exists(data_path)) {
    hist_old <- read_csv(data_path, show_col_types = FALSE) %>%
      filter(row_number() == n()) %>%
      select(date, open, high, low, close, vol, val)
  }

  # Get price, trading volume & value
  for (i in 1:2) {
    if (exists("hist_old")) {
      # Skip if last entry is up to date
      if (end_date == hist_old$date) return(1)

      # If split-adjusted price is the same, append to existing file...
      start_date <- hist_old$date
      hist <- get_hist(symbol, period, start_date, end_date, adjust)
      if (all(hist[1, ] == hist_old)) {
        hist <- hist[-1, ]
        append_existing <- TRUE
        break
      } else {
        rm("hist_old")
      }
    } else {
      # ... else replace entire file
      start_date <- end_date %m-% months(27)
      hist <- get_hist(symbol, period, start_date, end_date, adjust)
      append_existing <- FALSE
      break
    }
  }

  # Get valuation metrics
  hist_valuation <- get_hist_valuation(symbol)

  # Get fund flow data
  hist_fundflow <- get_hist_fundflow(
    symbol, index_comp$mkt[index_comp$symbol == symbol]
  )

  # Get market cost data
  hist_mktcost <- get_hist_mktcost(symbol, adjust)

  # Combine data & write to file
  data <- reduce(
    list(hist, hist_valuation, hist_fundflow, hist_mktcost),
    left_join,
    by = "date"
  )
  if (append_existing) {
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
