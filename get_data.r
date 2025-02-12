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
indcomp_path <- paste0(data_dir, "indcomp.csv")

dir.create(data_dir)

# Find constituents of CSI 300, 500, 1000 & 2000
indcomp <- foreach(
  index = index_list,
  .combine = "append"
) %dofuture% {
  list(get_index_comp(index))
} %>%
  rbindlist()

write.csv(indcomp, indcomp_path, quote = FALSE, row.names = FALSE)
tsprint(glue("Found {nrow(indcomp)} stocks."))

end_date <- as_tradedate(now() - hours(16))
start_date <- end_date %m-% months(27)
adjust <- "qfq"

# Download historical data for listed stocks
count <- foreach(
  symbol = indcomp$symbol,
  .combine = "c"
) %dofuture% {
  rm(list = c("data_path", "last_date", "data"))
  data_path <- paste0(data_dir, symbol, ".csv")
  if (file.exists(data_path)) {
    last_date <- last(read_csv(data_path, show_col_types = FALSE)$date)
    if (end_date == last_date) return(1)
  }
  data <- reduce(
    list(
      get_hist(symbol, start_date, end_date, adjust),
      get_hist_valuation(symbol),
      get_hist_fundflow(symbol, indcomp$mkt[indcomp$symbol == symbol]),
      get_hist_mktcost(symbol, adjust)
    ),
    left_join,
    by = "date"
  )
  write.csv(data, data_path, quote = FALSE, row.names = FALSE)
  return(1)
} %>%
  sum()

tsprint(glue("Retrieved historical data of {count} stocks."))

plan(sequential)
