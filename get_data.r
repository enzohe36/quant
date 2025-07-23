# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools
# conda activate myenv; Rscript get_data.r; Rscript get_data.r

rm(list = ls())

gc()

library(doFuture)
library(foreach)
library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"
dir.create(data_dir)
index_comp_path <- paste0(data_dir, "index_comp.csv")

# Define data acquisition parameters
period <- "daily"
end_date <- as_tradedate(now() - hours(16))
start_date_default <- NULL
adjust <- "hfq"

# Download list of stocks
index_comp <- foreach(
  index = c("000300", "000905", "000852", "932000"),
  .combine = "append"
) %dofuture% {
  list(get_index_comp(index))
} %>%
  rbindlist() %>%
  list(get_fundamentals(end_date)) %>%
  reduce(left_join, by = "symbol")
write.csv(index_comp, index_comp_path, quote = FALSE, row.names = FALSE)
tsprint(glue("Found {nrow(index_comp)} stocks."))

# Download historical stock data
count <- foreach(
  symbol = pull(index_comp, symbol),
  .combine = "c"
) %dofuture% {
  var <- c(
    "var", "data_path", "last_date", "start_date", "append_existing",
    "try_error", "data"
  )
  rm(list = var)

  data_path <- paste0(data_dir, symbol, ".csv")
  if (file.exists(data_path)) {
    last_date <- read_csv(data_path, show_col_types = FALSE) %>%
      pull(date) %>%
      last()
    if (end_date <= last_date) {
      return(1)
    } else {
      start_date <- last_date + days(1)
      append_existing <- TRUE
    }
  } else {
    start_date <- start_date_default
    append_existing <- FALSE
  }

  try_error <- try(
    data <- reduce(
      list(
        get_hist(symbol, period, start_date, end_date, adjust),
        get_val(symbol)
      ),
      left_join,
      by = "date"
    ),
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    return(0)
  } else {
    write_csv(data, data_path, append = append_existing)
    return(1)
  }
} %>%
  sum()
tsprint(glue("Retrieved historical data of {count} stocks."))

plan(sequential)
