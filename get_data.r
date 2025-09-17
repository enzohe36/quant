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

# plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"
dir.create(data_dir)

adjust_factors_dir <- "adjust_factors/"
dir.create(adjust_factors_dir)

indexcomp_path <- "indexcomp.csv"
log_path <- paste0("logs/log_", format(now(), "%Y%m%d_%H%M%S"), ".txt")

# Define data acquisition parameters
period <- "daily"
end_date <- as_tradedate(now() - hours(16))
start_date_default <- NULL
adjust <- "hfq"

# Download list of stocks
index <- "000985"
indexcomp <- get_indexcomp(index) %>%
  filter(str_detect(symbol, "^(0|3|6)")) %>%
  mutate(index_weight = index_weight / sum(index_weight))
write.csv(indexcomp, indexcomp_path, quote = FALSE, row.names = FALSE)
glue("Index contains {nrow(indexcomp)} stocks.") %>%
  tsprint()

symbols <- pull(indexcomp, symbol)
glue("Retrieving data for {length(symbols)} stocks.") %>%
  tsprint()

# TODO: Get dividend
# TODO: If exright date is between last entry on file and today, update adjust factors

i <- 1
count <- foreach(
  symbol = symbols,
  .combine = "c"
) %dopar% {
  var <- c(
    "adjust_factors", "adjust_factors_path", "try_error"
  )
  rm(list = var)

  adjust_factors_path <- paste0(adjust_factors_dir, symbol, ".csv")
  try_error <- try(
    adjust_factors <- get_adjust_factors(symbol, NULL, NULL, adjust),
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{i}/{length(symbols)}: Failed to retrieve adjust factors for {symbol}.") %>%
      tslog(log_path)
    i <- i + 1
    return(1)
  } else {
    write_csv(adjust_factors, adjust_factors_path)
    glue("{i}/{length(symbols)}: Retrieved adjust factors for {symbol}.") %>%
      tslog(log_path)
    Sys.sleep(1)
    i <- i + 1
    return(0)
  }
}
glue("Finished retrieving adjust factors; skipped {sum(count)} stocks.") %>%
  tsprint()

# TODO: Get spot data
# TODO: If the last date in all files is more than one tradeday before today, get historical data
# TODO: If else, update with spot data

i <- 1
count <- foreach(
  symbol = symbols,
  .combine = "c"
) %dopar% {
  var <- c(
    "append_existing", "data", "data_path", "start_date", "try_error"
  )
  rm(list = var)

  data_path <- paste0(data_dir, symbol, ".csv")
  if (file.exists(data_path)) {
    last_date <- read_csv(data_path, show_col_types = FALSE) %>%
      pull(date) %>%
      last()
    if (end_date <= last_date) {
      glue("{i}/{length(symbols)}: {symbol} is already up to date.") %>%
        tslog(log_path)
      i <- i + 1
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
    data <- get_hist(symbol, start_date, end_date, NULL),
    # TODO: get valuation data & write to separate files
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{i}/{length(symbols)}: Failed to retrieve historical data for {symbol}.") %>%
      tslog(log_path)
    i <- i + 1
    return(1)
  } else {
    write_csv(data, data_path, append = append_existing)
    glue("{i}/{length(symbols)}: Retrieved historical data for {symbol}.") %>%
      tslog(log_path)
    Sys.sleep(1)
    i <- i + 1
    return(0)
  }
}
glue("Finished retrieving historical data; skipped {sum(count)} stocks.") %>%
  tsprint()
