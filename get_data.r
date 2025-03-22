# python -m aktools

rm(list = ls())

gc()

library(doFuture)
library(foreach)
library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(tidyverse)

# Load custom settings & helper functions
source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"

dir.create(data_dir)

industry_list_path <- "industry_list_sw2021.csv"
index_comp_path <- paste0(data_dir, "index_comp.csv")
index_path <- paste0(data_dir, "index.csv")
treasury_path <- paste0(data_dir, "treasury.csv")

# Load list of industries & subindustries
industry_list <- read_csv(industry_list_path)

# Download list of stocks
index_comp <- foreach(
  index = c("000300", "000905", "000852", "932000"),
  .combine = "append"
) %dofuture% {
  list(get_index_comp(index))
} %>%
  rbindlist() %>%
  list(get_industry()) %>%
  reduce(left_join, by = "symb") %>%
  mutate(
    industry = sapply(
      industry,
      function(x) filter(industry_list, symb == x) %>% pull(primary)
    )
  )
write.csv(index_comp, index_comp_path, quote = FALSE, row.names = FALSE)
tsprint(glue("Found {nrow(index_comp)} stocks."))

# Define data parameters
period <- "daily"
end_date <- as_tradedate(now() - hours(16))
# start_date <- end_date %m-% months(27)
start_date <- end_date %m-% months(33) # For testing only
adjust <- "qfq"

# Use CSI All Share Index as market benchmark
index <- get_index("000985", start_date, end_date)
write.csv(index, index_path, quote = FALSE, row.names = FALSE)

# Use 10-yr treasury as risk-free benchmark
treasury <- get_treasury(start_date)
write.csv(treasury, treasury_path, quote = FALSE, row.names = FALSE)

# Download historical stock data
count <- foreach(
  symb = index_comp$symb,
  .combine = "c"
) %dofuture% {
  rm(list = c("data_path", "last_date", "try_error", "data"))

  data_path <- paste0(data_dir, symb, ".csv")
  if (file.exists(data_path)) {
    last_date <- read_csv(data_path, show_col_types = FALSE) %>%
      pull(date) %>%
      last()
    if (end_date == last_date) return(1)
  }

  try_error <- try(
    data <- reduce(
      list(
        get_hist(symb, period, start_date, end_date, adjust),
        get_val(symb)
      ),
      left_join,
      by = "date"
    ),
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    return(0)
  } else {
    write.csv(data, data_path, quote = FALSE, row.names = FALSE)
    return(1)
  }
} %>%
  sum()
tsprint(glue("Retrieved historical data of {count} stocks."))

plan(sequential)
