# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools
# conda activate myenv; Rscript get_data.r; Rscript get_data.r

rm(list = ls())

gc()

library(foreach)
library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

data_dir <- "data/"
hist_dir <- "data/hist/"
adjust_dir <- "data/adjust/"
delist_path <- "delist.csv"
log_path <- paste0("log/log_", format(now(), "%Y%m%d_%H%M%S"), ".log")

dir.create(data_dir)
dir.create(hist_dir)
dir.create(adjust_dir)
dir.create("log/")

end_date <- as_tradedate(now() - hours(16))

# get spot data
#   filter to keep sh & sz stocks
#   merge with dividend data
spot <- get_spot() %>%
  filter(str_detect(symbol, "^(0|3|6)")) %>%
  left_join(get_div(end_date), by = "symbol") %>%
  mutate(date = end_date)
glue("Retrieved spot data for {nrow(spot)} stocks.") %>%
  tsprint()

# get suspended stocks
#   filter & keep symbols that are currently suspended
susp <- get_susp(end_date) %>%
  filter(susp_start <= end_date & (susp_end >= end_date | is.na(susp_end))) %>%
  pull(symbol)
glue("{length(susp)} stocks are suspended.") %>%
  tsprint()

# get delisted stocks
# delist <- read_csv(delist_path, show_col_types = FALSE) %>%
#   pull(symbol)
delist <- as.character(c())
glue("{length(delist)} symbols are delisted/not in use.") %>%
  tsprint()

# generate symbols from spot data
#   filter out suspended & delisted symbols
symbols <- sprintf(
  "%06d",
  c(
    000001:(pull(spot, symbol) %>% .[str_detect(., "^00")] %>% max()),
    300001:(pull(spot, symbol) %>% .[str_detect(., "^30")] %>% max()),
    600000:(pull(spot, symbol) %>% .[str_detect(., "^60")] %>% max()),
    688001:(pull(spot, symbol) %>% .[str_detect(., "^68")] %>% max())
  )
) %>%
  .[!. %in% c(susp, delist)]
glue("Updating {length(symbols)} stocks...") %>%
  tsprint()

step_count <- 1
fail_count <- foreach(
  symbol = symbols,
  .combine = "c"
) %dopar% {
  vars <- c(
    "adjust", "adjust_path", "exright_date", "fail_count", "hist", "hist_path",
    "last_adjust", "last_date", "new_data", "prog", "spot_symbol", "try_error"
  )
  rm(list = vars)

  prog <- glue("{step_count}/{length(symbols)} {symbol}")
  fail_count <- 0
  hist_path <- paste0(hist_dir, symbol, ".csv")
  adjust_path <- paste0(adjust_dir, symbol, ".csv")
  spot_symbol <- filter(spot, symbol == !!symbol)

  # if hist file exists
  #   if last_date >= end_date
  #     no update
  #   if last_date = as_tradedate(end_date - 1)
  #     append spot data
  #   else
  #     append hist data
  # else
  #   create new hist file
  if (file.exists(hist_path)) {
    hist <- read_csv(hist_path, show_col_types = FALSE)
    last_date <- max(pull(hist, date))
    if (last_date >= end_date) {
      glue("{prog}: No update.") %>%
        tslog(log_path)
    } else if (last_date == as_tradedate(end_date - 1)) {
      hist <- bind_rows(
        hist,
        select(spot_symbol, date, open, high, low, close, volume, amount, to)
      )
      write_csv(hist, hist_path)
      glue("{prog}: Appended spot data.") %>%
        tslog(log_path)
    } else {
      try_error <- try(
        new_data <- get_hist(symbol, last_date + 1, end_date),
        silent = TRUE
      )
      if (inherits(try_error, "try-error")) {
        glue("{prog}: Failed to retrieve hist data.") %>%
          tslog(log_path)
        fail_count <- fail_count + 1
      } else {
        hist <- bind_rows(hist, new_data)
        write_csv(hist, hist_path)
        glue("{prog}: Appended hist data.") %>%
          tslog(log_path)
      }
    }
  } else {
    try_error <- try(
      hist <- get_hist(symbol, NULL, end_date),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      glue("{prog}: Failed to retrieve hist data.") %>%
        tslog(log_path)
      fail_count <- fail_count + 1
    } else {
      write_csv(hist, hist_path)
      glue("{prog}: Created hist file.") %>%
        tslog(log_path)
    }
  }

  # if adjust file exists
  #   if exright_date > last_adjust & exright_date <= end_date
  #     replace adjust file
  #   else
  #     no update
  # else
  #   create adjust file
  if (file.exists(adjust_path)) {
    adjust <- read_csv(adjust_path, show_col_types = FALSE)
    last_adjust <- max(pull(adjust, date))
    exright_date <- select(spot_symbol, exright_date) %>% pull()
    if (isTRUE(exright_date > last_adjust & exright_date <= end_date)) {
      try_error <- try(
        adjust <- get_adjust(symbol),
        silent = TRUE
      )
      if (inherits(try_error, "try-error")) {
        glue("{prog}: Failed to retrieve adjust data.") %>%
          tslog(log_path)
        fail_count <- fail_count + 1
      } else {
        write_csv(adjust, adjust_path)
        glue("{prog}: Replaced adjust file.") %>%
          tslog(log_path)
      }
    } else {
      glue("{prog}: No update.") %>%
        tslog(log_path)
    }
  } else {
    try_error <- try(
      adjust <- get_adjust(symbol),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      glue("{prog}: Failed to retrieve adjust data.") %>%
        tslog(log_path)
      fail_count <- fail_count + 1
    } else {
      write_csv(adjust, adjust_path)
      glue("{prog}: Created adjust file.") %>%
        tslog(log_path)
    }
  }

  step_count <- step_count + 1
  return(fail_count)
} %>%
  sum()
glue("Finished checking updates; {fail_count} failed.") %>%
  tsprint()
