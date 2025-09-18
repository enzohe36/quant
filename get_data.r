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
indexcomp_path <- "indexcomp.csv"
log_path <- paste0("log/log_", format(now(), "%Y%m%d_%H%M%S"), ".log")

dir.create(data_dir)
dir.create(hist_dir)
dir.create(adjust_dir)
dir.create("log/")

end_date <- as_tradedate(now() - hours(16))

# get indexcomp for 000985
#   filter indexcomp to keep sh & sz stocks
#   write to files
indexcomp <- get_indexcomp("000985") %>%
  filter(str_detect(symbol, "^(0|3|6)")) %>%
  mutate(weight = weight / sum(weight))
write_csv(indexcomp, indexcomp_path)
glue("Index contains {nrow(indexcomp)} stocks.") %>%
  tsprint()

# get spot data
#   filter symbols to keep what's in indexcomp
#   merge with dividend data
spot <- get_spot() %>%
  filter(symbol %in% pull(indexcomp, symbol)) %>%
  left_join(get_div(end_date), by = "symbol") %>%
  mutate(date = end_date)
glue("Retrieved spot data for {nrow(spot)} stocks.") %>%
  tsprint()

# for each symbol
i <- 1
symbols <- pull(indexcomp, symbol)
fail_count <- foreach(
  symbol = symbols,
  .combine = "c"
) %dopar% {
  prog <- glue("{i}/{length(symbols)} {symbol}")
  fail_count <- 0
  spot_symbol <- filter(spot, symbol == !!symbol)

  # if hist file is present
  #   if max date is today's tradedate
  #     no action
  #   else if max date is one day before today's tradedate
  #     append spot data to file
  #   else
  #     download missing data & append to file
  # else
  #   download all data
  hist_path <- paste0(hist_dir, symbol, ".csv")
  if (file.exists(hist_path)) {
    hist <- read_csv(hist_path, show_col_types = FALSE)
    if (max(pull(hist, date)) >= end_date) {
      glue("{prog}: Data is up to date.") %>%
        tslog(log_path)
    } else if (max(pull(hist, date)) == as_tradedate(end_date - 1)) {
      hist <- bind_rows(
        hist,
        select(
          spot_symbol,
          date, open, high, low, close, volume, amount, turnover
        )
      ) %>%
        arrange(date)
      write_csv(hist, hist_path)
      glue("{prog}: Appended spot data.") %>%
        tslog(log_path)
    } else {
      start_date <- max(pull(hist, date)) + 1
      try_error <- try(
        new_data <- get_hist(symbol, start_date, end_date),
        silent = TRUE
      )
      if (inherits(try_error, "try-error")) {
        glue("{prog}: Failed to append missing hist data.") %>%
          tslog(log_path)
        fail_count <- fail_count + 1
      } else {
        hist <- bind_rows(hist, new_data) %>%
          arrange(date)
        write_csv(hist, hist_path)
        glue("{prog}: Appended missing hist data.") %>%
          tslog(log_path)
      }
      Sys.sleep(1)
    }
  } else {
    try_error <- try(
      hist <- get_hist(symbol, NULL, end_date),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      glue("{prog}: Failed to create new hist file.") %>%
        tslog(log_path)
      fail_count <- fail_count + 1
    } else {
      write_csv(hist, hist_path)
      glue("{prog}: Created new hist file.") %>%
        tslog(log_path)
    }
    Sys.sleep(1)
  }

  # if adjust file is present
  #   if exright date is no later than today's tradedate
  #     and exright date is later than max date in adjust file
  #     download all data
  #   else
  #     no action
  # else
  #   download all data
  adjust_path <- paste0(adjust_dir, symbol, ".csv")
  if (file.exists(adjust_path)) {
    adjust <- read_csv(adjust_path, show_col_types = FALSE)
    exright_date <- select(spot_symbol, exright_date) %>% pull()
    if (
      isTRUE(exright_date <= end_date & exright_date > max(pull(adjust, date)))
    ) {
      try_error <- try(
        adjust <- get_adjust(symbol),
        silent = TRUE
      )
      if (inherits(try_error, "try-error")) {
        glue("{prog}: Failed to replace adjust file.") %>%
          tslog(log_path)
        fail_count <- fail_count + 1
      } else {
        write_csv(adjust, adjust_path)
        glue("{prog}: Replaced adjust file.") %>%
          tslog(log_path)
      }
      Sys.sleep(1)
    } else {
      glue("{prog}: Data is up to date.") %>%
        tslog(log_path)
    }
  } else {
    try_error <- try(
      adjust <- get_adjust(symbol),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      glue("{prog}: Failed to create new adjust file.") %>%
        tslog(log_path)
      fail_count <- fail_count + 1
    } else {
      write_csv(adjust, adjust_path)
      glue("{prog}: Created new adjust file.") %>%
        tslog(log_path)
    }
    Sys.sleep(1)
  }

  i <- i + 1
  return(fail_count)
}
glue("Updated {length(symbols) - sum(fail_count)} stocks; {sum(fail_count)} failed.") %>%
  tsprint()
