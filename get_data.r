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
hist_dir <- paste0(data_dir, "hist/")
shares_dir <- paste0(data_dir, "shares/")
adjust_dir <- paste0(data_dir, "adjust/")
spot_path <- paste0(data_dir, "spot.csv")

log_dir <- "logs/"
log_path <- paste0(log_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

dir.create(data_dir)
dir.create(hist_dir)
dir.create(shares_dir)
dir.create(adjust_dir)
dir.create(log_dir)

end_date <- as_tradedate(now() - hours(16))

combine_spot <- function() {
  get_spot() %>%
    right_join(get_symbols(), by = c("symbol", "date")) %>%
    left_join(get_susp(end_date), by = c("symbol", "date")) %>%
    left_join(get_div(end_date), by = "symbol") %>%
    mutate(
      delist = coalesce(delist, FALSE),
      susp = coalesce(susp, FALSE)
    )
}

if (file.exists(spot_path)) {
  spot <- read_csv(spot_path, show_col_types = FALSE)
  last_date <- max(pull(spot, date))
  if (last_date < end_date) {
    spot <- combine_spot()
    write_csv(spot, spot_path)
  }
} else {
  spot <- combine_spot()
  write_csv(spot, spot_path)
}
glue("Retrieved spot data for {nrow(spot)} symbols.") %>%
  tsprint()

symbols <- spot %>%
  filter(str_detect(symbol, "^(0|3|6)")) %>%
  filter(!delist & !susp) %>%
  pull(symbol)
glue("Checking updates for {length(symbols)} symbols...") %>%
  tsprint()

step_count <- 1
fail_count <- foreach(
  symbol = symbols,
  .combine = "c"
) %dopar% {
  vars <- c(
    "adjust", "adjust_path", "exright_date", "fail_count", "hist", "hist_path",
    "prog", "shares", "shares_path", "spot_symbol", "try_error"
  )
  rm(list = vars)

  prog <- glue("{step_count}/{length(symbols)} {symbol}")
  fail_count <- 0

  hist_path <- paste0(hist_dir, symbol, ".csv")
  shares_path <- paste0(shares_dir, symbol, ".csv")
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
      hist <- bind_rows(hist, select(spot_symbol, names(hist)))
      write_csv(hist, hist_path)
      glue("{prog}: Appended spot to hist.") %>%
        tslog(log_path)
    } else {
      try_error <- try(
        hist <- bind_rows(hist, get_hist(symbol, last_date + 1, end_date)),
        silent = TRUE
      )
      if (inherits(try_error, "try-error")) {
        glue("{prog}: Failed to retrieve hist.") %>%
          tslog(log_path)
        fail_count <- fail_count + 1
      } else {
        write_csv(hist, hist_path)
        glue("{prog}: Updated hist.") %>%
          tslog(log_path)
      }
    }
  } else {
    try_error <- try(
      hist <- get_hist(symbol, NULL, end_date),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      glue("{prog}: Failed to retrieve hist.") %>%
        tslog(log_path)
      fail_count <- fail_count + 1
    } else {
      write_csv(hist, hist_path)
      glue("{prog}: Created hist file.") %>%
        tslog(log_path)
    }
  }

  # Same logic as hist
  if (file.exists(shares_path)) {
    shares <- read_csv(shares_path, show_col_types = FALSE)
    last_date <- max(pull(shares, date))
    if (last_date >= end_date) {
      glue("{prog}: No update.") %>%
        tslog(log_path)
    } else if (last_date == as_tradedate(end_date - 1)) {
      shares <- bind_rows(shares, select(spot_symbol, names(shares)))
      write_csv(shares, shares_path)
      glue("{prog}: Appended spot to shares.") %>%
        tslog(log_path)
    } else {
      try_error <- try(
        shares <- shares %>%
          full_join(get_shares(symbol), by = names(.)) %>%
          full_join(select(spot_symbol, names(.)), by = names(.)) %>%
          distinct(date, .keep_all = TRUE) %>%
          arrange(date),
        silent = TRUE
      )
      if (inherits(try_error, "try-error")) {
        glue("{prog}: Failed to retrieve shares.") %>%
          tslog(log_path)
        fail_count <- fail_count + 1
      } else {
        write_csv(shares, shares_path)
        glue("{prog}: Updated shares.") %>%
          tslog(log_path)
      }
    }
  } else {
    try_error <- try(
      shares <- get_shares(symbol) %>%
        full_join(select(spot_symbol, names(.)), by = names(.)) %>%
        distinct(date, .keep_all = TRUE) %>%
        arrange(date),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      glue("{prog}: Failed to retrieve shares.") %>%
        tslog(log_path)
      fail_count <- fail_count + 1
    } else {
      write_csv(shares, shares_path)
      glue("{prog}: Created shares file.") %>%
        tslog(log_path)
    }
  }

  # if adjust file exists
  #   if exright_date > last_date & exright_date <= end_date
  #     replace adjust file
  #   else
  #     no update
  # else
  #   create adjust file
  if (file.exists(adjust_path)) {
    adjust <- read_csv(adjust_path, show_col_types = FALSE)
    last_date <- max(pull(adjust, date))
    exright_date <- select(spot_symbol, exright_date) %>% pull()
    if (isTRUE(exright_date > last_date & exright_date <= end_date)) {
      try_error <- try(
        adjust <- get_adjust(symbol),
        silent = TRUE
      )
      if (inherits(try_error, "try-error")) {
        glue("{prog}: Failed to retrieve adjust.") %>%
          tslog(log_path)
        fail_count <- fail_count + 1
      } else {
        write_csv(adjust, adjust_path)
        glue("{prog}: Updated adjust.") %>%
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
      glue("{prog}: Failed to retrieve adjust.") %>%
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
