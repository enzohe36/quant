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
adjust_dir <- paste0(data_dir, "adjust/")
mktcap_dir <- paste0(data_dir, "mktcap/")
val_dir <- paste0(data_dir, "val/")
spot_path <- paste0(data_dir, "spot.csv")

log_dir <- "logs/"
log_path <- paste0(log_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

dir.create(data_dir)
dir.create(hist_dir)
dir.create(adjust_dir)
dir.create(mktcap_dir)
dir.create(val_dir)
dir.create(log_dir)

end_date <- as_tradedate(now() - hours(16))

loop_get <- function(var, ...) {
  fail <- TRUE
  fail_count <- 1
  while (fail & fail_count < 3) {
    try_error <- try(
      data <- get(paste0("get_", var))(...),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      glue("Failed to retrieve {var} after {fail_count} tries.") %>%
        tsprint()
      fail_count <- fail_count + 1
    } else {
      return(data)
    }
  }
}

if (!file.exists(spot_path)) {
  symbols <- loop_get("symbols")
  susp <- loop_get("susp")
  spot <- loop_get("spot")
  adjust_change <- loop_get("adjust_change")
  shares_change <- loop_get("shares_change")
  val_change <- loop_get("val_change")
  spot <- symbols %>%
    left_join(susp, by = c("symbol", "date")) %>%
    left_join(spot, by = c("symbol", "date")) %>%
    left_join(adjust_change, by = c("symbol", "date")) %>%
    left_join(shares_change, by = c("symbol", "date")) %>%
    left_join(val_change, by = c("symbol", "date")) %>%
    mutate(
      delist = coalesce(delist, FALSE),
      susp = coalesce(susp, FALSE)
    )
  write_csv(spot, spot_path)
} else {
  spot <- read_csv(spot_path, show_col_types = FALSE)
  last_date <- max(pull(spot, date))
  if (last_date < end_date) {
    symbols <- loop_get("symbols")
    susp <- loop_get("susp")
    spot <- loop_get("spot")
    adjust_change <- loop_get("adjust_change")
    shares_change <- loop_get("shares_change")
    val_change <- loop_get("val_change")
    spot <- symbols %>%
      left_join(susp, by = c("symbol", "date")) %>%
      left_join(spot, by = c("symbol", "date")) %>%
      left_join(adjust_change, by = c("symbol", "date")) %>%
      left_join(shares_change, by = c("symbol", "date")) %>%
      left_join(val_change, by = c("symbol", "date")) %>%
      mutate(
        delist = coalesce(delist, FALSE),
        susp = coalesce(susp, FALSE)
      )
    write_csv(spot, spot_path)
  }
}
glue("Retrieved spot data for {nrow(spot)} symbols.") %>%
  tsprint()

symbols <- spot %>%
  filter(str_detect(symbol, "^(0|3|6)")) %>%
  # filter(!delist & !susp) %>%
  pull(symbol)
glue("Checking updates for {length(symbols)} symbols...") %>%
  tsprint()

step_count <- 1
fail_count <- foreach(
  symbol = symbols,
  .combine = "c"
) %dopar% {
  vars <- c(
    "adjust", "adjust_change_date", "adjust_path", "fail_count", "hist",
    "hist_path", "prog", "mktcap", "shares_change_date", "mktcap_path",
    "spot_symbol", "try_error", "val", "val_change_date", "val_path"
  )
  rm(list = vars)

  prog <- glue("{step_count}/{length(symbols)} {symbol}")
  fail_count <- 0

  hist_path <- paste0(hist_dir, symbol, ".csv")
  adjust_path <- paste0(adjust_dir, symbol, ".csv")
  mktcap_path <- paste0(mktcap_dir, symbol, ".csv")
  val_path <- paste0(val_dir, symbol, ".csv")

  spot_symbol <- filter(spot, symbol == !!symbol)

  # if hist file does not exist
  #   retrieve all hist
  #   create hist file
  # else
  #   if last date >= end date
  #     no update
  #   else if last date = end date - 1
  #     append spot to hist
  #   else
  #     retrieve new hist
  #     append new data to hist
  try_error <- try(
    if (!file.exists(hist_path)) {
      hist <- get_hist(symbol, NULL, end_date)
      write_csv(hist, hist_path)
      glue("{prog}: Created hist file.") %>%
        tslog(log_path)
    } else {
      hist <- read_csv(hist_path, show_col_types = FALSE)
      last_date <- max(pull(hist, date))
      if (
        isTRUE(last_date >= end_date | spot_symbol$delist | spot_symbol$susp)
      ) {
        glue("{prog}: No update.") %>%
          tslog(log_path)
      } else if (isTRUE(last_date == as_tradedate(end_date - 1))) {
        hist <- bind_rows(hist, select(spot_symbol, names(hist)))
        write_csv(hist, hist_path)
        glue("{prog}: Appended spot to hist.") %>%
          tslog(log_path)
      } else {
        hist <- bind_rows(hist, get_hist(symbol, last_date + 1, end_date))
        write_csv(hist, hist_path)
        glue("{prog}: Appended new data to hist.") %>%
          tslog(log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{prog}: Failed to retrieve hist.") %>%
      tslog(log_path)
    fail_count <- fail_count + 1
  }

  # if adjust file does not exist
  #   retrieve adjust
  #   create adjust file
  # else
  #   if last date < adjust change date <= end date
  #     retrieve adjust
  #     replace adjust file
  #   else
  #     no update
  try_error <- try(
    if (!file.exists(adjust_path)) {
      adjust <- get_adjust(symbol)
      write_csv(adjust, adjust_path)
      glue("{prog}: Created adjust file.") %>%
        tslog(log_path)
    } else {
      adjust <- read_csv(adjust_path, show_col_types = FALSE)
      last_date <- max(pull(adjust, date))
      adjust_change_date <- pull(spot_symbol, adjust_change_date)
      if (
        isTRUE(last_date < adjust_change_date & adjust_change_date <= end_date)
      ) {
        adjust <- get_adjust(symbol)
        write_csv(adjust, adjust_path)
        glue("{prog}: Replaced adjust file.") %>%
          tslog(log_path)
      } else {
        glue("{prog}: No update.") %>%
          tslog(log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{prog}: Failed to retrieve adjust.") %>%
      tslog(log_path)
    fail_count <- fail_count + 1
  }

  # if mktcap file does not exist
  #   retrieve mktcap
  #   create mktcap file
  # else
  #   if last date < mktcap change date <= end date
  #     retrieve mktcap
  #     replace mktcap file
  #   else
  #     no update
  try_error <- try(
    if (!file.exists(mktcap_path)) {
      mktcap <- get_mktcap(symbol)
      write_csv(mktcap, mktcap_path)
      glue("{prog}: Created mktcap file.") %>%
        tslog(log_path)
    } else {
      mktcap <- read_csv(mktcap_path, show_col_types = FALSE)
      last_date <- max(pull(mktcap, date))
      shares_change_date <- pull(spot_symbol, shares_change_date)
      if (
        isTRUE(last_date < shares_change_date & shares_change_date <= end_date)
      ) {
        mktcap <- get_mktcap(symbol)
        write_csv(mktcap, mktcap_path)
        glue("{prog}: Replaced mktcap file.") %>%
          tslog(log_path)
      } else {
        glue("{prog}: No update.") %>%
          tslog(log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{prog}: Failed to retrieve mktcap.") %>%
      tslog(log_path)
    fail_count <- fail_count + 1
  }

  # if val file does not exist
  #   retrieve val
  #   create val file
  # else
  #   if last date < val change date <= end date
  #     retrieve val
  #     replace val file
  #   else
  #     no update
  try_error <- try(
    if (!file.exists(val_path)) {
      val <- get_val(symbol)
      write_csv(val, val_path)
      glue("{prog}: Created val file.") %>%
        tslog(log_path)
    } else {
      val <- read_csv(val_path, show_col_types = FALSE)
      last_date <- max(pull(val, val_change_date))
      val_change_date <- pull(spot_symbol, val_change_date)
      if (
        isTRUE(last_date < val_change_date & val_change_date <= end_date)
      ) {
        val <- get_val(symbol)
        write_csv(val, val_path)
        glue("{prog}: Replaced val file.") %>%
          tslog(log_path)
      } else {
        glue("{prog}: No update.") %>%
          tslog(log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{prog}: Failed to retrieve val.") %>%
      tslog(log_path)
    fail_count <- fail_count + 1
  }

  step_count <- step_count + 1
  return(fail_count)
} %>%
  sum()

glue("Finished checking updates; {fail_count} failed.") %>%
  tsprint()
