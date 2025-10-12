# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools

rm(list = ls())

gc()

library(foreach)
library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

################################################################################

data_dir <- "data/"
hist_dir <- paste0(data_dir, "hist/")
adjust_dir <- paste0(data_dir, "adjust/")
mc_dir <- paste0(data_dir, "mc/")
val_dir <- paste0(data_dir, "val/")
spot_path <- paste0(data_dir, "spot.csv")

log_dir <- "logs/"
log_path <- paste0(log_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

dir.create(data_dir)
dir.create(hist_dir)
dir.create(adjust_dir)
dir.create(mc_dir)
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
      glue("Error retrieving {var} after {fail_count} try(s).") %>%
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

out <- foreach(
  symbol = symbols,
  .combine = "c"
) %dopar% {
  vars <- c(
    "hist", "hist_path", "adjust", "adjust_path", "adjust_change_date",
    "mc", "mc_path", "shares_change_date",
    "val", "val_path", "val_change_date", "spot_symbol",
    "try_error", "last_date"
  )
  rm(list = vars)

  hist_path <- paste0(hist_dir, symbol, ".csv")
  adjust_path <- paste0(adjust_dir, symbol, ".csv")
  mc_path <- paste0(mc_dir, symbol, ".csv")
  val_path <- paste0(val_dir, symbol, ".csv")

  spot_symbol <- filter(spot, symbol == !!symbol)

  # if hist file does not exist
  #   retrieve all hist
  #   create hist file
  # else
  #   if last date >= end date
  #     no action
  #   else if last date = end date - 1
  #     append spot to hist
  #   else
  #     retrieve new hist
  #     append new data to hist
  try_error <- try(
    if (!file.exists(hist_path)) {
      hist <- get_hist(symbol, NULL, end_date)
      write_csv(hist, hist_path)
      glue("{symbol}: Created hist file.") %>%
        tslog(log_path)
    } else {
      hist <- read_csv(hist_path, show_col_types = FALSE)
      last_date <- max(pull(hist, date))
      if (
        last_date >= end_date |
          pull(spot_symbol, delist) |
          pull(spot_symbol, susp)
      ) {
      } else if (isTRUE(last_date == as_tradedate(end_date - 1))) {
        hist <- bind_rows(hist, select(spot_symbol, names(hist)))
        write_csv(hist, hist_path)
        glue("{symbol}: Appended spot to hist.") %>%
          tslog(log_path)
      } else {
        hist <- bind_rows(hist, get_hist(symbol, last_date + 1, end_date))
        write_csv(hist, hist_path)
        glue("{symbol}: Appended new data to hist.") %>%
          tslog(log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{symbol}: Error retrieving hist.") %>%
      tslog(log_path)
  }

  # if adjust file does not exist
  #   retrieve adjust
  #   create adjust file
  # else
  #   if last date < adjust change date <= end date
  #     retrieve adjust
  #     replace adjust file
  #   else
  #     no action
  try_error <- try(
    if (!file.exists(adjust_path)) {
      adjust <- get_adjust(symbol)
      write_csv(adjust, adjust_path)
      glue("{symbol}: Created adjust file.") %>%
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
        glue("{symbol}: Replaced adjust file.") %>%
          tslog(log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{symbol}: Error retrieving adjust.") %>%
      tslog(log_path)
  }

  # if mc file does not exist
  #   retrieve mc
  #   create mc file
  # else
  #   if last date < mc change date <= end date
  #     retrieve mc
  #     replace mc file
  #   else
  #     no action
  try_error <- try(
    if (!file.exists(mc_path)) {
      mc <- get_mc(symbol)
      write_csv(mc, mc_path)
      glue("{symbol}: Created mc file.") %>%
        tslog(log_path)
    } else {
      mc <- read_csv(mc_path, show_col_types = FALSE)
      last_date <- max(pull(mc, date))
      shares_change_date <- pull(spot_symbol, shares_change_date)
      if (
        isTRUE(last_date < shares_change_date & shares_change_date <= end_date)
      ) {
        mc <- get_mc(symbol)
        write_csv(mc, mc_path)
        glue("{symbol}: Replaced mc file.") %>%
          tslog(log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{symbol}: Error retrieving mc.") %>%
      tslog(log_path)
  }

  # if val file does not exist
  #   retrieve val
  #   create val file
  # else
  #   if last date < val change date <= end date
  #     retrieve val
  #     replace val file
  #   else
  #     no action
  try_error <- try(
    if (!file.exists(val_path)) {
      val <- get_val(symbol)
      write_csv(val, val_path)
      glue("{symbol}: Created val file.") %>%
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
        glue("{symbol}: Replaced val file.") %>%
          tslog(log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{symbol}: Error retrieving val.") %>%
      tslog(log_path)
  }
} %>%
  sum()
