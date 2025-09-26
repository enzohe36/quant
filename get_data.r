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
shares_dir <- paste0(data_dir, "shares/")
val_dir <- paste0(data_dir, "val/")
spot_path <- paste0(data_dir, "spot.csv")

log_dir <- "logs/"
log_path <- paste0(log_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

dir.create(data_dir)
dir.create(hist_dir)
dir.create(adjust_dir)
dir.create(shares_dir)
dir.create(val_dir)
dir.create(log_dir)

end_date <- as_tradedate(now() - hours(16))

combine_spot <- function() {
  get_symbols() %>%
    left_join(get_susp(), by = c("symbol", "date")) %>%
    left_join(get_spot(), by = c("symbol", "date")) %>%
    left_join(get_div(end_date), by = c("symbol", "date")) %>%
    left_join(get_shares_change(), by = c("symbol", "date")) %>%
    left_join(get_val_change(), by = c("symbol", "date")) %>%
    mutate(
      delist = coalesce(delist, FALSE),
      susp = coalesce(susp, FALSE)
    )
}

if (!file.exists(spot_path)) {
  spot <- combine_spot()
  write_csv(spot, spot_path)
} else {
  spot <- read_csv(spot_path, show_col_types = FALSE)
  last_date <- max(pull(spot, date))
  if (last_date < end_date) {
    spot <- combine_spot()
    write_csv(spot, spot_path)
  }
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
    "last_date", "prog", "shares", "shares_path", "spot_symbol", "try_error"
  )
  rm(list = vars)

  prog <- glue("{step_count}/{length(symbols)} {symbol}")
  fail_count <- 0

  hist_path <- paste0(hist_dir, symbol, ".csv")
  adjust_path <- paste0(adjust_dir, symbol, ".csv")
  shares_path <- paste0(shares_dir, symbol, ".csv")
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
  #     update hist file
  #   else
  #     retrieve new hist
  #     append new data to old data
  #     update hist file
  if (!file.exists(hist_path)) {
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
  } else {
    hist <- read_csv(hist_path, show_col_types = FALSE)
    last_date <- max(pull(hist, date))
    if (isTRUE(last_date >= end_date)) {
      glue("{prog}: No update.") %>%
        tslog(log_path)
    } else if (isTRUE(last_date == end_date - 1)) {
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
        glue("{prog}: Appended new data to hist.") %>%
          tslog(log_path)
      }
    }
  }

  # if adjust file does not exist
  #   retrieve adjust
  #   create adjust file
  # else
  #   if last date < exright date <= end date
  #     retrieve adjust
  #     replace adjust file
  #   else
  #     no update
  if (!file.exists(adjust_path)) {
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
  } else {
    adjust <- read_csv(adjust_path, show_col_types = FALSE)
    last_date <- max(pull(adjust, date))
    exright_date <- pull(spot_symbol, exright_date)
    if (isTRUE(last_date < exright_date & exright_date <= end_date)) {
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
        glue("{prog}: Replaced adjust file.") %>%
          tslog(log_path)
      }
    } else {
      glue("{prog}: No update.") %>%
        tslog(log_path)
    }
  }

  # if shares file does not exist
  #   retrieve shares
  #   duplicate last line & change date to end date
  #   keep distinct dates
  #   create shares file
  # else
  #   if last date < shares change date <= end date
  #     retrieve shares
  #     full join with old data
  #     arrange by date
  #     duplicate last line & change date to end date
  #     keep distinct dates
  #     replace shares file
  #   else
  #     no update
  if (!file.exists(shares_path)) {
    try_error <- try(
      shares <- get_shares(symbol) %>%
        bind_rows(mutate(last(.), date = end_date)) %>%
        distinct(date, .keep_all = TRUE),
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
  } else {
    shares <- read_csv(shares_path, show_col_types = FALSE)
    last_date <- max(pull(shares, date))
    shares_change_date <- pull(spot_symbol, shares_change_date)
    if (
      isTRUE(last_date < shares_change_date & shares_change_date <= end_date)
    ) {
      try_error <- try(
        shares <- get_shares(symbol) %>%
          full_join(shares, by = names(.)) %>%
          arrange(date) %>%
          bind_rows(mutate(last(.), date = end_date)) %>%
          distinct(date, .keep_all = TRUE),
        silent = TRUE
      )
      if (inherits(try_error, "try-error")) {
        glue("{prog}: Failed to retrieve shares.") %>%
          tslog(log_path)
        fail_count <- fail_count + 1
      } else {
        write_csv(shares, shares_path)
        glue("{prog}: Replaced shares file.") %>%
          tslog(log_path)
      }
    } else {
      glue("{prog}: No update.") %>%
        tslog(log_path)
    }
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
  if (!file.exists(val_path)) {
    try_error <- try(
      val <- get_val(symbol),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      glue("{prog}: Failed to retrieve val.") %>%
        tslog(log_path)
      fail_count <- fail_count + 1
    } else {
      write_csv(val, val_path)
      glue("{prog}: Created val file.") %>%
        tslog(log_path)
    }
  } else {
    val <- read_csv(val_path, show_col_types = FALSE)
    last_date <- max(pull(val, date))
    if (ends_with(last_date, "09-30"))


    val_change_date <- pull(spot_symbol, val_change_date)
    end_quarter <- quarter(end_date %m-% months(3), "date_last")
    if (isTRUE(last_date < quarter(val_change_date %m-% months(3), "date_last") & val_change_date <= end_date)) {
      try_error <- try(
        val <- get_val(symbol),
        silent = TRUE
      )
      if (inherits(try_error, "try-error")) {
        glue("{prog}: Failed to retrieve val.") %>%
          tslog(log_path)
        fail_count <- fail_count + 1
      } else {
        write_csv(val, val_path)
        glue("{prog}: Replaced val file.") %>%
          tslog(log_path)
      }
    } else {
      glue("{prog}: No update.") %>%
        tslog(log_path)
    }
  }

  step_count <- step_count + 1
  return(fail_count)
} %>%
  sum()

glue("Finished checking updates; {fail_count} failed.") %>%
  tsprint()
