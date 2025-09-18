rm(list = ls())

gc()

library(doFuture)
library(foreach)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

plan(multisession, workers = availableCores() - 1)

hist_dir <- "data/hist/"
holidays_path <- "holidays.csv"

symbols <- list.files(hist_dir) %>%
  str_remove("\\.csv$")

if (length(symbols) > 0) {
  tradedates <- foreach(
    symbol = symbols,
    .combine = "c"
  ) %dofuture% {
    paste0(hist_dir, symbol, ".csv") %>%
      read_csv(show_col_types = FALSE) %>%
      pull(date)
  } %>%
    unique() %>%
    sort()
  holidays <- seq(first(tradedates), last(tradedates), by = "1 day") %>%
    .[!wday(., week_start = 1) %in% 6:7] %>%
    .[!.%in% tradedates]
} else {
  holidays <- as_date(c())
}

new_holidays <- c(
  mdy("January 1, 2025"),
  seq(mdy("January 28, 2025"), mdy("February 4, 2025"), by = "1 day"),
  mdy("January 26, 2025"),
  mdy("February 8, 2025"),
  seq(mdy("April 4, 2025"), mdy("April 6, 2025"), by = "1 day"),
  seq(mdy("May 1, 2025"), mdy("May 5, 2025"), by = "1 day"),
  mdy("April 27, 2025"),
  seq(mdy("May 31, 2025"), mdy("June 2, 2025"), by = "1 day"),
  seq(mdy("October 1, 2025"), mdy("October 8, 2025"), by = "1 day"),
  mdy("September 28, 2025"),
  mdy("October 11, 2025")
) %>%
  .[!wday(., week_start = 1) %in% 6:7]

holidays <- holidays %>%
  c(new_holidays[!new_holidays %in% .])
write_csv(tibble(date = holidays), holidays_path)

plan(sequential)
