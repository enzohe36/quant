rm(list = ls())

gc()

library(doFuture)
library(foreach)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"
indexcomp_path <- paste0(data_dir, "indexcomp.csv")
holidays_path <- paste0(data_dir, "holidays.csv")

indexcomp <- read_csv(indexcomp_path)

tradedates <- foreach(
  symbol = pull(indexcomp, symbol),
  .combine = "c"
) %dofuture% {
  paste0(data_dir, symbol, ".csv") %>%
    read_csv(show_col_types = FALSE) %>%
    pull(date) %>%
    list()
} %>%
  unlist() %>%
  unique() %>%
  sort() %>%
  as_date()

holidays_new <- c(
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

holidays <- seq(first(tradedates), last(tradedates), by = "1 day") %>%
  .[!wday(., week_start = 1) %in% 6:7] %>%
  .[!.%in% tradedates] %>%
  c(holidays_new[!holidays_new %in% .])

holidays <- tibble(date = holidays)
write_csv(holidays, holidays_path)

plan(sequential)
