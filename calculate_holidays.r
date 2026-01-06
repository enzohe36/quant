# PRESET =======================================================================

library(foreach)
library(doFuture)
library(data.table)
library(tidyverse)

source("scripts/misc.r")

data_dir <- "data/"
hist_dir <- paste0(data_dir, "hist/")

resources_dir <- "resources/"
holidays_path <- paste0(resources_dir, "holidays.txt")

# MAIN SCRIPT ==================================================================

dir.create(resources_dir)

new_holidays <- c(
  seq(mdy("January 1, 2026"), mdy("January 3, 2026"), "1 day"),
  seq(mdy("February 15, 2026"), mdy("February 23, 2026"), "1 day"),
  seq(mdy("April 4, 2026"), mdy("April 6, 2026"), "1 day"),
  seq(mdy("May 1, 2026"), mdy("May 5, 2026"), "1 day"),
  seq(mdy("June 19, 2026"), mdy("June 21, 2026"), "1 day"),
  seq(mdy("September 25, 2026"), mdy("September 27, 2026"), "1 day"),
  seq(mdy("October 1, 2026"), mdy("October 7, 2026"), "1 day")
) %>%
  .[!wday(.) %in% c(1, 7)]

holidays <- as_date(c())

if (dir.exists(hist_dir)) {
  files <- list.files(hist_dir)

  if (length(files) > 0) {
    plan(multisession, workers = availableCores() - 1)

    tradedays <- foreach(
      file = files,
      .combine = "c"
    ) %dofuture% {
      pull(read_csv(paste0(hist_dir, file), show_col_types = FALSE), date)
    } %>%
      unique()

    plan(sequential)

    holidays <- seq(min(tradedays), max(tradedays), by = "1 day") %>%
      .[!wday(.) %in% c(1, 7)] %>%
      .[!.%in% tradedays]
  }
}

holidays <- sort(unique(c(holidays, new_holidays)))
writeLines(as.character(holidays), holidays_path)
tsprint(str_glue("Updated holidays from {min(holidays)} to {max(holidays)}."))
