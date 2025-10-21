# library(foreach)
# library(doFuture)
# library(tidyverse)

holidays <- read_csv("data/holidays.csv", show_col_types = FALSE)$date

# ============================================================================

# .combine = "multiout", .multicombine = TRUE, .init = list(list(), list(), ...)
# https://stackoverflow.com/a/19801108
multiout <- function(lst1, ...) {
  lapply(
    seq_along(lst1),
    function(i) c(lst1[[i]], lapply(list(...), function(lst2) lst2[[i]]))
  )
}

tsprint <- function(v, ...) {
  write_args <- list(...)
  ts <- function(v) paste0("[", format(now(), "%H:%M:%S"), "] ", v)
  if (length(write_args) == 0) {
    writeLines(ts(v))
  } else {
    if (is.null(write_args$append)) write_args$append <- TRUE
    out <- sapply(ts(v), function(x) do.call("write", c(x, write_args)))
  }
}

# Redefines TTR::runSum
runSum <- function(x, n) {
  sapply(seq_along(x), function(i) {
    if (i < n) {
      return(NA_real_)   # not enough values before current
    }
    window <- x[(i - n + 1):i]
    return(sum(window))
  })
}

replace_missing <- function(x, replacement) {
  x[is.infinite(x) | is.na(x)] <- replacement
  return(x)
}

get_holidays <- function(hist_dir) {
  files <- list.files(hist_dir)

  if (length(files) == 0) {
    holidays <- as_date(c())
  } else {
    plan(multisession, workers = availableCores() - 1)

    tradedates <- foreach(
      file = files,
      .combine = "c"
    ) %dofuture% {
      read_csv(paste0(hist_dir, file), show_col_types = FALSE) %>%
        pull(date)
    } %>%
      unique()

    plan(sequential)

    holidays <- seq(min(tradedates), max(tradedates), by = "1 day") %>%
      .[!wday(., week_start = 1) %in% 6:7] %>%
      .[!.%in% tradedates]
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

  return(unique(c(holidays, new_holidays)))
}

as_tradedate <- function(datetime) {
  date <- as_date(datetime)
  tradedate <- lapply(
    date,
    function(date) {
      seq(date - weeks(3), date, "1 day") %>%
        .[!wday(., week_start = 1) %in% 6:7] %>%
        .[!.%in% holidays] %>%
        last()
    }
  ) %>%
    reduce(c)
  return(tradedate)
}
