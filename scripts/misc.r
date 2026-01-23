# PRESET =======================================================================

# library(foreach)
# library(doFuture)
# library(data.table)
# library(tidyverse)

resources_dir <- "resources/"
holidays_path <- paste0(resources_dir, "holidays.txt")

last_td_expr <- expr(as_tradeday(now() - hours(17)))
curr_td_expr <- expr(as_tradeday(now() - hours(9)))

# MISCELLANEOUS ================================================================

holidays <- as_date(readLines(holidays_path))

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

run_sum <- function(x, n) {
  sapply(
    seq_along(x),
    function(i) if (i < n) NA_real_ else sum(x[(i - n + 1):i])
  )
}

run_mean <- function(x, n) run_sum(x, n) / n

run_min <- function(x, n) {
  sapply(
    seq_along(x),
    function(i) if (i < n) NA_real_ else min(x[(i - n + 1):i])
  )
}

run_max <- function(x, n) {
  sapply(
    seq_along(x),
    function(i) if (i < n) NA_real_ else max(x[(i - n + 1):i])
  )
}

cumsum_na <- function(x) {
  first_non_na <- which(!is.na(x))[1]
  result <- x
  if (!is.na(first_non_na)) {
    result[first_non_na:length(x)] <- cumsum(x[first_non_na:length(x)])
  }
  return(result)
}

ema_na <- function(x, n = 10, ...) {
  first_non_na <- which(!is.na(x))[1]
  result <- rep(NA, length(x))
  if (isTRUE(length(x) - first_non_na + 1 >= n)) {
    result[first_non_na:length(x)] <- EMA(x[first_non_na:length(x)], n = n, ...)
  }
  return(result)
}

atr_na <- function(HLC, n = 14, ...) {
  first_non_na <- which(complete.cases(HLC))[1]
  num_rows <- nrow(HLC)
  result <- data.frame(
    tr = rep(NA, num_rows),
    atr = rep(NA, num_rows),
    truehigh = rep(NA, num_rows),
    truelow = rep(NA, num_rows)
  )
  if (isTRUE(num_rows - first_non_na + 1 > n)) {
    result[first_non_na:num_rows, ] <-
      ATR(HLC[first_non_na:num_rows, ], n = n, ...)
  }
  return(result)
}

replace_missing <- function(x, replacement) {
  x[is.infinite(x) | is.na(x)] <- replacement
  return(x)
}

as_tradeday <- function(datetime) {
  date <- as_date(datetime)
  tradeday <- lapply(
    date,
    function(date) {
      seq(date - weeks(3), date, "1 day") %>%
        .[!wday(.) %in% c(1, 7)] %>%
        .[!.%in% holidays] %>%
        last()
    }
  ) %>%
    reduce(c)
  return(tradeday)
}

first_td <- as_date("1990-12-19")
last_td <- eval(last_td_expr)
curr_td <- eval(curr_td_expr)
all_td <- seq(first_td, last_td, "1 day") %>%
  .[!wday(.) %in% c(1, 7)] %>%
  .[!.%in% holidays]

write_title <- function(string = "", total_length = 80) {
  string <- toupper(string)
  if (string == "") {
    output <- paste0("# ", strrep("=", total_length - 2))
  } else {
    n_equals <- total_length - nchar(string) - 3
    output <- paste0("# ", string, " ", strrep("=", max(0, n_equals)))
  }
  cat(output, "\n", sep = "")
}
