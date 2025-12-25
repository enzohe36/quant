# PRESET =======================================================================

# library(foreach)
# library(doFuture)
# library(tidyverse)

resource_dir <- "resources/"
holidays_path <- paste0(resource_dir, "holidays.txt")

last_td_expr <- expr(as_tradeday(now() - hours(17)))
curr_td_expr <- expr(as_tradeday(now() - hours(9)))

# MISCELLANEOUS ================================================================

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

as_tradeday <- function(datetime) {
  date <- as_date(datetime)
  tradeday <- lapply(
    date,
    function(date) {
      seq(date - weeks(3), date, "1 day") %>%
        .[!wday(., week_start = 1) %in% 6:7] %>%
        .[!.%in% as_date(readLines(holidays_path))] %>%
        last()
    }
  ) %>%
    reduce(c)
  return(tradeday)
}

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
