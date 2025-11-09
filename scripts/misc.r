# =============================== PRESET ==================================

# library(foreach)
# library(doFuture)
# library(tidyverse)

data_dir <- "data/"
holidays_path <- paste0(data_dir, "holidays.txt")

# ============================ MISCELLANEOUS ==============================

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

writeTitle <- function(str = "", level = 0, length = 75) {
  str <- toupper(str)
  pad_length <- floor((length - str_length(str) - 2) / 2)
  left <- c("#", paste0(rep("=", pad_length - 2), collapse = ""), str) %>%
    .[str_length(.) > 0] %>%
    paste0(collapse = " ") %>%
    str_sub(1, str_length(.) - str_length(str))
  right <- c(str, paste0(rep("=", length), collapse = "")) %>%
    .[str_length(.) > 0] %>%
    paste0(collapse = " ")
  paste0(left, right) %>%
    str_sub(1, length) %>%
    writeLines()
}
