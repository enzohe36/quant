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

plot_dual_y <- function(x, y1, y2,
                        xlab = "x",
                        y1lab = "y1",
                        y2lab = "y2",
                        title = "",
                        y1_color = "blue",
                        y2_color = "red") {
  # Calculate ranges for both series
  min_y1 <- min(y1, na.rm = TRUE)
  max_y1 <- max(y1, na.rm = TRUE)
  range_y1 <- max_y1 - min_y1

  min_y2 <- min(y2, na.rm = TRUE)
  max_y2 <- max(y2, na.rm = TRUE)
  range_y2 <- max_y2 - min_y2

  # Scale y2 to match y1's range
  df <- data.frame(
    x = x,
    y1 = y1,
    y2 = y2,
    y2_scaled = (y2 - min_y2) / range_y2 * range_y1 + min_y1
  )

  p <- ggplot(df, aes(x = x)) +
    geom_line(aes(y = y1, color = "y1"), linewidth = 0.5) +
    geom_line(aes(y = y2_scaled, color = "y2"), linewidth = 0.5) +
    scale_y_continuous(
      name = y1lab,
      sec.axis = sec_axis(~ (. - min_y1) / range_y1 * range_y2 + min_y2,
                         name = y2lab)
    ) +
    scale_color_manual(
      values = c("y1" = y1_color, "y2" = y2_color),
      labels = c(y1lab, y2lab)
    ) +
    labs(title = title, x = xlab, color = "") +
    theme_minimal() +
    theme(
      axis.title.y.left = element_text(color = y1_color),
      axis.text.y.left = element_text(color = y1_color),
      axis.title.y.right = element_text(color = y2_color),
      axis.text.y.right = element_text(color = y2_color)
    )

  print(p)
  invisible(p)
}

# Data Utilities ---------------------------------------------------------------

# Trailing sum over `period` rows. Returns NA for incomplete windows.
# Partial sums (some NAs in window) are computed — leading NAs get filtered
# downstream by trim_leading_na.
run_sum <- function(x, period) {
  n <- length(x)
  if (n < period) return(rep(NA_real_, n))
  mat <- embed(x, period)
  valid <- rowSums(!is.na(mat))
  sums <- ifelse(valid > 0, rowSums(mat, na.rm = TRUE), NA_real_)
  c(rep(NA_real_, period - 1), sums)
}

# Cross-sectional percentile rank. Output [-1, +1].
# Designed for data.table by-group usage:
#   feats[, (cols) := lapply(.SD, cross_pctrank), by = date, .SDcols = cols]
cross_pctrank <- function(x) {
  out <- rep(NA_real_, length(x))
  valid <- !is.na(x)
  if (sum(valid) <= 1) return(out)
  out[valid] <- 2 * rank(x[valid], ties.method = "average") / sum(valid) - 1
  out
}
