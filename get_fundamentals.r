# pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools

rm(list = ls())

gc()

library(doFuture)
library(foreach)
library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(microbenchmark)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"
indexcomp_path <- paste0(data_dir, "indexcomp.csv")
fundamentals_path <- paste0(data_dir, "fundamentals.csv")

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

quarters <- quarter(tradedates, type = "date_last") %>% unique()

max_try <- 10
xs <- c("bs", "is", "cfs", "div")

fundamentals <- foreach(
  task = expand.grid(quarter = quarters, x = xs) %>% split(seq_len(nrow(.))),
  .options.future = list(globals = structure(TRUE, add = paste0("get_", xs))),
  .combine = "comb",
  .multicombine = TRUE,
  .init = list(list(), list(), list())
) %dofuture% {
  var <- c("var", "quarter", "x", "t", "try_error", "data", "status")
  rm(list = var)

  quarter <- pull(task, quarter)
  x <- pull(task, x)

  for (i in seq_len(max_try)) {
    t <- microbenchmark(
      try_error <- try(data <- get(paste0("get_", x))(quarter), silent = TRUE),
      times = 1
    ) %>%
      pull(time)
    if (!inherits(try_error, "try-error")) {
      status <- TRUE
      break
    } else if (t < 10^9 | i == max_try) {
      data <- NULL
      status <- FALSE
      print(paste0(quarter, ": failed to get ", x))
      break
    }
  }

  return(list(x = x, data = data, status = status))
}

fundamentals <- split(fundamentals[["data"]], unlist(fundamentals[["x"]])) %>%
  lapply(rbindlist) %>%
  reduce(full_join, by = c("symbol", "date")) %>%
  filter(symbol %in% pull(indexcomp, symbol)) %>%
  distinct() %>%
  arrange(symbol, quarter)

# fundamentals <- rows_patch(
#   fundamentals, read_csv(fundamentals_path), by = c("symbol", "quarter")
# )

write_csv(fundamentals, fundamentals_path)

plan(sequential)


for (yr in 1991:2024) {
  try_error <- glue("{yr}-03-31") %>%
    as_date() %>%
    get_nsh() %>%
    head() %>%
    try()
  print(try_error)
}





bs <- get_bs(as_date("2024-03-30"))
is <- get_is(as_date("2024-03-30"))
cfs <- get_cfs(as_date("2024-03-30"))
div <- get_div(as_date("2024-03-30"))
data <- list(bs, is, cfs, div) %>%
  reduce(full_join, by = c("symbol", "quarter")) %>%
  filter(symbol == "000001")

val <- get_val("000001")
hist <- get_hist(
  "000001", "daily", as_date("2024-12-31"), as_date("2024-12-31"), NULL
)

close <- hist %>% pull(close)
tso <- filter(val, date == !!date) %>% pull (tso)
data %>% mutate(equity = assets - liabilities) %>%
  mutate(across(where(is.numeric), ~ close / (. / tso)))
filter(val, date == !!date)
