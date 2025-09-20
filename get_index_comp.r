rm(list = ls())

gc()

library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

data_dir <- "data/"
hist_dir <- paste0(data_dir, "hist/")
index_comp_path <- paste0(data_dir, "index_comp.csv")

end_date <- as_tradedate(now() - hours(16))

symbols <- str_remove(list.files(hist_dir), "\\.csv$") %>%
  tibble() %>%
  rename(symbol = 1)

index_comp <- list(
  get_index_comp("000300"),
  get_index_comp("000905"),
  get_index_comp("000852"),
  get_index_comp("932000")
) %>%
  rbindlist() %>%
  right_join(symbols, by = "symbol") %>%
  left_join(
    select(get_index_comp("000985"), symbol, weight),
    by = "symbol",
    suffix = c("", "_allshare")
  ) %>%
  arrange(symbol) %>%
  mutate(date = end_date, .before = 1)
write_csv(index_comp, index_comp_path)
