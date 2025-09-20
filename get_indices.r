rm(list = ls())

gc()

library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

data_dir <- "data/"
indices_path <- paste0(data_dir, "indices.csv")

end_date <- as_tradedate(now() - hours(16))

indices <- get_index_spot() %>%
  select(symbol, name, market) %>%
  mutate(date = end_date, .before = 1)
write_csv(indices, indices_path)
