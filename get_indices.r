# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools
# conda activate myenv; Rscript get_data.r; Rscript get_data.r

rm(list = ls())

gc()

library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

end_date <- as_tradedate(now() - hours(16))

indices <- get_indexspot() %>%
  select(market, symbol, name) %>%
  arrange(symbol) %>%
  mutate(date = end_date, .before = 1)
write_csv(indices, "indices.csv")
