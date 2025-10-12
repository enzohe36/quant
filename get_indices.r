# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools

rm(list = ls())

gc()

library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

################################################################################

data_dir <- "data/"
indices_path <- paste0(data_dir, "indices.csv")

indices <- get_index_spot() %>%
  select(symbol, name, market)
write_csv(indices, indices_path)
