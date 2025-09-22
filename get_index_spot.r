rm(list = ls())

gc()

library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

data_dir <- "data/"
index_spot_path <- paste0(data_dir, "index_spot.csv")

index_spot <- get_index_spot()
write_csv(index_spot, index_spot_path)