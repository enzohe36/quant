# python -m aktools

rm(list = ls())

library(doFuture)
library(foreach)
library(glue)
library(data.table)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

plan(multisession, workers = availableCores() - 1)

index <- "000905"
model_dir <- "models/"

data_list_path <- paste0(model_dir, "data_list_", index, ".rds")
model_class_path <- paste0(model_dir, "rf_class_", index, ".rds")
model_regr_path <- paste0(model_dir, "rf_regr_", index, ".rds")

data_list <- readRDS(data_list_path)
rf_class <- readRDS(model_class_path)
rf_regr <- readRDS(model_regr_path)

data_list <- foreach(
  data = data_list,
  .combine = "c"
) %dofuture% {
  date <- as_tradedate(now() - hours(16))
  filter(data, date == !!date)
} %>%
  rbindlist()



plan(sequential)
