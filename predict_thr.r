# python -m aktools

rm(list = ls())

gc()

library(doFuture)
library(foreach)
library(RCurl)
library(jsonlite)
library(glue)
library(data.table)
library(ranger)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

data_dir <- "data/"
model_dir <- "models/"

data_list_path <- paste0(data_dir, "data_list.rds")
train_path <- paste0(model_dir, "train.rds")
model_path <- paste0(model_dir, "rf_14.rds")

coi <- "a"

data_list <- readRDS(data_list_path)
train <- readRDS(train_path)
rf <- readRDS(model_path)

date <- as_tradedate(now() - hours(16))
date_train <- max(train$date)
class <- c("a", "b", "c", "d", "e")

plan(multisession, workers = availableCores() - 1)

new <- foreach(
  data = data_list,
  .combine = "append"
) %dofuture% {
  list(filter(data, date > !!date_train))
} %>%
  rbindlist()

plan(sequential)

y <- cbind(new, predict(rf, new)[["predictions"]]) %>%
  mutate(
    target = apply(
      select(., !!class), 1, function(v) !!class %>% .[match(max(v), v)]
    ),
    across(c(!!class), ~ NULL)
  ) %>%
  rbind(na.omit(rbindlist(data_list))) %>%
  split(.$date) %>%
  sapply(function(df) nrow(filter(df, target == coi)) / nrow(df)) %>%
  .[order(as_date(names(.)))]

ind <- get_index("000985", "sh", date %m-% months(3), date)
ind_roc <- get_roc(lead(ind$open), lead(ind$close, 10))
plot(
  ind$date, ind_roc,
  type = "l",
  xlab = "Date",
  ylab = "Index 10-day return"
)
lines(as_date(names(y)), normalize(y, ind_roc), col = "blue")
grid(nx = NULL, ny = NULL, lty = 1, col = "gray")
