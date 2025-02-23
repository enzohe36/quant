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

predict_thr <- function(
  coi,
  data_comb = parent.frame()$data_comb,
  date_train = parent.frame()$date_train,
  rf = parent.frame()$rf,
  actual = FALSE
) {
  if (actual) {
    data_comb <- na.omit(data_comb)
  } else {
    class <- c("a", "b", "c", "d", "e")
    old <- filter(data_comb, date <= date_train) %>% na.omit()
    new <- filter(data_comb, date > date_train)
    pred <- predict(rf, new)[["predictions"]]
    data_comb <- new %>%
      mutate(
        target = apply(!!pred, 1, function(v) !!class %>% .[match(max(v), v)])
      ) %>%
      rbind(old) %>%
      arrange(date, symbol)
  }
  thr <- split(data_comb, data_comb$date) %>%
    sapply(function(df) nrow(filter(df, target == coi)) / nrow(df))
  return(thr)
}

data_comb <- rbindlist(readRDS(data_list_path))
date_train <- max(readRDS(train_path)$date)
rf <- readRDS(model_path)
thr_a <- predict_thr("a")
thr_a_act <- predict_thr("a", actual = TRUE)
thr_e <- predict_thr("e")
thr_e_act <- predict_thr("e", actual = TRUE)

date <- as_tradedate(now() - hours(16))
ind <- get_index("000985", "sh", date %m-% months(3), date)
ind_roc <- get_roc(lead(ind$open), lead(ind$close, 10))

plot(
  ind$date, ind_roc,
  type = "l",
  main = model_path,
  xlab = "Date",
  ylab = glue("Index 10-day return"),
  panel.first = grid(nx = NULL, ny = NULL, lty = 1, col = "lightgray")
)
lines(
  as_date(names(thr_a)),
  normalize(thr_a, ind_roc),
  col = "red",
  lty = 2
)
lines(
  as_date(names(thr_a_act)),
  normalize(thr_a_act, ind_roc),
  col = "red",
)
lines(
  as_date(names(thr_e)),
  normalize(1 - thr_e, ind_roc),
  col = "blue",
  lty = 2
)
lines(
  as_date(names(thr_e_act)),
  normalize(1 - thr_e_act, ind_roc),
  col = "blue",
)
