library(jsonlite)
library(RCurl)
library(tidyverse)
library(TTR)
library(foreach)
library(doParallel)
library(glue)

Sys.setenv(TZ = "Asia/Shanghai")
Sys.setlocale(locale = "Chinese")

options(warn = -1)

# https://stackoverflow.com/a/25110203
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list = ls(name = env), pos = env)
}

normalize <- function(v) {
  return(
    (v - min(v, na.rm = TRUE)) / (max(v, na.rm = TRUE) - min(v, na.rm = TRUE))
  )
}

normalize0 <- function(v) {
  return(
    (0 - min(v, na.rm = TRUE)) / (max(v, na.rm = TRUE) - min(v, na.rm = TRUE))
  )
}

tnormalize <- function(v, t) {
  return(
    (v - runMin(v, t)) / (runMax(v, t) - runMin(v, t))
  )
}

# http://www.cftsc.com/qushizhibiao/610.html
adx_alt <- function(hlc, n = 14, m = 6) {
  h <- hlc[, 1]
  l <- hlc[, 2]
  c <- hlc[, 3]
  tr <- runSum(
    apply(cbind(h - l, abs(h - lag(c, 1)), abs(l - lag(c, 1))), 1, max), n
  )
  dh <- h - lag(h, 1)
  dl <- lag(l, 1) - l
  dmp <- runSum(ifelse(dh > 0 & dh > dl, dh, 0), n)
  dmn <- runSum(ifelse(dl > 0 & dl > dh, dl, 0), n)
  dip <- dmp / tr
  din <- dmn / tr
  adx <- SMA(abs(dip - din) / (dip + din), m)
  adxr <- (adx + lag(adx, m)) / 2
  df <- data.frame(adx, adxr)
  colnames(df) <- c("adx", "adxr")

  return(df)
}

ror <- function(v1, v2) {
  return((v2 - v1) / abs(v1))
}

# https://stackoverflow.com/a/19801108
multiout <- function(lst1, ...) {
  lapply(
    seq_along(lst1),
    function(i) c(lst1[[i]], lapply(list(...), function(lst2) lst2[[i]]))
  )
}

tsprint <- function(v) {
  v <- paste0("[", format(now(), "%H:%M:%S"), "] ", v)
  writeLines(v)
}