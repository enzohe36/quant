library(jsonlite)
library(RCurl)
library(tidyverse)
library(TTR)
library(foreach)
library(doParallel)

options(warn = -1)

# https://stackoverflow.com/a/25110203
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list = ls(name = env), pos = env)
}

normalize <- function(x) {
  return(
    (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  )
}

normalize0 <- function(x) {
  return(
    (0 - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  )
}

tnormalize <- function(x, t) {
  df <- data.frame(matrix(nrow = length(x), ncol = 0))
  for (i in 1:t) {
    df[, i] <- lag(x, i - 1)
  }

  return(
    (x - apply(df, 1, min)) / (apply(df, 1, max) - apply(df, 1, min))
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
  out <- cbind(adx, adxr)
  colnames(out) <- c("adx", "adxr")
  return(out)
}

pct_change <- function(x1, x2) {
  return((x2 - x1) / x1)
}

# https://stackoverflow.com/a/19801108
multiout <- function(x, ...) {
  lapply(
    seq_along(x),
    function(i) c(x[[i]], lapply(list(...), function(y) y[[i]]))
  )
}
