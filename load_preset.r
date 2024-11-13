library(jsonlite)
library(RCurl)
library(tidyverse)
library(TTR)
library(foreach)
library(doParallel)

options(warn = -1)

unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list = ls(name = env), pos = env)
}

normalize <- function(x) {
  return(
    (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  )
}

tnormalize <- function(x, t) {
  df <- foreach(
    i = 0:(t - 1),
    .combine = cbind,
    .packages = c("dplyr")
  ) %dopar% {
    return(lag(x, i))
  }

  return(
    (x - apply(df, 1, min)) / (apply(df, 1, max) - apply(df, 1, min))
  )
}
