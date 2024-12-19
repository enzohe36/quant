Sys.setenv(TZ = "Asia/Shanghai")
Sys.setlocale(locale = "Chinese")

options(warn = -1)

library(jsonlite)
library(RCurl)
library(foreach)
library(doParallel)
library(glue)
library(TTR)
library(signal)
library(data.table)
library(tidyverse)
