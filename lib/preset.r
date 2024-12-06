Sys.setenv(TZ = "Asia/Shanghai")
Sys.setlocale(locale = "Chinese")

options(warn = -1)

library(jsonlite)
library(RCurl)
library(tidyverse)
library(TTR)
library(foreach)
library(doParallel)
library(glue)
library(signal)
library(gplots)
