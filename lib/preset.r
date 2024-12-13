Sys.setenv(TZ = "Asia/Shanghai")
Sys.setlocale(locale = "Chinese")

library(jsonlite)
library(RCurl)
library(foreach)
library(doParallel)
library(glue)
library(gplots)
library(TTR)
library(signal)
library(data.table)
library(dtplyr)
library(tidyverse)

options(warn = -1)
