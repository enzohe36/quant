rm(list = ls())

gc()

library(foreach)
library(doFuture)
library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(TTR)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

################################################################################

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"
hist_dir <- paste0(data_dir, "hist/")
adjust_dir <- paste0(data_dir, "adjust/")
mc_dir <- paste0(data_dir, "mc/")
val_dir <- paste0(data_dir, "val/")

model_dir <- "models/"
data_combined_path <- paste0(model_dir, "data_combined.rds")

log_dir <- "logs/"
log_path <- paste0(log_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

dir.create(model_dir)
dir.create(log_dir)

end_date <- as_tradedate(now() - hours(16))
start_date <- end_date %m-% years(20)
quarters <- seq(
  as_date("1990-01-01"),
  end_date %m-% months(3),
  by = "1 day"
) %>%
  quarter("date_last") %>%
  unique()

symbols <- list.files(hist_dir) %>%
  str_remove("\\.csv$")

data_combined <- foreach (
  symbol = symbols,
  .combine = "c"
) %dofuture% {
  vars <- c(
    "hist_path", "adjust_path", "mc_path", "val_path",
    "hist", "adjust", "mc", "val", "data", "try_error"
  )
  rm(list = vars)

  hist_path <- paste0(hist_dir, symbol, ".csv")
  adjust_path <- paste0(adjust_dir, symbol, ".csv")
  mc_path <- paste0(mc_dir, symbol, ".csv")
  val_path <- paste0(val_dir, symbol, ".csv")

  hist <- read_csv(hist_path, show_col_types = FALSE)

  if (!file.exists(adjust_path)) {
    glue("{symbol} Missing adjust file.") %>%
      tslog(log_path)
    return(NULL)
  } else {
    adjust <- read_csv(adjust_path, show_col_types = FALSE) %>%
      mutate(adjust = adjust / last(adjust))
  }

  if (!file.exists(mc_path)) {
    glue("{symbol} Missing mc file.") %>%
      tslog(log_path)
    return(NULL)
  } else {
    mc <- read_csv(mc_path, show_col_types = FALSE)
  }

  try_error <- try(
    if (!file.exists(val_path)) {
      glue("{symbol} Missing val file.") %>%
        tslog(log_path)
      return(NULL)
    } else {
      val <- read_csv(val_path, show_col_types = FALSE) %>%
        full_join(tibble(date = !!quarters), by = "date") %>%
        arrange(date) %>%
        mutate(
          revenue = runSum(revenue, 4),
          np = runSum(np, 4),
          np_deduct = runSum(np_deduct, 4),
          cfps = runSum(cfps, 4),
          quarter = date
        )
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{symbol} Error reading val file.") %>%
      tslog(log_path)
    return(NULL)
  }

  try_error <- try(
    data <- hist %>%
      full_join(adjust, by = "date") %>%
      full_join(mc, by = "date") %>%
      full_join(val, by = "date") %>%
      arrange(date) %>%
      fill(names(hist), .direction = "down") %>%
      mutate(shares = mc * 10^8 / close) %>%
      fill(adjust, shares, quarter, .direction = "down") %>%
      mutate(
        mc = close * shares,
        equity = bvps * shares,
        cf = cfps * shares,
        across(c(open, high, low, close), ~ .x * adjust),
        volume = volume / adjust
      ) %>%
      group_by(quarter) %>%
      fill(np, np_deduct, equity, revenue, cf, .direction = "down") %>%
      ungroup() %>%
      mutate(
        pe = mc / np,
        pe_deduct = mc / np_deduct,
        pb = mc / equity,
        ps = mc / revenue,
        pcf = mc / cf,
        roe = np / equity,
        npm = np / revenue,
        symbol = !!symbol,
        across(c(amount, mc), ~ .x / 10^8)
      ) %>%
      select(symbol, names(hist), mc, pe, pe_deduct, pb, ps, pcf, roe, npm) %>%
      filter(date %in% pull(hist, date) & date >= !!start_date),
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    glue("{symbol} Error calculating val.") %>%
      tslog(log_path)
    return(NULL)
  }

  my_list <- list()
  my_list[[symbol]] <- data
  return(my_list)
}

saveRDS(data_combined, data_combined_path)

plan(sequential)
