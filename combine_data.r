# PRESET =======================================================================

library(xts)
library(DSTrading)
library(patchwork)
library(sn)
library(foreach)
library(doFuture)
library(data.table)
library(tidyverse)

source("scripts/ehlers.r")
source("scripts/features.r")
source("scripts/misc.r")

data_dir <- "data/"
hist_dir <- paste0(data_dir, "hist/")
adjust_dir <- paste0(data_dir, "adjust/")
mc_dir <- paste0(data_dir, "mc/")
val_dir <- paste0(data_dir, "val/")

data_combined_path <- paste0(data_dir, "data_combined.rds")

backtest_dir <- "backtest/"

logs_dir <- paste0(backtest_dir, "logs/")
log_path <- paste0(logs_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

zero_threshold <- 0.05
price_lookback <- 10
min_price_diff <- 0.5
price_lookforward <- 5
min_signal <- 0.35
min_osc_d1 <- -0.01
osc_lookback <- 40
max_osc_diff <- 1.9
price_rms_high <- 1.5
price_rms_low <- -1
min_required_length <- 0

# MAIN SCRIPT ==================================================================

dir.create(backtest_dir)
dir.create(logs_dir)

quarters_start <- unique(quarter(all_td, "date_first"))
quarters_start_td <- as_tradeday(quarters_start)
quarters_end <- quarters_start - 1

plan(multisession, workers = availableCores() - 1)

data_combined <- foreach(
  symbol = str_remove(list.files(hist_dir), "\\.csv$"),
  .combine = "c"
) %dofuture% {
  vars <- c(
    "adjust", "adjust_path", "data", "hist", "hist_path", "mc", "mc_path",
    "my_list", "try_error", "val", "val_path"
  )
  rm(list = vars)

  hist_path <- paste0(hist_dir, symbol, ".csv")
  adjust_path <- paste0(adjust_dir, symbol, ".csv")
  mc_path <- paste0(mc_dir, symbol, ".csv")
  val_path <- paste0(val_dir, symbol, ".csv")

  hist <- read_csv(hist_path, show_col_types = FALSE)

  if (!file.exists(adjust_path)) {
    tsprint(str_glue("{symbol}: Missing adjust file."), log_path)
    return(NULL)
  } else {
    try_error <- try(
      adjust <- read_csv(adjust_path, show_col_types = FALSE) %>%
        mutate(adjust = adjust / last(adjust)),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      tsprint(str_glue("{symbol}: Error reading adjust file."), log_path)
    }
  }

  if (!file.exists(mc_path)) {
    tsprint(str_glue("{symbol}: Missing mc file."), log_path)
    return(NULL)
  } else {
    mc <- read_csv(mc_path, show_col_types = FALSE)
  }

  if (!file.exists(val_path)) {
    tsprint(str_glue("{symbol}: Missing val file."), log_path)
    return(NULL)
  } else {
    try_error <- try(
      val <- read_csv(val_path, show_col_types = FALSE) %>%
        full_join(tibble(date = !!quarters_end), by = "date") %>%
        arrange(date) %>%
        mutate(
          revenue = run_sum(revenue, 4),
          np = run_sum(np, 4),
          np_deduct = run_sum(np_deduct, 4),
          cfps = run_sum(cfps, 4)
        ),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      tsprint(str_glue("{symbol}: Error reading val file."), log_path)
    }
  }

  try_error <- try(
    data <- hist %>%
      full_join(adjust, by = "date") %>%
      full_join(mc, by = "date") %>%
      full_join(val, by = "date") %>%
      arrange(date) %>%
      fill(names(hist), .direction = "down") %>%
      mutate(
        symbol = !!symbol,
        volume = volume * 100,
        to = to / 100,
        avg_price = amount / volume,
        shares = mc * 10^8 / close
      ) %>%
      fill(adjust, shares, .direction = "down") %>%
      mutate(
        mc = close * shares,
        equity = bvps * shares,
        cf = cfps * shares,
        across(c(open, high, low, close, avg_price), ~ .x * adjust),
        volume = volume / adjust
      ) %>%
      fill(np, np_deduct, equity, revenue, cf, .direction = "down") %>%
      filter(date %in% pull(hist, date)) %>%
      mutate(
        avg_cost = calculate_avg_cost(avg_price, to),
        quarter = quarter(date, with_year = TRUE)
      ) %>%
      group_by(quarter) %>%
      mutate(
        across(
          c(mc, to),
          ~ c(rep(NaN, n() - 1), mean(.x)),
          .names = "{col}_quarter"
        )
      ) %>%
      ungroup() %>%
      mutate(
        across(
          c(mc_quarter, to_quarter),
          ~ replace(.x, date < date[date %in% quarters_start_td][2], NA)
        )
      ) %>%
      fill(mc_quarter, to_quarter, .direction = "down") %>%
      select(
        symbol, names(hist), avg_cost, mc, np, np_deduct, equity, revenue, cf,
        quarter, mc_quarter, to_quarter
      ),
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    tsprint(str_glue("{symbol}: Error combining data."), log_path)
    return(NULL)
  }

  try_error <- try(
    data <- generate_features(
      data = data,
      zero_threshold = zero_threshold,
      price_lookback = price_lookback,
      min_price_diff = min_price_diff,
      price_lookforward = price_lookforward,
      min_signal = min_signal,
      min_osc_d1 = min_osc_d1,
      osc_lookback = osc_lookback,
      max_osc_diff = max_osc_diff,
      price_rms_high = price_rms_high,
      price_rms_low = price_rms_low
    ),
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    tsprint(str_glue("{symbol}: Error generating features."), log_path)
    return(NULL)
  }

  my_list <- list()
  my_list[[symbol]] <- data
  return(my_list)
}

plan(sequential)

saveRDS(data_combined, data_combined_path)
tsprint(str_glue("Combined {length(data_combined)} stocks."))
