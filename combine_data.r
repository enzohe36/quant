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
scale_params_path <- paste0(data_dir, "scale_params.csv")
data_train_path <- paste0(data_dir, "data_train.csv")
data_train_tr_path <- paste0(data_dir, "data_train_tr.csv")
example_path <- paste0(data_dir, "example.csv")

analysis_dir <- "analysis/"
logs_dir <- paste0(analysis_dir, "logs/")
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

last_td <- as_date("2026-01-23")
train_start <- last_td %m-% years(10)

set.seed(42)

# MAIN SCRIPT ==================================================================

dir.create(analysis_dir)
dir.create(logs_dir)

quarters_end <- quarter(all_td %m-% months(3), "date_last") %>%
  unique() %>%
  sort()

symbols <- str_remove(list.files(hist_dir), "\\.csv$")
tsprint(str_glue("Found {length(symbols)} stock histories."))

plan(multisession, workers = availableCores() - 1)

data_combined <- foreach(
  symbol = symbols,
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
          ocfps = run_sum(ocfps, 4)
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
        ocf = ocfps * shares,
        across(c(open, high, low, close, avg_price), ~ .x * adjust),
        volume = volume / adjust
      ) %>%
      fill(np, np_deduct, equity, revenue, ocf, .direction = "down") %>%
      filter(date %in% pull(hist, date)) %>%
      mutate(
        avg_cost = calculate_avg_cost(avg_price, to)
      ) %>%
      select(
        symbol, names(hist), avg_price, avg_cost, mc, np, np_deduct, equity,
        revenue, ocf
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

# data_combined <- readRDS(data_combined_path)

plan(multisession, workers = availableCores() - 1)

mkt_mc <- foreach(
  data = data_combined,
  .combine = "c"
) %dofuture% {
  data %>%
    right_join(tibble(date = all_td), by = "date") %>%
    select(date, mc) %>%
    filter(date <= last_td) %>%
    fill(mc, .direction = "down") %>%
    list()
} %>%
  rbindlist() %>%
  group_by(date) %>%
  summarize(mkt_mc = sum(mc, na.rm = TRUE))

data_train <- foreach(
  data = data_combined,
  .combine = "c"
) %dofuture% {
  data <- data %>%
    left_join(mkt_mc, by = "date") %>%
    mutate(
      amplitude = TR(tibble(high, low, close))[, "tr"] / lag(close),
      lr_mc_mkt = log(mc / mkt_mc),
      pe = mc / np,
      pe_deduct = mc / np_deduct,
      pb = mc / equity,
      ps = mc / revenue,
      pc = mc / ocf,
      npm = np / revenue,
      roe = np / equity,
      lr_ap_sma20 = log(avg_price / run_mean(close, 20)),
      lr_ap_sma120 = log(avg_price / run_mean(close, 120)),
      lr_ap_ema12 = log(avg_price / ema_na(close, 12)),
      lr_ap_ema50 = log(avg_price / ema_na(close, 50)),
      lr_ap_ac = log(avg_price / avg_cost),
      lr_ap_kama = log(avg_price / kama),
      lr_osc_sl = log(oscillator / signal_line)
    ) %>%
    select(
      symbol, date, open, close,
      avg_price, avg_cost, kama,
      amplitude, to, oscillator, price_rms, volume_rms,
      lr_mc_mkt, pe, pe_deduct, pb, ps, pc, npm, roe,
      lr_ap_sma20, lr_ap_sma120,
      lr_ap_ema12, lr_ap_ema50,
      lr_ap_ac, lr_ap_kama, lr_osc_sl
    ) %>%
    rename_with(~ paste0("p_", .x), c(avg_price, avg_cost, kama)) %>%
    rename_with(~ paste0("n_", .x), !matches("^(symbol|date|open|close|p_)"))

  data <- data %>%
    filter(date >= train_start) %>%
    na.omit()
  if (nrow(data) == 0) return(NULL) else return(list(data))
} %>%
  rbindlist()

plan(sequential)

scale_params <- data_train %>%
  calculate_scale_params(
    !matches("^(symbol|date|open|close)"), robust = TRUE
  ) %>%
  mutate(across(matches("^p"), ~ p_avg_price))
write_csv(scale_params, scale_params_path)

data_train <- scale_features(data_train, scale_params)
write_csv(data_train, data_train_path)
tsprint(str_glue("nrow(data_train) = {nrow(data_train)}"))

symbols_tr <- sample(unique(data_train$symbol), 100)
data_train_tr <- data_train %>%
  filter(symbol %in% symbols_tr)
write_csv(data_train_tr, data_train_tr_path)
tsprint(str_glue("nrow(data_train_tr) = {nrow(data_train_tr)}"))

example <- data_train[1:10, ]
write_csv(example, example_path)
