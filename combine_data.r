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
# train_path <- paste0(data_dir, "train_buy.csv")
# test_path <- paste0(data_dir, "test_buy.csv")

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

last_td <- eval(last_td_expr)
train_start <- last_td %m-% years(5)
test_start <- last_td %m-% years(1)

pred_length <- 20
freq_buy <- 0.025
freq_sell <- 0.025

set.seed(42)

# MAIN SCRIPT ==================================================================

dir.create(analysis_dir)
dir.create(logs_dir)

quarters_start <- unique(quarter(all_td, "date_first"))
quarters_start_td <- as_tradeday(quarters_start)
quarters_end <- quarters_start - 1

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
          ~ c(rep(NA_real_, n() - 1), mean(.x)),
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
        symbol, names(hist), avg_price, avg_cost, mc, np, np_deduct, equity,
        revenue, cf, quarter, mc_quarter, to_quarter
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

data_combined_filtered <- foreach(
  data = data_combined,
  .combine = "c"
) %dofuture% {
  data <- data %>%
    mutate(
      reward_buy = lead(run_max(avg_price, pred_length), pred_length) /
        avg_price - 1,
      reward_sell = avg_price /
        lead(run_min(avg_price, pred_length), pred_length) - 1,
      mc_sma60 = run_mean(mc, 60),
      to_sma60 = run_mean(to, 60),
      close_geq_kama = as.numeric(close >= kama),
      close_geq_avg_cost = as.numeric(close >= avg_cost),
      low_geq_kama = as.numeric(low >= kama),
      low_geq_avg_cost = as.numeric(low >= avg_cost),
      high_geq_kama = as.numeric(high >= kama),
      high_geq_avg_cost = as.numeric(high >= avg_cost),
      kama = kama / atr,
      avg_cost = avg_cost / atr
    ) %>%
    select(
      symbol, date, reward_buy, reward_sell,
      mc_sma60, to_sma60,
      close_geq_kama, close_geq_avg_cost,
      low_geq_kama, low_geq_avg_cost,
      high_geq_kama, high_geq_avg_cost,
      kama, avg_cost,
      oscillator, price_rms, volume_rms
    )

  data <- data %>%
    filter(date >= train_start) %>%
    na.omit()
  if (nrow(data) == 0) return(NULL) else return(list(data))
} %>%
  rbindlist()

plan(sequential)

cutoff_buy <- quantile(data_combined_filtered$reward_buy, 1 - freq_buy)
cutoff_sell <- quantile(data_combined_filtered$reward_sell, 1 - freq_sell)
data_combined_filtered <- data_combined_filtered %>%
  mutate(
    reward_buy = reward_buy - cutoff_buy,
    reward_sell = reward_sell - cutoff_sell
  )
write_csv(
  tibble(
    reward_type = c("buy", "sell"),
    cutoff = c(cutoff_buy, cutoff_sell)
  ),
  "data/reward_cutoff.csv"
)

symbols_tr <- sample(unique(data_combined_filtered$symbol), 100)

train_buy <- data_combined_filtered %>%
  mutate(
    reward = reward_buy,
    across(contains("reward_"), ~ NULL)
  ) %>%
  filter(date < test_start)
write_csv(train_buy, "data/train_buy.csv")
tsprint(str_glue("train_buy: {nrow(train_buy)} stocks."))

train_buy_tr <- train_buy %>%
  filter(symbol %in% symbols_tr)
write_csv(train_buy_tr, "data/train_buy_tr.csv")
tsprint(str_glue("train_buy_tr: {nrow(train_buy_tr)} stocks."))

test_buy <- data_combined_filtered %>%
  mutate(
    reward = reward_buy,
    across(contains("reward_"), ~ NULL)
  ) %>%
  filter(date >= test_start)
write_csv(test_buy, "data/test_buy.csv")
tsprint(str_glue("test_buy: {nrow(test_buy)} stocks."))

test_buy_tr <- test_buy %>%
  filter(symbol %in% symbols_tr)
write_csv(test_buy_tr, "data/test_buy_tr.csv")
tsprint(str_glue("test_buy_tr: {nrow(test_buy_tr)} stocks."))

train_sell <- data_combined_filtered %>%
  mutate(
    reward = reward_sell,
    across(contains("reward_"), ~ NULL)
  ) %>%
  filter(date < test_start)
write_csv(train_sell, "data/train_sell.csv")
tsprint(str_glue("train_sell: {nrow(train_sell)} stocks."))

train_sell_tr <- train_sell %>%
  filter(symbol %in% symbols_tr)
write_csv(train_sell_tr, "data/train_sell_tr.csv")
tsprint(str_glue("train_sell_tr: {nrow(train_sell_tr)} stocks."))

test_sell <- data_combined_filtered %>%
  mutate(
    reward = reward_sell,
    across(contains("reward_"), ~ NULL)
  ) %>%
  filter(date >= test_start)
write_csv(test_sell, "data/test_sell.csv")
tsprint(str_glue("test_sell: {nrow(test_sell)} stocks."))

test_sell_tr <- test_sell %>%
  filter(symbol %in% symbols_tr)
write_csv(test_sell_tr, "data/test_sell_tr.csv")
tsprint(str_glue("test_sell_tr: {nrow(test_sell_tr)} stocks."))

example <- train_buy[1:10, ]
write_csv(example, "data/example.csv")
