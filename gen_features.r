# =============================== PRESET ==================================

source_scripts(
  scripts = c("misc", "ehlers"),
  packages = c("tidyverse")
)

backtest_dir <- "backtest/"
data_combined_path <- paste0(backtest_dir, "data_combined.rds")
data_bt_path <- paste0(backtest_dir, "data_bt.rds")

end_date <- as_tradeday(now() - hours(16))
start_date <- end_date %m-% years(5)

zero_threshold <- 0.05
price_lookback <- 10
min_price_diff <- 0.5
price_lookforward <- 5
min_signal <- 0.35
min_osc_d1 <- -0.01
osc_lookback <- 40
max_osc_diff <- 1.9

# ============================= MAIN SCRIPT ===============================

data_combined <- if (exists("data_bt")) data_bt else readRDS(data_combined_path)

ts1 <- Sys.time()
data_bt <- generate_features(
  data_combined = data_combined,
  start_date = start_date,
  end_date = end_date,
  zero_threshold = zero_threshold,
  price_lookback = price_lookback,
  min_price_diff = min_price_diff,
  price_lookforward = price_lookforward,
  min_signal = min_signal,
  min_osc_d1 = min_osc_d1,
  osc_lookback = osc_lookback,
  max_osc_diff = max_osc_diff
)
ts2 <- Sys.time()
print(ts2 - ts1)

# saveRDS(data_bt, data_bt_path)

# Test

symbols <- sample(names(data_bt), 5)
start_date <- end_date %m-% years(1)
spot <- read_csv("data/spot_combined.csv", show_col_types = FALSE)

for (symbol in symbols) {
  data <- data_bt[[symbol]] %>%
    filter(date >= start_date & date <= end_date)
  plot <- plot_supersmoother_indicator(data, data, spot)
  print(plot)
}
