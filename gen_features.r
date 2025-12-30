# PRESET =======================================================================

source_scripts(
  scripts = c("misc", "ehlers"),
  packages = c("tidyverse")
)

data_dir <- "data/"
spot_combined_path <- paste0(data_dir, "spot_combined.csv")

backtest_dir <- "backtest/"
data_combined_path <- paste0(backtest_dir, "data_combined.rds")

end_date <- eval(last_td_expr)
start_date <- end_date %m-% years(1)

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

# MAIN SCRIPT ==================================================================

spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)
data_combined <- readRDS(data_combined_path)

data_combined <- gen_features(
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
  max_osc_diff = max_osc_diff,
  price_rms_high = price_rms_high,
  price_rms_low = price_rms_low
)

symbols <- sapply(
  data_combined,
  function(df) {
    if (
      df %>%
        filter(date == !!end_date & mc >= 10^10 & pe > 0 & oscillator <= 0) %>%
        pull(buy) %>%
        isTRUE()
    ) {
      unique(df$symbol)
    }
  }
) %>%
  unlist() %>%
  unname()

for (symbol in symbols) {
  image_path <- paste0(backtest_dir, symbol, ".png")
  data <- data_combined[[symbol]] %>%
    filter(date >= start_date & date <= end_date)
  spot <- filter(spot_combined, symbol == !!symbol)
  plot <- plot_indicators(data, spot)
  ggsave(image_path, plot)
}
