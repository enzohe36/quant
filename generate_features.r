# PRESET =======================================================================

library(foreach)
library(doFuture)
library(xts)
library(DSTrading)
library(patchwork)
library(tidyverse)

source("scripts/misc.r")
source("scripts/ehlers.r")

data_dir <- "data/"
data_combined_path <- paste0(data_dir, "data_combined.rds")
spot_combined_path <- paste0(data_dir, "spot_combined.csv")

backtest_dir <- "backtest/"

logs_dir <- "logs/"
log_path <- paste0(logs_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

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

dir.create(backtest_dir)
dir.create(logs_dir)

data_combined <- readRDS(data_combined_path)
spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)

plan(multisession, workers = availableCores() - 1)

data_combined <- foreach(
  data = data_combined,
  .combine = "c"
) %dofuture% {
  vars <- c("my_list", "symbol")
  rm(list = vars)

  symbol <- first(data$symbol)
  data <- generate_features(
    data = data,
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
  my_list <- list()
  my_list[[symbol]] <- data
  return(my_list)
}

plan(sequential)

symbols <- lapply(
  data_combined,
  function(df) {
    if (
      filter(
        df,
        date == !!end_date &
          mc >= 10^10 &
          pe > 0 &
          runMax(price_rms, 20) >= 2
      ) %>%
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
  # print(plot)
  ggsave(image_path, plot)
}
