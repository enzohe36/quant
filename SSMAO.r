# SuperSmoother MA Oscillator - R Implementation
# https://www.tradingview.com/scripts/ehlers/
# https://www.tradingview.com/script/e8DZtqQL/
# https://www.tradingview.com/script/559mGm7c/

# Required libraries
library(xts)
library(DSTrading)
library(tidyverse)
library(patchwork)

# ============================================================================
# Helper Functions
# ============================================================================

# SuperSmoother Function
supersmoother <- function(src, length) {
  a1 <- exp(-sqrt(2) * pi / length)
  b1 <- 2.0 * a1 * cos(sqrt(2) * pi / length)
  c2 <- b1
  c3 <- -a1 * a1
  c1 <- 1 - c2 - c3

  src_avg <- (src + c(src[1], src[-length(src)])) / 2

  ss <- stats::filter(
    x = c1 * src_avg,
    filter = c(c2, c3),
    method = "recursive",
    init = rep(0, 2)
  )

  return(as.numeric(ss))
}

# Exponential Moving Average
EMA <- function(x, n = 10) {
  library(zoo)

  alpha <- 2 / (n + 1)

  na_in_window <- rollapply(
    zoo(is.na(x)),
    width = n,
    FUN = any,
    align = "right",
    partial = TRUE,
    fill = FALSE
  )

  ema <- Reduce(
    f = function(prev, curr) {
      if (is.na(curr) || is.na(prev)) NA_real_
      else alpha * curr + (1 - alpha) * prev
    },
    x = x,
    accumulate = TRUE
  )

  ema[as.logical(na_in_window)] <- NA_real_

  return(ema)
}

# Average True Range
ATR <- function(hlc, n) {
  hlc <- as.matrix(hlc)
  high <- hlc[, 1]
  low <- hlc[, 2]
  close <- hlc[, 3]

  tr <- pmax(
    high - low,
    abs(high - c(high[1], close[-length(close)])),
    abs(low - c(low[1], close[-length(close)]))
  )

  return(EMA(tr, n))
}

# HSV to RGB conversion
hsv_to_rgb <- function(h, s, v) {
  c <- v * s
  x <- c * (1 - abs((h / 60) %% 2 - 1))
  m <- v - c

  r <- g <- b <- 0

  if (h < 60) {
    r <- c; g <- x; b <- 0
  } else if (h < 120) {
    r <- x; g <- c; b <- 0
  } else if (h < 180) {
    r <- 0; g <- c; b <- x
  } else if (h < 240) {
    r <- 0; g <- x; b <- c
  } else if (h < 300) {
    r <- x; g <- 0; b <- c
  } else {
    r <- c; g <- 0; b <- x
  }

  return(list(
    r = as.integer((r + m) * 255),
    g = as.integer((g + m) * 255),
    b = as.integer((b + m) * 255)
  ))
}

# Crossover detection
crossover <- function(x, y) {
  x_curr <- x
  y_curr <- y
  x_prev <- c(x[1], x[-length(x)])
  y_prev <- c(y[1], y[-length(y)])

  cross <- (x_curr > y_curr) & (x_prev <= y_prev)
  cross[1] <- FALSE

  return(cross)
}

# Crossunder detection
crossunder <- function(x, y) {
  x_curr <- x
  y_curr <- y
  x_prev <- c(x[1], x[-length(x)])
  y_prev <- c(y[1], y[-length(y)])

  cross <- (x_curr < y_curr) & (x_prev >= y_prev)
  cross[1] <- FALSE

  return(cross)
}

# Function to calculate high-pass filter coefficients
calculate_high_pass_filter_coefficients <- function(long_period) {
  exp_term <- exp(-1.414 * pi / long_period)
  cos_term <- 2 * exp_term * cos(1.414 * pi / long_period)
  cos_term_squared <- cos_term * cos_term
  exp_term_squared <- exp_term * exp_term

  hpc1 <- (1 + cos_term - exp_term_squared) / 4
  hpc2 <- cos_term
  hpc3 <- -exp_term_squared

  return(list(hpc1 = hpc1, hpc2 = hpc2, hpc3 = hpc3))
}

# Function to calculate super smoother filter coefficients
calculate_super_smoother_filter_coefficients <- function(short_period) {
  exp_term <- exp(-1.414 * pi / short_period)
  cos_term <- 2 * exp_term * cos(1.414 * pi / short_period)
  cos_term_squared <- cos_term * cos_term
  exp_term_squared <- exp_term * exp_term

  ssc1 <- 1 - cos_term - exp_term_squared
  ssc2 <- cos_term
  ssc3 <- -exp_term_squared

  return(list(ssc1 = ssc1, ssc2 = ssc2, ssc3 = ssc3))
}

# Main function to calculate Ehlers Loops
ehlers_loops <- function(data, long_period = 20, short_period = 125) {

  close <- data$close
  volume <- data$volume

  n <- length(close)

  hp_coef <- calculate_high_pass_filter_coefficients(long_period)
  ss_coef <- calculate_super_smoother_filter_coefficients(short_period)

  hpc1 <- hp_coef$hpc1
  hpc2 <- hp_coef$hpc2
  hpc3 <- hp_coef$hpc3

  ssc1 <- ss_coef$ssc1
  ssc2 <- ss_coef$ssc2
  ssc3 <- ss_coef$ssc3

  # --- PRICE PROCESSING (Vectorized) ---

  # High-pass filter: calculate second-order difference
  price_diff <- c(0, 0, diff(close, differences = 2))

  high_pass_filtered_price <- stats::filter(
    x = hpc1 * price_diff,
    filter = c(hpc2, hpc3),
    method = "recursive",
    init = c(0, 0)
  )
  high_pass_filtered_price <- as.numeric(high_pass_filtered_price)

  # Super smoother: moving average then filter
  hp_price_avg <- (high_pass_filtered_price + c(0, high_pass_filtered_price[-n])) / 2

  smoothed_price <- stats::filter(
    x = ssc1 * hp_price_avg,
    filter = c(ssc2, ssc3),
    method = "recursive",
    init = c(0, 0)
  )
  smoothed_price <- as.numeric(smoothed_price)

  # Mean square with exponential smoothing
  smoothed_price_squared <- smoothed_price * smoothed_price

  price_mean_square <- stats::filter(
    x = 0.0242 * smoothed_price_squared,
    filter = 0.9758,
    method = "recursive",
    init = 0
  )
  price_mean_square <- as.numeric(price_mean_square)

  # Root mean square (avoid division by zero)
  price_root_mean_square <- smoothed_price / sqrt(pmax(price_mean_square, 1e-10))
  price_root_mean_square[is.nan(price_root_mean_square) | is.infinite(price_root_mean_square)] <- 0

  # --- VOLUME PROCESSING (Vectorized) ---

  # High-pass filter: calculate second-order difference
  volume_diff <- c(0, 0, diff(volume, differences = 2))

  high_pass_filtered_volume <- stats::filter(
    x = hpc1 * volume_diff,
    filter = c(hpc2, hpc3),
    method = "recursive",
    init = c(0, 0)
  )
  high_pass_filtered_volume <- as.numeric(high_pass_filtered_volume)

  # Super smoother: moving average then filter
  hp_volume_avg <- (high_pass_filtered_volume + c(0, high_pass_filtered_volume[-n])) / 2

  smoothed_volume <- stats::filter(
    x = ssc1 * hp_volume_avg,
    filter = c(ssc2, ssc3),
    method = "recursive",
    init = c(0, 0)
  )
  smoothed_volume <- as.numeric(smoothed_volume)

  # Mean square with exponential smoothing
  smoothed_volume_squared <- smoothed_volume * smoothed_volume

  volume_mean_square <- stats::filter(
    x = 0.0242 * smoothed_volume_squared,
    filter = 0.9758,
    method = "recursive",
    init = 0
  )
  volume_mean_square <- as.numeric(volume_mean_square)

  # Root mean square (avoid division by zero)
  volume_root_mean_square <- smoothed_volume / sqrt(pmax(volume_mean_square, 1e-10))
  volume_root_mean_square[is.nan(volume_root_mean_square) | is.infinite(volume_root_mean_square)] <- 0

  return(data.frame(
    date = data$date,
    price_rms = price_root_mean_square,
    volume_rms = volume_root_mean_square
  ))
}

# ============================================================================
# Main Calculation Function
# ============================================================================

calculate_supersmoother_oscillator <- function(
  data,
  smoothing_length = 5,
  fast_length = 20,
  slow_length = 50,
  atr_length = 20,
  kama_nER = 10,
  kama_nFast = 2,
  kama_nSlow = 30,
  signal_sensitivity = 0.03,
  ehlers_long_period = 20,
  ehlers_short_period = 125
) {
  if (nrow(data) <= max(slow_length, ehlers_short_period) * 2) stop()

  close <- data$close
  hlc <- xts(data[, c("high", "low", "close")], order.by = data$date)

  smoothed_price <- supersmoother(close, smoothing_length)

  fast_ma <- EMA(smoothed_price, fast_length)
  slow_ma <- EMA(smoothed_price, slow_length)

  oscillator <- fast_ma - slow_ma
  oscillator_norm <- (oscillator / smoothed_price) * 100

  signal_line <- EMA(oscillator, 25)
  signal_line_norm <- (signal_line / smoothed_price) * 100

  atr <- ATR(hlc, atr_length)

  # Rate of change (first derivative)
  oscillator_roc <- c(0, diff(oscillator))
  oscillator_roc_norm <- tanh(oscillator_roc / (atr * 0.01))

  # Acceleration (second derivative)
  oscillator_acc <- oscillator_roc - lag(oscillator_roc)

  # Color hue: red for positive momentum, green for negative
  hue_raw <- 60 - oscillator_roc_norm * 60
  hue <- as.numeric(stats::filter(hue_raw, filter = 0.5, method = "recursive"))

  kama <- as.numeric(KAMA(hlc, nER = kama_nER, nFast = kama_nFast, nSlow = kama_nSlow, priceMethod = "Ehlers's"))

  # Calculate Ehlers Loops
  ehlers <- ehlers_loops(data, long_period = ehlers_long_period, short_period = ehlers_short_period)

  # Calculate rate of change for price_rms and volume_rms
  price_rms_roc <- c(0, diff(ehlers$price_rms))
  volume_rms_roc <- c(0, diff(ehlers$volume_rms))

  # Normalize and calculate hues for price_rms
  price_rms_roc_norm <- tanh(price_rms_roc / (atr * 0.01))
  price_rms_hue_raw <- 60 - price_rms_roc_norm * 60
  price_rms_hue <- as.numeric(stats::filter(price_rms_hue_raw, filter = 0.5, method = "recursive"))

  # Normalize and calculate hues for volume_rms
  volume_rms_roc_norm <- tanh(volume_rms_roc / (atr * 0.01))
  volume_rms_hue_raw <- 60 - volume_rms_roc_norm * 60
  volume_rms_hue <- as.numeric(stats::filter(volume_rms_hue_raw, filter = 0.5, method = "recursive"))

  result <- data.frame(
    date = data$date,
    smoothed_price = smoothed_price,
    fast_ma = fast_ma,
    slow_ma = slow_ma,
    oscillator = oscillator,
    oscillator_norm = oscillator_norm,
    signal_line = signal_line,
    signal_line_norm = signal_line_norm,
    atr = atr,
    oscillator_roc = oscillator_roc,
    oscillator_acc = oscillator_acc,
    hue = hue,
    kama = kama,
    price_rms = ehlers$price_rms,
    volume_rms = ehlers$volume_rms,
    price_rms_hue = price_rms_hue,
    volume_rms_hue = volume_rms_hue
  )

  return(result)
}

# ============================================================================
# Bullish Condition Test
# ============================================================================

if_bullish <- function(
  result,
  signal_sensitivity = 0.3
) {
  oscillator <- result$oscillator
  signal_line <- result$signal_line
  atr <- result$atr
  oscillator_roc <- result$oscillator_roc

  min_signal_threshold <- atr * signal_sensitivity

  ifelse(
    oscillator > signal_line + min_signal_threshold &
      oscillator_roc >= 0,
    TRUE,
    FALSE
  )
}

# ============================================================================
# Plotting Functions
# ============================================================================

plot_supersmoother_indicator <- function(data, result) {

  plot_data <- data.frame(
    index = 1:nrow(data),
    date = data$date,
    open = data$open,
    high = data$high,
    low = data$low,
    close = data$close,
    oscillator = result$oscillator_norm,
    signal_line = result$signal_line_norm,
    hue = result$hue,
    kama = result$kama,
    atr = result$atr,
    is_bullish = result$is_bullish,
    price_rms = result$price_rms,
    volume_rms = result$volume_rms,
    price_rms_hue = result$price_rms_hue,
    volume_rms_hue = result$volume_rms_hue,
    candle_color = ifelse(data$close >= data$open, "red", "green")
  )

  # Convert hues to RGB colors
  oscillator_colors <- sapply(plot_data$hue, function(h) {
    if (is.na(h)) return("#FFFF00")
    rgb_vals <- hsv_to_rgb(h, 1.0, 1.0)
    return(rgb(rgb_vals$r, rgb_vals$g, rgb_vals$b, maxColorValue = 255))
  })

  plot_data$oscillator_color <- oscillator_colors

  n_breaks <- min(10, nrow(plot_data))
  date_breaks <- seq(1, nrow(plot_data), length.out = n_breaks)
  date_labels <- format(plot_data$date[date_breaks], "%Y-%m-%d")

  # =========================================================================
  # Plot 1: Candlestick Chart
  # =========================================================================

  bullish_bg <- data.frame(
    index = plot_data$index,
    ymin = min(plot_data$low, na.rm = TRUE),
    ymax = max(plot_data$high, na.rm = TRUE),
    is_bullish = plot_data$is_bullish
  )

  p1 <- ggplot(plot_data, aes(x = index)) +
    geom_rect(data = bullish_bg[bullish_bg$is_bullish, ],
              aes(xmin = index - 0.5, xmax = index + 0.5,
                  ymin = ymin, ymax = ymax),
              fill = "red", alpha = 0.15, inherit.aes = FALSE) +
    # Gray band around KAMA
    geom_ribbon(aes(ymin = kama - atr * 2.5, ymax = kama + atr * 2.5),
                fill = "gray", alpha = 0.5) +
    geom_segment(aes(xend = index, y = low, yend = high),
                 color = "gray30", linewidth = 0.3) +
    geom_rect(aes(xmin = index - 0.4, xmax = index + 0.4,
                  ymin = pmin(open, close), ymax = pmax(open, close),
                  fill = candle_color),
              color = "gray30", linewidth = 0.3) +
    scale_fill_identity() +
    geom_line(aes(y = kama, color = "KAMA"), linewidth = 0.5) +
    scale_color_manual(
      name = "",
      values = c("KAMA" = "black"),
      labels = c("KAMA")
    ) +
    scale_x_continuous(breaks = date_breaks, labels = date_labels) +
    labs(title = "Price Chart with Signal Indicators",
         y = "Price", x = NULL) +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "gray80"),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "top",
      legend.justification = "center",
      legend.direction = "horizontal",
      legend.box.margin = margin(0, 0, 0, 0),
      legend.margin = margin(0, 0, 0, 0),
      legend.spacing = unit(0, "pt"),
      legend.box.spacing = unit(0, "pt"),
      legend.key.size = unit(0.8, "lines"),
      legend.text = element_text(size = 9)
    )

  # =========================================================================
  # Plot 2: Oscillator Subplot
  # =========================================================================

  oscillator_bullish_bg <- data.frame(
    index = plot_data$index,
    ymin = min(c(plot_data$oscillator, plot_data$signal_line), na.rm = TRUE),
    ymax = max(c(plot_data$oscillator, plot_data$signal_line), na.rm = TRUE),
    is_bullish = plot_data$is_bullish
  )

  p2 <- ggplot(plot_data, aes(x = index)) +
    geom_rect(data = oscillator_bullish_bg[oscillator_bullish_bg$is_bullish, ],
              aes(xmin = index - 0.5, xmax = index + 0.5,
                  ymin = ymin, ymax = ymax),
              fill = "red", alpha = 0.15, inherit.aes = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed",
               color = "gray50", linewidth = 0.5) +
    geom_line(aes(y = oscillator, color = oscillator_color, group = 1),
              linewidth = 1) +
    scale_color_identity() +
    geom_line(aes(y = signal_line, linetype = "Signal Line"),
              color = "black", linewidth = 1) +
    scale_linetype_manual(
      name = "",
      values = c("Signal Line" = "solid"),
      labels = c("Signal Line")
    ) +
    scale_x_continuous(breaks = date_breaks, labels = date_labels) +
    labs(title = "SuperSmoother MA Oscillator",
         y = "Oscillator Value", x = NULL) +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "gray80"),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "top",
      legend.justification = "center",
      legend.direction = "horizontal",
      legend.box.margin = margin(0, 0, 0, 0),
      legend.margin = margin(0, 0, 0, 0),
      legend.spacing = unit(0, "pt"),
      legend.box.spacing = unit(0, "pt"),
      legend.key.size = unit(0.8, "lines"),
      legend.text = element_text(size = 9)
    )

  # =========================================================================
  # Plot 3: Price & Volume RMS Subplot
  # =========================================================================

  rms_bullish_bg <- data.frame(
    index = plot_data$index,
    ymin = min(c(plot_data$price_rms, plot_data$volume_rms), na.rm = TRUE),
    ymax = max(c(plot_data$price_rms, plot_data$volume_rms), na.rm = TRUE),
    is_bullish = plot_data$is_bullish
  )

  p3 <- ggplot(plot_data, aes(x = index)) +
    geom_rect(data = rms_bullish_bg[rms_bullish_bg$is_bullish, ],
              aes(xmin = index - 0.5, xmax = index + 0.5,
                  ymin = ymin, ymax = ymax),
              fill = "red", alpha = 0.15, inherit.aes = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed",
               color = "gray50", linewidth = 0.5) +
    geom_line(aes(y = price_rms, color = "Price RMS"),
              linewidth = 1) +
    geom_line(aes(y = volume_rms, color = "Volume RMS"),
              linewidth = 1) +
    scale_color_manual(
      name = "",
      values = c("Price RMS" = "lightblue", "Volume RMS" = "navyblue"),
      labels = c("Price RMS", "Volume RMS")
    ) +
    scale_x_continuous(breaks = date_breaks, labels = date_labels) +
    labs(title = "Ehlers Loops",
         y = "RMS Value", x = "Date") +
    theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "gray80"),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "top",
      legend.justification = "center",
      legend.direction = "horizontal",
      legend.box.margin = margin(0, 0, 0, 0),
      legend.margin = margin(0, 0, 0, 0),
      legend.spacing.x = unit(10, "pt"),
      legend.box.spacing = unit(0, "pt"),
      legend.key.size = unit(0.8, "lines"),
      legend.text = element_text(size = 9)
    )

  combined_plot <- p1 / p2 / p3 +
    plot_layout(heights = c(2, 1, 1))

  return(combined_plot)
}

# ============================================================================
# Usage Example with Plotting
# ============================================================================

symbol <- "600114"
end_date <- today()
start_date <- end_date - years(1)

data <- read_csv(paste0("data/hist/", symbol, ".csv")) %>%
  left_join(read_csv(paste0("data/adjust/", symbol, ".csv")), by = "date") %>%
  arrange(date) %>%
  fill(adjust, .direction = "down") %>%
  mutate(
    across(c(open, high, low, close), ~ .x * adjust),
    volume = volume / adjust
  ) %>%
  filter(date >= start_date %m-% years(20))

result <- calculate_supersmoother_oscillator(data) %>%
  mutate(is_bullish = if_bullish(.))

plot <- plot_supersmoother_indicator(
  data = filter(data, date >= start_date & date <= end_date),
  result = filter(result, date >= start_date & date <= end_date)
)

print(plot)

ggsave("supersmoother_indicator.png", plot, width = 14, height = 10, dpi = 300)