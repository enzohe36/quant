# https://www.tradingview.com/scripts/ehlers/
# https://www.tradingview.com/script/e8DZtqQL/
# https://www.tradingview.com/script/559mGm7c/

# library(xts)
# library(DSTrading)
# library(patchwork)
# library(foreach)
# library(doFuture)
# library(tidyverse)

# ============================================================================
# Helper Functions
# ============================================================================

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

crossover <- function(x, y) replace_missing((x > y) & lag(x <= y), FALSE)

crossunder <- function(x, y) replace_missing((x < y) & lag(x >= y), FALSE)

momentum <- function(x) c(0, diff(x))

calculate_highpass_coefs <- function(highpass_cutoff) {
  exp_term <- exp(-sqrt(2) * pi / highpass_cutoff)
  cos_term <- 2 * exp_term * cos(sqrt(2) * pi / highpass_cutoff)
  exp_term_squared <- exp_term * exp_term

  hpc1 <- (1 + cos_term - exp_term_squared) / 4
  hpc2 <- cos_term
  hpc3 <- -exp_term_squared

  return(list(hpc1 = hpc1, hpc2 = hpc2, hpc3 = hpc3))
}

calculate_smoothing_coefs <- function(lowpass_cutoff) {
  exp_term <- exp(-sqrt(2) * pi / lowpass_cutoff)
  cos_term <- 2 * exp_term * cos(sqrt(2) * pi / lowpass_cutoff)
  exp_term_squared <- exp_term * exp_term

  sc1 <- 1 - cos_term - exp_term_squared
  sc2 <- cos_term
  sc3 <- -exp_term_squared

  return(list(sc1 = sc1, sc2 = sc2, sc3 = sc3))
}

# ============================================================================
# SuperSmoother MA Oscillator
# ============================================================================

calculate_supersmoother_oscillator <- function(
  data,
  smoothing_length = 5,
  fast_length = 20,
  slow_length = 50,
  atr_length = 20,
  hue_multiplier = 100
) {
  close <- data$close
  hlc <- xts(data[, c("high", "low", "close")], order.by = data$date)

  smoothed_price <- supersmoother(close, smoothing_length)

  fast_ma <- EMA(smoothed_price, fast_length)
  slow_ma <- EMA(smoothed_price, slow_length)

  atr <- as.numeric(ATR(hlc, atr_length)[, "atr"])
  smoothed_atr <- supersmoother(replace_missing(atr, 0), smoothing_length)

  oscillator <- (fast_ma - slow_ma) / smoothed_atr
  signal_line <- EMA(oscillator, 25)

  osc_d1 <- momentum(oscillator)
  osc_d1_norm <- tanh(osc_d1 * hue_multiplier)

  hue_raw <- 60 - osc_d1_norm * 60
  hue <- stats::filter(
    x = replace_missing(hue_raw, 0),
    filter = 0.5,
    method = "recursive"
  )

  return(data.frame(
    date = data$date,
    oscillator = oscillator,
    signal_line = signal_line,
    hue = hue
  ))
}

# ============================================================================
# KAMA
# ============================================================================

calculate_kama <- function(
  data,
  atr_length = 20,
  atr_multiplier = 3,
  nER = 10,
  nFast = 2,
  nSlow = 30
) {
  hlc <- xts(data[, c("high", "low", "close")], order.by = data$date)

  atr <- as.numeric(ATR(hlc, atr_length)[, "atr"])

  kama <- as.numeric(KAMA(hlc, nER, nFast, nSlow, "Ehlers's"))
  upper_bound <- kama + atr_multiplier * atr
  lower_bound <- kama - atr_multiplier * atr

  return(data.frame(
    date = data$date,
    kama = kama,
    upper_bound = upper_bound,
    lower_bound = lower_bound
  ))
}

# ============================================================================
# Ehlers Loops
# ============================================================================

calculate_ehlers_loops <- function(
  data,
  highpass_cutoff = 20,
  lowpass_cutoff = 125,
  alpha = 0.0242
) {
  close <- data$close
  volume <- data$volume
  n <- length(close)

  highpass_coefs <- calculate_highpass_coefs(highpass_cutoff)
  smoothing_coefs <- calculate_smoothing_coefs(lowpass_cutoff)

  hpc1 <- highpass_coefs$hpc1
  hpc2 <- highpass_coefs$hpc2
  hpc3 <- highpass_coefs$hpc3

  sc1 <- smoothing_coefs$sc1
  sc2 <- smoothing_coefs$sc2
  sc3 <- smoothing_coefs$sc3

  # Price processing
  price_diff <- momentum(momentum(close))

  highpass_price <- stats::filter(
    x = replace_missing(hpc1 * price_diff, 0),
    filter = c(hpc2, hpc3),
    method = "recursive",
    init = c(0, 0)
  )
  highpass_price <- as.numeric(highpass_price)

  highpass_price_avg <- (highpass_price + lag(highpass_price)) / 2

  smoothed_price <- stats::filter(
    x = replace_missing(sc1 * highpass_price_avg, 0),
    filter = c(sc2, sc3),
    method = "recursive",
    init = c(0, 0)
  )
  smoothed_price <- as.numeric(smoothed_price)

  price_ms <- stats::filter(
    x = alpha * smoothed_price^2,
    filter = 1 - alpha,
    method = "recursive",
    init = 0
  )
  price_ms <- as.numeric(price_ms)

  price_rms <- smoothed_price / sqrt(pmax(price_ms, 1e-10))
  price_rms[is.nan(price_rms) | is.infinite(price_rms)] <- 0

  # Volume processing
  volume_diff <- momentum(momentum(volume))

  highpass_volume <- stats::filter(
    x = replace_missing(hpc1 * volume_diff, 0),
    filter = c(hpc2, hpc3),
    method = "recursive",
    init = c(0, 0)
  )
  highpass_volume <- as.numeric(highpass_volume)

  highpass_volume_avg <- (highpass_volume + lag(highpass_volume)) / 2

  smoothed_volume <- stats::filter(
    x = replace_missing(sc1 * highpass_volume_avg, 0),
    filter = c(sc2, sc3),
    method = "recursive",
    init = c(0, 0)
  )
  smoothed_volume <- as.numeric(smoothed_volume)

  volume_ms <- stats::filter(
    x = alpha * smoothed_volume^2,
    filter = 1 - alpha,
    method = "recursive",
    init = 0
  )
  volume_ms <- as.numeric(volume_ms)

  volume_rms <- smoothed_volume / sqrt(pmax(volume_ms, 1e-10))
  volume_rms[is.nan(volume_rms) | is.infinite(volume_rms)] <- 0

  return(data.frame(
    date = data$date,
    price_rms = price_rms,
    volume_rms = volume_rms
  ))
}

# ============================================================================
# Bullish Condition Test
# ============================================================================

if_bullish <- function(
  result,
  zero_threshold = 0.05,
  price_lookback = 10,
  min_price_diff = 0.5,
  price_lookforward = 5,
  min_signal = 0.35,
  min_osc_d1 = -0.01,
  osc_lookback = 40,
  max_osc_diff = 1.9,
  min_required_length = 0
) {
  oscillator <- result$oscillator
  signal_line <- result$signal_line
  price_rms <- result$price_rms
  volume_rms <- result$volume_rms
  n <- nrow(result)

  # Price start conditions
  price_d1 <- momentum(price_rms)
  price_d2 <- momentum(price_d1)
  price_near_max <- (abs(price_d1) <= zero_threshold) & (price_d2 < 0)

  price_diff <- price_rms - runMin(price_rms, price_lookback)

  price_start <- (
    (price_near_max) & (price_diff >= min_price_diff)
  ) | (
    (price_d1 < -zero_threshold)
  )

  # Volume end conditions
  volume_d1 <- momentum(volume_rms)
  volume_end <- (volume_d1 > zero_threshold)

  # Detect run starts
  price_start_run_begins <- (price_start) & (!lag(price_start))
  volume_end_run_begins <- (volume_end) & (!lag(volume_end))

  # Track days since last price_start
  price_start_positions <- ifelse(price_start_run_begins, 1:n, 0)
  last_price_start_position <- cummax(replace_na(price_start_positions, 0))

  days_since_price_start <- ifelse(
    last_price_start_position > 0,
    (1:n) - last_price_start_position,
    Inf
  )

  # Protect volume_end runs that start within protection period
  volume_end_run_needs_protection <- volume_end_run_begins &
    (days_since_price_start < price_lookforward)

  volume_end_run_id <- cumsum(replace_na(volume_end_run_begins, FALSE))
  volume_end_run_id[!volume_end] <- 0

  protected_run_ids <- unique(volume_end_run_id[volume_end_run_needs_protection & volume_end])

  volume_end_protected <- volume_end
  volume_end_protected[volume_end_run_id %in% protected_run_ids] <- FALSE

  # Oscillator start conditions
  strong_signal <- oscillator - signal_line >= min_signal
  osc_d1 <- momentum(oscillator)
  osc_diff <- oscillator - runMin(oscillator, osc_lookback)
  osc_start <- (strong_signal) & (osc_d1 > min_osc_d1)

  is_bullish <- (
    (price_start & !volume_end_protected) |
      (osc_start)
  ) & (
    (osc_diff <= max_osc_diff)
  )

  # Set first min_required_length elements to FALSE
  warmup_length <- min(min_required_length, length(is_bullish))
  is_bullish[1:warmup_length] <- FALSE

  result$is_bullish <- is_bullish

  return(result)
}

# ============================================================================
# Feature Generation
# ============================================================================

generate_features <- function(
  data_combined,
  start_date,
  end_date,
  # supersmoother oscillator args
  smoothing_length = 5,
  fast_length = 20,
  slow_length = 50,
  atr_length = 20,
  hue_multiplier = 100,
  # kama args
  kama_atr_length = 20,
  atr_multiplier = 3,
  nER = 10,
  nFast = 2,
  nSlow = 30,
  # ehlers loops args
  highpass_cutoff = 20,
  lowpass_cutoff = 125,
  alpha = 0.0242,
  # if_bullish args
  zero_threshold = 0.05,
  price_lookback = 10,
  min_price_diff = 0.5,
  price_lookforward = 5,
  min_signal = 0.35,
  min_osc_d1 = -0.01,
  osc_lookback = 40,
  max_osc_diff = 1.9
) {
  # Calculate min_required_length
  min_required_length <- max(slow_length, nSlow, lowpass_cutoff) * 2
  start_date <- start_date - days(min_required_length * 2)

  # Process in parallel
  plan(multisession, workers = availableCores() - 1)

  data_combined <- foreach(data = data_combined, .combine = "c") %dofuture% {
    # Filter to keep last n_rows_needed rows (or all if fewer)
    data <- filter(data, date >= start_date & date <= end_date)
    if (nrow(data) <= min_required_length) return(NULL)

    # Calculate indicators
    result <- cbind(
      select(
        calculate_supersmoother_oscillator(
          data,
          smoothing_length = smoothing_length,
          fast_length = fast_length,
          slow_length = slow_length,
          atr_length = atr_length,
          hue_multiplier = hue_multiplier
        ),
        -date
      ),
      select(
        calculate_kama(
          data,
          atr_length = kama_atr_length,
          atr_multiplier = atr_multiplier,
          nER = nER,
          nFast = nFast,
          nSlow = nSlow
        ),
        -date
      ),
      select(
        calculate_ehlers_loops(
          data,
          highpass_cutoff = highpass_cutoff,
          lowpass_cutoff = lowpass_cutoff,
          alpha = alpha
        ),
        -date
      )
    ) %>%
      if_bullish(
        zero_threshold = zero_threshold,
        price_lookback = price_lookback,
        min_price_diff = min_price_diff,
        price_lookforward = price_lookforward,
        min_signal = min_signal,
        min_osc_d1 = min_osc_d1,
        osc_lookback = osc_lookback,
        max_osc_diff = max_osc_diff,
        min_required_length = min_required_length
      )

    # Combine with original data, keeping only is_bullish from result
    for (col_name in names(result)) data[[col_name]] <- result[[col_name]]
    data <- select(
      data,
      symbol, date, open, high, low, close, volume, mc,
      oscillator, signal_line, hue,
      kama, upper_bound, lower_bound,
      price_rms, volume_rms,
      is_bullish
    )

    # Return as list element for foreach to combine
    my_list <- list()
    my_list[[first(data$symbol)]] <- data
    return(my_list)
  }

  plan(sequential)

  return(data_combined)
}

# ============================================================================
# Plotting
# ============================================================================

plot_supersmoother_indicator <- function(data, result, spot) {
  # Extract symbol and name
  stock_symbol <- unique(data$symbol)
  stock_name <- filter(spot, symbol == stock_symbol)$name
  plot_title <- paste0(stock_symbol, " - ", stock_name)

  plot_data <- data.frame(
    index = 1:nrow(data),
    date = data$date,
    open = data$open,
    high = data$high,
    low = data$low,
    close = data$close,
    oscillator = result$oscillator,
    signal_line = result$signal_line,
    hue = result$hue,
    kama = result$kama,
    upper_bound = result$upper_bound,
    lower_bound = result$lower_bound,
    price_rms = result$price_rms,
    volume_rms = result$volume_rms,
    is_bullish = result$is_bullish,
    candle_color = ifelse(data$close >= data$open, "red", "green")
  )

  oscillator_colors <- sapply(plot_data$hue, function(h) {
    if (is.na(h)) return("#FFFF00")
    rgb_vals <- hsv_to_rgb(h, 1.0, 1.0)
    return(rgb(rgb_vals$r, rgb_vals$g, rgb_vals$b, maxColorValue = 255))
  })

  plot_data$oscillator_color <- oscillator_colors

  n_breaks <- min(10, nrow(plot_data))
  date_breaks <- seq(1, nrow(plot_data), length.out = n_breaks)
  date_labels <- format(plot_data$date[date_breaks], "%Y-%m-%d")

  # Calculate y-axis limits with padding
  price_range <- range(c(plot_data$low, plot_data$high, plot_data$lower_bound, plot_data$upper_bound), na.rm = TRUE)
  price_padding <- diff(price_range) * 0.02
  price_ylim <- c(price_range[1] - price_padding, price_range[2] + price_padding)

  osc_range <- range(c(plot_data$oscillator, plot_data$signal_line), na.rm = TRUE)
  osc_padding <- diff(osc_range) * 0.02
  osc_ylim <- c(osc_range[1] - osc_padding, osc_range[2] + osc_padding)

  rms_range <- range(c(plot_data$price_rms, plot_data$volume_rms), na.rm = TRUE)
  rms_padding <- diff(rms_range) * 0.02
  rms_ylim <- c(rms_range[1] - rms_padding, rms_range[2] + rms_padding)

  # Candlestick chart
  bullish_bg <- data.frame(
    index = plot_data$index,
    ymin = price_ylim[1],
    ymax = price_ylim[2],
    is_bullish = plot_data$is_bullish
  )

  p1 <- ggplot(plot_data, aes(x = index)) +
    geom_rect(data = bullish_bg[bullish_bg$is_bullish, ],
              aes(xmin = index - 0.5, xmax = index + 0.5,
                  ymin = ymin, ymax = ymax),
              fill = "red", alpha = 0.15, inherit.aes = FALSE) +
    geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound),
                fill = NA, color = "black", linewidth = 0.5) +
    geom_segment(aes(xend = index, y = low, yend = high),
                 color = "gray30", linewidth = 0.3) +
    geom_rect(aes(xmin = index - 0.4, xmax = index + 0.4,
                  ymin = pmin(open, close), ymax = pmax(open, close),
                  fill = candle_color),
              color = "gray30", linewidth = 0.3) +
    scale_fill_identity() +
    scale_x_continuous(breaks = date_breaks, labels = date_labels) +
    scale_y_continuous(limits = price_ylim, expand = c(0, 0)) +
    labs(title = plot_title, y = "Price", x = NULL) +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "gray80"),
      plot.title = element_text(hjust = 0.5, face = "bold", margin = margin(0, 0, 0, 0)),
      legend.position = "top",
      legend.justification = "center",
      legend.direction = "horizontal",
      legend.margin = margin(0, 0, -10, 0)
    )

  # Oscillator subplot
  oscillator_bullish_bg <- data.frame(
    index = plot_data$index,
    ymin = osc_ylim[1],
    ymax = osc_ylim[2],
    is_bullish = plot_data$is_bullish
  )

  p2 <- ggplot(plot_data, aes(x = index)) +
    geom_rect(data = oscillator_bullish_bg[oscillator_bullish_bg$is_bullish, ],
              aes(xmin = index - 0.5, xmax = index + 0.5,
                  ymin = ymin, ymax = ymax),
              fill = "red", alpha = 0.15, inherit.aes = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed",
               color = "gray50", linewidth = 0.5) +
    # Actual oscillator line with dynamic colors (no legend)
    geom_line(aes(y = oscillator, color = oscillator_color, group = 1),
              linewidth = 1) +
    scale_color_identity() +
    # Invisible dummy line for oscillator legend
    geom_line(aes(y = oscillator, linetype = "Oscillator"),
              color = "darkorange", linewidth = 1, alpha = 0) +
    # Signal line
    geom_line(aes(y = signal_line, linetype = "Signal Line"),
              color = "black", linewidth = 1) +
    scale_linetype_manual(
      name = "",
      values = c("Oscillator" = "solid", "Signal Line" = "solid"),
      guide = guide_legend(
        override.aes = list(
          color = c("darkorange", "black"),
          alpha = c(1, 1),
          linewidth = c(1, 1)
        )
      )
    ) +
    scale_x_continuous(breaks = date_breaks, labels = date_labels) +
    scale_y_continuous(limits = osc_ylim, expand = c(0, 0)) +
    labs(y = "Oscillator", x = NULL) +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "gray80"),
      legend.position = "top",
      legend.justification = "center",
      legend.direction = "horizontal",
      legend.margin = margin(-10, 0, -10, 0)
    )

  # RMS subplot
  rms_bullish_bg <- data.frame(
    index = plot_data$index,
    ymin = rms_ylim[1],
    ymax = rms_ylim[2],
    is_bullish = plot_data$is_bullish
  )

  p3 <- ggplot(plot_data, aes(x = index)) +
    geom_rect(data = rms_bullish_bg[rms_bullish_bg$is_bullish, ],
              aes(xmin = index - 0.5, xmax = index + 0.5,
                  ymin = ymin, ymax = ymax),
              fill = "red", alpha = 0.15, inherit.aes = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed",
               color = "gray50", linewidth = 0.5) +
    geom_line(aes(y = price_rms, color = "Price RMS"), linewidth = 1) +
    geom_line(aes(y = volume_rms, color = "Volume RMS"), linewidth = 1) +
    scale_color_manual(
      name = "",
      values = c("Price RMS" = "lightblue", "Volume RMS" = "navyblue")
    ) +
    scale_x_continuous(breaks = date_breaks, labels = date_labels) +
    scale_y_continuous(limits = rms_ylim, expand = c(0, 0)) +
    labs(y = "RMS", x = "Date") +
    theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "gray80"),
      legend.position = "top",
      legend.justification = "center",
      legend.direction = "horizontal",
      legend.margin = margin(-10, 0, -10, 0),
      axis.title.x = element_text(margin = margin(10, 0, 0, 0))
    )

  combined_plot <- p1 / p2 / p3 +
    plot_layout(heights = c(2, 1, 1))

  return(combined_plot)
}

# ============================================================================
# Usage Example
# ============================================================================

# symbol <- "300378"
# end_date <- today()
# start_date <- end_date - years(2)

# spot <- read_csv("data/spot.csv")  # Load spot dataframe with symbol and name columns
# hist <- read_csv(paste0("data/hist/", symbol, ".csv"))
# adjust <- read_csv(paste0("data/adjust/", symbol, ".csv"))
# data <- left_join(hist, adjust, by = "date") %>%
#   arrange(date) %>%
#   fill(adjust, .direction = "down") %>%
#   mutate(
#     symbol = symbol,
#     across(c(open, high, low, close), ~ .x * adjust),
#     volume = volume / adjust
#   ) %>%
#   filter(date %in% hist$date)

# result <- cbind(
#   calculate_supersmoother_oscillator(data),
#   select(calculate_kama(data), -date),
#   select(calculate_ehlers_loops(data), -date)
# ) %>%
#   if_bullish()

# plot <- plot_supersmoother_indicator(
#   data = filter(data, date >= start_date & date <= end_date),
#   result = filter(result, date >= start_date & date <= end_date),
#   spot = spot
# )

# print(plot)

# ggsave("supersmoother_indicator.png", plot, width = 14, height = 10, dpi = 300)