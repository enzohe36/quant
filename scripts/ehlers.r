# ehlers.r
# Ehlers DSP feature ensemble for RL stock trading.
#
# Feature ranges:
#   [-3, +3]  RMS-normalized (price spreads, slopes, volume)
#   [-1, +1]  Bounded oscillators (EBSW, elegant, CTI, correlations)
#   [-1, +1]  Running percentile rank (OHLC, fundamentals _rn)
#   [-1, +1]  Cross-sectional percentile rank (fundamentals _dn, applied externally)
#
# IIR warmup rows are set to NA_real_ (transient not yet settled).

HP_PERIOD  <- 120
LP_PERIOD  <- 30
RMS_PERIOD <- 60
PCT_PERIOD <- 240
GRID       <- c(30, 60, 120)

# HELPER FUNCTIONS =============================================================

safe_div <- function(a, b) ifelse(b != 0, a / b, 0)

clamp <- function(x, limit = 3) pmax(-limit, pmin(limit, x))

blank_warmup <- function(df, n_warmup) {
  if (n_warmup > 0 && n_warmup < nrow(df)) df[1:n_warmup, ] <- NA_real_
  df
}

# IIR FILTERS ==================================================================

butterworth_coefs <- function(period) {
  alpha <- sqrt(2) * pi / period
  a1    <- exp(-alpha)
  list(c2 = 2 * a1 * cos(alpha), c3 = -a1^2)
}

highpass_2pole <- function(x, period = HP_PERIOD) {
  bw <- butterworth_coefs(period)
  c1 <- (1 + bw$c2 - bw$c3) / 4
  u  <- stats::filter(x, c(c1, -2 * c1, c1), method = "convolution", sides = 1)
  u[is.na(u)] <- 0
  as.numeric(stats::filter(u, c(bw$c2, bw$c3), method = "recursive", init = c(0, 0)))
}

highpass_1pole <- function(x, period = HP_PERIOD) {
  a1 <- (1 - sin(2 * pi / period)) / cos(2 * pi / period)
  dx <- 0.5 * (1 + a1) * c(0, diff(x))
  as.numeric(stats::filter(dx, a1, method = "recursive", init = 0))
}

ultimate_smoother <- function(x, period = LP_PERIOD) {
  bw <- butterworth_coefs(period)
  c1 <- (1 + bw$c2 - bw$c3) / 4
  u  <- stats::filter(x, c(1 - c1, 2 * c1 - bw$c2, -(c1 + bw$c3)),
                      method = "convolution", sides = 1)
  u[is.na(u)] <- 0
  as.numeric(stats::filter(u, c(bw$c2, bw$c3), method = "recursive", init = c(0, 0)))
}

roofing_filter <- function(x, hp_period = HP_PERIOD, lp_period = LP_PERIOD) {
  ultimate_smoother(highpass_2pole(x, hp_period), lp_period)
}

laguerre_filter <- function(x, gamma) {
  n <- length(x)
  L0 <- as.numeric(stats::filter((1 - gamma) * x, gamma, method = "recursive", init = x[1]))

  cascade <- function(prev) {
    u    <- -gamma * prev + c(0, prev[-n])
    u[1] <- 0
    as.numeric(stats::filter(u, gamma, method = "recursive", init = 0))
  }

  L1 <- cascade(L0)
  L2 <- cascade(L1)
  L3 <- cascade(L2)
  (L0 + 2 * L1 + 2 * L2 + L3) / 6
}

# NORMALIZATION ================================================================

# Cross-sectional percentile rank, scaled to [-1, +1]. Uses dplyr::percent_rank.
cross_pctrank <- function(x) 2 * percent_rank(x) - 1

# Rolling midpoint percentile rank over a trailing window, scaled to [-1, +1].
running_pctrank <- function(x, period = PCT_PERIOD) {
  n <- length(x)
  if (n < period) return(rep(NA_real_, n))

  mat   <- embed(x, period)
  cur   <- mat[, 1]
  valid <- !is.na(mat)
  nv    <- rowSums(valid)
  lt    <- rowSums(valid & (mat < cur), na.rm = TRUE)
  eq    <- rowSums(valid & (mat == cur), na.rm = TRUE)
  rank  <- lt + 0.5 * (eq - 1)
  p     <- ifelse(nv > 1 & !is.na(cur), 2 * rank / (nv - 1) - 1, NA_real_)

  c(rep(NA_real_, period - 1), p)
}

# Exponential RMS automatic gain control, clamped to [-limit, +limit].
rms_normalize <- function(x, period = RMS_PERIOD, limit = 3) {
  alpha <- 2 / (period + 1)
  ms    <- as.numeric(stats::filter(alpha * x^2, 1 - alpha, method = "recursive", init = 0))
  clamp(ifelse(ms > 0, x / sqrt(ms), 0), limit)
}

# Rolling Pearson correlation via cumulative sums. O(n).
rolling_correlation <- function(a, b, window) {
  n <- length(a)
  if (window > n) return(rep(0, n))

  cs_a  <- c(0, cumsum(a))
  cs_b  <- c(0, cumsum(b))
  cs_ab <- c(0, cumsum(a * b))
  cs_a2 <- c(0, cumsum(a^2))
  cs_b2 <- c(0, cumsum(b^2))

  hi  <- (window + 1):(n + 1)
  lo  <- hi - window
  sa  <- cs_a[hi]  - cs_a[lo]
  sb  <- cs_b[hi]  - cs_b[lo]
  sab <- cs_ab[hi] - cs_ab[lo]
  sa2 <- cs_a2[hi] - cs_a2[lo]
  sb2 <- cs_b2[hi] - cs_b2[lo]

  denom <- (window * sa2 - sa^2) * (window * sb2 - sb^2)
  r     <- ifelse(denom > 0, (window * sab - sa * sb) / sqrt(denom), 0)
  r[is.na(r)] <- 0
  c(rep(0, window - 1), r)
}

# Correlation trend indicator: Pearson correlation of price vs linear ramp.
correlation_trend <- function(x, period) {
  n <- length(x)
  if (n < period) return(rep(0, n))

  ramp     <- seq_len(period)
  sum_ramp <- sum(ramp)
  ss_ramp  <- sum(ramp^2)

  cs_x  <- c(0, cumsum(x))
  cs_x2 <- c(0, cumsum(x^2))
  cs_jx <- c(0, cumsum(seq_len(n) * x))

  hi  <- period:n
  lo  <- hi - period + 1
  sx  <- cs_x[hi + 1]  - cs_x[lo]
  sx2 <- cs_x2[hi + 1] - cs_x2[lo]
  sxy <- (cs_jx[hi + 1] - cs_jx[lo]) - (lo - 1) * sx

  denom <- sqrt((period * sx2 - sx^2) * (period * ss_ramp - sum_ramp^2))
  r     <- ifelse(denom > 0, (period * sxy - sx * sum_ramp) / denom, 0)
  c(rep(0, period - 1), r)
}

# FEATURE GENERATORS ===========================================================

# Bandpass-filtered price at each grid period. Output [-3, +3].
feat_roofing <- function(x, periods = GRID) {
  out <- lapply(periods, function(p) rms_normalize(roofing_filter(x, 2 * p, p)))
  names(out) <- paste0("roof_", periods)
  blank_warmup(as.data.frame(out), max(2 * periods, RMS_PERIOD))
}

# Price deviation, trend slope, cross-scale divergence. Output [-3, +3].
feat_smoother <- function(x, periods = GRID) {
  smoothed <- lapply(periods, function(p) ultimate_smoother(x, p))
  names(smoothed) <- periods

  out <- list()
  for (p in periods) {
    out[[paste0("us_spread_", p)]] <- rms_normalize(x - smoothed[[as.character(p)]])
    out[[paste0("us_slope_", p)]]  <- rms_normalize(c(0, diff(smoothed[[as.character(p)]])))
  }
  for (i in seq_len(length(periods) - 1)) {
    fast <- smoothed[[as.character(periods[i])]]
    slow <- smoothed[[as.character(periods[i + 1])]]
    out[[paste0("us_xscale_", periods[i], "_", periods[i + 1])]] <- rms_normalize(fast - slow)
  }
  blank_warmup(as.data.frame(out), max(periods, RMS_PERIOD))
}

# Even Better Sinewave: cycle-phase detector. Output [-1, +1].
even_better_sinewave <- function(x, period) {
  filtered <- ultimate_smoother(highpass_1pole(x, period), period)
  power    <- as.numeric(stats::filter(0.05 * filtered^2, 0.95, method = "recursive", init = 0))
  clamp(ifelse(power > 0, filtered / sqrt(power), 0), 1)
}

feat_sinewave <- function(x, periods = GRID) {
  out <- lapply(periods, function(p) even_better_sinewave(x, p))
  names(out) <- paste0("ebsw_", periods)
  blank_warmup(as.data.frame(out), max(periods))
}

# Elegant oscillator: bounded momentum via tanh-normalized derivative. Output [-1, +1].
feat_elegant <- function(x, periods = GRID) {
  n     <- length(x)
  deriv <- c(0, 0, x[3:n] - x[1:(n - 2)])

  cs_sq     <- c(0, cumsum(deriv^2))
  idx       <- seq_len(n)
  start_idx <- pmax(1L, idx - RMS_PERIOD + 1L)
  rms_deriv <- sqrt((cs_sq[idx + 1] - cs_sq[start_idx]) / (idx - start_idx + 1L))

  bounded <- tanh(ifelse(rms_deriv > 0, deriv / rms_deriv, 0))

  out <- lapply(periods, function(p) clamp(ultimate_smoother(bounded, p), 1))
  names(out) <- paste0("eleg_", periods)
  blank_warmup(as.data.frame(out), max(periods, RMS_PERIOD))
}

# Correlation trend at grid periods + regime change rate. Output [-1, +1].
feat_trend <- function(x, periods = GRID) {
  out <- lapply(periods, function(p) correlation_trend(x, p))
  names(out) <- paste0("cti_", periods)
  out[["cti_chg"]] <- running_pctrank(c(0, abs(diff(correlation_trend(x, periods[2])))))
  blank_warmup(as.data.frame(out), max(periods, PCT_PERIOD))
}

# Laguerre spread: price minus adaptive smoother. Output [-3, +3].
feat_laguerre <- function(x, gammas = c(0.3, 0.6, 0.8)) {
  out <- lapply(gammas, function(g) rms_normalize(x - laguerre_filter(x, g)))
  names(out) <- paste0("lag_spread_", sub("\\.", "", gammas))
  blank_warmup(as.data.frame(out), RMS_PERIOD)
}

# MAMA/FAMA adaptive moving average spread and crossover. Output [-3, +3].
# Data-dependent alpha — cannot vectorize; requires loop.
mama_fama <- function(x, fast_limit = 0.5, slow_limit = 0.05) {
  n <- length(x)

  smooth <- as.numeric(stats::filter(x, c(4, 3, 2, 1) / 10, method = "convolution", sides = 1))
  smooth[is.na(smooth)] <- x[is.na(smooth)]

  hilbert_weights <- c(0.0962, 0, 0.5769, 0, -0.5769, 0, -0.0962)

  hilbert_fir <- function(sig, i) {
    sig[i]     * hilbert_weights[1] + sig[i - 2] * hilbert_weights[3] +
    sig[i - 4] * hilbert_weights[5] + sig[i - 6] * hilbert_weights[7]
  }

  detrender <- quad <- inphase <- j_inphase <- j_quad <- numeric(n)
  inphase2 <- quad2 <- re <- im <- numeric(n)
  period <- rep(20, n)
  phase  <- alpha <- numeric(n)
  mama   <- fama <- x

  if (n < 7) return(list(mama = mama, fama = fama))

  for (i in 7:n) {
    adj <- 0.075 * period[i - 1] + 0.54

    detrender[i] <- hilbert_fir(smooth, i) * adj
    quad[i]      <- hilbert_fir(detrender, i) * adj
    inphase[i]   <- detrender[i - 3]
    j_inphase[i] <- hilbert_fir(inphase, i) * adj
    j_quad[i]    <- hilbert_fir(quad, i) * adj

    inphase2[i] <- 0.2 * (inphase[i] - j_quad[i])    + 0.8 * inphase2[i - 1]
    quad2[i]    <- 0.2 * (quad[i]     + j_inphase[i]) + 0.8 * quad2[i - 1]

    re[i] <- 0.2 * (inphase2[i] * inphase2[i - 1] + quad2[i] * quad2[i - 1]) + 0.8 * re[i - 1]
    im[i] <- 0.2 * (inphase2[i] * quad2[i - 1]    - quad2[i] * inphase2[i - 1]) + 0.8 * im[i - 1]

    period[i] <- if (im[i] != 0 && re[i] != 0) 2 * pi / atan(im[i] / re[i]) else period[i - 1]
    period[i] <- max(6, min(50, max(0.67 * period[i - 1], min(1.5 * period[i - 1], period[i]))))
    period[i] <- 0.2 * period[i] + 0.8 * period[i - 1]

    phase[i] <- if (inphase[i] != 0) atan(quad[i] / inphase[i]) * 180 / pi else phase[i - 1]
    delta_phase <- max(1, phase[i - 1] - phase[i])
    alpha[i]    <- max(slow_limit, min(fast_limit, fast_limit / delta_phase))

    mama[i] <- alpha[i] * x[i]       + (1 - alpha[i]) * mama[i - 1]
    fama[i] <- 0.5 * alpha[i] * mama[i] + (1 - 0.5 * alpha[i]) * fama[i - 1]
  }

  list(mama = mama, fama = fama)
}

feat_mama <- function(x, fast_limit = 0.5, slow_limit = 0.05) {
  mf <- mama_fama(x, fast_limit, slow_limit)
  blank_warmup(data.frame(
    mama_spread = rms_normalize(x - mf$mama),
    mama_cross  = rms_normalize(mf$mama - mf$fama)
  ), RMS_PERIOD)
}

# Filtered volume + price-volume correlation. Output [-3, +3] / [-1, +1].
feat_volume <- function(close, volume, periods = GRID) {
  out <- list()
  for (p in periods) {
    filtered_price  <- roofing_filter(close, HP_PERIOD, p)
    filtered_volume <- roofing_filter(volume, HP_PERIOD, p)
    out[[paste0("vol_roof_", p)]]  <- rms_normalize(filtered_volume)
    out[[paste0("vol_pcorr_", p)]] <- rolling_correlation(filtered_price, filtered_volume, max(p, 30))
  }
  blank_warmup(as.data.frame(out), max(HP_PERIOD, GRID, RMS_PERIOD))
}

# Turnover-weighted EMA of VWAP (CYQ chip distribution).
# Time-varying alpha — cannot vectorize; requires loop.
avg_cost_basis <- function(amount, volume, turnover) {
  n        <- length(amount)
  vwap     <- amount / volume
  avg_cost <- numeric(n)
  avg_cost[1] <- vwap[1]
  for (i in 2:n) avg_cost[i] <- (1 - turnover[i]) * avg_cost[i - 1] + turnover[i] * vwap[i]
  first_full <- which(cumsum(turnover) >= 1)[1]
  if (is.na(first_full)) first_full <- n
  if (first_full > 1) avg_cost[1:(first_full - 1)] <- avg_cost[first_full]
  avg_cost
}

# Cost basis spread, slope, cross-scale. Output [-3, +3].
feat_cost <- function(close, avg_cost, periods = GRID) {
  out <- list(
    cost_spread = rms_normalize(close - avg_cost),
    cost_slope  = rms_normalize(c(0, diff(avg_cost)))
  )
  for (p in periods) {
    out[[paste0("cost_xscale_", p)]] <- rms_normalize(ultimate_smoother(close, p) - avg_cost)
  }
  blank_warmup(as.data.frame(out), max(periods, RMS_PERIOD))
}

# OHLC intrabar ratios, percentile-ranked. Output [-1, +1].
feat_ohlc <- function(open, high, low, close) {
  n     <- length(close)
  range <- high - low
  range[range == 0] <- 1e-8

  data.frame(
    ohlc_range      = range / close,
    ohlc_body_pos   = (close - low) / range,
    ohlc_upper_wick = (high - pmax(open, close)) / range,
    ohlc_lower_wick = (pmin(open, close) - low) / range,
    ohlc_gap        = c(0, open[-1] - close[-n]) / close
  ) %>%
    mutate(across(everything(), running_pctrank))
}

# Fundamental ratios: _dn (raw for cross-sectional norm) + _rn (temporal percentile rank).
# ROE excluded from _rn (pure accounting ratio, too sparse for temporal ranking).
feat_fundamental <- function(mc, np, np_deduct, equity, revenue, ocf) {
  raw <- data.frame(
    fund_ey_dn      = safe_div(np, mc),
    fund_eyd_dn     = safe_div(np_deduct, mc),
    fund_by_dn      = safe_div(equity, mc),
    fund_sy_dn      = safe_div(revenue, mc),
    fund_cfy_dn     = safe_div(ocf, mc),
    fund_roe_dn     = safe_div(np, equity),
    fund_accrual_dn = safe_div(np - ocf, mc)
  )

  temporal <- as.data.frame(lapply(select(raw, -fund_roe_dn), running_pctrank))
  names(temporal) <- sub("_dn$", "_rn", names(temporal))

  cbind(raw, temporal)
}

# Log market share: _dn (raw) + _rn (temporal percentile rank).
feat_market_share <- function(mc, mkt_mc) {
  raw  <- log(safe_div(mc, mkt_mc))
  data.frame(fund_lms_dn = raw, fund_lms_rn = running_pctrank(raw))
}

# ENSEMBLE =====================================================================

ehlers_features <- function(
    close, open = NULL, high = NULL, low = NULL,
    volume = NULL, amount = NULL, to = NULL,
    mc = NULL, mkt_mc = NULL,
    np = NULL, np_deduct = NULL, equity = NULL, revenue = NULL, ocf = NULL) {

  feats <- list(
    roofing  = feat_roofing(close),
    smoother = feat_smoother(close),
    sinewave = feat_sinewave(close),
    elegant  = feat_elegant(close),
    trend    = feat_trend(close),
    laguerre = feat_laguerre(close),
    mama     = feat_mama(close)
  )

  has_all <- function(...) !any(vapply(list(...), is.null, logical(1)))

  if (has_all(open, high, low))
    feats[["ohlc"]] <- feat_ohlc(open, high, low, close)

  if (!is.null(volume))
    feats[["volume"]] <- feat_volume(close, volume)

  if (has_all(amount, volume, to))
    feats[["cost"]] <- feat_cost(close, avg_cost_basis(amount, volume, to))

  if (has_all(mc, np, np_deduct, equity, revenue, ocf))
    feats[["fund"]] <- feat_fundamental(mc, np, np_deduct, equity, revenue, ocf)

  if (has_all(mc, mkt_mc)) {
    share <- feat_market_share(mc, mkt_mc)
    feats[["fund"]] <- if (is.null(feats[["fund"]])) share else cbind(feats[["fund"]], share)
  }

  result <- do.call(cbind, feats)
  mat    <- as.matrix(result)
  result[is.nan(mat) | is.infinite(mat)] <- NA_real_
  result
}

# VALIDATION ===================================================================

validate_features <- function(feats) {
  if (nrow(feats) > 1e4) {
    cat("Sampling 10k rows for validation...\n")
    feats <- slice_sample(feats, n = 1e4)
  }

  feat_cols <- select(feats, matches("^[a-z_]+\\."))
  mat       <- as.matrix(feat_cols)
  complete  <- mat[complete.cases(mat), ]

  cat("Valid rows:", nrow(complete), " Warmup NAs:", sum(is.na(mat)), "\n")
  cat("Bounds: [", round(min(complete), 4), ",", round(max(complete), 4), "]\n\n")

  cat(sprintf("%-35s %8s %8s %8s\n", "Feature", "NAs", "Min", "Max"))
  for (col in names(feat_cols)) {
    v <- feat_cols[[col]]
    cat(sprintf("%-35s %8d %8.4f %8.4f\n", col, sum(is.na(v)),
                min(v, na.rm = TRUE), max(v, na.rm = TRUE)))
  }

  groups <- sub("\\.[^.]+$", "", names(feat_cols))
  cat("\nGroups:\n")
  for (g in unique(groups)) cat(sprintf("  %-20s %3d\n", g, sum(groups == g)))
  cat(sprintf("  %-20s %3d\n", "TOTAL", ncol(feat_cols)))

  for (col in names(feat_cols)) hist(feat_cols[[col]], 30, xlab = col)
}

# DEMO =========================================================================

# # General workflow

# t0 <- proc.time()

# feats <- ehlers_features(...)

# elapsed <- (proc.time() - t0)[3]
# cat(nrow(feats), "x", ncol(feats), "in", round(elapsed, 3), "s\n")

# validate_features(feats)

# # For market features

# ehlers_features(
#   close     = mkt_data$close,
#   volume    = mkt_data$to,
#   mc        = mkt_data$mc,
#   np        = mkt_data$np,
#   np_deduct = mkt_data$np_deduct,
#   equity    = mkt_data$equity,
#   revenue   = mkt_data$revenue,
#   ocf       = mkt_data$ocf
# ) %>%
#   rename_with(~ paste0("mkt_", .x)) %>%
#   mutate(
#     date = mkt_data$date, .before = 1,
#     across(matches("_dn$"), ~ NULL)
#   ) %>%
#   filter(date >= train_start) %>%
#   na.omit() %>%
#   as_tibble()

# # For stock features

# ehlers_features(
#   close     = data$close,
#   open      = data$open,
#   high      = data$high,
#   low       = data$low,
#   volume    = data$volume,
#   amount    = data$amount,
#   to        = data$to,
#   mc        = data$mc,
#   mkt_mc    = left_join(data, mkt_data, by = "date")$mc.y,
#   np        = data$np,
#   np_deduct = data$np_deduct,
#   equity    = data$equity,
#   revenue   = data$revenue,
#   ocf       = data$ocf
# )
