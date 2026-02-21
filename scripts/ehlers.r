# ehlers_ensemble.r — Ehlers DSP feature ensemble for RL training.
#
# SuperSmoother where Ehlers specifies (EBSW, Elegant, Cybernetic, Roofing Filter).
# Ultimate Smoother where Ehlers specifies (USI, Laguerre L0, derived smoothers).
# Ehlers Apr 2024 shows US replacing SS in the roofing filter lowpass, but every
# subsequent indicator (incl. Cybernetic Osc Jun 2025) still explicitly uses SS.
#
# All outputs bounded: native indicators [-1, +1], sigma-normalized via clamp [-2.5, +2.5].
#
# Normalization naming convention:
#   run_*    — rolling statistical primitives (run_rms, run_cor, run_pctrank)
#   agc_*    — Ehlers AGC producing sigma-units, unbounded (agc_ema, agc_rolling)
#   norm_*   — final bounded output for RL (norm_ema, norm_rolling: [-2.5, +2.5]; norm_pctrank: [-1, +1])

HP_PERIOD <- 125
LP_PERIOD <- 20
GRID <- c(30, 60, 120)

# Helpers ----------------------------------------------------------------------

safe_div <- function(a, b) {
  ifelse(b != 0, a / b, 0)
}

clamp <- function(x, limit) {
  pmax(-limit, pmin(limit, x))
}

blank_warmup <- function(df, n_warmup) {
  df[seq_len(min(n_warmup, nrow(df))), ] <- NA_real_
  df
}

# Rolling Primitives -----------------------------------------------------------

# Expanding percentile rank. Output [0, 1]. O(n^2).
run_pctrank <- function(x) {
  n <- length(x)
  out <- numeric(n)
  for (i in seq_len(n)) {
    if (is.na(x[i])) {
      out[i] <- NA_real_
      next
    }
    past <- x[1:i]
    past <- past[!is.na(past)]
    out[i] <- if (length(past) <= 1) 0.5 else sum(past <= x[i]) / length(past)
  }
  out
}

# Rolling RMS over a fixed window.
run_rms <- function(x, window) {
  n <- length(x)
  if (n < window) return(rep(NA_real_, n))
  ms <- stats::filter(x^2, rep(1 / window, window), sides = 1)
  sqrt(ifelse(is.na(ms) | ms <= 0, NA_real_, ms))
}

# Rolling Pearson correlation via cumulative sums. O(n).
run_cor <- function(a, b, period) {
  n <- length(a)
  if (period > n) return(rep(NA_real_, n))

  cs_a <- c(0, cumsum(a))
  cs_b <- c(0, cumsum(b))
  cs_ab <- c(0, cumsum(a * b))
  cs_a2 <- c(0, cumsum(a^2))
  cs_b2 <- c(0, cumsum(b^2))

  hi <- (period + 1):(n + 1)
  lo <- hi - period
  sum_a <- cs_a[hi] - cs_a[lo]
  sum_b <- cs_b[hi] - cs_b[lo]
  sum_ab <- cs_ab[hi] - cs_ab[lo]
  sum_a2 <- cs_a2[hi] - cs_a2[lo]
  sum_b2 <- cs_b2[hi] - cs_b2[lo]

  denom <- (period * sum_a2 - sum_a^2) * (period * sum_b2 - sum_b^2)
  r <- ifelse(denom > 0, (period * sum_ab - sum_a * sum_b) / sqrt(denom), 0)
  r[is.na(r)] <- 0
  c(rep(NA_real_, period - 1), r)
}

# IIR Filters ------------------------------------------------------------------

butterworth_coefs <- function(period) {
  alpha <- sqrt(2) * pi / period
  a1 <- exp(-alpha)
  list(c2 = 2 * a1 * cos(alpha), c3 = -a1^2)
}

highpass_2pole <- function(x, period = HP_PERIOD) {
  n <- length(x)
  if (n < 3) return(rep(0, n))
  bw <- butterworth_coefs(period)
  c1 <- (1 + bw$c2 - bw$c3) / 4
  feed <- stats::filter(x, c(c1, -2 * c1, c1), sides = 1)
  feed[is.na(feed)] <- 0
  as.numeric(stats::filter(feed, c(bw$c2, bw$c3), method = "recursive", init = c(0, 0)))
}

highpass_1pole <- function(x, period) {
  n <- length(x)
  if (n < 2) return(rep(0, n))
  a1 <- (1 - sin(2 * pi / period)) / cos(2 * pi / period)
  feed <- 0.5 * (1 + a1) * c(0, diff(x))
  as.numeric(stats::filter(feed, a1, method = "recursive", init = 0))
}

ultimate_smoother <- function(x, period = LP_PERIOD) {
  n <- length(x)
  if (n < 3) return(x)
  bw <- butterworth_coefs(period)
  c1 <- (1 + bw$c2 - bw$c3) / 4
  feed <- stats::filter(x, c(1 - c1, 2 * c1 - bw$c2, -(c1 + bw$c3)), sides = 1)
  feed[is.na(feed)] <- 0
  as.numeric(stats::filter(feed, c(bw$c2, bw$c3), method = "recursive", init = c(0, 0)))
}

super_smoother <- function(x, period = LP_PERIOD) {
  n <- length(x)
  if (n < 2) return(x)
  bw <- butterworth_coefs(period)
  c1 <- 1 - bw$c2 - bw$c3
  avg <- stats::filter(x, c(0.5, 0.5), sides = 1)
  avg[is.na(avg)] <- x[is.na(avg)]
  as.numeric(stats::filter(c1 * avg, c(bw$c2, bw$c3), method = "recursive", init = c(0, 0)))
}

roofing_filter <- function(x, hp_period = HP_PERIOD, lp_period = LP_PERIOD) {
  super_smoother(highpass_2pole(x, hp_period), lp_period)
}

laguerre_stages <- function(x, gamma, period) {
  n <- length(x)
  L0 <- ultimate_smoother(x, period)
  cascade <- function(prev) {
    feed <- -gamma * prev + c(0, prev[-n])
    feed[1] <- 0
    as.numeric(stats::filter(feed, gamma, method = "recursive", init = 0))
  }
  L1 <- cascade(L0)
  L2 <- cascade(L1)
  L3 <- cascade(L2)
  L4 <- cascade(L3)
  list(
    filter = (L0 + 4 * L1 + 6 * L2 + 4 * L3 + L4) / 16,
    L0 = L0,
    L1 = L1
  )
}

# Normalization ----------------------------------------------------------------

# EMA-based AGC (Ehlers, alpha ~ 82-bar EMA). Output in sigma-units, unbounded.
agc_ema <- function(x, alpha = 0.0242) {
  ms <- as.numeric(stats::filter(alpha * x^2, 1 - alpha, method = "recursive", init = 0))
  ifelse(ms > 0, x / sqrt(ms), 0)
}

# Rolling-window AGC. Output in sigma-units, unbounded.
agc_rolling <- function(x, window = 100) {
  rms_val <- run_rms(x, window)
  ifelse(!is.na(rms_val) & rms_val > 0, x / rms_val, 0)
}

# Bounded EMA AGC: sigma-units / 2 -> clamp -> [-2.5, +2.5].
norm_ema <- function(x, alpha = 0.0242) {
  clamp(agc_ema(x, alpha) / 2, 2.5)
}

# Bounded rolling AGC: sigma-units / 2 -> clamp -> [-2.5, +2.5].
norm_rolling <- function(x, window = 100) {
  clamp(agc_rolling(x, window) / 2, 2.5)
}

# Expanding percentile rank rescaled to [-1, +1].
norm_pctrank <- function(x) {
  2 * run_pctrank(x) - 1
}

# Ehlers Indicators ------------------------------------------------------------

# Correlation Trend Indicator (TASC May 2020). Inherently [-1, +1].
correlation_trend <- function(x, period) {
  n <- length(x)
  if (n < period) return(rep(NA_real_, n))

  ramp <- seq_len(period)
  sum_ramp <- sum(ramp)
  ss_ramp <- sum(ramp^2)

  cs_x <- c(0, cumsum(x))
  cs_x2 <- c(0, cumsum(x^2))
  cs_jx <- c(0, cumsum(seq_len(n) * x))

  hi <- period:n
  lo <- hi - period + 1
  sum_x <- cs_x[hi + 1] - cs_x[lo]
  sum_x2 <- cs_x2[hi + 1] - cs_x2[lo]
  sum_xy <- (cs_jx[hi + 1] - cs_jx[lo]) - (lo - 1) * sum_x

  denom <- sqrt((period * sum_x2 - sum_x^2) * (period * ss_ramp - sum_ramp^2))
  r <- ifelse(denom > 0, (period * sum_xy - sum_x * sum_ramp) / denom, 0)
  c(rep(NA_real_, period - 1), r)
}

# Correlation Cycle Indicator (TASC Jun 2020).
# Returns pwr [0, 1] and angle [-90, +90] degrees.
correlation_cycle <- function(x, period) {
  n <- length(x)
  if (n < period) return(list(pwr = rep(NA_real_, n), angle = rep(NA_real_, n)))

  cos_ref <- cos(2 * pi * seq_len(period) / period)
  sin_ref <- sin(2 * pi * seq_len(period) / period)
  sum_cos <- sum(cos_ref)
  sum_sin <- sum(sin_ref)
  var_cos <- period * sum(cos_ref^2) - sum_cos^2
  var_sin <- period * sum(sin_ref^2) - sum_sin^2

  xcr_cos <- as.numeric(stats::filter(x, rev(cos_ref), sides = 1))
  xcr_sin <- as.numeric(stats::filter(x, rev(sin_ref), sides = 1))

  cs_x <- c(0, cumsum(x))
  cs_x2 <- c(0, cumsum(x^2))
  hi <- period:n
  lo <- hi - period + 1
  sum_x <- cs_x[hi + 1] - cs_x[lo]
  sum_x2 <- cs_x2[hi + 1] - cs_x2[lo]
  var_x <- period * sum_x2 - sum_x^2

  r_cos <- ifelse(var_x * var_cos > 0,
                  (period * xcr_cos[hi] - sum_x * sum_cos) / sqrt(var_x * var_cos), 0)
  r_sin <- ifelse(var_x * var_sin > 0,
                  (period * xcr_sin[hi] - sum_x * sum_sin) / sqrt(var_x * var_sin), 0)

  pad <- rep(NA_real_, period - 1)
  list(
    pwr = c(pad, pmin(sqrt(r_cos^2 + r_sin^2), 1)),
    angle = c(pad, ifelse(r_cos != 0, atan(r_sin / r_cos) * 180 / pi, 90 * sign(r_sin)))
  )
}

# Autocorrelation Periodogram (Cycle Analytics 2013, Ch. 8).
# Returns dominant cycle period in bars.
# Vectorized: autocorrelation via run_cor per lag, DFT via matrix multiply.
autocorrelation_periodogram <- function(x, min_period = 10, max_period = 48,
                                        hp_period = 48, lp_period = 10) {
  n <- length(x)
  filt <- roofing_filter(x, hp_period, lp_period)
  dom_cycle <- rep(NA_real_, n)
  start_bar <- 2 * max_period
  if (start_bar > n) return(dom_cycle)

  periods <- min_period:max_period
  lag_vals <- 0:max_period
  n_lags <- length(lag_vals)
  idx <- start_bar:n
  n_idx <- length(idx)

  # DFT basis matrices (lags x periods)
  cos_basis <- outer(lag_vals, periods, function(l, p) cos(2 * pi * l / p))
  sin_basis <- outer(lag_vals, periods, function(l, p) sin(2 * pi * l / p))

  # Autocorrelation matrix (bars x lags) via run_cor
  corr_mat <- matrix(0, nrow = n_idx, ncol = n_lags)
  corr_mat[, 1] <- 1  # lag 0: self-correlation

  for (k in 2:n_lags) {
    lag_val <- lag_vals[k]
    window <- max(lag_val, 3)
    shifted <- c(rep(0, lag_val), filt[1:(n - lag_val)])
    r <- run_cor(filt, shifted, window)
    corr_mat[, k] <- r[idx]
  }
  corr_mat[is.na(corr_mat)] <- 0

  # DFT: autocorrelation -> power at each candidate period (matrix multiply)
  cos_part <- corr_mat %*% cos_basis
  sin_part <- corr_mat %*% sin_basis
  pwr_mat <- cos_part^2 + sin_part^2

  # Normalize per bar and extract dominant cycle via center of gravity
  max_pwr <- do.call(pmax, c(as.data.frame(pwr_mat), 1e-10))
  pwr_norm <- pwr_mat / max_pwr
  sq_pwr <- pwr_norm^2
  masked <- sq_pwr * (sq_pwr > 0.25)
  total <- rowSums(masked)
  weighted <- as.numeric(masked %*% periods)
  dom_cycle[idx] <- ifelse(total > 0, weighted / total, (min_period + max_period) / 2)

  # EMA smooth (alpha = 0.2) per Ehlers
  dc_valid <- dom_cycle[idx]
  dc_valid <- as.numeric(stats::filter(0.2 * dc_valid, 0.8, method = "recursive",
                                       init = dc_valid[1]))
  dom_cycle[idx] <- dc_valid
  dom_cycle
}

# MAMA/FAMA (Rocket Science 2001, Ch. 8). Input: (H+L)/2.
mama_fama <- function(x, fast_limit = 0.5, slow_limit = 0.05) {
  n <- length(x)

  smooth <- as.numeric(stats::filter(x, c(4, 3, 2, 1) / 10, sides = 1))
  smooth[is.na(smooth)] <- x[is.na(smooth)]

  ht_coefs <- c(0.0962, 0.5769, -0.5769, -0.0962)
  hilbert_fir <- function(sig, i) {
    sig[i] * ht_coefs[1] + sig[i - 2] * ht_coefs[2] +
      sig[i - 4] * ht_coefs[3] + sig[i - 6] * ht_coefs[4]
  }

  detrender <- quad <- inphase <- j_inphase <- j_quad <- numeric(n)
  inphase2 <- quad2 <- re <- im <- numeric(n)
  period <- rep(20, n)
  phase <- alpha <- numeric(n)
  mama <- fama <- x

  if (n < 7) return(list(mama = mama, fama = fama))

  for (i in 7:n) {
    adj <- 0.075 * period[i - 1] + 0.54

    detrender[i] <- hilbert_fir(smooth, i) * adj
    quad[i] <- hilbert_fir(detrender, i) * adj
    inphase[i] <- detrender[i - 3]
    j_inphase[i] <- hilbert_fir(inphase, i) * adj
    j_quad[i] <- hilbert_fir(quad, i) * adj

    inphase2[i] <- 0.2 * (inphase[i] - j_quad[i]) + 0.8 * inphase2[i - 1]
    quad2[i] <- 0.2 * (quad[i] + j_inphase[i]) + 0.8 * quad2[i - 1]
    re[i] <- 0.2 * (inphase2[i] * inphase2[i - 1] + quad2[i] * quad2[i - 1]) + 0.8 * re[i - 1]
    im[i] <- 0.2 * (inphase2[i] * quad2[i - 1] - quad2[i] * inphase2[i - 1]) + 0.8 * im[i - 1]

    period[i] <- if (im[i] != 0 && re[i] != 0) 2 * pi / atan(im[i] / re[i]) else period[i - 1]
    period[i] <- max(6, min(50, max(0.67 * period[i - 1], min(1.5 * period[i - 1], period[i]))))
    period[i] <- 0.2 * period[i] + 0.8 * period[i - 1]

    phase[i] <- if (inphase[i] != 0) atan(quad[i] / inphase[i]) * 180 / pi else phase[i - 1]
    delta_phase <- max(1, phase[i - 1] - phase[i])
    alpha[i] <- max(slow_limit, min(fast_limit, fast_limit / delta_phase))

    mama[i] <- alpha[i] * x[i] + (1 - alpha[i]) * mama[i - 1]
    fama[i] <- 0.5 * alpha[i] * mama[i] + (1 - 0.5 * alpha[i]) * fama[i - 1]
  }
  list(mama = mama, fama = fama)
}

# Ehlers Feature Generators ----------------------------------------------------

# Cybernetic Oscillator (TASC Jun 2025) at grid periods.
# Octave bandpass: HP(2p) -> SS(p) -> rolling AGC -> clamp. Output [-2.5, +2.5].
feat_roofing <- function(x, periods = GRID) {
  out <- lapply(periods, function(p) {
    norm_rolling(roofing_filter(x, 2 * p, p), 100)
  })
  names(out) <- paste0("roof_", periods)
  blank_warmup(as.data.frame(out), max(2 * periods))
}

# Even Better Sinewave (Cycle Analytics 2013, Ch. 12) at grid periods.
# HP1(duration) -> SS(10) -> 3-bar wave/power -> normalize. Output [-1, +1].
feat_sinewave <- function(x, periods = GRID) {
  n <- length(x)
  out <- lapply(periods, function(p) {
    if (n < 3) return(rep(NA_real_, n))
    filtered <- super_smoother(highpass_1pole(x, p), 10)
    wave <- stats::filter(filtered, rep(1 / 3, 3), sides = 1)
    pwr <- stats::filter(filtered^2, rep(1 / 3, 3), sides = 1)
    wave[is.na(wave)] <- 0
    pwr[is.na(pwr)] <- 0
    clamp(ifelse(pwr > 0, as.numeric(wave) / sqrt(as.numeric(pwr)), 0), 1)
  })
  names(out) <- paste0("ebsw_", periods)
  blank_warmup(as.data.frame(out), max(periods))
}

# Elegant Oscillator (TASC Feb 2022). Single period per Ehlers' BandEdge = 20.
# Deriv(2-bar) -> SS(20) -> rolling AGC -> clamp. Output [-2.5, +2.5].
feat_elegant <- function(x, period = 20, rms_window = 50) {
  n <- length(x)
  if (n < 3) return(blank_warmup(data.frame(eleg = rep(NA_real_, n)), n))
  deriv <- c(0, 0, x[3:n] - x[1:(n - 2)])
  eleg <- norm_rolling(super_smoother(deriv, period), rms_window)
  blank_warmup(data.frame(eleg = eleg), max(rms_window, period))
}

# Correlation Trend Indicator (TASC May 2020) at grid periods.
# Inherently [-1, +1].
feat_trend <- function(x, periods = GRID) {
  out <- lapply(periods, function(p) correlation_trend(x, p))
  names(out) <- paste0("cti_", periods)
  blank_warmup(as.data.frame(out), max(periods))
}

# Laguerre Oscillator (TASC Jul 2025) at multiple gammas.
# US(L0) internally per Ehlers. Period = 30 per Ehlers default.
# Rolling AGC -> clamp. Output [-2.5, +2.5].
feat_laguerre <- function(x, gammas = c(0.3, 0.6, 0.8), period = 30) {
  out <- lapply(gammas, function(g) {
    lag <- laguerre_stages(x, g, period)
    norm_rolling(lag$L0 - lag$L1, 100)
  })
  names(out) <- paste0("lag_osc_", sub("\\.", "", gammas))
  blank_warmup(as.data.frame(out), period)
}

# MAMA spread and crossover (Rocket Science 2001, Ch. 8).
# Input: (H+L)/2. Self-adaptive — no period grid.
# EMA AGC -> clamp. Output [-2.5, +2.5].
feat_mama <- function(hl2, fast_limit = 0.5, slow_limit = 0.05) {
  mf <- mama_fama(hl2, fast_limit, slow_limit)
  blank_warmup(data.frame(
    mama_spread = norm_ema(hl2 - mf$mama),
    mama_cross = norm_ema(mf$mama - mf$fama)
  ), 50)
}

# Ultimate Strength Index (TASC Nov 2024) at grid periods.
# Inherently [-1, +1].
feat_usi <- function(x, periods = GRID) {
  changes <- c(0, diff(x))
  ups <- pmax(changes, 0)
  downs <- pmax(-changes, 0)
  out <- lapply(periods, function(p) {
    us_ups <- pmax(ultimate_smoother(ups, p), 0)
    us_downs <- pmax(ultimate_smoother(downs, p), 0)
    safe_div(us_ups - us_downs, us_ups + us_downs)
  })
  names(out) <- paste0("usi_", periods)
  blank_warmup(as.data.frame(out), max(periods))
}

# Correlation Cycle Indicator (TASC Jun 2020) at grid periods.
# pwr: [0,1] -> [-1,+1]. angle: [-90,+90]/90 -> [-1,+1].
feat_cycle <- function(x, periods = GRID) {
  out <- list()
  for (p in periods) {
    cc <- correlation_cycle(x, p)
    out[[paste0("cci_pwr_", p)]] <- 2 * cc$pwr - 1
    out[[paste0("cci_angle_", p)]] <- cc$angle / 90
  }
  blank_warmup(as.data.frame(out), max(periods))
}

# Dominant Cycle period (Cycle Analytics 2013, Ch. 8).
# Single estimate, rescaled to [-1, +1].
feat_dominant_cycle <- function(x, min_period = 10, max_period = 48) {
  dc <- autocorrelation_periodogram(x, min_period, max_period)
  dc_norm <- 2 * (dc - min_period) / (max_period - min_period) - 1
  blank_warmup(data.frame(dc_period = dc_norm), 2 * max_period)
}

# Derived Feature Generators ---------------------------------------------------

# Ultimate Smoother spread/slope/cross-scale at grid periods.
# EMA AGC -> clamp. Output [-2.5, +2.5].
feat_smoother <- function(x, periods = GRID) {
  all_periods <- sort(unique(c(periods, 2 * periods)))
  smoothed <- setNames(lapply(all_periods, function(p) ultimate_smoother(x, p)), all_periods)
  out <- list()
  for (p in periods) {
    key <- as.character(p)
    key2 <- as.character(2 * p)
    out[[paste0("us_spread_", p)]] <- norm_ema(x - smoothed[[key]])
    out[[paste0("us_slope_", p)]] <- norm_ema(c(0, diff(smoothed[[key]])))
    out[[paste0("us_xscale_", p)]] <- norm_ema(smoothed[[key]] - smoothed[[key2]])
  }
  blank_warmup(as.data.frame(out), max(2 * periods))
}

# Volume: octave bandpass EMA AGC -> clamp [-2.5, +2.5]; price-volume correlation [-1, +1].
feat_volume <- function(close, volume, periods = GRID) {
  out <- list()
  for (p in periods) {
    filt_price <- roofing_filter(close, 2 * p, p)
    filt_vol <- roofing_filter(volume, 2 * p, p)
    out[[paste0("vol_roof_", p)]] <- norm_ema(filt_vol)
    out[[paste0("vol_pcor_", p)]] <- run_cor(filt_price, filt_vol, max(p, 30))
  }
  blank_warmup(as.data.frame(out), max(2 * periods))
}

# Non-Ehlers Feature Generators ------------------------------------------------

# OHLC microstructure. All outputs [-1, +1].
# OHLC microstructure. TR-normalized, all outputs [-1, +1].
feat_ohlc <- function(open, high, low, close) {
  n <- length(close)
  prev_close <- c(open[1], close[-n])
  tr <- pmax(high - low, abs(high - prev_close), abs(low - prev_close))
  data.frame(
    ohlc_range = norm_pctrank(tr / close),
    ohlc_gap = safe_div(open - prev_close, tr),
    ohlc_body_pos = 2 * ifelse(tr != 0, (close - low) / tr, 0.5) - 1,
    ohlc_upper_wick = 2 * safe_div(high - pmax(open, close), tr) - 1,
    ohlc_lower_wick = 2 * safe_div(pmin(open, close) - low, tr) - 1
  )
}

# Turnover-weighted average cost (CYQ chip distribution).
# Uses VWAP (amount/volume) when available, falls back to close.
avg_cost_basis <- function(close, turnover, amount = NULL, volume = NULL) {
  n <- length(close)
  vwap <- if (!is.null(amount) && !is.null(volume)) amount / volume else close
  avg_cost <- numeric(n)
  avg_cost[1] <- vwap[1]
  for (i in 2:n) {
    avg_cost[i] <- (1 - turnover[i]) * avg_cost[i - 1] + turnover[i] * vwap[i]
  }
  avg_cost
}

# Cost basis spread/slope at single scale, cross-scale at grid periods.
# EMA AGC -> clamp. Output [-2.5, +2.5].
feat_cost <- function(close, avg_cost, periods = GRID) {
  out <- list(
    cost_spread = norm_ema(close - avg_cost),
    cost_slope = norm_ema(c(0, diff(avg_cost)))
  )
  for (p in periods) {
    out[[paste0("cost_xscale_", p)]] <- norm_ema(ultimate_smoother(close, p) - avg_cost)
  }
  blank_warmup(as.data.frame(out), max(periods))
}

# Fundamentals: _dn raw for cross-sectional norm; _rn pctrank [-1, +1].
# mkt_mc is optional — when NULL, market share columns are omitted.
feat_fundamental <- function(mc, np, np_deduct, equity, revenue, ocf, mkt_mc = NULL) {
  raw <- data.frame(
    fund_ey_dn = safe_div(np, mc),
    fund_eyd_dn = safe_div(np_deduct, mc),
    fund_by_dn = safe_div(equity, mc),
    fund_sy_dn = safe_div(revenue, mc),
    fund_cfy_dn = safe_div(ocf, mc),
    fund_roe_dn = safe_div(np, equity),
    fund_accrual_dn = safe_div(np - ocf, mc)
  )
  if (!is.null(mkt_mc)) {
    raw <- cbind(data.frame(fund_ms_dn = safe_div(mc, mkt_mc)), raw)
  }
  ranked <- as.data.frame(lapply(raw, norm_pctrank))
  names(ranked) <- sub("_dn$", "_rn", names(ranked))
  cbind(raw, ranked)
}

# Ensemble ---------------------------------------------------------------------

ehlers_features <- function(
    close, open = NULL, high = NULL, low = NULL,
    volume = NULL, amount = NULL, to = NULL,
    mc = NULL, mkt_mc = NULL,
    np = NULL, np_deduct = NULL, equity = NULL, revenue = NULL, ocf = NULL) {

  hl2 <- if (!is.null(high) && !is.null(low)) (high + low) / 2 else close
  has_all <- function(...) !any(vapply(list(...), is.null, logical(1)))

  feats <- list(
    roofing = feat_roofing(close),
    smoother = feat_smoother(close),
    sinewave = feat_sinewave(close),
    elegant = feat_elegant(close),
    trend = feat_trend(close),
    laguerre = feat_laguerre(close),
    mama = feat_mama(hl2),
    usi = feat_usi(close),
    cycle = feat_cycle(close),
    dc = feat_dominant_cycle(close)
  )

  if (has_all(open, high, low))
    feats$ohlc <- feat_ohlc(open, high, low, close)

  if (!is.null(volume))
    feats$volume <- feat_volume(close, volume)

  if (!is.null(to))
    feats$cost <- feat_cost(close, avg_cost_basis(close, to, amount, volume))

  if (has_all(mc, np, np_deduct, equity, revenue, ocf))
    feats$fund <- feat_fundamental(mc, np, np_deduct, equity, revenue, ocf, mkt_mc)

  # Prefix column names with group name to ensure group.feature pattern
  # (cbind with named list does not reliably prefix single-column data.frames)
  for (group in names(feats)) {
    names(feats[[group]]) <- paste0(group, ".", names(feats[[group]]))
  }

  result <- do.call(cbind, unname(feats))
  mat <- as.matrix(result)
  result[is.nan(mat) | is.infinite(mat)] <- NA_real_
  result
}

# Validation -------------------------------------------------------------------

validate_features <- function(feats, sample_n = 10000, plot = FALSE) {
  feats <- as.data.frame(feats)

  if (nrow(feats) > sample_n) {
    cat("Sampling", sample_n, "rows for validation...\n")
    feats <- feats[sample.int(nrow(feats), sample_n), ]
  }

  feat_cols <- feats[, grep("^[a-z_]+\\.", names(feats)), drop = FALSE]
  if (ncol(feat_cols) == 0) {
    cat("No feature columns found (expected group.name pattern).\n")
    return(invisible(NULL))
  }

  mat <- as.matrix(feat_cols)
  n_complete <- sum(complete.cases(mat))
  cat("Rows:", nrow(mat), " Complete:", n_complete,
      " Warmup NAs:", sum(is.na(mat)), "\n")

  bounded_cols <- grep("_dn$", names(feat_cols), invert = TRUE)
  if (length(bounded_cols) > 0) {
    bmat <- mat[, bounded_cols, drop = FALSE]
    bmat <- bmat[complete.cases(bmat), , drop = FALSE]
    cat("Bounded range: [", round(min(bmat), 4), ",", round(max(bmat), 4), "]\n")
  }

  cat(sprintf("\n%-35s %8s %8s %8s %8s %8s\n", "Feature", "NAs", "Min", "Mean", "Max", "SD"))
  cat(strrep("-", 80), "\n")

  for (col in names(feat_cols)) {
    v <- feat_cols[[col]]
    na_count <- sum(is.na(v))
    v_clean <- v[!is.na(v)]
    if (length(v_clean) == 0) {
      cat(sprintf("%-35s %8d %8s %8s %8s %8s\n", col, na_count, "NA", "NA", "NA", "NA"))
      next
    }
    vmin <- min(v_clean)
    vmean <- mean(v_clean)
    vmax <- max(v_clean)
    vsd <- sd(v_clean)
    cat(sprintf("%-35s %8d %8.3f %8.3f %8.3f %8.3f", col, na_count, vmin, vmean, vmax, vsd))
    cat("\n")
  }

  groups <- sub("\\.[^.]+$", "", names(feat_cols))
  cat("\nGroups:\n")
  for (g in unique(groups)) {
    cat(sprintf("  %-25s %3d\n", g, sum(groups == g)))
  }
  cat(sprintf("  %-25s %3d\n", "TOTAL", ncol(feat_cols)))

  if (plot) {
    for (col in names(feat_cols)) {
      v <- feat_cols[[col]]
      v <- v[!is.na(v)]
      if (length(v) > 0) hist(v, breaks = 30, main = col, xlab = col)
    }
  }

  invisible(NULL)
}
