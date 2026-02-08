# ehlers_ensemble.R
# Ehlers DSP feature ensemble for RL training.
# All filters built on the Ultimate Smoother (2-pole Butterworth, zero-lag passband).
# All outputs bounded: [-3,+3] RMS-normalized, [-1,+1] oscillators, raw fundamentals.
# Bars within warmup period are set to NA_real_ (IIR transient not yet settled).
# Fundamental features are raw ratios — normalize cross-sectionally with
# bestNormalize::orderNorm() across all stocks per date before feeding to RL.

HP_PERIOD  <- 120
LP_PERIOD  <- 20
RMS_PERIOD <- 60
GRID       <- c(30, 60, 120)


# FILTERS ======================================================================

# Shared 2-pole Butterworth coefficients.
# Ehlers: a1 = exp(-sqrt(2)*PI/Period), c2 = 2*a1*cos(sqrt(2)*PI/Period)
.bw_coefs <- function(period) {
  alpha <- sqrt(2) * pi / period
  a1    <- exp(-alpha)
  list(c2 = 2 * a1 * cos(alpha), c3 = -a1^2)
}

# 2-pole Butterworth highpass.
# Ehlers: HP[i] = c1*(x[i] - 2*x[i-1] + x[i-2]) + c2*HP[i-1] + c3*HP[i-2]
highpass <- function(x, period = HP_PERIOD) {
  bw <- .bw_coefs(period)
  c1 <- (1 + bw$c2 - bw$c3) / 4
  u  <- stats::filter(x, filter = c(c1, -2*c1, c1), method = "convolution", sides = 1)
  u[is.na(u)] <- 0
  as.numeric(stats::filter(u, filter = c(bw$c2, bw$c3), method = "recursive", init = c(0, 0)))
}

# 1-pole highpass.
# Ehlers: a1 = (1 - sin(2*PI/P)) / cos(2*PI/P)
#         HP[i] = 0.5*(1+a1)*(x[i] - x[i-1]) + a1*HP[i-1]
highpass1 <- function(x, period = HP_PERIOD) {
  a1 <- (1 - sin(2 * pi / period)) / cos(2 * pi / period)
  dx <- 0.5 * (1 + a1) * c(0, diff(x))
  as.numeric(stats::filter(dx, filter = a1, method = "recursive", init = 0))
}

# Ultimate Smoother (zero-lag passband, 2-pole Butterworth).
# Ehlers: US[i] = (1-c1)*x[i] + (2*c1-c2)*x[i-1] - (c1+c3)*x[i-2]
#                 + c2*US[i-1] + c3*US[i-2]
# LTI decomposition: FIR input side → IIR feedback via stats::filter.
ultimate_smoother <- function(x, period = LP_PERIOD) {
  bw <- .bw_coefs(period)
  c1 <- (1 + bw$c2 - bw$c3) / 4
  u  <- stats::filter(x, filter = c(1 - c1, 2*c1 - bw$c2, -(c1 + bw$c3)),
                      method = "convolution", sides = 1)
  u[is.na(u)] <- 0
  as.numeric(stats::filter(u, filter = c(bw$c2, bw$c3), method = "recursive", init = c(0, 0)))
}

# Roofing filter: 2-pole HP → Ultimate Smoother bandpass.
roofing <- function(x, hp_period = HP_PERIOD, lp_period = LP_PERIOD) {
  ultimate_smoother(highpass(x, hp_period), lp_period)
}

# Laguerre 4-stage IIR filter. Gamma controls lag/smoothness tradeoff.
# Ehlers: L0[i] = (1-g)*x[i] + g*L0[i-1]
#         Lk[i] = -g*L_{k-1}[i] + L_{k-1}[i-1] + g*Lk[i-1]   (k = 1,2,3)
#         Output = (L0 + 2*L1 + 2*L2 + L3) / 6
# Each stage is a 1st-order IIR with fixed gamma → stats::filter.
laguerre <- function(x, gamma) {
  n <- length(x)
  g <- gamma

  # Stage 0: L0[i] = (1-g)*x[i] + g*L0[i-1], L0[1] = x[1]
  # stats::filter: y[i] = input[i] + g*y[i-1]; init=x[1] gives y[1]=(1-g)*x[1]+g*x[1]=x[1]
  L0 <- as.numeric(stats::filter((1 - g) * x, filter = g, method = "recursive", init = x[1]))

  # Stages 1-3: Lk[i] = (-g*L_{k-1}[i] + L_{k-1}[i-1]) + g*Lk[i-1], Lk[1] = 0
  laguerre_stage <- function(prev) {
    u    <- -g * prev + c(0, prev[-n])
    u[1] <- 0
    as.numeric(stats::filter(u, filter = g, method = "recursive", init = 0))
  }

  L1 <- laguerre_stage(L0)
  L2 <- laguerre_stage(L1)
  L3 <- laguerre_stage(L2)
  (L0 + 2*L1 + 2*L2 + L3) / 6
}


# NORMALIZATION ================================================================

# Exponential RMS AGC. Output clamped to [-limit, +limit].
rms_normalize <- function(x, period = RMS_PERIOD, limit = 3) {
  alpha <- 2 / (period + 1)
  ms  <- as.numeric(stats::filter(alpha * x^2, filter = 1 - alpha,
                                   method = "recursive", init = 0))
  clamp(ifelse(ms > 0, x / sqrt(ms), 0), limit)
}

clamp <- function(x, limit = 3) pmax(-limit, pmin(limit, x))

# Set rows 1:warmup to NA_real_. Applied at end of each feat_* function.
.set_warmup <- function(out, warmup) {
  if (warmup > 0 && warmup < nrow(out)) out[1:warmup, ] <- NA_real_
  out
}

# Vectorized rolling Pearson correlation via cumulative sums. O(n).
rolling_cor <- function(a, b, win) {
  n <- length(a)
  if (win > n) return(rep(0, n))

  cum_a  <- c(0, cumsum(a))
  cum_b  <- c(0, cumsum(b))
  cum_ab <- c(0, cumsum(a * b))
  cum_a2 <- c(0, cumsum(a^2))
  cum_b2 <- c(0, cumsum(b^2))

  idx <- (win + 1):(n + 1)
  j   <- idx - win
  sa  <- cum_a[idx]  - cum_a[j]
  sb  <- cum_b[idx]  - cum_b[j]
  sab <- cum_ab[idx] - cum_ab[j]
  sa2 <- cum_a2[idx] - cum_a2[j]
  sb2 <- cum_b2[idx] - cum_b2[j]

  den <- (win * sa2 - sa^2) * (win * sb2 - sb^2)
  r   <- ifelse(den > 0, (win * sab - sa * sb) / sqrt(den), 0)
  r[is.na(r)] <- 0
  c(rep(0, win - 1), r)
}


# REGIME DETECTION =============================================================

# Correlation Trend Indicator: Pearson correlation of price vs linear ramp.
# Vectorized with cumulative sums. Output [-1, +1].
cti <- function(x, period) {
  n <- length(x)
  if (n < period) return(rep(0, n))

  y   <- seq_len(period)
  sy  <- sum(y)
  syy <- sum(y^2)

  cum_x  <- c(0, cumsum(x))
  cum_x2 <- c(0, cumsum(x^2))
  cum_jx <- c(0, cumsum(seq_len(n) * x))

  bars <- period:n
  a    <- bars - period + 1
  sx   <- cum_x[bars + 1]  - cum_x[a]
  sx2  <- cum_x2[bars + 1] - cum_x2[a]
  sxy  <- (cum_jx[bars + 1] - cum_jx[a]) - (a - 1) * sx

  den <- sqrt((period * sx2 - sx^2) * (period * syy - sy^2))
  r   <- ifelse(den > 0, (period * sxy - sx * sy) / den, 0)
  c(rep(0, period - 1), r)
}


# FEATURE GENERATORS ===========================================================

# Roofing + RMS at grid periods. Output [-3, +3].
# Warmup: max of HP, LP periods, and RMS settling.
feat_roofing <- function(x, periods = GRID) {
  out <- list()
  for (p in periods) out[[paste0("roof_", p)]] <- rms_normalize(roofing(x, HP_PERIOD, p))
  .set_warmup(as.data.frame(out), max(c(HP_PERIOD, periods, RMS_PERIOD)))
}

# Ultimate Smoother spread, slope, cross-scale. Output [-3, +3].
# Warmup: max of US periods and RMS settling.
feat_ultimate <- function(x, periods = GRID) {
  us_list <- list()
  out     <- list()
  for (p in periods) {
    us <- ultimate_smoother(x, p)
    us_list[[as.character(p)]] <- us
    out[[paste0("us_spread_", p)]] <- rms_normalize(x - us)
    out[[paste0("us_slope_", p)]]  <- rms_normalize(c(0, diff(us)))
  }
  for (i in 1:(length(periods) - 1))
    out[[paste0("us_xscale_", periods[i], "_", periods[i+1])]] <-
      rms_normalize(us_list[[as.character(periods[i])]] -
                    us_list[[as.character(periods[i+1])]])
  .set_warmup(as.data.frame(out), max(c(periods, RMS_PERIOD)))
}

# Even Better Sinewave. Matched bandpass: same period for HP1 and US.
# Ehlers: HP1 → UltimateSmoother → AGC(0.95/0.05).
# AGC via stats::filter: pwr is EMA of filt^2 with alpha=0.05.
# Output [-1, +1].
even_better_sinewave <- function(x, period) {
  filt <- ultimate_smoother(highpass1(x, period), period)
  pwr  <- as.numeric(stats::filter(0.05 * filt^2, filter = 0.95,
                                    method = "recursive", init = 0))
  wave <- ifelse(pwr > 0, filt / sqrt(pwr), 0)
  pmax(-1, pmin(1, wave))
}

# Warmup: max of matched bandpass periods (HP1+US both at p).
feat_ebsw <- function(x, periods = GRID) {
  out <- list()
  for (p in periods) out[[paste0("ebsw_", p)]] <- even_better_sinewave(x, p)
  .set_warmup(as.data.frame(out), max(periods))
}

# Elegant Oscillator: 2-bar deriv → RMS → IFT(tanh) → UltimateSmoother.
# Ehlers: Deriv = Close - Close[2]; nDeriv = Deriv / RMS(Deriv);
#         Elegant = UltimateSmoother(tanh(nDeriv)).
# Output [-1, +1].
# Warmup: max of US periods and RMS rolling window.
feat_elegant <- function(x, periods = GRID) {
  n     <- length(x)
  deriv <- c(0, 0, x[3:n] - x[1:(n-2)])

  # Vectorized rolling RMS via cumulative sums
  cum_sq    <- c(0, cumsum(deriv^2))
  idx       <- seq_len(n)
  start_idx <- pmax(1L, idx - RMS_PERIOD + 1L)
  rms_v     <- sqrt((cum_sq[idx + 1] - cum_sq[start_idx]) / (idx - start_idx + 1L))

  ift <- tanh(ifelse(rms_v > 0, deriv / rms_v, 0))

  out <- list()
  for (p in periods) out[[paste0("eleg_", p)]] <- clamp(ultimate_smoother(ift, p), 1)
  .set_warmup(as.data.frame(out), max(c(periods, RMS_PERIOD)))
}

# Correlation Trend Indicator at grid periods + change rate.
# Output [-1, +1] and [0, 1].
# Warmup: max of CTI rolling window periods.
feat_cti <- function(x, periods = GRID) {
  out <- list()
  for (p in periods) out[[paste0("cti_", p)]] <- cti(x, p)
  out[["cti_chg"]] <- clamp(c(0, abs(diff(cti(x, periods[2])))), 1)
  .set_warmup(as.data.frame(out), max(periods))
}

# Laguerre spread at three gamma values. Output [-3, +3].
# Warmup: RMS settling (Laguerre settles in ~4/(1-gamma) bars, well under RMS_PERIOD).
feat_laguerre <- function(x, gammas = c(0.3, 0.6, 0.8)) {
  out <- list()
  for (g in gammas)
    out[[paste0("lag_spread_", sub("\\.", "", as.character(g)))]] <-
      rms_normalize(x - laguerre(x, g))
  .set_warmup(as.data.frame(out), RMS_PERIOD)
}

# Volume features: filtered volume + price-volume correlation.
# Output: filtered [-3,+3], correlation [-1,+1].
# Warmup: max of HP, LP periods, RMS settling, and rolling_cor window.
feat_volume <- function(close, volume, periods = GRID) {
  out <- list()
  for (p in periods) {
    filt_p <- roofing(close,  HP_PERIOD, p)
    filt_v <- roofing(volume, HP_PERIOD, p)
    out[[paste0("vol_roof_", p)]]  <- rms_normalize(filt_v)
    out[[paste0("vol_pcorr_", p)]] <- rolling_cor(filt_p, filt_v, max(p, 30))
  }
  .set_warmup(as.data.frame(out), max(c(HP_PERIOD, periods, RMS_PERIOD)))
}


# AVERAGE COST =================================================================
# Turnover-weighted EMA of VWAP (CYQ chip distribution theory).
# avg_cost[i] = (1 - to[i]) * avg_cost[i-1] + to[i] * (amount[i] / volume[i])
# Time-varying alpha (to[i]) — cannot use stats::filter; requires loop.
# Becomes meaningful after cumulative turnover exceeds 1x (full float rotation).

calculate_avg_cost <- function(amount, volume, to) {
  n         <- length(amount)
  avg_price <- amount / volume
  avg_cost  <- numeric(n)
  avg_cost[1] <- avg_price[1]
  for (i in 2:n) avg_cost[i] <- (1 - to[i]) * avg_cost[i-1] + to[i] * avg_price[i]
  first_valid <- which(cumsum(to) >= 1)[1]
  if (is.na(first_valid)) first_valid <- n
  if (first_valid > 1) avg_cost[1:(first_valid - 1)] <- avg_cost[first_valid]
  avg_cost
}

# Cost basis features. Output [-3, +3].
# Warmup: max of US periods and RMS settling.
feat_cost <- function(close, avg_cost, periods = GRID) {
  out <- list()
  out[["cost_spread"]] <- rms_normalize(close - avg_cost)
  out[["cost_slope"]]  <- rms_normalize(c(0, diff(avg_cost)))
  for (p in periods)
    out[[paste0("cost_xscale_", p)]] <- rms_normalize(ultimate_smoother(close, p) - avg_cost)
  .set_warmup(as.data.frame(out), max(c(periods, RMS_PERIOD)))
}


# OHLC INTRABAR FEATURES =======================================================
# Fully vectorized. Range/gap [-3,+3], ratios [0,1].
# Warmup: RMS settling on range and gap signals.

feat_ohlc <- function(open, high, low, close) {
  n         <- length(close)
  range_raw <- high - low
  range_raw[range_raw == 0] <- 1e-8

  .set_warmup(data.frame(
    ohlc_range      = rms_normalize(range_raw / close),
    ohlc_body_pos   = (close - low) / range_raw,
    ohlc_upper_wick = (high - pmax(open, close)) / range_raw,
    ohlc_lower_wick = (pmin(open, close) - low) / range_raw,
    ohlc_gap        = rms_normalize(c(0, open[-1] - close[-n]) / close)
  ), RMS_PERIOD)
}


# FUNDAMENTAL FEATURES =========================================================
# Raw ratios — normalize cross-sectionally (across all stocks per date)
# with bestNormalize::orderNorm() before feeding to the RL model.
# No warmup — pure point-in-time ratios.

feat_fundamental <- function(mc, np, np_deduct, equity, revenue, ocf) {
  safe_div <- function(a, b) ifelse(b != 0, a / b, 0)
  data.frame(
    fund_ey      = safe_div(np, mc),
    fund_eyd     = safe_div(np_deduct, mc),
    fund_by      = safe_div(equity, mc),
    fund_sy      = safe_div(revenue, mc),
    fund_cfy     = safe_div(ocf, mc),
    fund_roe     = safe_div(np, equity),
    fund_accrual = safe_div(np - ocf, mc)
  )
}


# MAMA / FAMA ==================================================================
# MESA Adaptive Moving Average (Ehlers).
# Hilbert Transform → instantaneous phase → adaptive alpha from delta_phase.
# MAMA[i] = alpha[i]*x[i] + (1-alpha[i])*MAMA[i-1]
# FAMA[i] = 0.5*alpha[i]*MAMA[i] + (1-0.5*alpha[i])*FAMA[i-1]
# Data-dependent alpha — cannot use stats::filter; requires loop.

mama_fama <- function(x, fast_limit = 0.5, slow_limit = 0.05) {
  n <- length(x)

  # 4-bar WMA smooth
  smooth <- as.numeric(stats::filter(x, filter = c(4, 3, 2, 1) / 10,
                                      method = "convolution", sides = 1))
  smooth[is.na(smooth)] <- x[is.na(smooth)]

  # Hilbert Transform FIR coefficients (Ehlers' 7-tap approximation)
  ht <- c(0.0962, 0, 0.5769, 0, -0.5769, 0, -0.0962)

  # Hilbert Transform helper: apply FIR with period-adaptive gain
  ht_fir <- function(sig, i, adj) {
    sig[i]   * ht[1] + sig[i-2] * ht[3] +
    sig[i-4] * ht[5] + sig[i-6] * ht[7]
  } # ht[2,4,6] are 0 so omitted

  detrender <- Q1 <- I1 <- jI <- jQ <- numeric(n)
  I2 <- Q2 <- Re_v <- Im_v <- numeric(n)
  period <- rep(20, n)
  phase  <- alpha_v <- numeric(n)
  mama   <- fama <- x  # initialize to close

  if (n < 7) return(list(mama = mama, fama = fama))

  for (i in 7:n) {
    adj <- 0.075 * period[i-1] + 0.54

    # Smooth → Detrender via Hilbert
    detrender[i] <- ht_fir(smooth, i, adj) * adj

    # InPhase and Quadrature
    Q1[i] <- ht_fir(detrender, i, adj) * adj
    I1[i] <- detrender[i-3]

    # Advance phase by 90° via Hilbert
    jI[i] <- ht_fir(I1, i, adj) * adj
    jQ[i] <- ht_fir(Q1, i, adj) * adj

    # Phasor addition + EMA smooth (alpha = 0.2)
    I2[i] <- 0.2 * (I1[i] - jQ[i]) + 0.8 * I2[i-1]
    Q2[i] <- 0.2 * (Q1[i] + jI[i]) + 0.8 * Q2[i-1]

    # Homodyne Discriminator for period
    Re_v[i] <- 0.2 * (I2[i] * I2[i-1] + Q2[i] * Q2[i-1]) + 0.8 * Re_v[i-1]
    Im_v[i] <- 0.2 * (I2[i] * Q2[i-1] - Q2[i] * I2[i-1]) + 0.8 * Im_v[i-1]

    if (Im_v[i] != 0 && Re_v[i] != 0) {
      period[i] <- 2 * pi / atan(Im_v[i] / Re_v[i])
    } else {
      period[i] <- period[i-1]
    }
    period[i] <- max(6, min(50, max(0.67 * period[i-1], min(1.5 * period[i-1], period[i]))))
    period[i] <- 0.2 * period[i] + 0.8 * period[i-1]

    # Phase and adaptive alpha
    # Ehlers: EasyLanguage variables retain previous value when If block skips.
    phase[i] <- phase[i-1]
    if (I1[i] != 0) phase[i] <- atan(Q1[i] / I1[i]) * 180 / pi
    dp <- max(1, phase[i-1] - phase[i])
    alpha_v[i] <- max(slow_limit, min(fast_limit, fast_limit / dp))

    # MAMA and FAMA
    mama[i] <- alpha_v[i] * x[i] + (1 - alpha_v[i]) * mama[i-1]
    fama[i] <- 0.5 * alpha_v[i] * mama[i] + (1 - 0.5 * alpha_v[i]) * fama[i-1]
  }

  list(mama = mama, fama = fama)
}

# MAMA features: spread, crossover.
# Output: spread/cross [-3,+3].
# Warmup: RMS settling (Hilbert settles in ~50 bars, within RMS_PERIOD).
feat_mama <- function(x, fast_limit = 0.5, slow_limit = 0.05) {
  mf <- mama_fama(x, fast_limit, slow_limit)
  .set_warmup(data.frame(
    mama_spread = rms_normalize(x - mf$mama),
    mama_cross  = rms_normalize(mf$mama - mf$fama)
  ), RMS_PERIOD)
}


# ENSEMBLE =====================================================================

ehlers_features <- function(close, open = NULL, high = NULL, low = NULL,
    volume = NULL, amount = NULL, to = NULL,
    mc = NULL, np = NULL, np_deduct = NULL, equity = NULL, revenue = NULL, ocf = NULL) {

  feats <- list(
    roofing  = feat_roofing(close),
    ultimate = feat_ultimate(close),
    ebsw     = feat_ebsw(close),
    elegant  = feat_elegant(close),
    cti      = feat_cti(close),
    laguerre = feat_laguerre(close),
    mama     = feat_mama(close)
  )

  if (!is.null(volume))
    feats[["volume"]] <- feat_volume(close, volume)

  if (!is.null(amount) && !is.null(volume) && !is.null(to))
    feats[["cost"]] <- feat_cost(close, calculate_avg_cost(amount, volume, to))

  if (!is.null(open) && !is.null(high) && !is.null(low))
    feats[["ohlc"]] <- feat_ohlc(open, high, low, close)

  if (!is.null(mc) && !is.null(np) && !is.null(np_deduct) &&
      !is.null(equity) && !is.null(revenue) && !is.null(ocf))
    feats[["fund"]] <- feat_fundamental(mc, np, np_deduct, equity, revenue, ocf)

  result <- do.call(cbind, feats)
  mat    <- as.matrix(result)
  result[is.nan(mat) | is.infinite(mat)] <- NA_real_
  result
}


# Market-level features: use ehlers_features() with market aggregates,
# prefix feature names with "mkt_" for namespace separation:

# feats <- mkt_feats <- ehlers_features(
#   close         = mkt_data$close,
#   volume        = mkt_data$to,
#   mc            = mkt_data$mc,
#   np            = mkt_data$np,
#   np_deduct     = mkt_data$np_deduct,
#   equity        = mkt_data$equity,
#   revenue       = mkt_data$revenue,
#   ocf           = mkt_data$ocf
# ) %>%
#   rename_with(~ paste0("mkt_", .x)) %>%
#   as_tibble()


# VERIFICATION =================================================================
# Input: data.frame with columns from example.csv:
#   symbol, date, open, high, low, close, volume, amount, to,
#   mc, np, np_deduct, equity, revenue, ocf

# t0 <- proc.time()

# feats <- ehlers_features(
#   close         = data$close,
#   open          = data$open,
#   high          = data$high,
#   low           = data$low,
#   volume        = data$volume,
#   amount        = data$amount,
#   to            = data$to,
#   mc            = data$mc,
#   np            = data$np,
#   np_deduct     = data$np_deduct,
#   equity        = data$equity,
#   revenue       = data$revenue,
#   ocf           = data$ocf
# )

# elapsed <- (proc.time() - t0)[3]
# cat(nrow(feats), "x", ncol(feats), "in", round(elapsed, 3), "s\n")

# mat <- as.matrix(feats)
# valid <- mat[complete.cases(mat), ]
# cat("Valid rows:", nrow(valid), " Warmup NAs:", sum(is.na(mat)), "\n")
# cat("Bounds: [", round(min(valid), 4), ",", round(max(valid), 4), "]\n\n")

# cat(sprintf("%-35s %8s %8s %8s\n", "Feature", "NAs", "Min", "Max"))
# for (col in names(feats)) {
#   v <- feats[[col]]; w <- sum(is.na(v))
#   cat(sprintf("%-35s %8d %8.4f %8.4f\n", col, w,
#               min(v, na.rm = TRUE), max(v, na.rm = TRUE)))
# }

# groups <- sub("\\.[^.]+$", "", names(feats))
# cat("\nGroups:\n")
# for (g in unique(groups)) cat(sprintf("  %-20s %3d\n", g, sum(groups == g)))
# cat(sprintf("  %-20s %3d\n", "TOTAL", ncol(feats)))
