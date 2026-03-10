# Config =======================================================================

# feat_log_ratios.r — Log ratio space feature ensemble for RL training.

# Features:
#   price log ratio space, price KAMA log ratio,
#   turnover log ratio space, cross-sectional fundamentals.

# Log ratio space: log(x / lag(x, lb)) at geometrically spaced lookback periods.

# Helpers ======================================================================

safe_div <- function(a, b) {
  ifelse(b != 0, a / b, 0)
}

has_all <- function(...) !any(vapply(list(...), is.null, logical(1)))

blank_warmup <- function(df, n_warmup) {
  df[seq_len(min(n_warmup, nrow(df))), ] <- NA_real_
  df
}

# Feature Generators ===========================================================

# Log ratio space at geometrically spaced lookback periods.
# Output: data.frame with columns lr_<lookback>. Warmup blanked.
feat_log_ratio_space <- function(x, lookback_length = 240, n_ratio = 20, eps = 0) {
  n <- length(x)
  lookbacks <- ceiling((lookback_length^(1 / n_ratio))^c(0, seq_len(n_ratio)))
  lookbacks <- unique(lookbacks)
  log_ratios <- lapply(lookbacks, function(lb) {
    lagged <- if (lb >= n) rep(NA_real_, n) else c(rep(NA_real_, lb), x[1:(n - lb)])
    log((x + eps) / (lagged + eps))
  })
  names(log_ratios) <- paste0("lr_", lookbacks)
  blank_warmup(as.data.frame(log_ratios), max(lookbacks))
}

# KAMA log ratio. Kaufman adaptive moving average on (high + low) / 2.
# Output: data.frame with column lr_1.
feat_kama <- function(high, low, n_er = 10, n_fast = 2, n_slow = 30) {
  price <- (high + low) / 2
  n <- length(price)

  direction <- c(rep(NA_real_, n_er), price[(n_er + 1):n] - price[1:(n - n_er)])
  daily_diff <- c(0, abs(diff(price)))
  cs <- cumsum(daily_diff)
  volatility <- c(rep(NA_real_, n_er), cs[(n_er + 1):n] - cs[1:(n - n_er)])

  er <- direction / volatility
  er[is.infinite(er)] <- 0

  sc <- er * (2 / (1 + n_fast) - 2 / (1 + n_slow)) + 2 / (1 + n_slow)
  alpha <- sc * sc
  alpha[is.na(alpha)] <- 1

  kama <- numeric(n)
  kama[1:n_er] <- price[1:n_er]
  for (i in (n_er + 1):n) {
    kama[i] <- alpha[i] * price[i] + (1 - alpha[i]) * kama[i - 1]
  }

  lagged <- c(NA_real_, kama[1:(n - 1)])
  data.frame(lr_1 = log(kama / lagged))
}

# Fundamentals: _cn raw for cross-sectional norm.
feat_fundamental <- function(mc, np, np_deduct, equity, revenue, ocf, mkt_mc = NULL) {
  raw <- data.frame(
    ey_cn = safe_div(np, mc),
    eyd_cn = safe_div(np_deduct, mc),
    by_cn = safe_div(equity, mc),
    sy_cn = safe_div(revenue, mc),
    cfy_cn = safe_div(ocf, mc),
    roe_cn = safe_div(np, equity),
    accrual_cn = safe_div(np - ocf, mc)
  )
  if (!is.null(mkt_mc)) {
    raw <- cbind(data.frame(ms_cn = safe_div(mc, mkt_mc)), raw)
  }
  raw
}

# Ensemble =====================================================================

make_features <- function(
    close, open = NULL, high = NULL, low = NULL,
    volume = NULL, amount = NULL, to = NULL,
    mc = NULL, mkt_mc = NULL,
    np = NULL, np_deduct = NULL,
    equity = NULL, revenue = NULL, ocf = NULL
) {
  n <- length(close)
  feats <- list()

  # Price log ratio space
  feats$close <- feat_log_ratio_space(close)

  # Price KAMA log ratio (lookback = 1)
  if (has_all(high, low) && n > 10) {
    feats$kama <- feat_kama(high, low)
  } else if (has_all(high, low)) {
    feats$kama <- data.frame(lr_1 = rep(NA_real_, n))
  }

  # Turnover log ratio space
  if (!is.null(to)) {
    feats$to <- feat_log_ratio_space(to, eps = 1e-4)
  }

  # Cross-sectional fundamentals
  if (has_all(mc, np, np_deduct, equity, revenue, ocf))
    feats$fund <- feat_fundamental(mc, np, np_deduct, equity, revenue, ocf, mkt_mc)

  # Add group prefix
  for (group in names(feats)) {
    names(feats[[group]]) <- paste0(group, ".", names(feats[[group]]))
  }

  result <- do.call(cbind, unname(feats))
  mat <- as.matrix(result)
  result[is.nan(mat) | is.infinite(mat)] <- NA_real_
  result
}

# Validation ===================================================================

validate_features <- function(feats, plot = FALSE) {
  feat_names <- grep("^[a-z_]+\\.", names(feats), value = TRUE)
  if (length(feat_names) == 0) {
    cat("No feature columns found (expected group.name pattern).\n")
    return(invisible(NULL))
  }

  n <- nrow(feats)

  # Complete-row count: column-by-column to avoid materializing full matrix
  has_na <- rep(FALSE, n)
  total_nas <- 0L
  for (col in feat_names) {
    col_na <- is.na(feats[[col]])
    has_na <- has_na | col_na
    total_nas <- total_nas + sum(col_na)
  }
  n_complete <- sum(!has_na)
  rm(has_na)

  cat("Rows:", n, " Complete:", n_complete, " Total NAs:", total_nas, "\n")

  # Per-column stats
  cat(sprintf("\n%-35s %8s %8s %8s %8s %8s\n",
              "Feature", "NAs", "Min", "Mean", "Max", "SD"))
  cat(strrep("-", 80), "\n")

  for (col in feat_names) {
    v <- feats[[col]]
    na_count <- sum(is.na(v))
    if (na_count == length(v)) {
      cat(sprintf("%-35s %8d %8s %8s %8s %8s\n",
                  col, na_count, "NA", "NA", "NA", "NA"))
      next
    }
    vmin <- min(v, na.rm = TRUE)
    vmean <- mean(v, na.rm = TRUE)
    vmax <- max(v, na.rm = TRUE)
    vsd <- sd(v, na.rm = TRUE)

    cat(sprintf("%-35s %8d %8.3f %8.3f %8.3f %8.3f\n",
                col, na_count, vmin, vmean, vmax, vsd))
  }

  # Group summary
  groups <- sub("\\.[^.]+$", "", feat_names)
  cat("\nGroups:\n")
  for (g in unique(groups)) {
    cat(sprintf("  %-25s %3d\n", g, sum(groups == g)))
  }
  cat(sprintf("  %-25s %3d\n", "TOTAL", length(feat_names)))

  # Histograms: sample to keep plot rendering fast
  if (plot) {
    idx <- if (n > 50000) sample.int(n, 50000) else seq_len(n)
    for (col in feat_names) {
      v <- feats[[col]][idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) hist(v, breaks = 30, main = col, xlab = col)
    }
  }

  invisible(NULL)
}
