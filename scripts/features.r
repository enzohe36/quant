# PRESET =======================================================================

# library(sn)

# HELPER FUNCTIONS =============================================================

run_sum <- function(x, n) {
  sapply(
    seq_along(x),
    function(i) if (i < n) NA_real_ else sum(x[(i - n + 1):i])
  )
}

run_mean <- function(x, n) run_sum(x, n) / n

run_min <- function(x, n) {
  sapply(
    seq_along(x),
    function(i) if (i < n) NA_real_ else min(x[(i - n + 1):i])
  )
}

run_max <- function(x, n) {
  sapply(
    seq_along(x),
    function(i) if (i < n) NA_real_ else max(x[(i - n + 1):i])
  )
}

run_any <- function(x, n) {
  sapply(
    seq_along(x),
    function(i) if (i < n) NA_character_ else any(x[(i - n + 1):i])
  )
}

cumsum_na <- function(x) {
  first_non_na <- which(!is.na(x))[1]
  result <- x
  if (!is.na(first_non_na)) {
    result[first_non_na:length(x)] <- cumsum(x[first_non_na:length(x)])
  }
  return(result)
}

ema_na <- function(x, n = 10, ...) {
  first_non_na <- which(!is.na(x))[1]
  result <- rep(NA, length(x))
  if (isTRUE(length(x) - first_non_na + 1 >= n)) {
    result[first_non_na:length(x)] <- EMA(x[first_non_na:length(x)], n = n, ...)
  }
  return(result)
}

atr_na <- function(HLC, n = 14, ...) {
  first_non_na <- which(complete.cases(HLC))[1]
  num_rows <- nrow(HLC)
  result <- data.frame(
    tr = rep(NA, num_rows),
    atr = rep(NA, num_rows),
    truehigh = rep(NA, num_rows),
    truelow = rep(NA, num_rows)
  )
  if (isTRUE(num_rows - first_non_na + 1 > n)) {
    result[first_non_na:num_rows, ] <-
      ATR(HLC[first_non_na:num_rows, ], n = n, ...)
  }
  return(result)
}

replace_missing <- function(x, replacement) {
  x[is.infinite(x) | is.na(x)] <- replacement
  return(x)
}

calculate_avg_cost <- function(avg_price, to) {
  n <- length(avg_price)
  avg_cost <- numeric(n)
  avg_cost[1] <- avg_price[1]
  for (i in 2:n) {
    avg_cost[i] <- (1 - to[i]) * avg_cost[i - 1] + to[i] * avg_price[i]
  }
  avg_cost[which(is.na(cumsum_na(to)) | cumsum_na(to) <= 1)] <- NA_real_
  return(avg_cost)
}

normalize_sn <- function(x, silent = FALSE) {
  x <- replace_missing(x, NA_real_)

  if (!silent) {
    par(mfrow = c(1, 2))
    q_orig <- quantile(x, probs = c(0.001, 0.999), na.rm = TRUE)
    breaks_orig <- c(
      min(x, na.rm = TRUE),
      seq(q_orig[1], q_orig[2], length.out = 31),
      max(x, na.rm = TRUE)
    )
    hist(
      x, breaks = breaks_orig, probability = TRUE,
      main = "Original Distribution",
      xlab = "x", ylab = "Density",
      xlim = q_orig
    )
  }

  fit_sn <- selm(x ~ 1, family = "SN")
  xi <- coef(fit_sn, param.type = "DP")["xi"]
  omega <- coef(fit_sn, param.type = "DP")["omega"]
  alpha <- coef(fit_sn, param.type = "DP")["alpha"]

  if (!silent) {
    x_seq <- seq(q_orig[1], q_orig[2], length.out = 1000)
    lines(
      x_seq, dsn(x_seq, xi = xi, omega = omega, alpha = alpha),
      col = "red", lwd = 2
    )
  }

  eps <- .Machine$double.eps
  u <- psn(x, xi = xi, omega = omega, alpha = alpha)
  u <- pmax(pmin(u, 1 - eps), eps)
  x_symmetric <- qnorm(u)

  if (!silent) {
    q_trans <- quantile(x_symmetric, probs = c(0.001, 0.999), na.rm = TRUE)
    breaks_trans <- c(
      min(x_symmetric, na.rm = TRUE),
      seq(q_trans[1], q_trans[2], length.out = 31),
      max(x_symmetric, na.rm = TRUE)
    )
    hist(
      x_symmetric, breaks = breaks_trans, probability = TRUE,
      main = "Transformed Distribution",
      xlab = "Transformed x", ylab = "Density",
      xlim = q_trans
    )
    z_seq <- seq(q_trans[1], q_trans[2], length.out = 1000)
    lines(z_seq, dnorm(z_seq, mean = 0, sd = 1), col = "blue", lwd = 2)
    par(mfrow = c(1, 1))
  }

  if (!silent) {
    cat("Fitted Skewed Normal Parameters:\n")
    cat(sprintf("  Location (xi):     %.4f\n", xi))
    cat(sprintf("  Scale (omega):        %.4f\n", omega))
    cat(sprintf("  Shape/Skew (alpha):   %.4f\n", alpha))
    cat("\nTransformation: Skewed Normal -> Standard Normal N(0,1)\n")
    cat(sprintf("Original skewness:    %.4f\n",
                moments::skewness(x, na.rm = TRUE)))
    cat(sprintf("Transformed skewness: %.4f (should be ≈0)\n",
                moments::skewness(x_symmetric, na.rm = TRUE)))
  }

  return(x_symmetric)
}

calculate_scale_params <- function(data, ..., robust = FALSE) {
  center_fn <- if (robust) {
    \(x) median(x, na.rm = TRUE)
  } else {
    \(x) mean(x, na.rm = TRUE)
  }
  disp_fn <- if (robust) {
    \(x) mad(x, na.rm = TRUE)
  } else {
    \(x) sd(x, na.rm = TRUE)
  }

  data %>%
    select(...) %>%
    summarise(
      across(everything(), list(center = center_fn, disp = disp_fn))
    ) %>%
    pivot_longer(
      everything(),
      names_to = c("column", "stat"),
      names_sep = "_(?=(center|disp)$)"
    ) %>%
    pivot_wider(
      names_from = column,
      values_from = value
    )
}

scale_features <- function(data, scale_params) {
  center_vals <- scale_params %>% filter(stat == "center") %>% select(-stat)
  disp_vals <- scale_params %>% filter(stat == "disp") %>% select(-stat)

  cols_to_scale <- setdiff(names(scale_params), "stat")

  data %>%
    mutate(
      across(
        all_of(cols_to_scale),
        ~ {
          col_name <- cur_column()
          (. - center_vals[[col_name]]) / disp_vals[[col_name]]
        }
      )
    )
}
