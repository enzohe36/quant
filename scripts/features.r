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

normalize <- function(x, n_bins = NULL, plot = TRUE, silent = FALSE) {
  x <- replace_missing(x, NA_real_)
  if (is.null(n_bins)) n_bins <- ceiling(sqrt(length(na.omit(x))))

  # Calculate probability histogram
  if (plot & !silent) {
    par(mfrow = c(1, 2))
    hist(x, breaks = n_bins, probability = TRUE,
         main = "Original Distribution",
         xlab = "x", ylab = "Density")
  }

  # Fit skewed normal distribution
  fit <- selm(x ~ 1, family = "SN")
  params <- coef(fit, param.type = "DP")

  xi <- params["xi"]          # location
  omega <- params["omega"]    # scale
  alpha <- params["alpha"]    # shape

  # Add fitted curve to original plot
  if (plot & !silent) {
    x_seq <- seq(
      min(x, na.rm = TRUE),
      max(x, na.rm = TRUE),
      length.out = 1000
    )
    lines(x_seq, dsn(x_seq, xi = xi, omega = omega, alpha = alpha),
          col = "red", lwd = 2)
  }

  # Transform to symmetric normal using probability integral transformation
  # Step 1: Transform to uniform [0,1] using CDF of fitted skewed normal
  u <- psn(x, xi = xi, omega = omega, alpha = alpha)

  # Step 2: Transform to standard normal using inverse CDF (quantile function)
  x_symmetric <- qnorm(u)

  # Plot transformed data
  if (plot & !silent) {
    hist(x_symmetric, breaks = n_bins, probability = TRUE,
         main = "Transformed Distribution",
         xlab = "Transformed x", ylab = "Density")

    # Overlay theoretical N(0,1) curve
    z_seq <- seq(
      min(x_symmetric, na.rm = TRUE),
      max(x_symmetric, na.rm = TRUE),
      length.out = 1000
    )
    lines(z_seq, dnorm(z_seq, mean = 0, sd = 1), col = "blue", lwd = 2)

    par(mfrow = c(1, 1))
  }

  # Print fitted parameters and transformation info
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

generate_differences <- function(data, ..., n) {
  if (!is.data.frame(data)) {
    stop("data must be a data frame")
  }

  if (!is.numeric(n) || any(n < 1) || any(n != floor(n))) {
    stop("n must be a vector of positive integers")
  }

  lag_fns <- purrr::map(n, function(lag_n) {
    function(x) x - dplyr::lag(x, n = lag_n)
  })

  names(lag_fns) <- paste0("d", n)

  result <- data %>%
    mutate(
      across(
        .cols = c(...),
        .fns = lag_fns,
        .names = "{.col}_{.fn}"
      )
    )

  return(result)
}