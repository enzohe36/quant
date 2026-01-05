# PRESET =======================================================================

# library(sn)

# HELPER FUNCTIONS =============================================================

calculate_avg_cost <- function(avg_price, to) {
  n <- length(avg_price)
  avg_cost <- numeric(n)
  avg_cost[1] <- avg_price[1]
  for (i in 2:n) {
    avg_cost[i] <- (1 - to[i]) * avg_cost[i - 1] + to[i] * avg_price[i]
  }
  return(avg_cost)
}

normalize <- function(x, n_bins = 100, plot = TRUE, silent = FALSE) {
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
    cat(sprintf("  Location (ξ):     %.4f\n", xi))
    cat(sprintf("  Scale (ω):        %.4f\n", omega))
    cat(sprintf("  Shape/Skew (α):   %.4f\n", alpha))
    cat("\nTransformation: Skewed Normal -> Standard Normal N(0,1)\n")
    cat(sprintf("Original skewness:    %.4f\n",
                moments::skewness(x, na.rm = TRUE)))
    cat(sprintf("Transformed skewness: %.4f (should be ≈0)\n",
                moments::skewness(x_symmetric, na.rm = TRUE)))
  }

  return(x_symmetric)
}

calculate_wrm <- function(
  x, w = NULL, s_low = 2, s_high = 2, output_index = FALSE
) {
  w <- if (is.null(w)) rep(1, length(x))
  x_w <- x * w
  x_mean <- sum(x_w, na.rm = TRUE) / sum(w[!is.na(x_w)])
  x_sq_w <- w * (x - x_mean)^2
  x_sd <- sqrt(sum(x_sq_w, na.rm = TRUE) / sum(w[!is.na(x_sq_w)]))
  index <- which(x >= x_mean - s_low * x_sd & x <= x_mean + s_high * x_sd)
  if (output_index) {
    return(index)
  } else {
    x_f <- x[index]
    w_f <- w[index]
    x_f_w <- x_f * w_f
    return(sum(x_f_w, na.rm = TRUE) / sum(w_f[!is.na(x_f_w)]))
  }
}
