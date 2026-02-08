# PRESET =======================================================================

# library(sn)
# library(patchwork)
# library(bestNormalize)


# HELPER FUNCTIONS =============================================================

run_sum <- function(x, n) {
  sapply(
    seq_along(x),
    function(i) if (i < n) NA_real_ else sum(x[(i - n + 1):i])
  )
}

run_mean <- function(x, n) run_sum(x, n) / n

run_sd <- function(x, n) {
  sapply(
    seq_along(x),
    function(i) if (i < n) NA_real_ else sd(x[(i - n + 1):i])
  )
}

run_norm <- function(x, n) (x - run_mean(x, n)) / sd(x, n)

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


# PLOTTING =====================================================================

plot_dual_y <- function(x, y1, y2,
                        xlab = "x",
                        y1lab = "y1",
                        y2lab = "y2",
                        title = "",
                        y1_color = "blue",
                        y2_color = "red") {
  # Calculate ranges for both series
  min_y1 <- min(y1, na.rm = TRUE)
  max_y1 <- max(y1, na.rm = TRUE)
  range_y1 <- max_y1 - min_y1

  min_y2 <- min(y2, na.rm = TRUE)
  max_y2 <- max(y2, na.rm = TRUE)
  range_y2 <- max_y2 - min_y2

  # Scale y2 to match y1's range
  df <- data.frame(
    x = x,
    y1 = y1,
    y2 = y2,
    y2_scaled = (y2 - min_y2) / range_y2 * range_y1 + min_y1
  )

  p <- ggplot(df, aes(x = x)) +
    geom_line(aes(y = y1, color = "y1"), linewidth = 0.5) +
    geom_line(aes(y = y2_scaled, color = "y2"), linewidth = 0.5) +
    scale_y_continuous(
      name = y1lab,
      sec.axis = sec_axis(~ (. - min_y1) / range_y1 * range_y2 + min_y2,
                         name = y2lab)
    ) +
    scale_color_manual(
      values = c("y1" = y1_color, "y2" = y2_color),
      labels = c(y1lab, y2lab)
    ) +
    labs(title = title, x = xlab, color = "") +
    theme_minimal() +
    theme(
      axis.title.y.left = element_text(color = y1_color),
      axis.text.y.left = element_text(color = y1_color),
      axis.title.y.right = element_text(color = y2_color),
      axis.text.y.right = element_text(color = y2_color)
    )

  print(p)
  invisible(p)
}


# NORMALIZERS ==================================================================

# Main creation function
create_normalizers <- function(
  data, ..., method = c("scale", "robust_scale", "orderNorm")
) {
  method <- match.arg(method)

  selected_cols <- data %>%
    select(...) %>%
    names()

  normalizers <- map(selected_cols, function(col) {
    if (method %in% c("scale", "robust_scale")) {
      create_scale_normalizer(
        data[[col]], col, robust = method == "robust_scale"
      )
    } else {
      # Use bestNormalize::orderNorm directly
      create_ordernorm_wrapper(data[[col]], col)
    }
  })

  names(normalizers) <- selected_cols
  structure(normalizers, class = "normalizer_list")
}

combine_normalizers <- function(...) {
  all_norms <- list(...)
  combined <- unlist(all_norms, recursive = FALSE)
  structure(combined, class = "normalizer_list")
}

# Create scale normalizer (z-score or robust)
create_scale_normalizer <- function(x, col_name, robust = FALSE) {
  center <- if (robust) median(x, na.rm = TRUE) else mean(x, na.rm = TRUE)
  disp <- if (robust) mad(x, na.rm = TRUE) else sd(x, na.rm = TRUE)

  structure(
    list(
      column = col_name,
      method = if (robust) "robust_scale" else "scale",
      center = center,
      dispersion = disp
    ),
    class = "scale_normalizer"
  )
}

# Wrapper for bestNormalize::orderNorm object
create_ordernorm_wrapper <- function(x, col_name) {
  # Create the actual orderNorm object
  orq_obj <- orderNorm(x, warn = FALSE)

  # Wrap it with column info
  structure(
    list(
      column = col_name,
      method = "orderNorm",
      orq_object = orq_obj  # Store the actual bestNormalize object
    ),
    class = "ordernorm_wrapper"
  )
}

# Apply scale normalization to new data
predict.scale_normalizer <- function(object, newdata, ...) {
  (newdata - object$center) / object$dispersion
}

# Apply orderNorm using bestNormalize's predict method
predict.ordernorm_wrapper <- function(object, newdata, ...) {
  predict(object$orq_object, newdata = newdata, warn = FALSE)
}

# Apply all normalizers in a list to new data
predict.normalizer_list <- function(object, newdata, ...) {
  result <- newdata

  for (col_name in names(object)) {
    if (col_name %in% names(newdata)) {
      result[[col_name]] <- predict(object[[col_name]], newdata[[col_name]])
    }
  }

  result
}

# Print method for scale normalizer
print.scale_normalizer <- function(x, ...) {
  cat(sprintf("<%s Normalizer>\n", x$method))
  cat(sprintf("Column: %s\n", x$column))
  cat(sprintf("Center: %s\n", format(x$center, digits = 6)))
  cat(sprintf("Dispersion: %s\n", format(x$dispersion, digits = 6)))
  invisible(x)
}

# Print method for orderNorm wrapper
print.ordernorm_wrapper <- function(x, ...) {
  cat(sprintf("<%s Normalizer>\n", x$method))
  cat(sprintf("Column: %s\n", x$column))
  cat("bestNormalize::orderNorm object:\n")
  print(x$orq_object)
  invisible(x)
}

# Print method for normalizer list (brief overview)
print.normalizer_list <- function(x, ...) {
  cat(sprintf("Normalizer List (%d columns)\n", length(x)))
  cat(strrep("=", 50), "\n")

  methods <- sapply(x, function(obj) obj$method)
  for (method_type in unique(methods)) {
    cols <- names(x)[methods == method_type]
    cat(sprintf("\n%s (%d):\n", method_type, length(cols)))
    cat("  ", paste(cols, collapse = ", "), "\n")
  }

  cat("\nUse summary() for detailed information\n")
  invisible(x)
}

# Summary method for normalizer list (detailed view)
summary.normalizer_list <- function(object, ...) {
  cat(sprintf("Normalizer List Summary (%d columns)\n", length(object)))
  cat(strrep("=", 70), "\n\n")

  for (col_name in names(object)) {
    norm <- object[[col_name]]
    cat(sprintf("[%s] %s\n", norm$method, col_name))

    if (inherits(norm, "scale_normalizer")) {
      cat(sprintf("  Center: %s, Dispersion: %s\n",
                  format(norm$center, digits = 6),
                  format(norm$dispersion, digits = 6)))
    } else if (inherits(norm, "ordernorm_wrapper")) {
      cat(sprintf("  N observations: %d\n", norm$orq_object$n))
      cat(sprintf("  Ties present: %s\n", norm$orq_object$ties_status))
    }
  }

  invisible(object)
}
