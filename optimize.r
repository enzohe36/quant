calc_apy25 <- function(t_max, symbol_list = load[[1]], data_list = load[[2]]) {
  # Define parameters
  t_adx <- 70
  t_cci <- 51
  x_h <- 0.53
  x_l <- 0.31
  r_h <- 0.037
  t_min <- 14
  t_max <- 104

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    symbol = symbol_list,
    .combine = append,
    .export = c("tnormalize", "calc_adx", "calc_ror"),
    .packages = c("TTR", "tidyverse")
  ) %dopar% {
    data <- data_list[[symbol]]

    # Calculate predictor
    adx <- 1 - tnormalize(
      abs(calc_adx(data[, 3:5])[, 1] - calc_adx(data[, 3:5])[, 2]), t_adx
    )
    cci <- 1 - 2 * tnormalize(CCI(data[, 3:5]), t_cci)
    data$x <- adx * cci
    data$dx <- momentum(data$x, 5)

    data <- na.omit(data)

    trade <- c()
    s <- 1
    for (i in 1:nrow(data)) {
      if (i < s | data[i, "x"] < x_h | data[i, "dx"] <= 0) {
        next
      }
      for (j in i:nrow(data)) {
        if (
          (
            calc_ror(data[i, "close"], data[j, "close"]) >= r_h &
            data[j, "x"] <= x_l &
            j - i >= t_min
          ) | (
            j - i >= t_max
          )
        ) {
          s <- j
          break
        }
      }
      if (i < s) {
        r <- calc_ror(data[i, "close"], data[s, "close"])
        trade <- c(trade, r)
      }
    }

    apy <- sum(trade) / as.numeric(data[nrow(data), 1] - data[1, 1]) * 365

    return(apy)
  }
  unregister_dopar

  return(quantile(out, 0.25))
}

out <- optimize(
  calc_apy25, c(40, 120), tol = 1, maximum = TRUE
)

print(out)