backtest_min <- function(r_h, r_l) {
  # Define parameters
  symbol_list <- out0[[1]]
  data_list <- out0[[2]]
  t_adx <- 70
  t_cci <- 51
  x_h <- 0.53
  #r_h <- 0.1
  #r_l <- -0.5
  t_max <- 104

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    symbol = symbol_list,
    .combine = append,
    .export = c("tnormalize", "adx_alt", "ror"),
    .packages = c("TTR", "tidyverse")
  ) %dopar% {
    data <- data_list[[symbol]]

    # Calculate predictor
    adx <- 1 - tnormalize(
      abs(adx_alt(data[, 3:5])[, "adx"] - adx_alt(data[, 3:5])[, "adxr"]), t_adx
    )
    cci <- 1 - 2 * tnormalize(CCI(data[, 3:5]), t_cci)
    data$x <- adx * cci
    data$dx <- momentum(data$x, 5)

    data <- na.omit(data)

    trade <- data.frame()
    j <- 1
    for (i in 1:(nrow(data) - 1)) {
      if (!(i >= j & data[i, "x"] >= x_h & data[i, "dx"] > 0)) {
        next
      }
      for (j in (i + 1):nrow(data)) {
        if (
          ror(data[i, "close"], data[j, "high"]) >= r_h
        ) {
          r <- r_h
          break
        } else if (
          ror(data[i, "close"], data[j, "low"]) <= r_l
        ) {
          ifelse(
            ror(data[i, "close"], data[j, "high"]) > r_l,
            r <- r_l,
            r <- ror(data[i, "close"], data[j, "close"])
          )
          break
        } else if (
          j - i >= t_max
        ) {
          r <- ror(data[i, "close"], data[j, "close"])
          break
        }
      }
      trade <- rbind(
        trade, list(r, j - i)
      )
    }
    colnames(trade) <- c("r", "t")
    trade <- trade[trade$r >= 0.9^trade$t - 1 & trade$r <= 1.1^trade$t - 1, ]

    apy <- sum(trade$r) /
      as.numeric(data[nrow(data), "date"] - data[1, "date"]) * 365

    return(apy)
  }
  unregister_dopar

  write(
    paste0(r_h, ",", r_l, ",", mean(out), mean(out) / sd(out)),
    file = "optimization.csv",
    append = TRUE
  )
}

for (i in 1:nrow(out)) {
  for (j in 1:ncol(out)) {
    backtest_min(i / 100, -j / 10)
  }
}
