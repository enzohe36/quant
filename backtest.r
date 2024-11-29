backtest <- function(t_adx, t_cci, x_h, r_h, r_l, t_max, descriptive = TRUE) {
  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Started backtest()."
    ),
    quote = FALSE
  )

  # Define input
  symbol_list <- out0[[1]]
  data_list <- out0[[2]]

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  trade <- foreach(
    symbol = symbol_list,
    .combine = rbind,
    .export = c("tnormalize", "adx_alt", "ror"),
    .packages = c("TTR", "tidyverse")
  ) %dopar% {
    data <- data_list[[symbol]]

    # Calculate predictor
    adx <- adx_alt(data[, 3:5])
    adx <- 1 - tnormalize(abs(adx$adx - adx$adxr), t_adx)
    cci <- 1 - 2 * tnormalize(CCI(data[, 3:5]), t_cci)
    data$x <- adx * cci
    data$x1 <- lag(data$x, 1)
    data$dx <- momentum(data$x, 5)

    data <- na.omit(data)

    trade <- data.frame()
    for (i in 1:(nrow(data) - 1)) {
      r <- NaN
      if (!(data[i, "x"] >= x_h & data[i, "x1"] < x_h & data[i, "dx"] > 0)) next
      for (j in (i + 1):nrow(data)) {
        if (ror(data[i, "close"], data[j, "high"]) >= r_h) {
          r <- r_h
          break
        }
        if (ror(data[i, "close"], data[j, "low"]) <= r_l) {
          ifelse(
            ror(data[i, "close"], data[j, "high"]) > r_l,
            r <- r_l,
            r <- ror(data[i, "close"], data[j, "close"])
          )
          break
        }
        if (j - i >= t_max) {
          r <- ror(data[i, "close"], data[j, "close"])
          break
        }
      }
      trade <- rbind(
        trade, list(symbol, data[i, "date"], data[j, "date"], r, j - i)
      )
    }
    colnames(trade) <- c("symbol", "buy", "sell", "r", "t")
    trade <- trade[trade$r >= 0.9^trade$t - 1 & trade$r <= 1.1^trade$t - 1, ]
    trade$buy <- as.Date(trade$buy)
    trade$sell <- as.Date(trade$sell)

    trade <- na.omit(trade)

    return(trade)
  }
  unregister_dopar

  if (!descriptive) return(trade)

  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Backtested ", length(unique(trade$symbol)), " stocks."
    ),
    quote = FALSE
  )

  stats <- rbind(
    data.frame(
      r = quantile(trade$r),
      t = quantile(trade$t),
      t_cal = quantile(as.numeric(trade$sell - trade$buy))
    ),
    data.frame(
      r = NaN,
      t = NaN,
      t_cal = NaN,
      row.names = ""
    ),
    data.frame(
      r = c(mean(trade$r), sd(trade$r)),
      t = c(mean(trade$t), sd(trade$t)),
      t_cal = c(
        mean(as.numeric(trade$sell - trade$buy)),
        sd(as.numeric(trade$sell - trade$buy))
      ),
      row.names = c("mean", "sd")
    )
  )
  stats$r <- format(round(stats$r, 3), nsmall = 3)
  stats[, c("t", "t_cal")] <- round(stats[, c("t", "t_cal")])
  stats["", ] <- ""
  writeLines(c("", capture.output(stats)))

  hist <- hist(trade$r, breaks = 100)

  return(trade)
}
