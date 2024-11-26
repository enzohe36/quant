backtest <- function() {
  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Started backtest()."
    ),
    quote = FALSE
  )

  # Define parameters
  symbol_list <- out0[[1]]
  data_list <- out0[[2]]
  t_adx <- 70
  t_cci <- 51
  x_h <- 0.53
  r_h <- 0.1
  r_l <- -0.5
  t_max <- 104

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    symbol = symbol_list,
    .combine = multiout,
    .multicombine = TRUE,
    .init = list(list(), list()),
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
        trade, list(symbol, data[i, "date"], data[j, "date"], r, j - i)
      )
    }
    colnames(trade) <- c("symbol", "buy", "sell", "r", "t")
    trade <- trade[trade$r >= 0.9^trade$t - 1 & trade$r <= 1.1^trade$t - 1, ]
    trade$buy <- as.Date(trade$buy)
    trade$sell <- as.Date(trade$sell)

    apy <- data.frame(
      symbol = symbol,
      apy = sum(trade$r) /
        as.numeric(data[nrow(data), "date"] - data[1, "date"]) * 365,
      apy0 = ror(data[1, "close"], data[nrow(data), "close"]) /
        as.numeric(data[nrow(data), "date"] - data[1, "date"]) * 365
    )

    return(list(trade, apy))
  }
  unregister_dopar

  trade_list <- out[[1]]
  names(trade_list) <- do.call(
    c, lapply(trade_list, function(df) df[1, "symbol"])
  )
  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Backtested ", length(trade_list), " stocks."
    ),
    quote = FALSE
  )

  trade <- do.call(rbind, trade_list)
  apy <- do.call(rbind, out[[2]])[, "apy"]
  apy0 <- do.call(rbind, out[[2]])[, "apy0"]
  stats <- data.frame(
    r = quantile(trade$r),
    t = quantile(trade$t),
    t_cal = quantile(as.numeric(trade$sell - trade$buy)),
    apy = quantile(apy),
    apy0 = quantile(apy0)
  )
  stats[, c("r", "apy", "apy0")] <- format(
    round(stats[, c("r", "apy", "apy0")], 3), nsmall = 3
  )

  stats2 <- data.frame(
    r = c(mean(trade$r), sd(trade$r)),
    t = c(mean(trade$t), sd(trade$t)),
    t_cal = c(
      mean(as.numeric(trade$sell - trade$buy)),
      sd(as.numeric(trade$sell - trade$buy))
    ),
    apy = c(mean(apy), sd(apy)),
    apy0 = c(mean(apy0), sd(apy0))
  )
  rownames(stats2) <- c("mean", "sd")
  stats2[, c("r", "apy", "apy0")] <- format(
    round(stats2[, c("r", "apy", "apy0")], 3), nsmall = 3
  )
  stats2[, c("t", "t_cal")] <- round(stats2[, c("t", "t_cal")])

  stats <- rbind(stats, list("", "", "", "", ""), stats2)
  rownames(stats)[rownames(stats) == "1"] <- ""
  writeLines(c("", capture.output(stats)))

  hist <- hist(trade$r, breaks = 100)

  return(trade_list)
}