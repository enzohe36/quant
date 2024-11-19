backtest <- function(symbol_list = load[[1]], data_list = load[[2]]) {
  # Define parameters
  t_adx <- 70
  t_cci <- 51
  x_h <- 0.53
  x_l <- 0.31
  r_h <- 0.037
  t_min <- 14
  t_max <- 104

  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Started backtest()."
    ), quote = FALSE
  )

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    symbol = symbol_list,
    .combine = multiout,
    .multicombine = TRUE,
    .init = list(list(), list()),
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

    trade <- data.frame(matrix(nrow = 0, ncol = 4))
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
        trade <- rbind(
          trade, list(symbol, data[i, "date"], data[s, "date"], r)
        )
      }
    }
    colnames(trade) <- c("symbol", "buy", "sell", "r")
    trade[, 2] <- as.Date(trade[, 2])
    trade[, 3] <- as.Date(trade[, 3])

    apy <- data.frame(
      symbol = symbol,
      apy = sum(trade[, 4]) /
        as.numeric(data[nrow(data), 1] - data[1, 1]) * 365,
      apy0 = calc_ror(data[1, 5], data[nrow(data), 5]) /
        as.numeric(data[nrow(data), 1] - data[1, 1]) * 365
    )

    return(list(trade, apy))
  }
  unregister_dopar

  trade_list <- out[[1]]
  names(trade_list) <- do.call(c, lapply(trade_list, function(df) df[1, 1]))
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Backtested ", length(trade_list), " stocks."
    ), quote = FALSE
  )

  trade <- do.call(rbind, trade_list)
  apy <- do.call(rbind, out[[2]])[, 2]
  apy0 <- do.call(rbind, out[[2]])[, 3]
  stats <- data.frame(
    r = quantile(trade[, 4]),
    t = quantile(as.numeric(trade[, 3] - trade[, 2])),
    apy = quantile(apy),
    apy0 = quantile(apy0)
  )
  writeLines(c(
      "",
      capture.output(round(stats, 4))
    )
  )

  hist <- hist(trade[trade$r <= 1 & trade$r >= -1, 4], breaks = 100)

  return(trade_list)
}