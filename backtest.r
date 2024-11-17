# Define parameters
x_h <- 0.6
x_l <- 0.5
r_h <- 0.01
t_min <- 10
t_max <- 60

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
  .export = "pct_change"
) %dopar% {
  data <- data_list[[symbol]]
  data <- na.omit(data)
  data[, 1] <- as.Date(data[, 1])

  s <- 1
  trading <- data.frame(matrix(nrow = 0, ncol = 4))
  for (i in 1:nrow(data)) {
    if (i < s | data[i, "x"] < x_h | data[i, "dx"] <= 0) {
      next
    }
    for (j in i:nrow(data)) {
      if (
        (
          pct_change(data[i, "close"], data[j, "close"]) >= r_h &
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
      r <- pct_change(data[i, "close"], data[s, "close"])
      trading <- rbind(
        trading, list(symbol, data[i, "date"], data[s, "date"], r)
      )
    }
  }
  colnames(trading) <- c("symbol", "buy", "sell", "r")
  trading[, 2] <- as.Date(trading[, 2])
  trading[, 3] <- as.Date(trading[, 3])

  apy <- data.frame(
    symbol,
    sum(trading[, 4]) / as.numeric(data[nrow(data), 1] - data[1, 1]) * 365
  )
  colnames(apy) <- c("symbol", "apy")

  return(list(trading, apy))
}
unregister_dopar

trading_list <- out[[1]]
names(trading_list) <- do.call(c, lapply(trading_list, function(df) df[1, 1]))
print(paste0(
    format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
    " Backtested ", length(trading_list), " stocks."
  ), quote = FALSE
)

trading <- do.call(rbind, trading_list)
writeLines(c(
    "",
    capture.output(round(quantile(trading[, 4], seq(0, 1, 0.1)), 4))
  )
)

apy_low <- do.call(rbind, out[[2]])[, 2]
apy_high <- trading[, 4] / as.numeric(trading[, 3] - trading[, 2]) * 365
stats <- data.frame(
  Mean = c(
    mean(trading[, 4]),
    mean(as.numeric(trading[, 3] - trading[, 2])),
    mean(apy_low),
    mean(apy_high)
  ),
  SD = c(
    sd(trading[, 4]),
    sd(as.numeric(trading[, 3] - trading[, 2])),
    sd(apy_low),
    sd(apy_high)
  ),
  row.names = c("r", "t", "APY_low", "APY_high")
)
writeLines(c(
    "",
    capture.output(round(stats, 2))
  )
)

hist <- hist(trading[trading$r <= 1 & trading$r >= -1, 4], breaks = 100)

return(trading_list)
