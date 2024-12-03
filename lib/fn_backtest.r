backtest <- function(
  t_adx, t_cci, x_thr, t_max, r_max, r_min, descr = TRUE,
  symbol_list = out0[["symbol_list"]],
  data_list = out0[["data_list"]]
) {
  tsprint("Started backtest().")

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  trade <- foreach(
    symbol = symbol_list,
    .combine = rbind,
    .export = c("tnormalize", "adx_alt", "ror", "predictor", "t_adx", "t_cci"),
    .packages = c("TTR", "tidyverse")
  ) %dopar% {
    rm("data", "i", "j", "r", "trade")

    try(data <- data_list[[symbol]], silent = TRUE)
    if (is.null(data)) return(NULL)

    data <- predictor(data) %>% na.omit
    if (nrow(data) == 0) return(NULL)

    trade <- data.frame()
    for (i in 1:(nrow(data) - 1)) {
      if (
        !(data[i, "x"] >= x_thr & data[i, "x1"] < x_thr & data[i, "dx"] > 0)
      ) {
        next
      }
      for (j in (i + 1):nrow(data)) {
        r <- NaN
        if (ror(data[i, "close"], data[j, "high"]) >= r_max) {
          r <- r_max
          break
        }
        if (ror(data[i, "close"], data[j, "low"]) <= r_min) {
          ifelse(
            ror(data[i, "close"], data[j, "open"]) <= r_min,
            r <- ror(data[i, "close"], data[j, "open"]),
            r <- r_min
          )
          break
        }
        if (j - i == t_max) {
          r <- ror(data[i, "close"], data[j, "close"])
          break
        }
      }
      trade <- rbind(
        trade, list(symbol, data[i, "date"], data[j, "date"], r, j - i)
      )
    }
    if (nrow(trade) == 0) return(NULL)

    colnames(trade) <- c("symbol", "buy", "sell", "r", "t")
    trade$buy <- as.Date(trade$buy)
    trade$sell <- as.Date(trade$sell)
    trade <- na.omit(trade)
    return(trade)
  }
  unregister_dopar

  if (descr) {
    tsprint(glue("Backtested {length(unique(trade$symbol))} stocks."))

    stats_mean <- data.frame(
      r = c(mean(trade$r), sd(trade$r)),
      t = c(mean(trade$t), sd(trade$t)),
      t_cal = c(
        mean(as.numeric(trade$sell - trade$buy)),
        sd(as.numeric(trade$sell - trade$buy))
      ),
      row.names = c("mean", "sd")
    )
    stats_q <- data.frame(
      r = quantile(trade$r),
      t = quantile(trade$t),
      t_cal = quantile(as.numeric(trade$sell - trade$buy))
    )
    stats <- rbind(stats_q, stats_mean)
    stats$r <- format(round(stats$r, 3), nsmall = 3)
    stats[, c("t", "t_cal")] <- round(stats[, c("t", "t_cal")])

    v <- seq_len(nrow(stats_q))
    stats <- rbind(
      stats[v, ],
      setNames(
        data.frame(t(replicate(ncol(stats), "")), row.names = ""), names(stats)
      ),
      stats[-v, ]
    )
    print(stats)

    hist <- hist(trade$r, breaks = 100)
  }

  return(trade)
}
