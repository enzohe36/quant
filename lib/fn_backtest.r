backtest <- function(
  t_adx, t_cci, xa_thr, xb_thr, t_max, r_max, r_min, descr = TRUE,
  tb = tb0
) {
  tsprint("Started backtest().")

  tb = tb0; t0 <- now()

  core_count <- detectCores() - 1
  cl <- makeCluster(core_count)
  registerDoParallel(cl)
  tb <- foreach(
    tb_chunk = split(tb, ceiling(row_number(tb) / nrow(tb) * core_count)),
    .combine = rbind,
    .export = c(
      "normalize", "tnormalize", "ADX", "ROR", "get_predictor", "get_trade",
      "t_adx", "t_cci", "xa_thr", "xb_thr", "t_max", "r_max", "r_min"
    ),
    .packages = c("TTR", "signal", "tidyverse", "dtplyr")
  ) %dopar% {
    tb %>%
      lazy_dt() %>%
      mutate(data = lapply(data, get_predictor, t_adx, t_cci)) %>%
      mutate(
        trade = lapply(data, get_trade, xa_thr, xb_thr, t_max, r_max, r_min)
      ) %>%
      as_tibble()
  }
  unregister_dopar

  print(now() - t0); tb

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
