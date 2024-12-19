sample_apy <- function(
  trade, n_portfolio, t_apy, n_sample, end_date
) {
  end_date <- ymd(end_date)

  start_date_list <- filter(trade, buy <= end_date %m-% years(t_apy)) %>%
    select(buy) %>%
    pull() %>%
    unique() %>%
    sample(n_sample, replace = TRUE)

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    start_date = start_date_list,
    .packages = "tidyverse"
  ) %dopar% {
    rm("i", "portfolio", "portfolio_rem", "r", "trade_t", "trade_x", "x")

    end_date <- start_date %m+% years(1)

    portfolio <- filter(trade, buy < start_date & sell >= start_date) %>%
      slice_sample(n = min(nrow(.), n_portfolio)) %>%
      data.matrix()

    trade_t <- filter(trade, buy >= start_date & buy <= end_date) %>%
      slice_sample(n = nrow(.)) %>%
      data.matrix()

    r <- 0
    for (x in start_date:end_date) {
      i <- which(portfolio[, "sell"] == x)
      r <- sum(r, portfolio[i, "r"], na.rm = TRUE)
      portfolio <- portfolio[!seq_len(nrow(portfolio)) %in% i, , drop = FALSE]
      portfolio_rem <- n_portfolio - nrow(portfolio)
      if (portfolio_rem > 0) {
        trade_x <- trade_t[trade_t[, "buy"] == x, , drop = FALSE] %>%
          .[seq_len(min(nrow(.), portfolio_rem)), ]
        portfolio <- rbind(portfolio, trade_x)
      }
    }

    return(list(date = start_date, apy = r / n_portfolio / t_apy))
  }
  unregister_dopar

  apy <- rbindlist(out) %>%
    na.omit %>%
    arrange(date)
  return(apy)
}
