sample_apy <- function(
  .trade = trade, n_portfolio, t, n_sample
) {
  start_date_list <- filter(.trade, buy <= last(sell) %m-% years(t)) %>%
    .$buy %>%
    unique() %>%
    sample(n_sample, replace = TRUE)

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    start_date = start_date_list,
    .combine = multiout,
    .multicombine = TRUE,
    .init = list(list(), list()),
    .packages = "tidyverse"
  ) %dopar% {
    rm(
      "end_date", "i", "portfolio", "portfolio_rem", "r",
      "trade_t", "trade_x", "x"
    )

    end_date <- start_date %m+% years(1)

    portfolio <- filter(
      .trade, buy < start_date & sell >= start_date & sell <= end_date
    ) %>%
      slice_sample(n = min(nrow(.), n_portfolio)) %>%
      data.matrix()

    trade_t <- filter(.trade, buy >= start_date & buy <= end_date) %>%
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

    return(list(start_date, r / n_portfolio / t))
  }
  unregister_dopar

  apy <- data.frame(
    date = as.Date(unlist(out[[1]])), apy = unlist(out[[2]])
  ) %>%
    na.omit %>%
    .[order(.$date), ]
  return(apy)
}
