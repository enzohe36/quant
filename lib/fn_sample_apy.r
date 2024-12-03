sample_apy <- function(
  n_portfolio, t, n_sample,
  trade = out0[["trade"]]
) {
  tsprint("Started sample_apy().")

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    i = seq_len(n_sample),
    .combine = multiout,
    .multicombine = TRUE,
    .init = list(list(), list()),
    .packages = "tidyverse"
  ) %dopar% {
    rm("date", "j", "portfolio", "r", "trade_tr")

    trade <- trade[sample(seq_len(nrow(trade))), ] %>% .[order(.$buy), ]
    date <- unique(trade$buy) %>%
      .[. <= .[length(.)] - years(t)] %>%
      sample(., 1)
    trade_tr <- trade[
      trade$buy >= date & trade$sell <= date + years(t),
    ]

    portfolio <- data.frame()
    r <- 0
    for (j in seq_len(nrow(trade_tr))) {
      r <- sum(r, portfolio[portfolio$sell == trade_tr[j, "buy"], "r"])
      portfolio <- portfolio[portfolio$sell != trade_tr[j, "buy"], ]
      if (n_portfolio - nrow(portfolio) > 0) {
        portfolio <- rbind(portfolio, trade_tr[j, ])
      }
    }

    return(list(date, r / n_portfolio / t))
  }
  unregister_dopar

  tsprint(
    glue(
      "Sampled a {n_portfolio}-stock portfolio over {t} year(s) {n_sample} times."
    )
  )

  df <- data.frame(date = as.Date(unlist(out[[1]])), apy = unlist(out[[2]])) %>%
    na.omit %>%
    .[order(.$date), ]
  return(df)
}
