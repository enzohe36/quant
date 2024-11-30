sample_apy <- function(n_portfolio, t, n_sample) {
  tsprint("Started sample_apy().")

  # Define input
  trade_all <- out0[["trade"]]

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    i = 1:n_sample,
    .combine = multiout,
    .multicombine = TRUE,
    .init = list(list(), list()),
    .packages = "tidyverse"
  ) %dopar% {
    trade <- trade_all
    trade <- trade[sample(seq_len(nrow(trade))), ]
    trade <- trade[order(trade$buy), ]

    buy <- unique(trade$buy) %>%
      .[. <= .[length(.)] - years(t)] %>%
      sample(., 1)
    trade <- trade[
      trade$buy >= buy & trade$sell <= buy + years(t),
    ]

    portfolio <- data.frame()
    r <- 0
    for (j in seq_len(nrow(trade))) {
      r <- sum(r, portfolio[portfolio$sell == trade[j, "buy"], "r"])
      portfolio <- portfolio[portfolio$sell != trade[j, "buy"], ]
      if (n_portfolio - nrow(portfolio) > 0) {
        portfolio <- rbind(portfolio, trade[j, ])
      }
    }

    gc()

    return(list(buy, r / n_portfolio / t))
  }
  unregister_dopar

  tsprint(
    glue("Sampled a {n_portfolio}-stock portfolio over {t} years {n_sample} times.")
  )

  return(list(buy = as.Date(unlist(out[[1]])), apy = unlist(out[[2]])))
}
