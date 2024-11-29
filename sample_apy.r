sample_apy <- function(n_portfolio) {
  # Define input
  trade <- trade

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  apy <- foreach(i = 1:30, .combine = append) %dopar% {
    trade <- trade[sample(seq_len(nrow(trade))), ]
    trade <- trade[order(trade$buy), ]

    portfolio <- data.frame()
    r <- 0
    for (j in seq_len(nrow(trade))) {
      r <- sum(r, portfolio[portfolio$sell == trade[j, "buy"], "r"])
      portfolio <- portfolio[portfolio$sell != trade[j, "buy"], ]
      n <- n_portfolio - nrow(portfolio)
      if (n > 0) portfolio <- rbind(portfolio, trade[j, ])
    }

    return(r / n_portfolio / as.numeric(max(trade$sell) - min(trade$buy)) * 365)
  }
  unregister_dopar

  return(c(mean(apy), sd(apy)))
}
