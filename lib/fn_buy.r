buy <- function(
  symbol, cost = NaN, buy = NA,
  latest = get("latest", envir = .GlobalEnv),
  portfolio_path = "assets/portfolio.csv"
) {
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")
  buy <- ymd(buy)

  if (file.exists(portfolio_path)) {
    portfolio <- read.csv(
      portfolio_path,
      colClasses = c(buy = "Date", symbol = "character")
    )
  } else {
    portfolio <- data.frame(matrix(ncol = 4)) %>%
      `colnames<-`(c("buy", "symbol", "name", "cost")) %>%
      na.omit()
    write.csv(portfolio, portfolio_path, quote = FALSE, row.names = FALSE)
  }

  portfolio_i <- portfolio[portfolio$symbol == symbol, ]
  if (nrow(portfolio_i) == 0) {
    portfolio <- rbind(
      portfolio,
      list(
        buy = ifelse(is.na(buy), bizday(), buy),
        symbol = symbol,
        name = latest[latest$symbol == symbol, "name"],
        cost = cost
      )
    ) %>%
      mutate(buy = as_date(buy))
  } else {
    portfolio[portfolio$symbol == symbol, ] <- list(
      buy = ifelse(is.na(buy), portfolio_i$buy, buy),
      symbol = symbol,
      name = latest[latest$symbol == symbol, "name"],
      cost = ifelse(is.na(cost), portfolio_i$cost, cost)
    )
  }
  portfolio <- arrange(portfolio, symbol) %>%
    mutate(cost = format(round(cost, 3), nsmall = 3)) %>%
    `rownames<-`(seq_len(nrow(.)))
  write.csv(portfolio, portfolio_path, quote = FALSE, row.names = FALSE)
  print(portfolio)
}
