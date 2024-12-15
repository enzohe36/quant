buy <- function(
  symbol, cost = NaN, buy = NA,
  .latest = latest,
  portfolio_path = "assets/portfolio.csv"
) {
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")
  try(buy <- ymd(buy), silent = TRUE)

  if (!file.exists(portfolio_path)) {
    portfolio <- data.frame(matrix(ncol = 4)) %>%
      `colnames<-`(c("buy", "symbol", "name", "cost")) %>%
      na.omit()
    write.csv(portfolio, portfolio_path, quote = FALSE, row.names = FALSE)
  } else {
    portfolio <- read.csv(
      portfolio_path,
      colClasses = c(buy = "Date", symbol = "character")
    )
  }

  df <- portfolio[portfolio$symbol == symbol, ]
  if (nrow(df) == 0) {
    portfolio <- rbind(
      portfolio,
      list(
        buy = ifelse(is.na(buy), bizday(), buy),
        symbol = symbol,
        name = .latest[.latest$symbol == symbol, "name"],
        cost = cost
      )
    ) %>%
      mutate(buy = as_date(buy))
  } else {
    portfolio[portfolio$symbol == symbol, ] <- list(
      ifelse(is.na(buy), df$buy, buy),
      symbol,
      .latest[.latest$symbol == symbol, "name"],
      ifelse(is.na(cost), df$cost, cost)
    )
  }
  portfolio <- arrange(portfolio, symbol) %>%
    mutate(cost = format(round(cost, 3), nsmall = 3)) %>%
    `rownames<-`(seq_len(nrow(.)))
  write.csv(portfolio, portfolio_path, quote = FALSE, row.names = FALSE)
  print(portfolio)
}
