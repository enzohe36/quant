buy <- function(
  symbol, cost = NaN, date = NA,
  .latest = latest,
  portfolio_path = "assets/portfolio.csv"
) {
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")
  try(date <- ymd(date), silent = TRUE)

  if (!file.exists(portfolio_path)) portfolio <- data.frame()

  portfolio <- read.csv(
    portfolio_path,
    colClasses = c(date = "Date", symbol = "character")
  )

  df <- portfolio[portfolio$symbol == symbol, ]
  if (nrow(df) == 0) {
    portfolio <- rbind(
      portfolio,
      list(
        ifelse(is.na(date), bizday(), date),
        symbol,
        .latest[.latest$symbol == symbol, "name"],
        cost
      )
    )
    colnames(portfolio) <- c("date", "symbol", "name", "cost")
    portfolio$date <- as.Date(portfolio$date)
  } else {
    portfolio[portfolio$symbol == symbol, ] <- list(
      ifelse(is.na(date), df$date, date),
      symbol,
      .latest[.latest$symbol == symbol, "name"],
      ifelse(is.na(cost), df$cost, cost)
    )
  }
  portfolio <- portfolio[order(portfolio$symbol), ]
  portfolio$cost <- format(round(portfolio$cost, 3), nsmall = 3)
  write.csv(portfolio, portfolio_path, quote = FALSE, row.names = FALSE)

  rownames(portfolio) <- seq_len(nrow(portfolio))
  print(portfolio)
}
