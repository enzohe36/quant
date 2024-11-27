buy <- function(symbol, cost = NA, buy = NA) {
  # Define parameters
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")
  try(buy <- ymd(buy), silent = TRUE)
  latest <- out0[[3]]

  portfolio <- read.csv(
    "portfolio.csv",
    colClasses = c(symbol = "character", buy = "Date", cost = "numeric")
  )

  out <- portfolio[portfolio$symbol == symbol, ]
  if (nrow(out) == 0) {
    portfolio <- rbind(
      portfolio,
      list(
        symbol,
        latest[latest$symbol == symbol, "name"],
        ifelse(is.na(buy), date(now(tzone = "Asia/Shanghai")), buy),
        cost
      )
    )
    colnames(portfolio) <- c("symbol", "name", "buy", "cost")
    portfolio$buy <- as.Date(portfolio$buy)
  } else {
    portfolio[portfolio$symbol == symbol, ] <- list(
      symbol,
      latest[latest$symbol == symbol, "name"],
      ifelse(is.na(buy), portfolio[portfolio$symbol == symbol, "buy"], buy),
      ifelse(is.na(cost), portfolio[portfolio$symbol == symbol, "cost"], cost)
    )
  }

  portfolio <- portfolio[order(portfolio$symbol), ]
  portfolio$cost <- format(round(portfolio$cost, 3), nsmall = 3)
  write.csv(portfolio, "portfolio.csv", quote = FALSE, row.names = FALSE)
  print(portfolio, row.names = FALSE)
}

sell <- function(symbol = read.csv("portfolio.csv")[, "symbol"]) {
  # Define parameters
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")

  portfolio <- read.csv(
    "portfolio.csv",
    colClasses = c(symbol = "character", buy = "Date", cost = "numeric")
  )

  portfolio <- portfolio[portfolio$symbol != symbol, ]
  portfolio$cost <- format(round(portfolio$cost, 3), nsmall = 3)
  write.csv(portfolio, "portfolio.csv", quote = FALSE, row.names = FALSE)
  print(portfolio, row.names = FALSE)
}
