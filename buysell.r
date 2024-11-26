buy <- function(symbol, cost = NA, date = NA) {
  # Define parameters
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")
  try(date <- ymd(date), silent = TRUE)
  latest <- out0[[3]]

  portfolio <- read.csv(
    "portfolio.csv",
    colClasses = c(symbol = "character", cost = "numeric", date = "Date")
  )

  out <- portfolio[portfolio$symbol == symbol, ]
  if (nrow(out) == 0) {
    portfolio <- rbind(portfolio, list(
        symbol,
        latest[latest$symbol == symbol, "name"],
        cost,
        ifelse(is.na(date), date(now(tzone = "Asia/Shanghai")), date)
      )
    )
    colnames(portfolio) <- c("symbol", "name", "cost", "date")
    portfolio$date <- as.Date(portfolio$date)
  } else {
    portfolio[portfolio$symbol == symbol, ] <- list(
      symbol,
      latest[latest$symbol == symbol, "name"],
      ifelse(is.na(cost), portfolio[portfolio$symbol == symbol, "cost"], cost),
      ifelse(is.na(date), portfolio[portfolio$symbol == symbol, "date"], date)
    )
  }

  portfolio <- portfolio[order(portfolio$symbol), ]
  portfolio$cost <- format(round(portfolio$cost, 3), nsmall = 3)
  write.csv(
    portfolio, "portfolio.csv", row.names = FALSE, quote = FALSE
  )
  print(portfolio, row.names = FALSE)
}

sell <- function(symbol = read.csv("portfolio.csv")[, "symbol"]) {
  # Define parameters
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")

  portfolio <- read.csv(
    "portfolio.csv",
    colClasses = c(date = "Date", symbol = "character", cost = "numeric")
  )
  portfolio <- portfolio[portfolio$symbol != symbol, ]
  portfolio$cost <- format(round(portfolio$cost, 3), nsmall = 3)
  write.csv(
    portfolio, "portfolio.csv", row.names = FALSE, quote = FALSE
  )
  print(portfolio, row.names = FALSE)
}
