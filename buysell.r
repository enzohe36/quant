buy <- function(symbol, cost = NaN, date = NA) {
  # Format arguments
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")
  try(date <- ymd(date), silent = TRUE)

  # Define input
  latest <- out0[["latest"]]
  portfolio <- read.csv(
    "portfolio.csv",
    colClasses = c(date = "Date", symbol = "character")
  )

  df <- portfolio[portfolio$symbol == symbol, ]
  if (nrow(df) == 0) {
    portfolio <- rbind(
      portfolio,
      list(
        ifelse(is.na(date), today(), date),
        symbol,
        latest[latest$symbol == symbol, "name"],
        cost
      )
    )
    colnames(portfolio) <- c("date", "symbol", "name", "cost")
    portfolio$date <- as.Date(portfolio$date)
  } else {
    portfolio[portfolio$symbol == symbol, ] <- list(
      ifelse(is.na(date), df$date, date),
      symbol,
      latest[latest$symbol == symbol, "name"],
      ifelse(is.na(cost), df$cost, cost)
    )
  }
  portfolio <- portfolio[order(portfolio$symbol), ]
  portfolio$cost <- format(round(portfolio$cost, 3), nsmall = 3)
  write.csv(portfolio, "portfolio.csv", quote = FALSE, row.names = FALSE)
  rownames(portfolio) <- seq_len(nrow(portfolio))
  print(portfolio)
}

sell <- function(...) {
  # Format arguments
  symbol_list <- c(...)

  # Define input
  portfolio <- read.csv(
    "portfolio.csv",
    colClasses = c(date = "Date", symbol = "character")
  )

  if(length(symbol_list) != 0) {
    symbol_list <- formatC(
      as.integer(symbol_list), width = 6, format = "d", flag = "0"
    )
  } else {
    symbol_list <- portfolio[, "symbol"]
  }

  portfolio <- portfolio[!(portfolio$symbol %in% symbol_list), ]
  portfolio$cost <- format(round(portfolio$cost, 3), nsmall = 3)
  write.csv(portfolio, "portfolio.csv", quote = FALSE, row.names = FALSE)
  rownames(portfolio) <- seq_len(nrow(portfolio))
  print(portfolio)
}
