sell <- function(
  ...,
  portfolio_path = "assets/portfolio.csv"
) {
  portfolio <- read.csv(
    portfolio_path, colClasses = c(date = "Date", symbol = "character")
  )

  if(length(c(...)) != 0) {
    symbol_list <- formatC(
      as.integer(c(...)), width = 6, format = "d", flag = "0"
    )
  } else {
    symbol_list <- portfolio[, "symbol"]
  }

  portfolio <- portfolio[!(portfolio$symbol %in% symbol_list), ]
  portfolio$cost <- format(round(portfolio$cost, 3), nsmall = 3)
  write.csv(portfolio, portfolio_path, quote = FALSE, row.names = FALSE)

  rownames(portfolio) <- seq_len(nrow(portfolio))
  print(portfolio)
}
