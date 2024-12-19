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

  portfolio <- filter(portfolio, !symbol %in% symbol_list) %>%
    mutate(cost = format(round(cost, 3), nsmall = 3)) %>%
    `rownames<-`(seq_len(nrow(.)))
  write.csv(portfolio, portfolio_path, quote = FALSE, row.names = FALSE)
  print(portfolio)
}
