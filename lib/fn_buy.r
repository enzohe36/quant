buy <- function(
  symbol, cost = NaN, date = NA,
  latest = get("latest", envir = .GlobalEnv),
  portfolio_path = "assets/portfolio.csv"
) {
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")
  date <- ymd(date)

  if (file.exists(portfolio_path)) {
    portfolio <- read.csv(
      portfolio_path,
      colClasses = c(date = "Date", symbol = "character")
    )
  } else {
    portfolio <- data.frame(matrix(ncol = 4)) %>%
      `colnames<-`(c("date", "symbol", "name", "cost")) %>%
      na.omit()
    write.csv(portfolio, portfolio_path, quote = FALSE, row.names = FALSE)
  }

  portfolio_i <- portfolio[portfolio$symbol == symbol, ]
  if (nrow(portfolio_i) == 0) {
    portfolio <- rbind(
      portfolio,
      list(
        date = ifelse(is.na(date), as_tdate(today()), date),
        symbol = symbol,
        name = latest[latest$symbol == symbol, "name"],
        cost = cost
      )
    ) %>%
      mutate(date = as_date(date))
  } else {
    portfolio[portfolio$symbol == symbol, ] <- list(
      date = ifelse(is.na(date), portfolio_i$date, date),
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
