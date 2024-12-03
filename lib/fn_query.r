query <- function(
  ..., plot = TRUE,
  data_list = out0[["data_list"]],
  latest = out0[["latest"]],
  portfolio_path = "assets/portfolio.csv"
) {
  if(length(c(...)) != 0) {
    symbol_list <- formatC(
      as.integer(c(...)), width = 6, format = "d", flag = "0"
    )
  } else {
    portfolio <- read.csv(
      portfolio_path,
      colClasses = c(date = "Date", symbol = "character")
    )
    symbol_list <- portfolio[, "symbol"]
  }

  df <- latest[latest$symbol %in% symbol_list, ]
  df[, sapply(df, is.numeric)] <- format(
    df[, sapply(df, is.numeric)], nsmall = 2
  )
  rownames(df) <- seq_len(nrow(df))
  print(df)

  if (plot) {
    for (symbol in df$symbol) {
      data <- data_list[[symbol]] %>% .[.$date > today() - months(6), ]
      plot(
        data$date, 2 * normalize(data$close) - 1,
        type = "l",
        ylim = c(-1, 1),
        main = paste0(symbol, " ", latest[latest$symbol == symbol, "name"]),
        xlab = "", ylab = ""
      )
      lines(data$date, data$x, col = "red")
      legend(
        "topleft", legend = c("close", "x"), fill = c("black", "red")
      )
    }
  }
}