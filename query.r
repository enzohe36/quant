query <- function(symbol = read.csv("portfolio.csv")[, "symbol"], plot = TRUE) {
  # Format arguments
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")

  # Define input
  data_list <- out0[[2]]
  latest <- out0[[3]]

  df <- latest[latest$symbol %in% symbol, ]
  df[, sapply(df, is.numeric)] <- format(
    df[, sapply(df, is.numeric)], nsmall = 2
  )
  print(df, row.names = FALSE)

  if (plot) {
    for (symbol in df$symbol) {
      data <- data_list[[symbol]]
      data <- data[data$date > now(tzone = "Asia/Shanghai") - months(6), ]
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