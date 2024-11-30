query <- function(symbol = read.csv("portfolio.csv")[, "symbol"], plot = TRUE) {
  # Format arguments
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")

  # Define input
  data_list <- out0[["data_list"]]
  latest <- out0[["latest"]]

  df <- latest[latest$symbol %in% symbol, ]
  df[, sapply(df, is.numeric)] <- format(
    df[, sapply(df, is.numeric)], nsmall = 2
  )
  rownames(df) <- seq_len(nrow(df))
  print(df)

  if (plot) {
    for (symbol in df$symbol) {
      data <- data_list[[symbol]]
      data <- data[data$date > today() - months(6), ]
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