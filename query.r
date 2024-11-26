query <- function(symbol = read.csv("portfolio.csv")[, "symbol"]) {
  # Define parameters
  data_list <- out0[[2]]
  latest <- out0[[3]]

  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")

  out <- latest[latest$symbol %in% symbol, ]
  out[, sapply(out, is.numeric)] <- format(
    out[, sapply(out, is.numeric)], nsmall = 2
  )
  print(out, row.names = FALSE)

  for (symbol in out$symbol) {
    data <- data_list[[symbol]]
    data <- data[data$date > now(tzone = "Asia/Shanghai") - months(6), ]
    plot(
      data$date,
      2 * normalize(data$close) - 1,
      type = "l",
      xlab = "",
      ylab = "",
      main = paste0(symbol, " ", latest[latest$symbol == symbol, "name"]),
      ylim = c(-1, 1)
    )
    lines(data$date, data$x, col = "red")
    legend(
      x = "topleft", legend = c("close", "x"), fill = c("black", "red")
    )
  }
}