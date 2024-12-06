query <- function(
  symbol, date = NA, plot = TRUE,
  data_list = out0[["data_list"]],
  portfolio_path = "assets/portfolio.csv"
) {
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")
  date <- bizday(date)

  out <- data_list[[symbol]] %>% .[.$date == date, ]

  if (plot) {
    data <- data_list[[symbol]] %>%
      .[.$date >= date - months(6) & .$date <= date + months(6), ]

    plot(
      data$date, 2 * normalize(data$close) - 1,
      type = "l",
      ylim = c(-1, 1),
      main = glue("{symbol}: {date}"), xlab = "", ylab = ""
    )
    lines(data$date, data$x, col = "red")
    abline(v = date, col = "blue")
    legend(
      "topleft", legend = c("close", "x"), fill = c("black", "red")
    )
  }

  return(out)
}