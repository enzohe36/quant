query <- function(
  symbol, date = NA, plot = TRUE,
  data_list = get("data_list", envir = .GlobalEnv)
) {
  symbol <- formatC(as.integer(symbol), width = 6, format = "d", flag = "0")
  date <- as_tdate(date)

  out <- data_list[[symbol]] %>% filter(date == !!date)

  if (plot) {
    data <- data_list[[symbol]] %>%
      filter(date >= !!date - months(6) & date <= !!date + months(6))

    plot(
      data$date, 2 * normalize(data$close) - 1,
      type = "l",
      ylim = c(-1, 1),
      main = glue("{symbol}: {date}"), xlab = "", ylab = ""
    )
    lines(data$date, data$xa, col = "red")
    lines(data$date, data$xb, col = "orange")
    abline(v = date, col = "blue")
    legend(
      "topleft",
      legend = c("close", "xa", "xb"),
      fill = c("black", "red", "orange")
    )
  }

  return(out)
}