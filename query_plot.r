source("/Users/anzhouhe/Documents/quant/load_preset.r", encoding = "UTF-8")

# Assign input values
query <- readLines("query.txt")
data_list <- data_list

# ------------------------------------------------------------------------------

for (symbol in query) {
  data <- data_list[[symbol]]
  data <- data[data$date > now(tzone = "Asia/Shanghai") - months(6), ]
  data <- data[is.finite(data$r_max) & is.finite(data$r_min), ]
  h <- max(max(data$r_max, na.rm = TRUE), max(data$r_min, na.rm = TRUE))
  l <- min(min(data$r_max, na.rm = TRUE), min(data$r_min, na.rm = TRUE))
  r_max <- 2 * (data$r_max - l) / (h - l) - 1
  r_min <- 2 * (data$r_min - l) / (h - l) - 1
  r_0 <- 2 * (0 - l) / (h - l) - 1
  plot(
    data$date, data$x, type = "l", xlab = "", ylab = "",
    main = unique(data$symbol), ylim = c(-1, 1)
  )
  abline(h = r_0, col = "grey")
  lines(data$date, r_max, col = "red")
  lines(data$date, r_min, col = "blue")
}
