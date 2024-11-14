source("/Users/anzhouhe/Documents/quant/load_preset.r", encoding = "UTF-8")

# Assign input values
data_list <- data_list
query <- readLines("query.txt")

# ------------------------------------------------------------------------------

for (symbol in query) {
  data <- data_list[[symbol]]
  data <- data[data$date > now(tzone = "Asia/Shanghai") - months(6), ]
  data <- data[is.finite(data$r_med), ]
  plot(
    data$date, normalize(data$close),
    type = "l", lwd = 2,
    xlab = "", ylab = "", main = unique(data$symbol)
  )
  lines(data$date, normalize(data$r_med), col = "red")
  abline(h = normalize0(data$r_med), col = "red")
  lines(data$date, data$dx, col = "blue")
  lines(data$date, data$cci, col = "green4")
}
