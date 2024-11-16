source("load_preset.r", encoding = "UTF-8")

# Define parameters
query <- readLines("query.txt")
data_list <- data_list

# ------------------------------------------------------------------------------

for (symbol in query) {
  data <- data_list[[symbol]]
  data <- data[data$date > now(tzone = "Asia/Shanghai") - months(6), ]
  plot(
    data$date, 2 * normalize(data$close) - 1, type = "l",
    xlab = "", ylab = "", main = unique(data$symbol), ylim = c(-1, 1)
  )
  abline(h = normalize0(data$close), col = "grey")
  lines(data$date, data$x, col = "red")
}
