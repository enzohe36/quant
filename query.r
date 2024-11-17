print(paste0(
    format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
    " Started query()."
  ), quote = FALSE
)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
out <- foreach(
  symbol = q_list,
  .combine = rbind
) %dopar% {
  return(update[update$symbol == symbol, ])
}
unregister_dopar

out <- out[order(out$score, decreasing = TRUE), ]
out[, 4:10] <- format(round(out[, 4:10], 2), nsmall = 2)
writeLines(c("", capture.output(print(out, row.names = FALSE))))

for (symbol in out[, 2]) {
  data <- data_list[[symbol]]
  data <- data[data$date > now(tzone = "Asia/Shanghai") - months(6), ]
  plot(
    data$date,
    2 * normalize(data$close) - 1,
    type = "l",
    xlab = "",
    ylab = "",
    main = paste0(symbol, " ", update[update$symbol == symbol, "name"]),
    ylim = c(-1, 1)
  )
  lines(data$date, data$x, col = "red")
  legend(
    x = "topleft", legend = c("close", "x"), fill = c("black", "red")
  )
}
