print(paste0(
    format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
    " Started load_history()."
  ), quote = FALSE
)

symbol_list <- readLines("symbol_list.txt")

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
data_list <- foreach(
  symbol = symbol_list,
  .combine = append
) %dopar% {
  # [1]   date symbol high low close volume
  data <- read.csv(paste0("data/", symbol, ".csv"))
  data[, 1] <- as.Date(data[, 1])
  data[, 2] <- formatC(data$symbol, width = 6, format = "d", flag = "0")

  lst <- list()
  lst[[symbol]] <- data
  return(lst)
}
unregister_dopar

print(paste0(
    format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
    " Loaded ", length(data_list), " stocks."
  ), quote = FALSE
)

return(list(symbol_list, data_list))
