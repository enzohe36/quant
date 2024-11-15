source("/Users/anzhouhe/Documents/quant/load_preset.r", encoding = "UTF-8")

# Assign input values
pattern <- "^(00|60)"
t <- 5

# ------------------------------------------------------------------------------

# [1]   code name
symbol_list <- fromJSON(getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_info_a_code_name",
    .encoding = "utf-8"
  )
)
symbol_list <- symbol_list[, 1]
print(paste0(
    format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
    " Found ", length(symbol_list), " stock(s)."
  )
)

symbol_list <- symbol_list[grepl(pattern, symbol_list)]
symbol_list <- sample(symbol_list, 10) # For testing only
print(paste0(
    format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
    " Matched ", length(symbol_list), " stock(s) to ", pattern, "."
  )
)

dir.create("data")

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
symbol_list <- foreach(
  symbol = symbol_list,
  .combine = "c",
  .errorhandling = "remove",
  .packages = c("jsonlite", "RCurl", "tidyverse")
) %dopar% {
  # [1]   日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅
  # [11]  涨跌额 换手率
  data <- fromJSON(getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
      symbol = symbol,
      adjust = "qfq",
      start_date = format(now(tzone = "Asia/Shanghai") - years(t), "%Y%m%d"),
      .encoding = "utf-8"
    )
  )
  data <- data[, c(1, 2, 5, 6, 4, 7)]
  colnames(data) <- c("date", "symbol", "high", "low", "close", "volume")
  data[, 1] <- as.Date(data[, 1])

  # Skip stock with < 5 yr of history
  if (data[1, 1] > now(tzone = "Asia/Shanghai") - years(t) + days(2)) {
    next
  }

  write.csv(data, paste0("data/", symbol, ".csv"), row.names = FALSE)

  return(symbol)
}
unregister_dopar

writeLines(symbol_list, "symbol_list.txt")
print(paste0(
    format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
    " ", length(symbol_list), " stock(s) have ≥ ", t, " years of history;",
    " wrote to data/."
  )
)
