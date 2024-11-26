get_history <- function(pattern, adjust) {
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Started get_history()."
    ), quote = FALSE
  )

  # [1]   code name
  symbol_list <- fromJSON(getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_info_a_code_name",
      .encoding = "utf-8"
    )
  )
  symbol_list <- symbol_list[, 1]
  writeLines(symbol_list, "symbol_list.txt")
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Found ", length(symbol_list), " stocks;",
      " wrote to symbol_list.txt."
    ), quote = FALSE
  )

  symbol_list <- symbol_list[grepl(pattern, symbol_list)]
  #symbol_list <- sample(symbol_list, 10) # For testing only
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Matched ", length(symbol_list), " stocks to ", pattern, "."
    ), quote = FALSE
  )

  dir.create(paste0("data_", adjust))

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    symbol = symbol_list,
    .combine = "c",
    .errorhandling = "remove",
    .packages = c("jsonlite", "RCurl", "tidyverse")
  ) %dopar% {
    if (adjust == "qfq") {
      # [1]   日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅
      # [11]  涨跌额 换手率
      data <- fromJSON(getForm(
          uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
          symbol = symbol,
          adjust = adjust,
          start_date = format(
            now(tzone = "Asia/Shanghai") - years(1), "%Y%m%d"
          ),
          .encoding = "utf-8"
        )
      )
      data <- data[, c(1, 2, 5, 6, 4, 7)]
      colnames(data) <- c("date", "symbol", "high", "low", "close", "volume")
      write.csv(
        data, paste0("data_", adjust, "/", symbol, ".csv"), row.names = FALSE
      )

      i <- 1
    } else {
      out <- try(
        data <- read.csv(
          paste0("data_", adjust, "/", symbol, ".csv"),
          colClasses = c(date = "Date", symbol = "character")
        ),
        silent = TRUE
      )
      if (class(out) == "try-error") {
        data <- data.frame(matrix(nrow = 0, ncol = 6))
        start_date <- ""
      } else {
        start_date <- format(data[nrow(data), 1] + days(1), "%Y%m%d")
      }

      # [1]   日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅
      # [11]  涨跌额 换手率
      latest <- fromJSON(getForm(
          uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
          symbol = symbol,
          adjust = adjust,
          start_date = start_date,
          end_date = format(now(tzone = "Asia/Shanghai") - days(1), "%Y%m%d"),
          .encoding = "utf-8"
        )
      )

      if (length(latest) != 0) {
        data <- rbind(data, latest[, c(1, 2, 5, 6, 4, 7)])
        colnames(data) <- c("date", "symbol", "high", "low", "close", "volume")
        i <- 1
      } else {
        i <- 0
      }
    }

    return(i)
  }
  unregister_dopar

  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Updated ", sum(out), " local files."
    ), quote = FALSE
  )
}