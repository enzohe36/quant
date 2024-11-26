get_history <- function(pattern, adjust) {
  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Started get_history()."
    ),
    quote = FALSE
  )

  # [1]   code name
  symbol_list <- fromJSON(getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_info_a_code_name",
      .encoding = "utf-8"
    )
  )
  symbol_list <- symbol_list[, 1]
  writeLines(symbol_list, "symbol_list.txt")
  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Found ", length(symbol_list), " stocks;",
      " wrote to symbol_list.txt."
    ),
    quote = FALSE
  )

  symbol_list <- symbol_list[grepl(pattern, symbol_list)]
  #symbol_list <- sample(symbol_list, 10) # For testing only
  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Matched ", length(symbol_list), " stocks to ", pattern, "."
    ),
    quote = FALSE
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
    file <- paste0("data_", adjust, "/", symbol, ".csv")

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
      data$date <- as.Date(data$date)
      write.csv(data, file, row.names = FALSE, quote = FALSE)

      return(1)
    }

    out <- try(
      data <- read.csv(
        file, colClasses = c(date = "Date", symbol = "character")
      ),
      silent = TRUE
    )

    if (class(out) == "try-error") {
      # [1]   日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅
      # [11]  涨跌额 换手率
      data <- fromJSON(getForm(
          uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
          symbol = symbol,
          adjust = adjust,
          end_date = format(now(tzone = "Asia/Shanghai") - days(1), "%Y%m%d"),
          .encoding = "utf-8"
        )
      )
      data <- data[, c(1, 2, 5, 6, 4, 7)]
      colnames(data) <- c("date", "symbol", "high", "low", "close", "volume")
      data$date <- as.Date(data$date)
      write.csv(data, file, row.names = FALSE, quote = FALSE)

      return(1)
    }

    # [1]   日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅
    # [11]  涨跌额 换手率
    latest <- fromJSON(getForm(
        uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
        symbol = symbol,
        adjust = adjust,
        start_date = format(data[nrow(data), "date"] + days(1), "%Y%m%d"),
        end_date = format(now(tzone = "Asia/Shanghai") - days(1), "%Y%m%d"),
        .encoding = "utf-8"
      )
    )

    if (length(latest) != 0) {
      latest <- latest[, c(1, 2, 5, 6, 4, 7)]
      latest[, 1] <- as.Date(latest[, 1])
      write.table(
        latest,
        file,
        sep = ",",
        row.names = FALSE,
        col.names = FALSE,
        append = TRUE,
        quote = FALSE
      )

      return(1)
    }

    return(0)
  }
  unregister_dopar

  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Updated ", sum(out), " local files."
    ),
    quote = FALSE
  )
}