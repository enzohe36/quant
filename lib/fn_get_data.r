get_data <- function(
  pattern, adjust,
  symbol_list_path = "assets/symbol_list.txt",
  data_dir = paste0("data_", adjust, "/"),
  data_path_expr = expression(paste0(data_dir, symbol, ".csv"))
) {
  tsprint("Started get_data().")

  # [1]   code name
  symbol_list <- fromJSON(
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_info_a_code_name",
      .encoding = "utf-8"
    )
  )
  symbol_list <- symbol_list[, 1]
  writeLines(symbol_list, symbol_list_path)
  tsprint(
    glue("Found {length(symbol_list)} stocks; wrote to {symbol_list_path}.")
  )

  symbol_list <- symbol_list[grepl(pattern, symbol_list)]
  tsprint(glue("Matched {length(symbol_list)} stocks to \"pattern\"."))

  dir.create(data_dir)

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    symbol = symbol_list,
    .combine = "c",
    .export = "data_dir",
    .packages = c("jsonlite", "RCurl", "tidyverse")
  ) %dopar% {
    data_path <- eval(data_path_expr)

    if (adjust == "qfq") {
      # [1]   日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅
      # [11]  涨跌额 换手率
      data <- fromJSON(
        getForm(
          uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
          symbol = symbol,
          adjust = adjust,
          start_date = format(today() - years(1), "%Y%m%d"),
          .encoding = "utf-8"
        )
      )
      data <- data[, c(1, 2, 5, 6, 4, 7)]
      colnames(data) <- c("date", "symbol", "high", "low", "close", "volume")
      data$date <- as.Date(data$date)
      write.csv(data, data_path, quote = FALSE, row.names = FALSE)
      return(1)
    }

    error <- try(
      data <- read.csv(
        data_path, colClasses = c(date = "Date", symbol = "character")
      ),
      silent = TRUE
    )
    if (class(error) == "try-error") {
      # [1]   日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅
      # [11]  涨跌额 换手率
      data <- fromJSON(
        getForm(
          uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
          symbol = symbol,
          adjust = adjust,
          end_date = format(today() - days(1), "%Y%m%d"),
          .encoding = "utf-8"
        )
      )
      data <- data[, c(1, 2, 5, 6, 4, 7)]
      colnames(data) <- c("date", "symbol", "high", "low", "close", "volume")
      data$date <- as.Date(data$date)
      write.csv(data, data_path, quote = FALSE, row.names = FALSE)
      return(1)
    }

    # [1]   日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅
    # [11]  涨跌额 换手率
    latest <- fromJSON(
      getForm(
        uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
        symbol = symbol,
        adjust = adjust,
        start_date = format(data[nrow(data), "date"] + days(1), "%Y%m%d"),
        end_date = format(today() - days(1), "%Y%m%d"),
        .encoding = "utf-8"
      )
    )
    if (length(latest) != 0) {
      latest <- latest[, c(1, 2, 5, 6, 4, 7)]
      latest[, 1] <- as.Date(latest[, 1])
      write.table(
        latest,
        data_path, append = TRUE,
        quote = FALSE,
        sep = ",",
        row.names = FALSE, col.names = FALSE
      )
      return(1)
    }

    return(0)
  }
  unregister_dopar

  tsprint(glue("Updated {sum(out)} local files."))
}