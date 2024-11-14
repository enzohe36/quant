# python -m aktools
# (Source this script)
# get_history()
# ll <- load_history() # symbol_list, data_list
# ll <- get_update(ll[[1]], ll[[2]]) # symbol_list, data_list, data_latest
# query(ll[[3]])
# query_plot(ll[[2]])

library(jsonlite)
library(RCurl)
library(tidyverse)
library(TTR)
library(foreach)
library(doParallel)

options(warn = -1)

unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list = ls(name = env), pos = env)
}

normalize <- function(x) {
  return(
    (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  )
}

normalize0 <- function(x) {
  return(
    (0 - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  )
}

tnormalize <- function(x, t) {
  df <- foreach(
    i = 0:(t - 1),
    .combine = cbind,
    .packages = c("dplyr")
  ) %dopar% {
    return(lag(x, i))
  }

  return(
    (x - apply(df, 1, min)) / (apply(df, 1, max) - apply(df, 1, min))
  )
}

# ------------------------------------------------------------------------------

get_history <- function(pattern = "^(00|60)", t = 5) {
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
    .packages = c("jsonlite", "RCurl", "lubridate")
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
}

load_history <- function() {
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

    list <- list()
    list[[symbol]] <- data
    return(list)
  }
  unregister_dopar
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Loaded ", length(data_list), " stock(s)."
    )
  )

  return(list(symbol_list, data_list))
}

get_update <- function(
  symbol_list, data_list, t_r = 20, t_dx = 60, t_cci = 60
) {
  # [1]   序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高
  # [11]  最低 今开 昨收 量比 换手率 市盈率-动态 市净率 总市值 流通市值 涨速
  # [21]  5分钟涨跌 60日涨跌幅 年初至今涨跌幅 date
  data_update <- fromJSON(getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_a_spot_em",
      .encoding = "utf-8"
    )
  )
  data_update <- mutate(data_update, date = date(now(tzone = "Asia/Shanghai")))
  data_update <- data_update[, c(24, 2, 10, 11, 4, 7)]
  colnames(data_update) <- c("date", "symbol", "high", "low", "close", "volume")
  data_update <- data_update[data_update$symbol %in% symbol_list, ]
  data_update <- na.omit(data_update)
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Found update for ", nrow(data_update), " stock(s)."
    )
  )

  update <- function(df) {
    if (df[nrow(df), 1] == data_update[1, 1]) {
      df[nrow(df), ] <- data_update[data_update$symbol == df[1, 2], ]
    } else {
      df <- rbind(df, data_update[data_update$symbol == df[1, 2], ])
    }

    # Calculate modified DX
    df$dx <- 1 - tnormalize(
      (ADX(df[, 3:5])[, 1] - ADX(df[, 3:5])[, 2]) /
        (ADX(df[, 3:5])[, 1] + ADX(df[, 3:5])[, 2]),
      t_dx
    )

    # Calculate modified CCI
    df$cci <- 1 - tnormalize(CCI(df[, 3:5]), t_cci)

    # Calculate median return
    r <- foreach(
      i = 1:t_r,
      .combine = cbind,
      .packages = c("dplyr")
    ) %dopar% {
      return((lead(df$close, i) - df$close) / df$close)
    }
    df$r_med <- apply(r, 1, median)

    return(df)
  }
  data_list <- lapply(data_list, update)

  fundflow_dict <- data.frame(
    indicator = c("今日", "3日", "5日", "10日"),
    header = c("if1", "if3", "if5", "if10")
  )
  fundflow_list <- foreach(
    i = fundflow_dict[, 1],
    .packages = c("jsonlite", "RCurl")
  ) %dopar% {
    # [1]   序号 代码 名称
    # [4]   最新价 今日涨跌幅 今日主力净流入-净额
    # [7]   今日主力净流入-净占比 今日超大单净流入-净额 今日超大单净流入-净占比
    # [10]  今日大单净流入-净额 今日大单净流入-净占比 今日中单净流入-净额
    # [13]  今日中单净流入-净占比 今日小单净流入-净额 今日小单净流入-净占比
    fundflow <- fromJSON(getForm(
        uri = "http://127.0.0.1:8080/api/public/stock_individual_fund_flow_rank",
        indicator = i,
        .encoding = "utf-8"
      )
    )
    fundflow <- fundflow[, c(2, 3, 6)]
    colnames(fundflow) <- c(
      "symbol", "name", fundflow_dict[fundflow_dict$indicator == i, 2]
    )
    fundflow <- fundflow[fundflow$symbol %in% symbol_list, ]
    fundflow[, 3] <- as.numeric(fundflow[, 3])

    return(fundflow)
  }
  fundflow <- reduce(fundflow_list, full_join, by = join_by("symbol", "name"))
  fundflow[, 3:6] <- fundflow[, 3:6] / abs(fundflow[, 3])

  get_latest <- function(df) {merge(df[nrow(df), ], fundflow, by = "symbol")}
  data_latest <- bind_rows(lapply(data_list, get_latest))

  # [1]   symbol date high low close volume dx cci r_med name
  # [11]  if1 if3 if5 if10
  data_latest <- data_latest[, c(2, 1, 10, 7, 8, 11:14)]
  data_latest$score <- data_latest$dx + data_latest$cci +
    rowSums(data_latest[, 6:9] > 0) / 4
  data_latest[, 4:10] <- round(data_latest[, 4:10], 2)
  data_latest <- data_latest[order(data_latest$score, decreasing = TRUE), ]
  cat(
    capture.output(print(data_latest, row.names = FALSE)),
    file = "ranking.txt",
    sep = "\n"
  )
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Ranked ", nrow(data_latest), " stock(s);",
      " wrote to ranking.txt."
    )
  )
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " 小优选股助手建议您购买 ", data_latest[1, 3], " 哦！"
    )
  )

  return(list(symbol_list, data_list, data_latest))
}

query <- function(data_latest, query = readLines("query.txt")) {
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  data_query <- foreach(
    symbol = query,
    .combine = rbind
  ) %dopar% {
    return(data_latest[data_latest$symbol == symbol, ])
  }
  unregister_dopar

  data_query <- data_query[order(data_query$score, decreasing = TRUE), ]
  cat(
    capture.output(print(data_query, row.names = FALSE)),
    file = "ranking_query.txt",
    sep = "\n"
  )
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Queried ", nrow(data_query), " stock(s);",
      " wrote to ranking_query.txt"
    )
  )
}

query_plot <- function(data_list, query = readLines("query.txt")) {
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
}
