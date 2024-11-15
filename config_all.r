# python -m aktools
# (Source this script)
# get_history()
# ll <- load_history() # symbol_list, data_list
# ll <- get_update(ll[[1]], ll[[2]]) # symbol_list, data_list, data_latest
# query()
# query_plot()

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
  df <- data.frame(matrix(nrow = length(x), ncol = 0))
  for (i in 1:t) {
    df[, i] <- lag(x, i - 1)
  }

  return(
    (x - apply(df, 1, min)) / (apply(df, 1, max) - apply(df, 1, min))
  )
}

adx_alt <- function(hlc, n = 14, m = 6) {
  h <- hlc[, 1]
  l <- hlc[, 2]
  c <- hlc[, 3]
  tr <- runSum(
    apply(cbind(h - l, abs(h - lag(c, 1)), abs(l - lag(c, 1))), 1, max), n
  )
  dh <- h - lag(h, 1)
  dl <- lag(l, 1) - l
  dmp <- runSum(ifelse(dh > 0 & dh > dl, dh, 0), n)
  dmn <- runSum(ifelse(dl > 0 & dl > dh, dl, 0), n)
  dip <- dmp / tr
  din <- dmn / tr
  adx <- SMA(abs(dip - din) / (dip + din), m)
  adxr <- (adx + lag(adx, m)) / 2
  out <- cbind(adx, adxr)
  colnames(out) <- c("adx", "adxr")
  return(out)
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
  #symbol_list <- sample(symbol_list, 10) # For testing only
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
  symbol_list, data_list, t_adx = 60, t_cci = 60, t_r = 20
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

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  data_list <- foreach(
    symbol = symbol_list,
    .combine = append,
    .export = c("tnormalize", "adx_alt"),
    .packages = c("tidyverse", "TTR")
  ) %dopar% {
    data <- data_list[[symbol]]
    if (data[nrow(data), 1] == data_update[1, 1]) {
      data[nrow(data), ] <- data_update[data_update$symbol == data[1, 2], ]
    } else {
      data <- rbind(data, data_update[data_update$symbol == data[1, 2], ])
    }

    # Calculate predictor
    dmi_mod <- 1 - tnormalize(
      abs(adx_alt(data[, 3:5])[, 1] - adx_alt(data[, 3:5])[, 2]), t_adx
    )
    cci_mod <- 1 - 2 * tnormalize(CCI(data[, 3:5]), t_cci)
    data$x <- dmi_mod * cci_mod

    # Calculate return
    df <- data.frame(matrix(nrow = nrow(data), ncol = 0))
    for (i in 1:t_r) {
      df[, i] <- (lead(data$close, i) - data$close) / data$close
    }
    data$r_max <- apply(df, 1, max)
    data$r_min <- apply(df, 1, min)

    list <- list()
    list[[symbol]] <- data
    return(list)
  }
  unregister_dopar

  fundflow_dict <- data.frame(
    indicator = c("今日", "3日", "5日", "10日"),
    header = c("inflow1", "inflow3", "inflow5", "inflow10")
  )

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
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
  unregister_dopar

  fundflow <- reduce(fundflow_list, full_join, by = join_by("symbol", "name"))
  fundflow[, 3:6] <- fundflow[, 3:6] / abs(fundflow[, 3])

  get_latest <- function(df) merge(df[nrow(df), ], fundflow, by = "symbol")
  data_latest <- bind_rows(lapply(data_list, get_latest))

  # [1]   symbol date high low close volume x r_max r_min name
  # [2]   inflow1 inflow3 inflow5 inflow10
  data_latest <- data_latest[, c(2, 1, 10, 7, 11:14)]
  weight_inflow <- function(x) {
    return(
      ifelse(x[1] > 0, 0.4, 0) + ifelse(x[2] > 0, 0.3, 0) +
        ifelse(x[3] > 0, 0.2, 0) + ifelse(x[4] > 0, 0.1, 0)
    )
  }
  data_latest$score <- data_latest$x + apply(data_latest[, 5:8], 1, weight_inflow)
  data_latest <- data_latest[order(data_latest$score, decreasing = TRUE), ]
  data_latest[, 4:9] <- format(round(data_latest[, 4:9], 2), nsmall = 2)
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

query <- function(query = readLines("query.txt"), data_latest = ll[[3]]) {
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
  print(data_query, row.names = FALSE)
}

query_plot <- function(query = readLines("query.txt"), data_list = ll[[2]]) {
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
}
