# python -m aktools
# (Source this script)
# get_history()
# dl <- load_history() # symbol_list, data_list
# dl <- get_update(dl[[1]], dl[[2]]) # symbol_list, data_list, data_latest
# query()
# query_plot()
# rl <- backtest(dl[[1]], dl[[2]])

library(jsonlite)
library(RCurl)
library(tidyverse)
library(TTR)
library(foreach)
library(doParallel)

options(warn = -1)

# https://stackoverflow.com/a/25110203
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

# http://www.cftsc.com/qushizhibiao/610.html
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

pct_change <- function(x1, x2) {
  return((x2 - x1) / x1)
}

# https://stackoverflow.com/a/19801108
multiout <- function(x, ...) {
  lapply(
    seq_along(x),
    function(i) c(x[[i]], lapply(list(...), function(y) y[[i]]))
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

get_update <- function(symbol_list, data_list, t_adx = 60, t_cci = 60) {
  # [1]   序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高
  # [11]  最低 今开 昨收 量比 换手率 市盈率-动态 市净率 总市值 流通市值 涨速
  # [21]  5分钟涨跌 60日涨跌幅 年初至今涨跌幅 date
  update <- fromJSON(getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_a_spot_em",
      .encoding = "utf-8"
    )
  )
  update <- mutate(update, date = date(now(tzone = "Asia/Shanghai")))
  update <- update[, c(24, 2, 10, 11, 4, 7)]
  colnames(update) <- c("date", "symbol", "high", "low", "close", "volume")
  update <- update[update$symbol %in% symbol_list, ]
  update <- na.omit(update)
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Found update for ", nrow(update), " stock(s)."
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
    if (data[nrow(data), 1] == update[1, 1]) {
      data[nrow(data), ] <- update[update$symbol == data[1, 2], ]
    } else {
      data <- rbind(data, update[update$symbol == data[1, 2], ])
    }

    # Calculate predictor
    x_dmi <- 1 - tnormalize(
      abs(adx_alt(data[, 3:5])[, 1] - adx_alt(data[, 3:5])[, 2]), t_adx
    )
    x_cci <- 1 - 2 * tnormalize(CCI(data[, 3:5]), t_cci)
    data$x <- x_dmi * x_cci

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
  fundflow <- foreach(
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

  fundflow <- reduce(fundflow, full_join, by = join_by("symbol", "name"))
  fundflow[, 3:6] <- fundflow[, 3:6] / abs(fundflow[, 3])

  combine_update <- function(df) merge(df[nrow(df), ], fundflow, by = "symbol")
  update <- bind_rows(lapply(data_list, combine_update))

  # [1]   symbol date high low close volume x name inflow1 inflow3
  # [11]  inflow5 inflow10
  update <- update[, c(2, 1, 8, 7, 9:12)]
  weigh_inflow <- function(x) {
    return(
      ifelse(x[1] > 0, 0.4, 0) + ifelse(x[2] > 0, 0.3, 0) +
        ifelse(x[3] > 0, 0.2, 0) + ifelse(x[4] > 0, 0.1, 0)
    )
  }
  update$score <- update$x + apply(update[, 5:8], 1, weigh_inflow)
  update <- update[order(update$score, decreasing = TRUE), ]
  update[, 4:9] <- format(round(update[, 4:9], 2), nsmall = 2)
  cat(
    capture.output(print(update, row.names = FALSE)),
    file = "ranking.txt",
    sep = "\n"
  )
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Ranked ", nrow(update), " stock(s);",
      " wrote to ranking.txt."
    )
  )
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " 小优选股助手建议您购买 ", update[1, 3], " 哦！"
    )
  )

  return(list(symbol_list, data_list, update))
}

query <- function(query = readLines("query.txt"), update = dl[[3]]) {
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  result <- foreach(
    symbol = query,
    .combine = rbind
  ) %dopar% {
    return(update[update$symbol == symbol, ])
  }
  unregister_dopar

  result <- result[order(result$score, decreasing = TRUE), ]
  print(result, row.names = FALSE)
}

query_plot <- function(query = readLines("query.txt"), data_list = dl[[2]]) {
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
}

backtest <- function(
  symbol_list, data_list,
  x_b = 0.75, x_s = 0.5, r_thr = 0.01, t_min = 10, t_max = 40
) {
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    symbol = symbol_list,
    .combine = "multiout",
    .export = "pct_change",
    .multicombine = TRUE,
    .init = list(list(), list())
  ) %dopar% {
    data <- data_list[[symbol]]
    data <- na.omit(data)
    s <- 1
    r_list <- data.frame(matrix(nrow = 0, ncol = 4))
    for (i in 1:nrow(data)) {
      if (i < s) next
      if (data[i, "x"] < x_b) next
      for (j in i:nrow(data)) {
        if (!(
            data[j, "x"] <= x_s &
            pct_change(data[i, "close"], data[j, "close"]) >= r_thr &
            j - i >= t_min &
            j - i <= t_max
          )
        ) next
        s <- j
        break
      }
      r <- pct_change(data[i, "close"], data[s, "close"])
      r_list <- rbind(r_list, c(symbol, data[i, "date"], data[s, "date"], r))
    }

    r_list <- data.frame(
      symbol = r_list[, 1],
      buy = as.Date(as.numeric(r_list[, 2])),
      sell = as.Date(as.numeric(r_list[, 3])),
      r = as.numeric(r_list[, 4])
    )

    r_stats <- c(
      symbol,
      sum(r_list[, 4]) / as.numeric(data[nrow(data), 1] - data[1, 1]) * 365
    )

    return(list(r_stats, r_list))
  }
  unregister_dopar

  r_stats <- data.frame(do.call(rbind, out[[1]]))
  r_stats <- data.frame(symbol = r_stats[, 1], apy = as.numeric(r_stats[, 2]))

  r_list <- out[[2]]
  get_name <- function(df) {
    unique(df[, 1])
  }
  names(r_list) <- do.call(c, lapply(out[[2]], get_name))
  print(paste0(
      "Backtested ", length(r_list), " stocks;",
      " APY mean = ", round(mean(r_stats[, 2]), 2), ",",
      " CV = ", round(sd(r_stats[, 2]), 2), "."
    )
  )

  r_cat <- do.call(rbind, r_list)
  hist <- hist(r_cat[r_cat[, 4] <= 1, 4], breaks = 100, probability = TRUE)

  return(list(r_stats, r_list))
}