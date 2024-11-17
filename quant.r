# python -m aktools
# (Source this script)
# get_history()
# out <- load_history() # symbol_list, data_list
# out <- update(out[[1]], out[[2]]) # symbol_list, data_list, update
# trading_list <- backtest(out[[1]], out[[2]])
# query(out[[2]], out[[3]], q_list = "000001")

source("preset.r", encoding = "UTF-8")

get_history <- function() {
  # Define parameters
  pattern <- "^(00|60)"
  t <- 5

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
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Found ", length(symbol_list), " stocks."
    ), quote = FALSE
  )

  symbol_list <- symbol_list[grepl(pattern, symbol_list)]
  #symbol_list <- sample(symbol_list, 10) # For testing only
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Matched ", length(symbol_list), " stocks to ", pattern, "."
    ), quote = FALSE
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
      " ", length(symbol_list), " stocks have ≥ ", t, " years of history;",
      " wrote to data/."
    ), quote = FALSE
  )
}

load_history <- function() {
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
}

update <- function(symbol_list, data_list) {
  # Define parameters
  t_adx <- 60
  t_cci <- 60

  x_h <- 0.6
  x_l <- 0.5
  r_h <- 0.01
  t_min <- 10
  t_max <- 60

  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Started update()."
    ), quote = FALSE
  )

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

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  data_list <- foreach(
    symbol = symbol_list,
    .combine = append,
    .export = c("tnormalize", "adx_alt"),
    .packages = c("tidyverse", "TTR")
  ) %dopar% {
    data <- data_list[[symbol]]
    if (
      !all(
        (data[nrow(data), 3:6] == update[update$symbol == symbol, 3:6]) %in% TRUE
      )
    ) {
      if (data[nrow(data), 1] == update[update$symbol == symbol, 1]) {
        data[nrow(data), ] <- update[update$symbol == symbol, ]
      } else {
        data <- bind_rows(data, update[update$symbol == symbol, ])
      }
    }
    if (any(is.na(data[nrow(data), ]))) data <- data[-c(nrow(data)), ]

    # Calculate predictor
    adx <- 1 - tnormalize(
      abs(adx_alt(data[, 3:5])[, 1] - adx_alt(data[, 3:5])[, 2]), t_adx
    )
    cci <- 1 - 2 * tnormalize(CCI(data[, 3:5]), t_cci)
    data$x <- adx * cci
    data$dx <- momentum(data$x, 5)

    lst <- list()
    lst[[symbol]] <- data
    return(lst)
  }
  unregister_dopar

  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Checked ", nrow(update), " stocks for update."
    ), quote = FALSE
  )

  fundflow_dict <- data.frame(
    indicator = c("今日", "3日", "5日", "10日"),
    header = c("in1", "in3", "in5", "in10")
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

  # [1]   symbol date high low close volume x dx name in1
  # [11]  in3 in5 in10
  update <- bind_rows(lapply(
      data_list,
      function(df) merge(df[nrow(df), ], fundflow, by = "symbol")
    )
  )
  update <- update[, c(2, 1, 9, 7, 8, 10:13)]
  weigh_inflow <- function(v) {
    return(
      ifelse(v[1] > 0, 0.4, 0) + ifelse(v[2] > 0, 0.3, 0) +
        ifelse(v[3] > 0, 0.2, 0) + ifelse(v[4] > 0, 0.1, 0)
    )
  }
  update$score <- update$x + apply(update[, 6:9], 1, weigh_inflow)

  out <- update[update$x >= x_h & update$dx > 0, ]
  out <- out[order(out$score, decreasing = TRUE), ]
  out[, 4:10] <- format(round(out[, 4:10], 2), nsmall = 2)
  cat(
    capture.output(print(out, row.names = FALSE)) %>%
      gsub("symbol     name", "symbol    name", .),
    file = "ranking.txt",
    sep = "\n"
  )
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Ranked ", nrow(update), " stocks;",
      " wrote ", nrow(out), " to ranking.txt."
    ), quote = FALSE
  )
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " 小优选股助手建议您购买 ", out[1, 3], " 哦！"
    ), quote = FALSE
  )

  # Evaluate portfolio
  portfolio <- read.csv("portfolio.csv")
  portfolio[, 1] <- as.Date(portfolio[, 1])
  portfolio[, 2] <- formatC(portfolio[, 2], width = 6, format = "d", flag = "0")

  out <- data.frame(matrix(nrow = 0, ncol = 3))
  for (symbol in portfolio[, 2]) {
    data <- data_list[[symbol]]
    i <- which(data$date == portfolio[portfolio$symbol == symbol, 1])
    j <- nrow(data)
    if (
      (
        pct_change(data[i, "close"], data[j, "close"]) >= r_h &
        data[j, "x"] <= x_l &
        j - i >= t_min
      ) | (
        j - i >= t_max
      )
    ) {
      out <- rbind(
        out, list(symbol, update[update$symbol == symbol, 3], "SELL")
      )
    } else {
      out <- rbind(
        out, list(symbol, update[update$symbol == symbol, 3], "HOLD")
      )
    }
  }
  colnames(out) <- c("symbol", "name", "action")
  out <- out[order(out$action, decreasing = TRUE), ]
  writeLines(c("", capture.output(print(out, row.names = FALSE))))

  return(list(symbol_list, data_list, update))
}

backtest <- function(symbol_list, data_list) {
  # Define parameters
  x_h <- 0.6
  x_l <- 0.5
  r_h <- 0.01
  t_min <- 10
  t_max <- 60

  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Started backtest()."
    ), quote = FALSE
  )

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    symbol = symbol_list,
    .combine = multiout,
    .multicombine = TRUE,
    .init = list(list(), list()),
    .export = "pct_change"
  ) %dopar% {
    data <- data_list[[symbol]]
    data <- na.omit(data)
    data[, 1] <- as.Date(data[, 1])

    s <- 1
    trading <- data.frame(matrix(nrow = 0, ncol = 4))
    for (i in 1:nrow(data)) {
      if (i < s | data[i, "x"] < x_h | data[i, "dx"] <= 0) {
        next
      }
      for (j in i:nrow(data)) {
        if (
          (
            pct_change(data[i, "close"], data[j, "close"]) >= r_h &
            data[j, "x"] <= x_l &
            j - i >= t_min
          ) | (
            j - i >= t_max
          )
        ) {
          s <- j
          break
        }
      }
      if (i < s) {
        r <- pct_change(data[i, "close"], data[s, "close"])
        trading <- rbind(
          trading, list(symbol, data[i, "date"], data[s, "date"], r)
        )
      }
    }
    colnames(trading) <- c("symbol", "buy", "sell", "r")
    trading[, 2] <- as.Date(trading[, 2])
    trading[, 3] <- as.Date(trading[, 3])

    apy <- data.frame(
      symbol,
      sum(trading[, 4]) / as.numeric(data[nrow(data), 1] - data[1, 1]) * 365
    )
    colnames(apy) <- c("symbol", "apy")

    return(list(trading, apy))
  }
  unregister_dopar

  trading_list <- out[[1]]
  names(trading_list) <- do.call(c, lapply(trading_list, function(df) df[1, 1]))
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Backtested ", length(trading_list), " stocks."
    ), quote = FALSE
  )

  trading <- do.call(rbind, trading_list)
  writeLines(c(
      "",
      capture.output(round(quantile(trading[, 4], seq(0, 1, 0.1)), 4))
    )
  )

  apy_low <- do.call(rbind, out[[2]])[, 2]
  apy_high <- trading[, 4] / as.numeric(trading[, 3] - trading[, 2]) * 365
  stats <- data.frame(
    Mean = c(
      mean(trading[, 4]),
      mean(as.numeric(trading[, 3] - trading[, 2])),
      mean(apy_low),
      mean(apy_high)
    ),
    SD = c(
      sd(trading[, 4]),
      sd(as.numeric(trading[, 3] - trading[, 2])),
      sd(apy_low),
      sd(apy_high)
    ),
    row.names = c("r", "t", "APY_low", "APY_high")
  )
  writeLines(c(
      "",
      capture.output(round(stats, 2))
    )
  )

  hist <- hist(trading[trading$r <= 1 & trading$r >= -1, 4], breaks = 100)

  return(trading_list)
}

query <- function(data_list, update, q_list) {
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
}
