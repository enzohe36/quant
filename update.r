update <- function() {
  print(paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Started update()."
    ), quote = FALSE
  )

  # Define parameters
  symbol_list <- out0[[1]]
  data_list <- out0[[2]]
  t_adx <- 70
  t_cci <- 51
  x_h <- 0.53
  x_l <- 0.31
  r_h <- 0.037
  t_min <- 14
  t_max <- 104

  # [1]   序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高
  # [11]  最低 今开 昨收 量比 换手率 市盈率-动态 市净率 总市值 流通市值 涨速
  # [21]  5分钟涨跌 60日涨跌幅 年初至今涨跌幅 date
  latest <- fromJSON(getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_a_spot_em",
      .encoding = "utf-8"
    )
  )
  latest <- mutate(latest, date = date(now(tzone = "Asia/Shanghai")))
  latest <- latest[, c(24, 2, 10, 11, 4, 7)]
  colnames(latest) <- c("date", "symbol", "high", "low", "close", "volume")
  latest <- latest[latest$symbol %in% symbol_list, ]

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
        (data[nrow(data), 3:6] == latest[latest$symbol == symbol, 3:6])
          %in% TRUE
      )
    ) {
      if (data[nrow(data), 1] == latest[latest$symbol == symbol, 1]) {
        data[nrow(data), ] <- latest[latest$symbol == symbol, ]
      } else {
        data <- bind_rows(data, latest[latest$symbol == symbol, ])
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
      " Checked ", nrow(latest), " stocks for update."
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
        uri = paste0(
          "http://127.0.0.1:8080/api/public/stock_individual_fund_flow_rank"
        ),
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
  latest <- bind_rows(lapply(
      data_list,
      function(df) merge(df[nrow(df), ], fundflow, by = "symbol")
    )
  )
  latest <- latest[, c(2, 1, 9, 7, 8, 10:13)]
  weigh_inflow <- function(v) {
    return(
      ifelse(v[1] > 0, 0.4, 0) + ifelse(v[2] > 0, 0.3, 0) +
        ifelse(v[3] > 0, 0.2, 0) + ifelse(v[4] > 0, 0.1, 0)
    )
  }
  latest$score <- latest$x + apply(latest[, 6:9], 1, weigh_inflow)

  out <- latest[latest$x >= x_h & latest$dx > 0, ]
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
      " Ranked ", nrow(latest), " stocks;",
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
        ror(data[i, "close"], data[j, "close"]) >= r_h &
        data[j, "x"] <= x_l &
        j - i >= t_min
      ) | (
        j - i >= t_max
      )
    ) {
      out <- rbind(
        out, list(symbol, latest[latest$symbol == symbol, 3], "SELL")
      )
    } else {
      out <- rbind(
        out, list(symbol, latest[latest$symbol == symbol, 3], "HOLD")
      )
    }
  }
  colnames(out) <- c("symbol", "name", "action")
  out <- out[order(out$action, decreasing = TRUE), ]
  writeLines(c("", capture.output(print(out, row.names = FALSE))))

  return(list(symbol_list, data_list, latest))
}