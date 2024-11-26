update <- function() {
  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Started update()."
    ),
    quote = FALSE
  )

  # Define parameters
  symbol_list <- out0[[1]]
  data_list <- out0[[2]]
  t_adx <- 70
  t_cci <- 51
  x_h <- 0.53
  r_h <- 0.1
  r_l <- -0.5
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
      if (data[nrow(data), "date"] == latest[latest$symbol == symbol, "date"]) {
        data[nrow(data), ] <- latest[latest$symbol == symbol, ]
      } else {
        data <- bind_rows(data, latest[latest$symbol == symbol, ])
      }
    }
    if (any(is.na(data[nrow(data), ]))) data <- data[-c(nrow(data)), ]

    # Calculate predictor
    adx <- 1 - tnormalize(
      abs(adx_alt(data[, 3:5])[, "adx"] - adx_alt(data[, 3:5])[, "adxr"]), t_adx
    )
    cci <- 1 - 2 * tnormalize(CCI(data[, 3:5]), t_cci)
    data$x <- adx * cci
    data$dx <- momentum(data$x, 5)

    lst <- list()
    lst[[symbol]] <- data
    return(lst)
  }
  unregister_dopar

  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Checked ", nrow(latest), " stocks for update."
    ),
    quote = FALSE
  )

  fundflow_dict <- data.frame(
    indicator = c("今日", "3日", "5日", "10日"),
    header = c("in1", "in3", "in5", "in10")
  )

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  fundflow <- foreach(
    indicator = fundflow_dict$indicator,
    .packages = c("jsonlite", "RCurl")
  ) %dopar% {
    # [1]   序号 代码 名称 最新价 涨跌幅
    # [6]   主力净流入额 主力净流入占比 超大单净流入额 超大单净流入占比 大单净流入额
    # [11]  大单净流入占比 中单净流入额 中单净流入占比 小单净流入额 小单净流入占比
    fundflow <- fromJSON(getForm(
        uri = paste0(
          "http://127.0.0.1:8080/api/public/stock_individual_fund_flow_rank"
        ),
        indicator = indicator,
        .encoding = "utf-8"
      )
    )
    fundflow <- fundflow[, c(2, 3, 6)]

    header <- fundflow_dict[fundflow_dict$indicator == indicator, "header"]
    colnames(fundflow) <- c("symbol", "name", header)
    fundflow <- fundflow[fundflow$symbol %in% symbol_list, ]
    fundflow[, header] <- as.numeric(fundflow[, header])

    return(fundflow)
  }
  unregister_dopar

  fundflow <- reduce(fundflow, full_join, by = join_by("symbol", "name"))
  fundflow[, fundflow_dict$header] <- fundflow[, fundflow_dict$header] /
    abs(fundflow$in1)

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
  latest$score <- latest$x +
    apply(latest[, fundflow_dict$header], 1, weigh_inflow)
  latest <- latest[order(latest$score, decreasing = TRUE), ]
  latest[, sapply(latest, is.numeric)] <- round(
    latest[, sapply(latest, is.numeric)], 2
  )

  out <- latest[latest[, "x"] >= x_h & latest[, "dx"] > 0, ]
  out[, sapply(out, is.numeric)] <- format(
    out[, sapply(out, is.numeric)], nsmall = 2
  )
  cat(
    capture.output(print(out, row.names = FALSE)) %>%
      gsub("symbol     name", "symbol    name", .),
    file = "ranking.txt",
    sep = "\n"
  )
  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Ranked ", nrow(latest), " stocks;",
      " wrote ", nrow(out), " to ranking.txt."
    ),
    quote = FALSE
  )
  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " 小优选股助手建议您购买 ", out[1, "name"], " 哦！"
    ),
    quote = FALSE
  )

  # Evaluate portfolio
  portfolio <- read.csv(
    "portfolio.csv",
    colClasses = c(symbol = "character", cost = "numeric", date = "Date")
  )

  if (nrow(portfolio) != 0) {
    out <- data.frame()
    for (symbol in portfolio$symbol) {
      cost <- portfolio[portfolio$symbol == symbol, "cost"]
      date <- portfolio[portfolio$symbol == symbol, "date"]

      data <- data_list[[symbol]]
      i <- which(data$date == date)
      j <- nrow(data)
      r <- ror(cost, data[j, "close"])

      out <- rbind(out, list(
          symbol,
          latest[latest$symbol == symbol, "name"],
          cost,
          date,
          r,
          ifelse(r >= r_h | r <= r_l | j - i >= t_max, "SELL", "HOLD")
        )
      )
      colnames(out) <- c("symbol", "name", "cost", "date", "r", "action")
      out$date <- as.Date(out$date)
    }
    out$cost <- format(round(out$cost, 3), nsmall = 3)
    out$r <- format(round(out$r, 3), nsmall = 3)
    writeLines(c("", capture.output(print(out, row.names = FALSE))))
  }

  return(list(symbol_list, data_list, latest))
}