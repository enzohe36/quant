update <- function() {
  tsprint("Started update().")

  # Define input
  symbol_list <- out0[["symbol_list"]]
  data_list <- out0[["data_list"]]

  # Define parameters
  t_adx <- 20
  t_cci <- 10
  x_h <- 0.53
  r_h <- 0.09
  r_l <- -0.5
  t_max <- 105

  # [1]   序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高
  # [11]  最低 今开 昨收 量比 换手率 市盈率-动态 市净率 总市值 流通市值 涨速
  # [21]  5分钟涨跌 60日涨跌幅 年初至今涨跌幅 date
  latest <- fromJSON(
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_a_spot_em",
      .encoding = "utf-8"
    )
  )
  latest <- mutate(latest, date = today())
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

    df <- latest[latest$symbol == symbol, ]
    if (
      !all((data[nrow(data), 3:6] == df[, 3:6]) %in% TRUE)
    ) {
      ifelse(
        data[nrow(data), "date"] == df$date,
        data[nrow(data), ] <- df,
        data <- bind_rows(data, df)
      )
    }
    if (any(is.na(data[nrow(data), ]))) data <- data[-c(nrow(data)), ]

    # Calculate predictor
    adx <- adx_alt(data[, 3:5])
    adx <- 1 - tnormalize(abs(adx$adx - adx$adxr), t_adx)
    cci <- 1 - 2 * tnormalize(CCI(data[, 3:5]), t_cci)
    data$x <- adx * cci
    data$x1 <- lag(data$x, 1)
    data$dx <- momentum(data$x, 5)

    lst <- list()
    lst[[symbol]] <- data
    return(lst)
  }
  unregister_dopar

  tsprint(glue("Checked {nrow(latest)} stocks for update."))

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
    fundflow <- fromJSON(
      getForm(
        uri = paste0(
          "http://127.0.0.1:8080/api/public/stock_individual_fund_flow_rank"
        ),
        indicator = indicator,
        .encoding = "utf-8"
      )
    )
    fundflow <- fundflow[, c(2, 3, 7)]
    header <- fundflow_dict[fundflow_dict$indicator == indicator, "header"]
    colnames(fundflow) <- c("symbol", "name", header)
    fundflow <- fundflow[fundflow$symbol %in% symbol_list, ]
    fundflow[, header] <- as.numeric(fundflow[, header])

    return(fundflow)
  }
  unregister_dopar

  fundflow <- reduce(fundflow, full_join, by = join_by("symbol", "name"))
  fundflow[, fundflow_dict$header] <- fundflow[, fundflow_dict$header] / 100

  # [1]   symbol date high low close volume x dx name in1
  # [11]  in3 in5 in10
  latest <- bind_rows(
    lapply(
      data_list,
      function(df) merge(df[nrow(df), ], fundflow, by = "symbol")
    )
  )
  latest <- cbind(
    latest[, c("date", "symbol", "name")],
    latest[, 7:ncol(latest)] %>% .[, sapply(., is.numeric)]
  )
  latest$in_score <- apply(latest[, fundflow_dict$header], 1, sum)
  latest <- latest[order(latest$in_score, decreasing = TRUE), ]
  latest[, sapply(latest, is.numeric)] <- round(
    latest[, sapply(latest, is.numeric)], 2
  )

  df <- latest[latest$x >= x_h & latest$x1 < x_h & latest$dx > 0, ]
  df[, sapply(df, is.numeric)] <- format(
    df[, sapply(df, is.numeric)], nsmall = 2
  )
  cat(
    capture.output(print(df, row.names = FALSE)) %>%
      gsub("     name", "    name", .) %>%
      gsub("^ ", "", .),
    file = "ranking.txt",
    sep = "\n"
  )
  tsprint(
    glue("Ranked {nrow(latest)} stocks; wrote {nrow(df)} to ranking.txt.")
  )

  # Evaluate portfolio
  portfolio <- read.csv(
    "portfolio.csv",
    colClasses = c(date = "Date", symbol = "character")
  )

  if (nrow(portfolio) != 0) {
    out <- data.frame()
    for (symbol in portfolio$symbol) {
      df <- portfolio[portfolio$symbol == symbol, ]
      data <- data_list[[symbol]]

      i <- which(data$date == df$date)
      j <- nrow(data)
      r <- ror(df$cost, data[j, "close"])
      out <- rbind(
        out, list(
          df$date,
          symbol,
          latest[latest$symbol == symbol, "name"],
          df$cost,
          r,
          ifelse(r >= r_h | r <= r_l | j - i >= t_max, "SELL", "HOLD")
        )
      )
    }
    colnames(out) <- c("date", "symbol", "name", "cost", "r", "action")
    out$date <- as.character(as.Date(out$date))
    out[, c("cost", "r")] <- format(round(out[, c("cost", "r")], 3), nsmall = 3)
    out <- arrange(out, desc(action), symbol)
    rownames(out) <- seq_len(nrow(out))

    which_sell <- which(out$action == "SELL")
    if (length(which_sell) != 0) {
      out <- rbind(
        out[which_sell, ],
        setNames(
          data.frame(t(replicate(ncol(out), "")), row.names = ""), names(out)
        ),
        out[-which_sell, ]
      )
    }
    print(out)
  }

  return(
    list(symbol_list = symbol_list, data_list = data_list, latest = latest)
  )
}