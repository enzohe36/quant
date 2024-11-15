source("/Users/anzhouhe/Documents/quant/load_preset.r", encoding = "UTF-8")

# Assign input values
symbol_list <- symbol_list
data_list <- data_list
t_adx <- 60
t_cci <- 60
t_r <- 20

# ------------------------------------------------------------------------------

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

#return(list(symbol_list, data_list, data_latest))
