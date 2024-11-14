source("/Users/anzhouhe/Documents/quant/load_preset.r", encoding = "UTF-8")

# Assign input values
symbol_list <- symbol_list
data_list <- data_list
t_r <- 20
t_dx <- 60
t_cci <- 60

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

#return(list(symbol_list, data_list, data_latest))
