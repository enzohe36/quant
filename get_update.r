source("load_preset.r", encoding = "UTF-8")

# Define parameters
symbol_list <- symbol_list
data_list <- data_list
t_adx <- 60
t_cci <- 60

# ------------------------------------------------------------------------------

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

#return(list(symbol_list, data_list, update))
