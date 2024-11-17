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
