# https://stackoverflow.com/a/25110203
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list = ls(name = env), pos = env)
}

normalize <- function(v) {
  (v - min(v, na.rm = TRUE)) / (max(v, na.rm = TRUE) - min(v, na.rm = TRUE))
}

normalize0 <- function(v) {
  (0 - min(v, na.rm = TRUE)) / (max(v, na.rm = TRUE) - min(v, na.rm = TRUE))
}

tnormalize <- function(v, t) {
  (v - runMin(v, t)) / (runMax(v, t) - runMin(v, t))
}

# http://www.cftsc.com/qushizhibiao/610.html
ADX <- function(hlc, n = 14, m = 6) {
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
  df <- data.frame(adx, adxr)
  colnames(df) <- c("adx", "adxr")
  return(df)
}

ROR <- function(v1, v2) {
  (v2 - v1) / abs(v1)
}

# https://stackoverflow.com/a/19801108
multiout <- function(lst1, ...) {
  lapply(
    seq_along(lst1),
    function(i) c(lst1[[i]], lapply(list(...), function(lst2) lst2[[i]]))
  )
}

tsprint <- function(v) {
  v <- paste0("[", format(now(), "%H:%M:%S"), "] ", v)
  writeLines(v)
}

bizday <- function(date = NA) {
  date <- ymd(date)
  if (is.na(date)) date <- as_date(now() - hours(16))
  if (wday(date) == 1) date <- date - days(2)
  if (wday(date) == 7) date <- date - days(1)
  return(date)
}

em_data <- function(symbol, adjust, start_date, end_date) {
  data <- fromJSON(
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
      symbol = symbol,
      adjust = adjust,
      start_date = start_date,
      end_date = end_date,
      .encoding = "utf-8"
    )
  )
  # [1]   日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅
  # [11]  涨跌额 换手率
  data <- data[, c(1, 2, 3, 5, 6, 4, 7)]
  colnames(data) <- c(
    "date", "symbol", "open", "high", "low", "close", "volume"
  )
  data$date <- as.Date(data$date)
  return(data)
}

em_data_update <- function() {
  df <- fromJSON(
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_a_spot_em",
      .encoding = "utf-8"
    )
  )
  df <- mutate(df, date = bizday())
  # [1]   序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高
  # [11]  最低 今开 昨收 量比 换手率 市盈率-动态 市净率 总市值 流通市值 涨速
  # [21]  5分钟涨跌 60日涨跌幅 年初至今涨跌幅 date
  data_update <- df[, c(24, 2, 12, 10, 11, 4, 7)]
  colnames(data_update) <- c(
    "date", "symbol", "open", "high", "low", "close", "volume"
  )
  data_name <- df[, c(2, 3)]
  colnames(data_name) <- c("symbol", "name")
  return(list(data_update, data_name))
}

em_fundflow <- function(indicator, fundflow_dict) {
  fundflow <- fromJSON(
    getForm(
      uri = paste0(
        "http://127.0.0.1:8080/api/public/stock_individual_fund_flow_rank"
      ),
      indicator = indicator,
      .encoding = "utf-8"
    )
  )
  # [1]   序号 代码 名称 最新价 涨跌幅
  # [6]   主力净流入额 主力净流入占比 超大单净流入额 超大单净流入占比 大单净流入额
  # [11]  大单净流入占比 中单净流入额 中单净流入占比 小单净流入额 小单净流入占比
  fundflow <- fundflow[, c(2, 7)]
  header <- fundflow_dict[fundflow_dict$indicator == indicator, "header"]
  colnames(fundflow) <- c("symbol", header)
  fundflow[, header] <- as.numeric(fundflow[, header])
  return(fundflow)
}

em_index <- function(symbol) {
  data <- fromJSON(
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_index_daily_em",
      symbol = ifelse(
        grepl("^399", symbol), paste0("sz", symbol), paste0("sh", symbol)
      ),
      .encoding = "utf-8"
    )
  )
  data <- mutate(data, symbol = symbol)
  # [1]   date open close high low volume amount symbol
  data <- data[, c(1, 8, 2, 4, 5, 3, 6)]
  colnames(data) <- c(
    "date", "symbol", "open", "high", "low", "close", "volume"
  )
  data$date <- as.Date(data$date)
  return(data)
}

get_predictor <- function(data, t_adx, t_cci) {
  error <- try(
    {
      adx <- ADX(data[, c("high", "low", "close")])
      cci_n <- (
        1 - 2 * tnormalize(CCI(data[, c("high", "low", "close")]), t_cci)
      )
      data$xa <- (1 - normalize(abs(adx$adx - adx$adxr))) * cci_n
      data$xa1 <- lag(data$xa, 1)
      data$xad <- momentum(data$xa, 5)
      data$xb <- tnormalize(adx$adx, t_adx) * cci_n
      data$xb1 <- lag(data$xb, 1)
      data$xbd <- abs(momentum(data$xb, 5)) - abs(momentum(data$xb, 1))
      data$sg <- sgolayfilt(data$close, n = 7)
      data$sgd <- ROR(lag(data$sg, 10), data$sg)
    },
    silent = TRUE
  )
  if (class(error) == "try-error") {
    for (header in c("xa", "xa1", "xad", "xb", "xb1", "xbd", "sg", "sgd")) {
      data[, header] <- rep(NaN, nrow(data))
    }
  }
  return(data)
}
