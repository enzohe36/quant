Sys.setlocale(locale = "Chinese")
Sys.setenv(TZ = "Asia/Shanghai")

options(warn = -1)

set.seed(42)

aktools_path <- "http://127.0.0.1:8080/api/public/"

################################################################################
# Data update functions
################################################################################

get_index_spot <- function() {
  Sys.sleep(1)
  ts1 <- as_tradedate(now() - hours(16))
  data <- list(
    # 序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高 最低 今开 昨收 量比
    getForm(
      uri = paste0(aktools_path, "stock_zh_index_spot_em"),
      symbol = "上证系列指数",
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(market = "sh"),
    getForm(
      uri = paste0(aktools_path, "stock_zh_index_spot_em"),
      symbol = "深证系列指数",
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(market = "sz"),
    getForm(
      uri = paste0(aktools_path, "stock_zh_index_spot_em"),
      symbol = "中证系列指数",
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(market = "csi")
  ) %>%
    rbindlist(fill = TRUE)
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    mutate(
      symbol = `代码`,
      name = `名称`,
      market = market,
      date = !!ts1,
      open = `今开`,
      high = `最高`,
      low = `最低`,
      close = `最新价`,
      volume = `成交量`,
      amount = `成交额`
    ) %>%
    select(
      symbol, name, market, date, open, high, low, close, volume, amount
    ) %>%
    distinct(symbol, .keep_all = TRUE) %>%
    arrange(symbol)
  return(data)
}

get_index_hist <- function(symbol, start_date, end_date) {
  Sys.sleep(1)
  # date open close high low volume amount
  getForm(
    uri = paste0(aktools_path, "stock_zh_index_daily_em"),
    symbol = read_csv("data/indices.csv", show_col_types = FALSE) %>%
      filter(symbol == !!symbol) %>%
      pull(market) %>%
      paste0(symbol),
    start_date = format(start_date, "%Y%m%d"),
    end_date = format(end_date, "%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(date = as_date(date)) %>%
    select(date, open, high, low, close, volume, amount) %>%
    arrange(date)
}

get_index_comp <- function(symbol) {
  Sys.sleep(1)
  ts1 <- as_tradedate(now() - hours(16))
  # 日期 指数代码 指数名称 指数英文名称 成分券代码 成分券名称 成分券英文名称 交易所
  # 交易所英文名称 权重
  data <- getForm(
    uri = paste0(aktools_path, "index_stock_cons_weight_csindex"),
    symbol = symbol,
    .encoding = "utf-8"
  ) %>%
    fromJSON()
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    mutate(
      symbol = `成分券代码`,
      date = !!ts1,
      index = !!symbol,
      index_weight = `权重`
    ) %>%
    select(symbol, date, index, index_weight) %>%
    arrange(symbol)
  return(data)
}

get_symbols <- function() {
  Sys.sleep(1)
  ts1 <- as_tradedate(now() - hours(16))
  data <- list(
    # code name
    getForm(
      uri = paste0(aktools_path, "stock_info_a_code_name"),
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(
        symbol = code,
        name = name,
        delist = FALSE
      ),
    # 公司代码 公司简称 上市日期 暂停上市日期
    getForm(
      uri = paste0(aktools_path, "stock_info_sh_delist"),
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(
        symbol = `公司代码`,
        name = `公司简称`,
        delist = TRUE
      ),
    # 证券代码 证券简称 上市日期 终止上市日期
    getForm(
      uri = paste0(aktools_path, "stock_info_sz_delist"),
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(
        symbol = `证券代码`,
        name = `证券简称`,
        delist = TRUE
      )
  ) %>%
    rbindlist(fill = TRUE)
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    mutate(date = !!ts1) %>%
    select(symbol, name, date, delist) %>%
    distinct(symbol, .keep_all = TRUE) %>%
    arrange(symbol)
  return(data)
}

get_susp <- function() {
  Sys.sleep(1)
  ts1 <- as_tradedate(now() - hours(16))
  # 序号 代码 名称 停牌时间 停牌截止时间 停牌期限 停牌原因 所属市场 预计复牌时间
  data <- getForm(
    uri = paste0(aktools_path, "stock_tfp_em"),
    date = format(ts1, "%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON()
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    mutate(
      symbol = `代码`,
      date = !!ts1,
      susp = ifelse(
        `停牌时间` <= !!ts1 & (`停牌截止时间` >= !!ts1 | is.na(`停牌截止时间`)),
        TRUE,
        FALSE
      )
    ) %>%
    select(symbol, date, susp) %>%
    arrange(symbol)
  return(data)
}

# Refresh webpage if having connection error
# https://quote.eastmoney.com/center/gridlist.html#hs_a_board
get_spot <- function() {
  Sys.sleep(60)
  ts1 <- as_tradedate(now() - hours(16))
  # 序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高 最低 今开 昨收 量比 换手率
  # 市盈率-动态 市净率 总市值 流通市值 涨速 5分钟涨跌 60日涨跌幅 年初至今涨跌幅
  data <- getForm(
    uri = paste0(aktools_path, "stock_zh_a_spot_em"),
    .encoding = "utf-8"
  ) %>%
    fromJSON()
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    mutate(
      symbol = `代码`,
      date = !!ts1,
      open = `今开`,
      high = `最高`,
      low = `最低`,
      close = `最新价`,
      volume = `成交量`,
      amount = `成交额`
    ) %>%
    select(symbol, date, open, high, low, close, volume, amount) %>%
    arrange(symbol)
  return(data)
}

# Refresh webpage if having connection error
# https://data.eastmoney.com/yjfp/
get_adjust_change <- function() {
  Sys.sleep(60)
  ts1 <- as_tradedate(now() - hours(16))
  # 代码 名称 送转股份-送转总比例 送转股份-送转比例 送转股份-转股比例 现金分红-现金分红比例
  # 现金分红-股息率 每股收益 每股净资产 每股公积金 每股未分配利润 净利润同比增长 总股本
  # 预案公告日 股权登记日 除权除息日 方案进度 最新公告日期
  data <- list(
    getForm(
      uri = paste0(aktools_path, "stock_fhps_em"),
      date = quarter(ts1 %m-% months(3), "date_last") %>%
        format("%Y%m%d"),
      .encoding = "utf-8"
    ) %>%
      fromJSON(),
    getForm(
      uri = paste0(aktools_path, "stock_fhps_em"),
      date = quarter(ts1 %m-% months(6), "date_last") %>%
        format("%Y%m%d"),
      .encoding = "utf-8"
    ) %>%
      fromJSON(),
    getForm(
      uri = paste0(aktools_path, "stock_fhps_em"),
      date = quarter(ts1 %m-% months(9), "date_last") %>%
        format("%Y%m%d"),
      .encoding = "utf-8"
    ) %>%
      fromJSON(),
    getForm(
      uri = paste0(aktools_path, "stock_fhps_em"),
      date = quarter(ts1 %m-% months(12), "date_last") %>%
        format("%Y%m%d"),
      .encoding = "utf-8"
    ) %>%
      fromJSON()
  ) %>%
    rbindlist(fill = TRUE)
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    mutate(
      symbol = `代码`,
      adjust_change_date = as_date(`除权除息日`)
    ) %>%
    summarize(
      date = !!ts1,
      adjust_change_date = if_else(
        is.infinite(max(adjust_change_date, na.rm = TRUE)),
        as_date(NA),
        max(adjust_change_date, na.rm = TRUE)
      ),
      .by = symbol
    ) %>%
    arrange(symbol)
  return(data)
}

get_shares_change <- function() {
  Sys.sleep(1)
  ts1 <- as_tradedate(now() - hours(16))
  # 证券代码 证券简称 交易市场 公告日期 变动日期 变动原因 总股本 已流通股份 已流通比例
  # 流通受限股份
  data <- getForm(
    uri = paste0(aktools_path, "stock_hold_change_cninfo"),
    symbol = "全部",
    .encoding = "utf-8"
  ) %>%
    fromJSON()
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    mutate(
      symbol = `证券代码`,
      date = !!ts1,
      shares_change_date = as_date(`变动日期`)
    ) %>%
    select(symbol, date, shares_change_date) %>%
    arrange(symbol)
  return(data)
}

# Refresh webpage if having connection error
# https://data.eastmoney.com/bbsj/202003/yysj.html
get_val_change <- function() {
  Sys.sleep(60)
  ts1 <- as_tradedate(now() - hours(16))
  # 序号 股票代码 股票简称 首次预约时间 一次变更日期 二次变更日期 三次变更日期 实际披露时间
  data <- list(
    getForm(
      uri = paste0(aktools_path, "stock_yysj_em"),
      symbol = "沪深A股",
      date = quarter(ts1 %m-% months(3), "date_last") %>%
        format("%Y%m%d"),
      .encoding = "utf-8"
    ) %>%
      fromJSON(),
    getForm(
      uri = paste0(aktools_path, "stock_yysj_em"),
      symbol = "沪深A股",
      date = quarter(ts1 %m-% months(6), "date_last") %>%
        format("%Y%m%d"),
      .encoding = "utf-8"
    ) %>%
      fromJSON(),
    getForm(
      uri = paste0(aktools_path, "stock_yysj_em"),
      symbol = "沪深A股",
      date = quarter(ts1 %m-% months(9), "date_last") %>%
        format("%Y%m%d"),
      .encoding = "utf-8"
    ) %>%
      fromJSON(),
    getForm(
      uri = paste0(aktools_path, "stock_yysj_em"),
      symbol = "沪深A股",
      date = quarter(ts1 %m-% months(12), "date_last") %>%
        format("%Y%m%d"),
      .encoding = "utf-8"
    ) %>%
      fromJSON()
  ) %>%
    rbindlist(fill = TRUE)
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    mutate(
      symbol = `股票代码`,
      val_change_date = as_date(`实际披露时间`)
    ) %>%
    summarize(
      date = !!ts1,
      val_change_date = if_else(
        is.infinite(max(val_change_date, na.rm = TRUE)),
        as_date(NA),
        max(val_change_date, na.rm = TRUE)
      ),
      .by = symbol
    ) %>%
    arrange(symbol)
  return(data)
}

# Refresh webpage if having connection error
# https://quote.eastmoney.com/concept/sh603777.html?from=classic(示例)
get_hist <- function(symbol, start_date, end_date) {
  Sys.sleep(1)
  # 日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
  getForm(
    uri = paste0(aktools_path, "stock_zh_a_hist"),
    symbol = symbol,
    start_date = format(start_date, "%Y%m%d"),
    end_date = format(end_date, "%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      date = as_date(`日期`),
      open = `开盘`,
      high = `最高`,
      low = `最低`,
      close = `收盘`,
      volume = `成交量`,
      amount = `成交额`
    ) %>%
    select(date, open, high, low, close, volume, amount) %>%
    arrange(date)
}

get_adjust <- function(symbol) {
  Sys.sleep(1)
  # date hfq_factor
  getForm(
    uri = paste0(aktools_path, "stock_zh_a_daily"),
    symbol = paste0(
      case_when(
        str_detect(symbol, "^6") ~ "sh",
        str_detect(symbol, "^(0|3)") ~ "sz",
        TRUE ~ "bj"
      ),
      symbol
    ),
    adjust = "hfq-factor",
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      date = as_date(date),
      adjust = hfq_factor
    ) %>%
    select(date, adjust) %>%
    arrange(date)
}

get_mc <- function(symbol) {
  Sys.sleep(1)
  # date value
  getForm(
    uri = paste0(aktools_path, "stock_zh_valuation_baidu"),
    symbol = symbol,
    indicator = "总市值",
    period = "全部",
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      date = as_tradedate(date),
      mc = value
    ) %>%
    select(date, mc) %>%
    distinct(date, .keep_all = TRUE) %>%
    arrange(date)
}

get_val <- function(symbol) {
  Sys.sleep(1)
  # SECUCODE SECURITY_CODE SECURITY_NAME_ABBR ORG_CODE REPORT_DATE
  # SECURITY_TYPE_CODE EPSJB BPS PER_CAPITAL_RESERVE PER_UNASSIGN_PROFIT
  # PER_NETCASH TOTALOPERATEREVE GROSS_PROFIT PARENTNETPROFIT
  # DEDU_PARENT_PROFIT TOTALOPERATEREVETZ PARENTNETPROFITTZ DPNP_YOY_RATIO
  # YYZSRGDHBZC NETPROFITRPHBZC KFJLRGDHBZC ROE_DILUTED JROA GROSS_PROFIT_RATIO
  # NET_PROFIT_RATIO SEASON_LABEL
  getForm(
    uri = paste0(aktools_path, "stock_financial_analysis_indicator_em"),
    symbol = paste0(
      symbol,
      case_when(
        str_detect(symbol, "^6") ~ ".SH",
        str_detect(symbol, "^(0|3)") ~ ".SZ",
        TRUE ~ ".BJ"
      )
    ),
    indicator = "按单季度",
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      date = as_date(REPORT_DATE),
      val_change_date = as_tradedate(now() - hours(16)),
      revenue = TOTALOPERATEREVE,
      np = PARENTNETPROFIT,
      np_deduct = DEDU_PARENT_PROFIT,
      bvps = BPS,
      cfps = PER_NETCASH
    ) %>%
    select(date, val_change_date, revenue, np, np_deduct, bvps, cfps) %>%
    arrange(date)
}

get_val2 <- function(symbol) {
  Sys.sleep(1)
  # 数据日期 当日收盘价 当日涨跌幅 总市值 流通市值 总股本 流通股本 PE(TTM) PE(静) 市净率
  # PEG值 市现率 市销率
  getForm(
    uri = paste0(aktools_path, "stock_value_em"),
    symbol = symbol,
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      date = as_date(`数据日期`),
      mc = `总市值`,
      mc_float = `流通市值`,
      pe = `PE(TTM)`,
      peg = `PEG值`,
      pb = `市净率`,
      ps = `市现率`,
      pc = `市销率`
    ) %>%
    select(date, mc, mc_float, pe, peg, pb, ps, pc) %>%
    arrange(date)
}

################################################################################
# Math functions
################################################################################

# Redefines TTR::runSum
runSum <- function(x, n) {
  sapply(seq_along(x), function(i) {
    if (i < n) {
      return(NA_real_)   # not enough values before current
    }
    window <- x[(i - n + 1):i]
    if (any(is.na(window))) {
      return(NA_real_)
    } else {
      return(sum(window))
    }
  })
}

normalize <- function(v, range = c(0, 1), h = NULL) {
  min <- min(v, na.rm = TRUE)
  max <- max(v, na.rm = TRUE)
  range_min <- min(range, na.rm = TRUE)
  range_max <- max(range, na.rm = TRUE)
  if (!is.null(h)) v <- h
  v_norm <- (v - min) / (max - min) * (range_max - range_min) + range_min
  return(v_norm)
}

run_norm <- function(v, n) (v - runMin(v, n)) / (runMax(v, n) - runMin(v, n))

get_roc <- function(v1, v2) (v2 - v1) / v1

get_rmse <- function(v1, v2) sqrt(sum((v2 - v1) ^ 2) / length(v1))

run_whichmax <- function(v, n) {
  rollapply(
    v, width = n, align = "right", fill = NA,
    FUN = function(w) which.max(w)
  )
}

run_whichmin <- function(v, n) {
  rollapply(
    v, width = n, align = "right", fill = NA,
    FUN = function(w) which.min(w)
  )
}

run_varmax <- function(x, widths) {
  stopifnot(length(x) == length(widths))
  sapply(
    seq_along(x), function(i) {
      w <- widths[i]
      if (i < w | is.na(i < w)) return(NA)
      max(x[(i - w + 1):i], na.rm = TRUE)
    }
  )
}

run_varmin <- function(x, widths) {
  stopifnot(length(x) == length(widths))
  sapply(
    seq_along(x), function(i) {
      w <- widths[i]
      if (i < w | is.na(i < w)) return(NA)
      min(x[(i - w + 1):i], na.rm = TRUE)
    }
  )
}

fit_gaussian <- function(x, y) {
  nls(
    y ~ 1 / (s * sqrt(2 * pi)) * exp(-1 / 2 * ((x - m) / s) ^ 2),
    start = c(s = 1, m = 0)
  )
}

################################################################################
# Feature engineering functions
################################################################################

# Redefines TTR::ADX
# http://www.cftsc.com/qushizhibiao/610.html
ADX <- function(hlc, n = 14, m = 6) {
  hlc <- as.matrix(hlc)
  h <- hlc[, 1]
  l <- hlc[, 2]
  c <- hlc[, 3]
  tr <- runSum(TR(hlc), n)
  dh <- h - lag(h, 1)
  dl <- lag(l, 1) - l
  dmp <- runSum(ifelse(dh > 0 & dh > dl, dh, 0), n)
  dmn <- runSum(ifelse(dl > 0 & dl > dh, dl, 0), n)
  dip <- dmp / tr
  din <- dmn / tr
  adx <- SMA(abs(dip - din) / (dip + din), m)
  adxr <- (adx + lag(adx, m)) / 2
  diff <- abs(adx - adxr)
  result <- cbind(adx, adxr)
  colnames(result) <- c("adx", "adxr")
  return(result)
}

# Redefines TTR::TR
TR <- function(hlc, w = 1) {
  hlc <- as.matrix(hlc)
  h <- hlc[, 1]
  l <- hlc[, 2]
  c <- hlc[, 3]
  trueHigh <- pmax(runMax(h, w), lag(c, w), na.rm = TRUE)
  trueLow <- pmin(runMin(l, w), lag(c, w), na.rm = TRUE)
  tr <- trueHigh - trueLow
  result <- cbind(tr, trueHigh, trueLow)
  colnames(result) <- c("tr", "trueHigh", "trueLow")
  return(result)
}

# Redefines TTR::ATR
ATR <- function(hlc, n = 14, maType, ..., w = 1) {
  tr <- TR(hlc, w)
  maArgs <- list(n = n, ...)
  if (missing(maType)) {
    maType <- "EMA"
    if (is.null(maArgs$wilder)) maArgs$wilder <- TRUE
  }
  atr <- do.call(maType, c(list(tr[, 1]), maArgs))
  result <- cbind(tr[, 1], atr, tr[, 2:3])
  colnames(result) <- c("tr", "atr", "trueHigh", "trueLow")
  return(result)
}

# https://www.gupang.com/201207/0F31H1H012.html
get_trend <- function(v) {
  get_madiff <- function(v, n) 3 * WMA(v, n) - 2 * SMA(v, n)
  k1 <- get_madiff(EMA(v, 5), 6)
  k2 <- get_madiff(EMA(v, 8), 6)
  k3 <- get_madiff(EMA(v, 11), 6)
  k4 <- get_madiff(EMA(v, 14), 6)
  k5 <- get_madiff(EMA(v, 17), 6)
  k6 <- k1 + k2 + k3 + k4 - 4 * k5
  EMA(k6, 2)
}

# https://www.cnblogs.com/long136/p/18345060
get_maang <- function(hlc) {
  hlc <- as.matrix(hlc)
  h <- hlc[, 1]
  l <- hlc[, 2]
  c <- hlc[, 3]
  k1 <- SMA(c, 30)
  k2 <- SMA(h - l, 100) * 0.34
  atan((k1 - lag(k1, 1)) / k2) * 180 / pi
}

add_roc <- function(df, col = "close", periods = 1:20) {
  roc_matrix <- sapply(
    periods,
    function(n) (df[[col]] - lag(df[[col]], n)) / lag(df[[col]], n)
  )
  roc_df <- as.data.frame(roc_matrix)
  colnames(roc_df) <- paste0(col, "_roc", periods)
  cbind(df, roc_df)
}

################################################################################
# Utility functions
################################################################################

# .combine = "multiout", .multicombine = TRUE, .init = list(list(), list(), ...)
# https://stackoverflow.com/a/19801108
multiout <- function(lst1, ...) {
  lapply(
    seq_along(lst1),
    function(i) c(lst1[[i]], lapply(list(...), function(lst2) lst2[[i]]))
  )
}

ts <- function(v) paste0("[", format(now(), "%H:%M:%S"), "] ", v)

tsprint <- function(v) writeLines(ts(v))

tslog <- function(v, log_path) write(ts(v), log_path, append = TRUE)

as_tradedate <- function(datetime) {
  date <- as_date(datetime)
  holidays <- read_csv("data/holidays.csv", show_col_types = FALSE) %>%
    pull(date)
  tradedate <- lapply(
    date,
    function(date) {
      seq(date - weeks(3), date, "1 day") %>%
        .[!wday(., week_start = 1) %in% 6:7] %>%
        .[!.%in% holidays] %>%
        last()
    }
  ) %>%
    reduce(c)
  return(tradedate)
}
