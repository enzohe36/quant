# library(RCurl)
# library(jsonlite)
# library(data.table)
# library(glue)
# library(tidyverse)

aktools_path <- "http://127.0.0.1:8080/api/public/"
indices <- read_csv("data/indices.csv", show_col_types = FALSE)

# ============================================================================
# Index Data Retrievers
# ============================================================================

# https://quote.eastmoney.com/center/gridlist.html#index_sz
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

# http://quote.eastmoney.com/center/hszs.html
get_index_hist <- function(symbol, start_date, end_date) {
  Sys.sleep(1)
  # date open close high low volume amount
  getForm(
    uri = paste0(aktools_path, "stock_zh_index_daily_em"),
    symbol = paste0(filter(indices, symbol == !!symbol)$market, symbol),
    start_date = format(start_date, "%Y%m%d"),
    end_date = format(end_date, "%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(date = as_date(date)) %>%
    select(date, open, high, low, close, volume, amount) %>%
    arrange(date)
}

# http://www.csindex.com.cn/zh-CN/indices/index-detail/000300
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

# ============================================================================
# Stock Data Retrievers
# ============================================================================

# https://www.sse.com.cn/assortment/stock/list/share/
# https://www.szse.cn/market/product/stock/list/index.html
# https://www.bse.cn/nq/listedcompany.html
# https://www.sse.com.cn/assortment/stock/list/delisting/
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

# https://data.eastmoney.com/tfpxx/
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

# https://quote.eastmoney.com/center/gridlist.html#hs_a_board
get_spot <- function() {
  Sys.sleep(1)
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

# https://data.eastmoney.com/yjfp/
get_div <- function(date) {
  Sys.sleep(1)
  ts1 <- as_tradedate(now() - hours(16))
  # 代码 名称 送转股份-送转总比例 送转股份-送转比例 送转股份-转股比例 现金分红-现金分红比例
  # 现金分红-股息率 每股收益 每股净资产 每股公积金 每股未分配利润 净利润同比增长 总股本
  # 预案公告日 股权登记日 除权除息日 方案进度 最新公告日期
  data <- getForm(
    uri = paste0(aktools_path, "stock_fhps_em"),
    date = quarter(date %m-% months(3), "date_last") %>%
      format("%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON()
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    mutate(
      symbol = `代码`,
      adjust_change_date = as_date(`除权除息日`)
    ) %>%
    select(symbol, adjust_change_date) %>%
    arrange(symbol)
  return(data)
}

get_adjust_change <- function() {
  Sys.sleep(1)
  ts1 <- as_tradedate(now() - hours(16))
  data <- list(
    get_div(ts1),
    get_div(ts1 %m-% months(3)),
    get_div(ts1 %m-% months(6)),
    get_div(ts1 %m-% months(9))
  ) %>%
    rbindlist(fill = TRUE)
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    summarize(
      date = !!ts1,
      adjust_change_date = if_else(
        is.infinite(max(adjust_change_date, na.rm = TRUE)),
        as_date(NA),
        max(adjust_change_date, na.rm = TRUE)
      ),
      .by = symbol
    )
  return(data)
}

# https://webapi.cninfo.com.cn/#/thematicStatistics
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

# https://data.eastmoney.com/bbsj/202003/yysj.html
get_earnings_calendar <- function(date) {
  Sys.sleep(1)
  ts1 <- as_tradedate(now() - hours(16))
  # 序号 股票代码 股票简称 首次预约时间 一次变更日期 二次变更日期 三次变更日期 实际披露时间
  data <- getForm(
    uri = paste0(aktools_path, "stock_yysj_em"),
    symbol = "沪深A股",
    date = quarter(date %m-% months(3), "date_last") %>%
      format("%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON()
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    mutate(
      symbol = `股票代码`,
      val_change_date = as_date(`实际披露时间`)
    ) %>%
    select(symbol, val_change_date) %>%
    arrange(symbol)
  return(data)
}

get_val_change <- function() {
  Sys.sleep(1)
  ts1 <- as_tradedate(now() - hours(16))
  # 序号 股票代码 股票简称 首次预约时间 一次变更日期 二次变更日期 三次变更日期 实际披露时间
  data <- list(
    get_earnings_calendar(ts1),
    get_earnings_calendar(ts1 %m-% months(3)),
    get_earnings_calendar(ts1 %m-% months(6)),
    get_earnings_calendar(ts1 %m-% months(9))
  ) %>%
    rbindlist(fill = TRUE)
  ts2 <- as_tradedate(now() - hours(9))
  if (ts1 != ts2) stop(glue("Trade date changed from {ts1} to {ts2}!"))
  data <- data %>%
    summarize(
      date = !!ts1,
      val_change_date = if_else(
        is.infinite(max(val_change_date, na.rm = TRUE)),
        as_date(NA),
        max(val_change_date, na.rm = TRUE)
      ),
      .by = symbol
    )
  return(data)
}

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

# https://finance.sina.com.cn/realstock/company/sh600006/nc.shtml(示例)
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

# https://gushitong.baidu.com/stock/ab-002044
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

# https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code=SZ301389&color=b#/cwfx
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

# https://data.eastmoney.com/gzfx/detail/300766.html
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
