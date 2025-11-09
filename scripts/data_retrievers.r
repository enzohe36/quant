# =============================== PRESET ==================================

# library(RCurl)
# library(jsonlite)
# library(data.table)
# library(tidyverse)

data_dir <- "data/"
indices_path <- paste0(data_dir, "indices.csv")

default_end_date_expr <- expr(as_tradeday(now() - hours(16)))
current_tradeday_expr <- expr(as_tradeday(now() - hours(8)))

# ========================== HELPER FUNCTIONS =============================

aktools <- function(key, ...){
  Sys.sleep(1)
  ts1 <- eval(default_end_date_expr)
  result <- getForm(
    uri = paste0("http://127.0.0.1:8080/api/public/", key),
    ...,
    .encoding = "utf-8"
  ) %>%
    fromJSON()
  ts2 <- eval(current_tradeday_expr)
  if (ts1 != ts2) {
    stop(str_glue("Trade date changed from {ts1} to {ts2}!"))
  } else {
    return(result)
  }
  return(result)
}

loop_function <- function(func_name, ..., fail_max = 3, wait = 60) {
  fail_count <- 1
  while (fail_count <= fail_max) {
    try_error <- try(
      out <- do.call(func_name, list(...)),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      tsprint(
        str_glue("Error running {func_name}, attempt {fail_count}/{fail_max}.")
      )
      fail_count <- fail_count + 1
      Sys.sleep(wait)
    } else {
      return(out)
    }
  }
  stop("Maximum retries exceeded.")
}

# ============================= INDEX DATA ================================

# https://quote.eastmoney.com/center/gridlist.html#index_sz
get_index_spot <- function(index_type) {
  ts1 <- eval(default_end_date_expr)
  aktools(
    key = "stock_zh_index_spot_em",
    symbol = index_type
  ) %>%
    mutate(
      symbol = `代码`,
      name = `名称`,
      date = !!ts1,
      open = as.numeric(`今开`),
      high = as.numeric(`最高`),
      low = as.numeric(`最低`),
      close = as.numeric(`最新价`),
      volume = as.numeric(`成交量`),
      amount = as.numeric(`成交额`)
    ) %>%
    select(symbol, name, date, open, high, low, close, volume, amount) %>%
    arrange(symbol)
}

combine_indices <- function() {
  list(
    loop_function("get_index_spot", "上证系列指数") %>%
      mutate(market = "sh"),
    loop_function("get_index_spot", "深证系列指数") %>%
      mutate(market = "sz"),
    loop_function("get_index_spot", "中证系列指数") %>%
      mutate(market = "csi")
  ) %>%
    rbindlist(fill = TRUE) %>%
    select(symbol, name, market, date) %>%
    distinct(symbol, .keep_all = TRUE) %>%
    arrange(symbol)
}

# http://quote.eastmoney.com/center/hszs.html
get_index_hist <- function(symbol, start_date, end_date) {
  indices <- read_csv(indices_path, show_col_types = FALSE)

  # date open close high low volume amount
  aktools(
    key = "stock_zh_index_daily_em",
    symbol = paste0(filter(indices, symbol == !!symbol)$market, symbol),
    start_date = format(start_date, "%Y%m%d"),
    end_date = format(end_date, "%Y%m%d")
  ) %>%
    mutate(date = as_date(date)) %>%
    select(date, open, high, low, close, volume, amount) %>%
    arrange(date)
}

# http://www.csindex.com.cn/zh-CN/indices/index-detail/000300
get_index_comp <- function(symbol) {
  # 日期 指数代码 指数名称 指数英文名称 成分券代码 成分券名称 成分券英文名称 交易所
  # 交易所英文名称 权重
  aktools(
    key = "index_stock_cons_weight_csindex",
    symbol = symbol
  ) %>%
    mutate(
      symbol = `成分券代码`,
      index = !!symbol,
      index_weight = as.numeric(`权重`)
    ) %>%
    select(symbol, index, index_weight) %>%
    arrange(symbol)
}

# ============================== SPOT DATA ================================

# https://quote.eastmoney.com/center/gridlist.html#hs_a_board
get_spot <- function() {
  ts1 <- eval(default_end_date_expr)
  # 序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高 最低 今开 昨收 量比 换手率
  # 市盈率-动态 市净率 总市值 流通市值 涨速 5分钟涨跌 60日涨跌幅 年初至今涨跌幅
  aktools(
    key = "stock_zh_a_spot_em"
  ) %>%
    mutate(
      symbol = `代码`,
      name = `名称`,
      date = !!ts1,
      open = as.numeric(`今开`),
      high = as.numeric(`最高`),
      low = as.numeric(`最低`),
      close = as.numeric(`最新价`),
      volume = as.numeric(`成交量`),
      amount = as.numeric(`成交额`)
    ) %>%
    select(symbol, name, date, open, high, low, close, volume, amount) %>%
    arrange(symbol)
}

get_delist <- function() {
  symbols <- c(
    # 公司代码 公司简称 上市日期 暂停上市日期
    loop_function(
      "aktools",
      key = "stock_info_sh_delist"
    ) %>%
      pull(`公司代码`),
    # 证券代码 证券简称 上市日期 终止上市日期
    loop_function(
      "aktools",
      key = "stock_info_sz_delist"
    ) %>%
      pull(`证券代码`)
  )
  tibble(
    symbol = symbols,
    delist = TRUE
  ) %>%
    distinct(symbol, .keep_all = TRUE) %>%
    arrange(symbol)
}

# https://data.eastmoney.com/tfpxx/
get_susp <- function() {
  ts1 <- eval(default_end_date_expr)
  # 序号 代码 名称 停牌时间 停牌截止时间 停牌期限 停牌原因 所属市场 预计复牌时间
  aktools(
    key = "stock_tfp_em",
    date = format(ts1, "%Y%m%d")
  ) %>%
    mutate(
      symbol = `代码`,
      susp = ifelse(
        `停牌时间` <= !!ts1 & (`停牌截止时间` >= !!ts1 | is.na(`停牌截止时间`)),
        TRUE,
        FALSE
      )
    ) %>%
    select(symbol, susp) %>%
    arrange(symbol)
}

# https://data.eastmoney.com/yjfp/
get_adjust_change <- function(date) {
  # 代码 名称 送转股份-送转总比例 送转股份-送转比例 送转股份-转股比例 现金分红-现金分红比例
  # 现金分红-股息率 每股收益 每股净资产 每股公积金 每股未分配利润 净利润同比增长 总股本
  # 预案公告日 股权登记日 除权除息日 方案进度 最新公告日期
  aktools(
    key = "stock_fhps_em",
    date = quarter(date %m-% months(3), "date_last") %>%
      format("%Y%m%d")
  ) %>%
    mutate(
      symbol = `代码`,
      adjust_change_date = as_date(`除权除息日`)
    ) %>%
    select(symbol, adjust_change_date) %>%
    arrange(symbol)
}

combine_adjust_change <- function() {
  ts1 <- eval(default_end_date_expr)
  list(
    loop_function("get_adjust_change", ts1),
    loop_function("get_adjust_change", ts1 %m-% months(3)),
    loop_function("get_adjust_change", ts1 %m-% months(6)),
    loop_function("get_adjust_change", ts1 %m-% months(9))
  ) %>%
    rbindlist(fill = TRUE) %>%
    summarize(
      adjust_change_date = if_else(
        is.infinite(max(adjust_change_date, na.rm = TRUE)),
        as_date(NA),
        max(adjust_change_date, na.rm = TRUE)
      ),
      .by = symbol
    )
}

# https://webapi.cninfo.com.cn/#/thematicStatistics
get_shares_change <- function() {
  # 证券代码 证券简称 交易市场 公告日期 变动日期 变动原因 总股本 已流通股份 已流通比例
  # 流通受限股份
  aktools(
    key = "stock_hold_change_cninfo",
    symbol = "全部"
  ) %>%
    mutate(
      symbol = `证券代码`,
      shares_change_date = as_date(`变动日期`)
    ) %>%
    select(symbol, shares_change_date) %>%
    arrange(symbol)
}

# https://data.eastmoney.com/bbsj/202003/yysj.html
get_val_change <- function(date) {
  # 序号 股票代码 股票简称 首次预约时间 一次变更日期 二次变更日期 三次变更日期 实际披露时间
  aktools(
    key = "stock_yysj_em",
    symbol = "沪深A股",
    date = quarter(date %m-% months(3), "date_last") %>%
      format("%Y%m%d")
  ) %>%
    mutate(
      symbol = `股票代码`,
      val_change_date = as_date(`实际披露时间`)
    ) %>%
    select(symbol, val_change_date) %>%
    arrange(symbol)
}

combine_val_change <- function() {
  ts1 <- eval(default_end_date_expr)
  # 序号 股票代码 股票简称 首次预约时间 一次变更日期 二次变更日期 三次变更日期 实际披露时间
  list(
    loop_function("get_val_change", ts1),
    loop_function("get_val_change", ts1 %m-% months(3)),
    loop_function("get_val_change", ts1 %m-% months(6)),
    loop_function("get_val_change", ts1 %m-% months(9))
  ) %>%
    rbindlist(fill = TRUE) %>%
    summarize(
      val_change_date = if_else(
        is.infinite(max(val_change_date, na.rm = TRUE)),
        as_date(NA),
        max(val_change_date, na.rm = TRUE)
      ),
      .by = symbol
    )
}

combine_spot <- function() {
  spot <- loop_function("get_spot")
  delist <- loop_function("get_delist")
  susp <- loop_function("get_susp")
  adjust_change <- combine_adjust_change()
  shares_change <- loop_function("get_shares_change")
  val_change <- combine_val_change()

  spot %>%
    left_join(delist, by = "symbol") %>%
    left_join(susp, by = "symbol") %>%
    left_join(adjust_change, by = "symbol") %>%
    left_join(shares_change, by = "symbol") %>%
    left_join(val_change, by = "symbol") %>%
    mutate(
      delist = replace_na(delist, FALSE),
      susp = replace_na(susp, FALSE)
    )
}

# =========================== HISTORICAL DATA =============================

# https://quote.eastmoney.com/concept/sh603777.html?from=classic(示例)
get_hist <- function(symbol, start_date, end_date) {
  # 日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
  aktools(
    key = "stock_zh_a_hist",
    symbol = symbol,
    start_date = format(start_date, "%Y%m%d"),
    end_date = format(end_date, "%Y%m%d")
  ) %>%
    mutate(
      date = as_date(`日期`),
      open = as.numeric(`开盘`),
      high = as.numeric(`最高`),
      low = as.numeric(`最低`),
      close = as.numeric(`收盘`),
      volume = as.numeric(`成交量`),
      amount = as.numeric(`成交额`)
    ) %>%
    select(date, open, high, low, close, volume, amount) %>%
    arrange(date)
}

# https://finance.sina.com.cn/realstock/company/sh600006/nc.shtml(示例)
get_adjust <- function(symbol) {
  # date hfq_factor
  aktools(
    key = "stock_zh_a_daily",
    symbol = paste0(
      case_when(
        str_detect(symbol, "^6") ~ "sh",
        str_detect(symbol, "^(0|3)") ~ "sz",
        TRUE ~ "bj"
      ),
      symbol
    ),
    adjust = as.numeric("hfq-factor")
  ) %>%
    mutate(
      date = as_date(date),
      adjust = as.numeric(hfq_factor)
    ) %>%
    select(date, adjust) %>%
    arrange(date)
}

# https://gushitong.baidu.com/stock/ab-002044
get_mc <- function(symbol) {
  # date value
  aktools(
    key = "stock_zh_valuation_baidu",
    symbol = symbol,
    indicator = "总市值",
    period = "全部"
  ) %>%
    mutate(
      date = as_tradeday(date),
      mc = as.numeric(value)
    ) %>%
    select(date, mc) %>%
    distinct(date, .keep_all = TRUE) %>%
    arrange(date)
}

# https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code=SZ301389&color=b#/cwfx
get_val <- function(symbol) {
  # SECUCODE SECURITY_CODE SECURITY_NAME_ABBR ORG_CODE REPORT_DATE
  # SECURITY_TYPE_CODE EPSJB BPS PER_CAPITAL_RESERVE PER_UNASSIGN_PROFIT
  # PER_NETCASH TOTALOPERATEREVE GROSS_PROFIT PARENTNETPROFIT
  # DEDU_PARENT_PROFIT TOTALOPERATEREVETZ PARENTNETPROFITTZ DPNP_YOY_RATIO
  # YYZSRGDHBZC NETPROFITRPHBZC KFJLRGDHBZC ROE_DILUTED JROA GROSS_PROFIT_RATIO
  # NET_PROFIT_RATIO SEASON_LABEL
  aktools(
    key = "stock_financial_analysis_indicator_em",
    symbol = paste0(
      symbol,
      case_when(
        str_detect(symbol, "^6") ~ ".SH",
        str_detect(symbol, "^(0|3)") ~ ".SZ",
        TRUE ~ ".BJ"
      )
    ),
    indicator = "按单季度"
  ) %>%
    mutate(
      date = as_date(REPORT_DATE),
      val_change_date = as_tradeday(now() - hours(16)),
      revenue = as.numeric(TOTALOPERATEREVE),
      np = as.numeric(PARENTNETPROFIT),
      np_deduct = as.numeric(DEDU_PARENT_PROFIT),
      bvps = as.numeric(BPS),
      cfps = as.numeric(PER_NETCASH)
    ) %>%
    select(date, val_change_date, revenue, np, np_deduct, bvps, cfps) %>%
    arrange(date)
}
