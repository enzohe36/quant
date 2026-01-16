# PRESET =======================================================================

# library(RCurl)
# library(jsonlite)

resources_dir <- "resources/"
indices_path <- paste0(resources_dir, "indices.csv")

# HELPER FUNCTIONS =============================================================

indices <- if (file.exists(indices_path)) {
  read_csv(indices_path, show_col_types = FALSE)
}

aktools <- function(key, ...){
  args <- list(...) %>%
    lapply(function(x) if (is.character(x)) enc2utf8(x) else x)

  Sys.sleep(1)
  last_td <- eval(last_td_expr)
  curr_td <- eval(curr_td_expr)
  # if (last_td != curr_td) {
  #   stop(str_glue("Tradeday changed from {last_td} to {curr_td}!"))
  # }
  result <- do.call(
    getForm,
    c(
      list(uri = paste0("http://127.0.0.1:8080/api/public/", key)),
      args,
      list(.encoding = "utf-8")
    )
  ) %>%
    fromJSON()
  return(result)
}

loop_function <- function(func_name, ..., fail_max = 10, wait = 60) {
  fail_count <- 1
  while (fail_count <= fail_max) {
    try_error <- try(
      out <- do.call(func_name, list(...)),
      silent = TRUE
    )
    if (inherits(try_error, "try-error")) {
      error_msg <- attr(try_error, "condition")$message
      tsprint(
        str_glue("{func_name}() attempt {fail_count}/{fail_max}: {error_msg}")
      )
      fail_count <- fail_count + 1
      Sys.sleep(wait)
    } else {
      return(out)
    }
  }
  stop("Maximum retries exceeded.")
}

# SPOT DATA ====================================================================

get_indices <- function() {
  curr_td <- eval(curr_td_expr)
  # index_code display_name publish_date
  aktools("index_stock_info") %>%
    mutate(
      date = !!curr_td,
      source = "csi",
      index = index_code,
      index_name = display_name,
      stock_count = NA
    ) %>%
    select(date, source, index, index_name, stock_count) %>%
    arrange(index)
}

get_indices_csi <- function() {
  curr_td <- eval(curr_td_expr)
  # 指数代码 指数简称 指数全称 基日 基点 指数系列 样本数量 最新收盘 近一个月收益率 资产类别
  # 指数热点 指数币种 合作指数 跟踪产品 指数合规 指数类别 发布时间
  aktools("index_csindex_all") %>%
    mutate(
      date = !!curr_td,
      source = "csi",
      index = `指数代码`,
      index_name = `指数简称`,
      stock_count = as.integer(`样本数量`)
    ) %>%
    select(date, source, index, index_name, stock_count) %>%
    arrange(index)
}

get_indices_cni <- function() {
  curr_td <- eval(curr_td_expr)
  # 指数代码 指数简称 样本数 收盘点位 涨跌幅 PE滚动 成交量 成交额 总市值 自由流通市值
  aktools("index_all_cni") %>%
    mutate(
      date = !!curr_td,
      source = "cni",
      index = `指数代码`,
      index_name = `指数简称`,
      stock_count = as.integer(`样本数`)
    ) %>%
    select(date, source, index, index_name, stock_count) %>%
    arrange(index)
}

combine_indices <- function() {
  list(
    loop_function("get_indices"),
    loop_function("get_indices_csi"),
    loop_function("get_indices_cni")
  ) %>%
    rbindlist(fill = TRUE) %>%
    distinct(index, .keep_all = TRUE) %>%
    arrange(index)
}

# http://www.csindex.com.cn/zh-CN/indices/index-detail/000300
get_index_comp <- function(index) {
  index <- filter(indices, index == !!index)
  curr_td <- eval(curr_td_expr)
  if (index$source == "csi") {
    # 日期 指数代码 指数名称 指数英文名称 成分券代码 成分券名称 成分券英文名称 交易所
    # 交易所英文名称 权重
    data <- aktools(
      key = "index_stock_cons_weight_csindex",
      symbol = index$index
    ) %>%
      mutate(
        date = !!curr_td,
        index = !!index$index,
        index_name = !!index$index_name,
        symbol = `成分券代码`,
        name = `成分券名称`,
        weight = as.numeric(`权重`)
      )
  } else if (index$source == "cni") {
    # 日期 样本代码 样本简称 所属行业 总市值 权重
    data <- aktools(
      key = "index_detail_cni",
      symbol = index$index
    ) %>%
      filter(`日期` == max(`日期`)) %>%
      mutate(
        date = !!curr_td,
        index = !!index$index,
        index_name = !!index$index_name,
        symbol = `样本代码`,
        name = `样本简称`,
        weight = as.numeric(`权重`)
      )
  }
  data %>%
    select(date, index, index_name, symbol, name, weight) %>%
    arrange(desc(weight), symbol)
}

# https://quote.eastmoney.com/center/gridlist.html#hs_a_board
get_spot <- function() {
  curr_td <- eval(curr_td_expr)
  # 序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高 最低 今开 昨收 量比 换手率
  # 市盈率-动态 市净率 总市值 流通市值 涨速 5分钟涨跌 60日涨跌幅 年初至今涨跌幅
  aktools(
    key = "stock_zh_a_spot_em"
  ) %>%
    mutate(
      symbol = `代码`,
      name = `名称`,
      date = !!curr_td,
      open = as.numeric(`今开`),
      high = as.numeric(`最高`),
      low = as.numeric(`最低`),
      close = as.numeric(`最新价`),
      volume = as.numeric(`成交量`),
      amount = as.numeric(`成交额`),
      to = as.numeric(`换手率`)
    ) %>%
    select(symbol, name, date, open, high, low, close, volume, amount, to) %>%
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
  curr_td <- eval(curr_td_expr)
  # 序号 代码 名称 停牌时间 停牌截止时间 停牌期限 停牌原因 所属市场 预计复牌时间
  aktools(
    key = "stock_tfp_em",
    date = format(curr_td, "%Y%m%d")
  ) %>%
    mutate(
      symbol = `代码`,
      susp = ifelse(
        `停牌时间` <= !!curr_td &
          (`停牌截止时间` >= !!curr_td | is.na(`停牌截止时间`)),
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
  curr_td <- eval(curr_td_expr)
  m_delay <- ifelse(month(curr_td) == 1, 3, 0)
  list(
    loop_function("get_adjust_change", curr_td %m-% months(m_delay)),
    loop_function("get_adjust_change", curr_td %m-% months(3 + m_delay)),
    loop_function("get_adjust_change", curr_td %m-% months(6 + m_delay)),
    loop_function("get_adjust_change", curr_td %m-% months(9 + m_delay))
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
  curr_td <- eval(curr_td_expr)
  list(
    loop_function("get_val_change", curr_td),
    loop_function("get_val_change", curr_td %m-% months(3)),
    loop_function("get_val_change", curr_td %m-% months(6)),
    loop_function("get_val_change", curr_td %m-% months(9))
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
  loop_function("get_spot") %>%
    left_join(loop_function("get_delist"), by = "symbol") %>%
    left_join(loop_function("get_susp"), by = "symbol") %>%
    left_join(combine_adjust_change(), by = "symbol") %>%
    left_join(loop_function("get_shares_change"), by = "symbol") %>%
    left_join(combine_val_change(), by = "symbol") %>%
    mutate(
      delist = replace_na(delist, FALSE),
      susp = replace_na(susp, FALSE)
    ) %>%
    filter(str_detect(symbol, "^(0|3|6)")) %>%
    arrange(symbol)
}

# HISTORICAL DATA ==============================================================

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
      amount = as.numeric(`成交额`),
      to = as.numeric(`换手率`)
    ) %>%
    select(date, open, high, low, close, volume, amount, to) %>%
    arrange(date)
}

# https://finance.sina.com.cn/realstock/company/sh600006/nc.shtml(示例)
get_adjust <- function(symbol) {
  last_td <- eval(last_td_expr)
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
    adjust = "hfq-factor"
  ) %>%
    mutate(
      date = as_date(date),
      adjust = as.numeric(hfq_factor)
    ) %>%
    select(date, adjust) %>%
    arrange(date) %>%
    rbind(mutate(last(.), date = !!last_td)) %>%
    distinct(date, .keep_all = TRUE)
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
  last_td <- eval(last_td_expr)
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
      val_change_date = !!last_td,
      revenue = as.numeric(TOTALOPERATEREVE),
      np = as.numeric(PARENTNETPROFIT),
      np_deduct = as.numeric(DEDU_PARENT_PROFIT),
      bvps = as.numeric(BPS),
      cfps = as.numeric(PER_NETCASH)
    ) %>%
    select(date, val_change_date, revenue, np, np_deduct, bvps, cfps) %>%
    arrange(date)
}
