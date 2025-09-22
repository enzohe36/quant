Sys.setlocale(locale = "Chinese")
Sys.setenv(TZ = "Asia/Shanghai")

options(warn = -1)

################################################################################
# Data update functions
################################################################################

get_index_spot <- function() {
  Sys.sleep(1)
  list(
    # 序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高 最低 今开 昨收 量比
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_index_spot_em",
      symbol = "上证系列指数",
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(market = "sh"),
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_index_spot_em",
      symbol = "深证系列指数",
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(market = "sz"),
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_index_spot_em",
      symbol = "中证系列指数",
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(market = "csi")
  ) %>%
    rbindlist() %>%
    mutate(
      symbol = `代码`,
      name = `名称`,
      market = market,
      date = as_tradedate(now() - hours(16)),
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
    arrange(symbol)
}

get_index_hist <- function(symbol, start_date, end_date) {
  Sys.sleep(1)
  # date open close high low volume amount
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_index_daily_em",
    symbol = read_csv("data/index_spot.csv", show_col_types = FALSE) %>%
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
  # 日期 指数代码 指数名称 指数英文名称 成分券代码 成分券名称 成分券英文名称 交易所
  # 交易所英文名称 权重
  getForm(
    uri = "http://127.0.0.1:8080/api/public/index_stock_cons_weight_csindex",
    symbol = symbol,
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      symbol = `成分券代码`,
      date = as_date(`日期`),
      index = !!symbol,
      index_weight = `权重` / 100
    ) %>%
    select(symbol, date, index, index_weight) %>%
    arrange(symbol)
}

get_symbols <- function() {
  Sys.sleep(1)
  list(
    # code name
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_info_a_code_name",
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(
        symbol = code,
        date = as_tradedate(now() - hours(16)),
        delist = FALSE
      ) %>%
      select(symbol, date, delist),
    # 公司代码 公司简称 上市日期 暂停上市日期
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_info_sh_delist",
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(
        symbol = `公司代码`,
        date = as_tradedate(now() - hours(16)),
        delist = TRUE
      ) %>%
      select(symbol, date, delist),
    # 证券代码 证券简称 上市日期 终止上市日期
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_info_sz_delist",
      .encoding = "utf-8"
    ) %>%
      fromJSON() %>%
      mutate(
        symbol = `证券代码`,
        date = as_tradedate(now() - hours(16)),
        delist = TRUE
      ) %>%
      select(symbol, date, delist)
  ) %>%
    rbindlist() %>%
    unique() %>%
    arrange(symbol)
}

get_susp <- function(date) {
  Sys.sleep(1)
  # 序号 代码 名称 停牌时间 停牌截止时间 停牌期限 停牌原因 所属市场 预计复牌时间
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_tfp_em",
    date = format(date, "%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      symbol = `代码`,
      date = !!date,
      susp = ifelse(
        `停牌时间` <= !!date & (`停牌截止时间` >= !!date | is.na(`停牌截止时间`)),
        TRUE,
        FALSE
      )
    ) %>%
    select(symbol, date, susp) %>%
    arrange(symbol)
}

get_spot <- function() {
  Sys.sleep(1)
  # 序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高 最低 今开 昨收 量比 换手率
  # 市盈率-动态 市净率 总市值 流通市值 涨速 5分钟涨跌 60日涨跌幅 年初至今涨跌幅
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_a_spot_em",
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      symbol = `代码`,
      name = `名称`,
      date = as_tradedate(now() - hours(16)),
      open = `今开`,
      high = `最高`,
      low = `最低`,
      close = `最新价`,
      volume = `成交量`,
      amount = `成交额`,
      shares = round(`总市值` / `最新价`),
      shares_float = round(`流通市值` / `最新价`)
    ) %>%
    select(
      symbol, name, date, open, high, low, close, volume, amount,
      shares, shares_float
    ) %>%
    arrange(symbol)
}

get_hist <- function(symbol, start_date, end_date) {
  Sys.sleep(1)
  # 日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
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
    uri = "http://127.0.0.1:8080/api/public/stock_zh_a_daily",
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

get_shares <- function(symbol) {
  Sys.sleep(1)
  # 变更日期 总股本 流通受限股份 其他内资持股(受限) 境内法人持股(受限) 境内自然人持股(受限)
  # 已流通股份 已上市流通A股 变动原因
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_a_gbjg_em",
    symbol = paste0(
      symbol,
      case_when(
        str_detect(symbol, "^6") ~ ".SH",
        str_detect(symbol, "^(0|3)") ~ ".SZ",
        TRUE ~ ".BJ"
      )
    ),
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      date = as_date(`变更日期`),
      shares = `总股本`,
      shares_float = `已上市流通A股`
    ) %>%
    select(date, shares, shares_float) %>%
    arrange(date)
}

get_div <- function(date) {
  Sys.sleep(1)
  quarter <- quarter(date %m-% months(3), "date_last")
  # 代码 名称 送转股份-送转总比例 送转股份-送转比例 送转股份-转股比例 现金分红-现金分红比例
  # 现金分红-股息率 每股收益 每股净资产 每股公积金 每股未分配利润 净利润同比增长 总股本
  # 预案公告日 股权登记日 除权除息日 方案进度 最新公告日期
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_fhps_em",
    date = format(quarter, "%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      symbol = `代码`,
      quarter = !!quarter,
      dps = `现金分红-现金分红比例` / 10,
      exright_date = as_date(`除权除息日`)
    ) %>%
    select(symbol, quarter, dps, exright_date) %>%
    arrange(symbol)
}

get_shares_change <- function() {
  Sys.sleep(1)
  # 证券代码 证券简称 交易市场 公告日期 变动日期 变动原因 总股本 已流通股份 已流通比例
  # 流通受限股份
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_hold_change_cninfo",
    symbol = "全部",
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      symbol = `证券代码`,
      shares_change = as.Date(`变动日期`)
    ) %>%
    select(symbol, quarter, dps, exright_date) %>%
    arrange(symbol)
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
    uri = "http://127.0.0.1:8080/api/public/stock_financial_analysis_indicator_em",
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
      quarter = as_date(REPORT_DATE),
      revenue = TOTALOPERATEREVE,
      np = PARENTNETPROFIT,
      np_deduct = DEDU_PARENT_PROFIT,
      bvps = BPS,
      cfps = PER_NETCASH
    ) %>%
    select(quarter, revenue, np, np_deduct, bvps, cfps) %>%
    arrange(quarter)
}

################################################################################
# Math functions
################################################################################

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

run_pct <- function(v, n) {
  v <- unlist(v)
  sapply(
    seq_len(length(v)),
    function(i) {
      if (i < n) return(NaN)
      v[(i - n + 1):i] %>% percent_rank() %>% last()
    }
  )
}

get_roc <- function(v1, v2) (v2 - v1) / v1

get_rmse <- function(v1, v2) sqrt(sum((v2 - v1) ^ 2) / length(v1))

runwhich_min <- function(v, n) {
  rollapply(
    v, width = n, align = "right", fill = NA,
    FUN = function(w) which.min(w)
  )
}

runwhich_max <- function(v, n) {
  rollapply(
    v, width = n, align = "right", fill = NA,
    FUN = function(w) which.max(w)
  )
}

runmax_var <- function(x, widths) {
  stopifnot(length(x) == length(widths))
  sapply(
    seq_along(x), function(i) {
      w <- widths[i]
      if (i < w | is.na(i < w)) return(NA)
      max(x[(i - w + 1):i], na.rm = TRUE)
    }
  )
}

runmin_var <- function(x, widths) {
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

# http://www.cftsc.com/qushizhibiao/610.html
get_adx <- function(hlc, n = 14, m = 6) {
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
  diff <- abs(adx - adxr)
  out <- data.frame(adx = adx, adxr = adxr, diff = diff)
  return(out)
}

add_mom <- function(data, var_list, lag_list) {
  for (i in lag_list) {
    data <- mutate(
      data,
      across(
        !!var_list,
        ~ momentum(., i),
        .names = "{.col}_mom{i}"
      )
    )
  }
  return(data)
}

add_roc <- function(data, var_list, lag_list) {
  for (i in lag_list) {
    data <- mutate(
      data,
      across(
        !!var_list,
        ~ ROC(., i, "discrete"),
        .names = "{.col}_roc{i}"
      )
    )
  }
  return(data)
}

add_rocnorm <- function(data, var_list, lag_list, n) {
  for (i in lag_list) {
    data <- mutate(
      data,
      across(
        !!var_list,
        ~ ROC(., i, "discrete") %>% run_norm(n),
        .names = "{.col}_rocnorm{i}"
      )
    )
  }
  return(data)
}

add_sma <- function(data, var_list, lag_list) {
  for (i in lag_list) {
    data <- mutate(
      data,
      across(
        !!var_list,
        ~ SMA(., i),
        .names = "{.col}_sma{i}"
      )
    )
  }
  return(data)
}

add_sd <- function(data, var_list, lag_list) {
  for (i in lag_list) {
    data <- mutate(
      data,
      across(
        !!var_list,
        ~ runSD(., i),
        .names = "{.col}_sd{i}"
      )
    )
  }
  return(data)
}

################################################################################
# Utility functions
################################################################################

# https://stackoverflow.com/a/25110203
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list = ls(name = env), pos = env)
}

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
      seq(date - weeks(2), date, "1 day") %>%
        .[!wday(., week_start = 1) %in% 6:7] %>%
        .[!.%in% holidays] %>%
        last()
    }
  ) %>%
    reduce(c) %>%
    unique()
  return(tradedate)
}

predict_probrf <- function(rf, test) {
  predict(rf, test)[["predictions"]] %>%
    as_data_frame() %>%
    mutate(
      prob_max = apply(., 1, function(v) max(v)),
      pred = apply(., 1, function(v) names(v) %>% .[match(max(v), v)]) %>%
        as.factor(),
      target = !!test$target,
      symbol = !!test$symbol,
      date = !!test$date
    )
}
