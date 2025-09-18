Sys.setlocale(locale = "Chinese")
Sys.setenv(TZ = "Asia/Shanghai")

options(warn = -1)

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
  holidays <- read_csv("holidays.csv", show_col_types = FALSE) %>%
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

get_indexspot <- function() {
  # 序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高 最低 今开 昨收 量比
  list(
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
      open = `今开`,
      high = `最高`,
      low = `最低`,
      close = `最新价`,
      volume = `成交量`,
      amount = `成交额`
    ) %>%
    select(market, symbol, name, open, high, low, close, volume, amount)
}

get_indexhist <- function(symbol, start_date, end_date) {
  # date open close high low volume amount
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_index_daily_em",
    symbol = read_csv("indices.csv", show_col_types = FALSE) %>%
      filter(symbol == !!symbol) %>%
      pull(market) %>%
      paste0(symbol),
    start_date = format(start_date, "%Y%m%d"),
    end_date = format(end_date, "%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(date = as_date(date)) %>%
    select(date, open, high, low, close, volume, amount)
}

get_indexcomp <- function(symbol) {
  # 日期 指数代码 指数名称 指数英文名称 成分券代码 成分券名称 成分券英文名称 交易所
  # 交易所英文名称 权重
  getForm(
    uri = "http://127.0.0.1:8080/api/public/index_stock_cons_weight_csindex",
    symbol = symbol,
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      index = !!symbol,
      symbol = `成分券代码`,
      name = `成分券名称`,
      weight = `权重` / 100
    ) %>%
    select(index, symbol, name, weight)
}

get_div <- function(date) {
  # 代码 名称 送转股份-送转总比例 送转股份-送转比例 送转股份-转股比例 现金分红-现金分红比例
  # 现金分红-股息率 每股收益 每股净资产 每股公积金 每股未分配利润 净利润同比增长 总股本
  # 预案公告日 股权登记日 除权除息日 方案进度 最新公告日期
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_fhps_em",
    date = format(quarter(date, "date_last"), "%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      symbol = `代码`,
      div = `现金分红-现金分红比例` / 10,
      exright_date = as_date(`除权除息日`)
    ) %>%
    select(symbol, div, exright_date)
}

get_spot <- function() {
  # 序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高 最低 今开 昨收 量比 换手率 市盈率-动态 市净率 总市值 流通市值 涨速 5分钟涨跌 60日涨跌幅 年初至今涨跌幅
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_a_spot_em",
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      symbol = `代码`,
      name = `名称`,
      open = `今开`,
      high = `最高`,
      low = `最低`,
      close = `最新价`,
      volume = `成交量`,
      amount = `成交额`,
      turnover = round(`换手率` / 100, 4)
    ) %>%
    select(symbol, name, open, high, low, close, volume, amount, turnover)
}

get_hist <- function(symbol, start_date, end_date) {
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
      amount = `成交额`,
      turnover = round(`换手率` / 100, 4)
    ) %>%
    select(date, open, high, low, close, volume, amount, turnover)
}

get_adjust <- function(symbol) {
  # date hfq_factor
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_a_daily",
    symbol = paste0(ifelse(grepl("^6", symbol), "sh", "sz"), symbol),
    adjust = "hfq-factor",
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(date = as_date(date)) %>%
    rename(adjust = hfq_factor)
}

# get_val <- function(symbol) {
#   # 数据日期 当日收盘价 当日涨跌幅 总市值 流通市值 总股本 流通股本 PE(TTM) PE(静) 市净率
#   # PEG值 市现率 市销率
#   getForm(
#     uri = "http://127.0.0.1:8080/api/public/stock_value_em",
#     symbol = symbol,
#     .encoding = "utf-8"
#   ) %>%
#     fromJSON() %>%
#     mutate(
#       date = as_date(`数据日期`),
#       mktcap = `流通市值`,
#       pe = `PE(TTM)`,
#       pb = `市净率`,
#       peg = `PEG值`,
#       pcf = `市现率`,
#       ps = `市销率`,
#       across(!matches("^[a-z0-9_]+$"), ~ NULL)
#     )
# }

# get_mktcost <- function(symbol, adjust) {
#   # 日期 获利比例 平均成本 90成本-低 90成本-高 90集中度 70成本-低 70成本-高 70集中度
#   getForm(
#     uri = "http://127.0.0.1:8080/api/public/stock_cyq_em",
#     symbol = symbol,
#     adjust = adjust,
#     .encoding = "utf-8"
#   ) %>%
#     fromJSON() %>%
#     mutate(
#       date = as_date(`日期`),
#       mktcost = `平均成本`,
#       profitable = `获利比例`,
#       cr70 = `70集中度`,
#       cr90 = `90集中度`,
#       across(!matches("^[a-z0-9_]+$"), ~ NULL)
#     )
# }

# get_fundflow <- function(symbol, market) {
#   # 日期 收盘价 涨跌幅 主力净流入-净额 主力净流入-净占比 超大单净流入-净额
#   # 超大单净流入-净占比 大单净流入-净额 大单净流入-净占比 中单净流入-净额 中单净流入-净占比
#   # 小单净流入-净额 小单净流入-净占比
#   getForm(
#     uri = "http://127.0.0.1:8080/api/public/stock_individual_fund_flow",
#     stock = symbol,
#     market = market,
#     .encoding = "utf-8"
#   ) %>%
#     fromJSON() %>%
#     mutate(
#       date = as_date(`日期`),
#       amount_xl = `超大单净流入-净额`,
#       amount_l = `大单净流入-净额`,
#       amount_m = `中单净流入-净额`,
#       amount_s = `小单净流入-净额`,
#       across(!matches("^[a-z0-9_]+$"), ~ NULL)
#     )
# }

# get_val_est <- function(symbol) {
#   # 序号 股票代码 股票简称 报告名称 东财评级 机构 近一月个股研报数
#   # 2024-盈利预测-收益 2024-盈利预测-市盈率 2025-盈利预测-收益 2025-盈利预测-市盈率
#   # 2026-盈利预测-收益 2026-盈利预测-市盈率 行业 日期
#   getForm(
#     uri = "http://127.0.0.1:8080/api/public/stock_research_report_em",
#     symbol = symbol,
#     .encoding = "utf-8"
#   ) %>%
#     fromJSON() %>%
#     rename_with(
#       ~ str_replace_all(., "日期", "date") %>%
#         str_replace_all("东财评级", "rating") %>%
#         str_replace_all(".*收益", paste0("eps_", str_extract(., "[0-9]+"))) %>%
#         str_replace_all(".*市盈率", paste0("pe_", str_extract(., "[0-9]+")))
#     ) %>%
#     mutate(
#       date = as_date(date),
#       across(!matches("^[a-z0-9_]+$"), ~ NULL)
#     )
# }

# get_treasury <- function(start_date) {
#   # 日期 中国国债收益率2年 中国国债收益率5年 中国国债收益率10年 中国国债收益率30年
#   # 中国国债收益率10年-2年 中国GDP年增率 美国国债收益率2年 美国国债收益率5年
#   # 美国国债收益率10年 美国国债收益率30年 美国国债收益率10年-2年 美国GDP年增率
#   getForm(
#     uri = "http://127.0.0.1:8080/api/public/bond_zh_us_rate",
#     start_date = start_date,
#     .encoding = "utf-8"
#   ) %>%
#     fromJSON() %>%
#     rename_with(
#       ~ str_replace_all(., "日期", "date")%>%
#         str_replace_all("中国", "cn")%>%
#         str_replace_all("美国", "us")%>%
#         str_replace_all("国债收益率", "")%>%
#         str_replace_all("GDP年增率", "gdp_roc")%>%
#         str_replace_all("年", "y")
#     ) %>%
#     mutate(
#       date = as_date(date),
#       across(where(is.numeric), ~ . / 100),
#       across(contains("-"), ~ NULL)
#     )
# }

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

fit_gaussian <- function(x, y) {
  nls(
    y ~ 1 / (s * sqrt(2 * pi)) * exp(-1 / 2 * ((x - m) / s) ^ 2),
    start = c(s = 1, m = 0)
  )
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
