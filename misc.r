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
        .[!wday(.) %in% c(1, 7)] %>%
        .[!. %in% holidays] %>%
        .[. <= date] %>%
        last()
    }
  ) %>%
    reduce(c) %>%
    unique()
  return(tradedate)
}

get_index <- function(symbol, start_date, end_date) {
  # date open close high low volume amount
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_index_daily_em",
    symbol = paste0(ifelse(grepl("^3", symbol), "sz", "sh"), symbol),
    start_date = format(start_date, "%Y%m%d"),
    end_date = format(end_date, "%Y%m%d"),
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      date = as_date(date),
      vol = volume,
      amt = amount
    ) %>%
    select(date, open, high, low, close, vol, amt)
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
      symbol = `成分券代码`,
      name = `成分券名称`,
      index = !!symbol,
      index_weight = `权重` / 100
    ) %>%
    select(symbol, name, index, index_weight)
}

get_hist <- function(symbol, start_date, end_date, adjust) {
  # 日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
    symbol = symbol,
    period = "daily",
    start_date = format(start_date, "%Y%m%d"),
    end_date = format(end_date, "%Y%m%d"),
    adjust = adjust,
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      date = as_date(`日期`),
      open = `开盘`,
      high = `最高`,
      low = `最低`,
      close = `收盘`,
      vol = `成交量`,
      amt = `成交额`
    ) %>%
    select(date, open, high, low, close, vol, amt)
}

get_adjust_factors <- function(symbol, start_date, end_date, adjust) {
  # date qfq_factor
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_a_daily",
    symbol = paste0(ifelse(grepl("^6", symbol), "sh", "sz"), symbol),
    start_date = format(start_date, "%Y%m%d"),
    end_date = format(end_date, "%Y%m%d"),
    adjust = paste0(adjust, "-factor"),
    .encoding = "utf-8"
  ) %>%
    fromJSON() %>%
    mutate(
      date = as_date(date),
      adjust_factor = get(!!paste0(adjust, "_factor"))
    ) %>%
    select(date, adjust_factor)
}

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

# get_fundflow <- function(symbol, mkt) {
#   # 日期 收盘价 涨跌幅 主力净流入-净额 主力净流入-净占比 超大单净流入-净额
#   # 超大单净流入-净占比 大单净流入-净额 大单净流入-净占比 中单净流入-净额 中单净流入-净占比
#   # 小单净流入-净额 小单净流入-净占比
#   getForm(
#     uri = "http://127.0.0.1:8080/api/public/stock_individual_fund_flow",
#     stock = symbol,
#     market = mkt,
#     .encoding = "utf-8"
#   ) %>%
#     fromJSON() %>%
#     mutate(
#       date = as_date(`日期`),
#       amt_xl = `超大单净流入-净额`,
#       amt_l = `大单净流入-净额`,
#       amt_m = `中单净流入-净额`,
#       amt_s = `小单净流入-净额`,
#       across(!matches("^[a-z0-9_]+$"), ~ NULL)
#     )
# }

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
#       mktcap = `总市值`,
#       mktcap_float = `流通市值`,
#       pe = `PE(TTM)`,
#       pb = `市净率`,
#       peg = `PEG值`,
#       pocf = `市现率`,
#       ps = `市销率`,
#       across(!matches("^[a-z0-9_]+$"), ~NULL)
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
