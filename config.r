# Run the following commands in order:
# python -m aktools
# (Source this script)
# download_history()
# local <- load_history()
# ranking <- update_ranking(local[[1]], local[[2]], local[[3]], local[[4]])
# query(readLines("query.txt"), ranking)

library(jsonlite)
library(RCurl)
library(lubridate)
library(dplyr)
library(TTR)
library(foreach)
library(doParallel)

options(warn = -1)

# To unregister CPU cluster after foreach dopar
# https://stackoverflow.com/q/25097729
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

# To download historical data of selected stocks
download_history <- function(symbol_pattern = "^(00|60)") {
  # Download symbol list via AKTools
  # [1]   code name
  symbol_list <- fromJSON(getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_info_a_code_name",
      .encoding = "utf-8"
    )
  )
  symbol_list <- symbol_list[, 1]
  print(paste(length(symbol_list), "stocks were found."))
  symbol_list <- symbol_list[grepl(symbol_pattern, symbol_list)]
  print(paste(length(symbol_list), "stocks matched selection."))

  #symbol_list <- sample(symbol_list, 10) # FOR TESTING ONLY

  # Download historical data via AKTools
  dir.create("data")
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  symbol_list <- foreach(
    symbol = symbol_list,
    .combine = "c",
    .errorhandling = "remove",
    .packages = c("jsonlite", "RCurl", "lubridate")
  ) %dopar% {
    # [1]   日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅
    # [11]  涨跌额 换手率
    data <- fromJSON(getForm(
        uri = "http://127.0.0.1:8080/api/public/stock_zh_a_hist",
        symbol = symbol,
        adjust = "qfq",
        start_date = format(now(tzone = "Asia/Shanghai") - years(5), "%Y%m%d"),
        end_date = format(now(tzone = "Asia/Shanghai"), "%Y%m%d"),
        .encoding = "utf-8"
      )
    )
    data <- data[, c(1, 2, 5, 6, 4, 7)]
    colnames(data) <- c("date", "symbol", "high", "low", "close", "volume")
    data[, 1] <- as.Date(data[, 1])

    # Skip stock with < 5 yr of history
    if (data[1, 1] > now(tzone = "Asia/Shanghai") - years(5) + days(2)) {
      next
    }

    # Write historical data to file
    write.csv(data, paste0("data/", symbol, ".csv"), row.names = FALSE)

    return(symbol)
  }
  unregister_dopar

  # Write symbol list of downloaded stocks to file
  writeLines(symbol_list, "symbol_list.txt")
  print(paste(length(symbol_list), "stocks had >= five years of history."))
}

# To load local historical data into list
load_history <- function(period = 5) {
  # Read symbol list from file
  symbol_list <- readLines("symbol_list.txt")

  # Calculate return & predictor
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  data_rbind <- foreach(
    symbol = symbol_list,
    .combine = rbind,
    .errorhandling = "remove",
    .packages = c("dplyr", "TTR")
  ) %dopar% {
    # Read historical data from files
    # [1]   date symbol high low close volume
    data <- read.csv(paste0("data/", symbol, ".csv"))
    data[, 1] <- as.Date(data[, 1])
    data[, 2] <- formatC(data$symbol, width = 6, format = "d", flag = "0")

    # Calculate median return
    for (i in 1:period) {
      data <- cbind(
        data, (lead(data$close, i) - data$close) / data$close * 100
      )
      colnames(data)[ncol(data)] <- paste0("temp_", i)
    }
    data$r_med <- apply(
      select(data, matches("temp_[0-9]+")), 1, median, na.rm = TRUE
    )
    data <- data %>% select(-matches("^temp_"))

    # Calculate modified ADX
    data$temp_0 <- ADX(data[, c(3:5)])[, 4]
    for (i in 1:60) {
      data <- cbind(data, lag(data$temp_0, i))
      colnames(data)[ncol(data)] <- paste0("temp_", i)
    }
    data$temp_min <- apply(
      select(data, matches("temp_[0-9]+")), 1, min, na.rm = TRUE
    )
    data$temp_max <- apply(
      select(data, matches("temp_[0-9]+")), 1, max, na.rm = TRUE
    )
    data$adx_mod <- (data$temp_0 - data$temp_min) /
      (data$temp_max - data$temp_min)
    data <- data %>% select(-matches("^temp_"))

    # Calculate modified CCI
    data$temp_0 <- CCI(data[, c(3:5)])
    for (i in 1:60) {
      data <- cbind(data, lag(data$temp_0, i))
      colnames(data)[ncol(data)] <- paste0("temp_", i)
    }
    data$temp_min <- apply(
      select(data, matches("temp_[0-9]+")), 1, min, na.rm = TRUE
    )
    data$temp_max <- apply(
      select(data, matches("temp_[0-9]+")), 1, max, na.rm = TRUE
    )
    data$cci_mod <- -2 * (data$temp_0 - data$temp_min) /
      (data$temp_max - data$temp_min) + 1
    data <- data %>% select(-matches("^temp_"))

    # Calculate modified stochastics
    data$temp_0 <- stoch(data[, c(3:5)])[, 1]
    data$k_mod <- -2 * data$temp_0 + 1
    data <- data %>% select(-matches("^temp_"))

    # Calculate modified TRIX
    data$temp_0 <- TRIX(data[, 5])[, 1]
    for (i in 1:60) {
      data <- cbind(data, lag(data$temp_0, i))
      colnames(data)[ncol(data)] <- paste0("temp_", i)
    }
    data$temp_min <- apply(
      select(data, matches("temp_[0-9]+")), 1, min, na.rm = TRUE
    )
    data$temp_max <- apply(
      select(data, matches("temp_[0-9]+")), 1, max, na.rm = TRUE
    )
    data$trix_mod <- -2 * (data$temp_0 - data$temp_min) /
      (data$temp_max - data$temp_min) + 1
    data <- data %>% select(-matches("^temp_"))

    # Calculate correlation coefficient
    data$x <- data$adx_mod * (data$cci_mod + data$k_mod + data$trix_mod) / 3
    fit <- lm(r_med ~ x, data)
    data <- mutate(data, temp_0 = fit$coefficients[2])
    data <- mutate(
      data, temp_p = summary(fit)[["coefficients"]][, "Pr(>|t|)"][2]
    )
    data$poscorr <- ifelse(data$temp_0 > 0 & data$temp_p < 0.01, 1, 0)
    data <- data %>% select(-matches("^temp_"))

    return(data)
  }
  unregister_dopar

  # Calculate linear fit
  fit <- lm(r_med ~ x, data_rbind, subset = poscorr == 1)
  print(summary(fit))
  data_sample <- data_rbind[sample(nrow(data_rbind), 1000), ]
  plot(
    data_sample$x, data_sample$r_med, pch = "•"
  )
  abline(fit, col = 2)

  # Output last six months' data
  # [1]   date symbol high low close volume r_med adx_mod cci_mod k_mod
  # [11]  trix_mod x poscorr
  data_rbind <- data_rbind[, c(1:6, 13)]
  data_rbind <- data_rbind[
    data_rbind$date > now(tzone = "Asia/Shanghai") - months(6),
  ]
  data_list <- split(data_rbind, data_rbind$symbol)

  print(
    "Output: [[1]] symbol_list, [[2]] data_list, [[3]] intercept, [[4]] slope."
  )
  return(list(symbol_list, data_list, fit$coefficients[1], fit$coefficients[2]))
}

# To calculate real-time stock ranking
update_ranking <- function(
  symbol_list, data_list, intercept, slope, period = 5
) {
  # Download real-time data via Aktools
  # [1]   序号 代码 名称 最新价 涨跌幅 涨跌额 成交量 成交额 振幅 最高
  # [11]  最低 今开 昨收 量比 换手率 市盈率-动态 市净率 总市值 流通市值 涨速
  # [21]  5分钟涨跌 60日涨跌幅 年初至今涨跌幅
  update_list <- fromJSON(getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_zh_a_spot_em",
      .encoding = "utf-8"
    )
  )
  update_list <- mutate(update_list, as.Date(now(tzone = "Asia/Shanghai")))
  update_list <- update_list[, c(24, 2, 10, 11, 4, 7)]
  colnames(update_list) <- c("date", "symbol", "high", "low", "close", "volume")
  update_list <- split(update_list, update_list$symbol)

  # Calculate return & predictor
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  data_rbind <- foreach(
    symbol = symbol_list,
    .combine = rbind,
    .errorhandling = "remove",
    .packages = c("dplyr", "TTR")
  ) %dopar% {
    # Load historical data from list
    data <- data_list[[symbol]]

    # Append update if different from last row
    if (!all(update_list[[symbol]] == data[nrow(data), 1:6])) {
      update_list[[symbol]] <- mutate(
        update_list[[symbol]], poscorr = data[nrow(data), "poscorr"]
      )
      data <- rbind(data, update_list[[symbol]])
    }

    # Calculate median return
    for (i in 1:period) {
      data <- cbind(
        data, (lead(data$close, i) - data$close) / data$close * 100
      )
      colnames(data)[ncol(data)] <- paste0("temp_", i)
    }
    data$r_med <- apply(
      select(data, matches("temp_[0-9]+")), 1, median, na.rm = TRUE
    )
    data <- data %>% select(-matches("^temp_"))

    # Calculate modified ADX
    data$temp_0 <- ADX(data[, c(3:5)])[, 4]
    for (i in 1:60) {
      data <- cbind(data, lag(data$temp_0, i))
      colnames(data)[ncol(data)] <- paste0("temp_", i)
    }
    data$temp_min <- apply(
      select(data, matches("temp_[0-9]+")), 1, min, na.rm = TRUE
    )
    data$temp_max <- apply(
      select(data, matches("temp_[0-9]+")), 1, max, na.rm = TRUE
    )
    data$adx_mod <- (data$temp_0 - data$temp_min) /
      (data$temp_max - data$temp_min)
    data <- data %>% select(-matches("^temp_"))

    # Calculate modified CCI
    data$temp_0 <- CCI(data[, c(3:5)])
    for (i in 1:60) {
      data <- cbind(data, lag(data$temp_0, i))
      colnames(data)[ncol(data)] <- paste0("temp_", i)
    }
    data$temp_min <- apply(
      select(data, matches("temp_[0-9]+")), 1, min, na.rm = TRUE
    )
    data$temp_max <- apply(
      select(data, matches("temp_[0-9]+")), 1, max, na.rm = TRUE
    )
    data$cci_mod <- -2 * (data$temp_0 - data$temp_min) /
      (data$temp_max - data$temp_min) + 1
    data <- data %>% select(-matches("^temp_"))

    # Calculate modified stochastics
    data$temp_0 <- stoch(data[, c(3:5)])[, 1]
    data$k_mod <- -2 * data$temp_0 + 1
    data <- data %>% select(-matches("^temp_"))

    # Calculate modified TRIX
    data$temp_0 <- TRIX(data[, 5])[, 1]
    for (i in 1:60) {
      data <- cbind(data, lag(data$temp_0, i))
      colnames(data)[ncol(data)] <- paste0("temp_", i)
    }
    data$temp_min <- apply(
      select(data, matches("temp_[0-9]+")), 1, min, na.rm = TRUE
    )
    data$temp_max <- apply(
      select(data, matches("temp_[0-9]+")), 1, max, na.rm = TRUE
    )
    data$trix_mod <- -2 * (data$temp_0 - data$temp_min) /
      (data$temp_max - data$temp_min) + 1
    data <- data %>% select(-matches("^temp_"))

    # Calculate predictor
    data$x <- data$adx_mod * (data$cci_mod + data$k_mod + data$trix_mod) / 3

    # Clean up output
    # [1]   date symbol high low close volume poscorr r_med adx_mod cci_mod
    # [11]  k_mod trix_mod x
    data <- data[, c(1, 2, 7:13)]
    return(data)
  }
  unregister_dopar

  # Calculate expected median redurn
  ranking <- data_rbind[
    data_rbind$date == as.Date(now(tzone = "Asia/Shanghai")),
  ]
  ranking$r_med_theor <- ifelse(
    ranking$poscorr == 1, intercept + slope * ranking$x, NA
  )
  print(paste(nrow(ranking), "stocks were updated."))
  print(paste(
      nrow(ranking[ranking$poscorr == 1, ]),
      "stocks had positive correlation coefficients."
    )
  )

  # Sort stocks by predictor value
  # [1]   date symbol poscorr r_med adx_mod cci_mod k_mod trix_mod x r_med_theor
  ranking <- ranking[, c(1, 2, 5:10)]
  ranking[, 3:8] <- round(ranking[, 3:8], 4)
  ranking <- ranking[order(ranking$x, decreasing = TRUE), ]
  cat(
    capture.output(print(ranking, row.names = FALSE)),
    file = 'ranking.txt',
    sep = "\n"
  )

  print("Output: df ranking.")
  return(ranking)
}

# To query specific stocks
query <- function(query, ranking) {
  result <- data.frame()
  for (symbol in query) {
    result <- rbind(result, ranking[ranking$symbol == symbol, ])
  }
  result <- result[order(result$x, decreasing = TRUE), ]
  cat(
    capture.output(print(result, row.names = FALSE)),
    file = 'result.txt',
    sep = "\n"
  )
}
