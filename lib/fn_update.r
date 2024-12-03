update <- function(
  symbol_list = out0[["symbol_list"]],
  data_list = out0[["data_list"]],
  portfolio_path = "assets/portfolio.csv",
  ranking_path = "assets/ranking.txt"
) {
  tsprint("Started update().")

  # Define parameters
  t_adx <- 15
  t_cci <- 30
  x_thr <- 0.53
  t_max <- 105
  r_max <- 0.09
  r_min <- -0.5

  out <- em_data_update()
  data_update <- out[[1]] %>% .[.$symbol %in% symbol_list, ]
  data_name <- out[[2]]

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  data_list <- foreach(
    symbol = symbol_list,
    .combine = append,
    .export = c("tnormalize", "adx_alt", "predictor", "t_adx", "t_cci"),
    .packages = c("tidyverse", "TTR")
  ) %dopar% {
    rm("data", "df", "lst")

    data <- data_list[[symbol]]
    df <- data_update[data_update$symbol == symbol, ]
    if (
      !all((data[nrow(data), 2:7] == df[, 2:7]) %in% TRUE)
    ) {
      ifelse(
        data[nrow(data), "date"] == df$date,
        data[nrow(data), ] <- df,
        data <- bind_rows(data, df)
      )
    }
    if (any(is.na(data[nrow(data), ]))) data <- data[-nrow(data), ]

    data <- predictor(data)

    lst <- list()
    lst[[symbol]] <- data
    return(lst)
  }
  unregister_dopar

  tsprint(glue("Checked {length(data_list)} stocks for update."))

  fundflow_dict <- data.frame(
    indicator = c("今日", "3日", "5日", "10日"),
    header = c("in1", "in3", "in5", "in10")
  )

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  fundflow <- foreach(
    indicator = fundflow_dict$indicator,
    .export = "em_fundflow",
    .packages = c("jsonlite", "RCurl", "tidyverse")
  ) %dopar% {
    rm("fundflow")
    fundflow <- em_fundflow(indicator, fundflow_dict) %>%
      .[.$symbol %in% symbol_list, ]
    return(fundflow)
  }
  unregister_dopar

  fundflow <- reduce(fundflow, full_join, by = "symbol")
  fundflow[, fundflow_dict$header] <- fundflow[, fundflow_dict$header] / 100
  tsprint(glue("Downloaded fundflow of {nrow(fundflow)} stocks."))

  latest <- bind_rows(
    lapply(
      data_list,
      function(df) {
        merge(data_name, df[nrow(df), ], by = "symbol") %>%
          merge(fundflow, by = "symbol")
      }
    )
  )
  # [1]   symbol name date open high low close volume x x1
  # [11]  dx in1 in3 in5 in10
  latest <- latest[, c(3, 1, 2, 4:ncol(latest))] %>%
    .[, !colnames(.) %in% c("open", "high", "low", "close", "volume")]
  latest$in_score <- apply(latest[, fundflow_dict$header], 1, sum)
  latest <- latest[order(latest$in_score, decreasing = TRUE), ]

  ranking <- latest[latest$x >= x_thr & latest$x1 < x_thr & latest$dx > 0, ] %>%
    .[order(desc(.$in_score)), ] %>%
    na.omit(.)
  ranking[, sapply(ranking, is.numeric)] <- format(
    round(ranking[, sapply(ranking, is.numeric)], 2), nsmall = 2
  )
  cat(
    capture.output(print(ranking, row.names = FALSE)) %>%
      gsub("     name", "    name", .) %>%
      gsub("^ ", "", .),
    file = ranking_path,
    sep = "\n"
  )
  tsprint(
    glue(
      "Ranked {nrow(latest)} stocks; wrote {nrow(ranking)} to {ranking_path}."
    )
  )

  # Evaluate portfolio
  out0 <- list(
    symbol_list = symbol_list, data_list = data_list, latest = latest
  )
  if (!file.exists(portfolio_path)) return(out0)

  portfolio <- read.csv(
    portfolio_path,
    colClasses = c(date = "Date", symbol = "character")
  )
  if (nrow(portfolio) == 0) return(out0)

  out <- data.frame()
  for (symbol in portfolio$symbol) {
    df <- portfolio[portfolio$symbol == symbol, ]
    data <- data_list[[symbol]]

    i <- which(data$date == df$date)
    j <- nrow(data)
    r <- ror(df$cost, data[j, "close"])
    out <- rbind(
      out, list(
        df$date,
        symbol,
        latest[latest$symbol == symbol, "name"],
        df$cost,
        r,
        ifelse(r >= r_max | r <= r_min | j - i >= t_max, "SELL", "HOLD")
      )
    )
  }
  colnames(out) <- c("date", "symbol", "name", "cost", "r", "action")
  out$date <- as.character(as.Date(out$date))
  out <- arrange(out, desc(action), desc(r))
  out[, c("cost", "r")] <- format(round(out[, c("cost", "r")], 3), nsmall = 3)
  rownames(out) <- seq_len(nrow(out))

  v <- which(out$action == "SELL")
  if (length(v) != 0) {
    out <- rbind(
      out[v, ],
      setNames(
        data.frame(t(replicate(ncol(out), "")), row.names = ""), names(out)
      ),
      out[-v, ]
    )
  }
  print(out)

  return(out0)
}