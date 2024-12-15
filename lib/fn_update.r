update <- function(
  .data_list = data_list,
  portfolio_path = "assets/portfolio.csv",
  ranking_path = "tmp/ranking.txt"
) {
  tsprint("Started update().")

  # Define parameters
  t_adx <- 15
  t_cci <- 30
  xa_thr <- 0.5
  xb_thr <- 0.5
  t_max <- 105
  r_max <- 0.09
  r_min <- -0.5

  symbol_list <- names(.data_list)
  out <- em_data_update()
  data_update <- out[[1]] %>% .[.$symbol %in% symbol_list, ]
  data_name <- out[[2]]

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  .data_list <- foreach(
    symbol = symbol_list,
    .combine = append,
    .export = c(
      "get_predictor", "normalize", "tnormalize", "ADX", "ROR"
    ),
    .packages = c("TTR", "signal", "tidyverse")
  ) %dopar% {
    rm("data", "df", "lst")

    data <- .data_list[[symbol]]
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

    data <- get_predictor(data, t_adx, t_cci)

    lst <- list()
    lst[[symbol]] <- data
    return(lst)
  }
  unregister_dopar

  tsprint(glue("Checked {length(.data_list)} stocks for update."))

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
  tsprint(glue("Found fundflow of {nrow(fundflow)} stocks."))

  latest <- bind_rows(
    lapply(
      .data_list,
      function(df) {
        merge(data_name, df[nrow(df), ], by = "symbol") %>%
          merge(fundflow, by = "symbol")
      }
    )
  )
  # [1]   symbol name date open high low close volume xa xa1
  # [11]  xad xb xb1 xbd sgd in1 in3 in5 in10
  latest <- latest[, c(3, 1, 2, 4:ncol(latest))]
  latest$score <- apply(
    latest[, fundflow_dict$header], 1, function(v) {
      v[1] * 0.4 + v[2] * 0.3 + v[3] * 0.2 + v[4] * 0.1
    }
  )
  latest <- latest[order(latest$score, decreasing = TRUE), ]

  dir.create("tmp/")

  ranking <- latest[
    (
      (latest$xa >= xa_thr & latest$xa1 < xa_thr & latest$xad > 0) |
        (latest$xb >= xb_thr & latest$xb1 < xb_thr & latest$xbd > 0)
    ) & (
      latest$sgd <= 0
    ),
  ] %>%
    .[order(desc(.$score)), ] %>%
    na.omit(.) %>%
    .[,
      colnames(.) %in% c(
        "date", "symbol", "name",
        "xa", "xb", "sgd", fundflow_dict$header, "score"
      )
    ]
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
  out <- list(data_list = .data_list, latest = latest)
  if (!file.exists(portfolio_path)) return(out)

  portfolio <- read.csv(
    portfolio_path,
    colClasses = c(buy = "Date", symbol = "character")
  )
  if (nrow(portfolio) == 0) return(out)

  df <- data.frame()
  for (portfolio_i in split(portfolio, portfolio$symbol)) {
    symbol <- portfolio_i$symbol
    data <- .data_list[[symbol]]
    if (!any(data$date == portfolio_i$buy)) {
      stop(glue("Purchase date of {symbol} does not exist"))
    }
    r <- ROR(portfolio_i$cost, last(data$close))
    df <- rbind(
      df, list(
        portfolio_i$buy,
        symbol,
        latest[latest$symbol == symbol, "name"],
        portfolio_i$cost,
        r,
        ifelse(
          r >= r_max | r <= r_min | last(data$date) - portfolio_i$buy >= t_max,
          "SELL",
          "HOLD"
        )
      )
    )
  }
  df <- `colnames<-`(df, c("buy", "symbol", "name", "cost", "r", "action")) %>%
    arrange(desc(action), desc(r)) %>%
    mutate(
      buy = as.character(as_date(buy)),
      cost = format(round(cost, 3), nsmall = 3),
      r = format(round(r, 3), nsmall = 3)
    ) %>%
    `rownames<-`(seq_len(nrow(.)))

  v <- which(df$action == "SELL")
  if (length(v) != 0) {
    df <- rbind(
      df[v, ],
      setNames(
        data.frame(t(replicate(ncol(df), "")), row.names = ""), names(df)
      ),
      df[-v, ]
    )
  }
  print(df)

  return(out)
}