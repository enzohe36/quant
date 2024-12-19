update <- function(
  data_list = get("data_list", envir = .GlobalEnv),
  ranking_path = "tmp/ranking.txt",
  portfolio_path = "assets/portfolio.csv"
) {
  tsprint("Started update().")

  # Define parameters
  t_adx <- 143
  t_cci <- 156
  t_xad <- 5
  t_xbd <- 2
  t_sgd <- 16
  xa_h <- 0.4
  xb_h <- 0.27
  t_max <- 52
  r_max <- 0.06
  r_min <- -0.54

  symbol_list <- names(data_list)
  out <- em_data_update()
  data_update <- out[["data_update"]] %>% filter(symbol %in% symbol_list)
  data_name <- out[["data_name"]]

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  data_list <- foreach(
    data = data_list,
    .combine = "c",
    .export = c("normalize", "tnormalize", "ADX", "ROR", "get_predictor"),
    .packages = c("TTR", "signal", "tidyverse")
  ) %dopar% {
    rm("data_update_i", "lst", "symbol")

    symbol <- data[1, 2]

    data_update_i <- data_update[data_update$symbol == symbol, ]
    if (
      !all(
        select(data[nrow(data), ], is.numeric) ==
          select(data_update_i, is.numeric)
      ) %in%
        TRUE
    ) {
      ifelse(
        data[nrow(data), "date"] == data_update_i$date,
        data[nrow(data), ] <- data_update_i,
        data <- rbind(data, data_update_i)
      )
    }
    if (any(is.na(data[nrow(data), ]))) data <- data[-nrow(data), ]

    data <- get_predictor(data, t_adx, t_cci, t_xad, t_xbd, t_sgd)

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
    rm("fundflow", "header")

    header <- fundflow_dict[fundflow_dict$indicator == indicator, "header"]

    fundflow <- em_fundflow(indicator, header) %>%
      filter(symbol %in% symbol_list) %>%
      mutate
    fundflow[, header] <- fundflow[, header] / 100
    return(fundflow)
  }
  unregister_dopar

  fundflow <- reduce(fundflow, full_join, by = "symbol")
  tsprint(glue("Found fundflow of {nrow(fundflow)} stocks."))

  latest <- rbindlist(
    lapply(
      data_list,
      function(df, data_name, fundflow) {
        symbol <- df[1, "symbol"]
        out <- df[nrow(df), ] %>%
          mutate(
            name = filter(data_name, symbol == !!symbol) %>% pull(name),
            .after = symbol
          ) %>%
          cbind(filter(fundflow, symbol == !!symbol) %>% select(!symbol))
        return(out)
      },
      data_name, fundflow
    )
  ) %>%
    rowwise %>%
    mutate(r = close / open - 1, .after = volume) %>%
    mutate(score = in1 * 0.4 + in3 * 0.3 + in5 * 0.2 + in10 * 0.1) %>%
    as.data.frame()

  dir.create("tmp/")

  ranking <- filter(
    latest,
    r <= 0.05 &
      (
        (xa >= xa_h & xa1 < xa_h & xad > 0) |
          (xb >= xb_h & xb1 < xb_h & xbd > 0)
      ) &
      sgd <= 0
  ) %>%
    arrange(desc(score)) %>%
    na.omit() %>%
    select(date, symbol, name, r, xa, xb, in1, in3, in5, in10, score) %>%
    mutate(across(where(is.numeric), round, 3)) %>%
    mutate(across(where(is.numeric), format, nsmall = 2)) %>%
    as.data.frame()
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
  out <- list(data_list = data_list, latest = latest)
  if (!file.exists(portfolio_path)) return(out)

  portfolio <- read.csv(
    portfolio_path,
    colClasses = c(date = "Date", symbol = "character")
  )
  if (nrow(portfolio) == 0) return(out)

  portfolio_eval <- lapply(
    split(portfolio, portfolio$symbol),
    function(df, data_name, data_list = get("data_list", envir = .GlobalEnv)) {
      symbol <- df$symbol
      data <- data_list[[symbol]]
      if (!any(data$date == df$date)) stop(
        glue("Purchase date of {symbol} not found")
      )

      r <- ROR(df$cost, last(data$close))
      out <- list(
        date = df$date,
        symbol = symbol,
        name = filter(data_name, symbol == !!symbol) %>% pull(name),
        cost = df$cost,
        r = r,
        action = ifelse(
          r >= r_max | r <= r_min | last(data$date) - df$date >= t_max,
          "SELL",
          "HOLD"
        )
      )
      return(out)
    },
    data_name
  ) %>%
    rbindlist() %>%
    arrange(desc(action), desc(r)) %>%
    mutate(date = as.character(as_date(date))) %>%
    mutate(across(where(is.numeric), round, 3)) %>%
    mutate(across(where(is.numeric), format, nsmall = 2)) %>%
    `rownames<-`(seq_len(nrow(.))) %>%
    as.data.frame()

  v <- which(portfolio_eval$action == "SELL")
  if (length(v) != 0) {
    portfolio_eval <- rbind(
      portfolio_eval[v, ],
      setNames(
        data.frame(t(replicate(ncol(portfolio_eval), "")), row.names = ""),
        names(portfolio_eval)
      ),
      portfolio_eval[-v, ]
    )
  }
  print(portfolio_eval)

  return(out)
}