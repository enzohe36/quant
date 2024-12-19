backtest <- function(
  t_adx, t_cci, t_xad, t_xbd, t_sgd, xa_h, xb_h, t_max, r_max, r_min,
  data_list = get("data_list", envir = .GlobalEnv)
) {
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  trade_list <- foreach(
    data = data_list,
    .combine = "c",
    .export = c("normalize", "tnormalize", "ADX", "ROR", "get_predictor"),
    .packages = c("TTR", "signal", "tidyverse")
  ) %dopar% {
    rm("i", "j", "r", "symbol", "trade_list")

    symbol <- data[1, 2]

    data <- get_predictor(data, t_adx, t_cci, t_xad, t_xbd, t_sgd) %>%
      na.omit() %>%
      data.matrix()
    if (nrow(data) == 0) return(NULL)

    trade_list <- list()
    for (
      i in which(
        (
          data[, "xa"] >= xa_h &
            data[, "xa1"] < xa_h &
            data[, "xad"] > 0 &
            data[, "sgd"] <= 0
        ) | (
          data[, "xb"] >= xb_h &
            data[, "xb1"] < xb_h &
            data[, "xbd"] > 0 &
            data[, "sgd"] <= 0
        )
      )
    ) {
      r <- NaN
      for (j in seq_len(nrow(data))[-c(seq_len(i))]) {
        if (ROR(data[i, "close"], data[j, "high"]) >= r_max) {
          r <- r_max
          break
        }
        if (ROR(data[i, "close"], data[j, "low"]) <= r_min) {
          ifelse(
            ROR(data[i, "close"], data[j, "open"]) <= r_min,
            r <- ROR(data[i, "close"], data[j, "open"]),
            r <- r_min
          )
          break
        }
        if (data[j, "date"] - data[i, "date"] >= t_max) {
          r <- ROR(data[i, "close"], data[j, "close"])
          break
        }
      }
      trade_list[[i]] <- list(
        buy = data[i, "date"],
        sell = ifelse(is.na(r), NA, data[j, "date"]),
        symbol = symbol,
        r = r
      )
    }
    ifelse(length(trade_list) == 0, return(NULL), return(trade_list))
  }
  unregister_dopar

  trade <- rbindlist(trade_list) %>%
    filter(r > -1 | is.na(r)) %>%
    mutate(buy = as_date(buy), sell = as_date(sell)) %>%
    as.data.frame()
  return(trade)
}
