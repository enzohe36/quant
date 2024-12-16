backtest <- function(
  .data_list = data_list,
  .t_adx = t_adx,
  .t_cci = t_cci,
  .t_xad = t_xad,
  .t_xbd = t_xbd,
  .t_sgd = t_sgd,
  .xa_thr = xa_thr,
  .xb_thr = xb_thr,
  .t_max = t_max,
  .r_max = r_max,
  .r_min = r_min
) {
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  trade_list <- foreach(
    data = .data_list,
    .combine = append,
    .export = c(
      "get_predictor", "normalize", "tnormalize", "ADX", "ROR"
    ),
    .packages = c("TTR", "signal", "tidyverse")
  ) %dopar% {
    rm("i", "j", "r", "symbol", "trade_list")

    symbol <- data[1, 2]

    data <- get_predictor(data, .t_adx, .t_cci, .t_xad, .t_xbd, .t_sgd) %>%
      na.omit() %>%
      data.matrix()
    if (nrow(data) == 0) return(NULL)

    trade_list <- list()
    for (
      i in which(
        (
          data[, "xa"] >= .xa_thr &
            data[, "xa1"] < .xa_thr &
            data[, "xad"] > 0 &
            data[, "sgd"] <= 0
        ) | (
          data[, "xb"] >= .xb_thr &
            data[, "xb1"] < .xb_thr &
            data[, "xbd"] > 0 &
            data[, "sgd"] <= 0
        )
      )
    ) {
      for (j in i:nrow(data)) {
        r <- NaN
        if (ROR(data[i, "close"], data[j, "high"]) >= .r_max) {
          r <- .r_max
          break
        }
        if (ROR(data[i, "close"], data[j, "low"]) <= .r_min) {
          ifelse(
            ROR(data[i, "close"], data[j, "open"]) <= .r_min,
            r <- ROR(data[i, "close"], data[j, "open"]),
            r <- .r_min
          )
          break
        }
        if (data[j, "date"] - data[i, "date"] >= .t_max) {
          r <- ROR(data[i, "close"], data[j, "close"])
          break
        }
      }
      trade_list[[i]] <- list(
        data[i, "date"], data[j, "date"], symbol, r
      )
    }
    ifelse(length(trade_list) == 0, return(NULL), return(trade_list))
  }
  unregister_dopar

  trade <- rbindlist(trade_list) %>%
    as.data.frame() %>%
    `colnames<-`(c("buy", "sell", "symbol", "r")) %>%
    filter(r > -1 & buy < sell) %>%
    mutate(buy = as_date(buy), sell = as_date(sell))
  return(trade)
}
