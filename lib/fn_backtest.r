backtest <- function(
  .data_list = data_list,
  .t_adx = t_adx,
  .t_cci = t_cci,
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
    .export = c("normalize", "tnormalize", "ADX", "ROR"),
    .packages = c("TTR", "signal", "tidyverse")
  ) %dopar% {
    symbol <- data[1, "symbol"]

    data_name <- c("date", "open", "high", "low", "close", "volume")
    data <- data.matrix(data)[, data_name]

    error <- try(
      {
        adx <- ADX(data[, c("high", "low", "close")])
        cci_n <- (
          1 - 2 * tnormalize(CCI(data[, c("high", "low", "close")]), .t_cci)
        )

        predictor_list <- list() %>%
          c(xa = list((1 - normalize(abs(adx$adx - adx$adxr))) * cci_n)) %>%
          c(xa1 = list(lag(.$xa, 1))) %>%
          c(xad = list(momentum(.$xa, 5))) %>%
          c(xb = list(tnormalize(adx$adx, .t_adx) * cci_n)) %>%
          c(xb1 = list(lag(.$xb, 1))) %>%
          c(xbd = list(abs(momentum(.$xb, 5)) - abs(momentum(.$xb, 1)))) %>%
          c(sg = list(sgolayfilt(data[, "close"], n = 7))) %>%
          c(sgd = list(ROR(lag(.$sg, 10), .$sg)))
      },
      silent = TRUE
    )
    if (class(error) == "try-error") {
      predictor_name <- c("xa", "xa1", "xad", "xb", "xb1", "xbd", "sg", "sgd")
      predictor_list <- lapply(
        predictor_name, function(str) rep(NaN, length(date))
      ) %>%
        `names<-`(predictor_name)
    }

    if (any(sapply(predictor_list, function(v) all(is.na(v))))) return(NULL)

    trade_list <- list()
    for (
      i in which(
        (
          predictor_list$xa >= .xa_thr &
            predictor_list$xa1 < .xa_thr &
            predictor_list$xad > 0 &
            predictor_list$sgd <= 0
        ) | (
          predictor_list$xb >= .xb_thr &
            predictor_list$xb1 < .xb_thr &
            predictor_list$xbd > 0 &
            predictor_list$sgd <= 0
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
        if (j - i >= .t_max) {
          r <- ROR(data[i, "close"], data[j, "close"])
          break
        }
      }
      trade_list[[i]] <- list(
        symbol, data[i, "date"], data[j, "date"], r, j - i
      )
    }
    ifelse(length(trade_list) == 0, return(NULL), return(trade_list))
  }
  unregister_dopar

  trade <- rbindlist(trade_list) %>%
    as.data.frame() %>%
    `colnames<-`(c("symbol", "buy", "sell", "r", "t")) %>%
    filter(r > -1 & buy < sell) %>%
    mutate(buy = as_date(buy), sell = as_date(sell))
  return(trade)
}
