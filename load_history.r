load_history <- function(pattern, adjust, start_date, end_date) {
  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Started load_history()."
    ),
    quote = FALSE
  )

  # Define parameters
  start_date <- as.Date(start_date)
  end_date <- as.Date(end_date)

  symbol_list <- readLines("symbol_list.txt")
  symbol_list <- symbol_list[grepl(pattern, symbol_list)]

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    symbol = symbol_list,
    .combine = multiout,
    .multicombine = TRUE,
    .init = list(list(), list()),
    .errorhandling = "remove",
    .packages = "tidyverse"
  ) %dopar% {
    data <- read.csv(
      paste0("data_", adjust, "/", symbol, ".csv"),
      colClasses = c(date = "Date", symbol = "character")
    )
    data <- data[data$date >= start_date & data$date <= end_date, ]
    if (data[1, "date"] > start_date + days(2)) next

    return(list(symbol, data))
  }
  unregister_dopar

  symbol_list <- do.call(rbind, out[[1]])[, 1]

  data_list <- out[[2]]
  names(data_list) <- do.call(
    c, lapply(data_list, function(df) df[1, "symbol"])
  )

  print(
    paste0(
      format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
      " Loaded ", length(data_list), " stocks from data_", adjust, "/."
    ),
    quote = FALSE
  )

  return(list(symbol_list, data_list))
}