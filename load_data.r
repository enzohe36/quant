load_data <- function(pattern, adjust, start_date = NA, end_date = NA) {
  tsprint("Started load_data().")

  # Format arguments
  start_date <- ymd(start_date)
  end_date <- ymd(end_date)

  # Define input
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
    # Define input
    data <- read.csv(
      paste0("data_", adjust, "/", symbol, ".csv"),
      colClasses = c(date = "Date", symbol = "character")
    )

    if (is.na(start_date)) start_date <- data[1, "date"]
    if (is.na(end_date)) end_date <- data[nrow(data), "date"]
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
  tsprint(glue("Loaded {length(data_list)} stocks from data_{adjust}/."))

  return(list(symbol_list = symbol_list, data_list = data_list))
}