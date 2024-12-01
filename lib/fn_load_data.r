load_data <- function(
  pattern, adjust, start_date = NA, end_date = NA,
  symbol_list_path = "assets/symbol_list.txt",
  data_dir = paste0("data_", adjust, "/"),
  data_path_expr = expression(paste0(data_dir, symbol, ".csv"))
) {
  tsprint("Started load_data().")

  start_date <- ymd(start_date)
  end_date <- ymd(end_date)

  symbol_list <- readLines(symbol_list_path)
  symbol_list <- symbol_list[grepl(pattern, symbol_list)]

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  out <- foreach(
    symbol = symbol_list,
    .combine = multiout,
    .multicombine = TRUE,
    .init = list(list(), list()),
    .export = "data_dir",
    .packages = "tidyverse"
  ) %dopar% {
    data_path <- eval(data_path_expr)
    data <- read.csv(
      data_path,
      colClasses = c(date = "Date", symbol = "character")
    )
    if (is.na(start_date)) start_date <- data[1, "date"]
    if (is.na(end_date)) end_date <- data[nrow(data), "date"]
    data <- data[data$date >= start_date & data$date <= end_date, ]
    return(list(symbol, data))
  }
  unregister_dopar

  symbol_list <- do.call(rbind, out[[1]])[, 1]

  data_list <- out[[2]]
  names(data_list) <- do.call(
    c, lapply(data_list, function(df) df[1, "symbol"])
  )
  tsprint(glue("Loaded {length(data_list)} stocks from {data_dir}."))

  return(list(symbol_list = symbol_list, data_list = data_list))
}