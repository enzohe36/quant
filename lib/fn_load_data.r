load_data <- function(
  pattern, adjust, start_date = NA, end_date = NA,
  symbol_list_path = "assets/symbol_list.csv",
  data_dir = paste0("data_", adjust, "/"),
  data_path_expr = expression(paste0(data_dir, symbol, ".csv"))
) {
  tsprint("Started load_data().")

  start_date <- ymd(start_date)
  end_date <- ymd(end_date)

  symbol_list <- readLines(symbol_list_path) %>% .[grepl(pattern, .)]

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  lst <- foreach(
    symbol = symbol_list,
    .combine = multiout,
    .multicombine = TRUE,
    .init = list(list(), list()),
    .export = "data_dir",
    .packages = "tidyverse"
  ) %dopar% {
    rm("data", "data_path")

    data_path <- eval(data_path_expr)
    if(file.exists(data_path)) {
      data <- read.csv(
        data_path, colClasses = c(date = "Date", symbol = "character")
      )
    }
    if (is.na(start_date)) start_date <- data[1, "date"]
    if (is.na(end_date)) end_date <- data[nrow(data), "date"]
    data <- data[data$date >= start_date & data$date <= end_date, ]

    return(list(symbol, data))
  }
  unregister_dopar

  tb <- tibble(symbol = unlist(lst[[1]]), data = lst[[2]]) %>%
    filter(sapply(data, nrow) != 0)
  tsprint(glue("Loaded {nrow(tb)} stocks from {data_dir}."))
  return(tb)
}