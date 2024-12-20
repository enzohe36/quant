load_data <- function(
  pattern, adjust, start_date = NA, end_date = NA,
  data_dir = paste0("data_", adjust, "/"),
  symbol_list_path = paste0(data_dir, "symbol_list.csv"),
  data_path_expr = expression(paste0(data_dir, symbol, ".csv"))
) {
  tsprint("Started load_data().")

  start_date <- ymd(start_date)
  end_date <- ymd(end_date)

  symbol_list <- readLines(symbol_list_path) %>% .[grepl(pattern, .)]

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  data_list <- foreach(
    symbol = symbol_list,
    .combine = "c",
    .export = "data_dir",
    .packages = "tidyverse"
  ) %dopar% {
    rm("data", "data_path", "lst")

    data_path <- eval(data_path_expr)
    if (file.exists(data_path)) {
      data <- read.csv(
        data_path, colClasses = c(date = "Date", symbol = "character")
      )
    }
    if (is.na(start_date)) start_date <- data[1, "date"]
    if (is.na(end_date)) end_date <- data[nrow(data), "date"]
    data <- filter(data, date >= start_date & date <= end_date)

    lst <- list()
    lst[[symbol]] <- data
    return(lst)
  }
  unregister_dopar

  tsprint(glue("Loaded {length(data_list)} stocks from {data_dir}."))

  return(data_list)
}