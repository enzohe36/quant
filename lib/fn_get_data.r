get_data <- function(
  pattern, adjust,
  data_dir = paste0("data_", adjust, "/"),
  symbol_list_path = paste0(data_dir, "symbol_list.csv"),
  data_path_expr = expression(paste0(data_dir, symbol, ".csv"))
) {
  tsprint("Started get_data().")

  # [1]   code name
  symbol_list <- fromJSON(
    getForm(
      uri = "http://127.0.0.1:8080/api/public/stock_info_a_code_name",
      .encoding = "utf-8"
    )
  )
  symbol_list <- symbol_list[, 1]
  writeLines(symbol_list, symbol_list_path)
  tsprint(
    glue("Found {length(symbol_list)} stocks; wrote to {symbol_list_path}.")
  )

  symbol_list <- symbol_list[grepl(pattern, symbol_list)]
  tsprint(glue("Matched {length(symbol_list)} stocks to \"pattern\"."))

  dir.create(data_dir)

  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  count <- foreach(
    symbol = symbol_list,
    .combine = "c",
    .export = c("data_dir", "em_data", "bizday"),
    .packages = c("jsonlite", "RCurl", "tidyverse")
  ) %dopar% {
    rm("data", "data_old", "data_path", "end_date", "i", "start_date")

    data_path <- eval(data_path_expr)
    if(file.exists(data_path)) {
      data_old <- read.csv(
        data_path, colClasses = c(date = "Date", symbol = "character")
      )
    }

    for (i in 1:2) {
      end_date <- format(bizday(), "%Y%m%d")
      if (exists("data_old")) {
        start_date <- format(data_old[nrow(data_old), "date"], "%Y%m%d")
        data <- em_data(symbol, adjust, start_date, end_date)
        if (all(data[1, ] == data_old[nrow(data_old), ])) {
          data <- data[-1, ]
          write.table(
            data,
            data_path, append = TRUE,
            quote = FALSE,
            sep = ",",
            row.names = FALSE, col.names = FALSE
          )
          break
        } else {
          rm("data_old")
        }
      } else {
        ifelse (
          adjust == "qfq",
          start_date <- format(bizday() %m-% years(1), "%Y%m%d"),
          start_date <- ""
        )
        data <- em_data(symbol, adjust, start_date, end_date)
        write.csv(data, data_path, quote = FALSE, row.names = FALSE)
        break
      }
    }

    ifelse(nrow(data) != 0, return(1), return(0))
  }
  unregister_dopar

  tsprint(glue("Updated {sum(count)} local files."))
}
