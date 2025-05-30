rm(list = ls())

gc()

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"
index_comp_path <- paste0(data_dir, "index_comp.csv")
data_comb_path <- paste0(data_dir, "data_comb.rds")

index_comp <- read_csv(index_comp_path)

data_list <- foreach(
  symbol = index_comp$symbol,
  .combine = "append"
) %dofuture% {
  rm(list = c("data_path", "data", "index_comp_i", "lst"))

  data_path <- paste0(data_dir, symbol, ".csv")
  if (!file.exists(data_path)) return(NULL)

  index_comp_i <- filter(index_comp, symbol == !!symbol)

  data <- read_csv(data_path, show_col_types = FALSE) %>%
    mutate(
      symbol = !!index_comp_i$symbol,
      name = !!index_comp_i$name,
      index = !!index_comp_i$index,
      industry = !!index_comp_i$industry,
      .before = date
    )

  lst <- list()
  lst[[symbol]] <- data
  return(lst)
}

date_all <- tibble(
  date = lapply(data_list, function(df) df$date) %>%
    do.call(what = c) %>%
    unique() %>%
    sort()
)

data_comb <- data_list %>%
  lapply(
    function(df) {
      list(date_all, df) %>%
        reduce(left_join, by = "date") %>%
        fill(!c(open, high, low), .direction = "down") %>%
        mutate(across(c(open, high, low), ~ coalesce(., close)))
    }
  ) %>%
  rbindlist() %>%
  na.omit()

start_date <- date_all %>%
  filter(
    date <= today() %m-% years(5),
    lead(date, 1) > today() %m-% years(5)
  ) %>%
  pull(date)

plot_hist <- function(symbol_list, name) {
  data <- data_comb %>%
    filter(date > !!start_date %m-% years(1), symbol %in% !!symbol_list) %>%
    group_by(symbol) %>%
    mutate(
      bias_sma100 = get_roc(SMA(close, 100), close),
      turnover = SMA(amt / mktcap, 20)
    ) %>%
    group_by(date) %>%
    summarise(across(is.numeric, median, na.rm = TRUE)) %>%
    filter(date > !!start_date)
  param_list <- c(
    "close", "bias_sma100", "turnover", "pe", "pocf"
  )
  for (param in param_list) {
    plot(
      pull(data, date), pull(data, param),
      type = "l",
      main = name,
      xlab = "date", ylab = param,
      col = which(param_list == param)
    )
    for (i in quantile(pull(data, param), na.rm = TRUE)[2:4]) {
      abline(h = i, lty = 2, col = which(param_list == param))
    }
    abline(
      h = last(pull(data, param)), col = which(param_list == param)
    )
  }
}

query_hist <- function(symbol) {
  par(mfrow = c(3, 5))

  index <- index_comp %>%
    filter(symbol == !!symbol) %>%
    pull(index)
  symbol_list <- data_comb %>%
    filter(date == !!start_date, index == !!index) %>%
    pull(symbol)
  plot_hist(symbol_list, index)

  industry <- index_comp %>%
    filter(symbol == !!symbol) %>%
    pull(industry)
  symbol_list <- data_comb %>%
    filter(date == !!start_date, industry == !!industry) %>%
    pull(symbol)
  plot_hist(symbol_list, industry)

  name <- index_comp %>%
    filter(symbol == !!symbol) %>%
    pull(name)
  plot_hist(symbol, name)
}

# "002384", "603259", "300308", "603893", "002050", "601689", "600206", "300450"

for (
  i in c(
    "002384", "603259", "300308", "603893", "002050", "601689", "600206",
    "300450"
  )
) {
  query_hist(i)
}

plan(sequential)
