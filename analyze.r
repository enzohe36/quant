# PRESET =======================================================================

library(xts)
library(DSTrading)
library(patchwork)
library(sn)
library(foreach)
library(doFuture)
library(data.table)
library(tidyverse)

source("scripts/ehlers.r")
source("scripts/features.r")
source("scripts/misc.r")

data_dir <- "data/"
data_combined_path <- paste0(data_dir, "data_combined.rds")
spot_combined_path <- paste0(data_dir, "spot_combined.csv")

resources_dir <- "resources/"
index_comp_path <- paste0(resources_dir, "index_comp.csv")

backtest_dir <- "backtest/"

logs_dir <- paste0(backtest_dir, "logs/")
log_path <- paste0(logs_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

# last_td <- eval(last_td_expr)
last_td <- as_date("2026-01-07")

# HELPER FUNCTIONS =============================================================

filter_index_comp <- function(data) {
  mc_quarter_norm <- normalize(log(data$mc_quarter), silent = TRUE)
  to_quarter_norm <- normalize(log(data$to_quarter), silent = TRUE)

  index_comp <- data %>%
    filter(
      across(everything(), ~ !is.na(.x)),
      # mc_quarter_norm <= 0,
      # to_quarter_norm >= -Inf,
      # mc_quarter / np >= 0 & mc_quarter / np <= 10
    ) %>%
    # slice_sample(n = 20) %>%
    mutate(shares = mc / close) %>%
    select(symbol, shares)

  return(index_comp)
}

# STOCK ANALYSIS ===============================================================

dir.create(backtest_dir)
dir.create(logs_dir)

data_combined <- readRDS(data_combined_path)
spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)

# index_comp <- read_csv(index_comp_path, show_col_types = FALSE)

# plan(multisession, workers = availableCores() - 1)

# symbols <- foreach(
#   data = data_combined %>% .[names(.) %in% index_comp$symbol],
#   .combine = "c"
# ) %dofuture% {
#   filter(
#     data,
#     date == !!last_td &
#       buy &
#       mc_quarter >= 10^10 &
#       mc_quarter / np_deduct >= 0 & mc_quarter / np_deduct <= 300
#   ) %>%
#     pull(symbol)
# }

# plan(sequential)

# symbols <- c(
#   "300720", "001301", "688766", "688656", "688122", "301018", "300450", "600703", "301069", "300946", "600562", "300455", "300857", "688099", "603893", "300885", "688608", "301291", "688027", "300757", "301308", "002384", "600114", "688120", "688709", "300827", "688472", "688249", "300660", "688012", "002850", "002518", "601231", "688525", "300408", "002558", "688256", "688002", "300655", "688210", "688670", "688559", "688170", "688234", "300316", "301488", "300019", "688160", "300652", "300037", "002008", "300378", "301207", "600711", "600863", "300516", "688200", "301606", "688196", "603530", "688128", "605008", "688172", "600483", "601869", "688409", "688599", "300446", "301010", "301117", "688333", "688256", "688041", "688709", "603893", "688018", "688099", "688608", "688591", "301308", "603019", "300857", "688072", "688012", "002371", "688234", "300316", "300475", "600703", "600330", "300655", "603650", "300398", "002384", "300308", "300502", "000988", "002281", "300620", "688027", "300450", "688499", "688155", "688411", "300274", "300827", "605117", "688472", "688676", "300037", "002407", "301358", "300073", "300080", "002050", "601689", "300660", "603009", "300100", "300580", "300652", "002896", "002472", "603728", "688160", "002008", "300946", "002850", "300953", "603662", "688322", "688400", "301076", "688716", "600114", "688210", "300885", "600392", "600111", "000831", "300748", "600366", "000970", "300127", "600206", "300618", "603799", "000603", "000737", "300199", "688117", "688235", "688131", "688617", "688236", "301091", "688629", "688002", "300768", "688631", "301236", "300339", "300378"
# )

# symbols <- "000988"

# for (symbol in symbols) {
#   image_path <- paste0(backtest_dir, symbol, ".png")
#   data <- data_combined[[symbol]] %>%
#     filter(date >= last_td %m-% years(2) & date <= last_td)
#   name <- pull(filter(spot_combined, symbol == !!symbol), name)
#   plot <- plot_indicators(data, plot_title = paste0(symbol, " - ", name))
#   # print(plot)
#   ggsave(image_path, plot)
# }

# MARKET ANALYSIS ==============================================================

plan(multisession, workers = availableCores() - 1)

data_split <- foreach(
  data = data_combined,
  .combine = "c"
) %dofuture% {
  vars <- c("all_td_data", "susp", "symbol")
  rm(list = vars)

  symbol <- data$symbol[1]
  susp <- pull(filter(spot_combined, symbol == !!symbol), susp)
  all_td_data <- all_td %>%
    .[. >= first(data$date) & . <= ifelse(susp, last_td, last(data$date))]

  data <- data %>%
    right_join(tibble(date = all_td_data), by = "date") %>%
    arrange(date) %>%
    fill(
      symbol, close, mc, np, np_deduct, equity, revenue, cf,
      mc_quarter, to_quarter,
      .direction = "down"
    ) %>%
    mutate(quarter = quarter(date, with_year = TRUE)) %>%
    select(
      symbol, date, close, mc, np, np_deduct, equity, revenue, cf,
      quarter, mc_quarter, to_quarter,
    ) %>%
    filter(date >= as_tradeday(as_date("2004-01-01") - 1))

  return(list(data))
} %>%
  rbindlist() %>%
  split(.$date) %>%
  .[order(names(.))]

plan(sequential)

index_comp <- filter_index_comp(data_split[[1]])
adjust <- 1

index <- foreach(
  ind = 5346,
  .combine = "c"
) %do% {
  vars <- c("data", "data_1", "data_1_comp", "data_comp")
  rm(list = vars)

  data <- data_split[[ind]]
  data_1 <- data_split[[max(ind - 1, 1)]]

  data_comp <- left_join(index_comp, data, by = "symbol")
  data_1_comp <- left_join(index_comp, data_1, by = "symbol")

  if (first(data$quarter) != first(data_1$quarter)) {
    index_comp <<- filter_index_comp(data_1)
    data_1_comp_new <- left_join(index_comp, data_1, by = "symbol")

    adjust <- sum(data_1_comp$close * data_1_comp$shares) /
      sum(data_1_comp_new$close * data_1_comp_new$shares)
    if (is.na(adjust) | adjust <= 0) adjust <- 1

    index_comp <<- mutate(index_comp, shares = shares * adjust)
    data_comp <- left_join(index_comp, data, by = "symbol")
    data_1_comp <- left_join(index_comp, data_1, by = "symbol")
  }

  symbols_reject <- left_join(data_comp, data_1_comp, by = "symbol") %>%
    filter(
      case_when(
        str_detect(symbol, "^(00|60)") ~ abs(close.x / close.y - 1) >= 0.105,
        str_detect(symbol, "^68") ~ abs(close.x / close.y - 1) >= 0.205,
        str_detect(symbol, "^30") & first(data$date) < as_date("2020-08-24") ~
          abs(close.x / close.y - 1) >= 0.105,
        TRUE ~ abs(close.x / close.y - 1) >= 0.205
      )
    ) %>%
    pull(symbol)
  if (length(symbols_reject) > 0) {
    data_comp <- filter(data_comp, !symbol %in% symbols_reject) %>%
      right_join(select(index_comp, symbol), by = "symbol")
    tsprint(str_glue("{first(data$date)}: {paste(symbols_reject)}"), log_path)
  }

  if (length(na.omit(data_comp$close)) < nrow(index_comp)) {
    data_1_comp_new <- filter(
      data_1_comp,
      symbol %in% !!data_comp$symbol[!is.na(data_comp$close)]
    )

    adjust <- sum(data_1_comp$close * data_1_comp$shares) /
      sum(data_1_comp_new$close * data_1_comp_new$shares)
    if (is.na(adjust) | adjust <= 0) adjust <- 1

    index_comp <<- index_comp %>%
      filter(symbol %in% data_1_comp_new$symbol) %>%
      mutate(shares = shares * adjust)
    data_comp <- left_join(index_comp, data, by = "symbol")
  }

  data_comp %>%
    summarize(
      count = n(),
      adjust = adjust,
      close = sum(close * shares),
      across(c(mc, np, np_deduct, equity, revenue, cf), sum),
      # pe = mc / np,
      # pe_deduct = mc / np_deduct,
      # pb = mc / equity,
      # ps = mc / revenue,
      # pcf = mc / cf,
      # roe = np / equity,
      # npm = np / revenue,
      .by = date
    ) %>%
    select(
      date, count, adjust, close, mc, np, np_deduct, equity, revenue, cf
    ) %>%
    list()
} %>%
  rbindlist() %>%
  filter(!is.na(close))

plot <- ggplot(index, aes(x = date)) +
  geom_line(aes(y = close / first(close)), color = "black", linewidth = 0.5) +
  theme_minimal()
print(plot)

plot <- ggplot(index, aes(x = date)) +
  geom_line(aes(y = count), color = "black", linewidth = 0.5) +
  theme_minimal()
print(plot)

plot <- ggplot(index, aes(x = date)) +
  geom_line(aes(y = adjust), color = "black", linewidth = 0.5) +
  theme_minimal()
print(plot)

plot <- ggplot(index, aes(x = date)) +
  geom_line(aes(y = mc), color = "black", linewidth = 0.5) +
  theme_minimal()
print(plot)

plot <- ggplot(index, aes(x = date)) +
  geom_line(aes(y = mc / np), color = "black", linewidth = 0.5) +
  theme_minimal()
print(plot)

s <- which(abs(index$close / lag(index$close) - 1) > 0.105)
d <- as.character(index$date[s])
print(which(names(data_split) %in% d))
