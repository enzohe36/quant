# PRESET =======================================================================

library(foreach)
library(doFuture)
library(xts)
library(DSTrading)
library(patchwork)
library(data.table)
library(tidyverse)

source("scripts/misc.r")
source("scripts/features.r")
source("scripts/ehlers.r")

data_dir <- "data/"
data_combined_path <- paste0(data_dir, "data_combined.rds")
spot_combined_path <- paste0(data_dir, "spot_combined.csv")

backtest_dir <- "backtest/"

logs_dir <- "logs/"
log_path <- paste0(logs_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

end_date <- eval(last_td_expr)
start_date <- end_date %m-% years(20)

zero_threshold <- 0.05
price_lookback <- 10
min_price_diff <- 0.5
price_lookforward <- 5
min_signal <- 0.35
min_osc_d1 <- -0.01
osc_lookback <- 40
max_osc_diff <- 1.9
price_rms_high <- 1.5
price_rms_low <- -1

# STOCK ANALYSIS ===============================================================

dir.create(backtest_dir)
dir.create(logs_dir)

data_combined <- readRDS(data_combined_path)
spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)

plan(multisession, workers = availableCores() - 1)

data_combined_gf <- foreach(
  data = data_combined,
  .combine = "c"
) %dofuture% {
  vars <- c("my_list", "symbol")
  rm(list = vars)

  symbol <- first(data$symbol)
  data <- generate_features(
    data = data,
    start_date = start_date,
    end_date = end_date,
    zero_threshold = zero_threshold,
    price_lookback = price_lookback,
    min_price_diff = min_price_diff,
    price_lookforward = price_lookforward,
    min_signal = min_signal,
    min_osc_d1 = min_osc_d1,
    osc_lookback = osc_lookback,
    max_osc_diff = max_osc_diff,
    price_rms_high = price_rms_high,
    price_rms_low = price_rms_low
  )
  my_list <- list()
  my_list[[symbol]] <- data
  return(my_list)
}

plan(sequential)

# symbols <- lapply(
#   data_combined_gf,
#   function(df) {
#     if (
#       filter(
#         df,
#         date == !!end_date &
#           mc >= 10^10 &
#           pe > 0 &
#           runMax(price_rms, 20) >= 2
#       ) %>%
#         pull(buy) %>%
#         isTRUE()
#     ) {
#       unique(df$symbol)
#     }
#   }
# ) %>%
#   unlist() %>%
#   unname()

# symbols <- c(
#   "300720", "001301", "688766", "688656", "688122", "301018", "300450", "600703", "301069", "300946", "600562", "300455", "300857", "688099", "603893", "300885", "688608", "301291", "688027", "300757", "301308", "002384", "600114", "688120", "688709", "300827", "688472", "688249", "300660", "688012", "002850", "002518", "601231", "688525", "300408", "002558", "688256", "688002", "300655", "688210", "688670", "688559", "688170", "688234", "300316", "301488", "300019", "688160", "300652", "300037", "002008", "300378", "301207", "600711", "600863", "300516", "688200", "301606", "688196", "603530", "688128", "605008", "688172", "600483", "601869", "688409", "688599", "300446", "301010", "301117", "688333", "688256", "688041", "688709", "603893", "688018", "688099", "688608", "688591", "301308", "603019", "300857", "688072", "688012", "002371", "688234", "300316", "300475", "600703", "600330", "300655", "603650", "300398", "002384", "300308", "300502", "000988", "002281", "300620", "688027", "300450", "688499", "688155", "688411", "300274", "300827", "605117", "688472", "688676", "300037", "002407", "301358", "300073", "300080", "002050", "601689", "300660", "603009", "300100", "300580", "300652", "002896", "002472", "603728", "688160", "002008", "300946", "002850", "300953", "603662", "688322", "688400", "301076", "688716", "600114", "688210", "300885", "600392", "600111", "000831", "300748", "600366", "000970", "300127", "600206", "300618", "603799", "000603", "000737", "300199", "688117", "688235", "688131", "688617", "688236", "301091", "688629", "688002", "300768", "688631", "301236", "300339", "300378"
# )

symbols <- "300720"

for (symbol in symbols) {
  image_path <- paste0(backtest_dir, symbol, ".png")
  data <- data_combined_gf[[symbol]] %>%
    filter(date >= end_date %m-% years(1) & date <= end_date)
  name <- pull(filter(spot_combined, symbol == !!symbol), name)
  plot <- plot_indicators(data, plot_title = paste0(symbol, " - ", name))
  print(plot)
  ggsave(image_path, plot)
}

# MARKET ANALYSIS ==============================================================

plan(multisession, workers = availableCores() - 1)

data_market <- foreach(
  data = rbindlist(data_combined_gf) %>% split(.$date),
  .combine = "c"
) %dofuture% {
  ind <- data$mc_float %>%
    log() %>%
    normalize(silent = TRUE)
  ind <- which(!is.na(ind) & ind >= 2)
  data <- data[ind, ]$mc / 10^8
  list(
    summarize(
      data,
      r = weighted.mean(r, mc_float, na.rm = TRUE),
      dev = weighted.mean(dev, mc_float, na.rm = TRUE),
      count = n(),
      .by = date
    )
  )
} %>%
  rbindlist() %>%
  arrange(date) %>%
  mutate(
    close = cumprod(r),
    avg_cost = close / dev
  )

plan(sequential)

plot <- ggplot(data_market, aes(x = date)) +
  geom_line(aes(y = close), color = "black", linewidth = 0.5) +
  geom_line(aes(y = avg_cost), color = "blue", linewidth = 0.5) +
  theme_minimal()
print(plot)

plot <- ggplot(data_market, aes(x = date)) +
  geom_line(aes(y = mc), color = "black", linewidth = 0.5) +
  theme_minimal()
print(plot)

plot <- ggplot(data_market, aes(x = date)) +
  geom_line(aes(y = count), color = "black", linewidth = 0.5) +
  theme_minimal()
print(plot)

plot <- ggplot(data_market, aes(x = date)) +
  geom_line(aes(y = close / avg_cost - 1), color = "black", linewidth = 0.5) +
  theme_minimal()
print(plot)

plot <- ggplot(data_market, aes(x = date)) +
  geom_line(aes(y = pe), color = "black", linewidth = 0.5) +
  theme_minimal()
print(plot)
