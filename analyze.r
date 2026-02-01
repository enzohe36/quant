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
watchlist_path <- paste0(resources_dir, "watchlist.txt")

analysis_dir <- "analysis/"

logs_dir <- paste0(analysis_dir, "logs/")
log_path <- paste0(logs_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

last_td <- eval(last_td_expr)

# MAIN SCRIPT ==================================================================

dir.create(analysis_dir)
dir.create(logs_dir)

data_combined <- readRDS(data_combined_path)
spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)

symbols <- c(
  pull(read_csv(index_comp_path, show_col_types = FALSE), symbol),
  readLines(watchlist_path)
) %>%
  unique() %>%
  sort()

plan(multisession, workers = availableCores() - 1)

symbols_filtered <- foreach(
  data = data_combined %>% .[names(.) %in% symbols],
  .combine = "c"
) %dofuture% {
  data <- data %>%
    mutate(
      mc_sma60 = run_mean(mc, 60),
      to_sma60 = run_mean(to, 60)
    ) %>%
    filter(
      data,
      date == !!last_td,
      mc_sma60 >= 5 * 10^9,
      mc_sma60 / np_deduct > 0 & mc_sma60 / np_deduct <= 300,
      to_sma60 >= 0.01,
      close >= 20,
      close / run_min(close, 20) <= 1.3
    )
  if (nrow(data) == 0) return(NULL)
  return(data$symbol)
}

count <- foreach(
  data = data_combined %>% .[names(.) %in% symbols_filtered],
  .combine = "c"
) %dofuture% {
  vars <- c("image_path", "name", "plot", "symbol")
  rm(list = vars)

  data <- filter(data, date >= ymd(20250101))
  symbol <- first(data$symbol)
  image_path <- paste0(analysis_dir, symbol, ".png")
  name <- filter(spot_combined, symbol == !!symbol)$name
  plot <- plot_indicators(data, plot_title = paste0(symbol, " - ", name))
  # print(plot)
  suppressMessages(ggsave(image_path, plot))
  return(1)
} %>%
  sum()

plan(sequential)

tsprint(str_glue("Generated {count} stock analysis plots."))
