rm(list = ls())
gc()

library(doFuture)
library(foreach)
library(TTR)
library(data.table)
library(glue)
library(tidyverse)

# Load custom settings & helper functions
source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"

index_comp_path <- paste0(data_dir, "index_comp.csv")

model_dir <- "models/"

dir.create(model_dir)

data_comb_path <- paste0(model_dir, "data_comb.rds")
data_comb_trim_path <- paste0(model_dir, "data_comb_trim.rds")

index_comp <- read_csv(index_comp_path)
t_obs <- 10

# Load historical stock data & generate features
data_comb <- foreach(
  symbol = index_comp$symbol,
  .combine = "append"
) %dofuture% {
  var <- c("var", "data_path", "index_comp_i", "data", "try_error", "lst")
  rm(list = var)

  data_path <- paste0(data_dir, symbol, ".csv")
  if (!file.exists(data_path)) return(NULL)

  index_comp_i <- filter(index_comp, symbol == !!symbol)

  # Skip stocks with insufficient history
  data <- read_csv(data_path, show_col_types = FALSE) %>%
    mutate(
      symbol = !!index_comp_i$symbol,
      name = !!index_comp_i$name,
      index = !!index_comp_i$index,
      industry = !!index_comp_i$industry,
      .before = date
    )

  try_error <- try(
    data <- data %>%
      mutate(
        target = get_roc(close, lead(runMax(close, t_obs), t_obs)),
        obv = OBV(close, vol)
      ) %>%
      add_smaroc(c("close", "obv"), c(5, 20, 60, 120, 240)),
    silent = TRUE
  )

  lst <- list()
  lst[[symbol]] <- data
  return(lst)
} %>%
  rbindlist(fill = TRUE)
saveRDS(data_comb, data_comb_path)
tsprint(glue("Loaded {length(index_comp$symbol)} stocks."))

data_comb_trim <- data_comb %>%
  na.omit() %>%
  select(symbol:date, target, contains("_smaroc")) %>%
  filter(index %in% c("000300", "000905")) %>%
  group_by(date) %>%
  mutate(
    target = findInterval(
      target, c(-Inf, quantile(target, na.rm = TRUE)[2:4], Inf)
    ) %>%
      as.factor()
  ) %>%
  ungroup()
saveRDS(data_comb_trim, data_comb_trim_path)