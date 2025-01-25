rm(list = ls())

gc()

library(doFuture)
library(foreach)
library(combinat)
library(glue)
library(data.table)
library(TTR)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"

index_comp_path <- paste0(data_dir, "index_comp.csv")
stats_path <- paste0(data_dir, "stats.pdf")
data_list_path <- paste0(data_dir, "data_list.rds")

index_comp <- read_csv(index_comp_path)

# Set return period
hold_period <- 10

# Generate features from downloaded data
data_list <- foreach(
  symbol = index_comp$symbol,
  .combine = "append"
) %dofuture% {
  rm(list = c("data", "data_path", "lst"))

  data_path <- paste0(data_dir, symbol, ".csv")
  if (!file.exists(data_path)) return(NULL)

  # Skip stocks with ≤ 540 days of history
  data <- read_csv(data_path, show_col_types = FALSE)
  if (nrow(data) <= 480) return(NULL)

  index_comp_i <- filter(index_comp, symbol == !!symbol)

  data <- data %>%
    # Add basic stock info
    mutate(
      index = !!index_comp_i$index,
      symbol = !!symbol,
      name = !!index_comp_i$name,
      .before = date
    ) %>%
    mutate(
      r = get_roc(lead(open), lead(close, !!hold_period)),
      .after = date
    ) %>%
    # Generate potentially useful features (to be optimized)
    mutate(
      mktcost_ror = get_roc(mktcost, close),
      adx = get_adx(cbind(high, low, close))[, "adx"],
      adx_diff = get_adx(cbind(high, low, close))[, "diff"],
      cci = CCI(cbind(high, low, close)),
      rsi = RSI(close),
      stoch = stoch(cbind(high, low, close))[, "fastK"],
      boll = BBands(cbind(high, low, close))[, "pctB"],
      amp = (runMax(high, !!hold_period) - runMin(low, !!hold_period)) /
        lag(close, !!hold_period),
      val_main = val_xl + val_l,
      turnover = vol * close / mktcap_float,
      turnover_main = val_main / mktcap_float,
      close_tnorm = tnormalize(close, 240)
    ) %>%
    add_mom(c("adx", "adx_diff", "cci", "rsi", "stoch", "boll"), c(5, 10)) %>%
    add_roc(c("close", "vol"), c(5, 10, 20, 60, 120, 240)) %>%
    add_roc(c("close"), c(hold_period)) %>%
    add_roc(c("val_main"), c(5, 10, 20)) %>%
    add_mom(c("close_tnorm", "turnover"), c(5, 10, 20, 60, 120, 240)) %>%
    add_mom(c("turnover_main"), c(5, 10, 20)) %>%
    add_pctma(c("close", "vol"), c(5, 10, 20, 60, 120, 240)) %>%
    add_pctma(c("val_main"), c(5, 10, 20)) %>%
    add_ma(c("close_tnorm", "turnover"), c(5, 10, 20, 60, 120, 240)) %>%
    add_ma(c("turnover_main"), c(5, 10, 20)) %>%
    add_sd(
      c(paste0("close_roc", hold_period), "vol", "turnover"),
      c(20, 60, 120, 240)
    ) %>%
    # Delete irrelevant features
    select(
      -c(high, low, vol, matches("^val($|_[a-z]+$)"), mktcost)
    ) %>%
    # Keep only last 60 trading days' data
    filter(date >= today() %m-% months(3))

  lst <- list()
  lst[[symbol]] <- data
  return(lst)
}

pdf(stats_path)

# Calculate mean & sd of market return
data_comb <- rbindlist(data_list) %>% na.omit()
h <- hist(data_comb$r, breaks = 1000, plot = FALSE)
nfit <- fit_normal(h$mids, h$density)
coeff <- coef(summary(nfit))[, "Estimate"]
thr <- coeff["m"] + coeff["s"] * c(0.8391, 0.2513, -0.2513, -0.8391)
col <- c("#FF0000", "#D45555", "#A9A9A9", "#55D455", "#00FF00")
col_h <- ifelse(h$mids >= thr[1], col[1], NA) %>%
  coalesce(ifelse(h$mids >= thr[2] & h$mids < thr[1], col[2], NA)) %>%
  coalesce(ifelse(h$mids >= thr[3] & h$mids < thr[2], col[3], NA)) %>%
  coalesce(ifelse(h$mids >= thr[4] & h$mids < thr[3], col[4], NA)) %>%
  coalesce(ifelse(h$mids < thr[4], col[5], NA))
plot(
  h,
  freq = FALSE,
  col = col_h,
  border = col_h,
  main = "",
  xlab = glue("{hold_period}-day return")
)
lines(h$mids, predict(nfit, h$mids), type = "l")

# Divide stocks into roughly equal-sized classes
data_list <- foreach(
  data = data_list,
  .combine = "append"
) %dofuture% {
  data <- mutate(
    data,
    target = ifelse(r >= !!thr[1], "a", NA) %>%
      coalesce(ifelse(r >= !!thr[2] & r < !!thr[1], "b", NA)) %>%
      coalesce(ifelse(r >= !!thr[3] & r < !!thr[2], "c", NA)) %>%
      coalesce(ifelse(r >= !!thr[4] & r < !!thr[3], "d", NA)) %>%
      coalesce(ifelse(r < !!thr[4], "e", NA)) %>%
      as.factor(),
    .after = r
  ) %>%
    relocate(open, close, .before = target)
  lst <- list()
  lst[[data$symbol[1]]] <- data
  return(lst)
}

saveRDS(data_list, data_list_path)
tsprint(glue("Generated features for {length(data_list)} stocks."))

data_comb <- rbindlist(data_list) %>% na.omit()
barplot(
  table(data_comb$target) %>% .[order(names(.))],
  xlab = "Class",
  ylab = "Frequency",
  col = col
)

dev.off()

plan(sequential)
