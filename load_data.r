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

indcomp_path <- paste0(data_dir, "indcomp.csv")
stats_path <- paste0(data_dir, "stats.pdf")
data_list_path <- paste0(data_dir, "data_list.rds")

indcomp <- read_csv(indcomp_path)

# Set return period
hold_period <- 10

# Generate features from downloaded data
data_list <- foreach(
  symbol = indcomp$symbol,
  .combine = "append"
) %dofuture% {
  rm(list = c("data_path", "data", "indcomp", "lst"))

  data_path <- paste0(data_dir, symbol, ".csv")
  if (!file.exists(data_path)) return(NULL)

  # Skip stocks with ≤ 540 days of history
  data <- read_csv(data_path)
  if (nrow(data) <= 480) return(NULL)

  indcomp_i <- filter(indcomp, symbol == !!symbol)

  data <- data %>%
    # Add basic stock info
    mutate(
      index = !!indcomp_i$index,
      symbol = !!symbol,
      name = !!indcomp_i$name,
      .before = date
    ) %>%
    mutate(
      r = get_roc(lead(open), lead(close, !!hold_period)),
      .after = date
    ) %>%
    # Generate potentially useful features (to be optimized)
    mutate(
      val_main = val_xl + val_l,
      .after = val_s
    ) %>%
    mutate(
      mktcost_ror = get_roc(mktcost, close),
      .after = mktcost
    ) %>%
    mutate(
      close_tnorm = tnormalize(close, 240),
      turnover = vol * close / mktcap_float,
      turnover_main = val_main / mktcap_float
    ) %>%
    add_roc(c("close"), c(5, 10, 20, 60, 120, 240)) %>%
    add_roc(c("close"), hold_period) %>%
    add_tnroc(c("close", "vol"), c(5, 10, 20, 60, 120, 240), 240) %>%
    add_tnroc(c("close"), hold_period, 240) %>%
    add_pctsma(c("close", "vol"), c(5, 10, 20, 60, 120, 240)) %>%
    add_pctsma(c("val_main"), c(5, 10, 20)) %>%
    add_sma(c("close_tnorm", "turnover"), c(1, 5, 10, 20, 60, 120, 240)) %>%
    add_sma(c("turnover_main"), c(1, 5, 10, 20)) %>%
    add_sd(
      c(paste0("close_roc", hold_period), "vol", "turnover"),
      c(20, 60, 120, 240)
    )

  lst <- list()
  lst[[symbol]] <- data
  return(lst)
}

data_comb <- rbindlist(data_list) %>% na.omit()
end_date <- max(data_comb$date)

pdf(stats_path)

# Calculate mean & sd of market return
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
  rm(list = c("lst"))

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
    filter(date >= !!end_date %m-% months(3))

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
