rm(list = ls())

source("lib/preset.r", encoding = "UTF-8")
source("lib/misc.r", encoding = "UTF-8")
source("lib/fn_load_data.r", encoding = "UTF-8")
source("lib/fn_backtest.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

# Define parameters
t_adx <- 15
t_cci <- 30
xa_thr <- 0.5
xb_thr <- 0.5
t_max <- 105
r_max <- 0.09
r_min <- -0.5

data_list <- load_data("^(00|60)", "hfq", 20190628, 20240628)

runtime <- system.time(trade <- backtest())
tsprint(glue("Total time: {runtime[3]} s."))

stats_mean <- data.frame(
  r = c(mean(trade$r), sd(trade$r)),
  t = c(mean(trade$t), sd(trade$t)),
  t_cal = c(
    mean(as.numeric(trade$sell - trade$buy)),
    sd(as.numeric(trade$sell - trade$buy))
  ),
  row.names = c("mean", "sd")
)
stats_q <- data.frame(
  r = quantile(trade$r),
  t = quantile(trade$t),
  t_cal = quantile(as.numeric(trade$sell - trade$buy))
)
stats <- rbind(stats_q, stats_mean)
stats$r <- format(round(stats$r, 3), nsmall = 3)
stats[, c("t", "t_cal")] <- round(stats[, c("t", "t_cal")])

v <- seq_len(nrow(stats_q))
stats <- rbind(
  stats[v, ],
  setNames(
    data.frame(t(replicate(ncol(stats), "")), row.names = ""), names(stats)
  ),
  stats[-v, ]
)
print(stats)

hist <- hist(trade$r, breaks = 100)
