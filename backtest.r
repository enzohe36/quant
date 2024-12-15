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

t <- as.numeric(trade$sell - trade$buy)
stats <- rbind(
  data.frame(r = quantile(trade$r), t = quantile(t)),
  data.frame(
    r = c(mean(trade$r), sd(trade$r)),
    t = c(mean(t), sd(t)),
    row.names = c("mean", "sd")
  )
) %>%
  mutate(
    r = format(round(r, 3), nsmall = 3),
    t = round(t)
  )
stats <- rbind(
  stats[1:5, ],
  setNames(
    data.frame(t(replicate(ncol(stats), "")), row.names = ""), names(stats)
  ),
  stats[-c(1:5), ]
)
print(stats)

hist <- hist(trade$r, breaks = 100)
