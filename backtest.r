# python -m aktools

rm(list = ls())

source("lib/preset.r", encoding = "UTF-8")
source("lib/misc.r", encoding = "UTF-8")
source("lib/fn_load_data.r", encoding = "UTF-8")
source("lib/fn_backtest.r", encoding = "UTF-8")
source("lib/fn_sample_apy.r", encoding = "UTF-8")

trade_path <- "tmp/trade.csv"
apy_path <- "tmp/apy.csv"
index_path <- "tmp/index.csv"

# ------------------------------------------------------------------------------

# Define parameters
t_adx <- 143
t_cci <- 156
t_xad <- 5
t_xbd <- 2
t_sgd <- 16
xa_thr <- 0.4
xb_thr <- 0.27
t_max <- 52
r_max <- 0.06
r_min <- -0.54

start_date <- ymd(20210628)
end_date <- ymd(20240628)
n_portfolio <- 30
t <- 1
n_sample <- 1000

dir.create("tmp")

data_list <- load_data("^(00|60)", "hfq", start_date, end_date)

runtime <- system.time(trade <- backtest())
tsprint(glue("Total time: {runtime[3]} s."))
write.csv(trade, trade_path, quote = FALSE, row.names = FALSE)

# trade <- read.csv(trade_path, colClasses = c(buy = "Date", sell = "Date", symbol = "character"))

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

runtime <- system.time(apy <- sample_apy(trade, n_portfolio, t, n_sample))
tsprint(glue("Total time: {runtime[3]} s."))
write.csv(apy, apy_path, quote = FALSE, row.names = FALSE)

hist(apy$date, breaks = 100)

# index <- em_index("000906") %>% select(date, close)
# write.csv(index, index_path, quote = FALSE, row.names = FALSE)

index <- read.csv(index_path, colClasses = c(date = "Date")) %>%
  filter(date >= start_date & date <= end_date)

apy_mean <- split(apy, f = apy$date) %>%
  sapply(function(df) mean(df$apy))
index_n <- normalize(index$close) * (max(apy_mean) - min(apy_mean)) +
  min(apy_mean)
plot(index$date, sgolayfilt(index_n, n = 7), type = "l")
lines(unique(apy$date), sgolayfilt(apy_mean, n = 7), col = "red")
abline(h = 0, col = "darkred")

opt_lm <- function(t) {
  index$close_tn <- tnormalize(index$close, t)
  df <- reduce(list(apy, index), full_join, by = "date")
  fit <- lm(df$apy ~ df$close_tn)

  return(fit$coefficients[2])
}
opt <- optimize(opt_lm, c(1, 250), tol = 0.01)
print(opt)

index$close_tn <- tnormalize(index$close, round(opt$minimum))
df <- reduce(list(apy, index), full_join, by = "date") %>% na.omit
plot(df$close_tn, df$apy, pch = 20, cex = 0.5)

fit <- lm(apy ~ close_tn, df)
print(summary(fit))

ci_x <- data.frame(close_tn = seq(0, 1, 0.01))
ci <- as.data.frame(predict(fit, newdata = ci_x, interval = "prediction"))
lines(ci_x$close_tn, ci$fit, col="red")
lines(ci_x$close_tn, ci$lwr, col = "blue", lty = 2)
lines(ci_x$close_tn, ci$upr, col = "blue", lty = 2)
