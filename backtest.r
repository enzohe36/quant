# python -m aktools

rm(list = ls())

source("lib/preset.r", encoding = "UTF-8")
source("lib/misc.r", encoding = "UTF-8")
source("lib/fn_as_tdate.r", encoding = "UTF-8")

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
xa_h <- 0.4
xb_h <- 0.27
t_max <- 52
r_max <- 0.06
r_min <- -0.54

start_date <- 20191129
end_date <- 20241129

n_portfolio <- 30
t_apy <- 1
n_sample <- 1000

dir.create("tmp")

data_list <- load_data("^(00|60)", "hfq", start_date, end_date)

runtime <- system.time(
  trade <- backtest(
    t_adx, t_cci, t_xad, t_xbd, t_sgd, xa_h, xb_h, t_max, r_max, r_min
  )
)
tsprint(glue("Total time: {runtime[3]} s."))
write.csv(trade, trade_path, quote = FALSE, row.names = FALSE)

t_trade <- as.numeric(trade$sell - trade$buy)
stats <- rbind(
  data.frame(
    r = quantile(trade$r, na.rm = TRUE),
    t_trade = quantile(t_trade, na.rm = TRUE)
  ),
  data.frame(
    r = c(mean(trade$r, na.rm = TRUE), sd(trade$r, na.rm = TRUE)),
    t_trade = c(mean(t_trade, na.rm = TRUE), sd(t_trade, na.rm = TRUE)),
    row.names = c("mean", "sd")
  )
) %>%
  mutate(
    r = format(round(r, 3), nsmall = 3),
    t_trade = round(t_trade)
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

runtime <- system.time(
  apy <- sample_apy(trade, n_portfolio, t_apy, n_sample, end_date)
)
tsprint(glue("Total time: {runtime[3]} s."))
write.csv(apy, apy_path, quote = FALSE, row.names = FALSE)

hist(apy$date, breaks = 100)

# index <- em_index("000906")
# write.csv(index, index_path, quote = FALSE, row.names = FALSE)

index <- read.csv(index_path, colClasses = c(date = "Date")) %>%
  select(date, close) %>%
  filter(date >= ymd(start_date) & date <= ymd(end_date))

apy_mean <- split(apy, f = apy$date) %>%
  sapply(function(df) mean(df$apy))
index_n <- normalize(index$close) * (max(apy_mean) - min(apy_mean)) +
  min(apy_mean)
plot(index$date, sgolayfilt(index_n, n = 7), type = "l")
lines(unique(apy$date), sgolayfilt(apy_mean, n = 7), col = "red")
abline(h = 0, col = "darkred")

opt_lm <- function(t_index) {
  index$close_tn <- tnormalize(index$close, t_index)
  apy <- reduce(list(apy, index), full_join, by = "date")
  fit <- lm(apy$apy ~ apy$close_tn)
  return(fit$coefficients[2])
}
opt <- optimize(opt_lm, c(1, 250), tol = 0.01)
print(opt)

index$close_tn <- tnormalize(index$close, round(opt$minimum))
apy <- reduce(list(apy, index), full_join, by = "date") %>% na.omit
plot(apy$close_tn, apy$apy, pch = 20, cex = 0.5)

fit <- lm(apy ~ close_tn, apy)
print(summary(fit))

ci_x <- data.frame(close_tn = seq(0, 1, 0.01))
ci <- as.data.frame(predict(fit, newdata = ci_x, interval = "prediction"))
lines(ci_x$close_tn, ci$fit, col="red")
lines(ci_x$close_tn, ci$lwr, col = "blue", lty = 2)
lines(ci_x$close_tn, ci$upr, col = "blue", lty = 2)
