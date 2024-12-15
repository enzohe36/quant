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
t_adx <- 15
t_cci <- 30
xa_thr <- 0.5
xb_thr <- 0.5
t_max <- 105
r_max <- 0.09
r_min <- -0.5

dir.create("tmp")

# data_list <- load_data("^(00|60)", "hfq", 20140628, 20240628)
# trade <- backtest()
# write.csv(trade, trade_path, quote = FALSE, row.names = FALSE)

trade <- read.csv(
  trade_path,
  colClasses = c(symbol = "character", buy = "Date", sell = "Date")
)

runtime <- system.time(apy <- sample_apy(trade, 30, 1, 1000))
tsprint(glue("Total time: {runtime[3]} s."))
write.csv(apy, apy_path, quote = FALSE, row.names = FALSE)

# apy <- read.csv(apy_path, colClasses = c(date = "Date"))

hist(apy$date, breaks = 100)

# index <- em_index("000300") %>%
#   select(date, close) %>%
#   filter(date >= min(apy$date) & date <= max(apy$date))
# write.csv(index, index_path, quote = FALSE, row.names = FALSE)

index <- read.csv(index_path, colClasses = c(date = "Date"))

apy_mean <- split(apy, f = apy$date) %>%
  sapply(function(df) mean(df$apy))
index_n <- normalize(index$close) * (max(apy_mean) - min(apy_mean)) +
  min(apy_mean)
plot(index$date, sgolayfilt(index_n, n = 7), type = "l")
lines(unique(apy$date), sgolayfilt(apy_mean, n = 7), col = "red")

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
