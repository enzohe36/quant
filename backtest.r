# python -m aktools

source("lib/preset.r", encoding = "UTF-8")

source("lib/fn_get_data.r", encoding = "UTF-8")
source("lib/fn_load_data.r", encoding = "UTF-8")
source("lib/fn_backtest.r", encoding = "UTF-8")
source("lib/fn_sample_apy.r", encoding = "UTF-8")

library(signal)

trade_path <- "tmp/trade.csv"
apy_path <- "tmp/apy.csv"
index_path <- "tmp/index.csv"

# ------------------------------------------------------------------------------

# Define parameters
t_adx <- 15
t_cci <- 30
x_thr <- 0.53
t_max <- 105
r_max <- 0.09
r_min <- -0.5

# get_data("^(00|60)", "hfq")

# dir.create("tmp")

# out0 <- load_data("^(00|60)", "hfq", 20141129, 20241129)
# out0[["trade"]] <- backtest(t_adx, t_cci, x_thr, t_max, r_max, r_min)
# write.csv(out0[["trade"]], trade_path, quote = FALSE, row.names = FALSE)

out0 <- list()
out0[["trade"]] <- read.csv(
  trade_path,
  colClasses = c(symbol = "character", buy = "Date", sell = "Date")
)
apy <- sample_apy(30, 1, 10000)
write.csv(apy, apy_path, quote = FALSE, row.names = FALSE)

hist(apy$date, breaks = 100)

index <- em_index("000300")
index <- index[
  index$date >= min(apy$date) & index$date <= max(apy$date), c("date", "close")
]
write.csv(index, index_path, quote = FALSE, row.names = FALSE)

apy_mean <- split(apy, f = apy$date) %>% sapply(function(df) mean(df$apy))
index_n <- normalize(index$close) * (max(apy_mean) - min(apy_mean)) +
  min(apy_mean)
plot(index$date, sgolayfilt(index_n), type = "l")
lines(unique(apy$date), sgolayfilt(apy_mean), col = "red")

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
