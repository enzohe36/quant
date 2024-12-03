source("lib/preset.r", encoding = "UTF-8")

source("lib/fn_get_data.r", encoding = "UTF-8")
source("lib/fn_load_data.r", encoding = "UTF-8")
source("lib/fn_backtest.r", encoding = "UTF-8")
source("lib/fn_sample_apy.r", encoding = "UTF-8")

library(signal)

# python -m aktools

trade_path <- "assets/trade10.csv"
apy_path <- "assets/apy30.csv"
csi300_path <- "assets/csi300.csv"

# ------------------------------------------------------------------------------

# get_data("^(00|60)", "hfq")

# out0 <- load_data("^(00|60)", "hfq", today() - years(10), today())

# out0[["trade"]] <- backtest(20, 10, 0.53, 0.09, -0.5, 105)

# write.csv(out0[["trade"]], trade_path, quote = FALSE, row.names = FALSE)

out0 <- list()
out0[["trade"]] <- read.csv(
  trade_path,
  colClasses = c(symbol = "character", buy = "Date", sell = "Date")
)
out0[["apy"]] <- sample_apy(30, 1, 1000)
if (!file.exists(apy_path)) {
  write.csv(out0[["apy"]], apy_path, quote = FALSE, row.names = FALSE)
} else {
  write.table(
    out0[["apy"]],
    apy_path, append = TRUE,
    sep = ",",
    row.names = FALSE, col.names = FALSE
  )
}

hist(out0[["apy"]][, "date"], breaks = 120)

apy <- read.csv(apy_path, colClasses = c(date = "Date"))
apy <- apy[order(apy$date), ]

# [1]   date open close high low volume amount symbol
csi300 <- fromJSON(
  getForm(
    uri = "http://127.0.0.1:8080/api/public/stock_zh_index_daily_em",
    symbol = "sh000300",
    .encoding = "utf-8"
  )
)
csi300 <- csi300[, c(1, 3)]
colnames(csi300) <- c("date", "csi300")
csi300$date <- as.Date(csi300$date)
write.csv(csi300, csi300_path, quote = FALSE, row.names = FALSE)

csi300 <- csi300[csi300$date >= min(apy$date) & csi300$date <= max(apy$date), ]

apy_mean <- split(apy, f = apy$date) %>% sapply(function(df) mean(df$apy))
csi300_n <- normalize(csi300$csi300) * (max(apy_mean) - min(apy_mean)) +
  min(apy_mean)
plot(csi300$date, sgolayfilt(csi300_n), type = "l")
lines(unique(apy$date), sgolayfilt(apy_mean), col = "red")

opt_lm <- function(t) {
  csi300$csi300_tn <- tnormalize(csi300$csi300, t)
  df <- reduce(list(apy, csi300), full_join, by = "date")
  fit <- lm(df$apy ~ df$csi300_tn)

  return(fit$coefficients[2])
}
opt <- optimize(opt_lm, c(1, 250), tol = 0.01)
print(opt)

csi300$csi300_tn <- tnormalize(csi300$csi300, round(opt$minimum))
df <- reduce(list(apy, csi300), full_join, by = "date") %>% na.omit
plot(df$csi300_tn, df$apy, pch = 20, cex = 0.5)

fit <- lm(apy ~ csi300_tn, df)
print(summary(fit))

ci_x <- seq(-0.1, 1.1, 0.01)
ci <- predict(
  fit,
  newdata = data.frame(csi300_tn = ci_x),
  interval = "prediction",
  level = 0.95
)
abline(fit, col="red")
lines(ci_x, ci[, 2], col = "blue", lty = 2)
lines(ci_x, ci[, 3], col = "blue", lty = 2)
