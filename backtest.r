source("load_preset.r", encoding = "UTF-8")

# Define parameters
symbol_list <- symbol_list
data_list <- data_list
x_b <- 0.75
x_s <- 0.5
r_thr <- 0.01
t_min <- 10
t_max <- 40

# ------------------------------------------------------------------------------

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
out <- foreach(
  symbol = symbol_list,
  .combine = "multiout",
  .export = "pct_change",
  .multicombine = TRUE,
  .init = list(list(), list())
) %dopar% {
  data <- data_list[[symbol]]
  data <- na.omit(data)
  s <- 1
  r_list <- data.frame(matrix(nrow = 0, ncol = 4))
  for (i in 1:nrow(data)) {
    if (i < s) next
    if (data[i, "x"] < x_b) next
    for (j in i:nrow(data)) {
      if (!(
          data[j, "x"] <= x_s &
          pct_change(data[i, "close"], data[j, "close"]) >= r_thr &
          j - i >= t_min &
          j - i <= t_max
        )
      ) next
      s <- j
      break
    }
    r <- pct_change(data[i, "close"], data[s, "close"])
    r_list <- rbind(r_list, c(symbol, data[i, "date"], data[s, "date"], r))
  }

  r_list <- data.frame(
    symbol = r_list[, 1],
    buy = as.Date(as.numeric(r_list[, 2])),
    sell = as.Date(as.numeric(r_list[, 3])),
    r = as.numeric(r_list[, 4])
  )

  r_stats <- c(
    symbol,
    sum(r_list[, 4]) / as.numeric(data[nrow(data), 1] - data[1, 1]) * 365
  )

  return(list(r_stats, r_list))
}
unregister_dopar

r_stats <- data.frame(do.call(rbind, out[[1]]))
r_stats <- data.frame(symbol = r_stats[, 1], apy = as.numeric(r_stats[, 2]))

r_list <- out[[2]]
get_name <- function(df) {
  unique(df[, 1])
}
names(r_list) <- do.call(c, lapply(out[[2]], get_name))
print(paste0(
    "Backtested ", length(r_list), " stocks;",
    " APY mean = ", round(mean(r_stats[, 2]), 2), ",",
    " CV = ", round(sd(r_stats[, 2]), 2), "."
  )
)

r_cat <- do.call(rbind, r_list)
hist <- hist(r_cat[r_cat[, 4] <= 1, 4], breaks = 100, probability = TRUE)

#return(list(r_stats, r_list))
