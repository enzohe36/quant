t_adx <- 6

data <- data_list[[1]]
data <- cbind(data, ADX())
data$ADX <- (SMA(diff) * 5 + diff) / 6