source("/Users/anzhouhe/Documents/quant/load_preset.r", encoding = "UTF-8")

# Assign input values
query <- readLines("query.txt")
data_latest <- data_latest

# ------------------------------------------------------------------------------

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
data_query <- foreach(
  symbol = query,
  .combine = rbind
) %dopar% {
  return(data_latest[data_latest$symbol == symbol, ])
}
unregister_dopar

data_query <- data_query[order(data_query$score, decreasing = TRUE), ]
print(data_query, row.names = FALSE)