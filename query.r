source("/Users/anzhouhe/Documents/quant/load_preset.r", encoding = "UTF-8")

# Assign input values
data_latest <- data_latest
query <- readLines("query.txt")

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
cat(
  capture.output(print(data_query, row.names = FALSE)),
  file = "ranking_query.txt",
  sep = "\n"
)
print(paste0(
    format(now(tzone = "Asia/Shanghai"), "%H:%M:%S"),
    " Queried ", nrow(data_query), " stock(s);",
    " wrote to ranking_query.txt"
  )
)
