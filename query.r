source("load_preset.r", encoding = "UTF-8")

# Define parameters
query <- readLines("query.txt")
update <- update

# ------------------------------------------------------------------------------

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
result <- foreach(
  symbol = query,
  .combine = rbind
) %dopar% {
  return(update[update$symbol == symbol, ])
}
unregister_dopar

result <- result[order(result$score, decreasing = TRUE), ]
print(result, row.names = FALSE)
