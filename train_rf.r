rm(list = ls())

gc()

library(ranger)
library(caret)

options(ranger.num.threads = availableCores(omit = 1))

plan(multisession, workers = availableCores() - 1)

model_dir <- "models/"
train_path <- paste0(model_dir, "train.rds")
test_path <- paste0(model_dir, "test.rds")
model_path <- paste0(model_dir, "model.rds")

train <- readRDS(train_path)
test <- readRDS(test_path)

tune_rf <- function(
  train,
  test,
  num_trees = 500,
  mtry = NULL,
  min_node_size = NULL,
  replace = TRUE
) {
  gc()

  model <- ranger(
    target ~ .,
    data = train,
    num.trees = num_trees,
    mtry = mtry,
    min.node.size = min_node_size,
    replace = replace,
    verbose = FALSE
  )

  cm <- confusionMatrix(predict(model, test)[["predictions"]], test$target)

  out <- data.frame(
    num_trees = num_trees,
    mtry = mtry,
    min_node_size = min_node_size,
    replace = replace,
    oob_acc = 1 - model$prediction.error
  )
  for (i in levels(train$target) %>% sort()) {
    out <- mutate(
      out,
      "test_acc_{i}" := !!cm$byClass[glue("Class: {i}"), "Balanced Accuracy"]
    )
  }
  print(out)
  return(out)
}

res <- list()

for (num_trees in (1:5) * 500)
for (mtry in floor(sqrt(ncol(train) - 1) * 1.5^(-5:5)) %>% unique())
for (min_node_size in 1:10)
for (replace in c(TRUE, FALSE)) {
  res <- tune_rf(train, test, num_trees, mtry, min_node_size, replace) %>%
    list() %>%
    append(res)
}

res <- rbindlist(res)
res_best <- arrange(res, test_acc_1 + test_acc_4) %>% last()
print(res_best)

model <- ranger(
  target ~ .,
  data = train,
  num.trees = res_best$num_trees,
  mtry = res_best$mtry,
  min.node.size = res_best$min_node_size,
  replace = res_best$replace,
  verbose = FALSE
)
saveRDS(model, model_path)
