rm(list = ls())
gc()

library(doFuture)
library(foreach)
library(TTR)
library(data.table)
library(glue)
library(ranger)
library(caret)
library(tidyverse)

# Load custom settings & helper functions
source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

plan(multisession, workers = availableCores() - 1)

model_dir <- "models/"

data_comb_trim_path <- paste0(model_dir, "data_comb_trim.rds")
rf_path <- paste0(model_dir, "rf.rds")

t_obs <- 5
t_train <- 1200
t_test <- 10

data_comb_trim <- read_rds(data_comb_trim_path)

predict_probrf <- function(rf, test) {
  predict(rf, test)[["predictions"]] %>%
    as_data_frame() %>%
    mutate(
      prob_max = apply(., 1, function(v) max(v)),
      pred = apply(., 1, function(v) names(v) %>% .[match(max(v), v)]) %>%
        as.factor(),
      target = !!test$target,
      symbol = !!test$symbol,
      date = !!test$date
    )
}

date_all <- unique(data_comb_trim$date) %>% sort()

for (i in seq_len(t_obs + 1)) {
  rm("obs")
  gc()

  if (i == 1) {
    date_train <- head(date_all, -(t_obs + t_test)) %>% tail(t_train)
    glue("Iteration {i}: [{first(date_train)}, {last(date_train)}]") %>%
      writeLines()
    train <- filter(data_comb_trim, date %in% date_train)
  } else {
    date_obs <- date_all %>% .[. > max(date_train)] %>% nth(i - 1)
    glue("Iteration {i}: {date_obs}") %>% writeLines()
    obs <- filter(data_comb_trim, date == date_obs)
    obs <- mutate(obs, target = predict_probrf(rf, obs)$pred)
    train <- rbindlist(list(train, obs)) %>% na.omit()
  }

  rf <- ranger(
    target ~ .,
    data = select(train, -c(symbol:date)),
    replace = FALSE,
    num.trees = 1000,
    probability = TRUE
  )
  glue("OOB error = {rf$prediction.error}") %>% writeLines()
  writeLines("")
}

for (i in t_test) {
  date_test <- date_all %>% .[. > max(date_train)] %>% nth(t_obs + i)
  glue("Test {i}: {date_test}") %>% writeLines()

  test <- filter(data_comb_trim, date == date_test)
  comp <- predict_probrf(rf, test) %>% filter(prob_max > 0.5)
  cm <- confusionMatrix(comp$pred, comp$target)
  print(cm)
  writeLines("")
}
