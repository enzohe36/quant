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

data_dir <- "data/"

index_comp_path <- paste0(data_dir, "index_comp.csv")
data_comb_path <- paste0(data_dir, "data_comb.rds")

model_dir <- "models/"

dir.create(model_dir)

target_q_path <- paste0(model_dir, "target_q.rds")
test_path <- paste0(model_dir, "test.rds")
rf_path <- paste0(model_dir, "rf.rds")

index_comp <- read_csv(index_comp_path)

# Set time parameters
end_date <- as_tradedate(now() - hours(16))
t_train <- 3 # In months
t_lag <- 15 # In months
t_hold <- 20 # In days

# Load historical stock data & generate features
data_comb <- foreach(
  symbol = index_comp$symbol,
  .combine = "append"
) %dofuture% {
  rm(list = c("data_path", "data", "index_comp_i", "lst"))

  data_path <- paste0(data_dir, symbol, ".csv")
  if (!file.exists(data_path)) return(NULL)

  # Skip stocks with insufficient history
  data <- read_csv(data_path, show_col_types = FALSE)
  if (min(data$date) > end_date %m-% months(t_train + t_lag)) return(NULL)

  index_comp_i <- filter(index_comp, symbol == !!symbol)

  data <- data %>%
    # Add basic stock info
    mutate(
      symbol = !!index_comp_i$symbol,
      name = !!index_comp_i$name,
      index = !!index_comp_i$index,
      industry = !!index_comp_i$industry,
      .before = date
    ) %>%
    # Add features
    mutate(
      target = get_roc(close, lead(runMax(close, t_hold), t_hold)),
      r_sd240 = runSD(ROC(close, t_hold), t_hold),
      close_pctmin240 = close / runMin(close, 240),
      .after = date
    ) %>%
    add_sma("close", c(20, 100)) %>% # Change in price trend
    add_roc("close_sma20", c(10, 20, 40)) %>%
    add_roc("close_sma100", c(50, 100, 200))

  lst <- list()
  lst[[symbol]] <- data
  return(lst)
} %>%
  rbindlist() %>%
  mutate(
    mktcost = get_roc(mktcost, close),
    mktcap = mktcap / 10^10,
    across(c(open:amt, pe:ps, matches("^(close|vol)_sma[0-9]+$")), ~ NULL)
  )

target_q <- c(-Inf, quantile(data_comb$target, na.rm = TRUE)[2:4], Inf)
saveRDS(target_q, target_q_path)

data_comb <- mutate(
  data_comb, target = findInterval(target, target_q) %>% as.factor()
)
saveRDS(data_comb, data_comb_path)

train <- na.omit(data_comb) %>%
  slice_sample(prop = 0.8, by = c("date", "index", "industry", "target"))
test <- na.omit(data_comb) %>%
  anti_join(train, by = c("date", "symbol"))
saveRDS(test, test_path)

# Train random forest model
rf <- ranger(
  target ~ .,
  data = select(train, -c(symbol:date)),
  importance = "permutation",
  probability = TRUE
)

get_cm <- function(rf, test, prob_thr = 0.5) {
  compar <- predict(rf, test)[["predictions"]] %>%
    as_data_frame() %>%
    mutate(
      prob_max = apply(., 1, function(v) max(v)),
      pred = apply(., 1, function(v) names(v) %>% .[match(max(v), v)]) %>%
        as.factor(),
      target = !!test$target
    ) %>%
    filter(prob_max > !!prob_thr)
  cm <- table(Prediction = compar$pred, Target = compar$target) %>%
    confusionMatrix()
  return(cm)
}

cm <- get_cm(rf, test)
print(cm)

par(mar = c(5.1, 9, 4.1, 2.1))
barplot(
  rf[["variable.importance"]] %>% sort(),
  horiz = TRUE, las = 1,
  xlab = "Variable importance"
)

print_eval <- function(rf, test, t1, t2 = NULL) {
  cm <- get_cm(rf, test)
  acc <- round(cm$overall["Accuracy"], 3)
  if (is.null(t2)) t_out <- glue("{t1}") else t_out <- glue("{t1}-{t2}")
  c(
    paste(rep("-", 40), collapse = ""),
    glue("t = {t_out}, acc = {acc}, n = {nrow(test)}"),
    "",
    "Target distribution:",
    capture.output(table(test$target))[-1],
    "",
    "Confusion matrix (P > 0.5 only):",
    capture.output(cm$table)
  ) %>%
    writeLines()
}

# Evaluate model on test set
print_eval(rf, test, 0)

for (lag in seq(40, 210, 10) %>% rev()) { # For testing only
  test <- data_comb %>%
    filter(
      date > max(max(date) %m-% months(7), max(date) - days(lag)),
      date <= max(date) - days(lag - 10)
    ) %>%
    na.omit()
  print_eval(rf, test, 211 - lag, 220 - lag)
}

plan(sequential)
