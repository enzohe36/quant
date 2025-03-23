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
index_path <- paste0(data_dir, "index.csv")
treasury_path <- paste0(data_dir, "treasury.csv")
data_comb_path <- paste0(data_dir, "data_comb.rds")

model_dir <- "models/"

dir.create(model_dir)

beta_q_path <- paste0(model_dir, "beta_q.txt")
test_path <- paste0(model_dir, "test.rds")
rf_path <- paste0(model_dir, "rf.rds")

index_comp <- read_csv(index_comp_path)

# Set observation window
hold_period <- 20

# Load market & risk-free benchmark
index <- read_csv(index_path) %>%
  mutate(r_index = get_roc(close, lead(close, hold_period))) %>%
  select(date, r_index)
treasury <- read_csv(treasury_path) %>%
  select(date, cn10y)

# Load historical stock data & generate features
data_comb <- foreach(
  symb = index_comp$symb,
  .combine = "append"
) %dofuture% {
  rm(list = c("data_path", "data", "index_comp_i", "lst"))

  data_path <- paste0(data_dir, symb, ".csv")
  if (!file.exists(data_path)) return(NULL)

  # Skip stocks with insufficient history
  data <- read_csv(data_path, show_col_types = FALSE)
  # if (nrow(data) <= 720) return(NULL)
  if (nrow(data) <= 840) return(NULL) # For testing only

  index_comp_i <- filter(index_comp, symb == !!symb)

  data <- data %>%
    # Add basic stock info
    mutate(
      symb = !!index_comp_i$symb,
      name = !!index_comp_i$name,
      index = !!index_comp_i$index,
      industry = !!index_comp_i$industry,
      .before = date
    ) %>%
    # Add features
    mutate(
      r = lead(ROC(close, hold_period), hold_period),
      r_sd240 = runSD(ROC(close, hold_period), 240), # Price volatility
      close_pctmin240 = close / runMin(close, 240), # Past growth
      .after = date
    ) %>%
    add_sma("close", c(20, 100)) %>% # Change in price trend
    add_roc("close_sma20", c(10, 20, 40)) %>%
    add_roc("close_sma100", c(50, 100, 200)) %>%
    add_sma("vol", 20) %>% # Change in volume trend
    add_roc("vol_sma20", c(10, 20, 40))

  lst <- list()
  lst[[symb]] <- data
  return(lst)
} %>%
  rbindlist() %>%
  # filter(date > max(date) %m-% months(36)) %>%
  filter(date > max(date) %m-% months(42)) %>% # For testing only
  # Calculate target value
  list(index, treasury) %>%
  reduce(left_join, by = "date") %>%
  mutate(beta = (r - cn10y) / (r_index - cn10y), .after = date)

# Calculate target quantiles
beta_q <- quantile(data_comb$beta, na.rm = TRUE)[2:4]
writeLines(as.character(beta_q), beta_q_path)

data_comb <- data_comb %>%
  # Categorize target value by quantiles
  mutate(
    target = beta %>% findInterval(c(-Inf, !!beta_q, Inf)) %>% as.factor(),
    .after = date
  ) %>%
  # Categorize valuation metrics by industry quantiles
  group_by(industry, date) %>%
  mutate(
    across(
      c(pe, pb, peg, pcf, ps),
      ~ abs(.) %>%
        findInterval(c(-Inf, quantile(., na.rm = TRUE)[2:4], Inf)) %>%
        as.factor()
    )
  ) %>%
  ungroup() %>%
  # Remove temporary features
  mutate(
    across(
      c(beta, r, open:amt, matches("^(close|vol)_sma[0-9]+$"), r_index, cn10y),
      ~ NULL
    )
  )
saveRDS(data_comb, data_comb_path)

# Split data into training & test sets
train <- data_comb %>%
  filter(date <= max(date) %m-% months(7)) %>% # For testing only
  na.omit() %>%
  slice_sample(prop = 0.8, by = c("index", "industry", "date", "target"))
test <- data_comb %>%
  filter(date <= max(date) %m-% months(7)) %>% # For testing only
  na.omit() %>%
  anti_join(train, by = c("symb", "date"))
saveRDS(test, test_path)

# Train random forest model
rf <- ranger(
  target ~ .,
  data = select(train, -c(symb:date)),
  importance = "permutation",
  probability = TRUE
)
saveRDS(rf, rf_path)

get_cm <- function(rf, test, prob_thr = 0.5) {
  compar <- predict(rf, test)[["predictions"]] %>%
    as_data_frame() %>%
    mutate(
      prob_max = apply(., 1, function(v) max(v)),
      pred = apply(., 1, function(v) names(v) %>% .[match(max(v), v)]) %>%
        as.factor(),
    ) %>%
    cbind(select(test, symb:target)) %>%
    filter(prob_max > !!prob_thr)
  cm <- table(Prediction = compar$pred, Target = compar$target) %>%
    confusionMatrix()
  return(cm)
}

# Evaluate model on test set
cm <- get_cm(rf, test)
acc <- round(cm$overall["Accuracy"], 3)
c(
  paste(rep("-", 40), collapse = ""),
  glue("t = 0 d, acc = {acc}, n = {nrow(test)}"),
  "",
  "Target distribution:",
  capture.output(table(test$target))[-1],
  "",
  "Confusion matrix (P > 0.5 only):",
  capture.output(cm$table)
) %>%
  writeLines()

# Evaluate model on future data (for testing only)
for (lag in seq(40, 210, 10) %>% rev()) {
  test <- data_comb %>%
    filter(
      date > max(max(date) %m-% months(7), max(date) - days(lag)),
      date <= max(date) - days(lag - 10)
    ) %>%
    na.omit()
  cm <- get_cm(rf, test)
  acc <- round(cm$overall["Accuracy"], 3)
  c(
    paste(rep("-", 40), collapse = ""),
    glue("t = {211 - lag}-{220 - lag} d, acc = {acc}, n = {nrow(test)}"),
    "",
    "Target distribution:",
    capture.output(table(test$target))[-1],
    "",
    "Confusion matrix (P > 0.5 only):",
    capture.output(cm$table)
  ) %>%
    writeLines()
}

plan(sequential)
