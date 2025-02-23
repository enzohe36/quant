rm(list = ls())

gc()

library(doFuture)
library(data.table)
library(glue)
library(caret)
library(ranger)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

data_dir <- "data/"
model_dir <- "models/"

dir.create(model_dir)

data_list_path <- paste0(data_dir, "data_list.rds")
nfit_path <- paste0(model_dir, "nfit.rds")
train_path <- paste0(model_dir, "train.rds")
test_path <- paste0(model_dir, "test.rds")
feat_path <- paste0(model_dir, "feat.rds")

# Calculate mean & sd of market return
data_comb <- readRDS(data_list_path) %>% rbindlist() %>% na.omit()
h <- hist(data_comb$r, breaks = 1000, plot = FALSE)
nfit <- fit_normal(h$mids, h$density)
saveRDS(nfit, nfit_path)

# Split data into training & test sets
train_ind <- createDataPartition(
  as.factor(data_comb$date), p = 0.8, list = FALSE
)
train <- slice(data_comb, train_ind) %>% slice_sample(n = 1000)
saveRDS(train, train_path)

test <- slice(data_comb, -train_ind)
saveRDS(test, test_path)

# Evaluate model on test set
eval_model <- function(
  model, test, print_table = FALSE, coi = "a", prob_thr = 0.5, ...
) {
  class <- c("a", "b", "c", "d", "e")

  # Keep only predictions with p > 0.5
  compar <- predict(model, test)[["predictions"]] %>%
    as_data_frame() %>%
    mutate(
      prob_max = apply(., 1, function(v) max(v)),
      pred = apply(., 1, function(v) !!class %>% .[match(max(v), v)]) %>%
        as.factor(),
      target = !!test$target
    ) %>%
    filter(prob_max > !!prob_thr)

  # Calculate accuracy & discovery rate (for high return)
  cm <- confusionMatrix(compar$pred, compar$target)
  if (print_table) print(cm$table)

  p_acc <- unname(cm$overall["AccuracyPValue"])
  acc <- ifelse(isTRUE(p_acc < 0.05), unname(cm$overall["Accuracy"]), 0)

  n_true <- sum(sapply(coi, function(str) cm$table[str, str]))
  n_coi <- sum(sapply(coi, function(str) cm$table[str, ]))
  acc_coi <- ifelse(isTRUE(p_acc < 0.05) & n_coi != 0, n_true / n_coi, 0)

  eval <- c(acc = acc, acc_coi = acc_coi, n_coi = n_coi)
  return(eval)
}

# Tune mtry using k-fold cross-validation
tune_mtry <- function(
  train_folds, mtry_list, verbose = FALSE, ...
) {
  res <- c()

  for (mtry in mtry_list) for (i in seq_along(train_folds)) {
    # Split data into training & validation sets
    train_i <- rbindlist(train_folds[-i])
    valid_i <- train_folds[[i]]

    # Grow a smaller forest for faster speed
    rf <- ranger(
      target ~ .,
      data = train_i,
      num.trees = 100,
      mtry = mtry,
      importance = "permutation",
      probability = TRUE,
      verbose = FALSE
    )

    # Evaluate model on validation set
    eval <- eval_model(rf, valid_i, ...)
    res <- c(res, list(as.list(c(mtry = mtry, eval))))
    eval_out <- paste(round(eval, 3), collapse = ", ")
    if (verbose) tsprint(glue("mtry = {mtry}, fold {i}: {eval_out}."))
  }

  # Find best mtry by TDR
  tune <- rbindlist(res) %>%
    group_by(mtry) %>%
    summarise(across(everything(), mean)) %>%
    filter(round(acc_coi, 2) == max(round(acc_coi, 2))) %>%
    filter(n_coi == max(n_coi)) %>%
    filter(round(acc, 2) == max(round(acc, 2))) %>%
    first() %>%
    unlist()
  return(tune)
}

# Iteratively optimize feature combinations
select_feat <- function(
  train, feat_base, feat_grid, n_try, do_best = TRUE, ...
) {
  row_list <- sample(nrow(feat_grid), n_try)
  best_list <- c()
  n_step <- n_try * (
    sum(sapply(feat_grid, function(v) length(unique(v)))) -
      ncol(feat_grid) + 1
  )

  for (row in row_list) {
    # Create cross-validation folds
    train_folds <- createFolds(as.factor(train$date), k = 5) %>%
      lapply(function(v) select(train, target:last_col()) %>% slice(v))

    # Choose one combination as initial condition
    init <- unlist(feat_grid[row, ])
    col_list <- sample(ncol(feat_grid))
    res <- c()

    for (col in col_list) {
      # Choose one category in initial condition to optimize
      var_list <- unique(feat_grid[, col]) %>% .[sample(length(.))]
      if (col != col_list[1]) var_list <- var_list[var_list != init[col]]

      # Evaluate all choices in category
      for (var in var_list) {
        init[col] <- var
        feat_i <- paste(init, collapse = "|")
        train_folds_trim <- lapply(
          train_folds, function(df) select(df, feat_base, matches(feat_i))
        )
        mtry <- floor(sqrt(ncol(train_folds_trim[[1]]) - 1))
        eval <- tune_mtry(train_folds_trim, mtry, ...) %>%
          .[names(.) != "mtry"]
        eval_out <- paste(round(eval, 3), collapse = ", ")
        res <- c(res, list(append(as.list(init), as.list(eval))))

        # Report progress
        n_step1 <- length(res) +
          n_step * (which(row_list == row) - 1) / length(row_list)
        tsprint(
          glue("Step {n_step1}/{n_step}: [{row}, {col}] = {var}; {eval_out}")
        )
      }

      # Update initial condition with best choice
      best <- rbindlist(res) %>%
        filter(round(acc_coi, 2) == max(round(acc_coi, 2))) %>%
        filter(n_coi == max(n_coi)) %>%
        filter(round(acc, 2) == max(round(acc, 2))) %>%
        first()
      init <- unlist(best)[seq_along(init)]
    }

    # Remove duplicate combinations
    best_list <- c(best_list, list(best))
    feat <- rbindlist(best_list, fill = TRUE) %>% select(seq_along(init))
    if (nrow(distinct(feat)) == nrow(head(feat, -1))) {
      best_list <- head(best_list, -1)
      writeLines(c("Skipping duplicate combination...", ""))
      next
    }

    if (!do_best) next

    i <- nrow(feat)

    model_path <- paste0(model_dir, "rf_", i, ".rds")
    cm_path <- paste0(model_dir, "rf_", i, "_cm.txt")
    imp_path <- paste0(model_dir, "rf_", i, "_imp.pdf")

    # Tune mtry
    feat_i <- paste(init, collapse = "|")
    train_folds_trim <- lapply(
      train_folds, function(df) select(df, feat_base, matches(feat_i))
    )
    mtry_list <- floor(1.5^(-3:3) * sqrt(ncol(train_folds_trim[[1]]) - 1)) %>%
      sapply(max, 1) %>%
      sapply(min, ncol(train_folds_trim[[1]]) - 1) %>%
      c(1) %>%
      unique() %>%
      sort()
    mtry_best <- tune_mtry(train_folds_trim, mtry_list, ...) %>%
      .[names(.) == "mtry"] %>%
      unname()

    # Grow a larger forest for higher accuracy
    rf <- ranger(
      target ~ .,
      data = rbindlist(train_folds_trim),
      mtry = mtry_best,
      importance = "permutation",
      probability = TRUE,
      verbose = FALSE
    )
    saveRDS(rf, model_path)

    # Evaluate model on reserved test set
    cm_table <- capture.output(
      eval <- eval_model(rf, test, print_table = TRUE, prob_thr = 0, ...)
    )
    best_list[length(best_list)] <- as.list(init) %>%
      append(c(model = i, mtry = mtry_best, eval)) %>%
      list()

    # Format output
    best_out <- last(best_list) %>%
      lapply(function(x) if (is.numeric(x)) round(x, 3) else x) %>%
      unlist() %>%
      cbind(str_pad(names(.), max(str_length(names(.))), "right")) %>%
      apply(1, function(v) paste(c(v[2], v[1]), collapse = " = "))
    cm <- c(cm_table, "", best_out, "")
    write(cm, cm_path)
    writeLines(cm)

    pdf(imp_path, height = 9.5)
    par(mar = c(5.1, 9, 4.1, 2.1))

    barplot(
      rf[["variable.importance"]] %>% sort(),
      horiz = TRUE, las = 1, cex.names = 0.8,
      xlab = "Variable importance",
      main = glue("Model {i}")
    )

    dev.off()
  }

  feat <- rbindlist(best_list) %>%
    relocate(model, .before = 1)
  return(feat)
}

hold_period <- 10

# Define sample space of all feature combinations
feat_grid <- expand.grid(
  valuation = c(
    "$^", "^(mktcap|pe|pb|pc|ps)"
  ),
  mktcost = c(
    "$^", "^(profitable|cr|mktcost_ror)"
  ),
  trend = c(
    "$^", "^adx($|_mom)", "^adx_diff"
  ),
  oscillator = c(
    "$^", "^cci", "^rsi", "^stoch", "^boll"
  ),
  price = c(
    "$^", "^close_roc[0-9]+$", "^close_pctma", "^close_tnorm($|_ma)"
  ),
  vol = c(
    "$^", "^vol_pctma", "^turnover($|_ma[0-9]+$)"
  ),
  val_main = c(
    "$^", "^val_main_pctma", "^turnover_main($|_ma)"
  ),
  volatility = c(
    "$^", "^amp", paste0("^close_roc", hold_period, "_sd"),
    "^vol_sd", "^turnover_sd"
  ),
  stringsAsFactors = FALSE
)
feat_base <- names(select(train, target:last_col())) %>%
  .[!grepl(paste(unique(unlist(feat_grid)), collapse = "|"), .)]

n_try <- 3

# Select features with different initial conditions
feat <- select_feat(train, feat_base, feat_grid, n_try)
saveRDS(feat, feat_path)
