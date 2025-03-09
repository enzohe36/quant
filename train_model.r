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
train <- slice_sample(data_comb, prop = 0.8, by = c("index", "date"))
saveRDS(train, train_path)

test <- anti_join(data_comb, train, by = c("symbol", "date"))
saveRDS(test, test_path)

# Evaluate model on test set
eval_model <- function(
  model, test, print_table = FALSE, coi = "a", prob_thr = 0, ...
) {
  class <- c("a", "b", "c", "d", "e")

  # Keep only predictions with p > 0.5
  compar <- predict(model, test)[["predictions"]] %>%
    as_data_frame() %>%
    select(!!class) %>%
    mutate(
      prob_max = apply(., 1, function(v) max(v)),
      pred = apply(., 1, function(v) !!class %>% .[match(max(v), v)]) %>%
        as.factor(),
      target = !!test$target
    ) %>%
    filter(prob_max > !!prob_thr)

  # Calculate accuracy & discovery rate (for high return)
  cm <- table(
    Prediction = factor(compar$pred, levels = class),
    Reference = factor(compar$target, levels = class)
  ) %>%
    confusionMatrix()
  if (print_table) print(cm$table)

  p_acc <- unname(cm$overall["AccuracyPValue"])
  acc <- ifelse(isTRUE(p_acc < 0.05), unname(cm$overall["Accuracy"]), 0)

  n_true <- sum(sapply(coi, function(str) cm$table[str, str]))
  n_coi <- sum(sapply(coi, function(str) cm$table[str, ]))
  acc_coi <- ifelse(isTRUE(p_acc < 0.05) & n_coi != 0, n_true / n_coi, 0)

  eval <- c(acc = acc, acc_coi = acc_coi, n_coi = n_coi)
  return(eval)
}

# Tune ntree & mtry using k-fold cross-validation
tune_rf <- function(
  train_folds, ntree_list, mtry_list, verbose = FALSE, ...
) {
  res <- c()

  for (ntree in ntree_list)
  for (mtry in mtry_list)
  for (i in seq_along(train_folds)) {
    # Split data into training & validation sets
    train_i <- rbindlist(train_folds[-i])
    valid_i <- train_folds[[i]]

    rf <- ranger(
      target ~ .,
      data = train_i,
      num.trees = ntree,
      mtry = mtry,
      importance = "permutation",
      probability = TRUE,
      verbose = FALSE
    )

    # Evaluate model on validation set
    eval <- eval_model(rf, valid_i, ...)
    res <- c(res, list(as.list(c(ntree = ntree, mtry = mtry, eval))))
    eval_out <- paste(round(eval, 3), collapse = ", ")
    if (verbose) {
      tsprint(glue("ntree = {ntree}, mtry = {mtry}, fold {i}: {eval_out}."))
    }
  }

  # Find best ntree & mtry by TDR
  tune <- rbindlist(res) %>%
    group_by(ntree, mtry) %>%
    summarise(across(everything(), mean)) %>%
    ungroup() %>%
    filter(round(acc_coi, 2) == max(round(acc_coi, 2))) %>%
    filter(round(acc, 2) == max(round(acc, 2))) %>%
    filter(ntree * mtry <= min(ntree * mtry)) %>%
    unlist()
  return(tune)
}

# Iteratively optimize feature combinations
select_feat <- function(
  train, feat_grid, n_try, do_best = TRUE, ...
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
      lapply(function(v) slice(train, v))

    # Choose one combination as initial condition
    init <- unlist(feat_grid[row, ])
    col_list <- sample(ncol(feat_grid))
    res <- c()

    for (col in col_list) {
      # Choose one category in initial condition to optimize
      var_list <- unique(feat_grid[, col])
      if (col != col_list[1]) var_list <- var_list[var_list != init[col]]

      # Evaluate all choices in category
      for (var in var_list) {
        init[col] <- var
        feat_i <- paste(init, collapse = "|")
        train_folds_trim <- lapply(
          train_folds, function(df) select(df, target, matches(feat_i))
        )
        ntree <- 100
        mtry <- floor(sqrt(ncol(train_folds_trim[[1]]) - 1))
        eval <- tune_rf(train_folds_trim, ntree, mtry, ...) %>%
          .[!names(.) %in% c("ntree", "mtry")]
        eval_out <- paste(round(eval, 3), collapse = ", ")
        res <- c(res, list(append(as.list(init), as.list(eval))))

        # Report progress
        n_step1 <- length(res) +
          n_step * (which(row_list == row) - 1) / length(row_list)
        tsprint(
          glue("Step {n_step1}/{n_step}: [{row}, {col}] = {var}; {eval_out}.")
        )
      }

      # Update initial condition with best choice
      best <- rbindlist(res) %>%
        filter(round(acc_coi, 2) == max(round(acc_coi, 2))) %>%
        filter(round(acc, 2) == max(round(acc, 2))) %>%
        slice_sample(n = 1)
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

    # Tune ntree & mtry
    feat_i <- paste(init, collapse = "|")
    train_folds_trim <- lapply(
      train_folds, function(df) select(df, target, matches(feat_i))
    )
    ntree_list <- seq(250, 1000, length.out = 4)
    mtry_list <- floor(2^(-2:2) * sqrt(ncol(train_folds_trim[[1]]) - 1)) %>%
      sapply(max, 1) %>%
      sapply(min, ncol(train_folds_trim[[1]]) - 1) %>%
      unique()
    tune <- tune_rf(train_folds_trim, ntree_list, mtry_list, ...)
    ntree_best <- unname(tune[names(tune) == "ntree"])
    mtry_best <- unname(tune[names(tune) == "mtry"])

    # Grow a larger forest for higher accuracy
    rf <- ranger(
      target ~ .,
      data = rbindlist(train_folds_trim),
      num.trees = ntree_best,
      mtry = mtry_best,
      importance = "permutation",
      probability = TRUE,
      verbose = FALSE
    )
    saveRDS(rf, model_path)

    # Evaluate model on reserved test set
    cm_table <- capture.output(
      eval <- eval_model(rf, test, print_table = TRUE, prob_thr = 0.5, ...)
    )
    best_list[length(best_list)] <- as.list(init) %>%
      append(c(model = i, ntree = ntree_best, mtry = mtry_best, eval)) %>%
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
    "$^", "^(mktcap|pe$|peg|pb|pc|ps)"
  ),
  mktcost = c(
    "$^", "^(profitable|cr|mktcost_ror)"
  ),
  price = c(
    "^close_roc[0-9]+$", "^close_tnroc", "^close_pctsma", "^close_tnorm_sma"
  ),
  vol = c(
    "^vol_tnroc", "^vol_pctsma", "^turnover_sma"
  ),
  val_main = c(
    "$^", "^val_main_pctsma", "^turnover_main_sma"
  ),
  volatility = c(
    "$^", paste0("^close_roc", hold_period, "_sd"), "^vol_sd", "^turnover_sd"
  ),
  stringsAsFactors = FALSE
)

n_try <- 30

# Select features with different initial conditions
feat <- select_feat(train, feat_grid, n_try)
saveRDS(feat, feat_path)

# Debug
if (FALSE) {
  for (i in seq_along(names(feat_grid))) {
    name <- names(feat_grid)[i]
    assign(
      name,
      names(data_comb) %>%
        .[grepl(paste(unique(feat_grid[, name]), collapse = "|"), .)]
    )
    if (i == 1) print(intersect(names(feat_grid)[i], last(names(feat_grid))))
    if (i > 1) print(intersect(names(feat_grid)[i], names(feat_grid)[i - 1]))
    if (i == length(names(feat_grid))) {
      print(
        names(data_comb) %>%
          .[!grepl(paste(unique(unlist(feat_grid)), collapse = "|"), .)]
      )
    }
  }
}
