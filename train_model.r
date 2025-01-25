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
data_comb <- rbindlist(readRDS(data_list_path)) %>% na.omit()
h <- hist(data_comb$r, breaks = 1000, plot = FALSE)
nfit <- fit_normal(h$mids, h$density)
saveRDS(nfit, nfit_path)

# Split data into training & test sets
ind <- createDataPartition(as.factor(data_comb$date), p = 0.8, list = FALSE)
train <- slice(data_comb, ind)
saveRDS(train, train_path)

test <- slice(data_comb, -ind)
saveRDS(test, test_path)

# Evaluate model on test set
eval_model <- function(model, test, print_table = FALSE, pos = "a", ...) {
  class <- as.factor(c("a", "b", "c", "d", "e"))

  # Keep only predictions with p > 0.5
  res <- predict(model, test)[["predictions"]] %>%
    as_data_frame() %>%
    mutate(
      prob_max = apply(., 1, function(v) max(v)),
      pred = apply(., 1, function(v) class[match(max(v), v)]),
      target = test$target
    ) %>%
    filter(prob_max > 0.5)

  # Calculate accuracy & discovery rate (for high return)
  cm <- confusionMatrix(res$pred, res$target)
  if (print_table) print(cm$table)

  p_acc <- unname(cm$overall["AccuracyPValue"])
  acc <- ifelse(isTRUE(p_acc < 0.05), unname(cm$overall["Accuracy"]), 0)

  n_true <- sum(sapply(which(class %in% pos), function(n) cm$table[n, n]))
  n_tdr <- sum(sapply(which(class %in% pos), function(n) cm$table[n, ]))
  tdr <- ifelse(isTRUE(p_acc < 0.05) & n_tdr != 0, n_true / n_tdr, 0)

  eval <- c(acc = acc, tdr = tdr, n_tdr = n_tdr)
  return(eval)
}

# Tune mtry using k-fold cross-validation
tune_mtry <- function(train_folds, mtry_list, verbose = FALSE, ...) {
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
    if (verbose) {
      glue(
        "mtry = {mtry}, fold {i}: ",
        "acc = {round(eval[\"acc\"], 3)}, ",
        "tdr = {round(eval[\"tdr\"], 3)}, ",
        "n_tdr = {round(eval[\"n_tdr\"], 3)}.",
      ) %>%
        tsprint()
    }
  }

  # Find best mtry by TDR
  tune <- rbindlist(res) %>%
    group_by(mtry) %>%
    summarise(across(everything(), mean)) %>%
    filter(round(tdr, 2) == max(round(tdr, 2))) %>%
    filter(n_tdr == max(n_tdr)) %>%
    filter(round(acc, 2) == max(round(acc, 2))) %>%
    first() %>%
    unlist()
  return(tune)
}

# Iteratively optimize feature combinations
select_feat <- function(
  train, feat_base, feat_grid, n_try, do_best = TRUE, ...
) {
  row_list <- seq_len(nrow(feat_grid))
  best_list <- c()
  n_step <- n_try * (
    sum(sapply(feat_grid, function(v) length(unique(v)))) -
      ncol(feat_grid) + 1
  )

  for (i in seq_len(n_try)) {
    # Create cross-validation folds
    train_folds <- createFolds(as.factor(train$date), k = 5) %>%
      lapply(function(v) select(train, target:last_col()) %>% slice(v))

    # Choose one combination as initial condition
    row <- row_list[sample(length(row_list), 1)]
    init <- unlist(feat_grid[row, ])
    col_list <- seq_len(ncol(feat_grid))
    res <- c()

    for (j in seq_along(col_list)) {
      # Choose one category in initial condition to optimize
      col <- col_list[sample(length(col_list), 1)]
      var_list <- unique(feat_grid[, col])
      if (j > 1) var_list <- var_list[var_list != init[col]]

      # Evaluate all choices in category
      for (k in seq_along(var_list)) {
        init[col] <- var_list[k]
        feat_k <- paste(init, collapse = "|")
        train_folds_trim <- lapply(
          train_folds, function(df) select(df, feat_base, matches(feat_k))
        )
        mtry <- floor(sqrt(ncol(train_folds_trim[[1]]) - 1))
        tune <- tune_mtry(train_folds_trim, mtry, ...) %>%
          .[names(.) != "mtry"]
        res <- c(res, list(append(as.list(init), as.list(tune))))

        # Report progress
        n_step1 <- length(res) + n_step * (i - 1) / n_try
        glue(
          "Step {n_step1}/{n_step}: ",
          "acc = {round(tune[\"acc\"], 3)}, ",
          "tdr = {round(tune[\"tdr\"], 3)}, ",
          "n_tdr = {round(tune[\"n_tdr\"], 3)}."
        ) %>%
          tsprint()
      }

      # Update initial condition with best choice
      best <- rbindlist(res) %>%
        filter(round(tdr, 2) == max(round(tdr, 2))) %>%
        filter(n_tdr == max(n_tdr)) %>%
        filter(round(acc, 2) == max(round(acc, 2))) %>%
        first() %>%
        unlist()
      init <- best[seq_along(init)]
      col_list <- col_list[col_list != col]
    }

    # Remove duplicate combinations
    row_list <- row_list[row_list != row]
    best_list <- c(best_list, list(as.list(best)))
    feat <- rbindlist(best_list, fill = TRUE) %>% select(seq_along(init))
    if (nrow(distinct(feat)) == nrow(head(feat, -1))) {
      best_list <- head(best_list, -1)
      next
    }

    if (!do_best) next

    model_path <- paste0(model_dir, "rf_", nrow(feat), ".rds")
    eval_path <- paste0(model_dir, "rf_", nrow(feat), "_eval.txt")
    imp_path <- paste0(model_dir, "rf_", nrow(feat), "_imp.pdf")

    # Tune mtry
    feat_i <- paste(init, collapse = "|")
    train_folds_trim <- lapply(
      train_folds, function(df) select(df, feat_base, matches(feat_i))
    )
    mtry_list <- unique(
      c(1, floor(1.5 ^ (-2:2) * sqrt(ncol(train_folds_trim[[1]]) - 1)))
    )
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
      eval <- eval_model(rf, test, print_table = TRUE, ...)
    )
    best_list[length(best_list)] <- append(
      as.list(init), as.list(c(model = nrow(feat), mtry = mtry_best, eval))
    ) %>%
      list()
    eval_out <- glue(
      "mtry = {mtry_best}, ",
      "acc = {round(eval[\"acc\"], 3)}, ",
      "tdr = {round(eval[\"tdr\"], 3)}, ",
      "n_tdr = {round(eval[\"n_tdr\"], 3)}.",
    )

    # Format output
    title <- paste(
      c(
        "-------------------- Model ", str_pad(nrow(feat), 2, pad = 0),
        " --------------------"
      ),
      collapse = ""
    )
    eval <- c(title, cm_table, "", eval_out, "")
    write(eval, eval_path)
    writeLines(eval)

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

  return(rbindlist(best_list))
}

hold_period <- 10

# Define sample space of all feature combinations
feat_grid <- expand.grid(
  valuation = c("$^", "^(mktcap|pe|pb|pc|ps)"),
  mktcost = c("$^", "^(profitable|cr|mktcost_ror)"),
  trend = c("$^", "^adx($|_mom)", "^adx_diff"),
  oscillator = c("$^", "^cci", "^rsi", "^stoch", "^boll"),
  price = c(
    "$^", "^close_roc[0-9]+$", "^close_pctma",
    "^close_tnorm($|_mom)", "^close_tnorm($|_ma)"
  ),
  vol = c(
    "$^", "^vol_roc", "^vol_pctma",
    "^turnover($|_mom)", "^turnover($|_ma[0-9]+$)"
  ),
  val_main = c(
    "$^", "^val_main_roc", "^val_main_pctma",
    "^turnover_main($|_mom)", "^turnover_main($|_ma)"
  ),
  volatility = c(
    "$^", "^amp", paste0("^close_roc", hold_period, "_sd"),
    "^vol_sd", "^turnover_sd"
  ),
  stringsAsFactors = FALSE
)
feat_base <- names(select(train, target:last_col())) %>%
  .[!grepl(paste(unique(unlist(feat_grid)), collapse = "|"), .)]

n_try <- 30

# Select features with different initial conditions
feat <- select_feat(train, feat_base, feat_grid, n_try) %>%
  arrange(desc(tdr * n_tdr))
saveRDS(feat, feat_path)
print(feat)
