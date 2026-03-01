# PRESET =======================================================================

library(foreach)
library(doFuture)
library(data.table)
library(tidyverse)

source("scripts/ehlers.r")
source("scripts/misc.r")

data_dir <- "data/"
hist_dir <- paste0(data_dir, "hist/")
adjust_dir <- paste0(data_dir, "adjust/")
mc_dir <- paste0(data_dir, "mc/")
val_dir <- paste0(data_dir, "val/")

spot_combined_path <- paste0(data_dir, "spot_combined.csv")

data_combined_path <- paste0(data_dir, "data_combined.rds")
train_path <- paste0(data_dir, "train.csv")
val_path <- paste0(data_dir, "val.csv")
test_path <- paste0(data_dir, "test.csv")
example_path <- paste0(data_dir, "example.csv")

mkt_data_path <- paste0(data_dir, "mkt_data.rds")
mkt_feats_path <- paste0(data_dir, "mkt_feats.csv")

batch_dir <- paste0(data_dir, "feat_batches/")
dir.create(batch_dir)

batch_size <- 500

analysis_dir <- "analysis/"
dir.create(analysis_dir)

logs_dir <- paste0(analysis_dir, "logs/")
dir.create(logs_dir)

log_path <- paste0(logs_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

last_td <- as_date("2026-01-23")
train_start <- last_td %m-% years(10)
val_start <- last_td %m-% years(2)
test_start <- last_td %m-% years(1)

set.seed(42)

# STOCK PREPROCESSING ==========================================================

spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)

quarters_end <- quarter(all_td %m-% months(3), "date_last") %>%
  unique() %>%
  sort()

symbols <- str_remove(list.files(hist_dir), "\\.csv$")
tsprint(str_glue("Found {length(symbols)} stock histories."))

plan(multisession, workers = availableCores() - 1)

data_combined <- foreach(
  symbol = symbols,
  .combine = "c"
) %dofuture% {
  hist_path <- paste0(hist_dir, symbol, ".csv")
  adjust_path <- paste0(adjust_dir, symbol, ".csv")
  mc_path <- paste0(mc_dir, symbol, ".csv")
  val_path <- paste0(val_dir, symbol, ".csv")

  if (!file.exists(adjust_path)) {
    tsprint(str_glue("{symbol}: Missing adjust file."), log_path)
    return(NULL)
  }
  if (!file.exists(mc_path)) {
    tsprint(str_glue("{symbol}: Missing mc file."), log_path)
    return(NULL)
  }
  if (!file.exists(val_path)) {
    tsprint(str_glue("{symbol}: Missing val file."), log_path)
    return(NULL)
  }

  hist <- read_csv(hist_path, show_col_types = FALSE) %>%
    mutate(
      symbol = !!symbol,
      volume = volume * 100,
      to = to / 100
    ) %>%
    relocate(symbol)

  adjust <- read_csv(adjust_path, show_col_types = FALSE) %>%
    mutate(adjust = adjust / last(adjust))

  mc <- read_csv(mc_path, show_col_types = FALSE) %>%
    mutate(mc = mc * 1e8)

  val <- read_csv(val_path, show_col_types = FALSE) %>%
    full_join(tibble(date = !!quarters_end), by = "date") %>%
    arrange(date) %>%
    mutate(
      revenue = run_sum(revenue, 4),
      np = run_sum(np, 4),
      np_deduct = run_sum(np_deduct, 4),
      ocfps = run_sum(ocfps, 4),
      date = case_when(
        month(date) == 3  ~ make_date(year(date), 5, 1),
        month(date) == 6  ~ make_date(year(date), 9, 1),
        month(date) == 9  ~ make_date(year(date), 11, 1),
        month(date) == 12 ~ make_date(year(date) + 1L, 5, 1)
      )
    )

  data <- hist %>%
    full_join(adjust, by = "date") %>%
    full_join(mc, by = "date") %>%
    full_join(val, by = "date") %>%
    arrange(date) %>%
    fill(names(hist), adjust) %>%
    mutate(shares = mc / close) %>%
    fill(shares) %>%
    mutate(
      mc = close * shares,
      equity = bvps * shares,
      ocf = ocfps * shares,
      across(c(open, high, low, close), ~ .x * adjust),
      volume = volume / adjust
    ) %>%
    fill(np, np_deduct, equity, revenue, ocf) %>%
    filter(date %in% pull(hist, date)) %>%
    select(names(hist), mc, np, np_deduct, equity, revenue, ocf)

  my_list <- list()
  my_list[[symbol]] <- data
  return(my_list)
}

plan(sequential)
saveRDS(data_combined, data_combined_path)
tsprint(str_glue("Combined {length(data_combined)} stocks."))

# MARKET PREPROCESSING =========================================================

# spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)
# data_combined <- readRDS(data_combined_path)

plan(multisession, workers = availableCores() - 1)

mkt_data <- foreach(
  data = data_combined,
  .combine = "c"
) %dofuture% {
  if (nrow(data) == 0) return(NULL)
  data %>%
    right_join(tibble(date = all_td), by = "date") %>%
    filter(
      date >= min(data$date),
      date <= if_else(
        filter(spot_combined, symbol == data$symbol[1])$delist,
        max(data$date),
        last_td
      )
    ) %>%
    arrange(date) %>%
    fill(close, mc, np, np_deduct, equity, revenue, ocf) %>%
    mutate(
      close_r = close / lag(close),
      across(c(volume, amount, to), ~ replace_na(.x, 0)),
      mc_traded = volume * close,
      mc_float = (mc_traded / to) %>%
        replace(is.na(.) | is.infinite(.), NA_real_)
    ) %>%
    fill(mc_float) %>%
    group_by(quarter(date, "date_last")) %>%
    mutate(weight = c(rep(NA_real_, n() - 1), last(mc_float))) %>%
    ungroup() %>%
    mutate(weight = lag(weight)) %>%
    fill(weight) %>%
    select(
      date, close_r, weight, mc_traded, mc_float,
      mc, np, np_deduct, equity, revenue, ocf
    ) %>%
    na.omit() %>%
    list()
} %>%
  rbindlist() %>%
  group_by(date) %>%
  summarize(
    n = sum(weight > 0),
    close_r = weighted.mean(close_r, weight),
    to = sum(mc_traded) / sum(mc_float),
    across(c(mc, np, np_deduct, equity, revenue, ocf), sum)
  ) %>%
  ungroup() %>%
  arrange(date) %>%
  mutate(close = cumprod(close_r) / first(close_r))

plan(sequential)
saveRDS(mkt_data, mkt_data_path)
tsprint(str_glue("Generated market data from {min(mkt_data$date)} to {max(mkt_data$date)}."))

# MARKET FEATURES ==============================================================

# mkt_data <- readRDS(mkt_data_path)

t0 <- proc.time()

mkt_feats <- ehlers_features(
  close     = mkt_data$close,
  volume    = mkt_data$to,
  to        = mkt_data$to,
  mc        = mkt_data$mc,
  np        = mkt_data$np,
  np_deduct = mkt_data$np_deduct,
  equity    = mkt_data$equity,
  revenue   = mkt_data$revenue,
  ocf       = mkt_data$ocf
) %>%
  rename_with(~ paste0("mkt_", .x)) %>%
  mutate(
    date = mkt_data$date, .before = 1,
    across(matches("_dn$"), ~ NULL)
  ) %>%
  filter(date >= train_start) %>%
  na.omit() %>%
  as_tibble()

elapsed <- (proc.time() - t0)[3]
cat(nrow(mkt_feats), "x", ncol(mkt_feats), "in", round(elapsed, 3), "s\n")

validate_features(mkt_feats)

write_csv(mkt_feats, mkt_feats_path)
tsprint(str_glue("nrow(mkt_feats) = {nrow(mkt_feats)}"))

# STOCK FEATURES ===============================================================

# data_combined <- readRDS(data_combined_path)
# mkt_data <- readRDS(mkt_data_path)

t0 <- proc.time()
plan(multisession, workers = availableCores() - 1)

batches <- split(seq_along(data_combined), ceiling(seq_along(data_combined) / batch_size))

for (b in seq_along(batches)) {
  batch_feats <- foreach(
    data = data_combined[batches[[b]]],
    .combine = "c"
  ) %dofuture% {
    data %>%
      select(symbol, date, open, close) %>%
      bind_cols(
        ehlers_features(
          close     = data$close,
          open      = data$open,
          high      = data$high,
          low       = data$low,
          volume    = data$volume,
          amount    = data$amount,
          to        = data$to,
          mc        = data$mc,
          mkt_mc    = left_join(data, mkt_data, by = "date")$mc.y,
          np        = data$np,
          np_deduct = data$np_deduct,
          equity    = data$equity,
          revenue   = data$revenue,
          ocf       = data$ocf
        )
      ) %>%
      filter(date >= train_start) %>%
      na.omit() %>%
      list()
  } %>%
    rbindlist()

  fwrite(batch_feats, paste0(batch_dir, "batch_", b, ".csv"))
  rm(batch_feats)
  gc()
  tsprint(str_glue("Batch {b}/{length(batches)} done."))
}

plan(sequential)

feats <- rbindlist(lapply(list.files(batch_dir, full.names = TRUE), fread, colClasses = c(symbol = "character")))
unlink(batch_dir, recursive = TRUE)

setDT(feats)
dn_cols <- grep("_dn$", names(feats), value = TRUE)
feats[, (dn_cols) := lapply(.SD, cross_pctrank), by = date, .SDcols = dn_cols]
tsprint(str_glue("Generated features for {length(unique(feats$symbol))} stocks."))

elapsed <- (proc.time() - t0)[3]
cat(nrow(feats), "x", ncol(feats), "in", round(elapsed, 3), "s\n")

validate_features(feats, plot = TRUE)

train <- filter(feats, date < val_start)
write_csv(train, train_path)
tsprint(str_glue("nrow(train) = {nrow(train)}"))

val <- filter(feats, date >= val_start & date < test_start)
write_csv(val, val_path)
tsprint(str_glue("nrow(val) = {nrow(val)}"))

test <- filter(feats, date >= test_start)
write_csv(test, test_path)
tsprint(str_glue("nrow(test) = {nrow(test)}"))

example <- filter(train, symbol == "002384")
write_csv(example, example_path)
tsprint(str_glue("nrow(example) = {nrow(example)}"))
