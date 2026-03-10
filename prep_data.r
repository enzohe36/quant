# Config =======================================================================

library(foreach)
library(doFuture)
library(data.table)
library(tidyverse)

source("scripts/feat_log_ratios.r")
source("scripts/misc.r")

data_dir <- "data/"
hist_dir <- paste0(data_dir, "hist/")
adjust_dir <- paste0(data_dir, "adjust/")
mc_dir <- paste0(data_dir, "mc/")
val_dir <- paste0(data_dir, "val/")

spot_combined_path <- paste0(data_dir, "spot_combined.csv")

data_combined_path <- paste0(data_dir, "data_combined.rds")
mkt_data_path <- paste0(data_dir, "mkt_data.rds")
feats_path <- paste0(data_dir, "feats.csv")
feats_new_path <- paste0(data_dir, "feats_new.csv")
feats_example_path <- paste0(data_dir, "feats_example.csv")

lookback <- 60

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

set.seed(42)

# Stock Preprocessing ==========================================================

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
    mutate(
      wide = str_starts(symbol, "68") | (str_starts(symbol, "30") & date >= "2020-08-24"),
      lo = if_else(wide, 0.795, 0.895),
      hi = if_else(wide, 1.205, 1.105),
      overnight = close / lag(close),
      fix = case_when(
        !is.na(overnight) & overnight < lo ~ lo / overnight,
        !is.na(overnight) & overnight > hi ~ hi / overnight,
        TRUE ~ 1.0
      ),
      anomaly_adjust = cumprod(fix) / prod(fix),
      across(c(open, high, low, close), ~ .x * anomaly_adjust),
      volume = volume / anomaly_adjust
    ) %>%
    select(names(hist), mc, np, np_deduct, equity, revenue, ocf)

  # Fill suspended dates with all trading dates
  sym_delist <- filter(spot_combined, symbol == !!symbol)$delist
  hist_start <- min(data$date)
  hist_end <- if (sym_delist) max(data$date) else last_td

  data <- data %>%
    right_join(tibble(date = all_td), by = "date") %>%
    filter(date >= hist_start, date <= hist_end) %>%
    arrange(date) %>%
    mutate(
      susp = is.na(close),
      symbol = !!symbol
    ) %>%
    fill(close) %>%
    mutate(
      across(c(open, high, low), ~ if_else(susp, close, .x)),
      across(c(volume, amount, to), ~ if_else(susp, 0, .x))
    ) %>%
    fill(mc, np, np_deduct, equity, revenue, ocf)

  my_list <- list()
  my_list[[symbol]] <- data
  return(my_list)
}

plan(sequential)
saveRDS(data_combined, data_combined_path)
tsprint(str_glue("Combined {length(data_combined)} stocks."))

# Market Preprocessing =========================================================

# spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)
# data_combined <- readRDS(data_combined_path)

plan(multisession, workers = availableCores() - 1)

mkt_data <- foreach(
  data = data_combined,
  .combine = "c"
) %dofuture% {
  if (nrow(data) == 0) return(NULL)
  data %>%
    mutate(
      close_r = close / lag(close),
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

# Feature Generation ===========================================================

# spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)
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
      mutate(price = lead(open)) %>%
      select(symbol, date, price, susp) %>%
      bind_cols(
        make_features(
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
      drop_leading_na() %>%
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
cn_cols <- grep("_cn$", names(feats), value = TRUE)
feats[, (cn_cols) := lapply(.SD, cross_pctrank), by = date, .SDcols = cn_cols]
tsprint(str_glue("Generated features for {length(unique(feats$symbol))} stocks."))

delist_map <- as.data.table(spot_combined)[, .(symbol, delist)]
setDT(delist_map)
delist_map[, symbol := as.character(symbol)]
feats <- merge(feats, delist_map, by = "symbol", all.x = TRUE)
feats[is.na(delist), delist := FALSE]
tsprint(str_glue("Delisted stocks in feats: {sum(feats$delist[!duplicated(feats$symbol)])}"))

elapsed <- (proc.time() - t0)[3]
cat(nrow(feats), "x", ncol(feats), "in", round(elapsed, 3), "s\n")

validate_features(feats, plot = TRUE)

new_dates <- tail(sort(unique(feats$date)), lookback)
feats_new <- feats[date %in% new_dates]
write_csv(feats_new, feats_new_path)
tsprint(str_glue("nrow(feats_new) = {nrow(feats_new)}"))

feats <- na.omit(feats)
write_csv(feats, feats_path)
tsprint(str_glue("nrow(feats) = {nrow(feats)}"))

feats_example <- filter(feats, symbol == "002384")
write_csv(feats_example, feats_example_path)
tsprint(str_glue("nrow(feats_example) = {nrow(feats_example)}"))
