# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools

rm(list = ls())

gc()

source("r_settings.r", encoding = "UTF-8")

scripts <- c("misc.r", "data_retrievers.r")
load_pkgs(scripts) # No tidyverse
library(foreach)
library(tidyverse)

source_scripts(scripts)

# ============================================================================
# Paths and Parameters
# ============================================================================

data_dir <- "data/"
hist_dir <- paste0(data_dir, "hist/")
adjust_dir <- paste0(data_dir, "adjust/")
mc_dir <- paste0(data_dir, "mc/")
val_dir <- paste0(data_dir, "val/")

holidays_path <- paste0(data_dir, "holidays.csv")
indices_path <- paste0(data_dir, "indices.csv")
spot_combined_path <- paste0(data_dir, "spot_combined.csv")

log_dir <- "logs/"

log_path <- paste0(log_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

dir.create(data_dir)
dir.create(hist_dir)
dir.create(adjust_dir)
dir.create(mc_dir)
dir.create(val_dir)
dir.create(log_dir)

end_date <- as_tradedate(now() - hours(16))

# ============================================================================
# Spot Data Update
# ============================================================================

if (!file.exists(spot_combined_path)) {
  spot_combined <- combine_spot()
  write_csv(spot_combined, spot_combined_path)
} else {
  spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)
  last_date <- max(pull(spot_combined, date))
  if (last_date < end_date) {
    spot_combined <- combine_spot()
    write_csv(spot_combined, spot_combined_path)
  }
}
tsprint(glue("Retrieved spot data for {nrow(spot_combined)} symbols."))

# indices <- select(get_index_spot(), c(symbol, name, market))
# write_csv(indices, indices_path)
# tsprint(glue("Updated {nrow(indices)} indices."))

# ============================================================================
# Historical Data Update
# ============================================================================

symbols <- spot_combined %>%
  filter(str_detect(symbol, "^(0|3|6)")) %>%
  # filter(!delist & !susp) %>%
  pull(symbol)

out <- foreach(
  symbol = symbols,
  .combine = "c"
) %do% {
  vars <- c(
    "hist", "hist_path", "adjust", "adjust_path", "adjust_change_date",
    "mc", "mc_path", "shares_change_date",
    "val", "val_path", "val_change_date", "spot_symbol",
    "try_error", "last_date"
  )
  rm(list = vars)

  hist_path <- paste0(hist_dir, symbol, ".csv")
  adjust_path <- paste0(adjust_dir, symbol, ".csv")
  mc_path <- paste0(mc_dir, symbol, ".csv")
  val_path <- paste0(val_dir, symbol, ".csv")

  spot_symbol <- filter(spot_combined, symbol == !!symbol)

  try_error <- try(
    if (!file.exists(hist_path)) {
      hist <- get_hist(symbol, NULL, end_date)
      write_csv(hist, hist_path)
      tsprint(glue("{symbol}: Created hist file."), log_path)
    } else {
      hist <- read_csv(hist_path, show_col_types = FALSE)
      last_date <- max(pull(hist, date))
      if (
        last_date >= end_date |
          pull(spot_symbol, delist) |
          pull(spot_symbol, susp)
      ) {
      } else if (isTRUE(last_date == as_tradedate(end_date - 1))) {
        hist <- bind_rows(hist, select(spot_symbol, names(hist)))
        write_csv(hist, hist_path)
        glue("{symbol}: Appended spot to hist.") %>%
          tsprint(log_path)
      } else {
        hist <- bind_rows(hist, get_hist(symbol, last_date + 1, end_date))
        write_csv(hist, hist_path)
        tsprint(glue("{symbol}: Appended new data to hist."), log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    tsprint(glue("{symbol}: Error retrieving hist."), log_path)
  }

  try_error <- try(
    if (!file.exists(adjust_path)) {
      adjust <- get_adjust(symbol)
      write_csv(adjust, adjust_path)
      tsprint(glue("{symbol}: Created adjust file."), log_path)
    } else {
      adjust <- read_csv(adjust_path, show_col_types = FALSE)
      last_date <- max(pull(adjust, date))
      adjust_change_date <- pull(spot_symbol, adjust_change_date)
      if (
        isTRUE(last_date < adjust_change_date & adjust_change_date <= end_date)
      ) {
        adjust <- get_adjust(symbol)
        write_csv(adjust, adjust_path)
        tsprint(glue("{symbol}: Replaced adjust file."), log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    tsprint(glue("{symbol}: Error retrieving adjust."), log_path)
  }

  try_error <- try(
    if (!file.exists(mc_path)) {
      mc <- get_mc(symbol)
      write_csv(mc, mc_path)
      tsprint(glue("{symbol}: Created mc file."), log_path)
    } else {
      mc <- read_csv(mc_path, show_col_types = FALSE)
      last_date <- max(pull(mc, date))
      shares_change_date <- pull(spot_symbol, shares_change_date)
      if (
        isTRUE(last_date < shares_change_date & shares_change_date <= end_date)
      ) {
        mc <- get_mc(symbol)
        write_csv(mc, mc_path)
        tsprint(glue("{symbol}: Replaced mc file."), log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    tsprint(glue("{symbol}: Error retrieving mc."), log_path)
  }

  try_error <- try(
    if (!file.exists(val_path)) {
      val <- get_val(symbol)
      write_csv(val, val_path)
      tsprint(glue("{symbol}: Created val file."), log_path)
    } else {
      val <- read_csv(val_path, show_col_types = FALSE)
      last_date <- max(pull(val, val_change_date))
      val_change_date <- pull(spot_symbol, val_change_date)
      if (
        isTRUE(last_date < val_change_date & val_change_date <= end_date)
      ) {
        val <- get_val(symbol)
        write_csv(val, val_path)
        tsprint(glue("{symbol}: Replaced val file."), log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    tsprint(glue("{symbol}: Error retrieving val."), log_path)
  }
}

tsprint(glue("Updated {length(symbols)} symbols."))

holidays <- get_holidays(hist_dir)
write_csv(tibble(date = holidays), paste0(data_dir, "holidays.csv"))
tsprint(glue("Updated holidays from {min(holidays)} to {max(holidays)}."))
