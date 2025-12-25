# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools

# PRESET =======================================================================

source_scripts(
  scripts = c("misc", "data_retrievers"),
  packages = c("foreach", "tidyverse")
)

data_dir <- "data/"
spot_combined_path <- paste0(data_dir, "spot_combined.csv")

hist_dir <- paste0(data_dir, "hist/")
adjust_dir <- paste0(data_dir, "adjust/")
mc_dir <- paste0(data_dir, "mc/")
val_dir <- paste0(data_dir, "val/")

log_dir <- "logs/"
log_path <- paste0(log_dir, format(now(), "%Y%m%d_%H%M%S"), ".log")

last_td <- eval(last_td_expr)

# SPOT DATA ====================================================================

dir.create(data_dir)
dir.create(hist_dir)
dir.create(adjust_dir)
dir.create(mc_dir)
dir.create(val_dir)
dir.create(log_dir)

if (!file.exists(spot_combined_path)) {
  spot_combined <- combine_spot()
  write_csv(spot_combined, spot_combined_path)
} else {
  spot_combined <- read_csv(spot_combined_path, show_col_types = FALSE)
  last_date <- max(pull(spot_combined, date))
  if (last_date < last_td) {
    spot_combined <- combine_spot()
    write_csv(spot_combined, spot_combined_path)
  }
}
tsprint(str_glue("Retrieved spot data for {nrow(spot_combined)} stocks."))

# HISTORICAL DATA ==============================================================

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
      hist <- get_hist(symbol, NULL, last_td)
      write_csv(hist, hist_path)
      tsprint(str_glue("{symbol}: Created hist file."), log_path)
    } else {
      hist <- read_csv(hist_path, show_col_types = FALSE)
      last_date <- max(pull(hist, date))
      if (
        last_date >= last_td |
          pull(spot_symbol, delist) |
          pull(spot_symbol, susp)
      ) {
      } else if (isTRUE(last_date == as_tradeday(last_td - 1))) {
        hist <- bind_rows(hist, select(spot_symbol, names(hist)))
        write_csv(hist, hist_path)
        str_glue("{symbol}: Appended spot to hist.") %>%
          tsprint(log_path)
      } else {
        hist <- bind_rows(hist, get_hist(symbol, last_date + 1, last_td))
        write_csv(hist, hist_path)
        tsprint(str_glue("{symbol}: Appended new data to hist."), log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    tsprint(str_glue("{symbol}: Error retrieving hist."), log_path)
  }

  try_error <- try(
    if (!file.exists(adjust_path)) {
      adjust <- get_adjust(symbol)
      write_csv(adjust, adjust_path)
      tsprint(str_glue("{symbol}: Created adjust file."), log_path)
    } else {
      adjust <- read_csv(adjust_path, show_col_types = FALSE)
      last_date <- max(pull(adjust, date))
      adjust_change_date <- pull(spot_symbol, adjust_change_date)
      if (
        isTRUE(last_date < adjust_change_date & adjust_change_date <= last_td)
      ) {
        adjust <- get_adjust(symbol)
        write_csv(adjust, adjust_path)
        tsprint(str_glue("{symbol}: Replaced adjust file."), log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    tsprint(str_glue("{symbol}: Error retrieving adjust."), log_path)
  }

  try_error <- try(
    if (!file.exists(mc_path)) {
      mc <- get_mc(symbol)
      write_csv(mc, mc_path)
      tsprint(str_glue("{symbol}: Created mc file."), log_path)
    } else {
      mc <- read_csv(mc_path, show_col_types = FALSE)
      last_date <- max(pull(mc, date))
      shares_change_date <- pull(spot_symbol, shares_change_date)
      if (
        isTRUE(last_date < shares_change_date & shares_change_date <= last_td)
      ) {
        mc <- get_mc(symbol)
        write_csv(mc, mc_path)
        tsprint(str_glue("{symbol}: Replaced mc file."), log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    tsprint(str_glue("{symbol}: Error retrieving mc."), log_path)
  }

  try_error <- try(
    if (!file.exists(val_path)) {
      val <- get_val(symbol)
      write_csv(val, val_path)
      tsprint(str_glue("{symbol}: Created val file."), log_path)
    } else {
      val <- read_csv(val_path, show_col_types = FALSE)
      last_date <- max(pull(val, val_change_date))
      val_change_date <- pull(spot_symbol, val_change_date)
      if (
        isTRUE(last_date < val_change_date & val_change_date <= last_td)
      ) {
        val <- get_val(symbol)
        write_csv(val, val_path)
        tsprint(str_glue("{symbol}: Replaced val file."), log_path)
      }
    },
    silent = TRUE
  )
  if (inherits(try_error, "try-error")) {
    tsprint(str_glue("{symbol}: Error retrieving val."), log_path)
  }
}

tsprint(str_glue("Updated {length(symbols)} stocks."))
