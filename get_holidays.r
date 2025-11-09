# =============================== PRESET ==================================

source_scripts(
  scripts = c("misc"),
  packages = c("foreach", "doFuture", "data.table", "tidyverse")
)

data_dir <- "data/"
holidays_path <- paste0(data_dir, "holidays.txt")

hist_dir <- paste0(data_dir, "hist/")

# ============================= MAIN SCRIPT ===============================

plan(multisession, workers = availableCores() - 1)

dir.create(data_dir)

new_holidays <- c(
  mdy("January 1, 2025"),
  seq(mdy("January 28, 2025"), mdy("February 4, 2025"), by = "1 day"),
  mdy("January 26, 2025"),
  mdy("February 8, 2025"),
  seq(mdy("April 4, 2025"), mdy("April 6, 2025"), by = "1 day"),
  seq(mdy("May 1, 2025"), mdy("May 5, 2025"), by = "1 day"),
  mdy("April 27, 2025"),
  seq(mdy("May 31, 2025"), mdy("June 2, 2025"), by = "1 day"),
  seq(mdy("October 1, 2025"), mdy("October 8, 2025"), by = "1 day"),
  mdy("September 28, 2025"),
  mdy("October 11, 2025")
) %>%
  .[!wday(., week_start = 1) %in% 6:7]

holidays <- as_date(c())

if (dir.exists(hist_dir)) {
  files <- list.files(hist_dir)

  if (length(files) > 0) {
    tradedays <- foreach(
      file = files,
      .combine = "c"
    ) %dofuture% {
      read_csv(paste0(hist_dir, file), show_col_types = FALSE) %>%
        pull(date)
    } %>%
      unique()

    holidays <- seq(min(tradedays), max(tradedays), by = "1 day") %>%
      .[!wday(., week_start = 1) %in% 6:7] %>%
      .[!.%in% tradedays]
  }
}

holidays <- sort(unique(c(holidays, new_holidays)))
writeLines(as.character(holidays), holidays_path)
tsprint(str_glue("Updated holidays from {min(holidays)} to {max(holidays)}."))

plan(sequential)
