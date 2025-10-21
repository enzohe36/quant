options(warn = -1)

Sys.setlocale(locale = "Chinese")
Sys.setenv(TZ = "Asia/Shanghai")

set.seed(42)

scripts_dir <- "scripts/"

load_pkgs <- function(scripts) {
  scripts <- paste0(scripts_dir, scripts)
  out <- sapply(
    scripts,
    function(script) {
      lines <- readLines(script, encoding = "UTF-8")
      gsub(
        "^.*library\\(([^)]+)\\)$",
        "\\1",
        lines[grepl("^.*library\\(([^)]+)\\)$", lines)]
      )
    }
  ) |>
    unlist() |>
    unique() |>
    setdiff("tidyverse") |>
    sapply(library, character.only = TRUE)
}

source_scripts <- function(scripts) {
  scripts <- paste0(scripts_dir, scripts)
  out <- sapply(scripts, source, encoding = "UTF-8")
}
