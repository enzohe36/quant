# For Windows compatibility
if (.Platform$OS.type == "windows") {
  options(radian.auto_match = FALSE)
  options(radian.auto_indentation = FALSE)
  options(radian.complete_while_typing = FALSE)

  if (interactive() && Sys.getenv("TERM_PROGRAM") == "vscode") {
    if ("httpgd" %in% .packages(all.available = TRUE)) {
      options(vsc.rstudioapi = TRUE)
      options(vsc.use_httpgd = TRUE)
      options(vsc.plot = FALSE)
      options(device = function(...) httpgd::hgd(silent = FALSE))
    }
  }

  Sys.setlocale("LC_ALL", "chs")
}

options(warn = -1)
options(repos = c(CRAN = "https://cloud.r-project.org"))
options(encoding = "UTF-8")

Sys.setenv(TZ = "Asia/Shanghai")

# conflictRules("dplyr", exclude = "lag")

source_scripts <- function(scripts, packages) {
  scripts <- paste0("scripts/", scripts, ".r")
  packages <- scripts |>
    sapply(
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
    c(packages) |>
    unique()
  if ("tidyverse" %in% packages) {
    packages <- c(packages[packages != "tidyverse"], "tidyverse")
  }
  out <- sapply(packages, library, character.only = TRUE)
  out <- sapply(scripts, source)
}
