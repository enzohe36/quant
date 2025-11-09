options(radian.auto_match = FALSE)
options(radian.auto_indentation = FALSE)
options(radian.complete_while_typing = FALSE)
options(warn = -1)
options(repos = c(CRAN = "https://cloud.r-project.org"))

Sys.setlocale(locale = "Chinese")
Sys.setenv(TZ = "Asia/Shanghai")

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
  out <- sapply(scripts, source, encoding = "UTF-8")
}
