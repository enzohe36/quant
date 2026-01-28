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
}

options(warn = -1)
options(repos = c(CRAN = "https://cloud.r-project.org"))
options(encoding = "UTF-8")

Sys.setenv(TZ = "Asia/Shanghai")
