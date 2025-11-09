# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools

# =============================== PRESET ==================================

source_scripts(
  scripts = c("misc", "data_retrievers"),
  packages = c("foreach", "doFuture", "data.table", "tidyverse")
)

data_dir <- "data/"
spot_combined_path <- paste0(data_dir, "spot_combined.csv")

resource_dir <- "resources/"
name_repl_path <- paste0(resource_dir, "name_repl.csv")

report_dir <- "reports/"
prompt_path <- paste0(report_dir, "prompt.txt")

industries <- c("算力")

# ============================= MAIN SCRIPT ===============================

plan(multisession, workers = availableCores() - 1)

name_repl <- read_csv(name_repl_path, show_col_types = FALSE)

analysis <- foreach(
  industry = industries,
  .combine = "multiout",
  .multicombine = TRUE,
  .init = list(list(), list())
) %dofuture% {
  report_path <- paste0(report_dir, "report_", industry, ".txt")

  report <- read_file(report_path) %>%
    str_replace_all("\n\n", "\n")
  report <- name_repl %>%
    {setNames(.$replacement, .$search)} %>%
    str_replace_all(string = report, pattern = .)

  spot_combined_trim <- spot_combined %>%
    mutate(count = sapply(name, function(x) str_count(report, x))) %>%
    arrange(desc(count)) %>%
    filter(count >= 1)
  return(list(spot_combined_trim, report))
}

spot_combined_trim <- rbindlist(analysis[[1]]) %>%
  group_by(symbol) %>%
  summarise(
    symbol = first(symbol),
    name = first(name),
    count = max(count),
    .groups = "drop"
  )

names <- spot_combined_trim %>%
  pull(name) %>%
  paste(collapse = "，") %>%
  str_replace_all("机器人", "") %>%
  str_replace_all("驱动力", "") %>%
  str_replace_all("比亚迪", "")

report <- analysis[[2]] %>%
  map_chr(~ .x[1]) %>%
  paste(collapse = "\n")

c(
  str_glue("根据给出的研报节选，为股票池中的所有股票按主营业务分类。要求如下："),
  "- 确保股票池中的所有股票仅分类在最为相关的细分领域下；",
  "- 确保所选股票范围严格限定在股票池内；",
  "- 在Artifacts以Markdown表格的形式输出所有符合条件的股票，包含股票名称、非常简短的推荐理由，优先列出重点推荐股票。",
  "- 不要使用emoji表情符号。",
  "\n股票池：",
  names,
  "\n研报节选：",
  report
) %>%
  writeLines(prompt_path)

plan(sequential)
