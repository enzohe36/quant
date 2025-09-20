# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools
# conda activate myenv; Rscript get_data.r; Rscript get_data.r

rm(list = ls())

gc()

library(doFuture)
library(foreach)
library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"
index_comp_path <- paste0(data_dir, "index_comp.csv")

report_dir <- "reports/"
name_repl_path <- paste0(report_dir, "name_repl.csv")
prompt_path <- paste0(report_dir, "prompt.txt")

industries <- c("算力")
# industries <- c("通用设备", "专用设备", "电池")

# index_comp <- get_index_comp("000985") %>%
#   list(get_fundamentals(as_tradedate(now() - hours(16)))) %>%
#   reduce(left_join, by = "symbol")
# write.csv(index_comp, index_comp_path, quote = FALSE, row.names = FALSE)
# tsprint(glue("Found {nrow(index_comp)} stocks."))

index_comp <- read_csv(index_comp_path, show_col_types = FALSE) %>%
  select(symbol, name)
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

  index_comp_trim <- index_comp %>%
    mutate(count = sapply(name, function(x) str_count(report, x))) %>%
    arrange(desc(count)) %>%
    filter(count >= 1)
  return(list(index_comp_trim, report))
}

index_comp_trim <- rbindlist(analysis[[1]]) %>%
  group_by(symbol) %>%
  summarise(
    symbol = first(symbol),
    name = first(name),
    count = max(count),
    .groups = "drop"
  )

names <- index_comp_trim %>%
  pull(name) %>%
  paste(collapse = "，") %>%
  str_replace_all("机器人", "") %>%
  str_replace_all("驱动力", "") %>%
  str_replace_all("比亚迪", "")

report <- analysis[[2]] %>%
  map_chr(~ .x[1]) %>%
  paste(collapse = "\n")

c(
  glue("根据给出的研报节选，为股票池中的所有股票按主营业务分类。要求如下："),
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
