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
name_replacement_path <- paste0(report_dir, "name_replacement.csv")
prompt_path <- paste0(report_dir, "prompt.txt")

industries <- c("半导体", "电子元件", "通信设备")
# industries <- c("通用设备", "电池")

concepts <- paste0(c("算力芯片", "AI SoC", "PCB", "光模块"), collapse = "，")
# concepts <- paste0(c("机器人执行器", "固态电池", "光伏设备"), collapse = "，")

# index_comp <- get_index_comp("000985") %>%
#   list(get_fundamentals(as_tradedate(now() - hours(16)))) %>%
#   reduce(left_join, by = "symbol")
# write.csv(index_comp, index_comp_path, quote = FALSE, row.names = FALSE)
# tsprint(glue("Found {nrow(index_comp)} stocks."))

index_comp <- read_csv(index_comp_path, show_col_types = FALSE) %>%
  select(symbol, name)
name_replacement <- read_csv(name_replacement_path, show_col_types = FALSE)

analysis <- foreach(
  industry = industries,
  .combine = "multiout",
  .multicombine = TRUE,
  .init = list(list(), list())
) %dofuture% {
  report_path <- paste0(report_dir, "report_", industry, ".txt")

  report <- read_file(report_path) %>%
    str_replace_all("\n\n", "\n")
  report <- name_replacement %>%
    {setNames(.$replacement, .$search)} %>%
    str_replace_all(string = report, pattern = .)

  index_comp_trim <- index_comp %>%
    mutate(count = sapply(name, function(x) str_count(report, x))) %>%
    arrange(desc(count)) %>%
    filter(count > 1)
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
  str_replace_all("机器人", "")

report <- analysis[[2]] %>%
  map_chr(~ .x[1]) %>%
  paste(collapse = "\n")

c(
  "根据以下科技行业研报，从股票池中为下列每个概念板块精选最具代表性的股票。请按以下逻辑思考：",
  "1. 为每个概念板块找出所有相关的股票，不要限制每个板块的股票数量；",
  "2. 检查是否有遗漏的股票，并补充进概念板块，不要限制每个板块的股票数量；",
  "3. 再次检查是否有遗漏的股票，并补充进概念板块，不要限制每个板块的股票数量；",
  "4. 确认所选股票范围严格限定在股票池内；",
  "5. 根据推荐频率和重要性，为每个板块中的所有股票从高到低进行排序；",
  "6. 输出所有符合条件的股票，包含股票名称、简短的推荐理由，并注明是否是重点推荐股票；",
  "7. 把输出在Artifacts中整理为Markdown表格。",
  "\n概念板块：",
  concepts,
  "\n股票池：",
  names,
  "\n科技行业研报：",
  report
) %>%
  writeLines(prompt_path)

plan(sequential)
