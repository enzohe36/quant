# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools

# PRESET =======================================================================

library(RCurl)
library(jsonlite)
library(foreach)
library(doFuture)
library(data.table)
library(tidyverse)

source("scripts/data_retrievers.r")
source("scripts/misc.r")

resources_dir <- "resources/"
indices_path <- paste0(resources_dir, "indices.csv")
index_comp_path <- paste0(resources_dir, "index_comp.csv")

# MAIN SCRIPT ==================================================================

dir.create(resources_dir)

indices <- combine_indices()
write_csv(indices, indices_path)
tsprint(str_glue("Retrieved {nrow(indices)} indices."))

index_list <- c(
  "930713", # CS人工智
  "000685", # 科创芯片
  "931743", # 半导体材料设备
  "931079", # 5G通信
  "980022", # 机器人产业
  "980032", # 新能电池
  "980018", # 卫星通信
  "930601", # 中证软件
  "931151", # 光伏产业
  "H11059", # 工业有色
  "930598" # 稀土产业
)

index_comp <- foreach(
  index = index_list,
  .combine = "c"
) %do% {
  list(loop_function("get_index_comp", index))
} %>%
  rbindlist() %>%
  filter(str_detect(symbol, "^(0|3|6)"))
write_csv(index_comp, index_comp_path)
tsprint(str_glue("Retrieved {nrow(index_comp)} index components."))
