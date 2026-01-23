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
  "H30184", "931743", "000685", "980017", "932066", "932139", # 半导体
  "H30590", "399283", "932438", "980022", "980071", # 机器人
  "931160", "931079", "931271", "932065", # 通信
  "930713", "931071", "932359", "399284", "931441", "932456", "980107",
  "980112", # 人工智能
  "931994", "932611", # 电网设备
  "399265", "931152", "931440", "931639", "932545", "980086", # 创新药
  "980032", "931746", "931747", "932090", "932246" # 储能
)

index_comp <- foreach(
  index = index_list,
  .combine = "c"
) %do% {
  print(index)
  list(loop_function("combine_index_comp", index))
} %>%
  rbindlist() %>%
  filter(str_detect(symbol, "^(0|3|6)"))
write_csv(index_comp, index_comp_path)
tsprint(str_glue("Retrieved {nrow(index_comp)} index components."))
