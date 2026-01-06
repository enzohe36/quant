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

indices <- get_indices()
write_csv(indices, indices_path)
tsprint(str_glue("Retrieved {nrow(indices)} indices."))

index_names <- c(
  # 机器人
  "932438", "H30590",
  # 新材料
  "H30597",
  # 半导体
  "931743", "932066", "932448", "932139", "H30184",
  # 芯片
  "932040", "H30007",
  # 航天/空天
  "932419", "H30213", "932116", "932143", "930875",
  # 卫星
  "931585", "931594",
  # 储能
  "931746", "931747", "932246", "932090",
  # 电池
  "931555", "931664", "931719",
  # 稀土
  "930598",
  # 有色
  "000811", "930708", "931892", "H11059", "000819", "932112",
  # 光伏
  "931151", "931798", "931528",
  # 计算机
  "930651", "932067", "H30182",
  # 电子
  "930652", "931461", "931483", "931494", "H30190", "399811", "932138",
  "H30183",
  # 通信
  "930852", "931079", "931144", "931160", "931271", "931723", "932065",
  "932141", "932145"
)

index_comp <- list()
for (index_name in index_names) {
  index_comp <- c(index_comp, list(loop_function("get_index_comp", index_name)))
}
index_comp <- rbindlist(index_comp)
write_csv(index_comp, index_comp_path)
tsprint(str_glue("Retrieved components of {length(index_names)} indices."))
