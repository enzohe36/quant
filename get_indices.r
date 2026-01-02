# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools

# PRESET =======================================================================

library(foreach)
library(doFuture)
library(RCurl)
library(jsonlite)
library(data.table)
library(tidyverse)

source("scripts/misc.r")
source("scripts/data_retrievers.r")

resources_dir <- "resources/"
indices_path <- paste0(resources_dir, "indices.csv")
index_comp_path <- paste0(resources_dir, "index_comp.csv")

index <- "000985"

# MAIN SCRIPT ==================================================================

dir.create(resources_dir)

indices <- combine_indices()
write_csv(indices, indices_path)
tsprint(str_glue("Updated {nrow(indices)} indices."))

index_comp <- get_index_comp(index)
write_csv(index_comp, index_comp_path)
tsprint(str_glue("Retrieved components for index {index}."))
