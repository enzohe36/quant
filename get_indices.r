# conda activate myenv; pip install aktools --upgrade -i https://pypi.org/simple; pip install akshare --upgrade -i https://pypi.org/simple; python -m aktools

# =============================== PRESET ==================================

source_scripts(
  scripts = c("misc", "data_retrievers"),
  packages = c()
)

data_dir <- "data/"
indices_path <- paste0(data_dir, "indices.csv")

# ============================= MAIN SCRIPT ===============================

dir.create(data_dir)

indices <- combine_indices()
write_csv(indices, indices_path)
tsprint(str_glue("Retrieved {nrow(indices)} indices."))
