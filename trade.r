# python -m aktools

rm(list = ls())

source("lib/preset.r", encoding = "UTF-8")
source("lib/misc.r", encoding = "UTF-8")
source("lib/fn_get_data.r", encoding = "UTF-8")
source("lib/fn_load_data.r", encoding = "UTF-8")
source("lib/fn_update.r", encoding = "UTF-8")
source("lib/fn_query.r", encoding = "UTF-8")
source("lib/fn_buy.r", encoding = "UTF-8")
source("lib/fn_sell.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

# get_data("^(00|60)", "qfq")

data_list <- load_data("^(00|60)", "qfq")

out <- update()
data_list <- out[["data_list"]]
latest <- out[["latest"]]

# query(000001, 20240527)

# buy(000001, 12.345, 20240527)

# sell(000001)
