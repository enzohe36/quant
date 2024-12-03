source("lib/preset.r", encoding = "UTF-8")

source("lib/fn_get_data.r", encoding = "UTF-8")
source("lib/fn_load_data.r", encoding = "UTF-8")
source("lib/fn_update.r", encoding = "UTF-8")
source("lib/fn_query.r", encoding = "UTF-8")
source("lib/fn_buy.r", encoding = "UTF-8")
source("lib/fn_sell.r", encoding = "UTF-8")

# python -m aktools

# ------------------------------------------------------------------------------

# get_data("^(00|60)", "qfq")

# out0 <- load_data("^(00|60)", "qfq")

out0 <- update()

# query(plot = FALSE)

# buy(000001, 12.345, 20240527)

# sell(000001)
