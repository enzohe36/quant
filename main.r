source("preset.r", encoding = "UTF-8")

source("get_data.r", encoding = "UTF-8")
source("load_data.r", encoding = "UTF-8")
source("update.r", encoding = "UTF-8")
source("query.r", encoding = "UTF-8")
source("buysell.r", encoding = "UTF-8")
source("backtest.r", encoding = "UTF-8")
source("sample_apy.r", encoding = "UTF-8")

# python -m aktools

# ------------------------------------------------------------------------------

# get_data("^(00|60)", "qfq")

# out0 <- load_data("^(00|60)", "qfq")

# out0 <- update()

# query(plot = FALSE)

# buy(000001, 12.345, 20240527)

# sell()

# ------------------------------------------------------------------------------

# get_data("^(00|60)", "hfq")

# out0 <- load_data("^(00|60)", "hfq", today() - years(10), today())

# out0[["trade"]] <- backtest(20, 10, 0.53, 0.09, -0.5, 105)

# out1 <- sample_apy(30, 1, 1000)
