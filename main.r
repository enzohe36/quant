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

# out0 <- load_data("^(00|60)", "qfq", date(now() - years(1)), date(now()))

# out0 <- update()

# query(plot = FALSE)

# buy(000001, 12.345, 20241127)

# sell(000001)

# ------------------------------------------------------------------------------

# get_data("^(00|60)", "hfq")

# out0 <- load_data("^(00|60)", "hfq", date(now() - years(5)), date(now()))

# trade <- backtest(20, 10, 0.53, 0.09, -0.5, 105)

# print(sample_apy(30))
