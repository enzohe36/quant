source("preset.r", encoding = "UTF-8")

source("backtest_min.r", encoding = "UTF-8")
source("backtest.r", encoding = "UTF-8")
source("buysell.r", encoding = "UTF-8")
source("get_history.r", encoding = "UTF-8")
source("load_history.r", encoding = "UTF-8")
source("query.r", encoding = "UTF-8")
source("update.r", encoding = "UTF-8")

# python -m aktools

# ------------------------------------------------------------------------------

# get_history("^(00|60)", "qfq")

# out0 <- load_history("^(00|60)", "qfq", date(now() - years(1)), date(now()))

# out0 <- update()

# query()

# buy(000010, 2.68, 20241127)

# sell(000001)

# ------------------------------------------------------------------------------

# get_history("^(00|60)", "hfq")

# out0 <- load_history("^(00|60)", "hfq", date(now() - years(5)), date(now()))

# trade_list <- backtest()
