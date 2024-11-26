source("preset.r", encoding = "UTF-8")
source("get_history.r", encoding = "UTF-8")
source("load_history.r", encoding = "UTF-8")
source("backtest.r", encoding = "UTF-8")
source("update.r", encoding = "UTF-8")
source("query.r", encoding = "UTF-8")
source("buysell.r", encoding = "UTF-8")

# python -m aktools

# ------------------------------------------------------------------------------

# get_history("^(00|60)", "qfq")

# out0 <- load_history("^(00|60)", "qfq", date(now() - months(12)), date(now()))

# out0 <- update()

# query()

# buy(000001, 20, 20241024)

# sell(000001)

# ------------------------------------------------------------------------------

# get_history("^(00|60)", "hfq")

# out0 <- load_history("^(00|60)", "hfq", date(now() - months(66)), date(now() - months(6)))

# trade_list <- backtest()
