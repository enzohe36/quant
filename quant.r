source("preset.r", encoding = "UTF-8")
source("get_history.r", encoding = "UTF-8")
source("load_history.r", encoding = "UTF-8")
source("backtest.r", encoding = "UTF-8")
source("update.r", encoding = "UTF-8")
source("query.r", encoding = "UTF-8")

# python -m aktools

# get_history("^(00|60)", "qfq")

# out0 <- load_history("^(00|60)", "hfq", date(now() - years(5)), date(now()))

# trade_list <- backtest()

# out0 <- update()

# query(formatC(read.csv("portfolio.csv")[, 2], width = 6, format = "d", flag = "0"))
