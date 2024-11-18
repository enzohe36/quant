source("preset.r", encoding = "UTF-8")
source("get_history.r", encoding = "UTF-8")
source("load_history.r", encoding = "UTF-8")
source("update.r", encoding = "UTF-8")
source("query.r", encoding = "UTF-8")
source("backtest.r", encoding = "UTF-8")

# python -m aktools

# get_history("^(00|60)", "qfq", 5)

# load <- load_history("hfq") # symbol_list, data_list

# load <- update() # symbol_list, data_list, latest

# query(formatC(read.csv("portfolio.csv")[, 2], width = 6, format = "d", flag = "0"))

# trade_list <- backtest()
