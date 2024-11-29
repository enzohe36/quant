source("preset.r", encoding = "UTF-8")

source("backtest_min.r", encoding = "UTF-8")
source("backtest.r", encoding = "UTF-8")
source("buysell.r", encoding = "UTF-8")
source("get_data.r", encoding = "UTF-8")
source("load_data.r", encoding = "UTF-8")
source("query.r", encoding = "UTF-8")
source("update.r", encoding = "UTF-8")

# python -m aktools

# ------------------------------------------------------------------------------

# get_data("^(00|60)", "qfq")

# out0 <- load_data("^(00|60)", "hfq", date(now() - years(5)), date(now()))

# writeLines(out0[[1]], "symbol_list.txt")

# file.copy("symbol_list.bak", "symbol_list.txt", overwrite = TRUE)

# out0 <- load_data("^(00|60)", "qfq", date(now() - years(1)), date(now()))

# out0 <- update()

# query(plot = FALSE)

# buy(000001, 12.345, 20241127)

# sell(000001)

# ------------------------------------------------------------------------------

# get_data("^(00|60)", "hfq")

# out0 <- load_data("^(00|60)", "hfq", date(now() - years(5)), date(now()))

# trade_list <- backtest()
