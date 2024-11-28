source("preset.r", encoding = "UTF-8")

source("backtest_min.r", encoding = "UTF-8")
source("load_data.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

arg_list <- commandArgs(trailingOnly = TRUE)
for (i in 1:length(arg_list)) {
  eval(parse(text = arg_list[i]))
}

out0 <- load_data("^(00|60)", "hfq", ymd("2019-05-26"), ymd("2024-05-26"))

for (var in var_seq) {
  assign(var_name, var)
  backtest_min(t_adx, t_cci, x_h, r_h, r_l, t_max)
  gc()
}
