source("preset.r", encoding = "UTF-8")

source("load_data.r", encoding = "UTF-8")
source("backtest.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

arg_list <- commandArgs(trailingOnly = TRUE)
for (i in seq_along(arg_list)) {
  eval(parse(text = arg_list[i]))
}

out0 <- load_data("^(00|60)", "hfq", ymd(20190526), ymd(20240526))

for (var in var_seq) {
  assign(var_name, var)
  trade <- backtest(t_adx, t_cci, x_h, r_h, r_l, t_max, descriptive = FALSE)
  write(
    paste(
      t_adx, t_cci, x_h, r_h, r_l, t_max, mean(trade$r), sd(trade$r),
      sep = ","
    ),
    file = "param.csv",
    append = TRUE
  )
  gc()
}
