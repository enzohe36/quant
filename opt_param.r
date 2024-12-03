source("lib/preset.r", encoding = "UTF-8")

source("lib/fn_load_data.r", encoding = "UTF-8")
source("lib/fn_backtest.r", encoding = "UTF-8")

param_path <- "assets/param.csv"

# ------------------------------------------------------------------------------

arg_list <- commandArgs(trailingOnly = TRUE)
for (i in seq_along(arg_list)) {
  eval(parse(text = arg_list[i]))
}

out0 <- load_data("^(00|60)", "hfq", 20191129, 20241129)

for (var in var_seq) {
  assign(var_name, var)
  trade <- backtest(t_adx, t_cci, x_h, r_h, r_l, t_max, descriptive = FALSE)
  if (!file.exists(param_path)) {
    write.csv(
      data.frame(
        t_adx, t_cci, x_h, r_h, r_l, t_max,
        mean = mean(trade$r), sd = sd(trade$r)
      ),
      param_path,
      quote = FALSE,
      row.names = FALSE
    )
  } else {
    write.table(
      data.frame(
        t_adx, t_cci, x_h, r_h, r_l, t_max,
        mean = mean(trade$r), sd = sd(trade$r)
      ),
      param_path, append = TRUE,
      quote = FALSE,
      sep = ",",
      row.names = FALSE, col.names = FALSE
    )
  }
  gc()
}
