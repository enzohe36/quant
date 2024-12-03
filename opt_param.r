source("lib/preset.r", encoding = "UTF-8")

source("lib/fn_load_data.r", encoding = "UTF-8")
source("lib/fn_backtest.r", encoding = "UTF-8")

param_path <- "assets/param.csv"

# ------------------------------------------------------------------------------

arg_list <- commandArgs(trailingOnly = TRUE)
for (i in seq_along(arg_list)) {
  eval(parse(text = arg_list[i]))
}

out0 <- load_data("^(00|60)", "hfq", 20190628, 20240628)

for (var1 in var_seq1) for (var2 in var_seq2) {
  assign(var_name1, var1)
  assign(var_name2, var2)

  trade <- backtest(t_adx, t_cci, x_thr, t_max, r_max, r_min, descr = FALSE)
  if (!file.exists(param_path)) {
    write.csv(
      data.frame(
        t_adx, t_cci, x_thr, t_max, r_max, r_min,
        mean = mean(trade$r), sd = sd(trade$r)
      ),
      param_path,
      quote = FALSE,
      row.names = FALSE
    )
  } else {
    write.table(
      data.frame(
        t_adx, t_cci, x_thr, t_max, r_max, r_min,
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
