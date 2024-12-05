for ($x_thr = 0.4; $x_thr -le 0.6; $x_thr = $x_thr + 0.05) {
  for ($t_max = 100; $t_max -le 120; $t_max = $t_max + 5) {
    Rscript opt_param.r `
      "var_name1 <- 't_adx'" `
      "var_seq1 <- seq(5, 45, 5)" `
      "var_name2 <- 't_cci'" `
      "var_seq2 <- seq(45, 50, 5)" `
      "x_thr <- $x_thr" `
      "t_max <- $t_max" `
      "r_max <- 0.1" `
      "r_min <- -0.5"
  }
}
