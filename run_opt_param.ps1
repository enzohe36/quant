for ($t_cci = 50; $t_cci -le 65; $t_cci = $t_cci + 5) {
  for ($x_thr = 0.45; $x_thr -le 0.65; $x_thr = $x_thr + 0.05) {
    Rscript opt_param.r `
      "var_name1 <- 't_adx'" `
      "var_seq1 <- seq(5, 40, 5)" `
      "var_name2 <- 't_max'" `
      "var_seq2 <- c(120, 125)" `
      "t_cci <- $t_cci" `
      "x_thr <- $x_thr" `
      "r_max <- 0.09" `
      "r_min <- -0.5"
  }
}
