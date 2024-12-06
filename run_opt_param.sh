for t_cci in $(seq 50 5 65); do
  for x_thr in $(seq 0.45 0.05 0.65); do
    Rscript opt_param.r \
      "var_name1 <- 't_adx'" \
      "var_seq1 <- seq(5, 40, 5)" \
      "var_name2 <- 't_max'" \
      "var_seq2 <- c(110, 115)" \
      "t_cci <- $t_cci" \
      "x_thr <- $x_thr" \
      "r_max <- 0.09" \
      "r_min <- -0.5"
  done
done
