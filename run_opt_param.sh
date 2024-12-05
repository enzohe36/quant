for x_thr in $(seq 0.4 0.05 0.6); do
  for t_max in $(seq 100 5 120); do
    Rscript opt_param.r \
      "var_name1 <- 't_adx'" \
      "var_seq1 <- seq(5, 45, 5)" \
      "var_name2 <- 't_cci'" \
      "var_seq2 <- seq(45, 50, 5)" \
      "x_thr <- $x_thr" \
      "t_max <- $t_max" \
      "r_max <- 0.1" \
      "r_min <- -0.5"
  done
done
