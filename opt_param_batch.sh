for i in $(seq 0.4 0.05 0.6); do
  for j in $(seq 100 5 120); do
    Rscript opt_param.r \
      "var_name1 <- \"t_adx\"" \
      "var_seq1 <- seq(5, 25, 5)" \
      "var_name2 <- \"t_cci\"" \
      "var_seq2 <- seq(10, 30, 5)" \
      "t_adx <- 20" \
      "t_cci <- 25" \
      "x_thr <- $i" \
      "t_max <- $j" \
      "r_max <- 0.1" \
      "r_min <- -0.5"
  done
done
