for i in $(seq 5 5 100); do
  Rscript opt_param.r \
    "var_name <- \"t_cci\"" \
    "var_seq <- seq(5, 100, 5)" \
    "t_adx <- $i" \
    "t_cci <- 10" \
    "x_h <- 0.53" \
    "r_h <- 0.09" \
    "r_l <- -0.5" \
    "t_max <- 105"
done
