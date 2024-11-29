for i in $(seq 0.5 0.01 1); do
  Rscript opt_param.r \
    "t_adx <- 20" \
    "t_cci <- 25" \
    "x_h <- $i" \
    "r_h <- 0.09" \
    "r_l <- -0.5" \
    "t_max <- 105" \
    "var_name <- \"t_max\"" \
    "var_seq <- seq(60, 120, 5)"
done
