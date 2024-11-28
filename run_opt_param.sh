for i in $(seq 0.1 0.01 0.1); do
  Rscript opt_param.r \
    "var_name <- as.character(\"r_l\")" \
    "var_seq <- seq(-0.5, -0.05, 0.05)" \
    "t_adx <- 20" \
    "t_cci <- 25" \
    "x_h <- 0.53" \
    "r_h <- $i" \
    "r_l <- -0.5" \
    "t_max <- 104"
done
