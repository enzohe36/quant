rm(list = ls())

source("lib/preset.r", encoding = "UTF-8")
source("lib/misc.r", encoding = "UTF-8")
source("lib/fn_load_data.r", encoding = "UTF-8")
source("lib/fn_backtest.r", encoding = "UTF-8")
source("lib/fn_sample_apy.r", encoding = "UTF-8")

var_dict_path <- "assets/var_dict.csv"
param_path <- "assets/param_20241217.csv"

# ------------------------------------------------------------------------------

start_date <- ymd(20210628)
end_date <- ymd(20240628)
n_portfolio <- 30
t <- 1
n_sample <- 1000

data_list <- load_data("^(00|60)", "hfq", start_date, end_date)

var_dict <- read.csv(var_dict_path)

if (!file.exists(param_path)) {
  var_list <- lapply(
    var_dict$name,
    function(str) {
      var_dict_i <- var_dict[var_dict$name == str, ]
      sample(seq(var_dict_i$min, var_dict_i$max, var_dict_i$prec), 10)
    }
  ) %>%
    `names<-`(var_dict$name)
  param <- do.call(Map, c(f = c, var_list)) %>%
    lapply(
      function(v) {
        for (str in names(v)) assign(str, v[str])
        apy <- backtest(
          data_list,
          t_adx, t_cci, t_xad, t_xbd, t_sgd, xa_thr, xb_thr, t_max, r_max, r_min
        ) %>%
          sample_apy(n_portfolio, t, n_sample)
        return(c(as.list(v), apy_mean = mean(apy$apy), apy_sd = sd(apy$apy)))
      }
    ) %>%
    rbindlist() %>%
    as.data.frame()
  write.csv(param, param_path, quote = FALSE, row.names = FALSE)
} else {
  param <- read.csv(param_path)
}

round_assign <- function(str, val, ...) {
  var_dict_i <- var_dict[var_dict$name == str, ]
  val <- round(val / var_dict_i$prec) * var_dict_i$prec
  assign(str, val, ...)
}

get_score <- function(
  apy_mean = numeric(0), apy_sd = numeric(0), t_max = numeric(0)
) {
  apy_mean <- c(param$apy_mean, apy_mean)
  apy_sd <- c(param$apy_sd, apy_sd)
  t_max <- c(param$t_max, t_max)
  t_max_min <- var_dict[var_dict$name == "t_max", "min"]
  t_max_max <- var_dict[var_dict$name == "t_max", "max"]
  score <- normalize(apy_mean) +
    normalize(apy_mean / apy_sd) -
    (mean(apy_mean) - min(apy_mean)) / (max(apy_mean) - min(apy_mean)) -
    (t_max - t_max_min) / (t_max_max - t_max_min)
  return(score)
}

opt_param <- function(var) {
  round_assign(var_name, var, envir = .GlobalEnv)
  apy <- backtest(
    data_list,
    t_adx, t_cci, t_xad, t_xbd, t_sgd, xa_thr, xb_thr, t_max, r_max, r_min
  ) %>%
    sample_apy(n_portfolio, t, n_sample)
  assign("apy_mean", mean(apy$apy), envir = .GlobalEnv)
  assign("apy_sd", sd(apy$apy), envir = .GlobalEnv)
  score <- get_score(apy_mean, apy_sd, t_max) %>% last()
  writeLines(glue("{var_name} = {var}, score = {score}"))
  return(score)
}

for (var_name in sample(var_dict$name, 3)) {
  var_dict_i <- var_dict[var_dict$name == var_name, ]

  q1 <- seq(var_dict_i$min, var_dict_i$max, var_dict_i$prec) %>%
    abs() %>%
    sort() %>%
    quantile(0.25) %>%
    `/`(var_dict_i$prec) %>%
    round() %>%
    `*`(var_dict_i$prec)

  # Assign initial values
  score <- get_score()
  param_best <- param[score == max(score), ] %>%
    .[sample(nrow(.), 1), ]
  param_init <- (tail(param, 1) + param_best) / 2
  for (str in var_dict$name) {
    round_assign(str, param_init[, str], envir = .GlobalEnv)
  }

  # Update selected parameter
  runtime <- system.time(
    opt <- optimize(
      opt_param,
      c(
        max(var_dict_i$min, get(var_name) - q1),
        min(var_dict_i$max, get(var_name) + q1)
      ),
      maximum = TRUE,
      tol = var_dict_i$prec
    )
  )
  tsprint(glue("Total time: {runtime[3]} s."))

  param <- rbind(
    param,
    lapply(colnames(param), function(str) get(str)) %>%
      `names<-`(colnames(param))
  )
  write.csv(param, param_path, quote = FALSE, row.names = FALSE)
}
