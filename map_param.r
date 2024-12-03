source("lib/preset.r", encoding = "UTF-8")

library(gplots)

param_path <- "assets/param_20241127.csv"

# ------------------------------------------------------------------------------

# Define parameters
t_adx <- 20 # 15; 5 to 25
t_cci <- 25 # 30; 10 to 30
x_thr <- 0.5 # 0.53; 0.4 to 0.6
t_max <- 105 # 105; 100 to 120
r_max <- 0.09 # 0.09
r_min <- -0.5 # -0.5

v1_name <- "r_max"
v2_name <- "r_min"

param <- read.csv(param_path)
v1 <- sort(unique(param[, v1_name]))
v2 <- sort(unique(param[, v2_name]))

m_mean <- matrix(
  nrow = length(v2),
  ncol = length(v1),
  dimnames = list(v2, v1)
)
m_icv <- m_mean

get_row <- function(t_adx, t_cci, x_thr, t_max, r_max, r_min) {
  param[
    param$t_adx == t_adx &
      param$t_cci == t_cci &
      param$x_thr == x_thr &
      param$t_max == t_max &
      param$r_max == r_max &
      param$r_min == r_min,
  ]
}

for (i in v1) for (j in v2) {
  assign(v1_name, i)
  assign(v2_name, j)

  get_value <- function(col_name) {
    get_row(t_adx, t_cci, x_thr, t_max, r_max, r_min)[, col_name]
  }

  m_mean[as.character(j), as.character(i)] <- ifelse(
    length(get_value("mean")) != 0, get_value("mean"), NA
  )
  m_icv[as.character(j), as.character(i)] <- ifelse(
    length(get_value("mean")) != 0, get_value("mean") / get_value("sd"), NA
  )
}

logit <- function(v) log(v / (1 - v))
m <- logit(normalize(m_mean)) + logit(normalize(m_icv))
m[is.infinite(m)] <- NA

heatmap.2(
  m,
  Rowv = FALSE, Colv = FALSE,
  dendrogram = "none",
  col = bluered(10),
  na.color = "grey",
  trace = "none",
  main = "logit(norm(mean)) + logit(norm(1 / CV))",
  xlab = v1_name, ylab = v2_name
)
