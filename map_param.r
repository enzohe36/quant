source("lib/preset.r", encoding = "UTF-8")

library(gplots)

param_path <- "assets/param.csv"

# ------------------------------------------------------------------------------

# Define parameters
t_adx <- 20 # 20; seq 5 5 30
t_cci <- 25 # 10; seq 5 5 40
x_h <- 0.5 # 0.53; seq 0.2 0.05 0.6
r_h <- 0.1 # 0.09; seq 0.05 0.05 0.3
r_l <- -0.5 # -0.5; seq -0.7 0.05 -0.3
t_max <- 105 # 105; seq 100 5 120

v1_name <- "t_adx"
v2_name <- "t_cci"

param <- read.csv(param_path)
v1 <- sort(unique(param[, v1_name]))
v2 <- sort(unique(param[, v2_name]))

m_mean <- matrix(
  nrow = length(v2),
  ncol = length(v1),
  dimnames = list(v2, v1)
)
m_icv <- m_mean

get_row <- function(t_adx, t_cci, x_h, r_h, r_l, t_max) {
  param[
    param$t_adx == t_adx &
      param$t_cci == t_cci &
      param$x_h == x_h &
      param$r_h == r_h &
      param$r_l == r_l &
      param$t_max == t_max,
  ]
}

for (i in v1) for (j in v2) {
  assign(v1_name, i)
  assign(v2_name, j)

  get_value <- function(col_name) {
    get_row(t_adx, t_cci, x_h, r_h, r_l, t_max)[, col_name]
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
