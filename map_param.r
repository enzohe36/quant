source("preset.r", encoding = "UTF-8")
library(gplots)

v1_name <- "r_h"
v2_name <- "r_l"

# ------------------------------------------------------------------------------

# Define input
param <- read.csv("param.csv")

# Define parameters
t_adx <- 20
t_cci <- 25
x_h <- 0.5
r_h <- 0.09
r_l <- -0.5
t_max <- 105

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

heatmap.2(
  m_mean,
  Rowv = FALSE, Colv = FALSE,
  dendrogram = "none",
  col = bluered(100),
  na.color = "grey",
  trace = "none",
  main = "Mean",
  xlab = v1_name, ylab = v2_name
)

heatmap.2(
  m_icv,
  Rowv = FALSE, Colv = FALSE,
  dendrogram = "none",
  col = bluered(100),
  na.color = "grey",
  trace = "none",
  main = "1 / CV",
  xlab = v1_name, ylab = v2_name
)
