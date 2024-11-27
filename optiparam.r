args = commandArgs(trailingOnly=TRUE)

source("preset.r", encoding = "UTF-8")

source("backtest_min.r", encoding = "UTF-8")
source("load_history.r", encoding = "UTF-8")

out0 <- load_history("^(00|60)", "hfq", ymd("2019-05-26"), ymd("2024-05-26"))

optiparam <- function(x1, x2_from, x2_to, x2_by) {
  # Define parameters
  t_adx <- 20
  t_cci <- 25
  x_h <- 0.53
  r_h <- x1
  # r_l <- -0.5
  t_max <- 104

  for (r_l in seq(x2_from, x2_to, x2_by)) {
    backtest_min(t_adx, t_cci, x_h, r_h, r_l, t_max)
    gc()
  }
}

eval(parse(text = args[1]))

# v1 <- "t_adx"
# v2 <- "t_cci"
# get_value <- function(i, j, funct) {
#   return(
#     df[
#       df$t_adx == i &
#         df$t_cci == j &
#         df$x_h == 0.53 &
#         df$r_h == 0.1 &
#         df$r_l == -0.5 &
#         df$t_max == 104,
#       funct
#     ]
#   )
# }
#
# df <- read.csv("param.csv")
# df <- df %>% arrange(across(everything()))
#
# m_mean <- matrix(
#   nrow = length(unique(df[, v2])),
#   ncol = length(unique(df[, v1])),
#   dimnames = list(unique(df[, v2]), unique(df[, v1]))
# )
# m_icv <- m_mean
# for (i in unique(df[, v1])) {
#   for (j in unique(df[, v2])) {
#     m_mean[as.character(j), as.character(i)] <- ifelse(
#       length(get_value(i, j, "mean") != 0),
#       get_value(i, j, "mean"),
#       NA
#     )
#     m_icv[as.character(j), as.character(i)] <- ifelse(
#       length(get_value(i, j, "mean") != 0),
#       get_value(i, j, "mean") / get_value(i, j, "sd"),
#       NA
#     )
#   }
# }
#
# heatmap.2(
#   m_mean,
#   Rowv = FALSE, Colv = FALSE, dendrogram = "none",
#   col = bluered(100),
#   na.color = "grey",
#   trace = "none",
#   main = "Mean", xlab = "V1", ylab = "V2"
# )
# heatmap.2(
#   m_icv,
#   Rowv = FALSE, Colv = FALSE, dendrogram = "none",
#   col = bluered(100),
#   na.color = "grey",
#   trace = "none",
#   main = "1 / CV", xlab = "V1", ylab = "V2"
# )
#