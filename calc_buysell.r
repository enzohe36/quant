# A1:=FORCAST(EMA(CLOSE,5),6);
# A2:=FORCAST(EMA(CLOSE,8),6);
# A3:=FORCAST(EMA(CLOSE,11),6);
# A4:=FORCAST(EMA(CLOSE,14),6);
# A5:=FORCAST(EMA(CLOSE,17),6);
# B:=A1+A2+A3+A4-4*A5;
# TOWERC:=EMA(B,2);
# STICKLINE(TOWERC>=REF(TOWERC,1),TOWERC,REF(TOWERC,1),1,0),COLORRED;
# STICKLINE(TOWERC<REF(TOWERC,1),TOWERC,REF(TOWERC,1),1,0),COLORGREEN;


rm(list = ls())

gc()

library(foreach)
library(doFuture)
library(RCurl)
library(jsonlite)
library(data.table)
library(glue)
library(TTR)
library(slider)
library(zoo)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

end_date <- as_tradedate(now() - hours(16))

madiff <- function(v, n) 3 * WMA(v, n) - 2 * SMA(v, n)
dktrend <- function(v) {
  a1 <- madiff(EMA(v, 5), 6)
  a2 <- madiff(EMA(v, 8), 6)
  a3 <- madiff(EMA(v, 11), 6)
  a4 <- madiff(EMA(v, 14), 6)
  a5 <- madiff(EMA(v, 17), 6)
  b <- a1 + a2 + a3 + a4 - 4 * a5
  c <- EMA(b, 2)
  return(c)
}

# N:=30;
# MA1:=MA(C,N);
# A1:=MA(H-L,100)*0.34;
# 均线角度:ATAN((MA1-REF(MA1,1))/A1)*180/3.1416;
# 30,COLORRED;
# -30,COLORGREEN;

maang <- function(h, l, c, n) {
  a1 <- SMA(c, n)
  a2 <- SMA(h - l, 100) * 0.34
  ang <- atan((a1 - lag(a1, 1)) / a2) * 180 / pi
  return(ang)
}

n <- 20
roc_threshold <- 0.2

symbols <- list.files("data/hist/") %>%
  str_remove("\\.csv$")

# plan(multisession, workers = availableCores() - 1)

# out <- foreach (
#   symbol = symbols,
#   .combine = "c"
# ) %dofuture% {
#   data <- read_csv(
#     paste0("data/hist/", symbol, ".csv"), show_col_types = FALSE
#   ) %>%
#     full_join(
#       read_csv(paste0("data/adjust/", symbol, ".csv"), show_col_types = FALSE),
#       by = "date"
#     ) %>%
#     arrange(date) %>%
#     fill(adjust, .direction = "down") %>%
#     na.omit() %>%
#     mutate(
#       symbol = !!symbol,
#       across(c(open, high, low, close), ~ .x * adjust),
#       volume = volume / adjust,
#       tp = (high + low) / 2,
#       dk_close = dktrend(close),
#       dk_volume = dktrend(volume),
#       max_after_min = n - runwhich_min(close, n),
#       min_after_max = n - runwhich_max(close, n),
#       maang = maang(high, low, close, 30),
#       col = case_when(
#         dk_close - lag(dk_close) > 0 &
#           (dk_close - lag(dk_close)) < (lag(dk_close) - lag(dk_close, 2)) &
#           dk_volume - lag(dk_volume) < 0 &
#           slide_lgl(
#             maang, ~ any(.x >= 30, na.rm = TRUE),
#             .before = n - 1, .complete = FALSE
#           ) &
#           runmax_var(close, max_after_min) / runMin(close, n) > 1 +
#             roc_threshold ~
#           "green",
#         dk_close - lag(dk_close) < 0 &
#           (dk_close - lag(dk_close)) > (lag(dk_close) - lag(dk_close, 2)) &
#           (
#             dk_volume - lag(dk_volume, 3) > 0 |
#               dk_volume == runMin(dk_volume, n * 2)
#           ) &
#           runmin_var(close, min_after_max) / runMax(close, n) < 1 -
#             roc_threshold ~
#           "red",
#         TRUE ~ NA
#       ),
#       out = case_when(
#         slide_lgl(
#           col, ~ any(.x == "green", na.rm = TRUE),
#           .before = 9, .complete = FALSE
#         ) ~
#           "green",
#         slide_lgl(
#           col, ~ any(.x == "red", na.rm = TRUE),
#           .before = 9, .complete = FALSE
#         ) ~
#           "red",
#         TRUE ~ NA
#       )
#     ) %>%
#     filter(date == end_date, !is.na(out)) %>%
#     try(silent = TRUE)
#   if (inherits(data, "try-error")) return(NULL) else return(list(data))
# } %>%
#   rbindlist()

# plan(sequential)

out <- read_csv("temp_out.csv")

symbols <- out %>%
  filter(amount > 10^9, out == "red") %>%
  pull(symbol)

names <- read_csv("data/symbols.csv", show_col_types = FALSE) %>%
  filter(symbol %in% symbols) %>%
  pull(name)

print(data.frame(symbol = symbols, name = names))

for (symbol in symbols) {
  name <- read_csv("data/symbols.csv", show_col_types = FALSE) %>%
    filter(symbol == !!symbol) %>%
    pull(name)
  data <- read_csv(
    paste0("data/hist/", symbol, ".csv"), show_col_types = FALSE
  ) %>%
    full_join(
      read_csv(paste0("data/adjust/", symbol, ".csv"), show_col_types = FALSE),
      by = "date"
    ) %>%
    arrange(date) %>%
    fill(adjust, .direction = "down") %>%
    na.omit() %>%
    mutate(
      symbol = !!symbol,
      across(c(open, high, low, close), ~ .x * adjust),
      volume = volume / adjust,
      tp = (high + low) / 2,
      dk_close = dktrend(close),
      dk_volume = dktrend(volume),
      max_after_min = n - runwhich_min(close, n),
      min_after_max = n - runwhich_max(close, n),
      maang = maang(high, low, close, 30),
      col = case_when(
        dk_close - lag(dk_close) > 0 &
          (dk_close - lag(dk_close)) < (lag(dk_close) - lag(dk_close, 2)) &
          dk_volume - lag(dk_volume) < 0 &
          slide_lgl(
            maang, ~ any(.x >= 30, na.rm = TRUE),
            .before = n - 1, .complete = FALSE
          ) &
          runmax_var(close, max_after_min) / runMin(close, n) > 1 +
            roc_threshold ~
          "green",
        dk_close - lag(dk_close) < 0 &
          (dk_close - lag(dk_close)) > (lag(dk_close) - lag(dk_close, 2)) &
          (
            dk_volume - lag(dk_volume, 3) > 0 |
              dk_volume == runMin(dk_volume, n * 2)
          ) &
          runmin_var(close, min_after_max) / runMax(close, n) < 1 -
            roc_threshold ~
          "red",
        TRUE ~ NA
      )
    ) %>%
    tail(240) %>%
    mutate(close = (close / first(close) - 1) * 100)
  plot(
    data$date, data$close, col = "black", type = "l",
    main = glue("{symbol} {name}")
  )
  points(data$date, data$close, col = data$col, pch = 16)
  readline(prompt = "Next?")
}

symbols <- symbols[responses == "y"]
writeLines(
  symbols,
  "/Users/anzhouhe/Library/Containers/com.zszq.Mac2020/Data/Documents/user_guest/MR.blk"
)
writeLines(
  c(
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
    "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">",
    "<plist version=\"1.0\">",
    "<dict>",
    paste0("	<key>", symbols, "</key>\n	<string>20250921|1.00</string>"),
    "</dict>",
    "</plist>"
  ),
  "/Users/anzhouhe/Library/Containers/com.zszq.Mac2020/Data/Documents/user_guest/MR.blkdict"
)
