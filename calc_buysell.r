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

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"
hist_dir <- paste0(data_dir, "hist/")
adjust_dir <- paste0(data_dir, "adjust/")
mktcap_dir <- paste0(data_dir, "mktcap/")
val_dir <- paste0(data_dir, "val/")
combined_path <- paste0(data_dir, "combined.rds")

end_date <- as_tradedate(now() - hours(16))
start_date <- end_date - years(20)
quarters <- seq(
  start_date %m-% months(3),
  end_date %m-% months(3),
  by = "1 day"
) %>%
  quarter("date_last") %>%
  unique()

# A1:=FORCAST(EMA(CLOSE,5),6);
# A2:=FORCAST(EMA(CLOSE,8),6);
# A3:=FORCAST(EMA(CLOSE,11),6);
# A4:=FORCAST(EMA(CLOSE,14),6);
# A5:=FORCAST(EMA(CLOSE,17),6);
# B:=A1+A2+A3+A4-4*A5;
# TOWERC:=EMA(B,2);
# STICKLINE(TOWERC>=REF(TOWERC,1),TOWERC,REF(TOWERC,1),1,0),COLORRED;
# STICKLINE(TOWERC<REF(TOWERC,1),TOWERC,REF(TOWERC,1),1,0),COLORGREEN;

madiff <- function(v, n) {
  3 * WMA(v, n) - 2 * SMA(v, n)
}

get_dk <- function(v) {
  a1 <- madiff(EMA(v, 5), 6)
  a2 <- madiff(EMA(v, 8), 6)
  a3 <- madiff(EMA(v, 11), 6)
  a4 <- madiff(EMA(v, 14), 6)
  a5 <- madiff(EMA(v, 17), 6)
  b <- a1 + a2 + a3 + a4 - 4 * a5
  dk <- EMA(b, 2)
  return(dk)
}

# N:=30;
# MA1:=MA(C,N);
# A1:=MA(H-L,100)*0.34;
# 均线角度:ATAN((MA1-REF(MA1,1))/A1)*180/3.1416;
# 30,COLORRED;
# -30,COLORGREEN;

maang <- function(h, l, c) {
  a1 <- SMA(c, 30)
  a2 <- SMA(h - l, 100) * 0.34
  ang <- atan((a1 - lag(a1, 1)) / a2) * 180 / pi
  return(ang)
}

n <- 20
roc_threshold <- 0.2

symbols <- list.files(hist_dir) %>%
  str_remove("\\.csv$")

combined <- foreach (
  symbol = symbols,
  .combine = "c"
) %dofuture% {
  print(symbol)

  hist_path <- paste0(hist_dir, symbol, ".csv")
  adjust_path <- paste0(adjust_dir, symbol, ".csv")
  mktcap_path <- paste0(mktcap_dir, symbol, ".csv")
  val_path <- paste0(val_dir, symbol, ".csv")

  hist <- read_csv(hist_path, show_col_types = FALSE)
  if (nrow(hist) < max(100, n)) return(NULL)
  if (last(SMA(hist$amount, 3)) < 10^9) return(NULL)

  if (!file.exists(adjust_path)) {
    return(NULL)
  } else {
    adjust <- read_csv(adjust_path, show_col_types = FALSE)
  }

  if (!file.exists(mktcap_path)) {
    return(NULL)
  } else {
    mktcap <- read_csv(mktcap_path, show_col_types = FALSE)
  }

  if (!file.exists(val_path)) {
    return(NULL)
  } else {
    val <- read_csv(val_path, show_col_types = FALSE) %>%
      full_join(tibble(date = !!quarters), by = "date") %>%
      arrange(date) %>%
      mutate(
        revenue = runSum(revenue, 4),
        np = runSum(np, 4),
        np_deduct = runSum(np_deduct, 4),
        cfps = runSum(cfps, 4),
        quarter = date
      )
  }

  # combine hist, adjust, mktcap, val by date
  # arrange by date
  # fill down names(hist), names(adjust), names(mktcap)
  # calculate shares
  # fill down shares, quarter
  # calculate mktcap, equity, cf, adjusted ohlcv
  # group by quarter
  # fill down np, np_deduct, equity, revenue, cf
  # ungroup
  # calculate pe, pe_deduct, pb, ps, pcf, roe
  # add symbol
  # select symbol, date, ohlcv, amount, financials
  # filter by hist dates
  data <- hist %>%
    full_join(adjust, by = "date") %>%
    full_join(mktcap, by = "date") %>%
    full_join(val, by = "date") %>%
    arrange(date) %>%
    fill(names(hist), names(adjust), names(mktcap), .direction = "down") %>%
    mutate(shares = mktcap / close * 10^8) %>%
    fill(shares, quarter, .direction = "down") %>%
    mutate(
      mktcap = shares * close,
      equity = bvps * shares,
      cf = cfps * shares,
      across(c(open, high, low, close), ~ .x * adjust),
      volume = volume / adjust
    ) %>%
    group_by(quarter) %>%
    fill(np, np_deduct, equity, revenue, cf, .direction = "down") %>%
    ungroup() %>%
    mutate(
      pe = mktcap / np,
      pe_deduct = mktcap / np_deduct,
      pb = mktcap / equity,
      ps = mktcap / revenue,
      pcf = mktcap / cf,
      roe = np / equity,
      symbol = !!symbol,
      across(c(open, high, low, close, pe, pe_deduct, pb, ps, pcf), round, 2),
      volume = round(volume),
      amount = round(amount / 10^8, 2),
      roe = round(roe, 4)
    ) %>%
    select(symbol, names(hist), pe, pe_deduct, pb, ps, pcf, roe) %>%
    filter(date %in% hist$date & date >= start_date)




  data <- data %>%
    mutate(
      close_dk = get_dk(close),
      volume_dk = get_dk(volume),
      tp = (high + low) / 2,
      min_window = n - run_whichmax(high, n),
      max_window = n - run_whichmin(low, n),
      roc_nmin = run_varmin(low, min_window) / runMax(high, n),
      roc_nmax = run_varmax(high, max_window) / runMin(low, n),
      # maang = maang(high, low, close),
      buy = (
        # close_dk decreases, volume_dk increases
        close_dk - lag(close_dk) < 0 &
          close_dk - lag(close_dk) > lag(close_dk - lag(close_dk)) &
          volume_dk - lag(volume_dk) > 0 &
          roc_nmin < 1 - roc_threshold
      ),
      sell = (
        # close_dk increases, volume_dk decreases
        close_dk - lag(close_dk) > 0 &
          close_dk - lag(close_dk) < lag(close_dk - lag(close_dk)) &
          volume_dk - lag(volume_dk) < 0 &
          roc_nmax > 1 + roc_threshold
      )
    ) %>%
    select(
      symbol, date, close, pe, pe_deduct, pb, ps, pcf, roe, buy, sell
    )

  return(list(data))
} %>%
  rbindlist()
saveRDS(combined, combined_path)

# plot(data$date, data$close, type = "l")
# points(data$date, data$close, col = ifelse(data$buy, "red", NA), pch = 16)
# points(data$date, data$close, col = ifelse(data$sell, "green", NA), pch = 16)

plan(sequential)














slide_lgl(maang, ~ any(.x >= 30, na.rm = TRUE), .before = n - 1)

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
      close_dk = dk(close),
      volume_dk = dk(volume),
      max_after_min = n - runwhich_min(close, n),
      min_after_max = n - runwhich_max(close, n),
      maang = maang(high, low, close, 30),
      col = case_when(
        close_dk - lag(close_dk) > 0 &
          (close_dk - lag(close_dk)) < (lag(close_dk) - lag(close_dk, 2)) &
          volume_dk - lag(volume_dk) < 0 &
          slide_lgl(
            maang, ~ any(.x >= 30, na.rm = TRUE),
            .before = n - 1, .complete = FALSE
          ) &
          runmax_var(close, max_after_min) / runMin(close, n) > 1 +
            roc_threshold ~
          "green",
        close_dk - lag(close_dk) < 0 &
          (close_dk - lag(close_dk)) > (lag(close_dk) - lag(close_dk, 2)) &
          (
            volume_dk - lag(volume_dk, 3) > 0 |
              volume_dk == runMin(volume_dk, n * 2)
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
