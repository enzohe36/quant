rm(list = ls())

gc()

library(doFuture)
library(foreach)
library(RCurl)
library(jsonlite)
library(glue)
library(data.table)
library(ranger)
library(tidyverse)

source("misc.r", encoding = "UTF-8")

# ------------------------------------------------------------------------------

data_dir <- "data/"
model_dir <- "models/"

data_list_path <- paste0(data_dir, "data_list.rds")
nfit_path <- paste0(model_dir, "nfit.rds")
model_path <- paste0(model_dir, "rf_14.rds")
portfolio_path <- "portfolio.csv"

data_list <- readRDS(data_list_path)
coeff <- coef(summary(readRDS(nfit_path)))[, "Estimate"]
rf <- readRDS(model_path)
if (!file.exists(portfolio_path)) file.create(portfolio_path)
portfolio <- read_csv(portfolio_path)
if (nrow(portfolio) == 0) {
  portfolio <- tibble()
} else {
  portfolio <- mutate(portfolio, across(c(index, symbol), as.character))
}

update_portfolio <- function(
  ...,
  short = FALSE,
  out = parent.frame()$out,
  portfolio = parent.frame()$portfolio
) {
  portfolio_list <- portfolio[complete.cases(portfolio), ] %>%
    split(seq_len(nrow(.)))
  out_list <- portfolio[!complete.cases(portfolio), ] %>%
    split(seq_len(nrow(.))) %>%
    c(slice(out, ...) %>% mutate(short = !!short) %>% split(seq_len(nrow(.))))

  out_list <- foreach(
    df = out_list,
    .combine = "append"
  ) %dofuture% {
    rm(list = c("data", "open1", "stop"))
    data <- data_list[[df$symbol]]
    open1 <- slice(data, which(date == !!df$date) + 1) %>% pull(open)
    stop <- ifelse(
      df$short,
      open1 * (1 - (coeff["m"] + 2 * coeff["s"])),
      open1 * (1 + (coeff["m"] + 2 * coeff["s"]))
    ) %>%
      round(2)
    df <- mutate(df, prob = round(prob, 3), stop = !!stop)
    return(list(df))
  } %>%
    c(portfolio_list)

  portfolio <- foreach(
    df = out_list,
    .combine = "append"
  ) %dofuture% {
    rm(list = c("data", "t", "t_max"))
    data <- data_list[[df$symbol]]
    t <- nrow(data) - which(data$date == df$date)
    t_max <- ifelse(df$pred == "e" & !df$short, 2, 10)
    df <- mutate(
      df,
      action = ifelse(
        isTRUE(short & last(!!data$close) <= stop) |
          isTRUE(!short & last(!!data$close) >= stop) |
          !!t >= !!t_max,
        "SELL",
        "HOLD"
      )
    )
    return(list(df))
  } %>%
    rbindlist() %>%
    arrange(desc(action), short, symbol, date)

  return(portfolio)
}

trim_portfolio <- function(..., portfolio = portfolio) {
  slice(portfolio, -c(...))
}

date <- as_tradedate(now() - hours(16))
class <- c("a", "b", "c", "d", "e")
cond_expr <- expression(prob > 0.5 & pred == "a")
val_unit <- 20000

plan(multisession, workers = availableCores() - 1)

new <- foreach(
  data = data_list,
  .combine = "append"
) %dofuture% {
  list(filter(data, date %in% !!date))
} %>%
  rbindlist()

plan(sequential)

pred <- cbind(
  select(new, index:date, close), predict(rf, new)[["predictions"]]
) %>%
  mutate(
    prob = apply(select(., !!class), 1, function(v) max(v)),
    pred = apply(
      select(., !!class), 1, function(v) !!class %>% .[match(max(v), v)]
    ),
    vol = apply(
      select(., symbol, close),
      1,
      function(v) {
        vol_min <- ifelse(grepl("^688", v["symbol"]), 200, 100)
        max(vol_min, round(!!val_unit / as.numeric(v["close"]) / 100) * 100)
      }
    ),
    across(c(close, !!class), ~ NULL)
  )
out <- filter(pred, eval(cond_expr)) %>% arrange(pred, desc(prob))
print(out, nrow = Inf)

# portfolio <- update_portfolio(1:8)
# portfolio <- trim_portfolio(1:4)
# write_csv(portfolio, portfolio_path)
