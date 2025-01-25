rm(list = ls())

gc()

library(doFuture)
library(foreach)
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
portfolio <- if (nrow(read_csv(portfolio_path)) == 0) {
  tibble()
} else {
  read_csv(portfolio_path, col_types = c(index = "c", symbol = "c"))
}

update_portfolio <- function(
  ..., short = FALSE, .out = out, .portfolio = portfolio
) {
  portfolio_list <- .portfolio[complete.cases(.portfolio), ] %>%
    split(seq_len(nrow(.)))
  out_list <- .portfolio[!complete.cases(.portfolio), ] %>%
    split(seq_len(nrow(.))) %>%
    c(slice(.out, ...) %>% mutate(short = !!short) %>% split(seq_len(nrow(.))))

  out_list <- foreach(
    df = out_list,
    .combine = "append"
  ) %dofuture% {
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
    data <- data_list[[df$symbol]]
    t <- nrow(data) - which(data$date == df$date)
    t_max <- ifelse(df$class == "e" & !df$short, 3, 10)
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

# date <- as_date("2025-01-14"):as_date(now() - hours(16))
date <- as_date(now() - hours(16))

class <- c("a", "b", "c", "d", "e")
cond_expr <- expression(pred$prob > 0.5 & pred$class == "e")
val_unit <- 20000

plan(multisession, workers = availableCores() - 1)

new <- foreach(
  data = data_list,
  .combine = "append"
) %dofuture% {
  list(filter(data, date %in% as_tradedate(!!date)))
} %>%
  rbindlist()

pred <- cbind(
  select(new, index:date, close), predict(rf, new)[["predictions"]]
) %>%
  mutate(
    prob = apply(select(., class), 1, function(v) max(v)),
    class = apply(select(., class), 1, function(v) class[match(max(v), v)]),
    vol = round(!!val_unit / close / 100) * 100,
    across(c(!!class, close), ~ NULL)
  )
out <- filter(pred, eval(cond_expr)) %>% arrange(class, desc(prob))
print(out)

num_e <- filter(pred, prob > 0.5) %>%
  split(.$date) %>%
  sapply(function(df) length(df$class[df$class == "e"]))
plot(
  names(num_e) %>% as_date(), num_e,
  type = "l",
  xlab = "Date", ylab = "count(e)]"
)

# portfolio <- update_portfolio(1:2)
# write_csv(portfolio, portfolio_path)
# print(portfolio)

plan(sequential)
