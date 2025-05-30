rm(list = ls())

gc()

plan(multisession, workers = availableCores() - 1)

data_dir <- "data/"
index_comp_path <- paste0(data_dir, "index_comp.csv")

model_dir <- "models/"
dir.create(model_dir)
data_comb_path <- paste0(model_dir, "data_comb.rds")
train_path <- paste0(model_dir, "train.rds")
test_path <- paste0(model_dir, "test.rds")

t_obs <- 20
t_train <- 1200
t_test <- 20

index_comp <- read_csv(index_comp_path)

# Load historical stock data & generate features
data_comb <- foreach(
  symbol = index_comp$symbol,
  .combine = "append"
) %dofuture% {
  var <- c("var", "data_path", "index_comp_i", "data", "try_error", "lst")
  rm(list = var)

  data_path <- paste0(data_dir, symbol, ".csv")
  if (!file.exists(data_path)) return(NULL)

  index_comp_i <- filter(index_comp, symbol == !!symbol)

  # Skip stocks with insufficient history
  data <- read_csv(data_path, show_col_types = FALSE) %>%
    mutate(
      symbol = !!index_comp_i$symbol,
      name = !!index_comp_i$name,
      index = !!index_comp_i$index,
      industry = !!index_comp_i$industry,
      .before = date
    )

  try_error <- try(
    data <- data %>%
      mutate(
        target = get_roc(close, lead(close, t_obs)),
        obv = OBV(close, vol)
      ) %>%
      add_rocnorm(c("close", "obv"), c(1:19, (1:12) * 20), 240),
    silent = TRUE
  )

  lst <- list()
  lst[[symbol]] <- data
  return(lst)
} %>%
  rbindlist(fill = TRUE)
saveRDS(data_comb, data_comb_path)
tsprint(glue("Loaded {length(index_comp$symbol)} stocks."))

# data_comb <- read_rds(data_comb_path)

data_comb_trim <- data_comb %>%
  na.omit() %>%
  select(symbol:date, target, contains("_rocnorm")) %>%
  filter(index %in% c("000300", "000905")) %>%
  group_by(date) %>%
  mutate(
    target = findInterval(
      target, c(-Inf, quantile(target, na.rm = TRUE)[2:4], Inf)
    ) %>%
      as.factor()
  ) %>%
  ungroup()

date_all <- unique(data_comb_trim$date) %>% sort()
date_train <- head(date_all, -t_obs - t_test) %>% tail(t_train)
train <- filter(data_comb_trim, date %in% date_train) %>% select(-(symbol:date))
saveRDS(train, train_path)
glue("training = [{first(date_train)}, {last(date_train)}]") %>% writeLines()

date_test <- tail(date_all, t_test)
test <- filter(data_comb_trim, date == date_test) %>% select(-(symbol:date))
saveRDS(test, test_path)
glue("test = [{first(date_test)}, {last(date_test)}]") %>% writeLines()

plan(sequential)
