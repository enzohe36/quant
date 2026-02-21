library(tidyverse)

results <- read_csv("models/checkpoints_20260218_drop00_mdrop05/test_results.csv") %>%
  split(.$symbol)

idx_sample <- sample(length(results), 10)

for (idx in idx_sample) {
  df <- results[[idx]] %>%
    arrange(date) %>%
    mutate(
      color = case_when(
        position == 1 ~ "red",
        position == -1 ~ "forestgreen",
        TRUE ~ "black"
      )
    )

  plot <- ggplot(df, aes(x = date, y = price)) +
    geom_segment(aes(xend = lead(date), yend = lead(price), color = color)) +
    scale_color_identity() +
    labs(x = df$symbol[1]) +
    theme_minimal()
  print(plot)
}
