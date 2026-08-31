library(tidyverse)
library(nanoparquet)
label <- "progression_to_death"
id <- "alias_filled"
df <- nanoparquet::read_parquet("data/processed/analytic-dataset.parquet")
to_patient <- function(d) {
  d <- d %>%
    select(-observation) %>%
    group_by(.data[[id]], .data[[label]]) %>%
    summarise(across(everything(), ~mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    select(-all_of(id))
  d <- d[sapply(d, function(x) n_distinct(x, na.rm = TRUE) >= 2)]
  d
}
all_obs <- to_patient(df)
selected_feats_csv <- read_csv("output/selected-features.csv", show_col_types = FALSE)
selected_feats <- selected_feats_csv$feature
r_feats <- setdiff(names(all_obs), label)
sym_diff <- union(setdiff(r_feats, selected_feats), setdiff(selected_feats, r_feats))
print(sym_diff)
