library(tidyverse)
library(nanoparquet)
df <- nanoparquet::read_parquet("data/processed/analytic-dataset.parquet")
# Check if sex is constant per patient
check_const <- df %>%
  group_by(alias_filled) %>%
  summarise(distinct_sex = n_distinct(sex)) %>%
  summarise(all_const = all(distinct_sex == 1)) %>%
  pull(all_const)
print(paste("Sex constant per patient:", check_const))
