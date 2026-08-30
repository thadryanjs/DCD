# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

```{r}
library(dplyr)
library(readxl)

df_pos <- read_excel("data/positive-cases-Jennys-data-Edited.xlsx") %>%
  mutate(row_number = row_number())

df_pos_sample_before_head <- df_pos %>%
  select(row_number, Alias, Age) %>%
  head(27)

df_pos_sample_before_head %>%
  print(n=Inf)

df_pos_sample_before_tail <- df_pos %>%
  select(row_number, Alias, Age) %>%
  tail(27)

df_pos_sample_before_tail %>%
  print(n=Inf)
```


# %% [markdown]
This allows us to see two full 'cycles' of how alias works at the top and bottom of the script. It seems clear that this is a merged cell artifact and we need to populate forward until we hit the next number.

# %% [code]
n_rows <- nrow(df_pos)

for (i in 1:n_rows) {

  current_val <- df_pos$Alias[i]
  next_val <- df_pos$Alias[i+1]

  # make sure we don't run of the end
  if (i == n_rows) {
    break
  } else {
    if (!is.na(current_val)) {
      if (is.numeric(current_val) && is.na(next_val)) {
        df_pos$Alias[i+1] <- current_val
      }
    }
  }
}

df_pos_sample_after_head <- df_pos %>%
  select(row_number, Alias, Age) %>%
  head(27)

df_pos_sample_after_tail <- df_pos %>%
  select(row_number, Alias, Age) %>%
  tail(27)

# %% [code]
df_pos_sample_before_head

# %% [code]
df_pos_sample_after_head

# %% [code]
df_pos_sample_before_tail

# %% [code]
df_pos_sample_after_tail


# %% [code]
df_neg <- read_excel("data/negative-cases-ANR-Data.xlsx") %>%
  mutate(row_number = row_number())

df_pos[10:30, ]
