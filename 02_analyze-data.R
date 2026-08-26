# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
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

# %% [markdown]
# # Mixed-Effects Analysis of Ophthalmology Cases
# This script performs statistical testing to compare clinical features between patients 
# who progressed to death vs those who did not. It handles repeated observations per patient 
# using Generalized Linear Mixed Models (GLMM) to ensure statistical independence.
#
# %% [code]
library(pacman)
p_load(arrow, tidyverse, lme4, lmerTest, broom.mixed, corrplot)

# %% [markdown]
# ## Configuration
#
# %% [code]
paths <- list(
  data = "data/processed",
  output = "output"
)
if (!dir.exists(paths$output)) dir.create(paths$output)

config <- list(
  label = "progression_to_death",
  id = "alias_filled",
  obs = "observation"
)

# %% [markdown]
# ## Load Data
#
# %% [code]
df <- read_parquet(file.path(paths$data, "analytic-dataset.parquet"))

print(
    sprintf("Dataset loaded: %d rows x %d cols", 
             nrow(df), ncol(df))
)

# %% [markdown]
# ### Identify Features
#
# %% [code]
numeric_features <- df %>%
  select(where(is.numeric), -all_of(c(config$label, config$id, config$obs))) %>%
  names()

print(
    sprintf("Analyzing %d features across %d patients...", 
             length(numeric_features), 
             df %>% distinct(!!sym(config$id)) %>% nrow())
)

# %% [markdown]
# ## Patient-Level Aggregation
# We aggregate observations to the patient level (mean) for correlation analysis.
#
# %% [code]
df_patient <- df %>%
  group_by(!!sym(config$id), !!sym(config$label)) %>%
  summarise(across(all_of(numeric_features), ~mean(.x, na.rm = TRUE)),
            .groups = "drop")

print(
    sprintf("Patient-level dataset: %d patients x %d features",
            nrow(df_patient), ncol(df_patient) - 2)
)

# %% [markdown]
# ## GLMM Analysis
# Fit: outcome ~ feature + (1 | patient_id)
# The random intercept `(1 | patient_id)` accounts for repeated observations per person.
#
# %% [code]
run_glmm <- function(feature, data, label, id) {
  temp <- data %>% select(all_of(c(label, id, feature))) %>% drop_na()

  if (nrow(temp) < 10) return(NULL)

  formula <- as.formula(paste(label, "~", feature, "+ (1 |", id, ")"))

  tryCatch({
    model <- glmer(formula, data = temp, family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))

    tidy_mod <- tidy(model, effects = "fixed", conf.int = TRUE) %>%
      filter(term == feature) %>%
      mutate(feature = feature,
             test = "GLMM (Binomial)")

    list(
      stats = tidy_mod,
      means = temp %>%
        group_by(!!sym(label)) %>%
        summarise(across(all_of(feature), ~mean(.x, na.rm = TRUE)), .groups = "drop") %>%
        pivot_wider(names_from = !!sym(label), values_from = all_of(feature),
                    names_prefix = "mean_")
    )
  }, error = function(e) NULL)
}

# %% [code]
results <- map(numeric_features, ~run_glmm(.x, df, config$label, config$id)) %>%
  compact() %>%
  map_dfr(~left_join(.x$stats, .x$means, by = "feature")) %>%
  rename(estimate = estimate,
         std_error = std.error,
         z_value = statistic,
         p_value = p.value,
         ci_low = conf.low,
         ci_high = conf.high) %>%
  mutate(mean_positive = `mean_1`,
         mean_negative = `mean_0`,
         diff = mean_positive - mean_negative) %>%
  select(feature, mean_positive, mean_negative, diff, p_value, estimate, z_value, ci_low, ci_high, test) %>%
  arrange(p_value)

print(
    sprintf("✓ GLMM completed: %d features analyzed", nrow(results))
)

# %% [markdown]
# ## Save Results
#
# %% [code]
write_csv(results, file.path(paths$data, "feature_analysis.csv"))

print(
    sprintf("✓ Results saved: %s", file.path(paths$data, "feature_analysis.csv"))
)

# %% [markdown]
# ## Correlation Heatmap (Patient Level)
#
# %% [code]
corr_mat <- df_patient %>%
  select(all_of(numeric_features)) %>%
  cor(use = "pairwise.complete.obs")

print(
    sprintf("Correlation matrix shape: %d x %d", nrow(corr_mat), ncol(corr_mat))
)

# %% [code]
png(file.path(paths$output, "feature_correlation_matrix.png"),
    width = 1400, height = 1200, res = 150)
corrplot(corr_mat,
         method = "color",
         type = "upper",
         tl.col = "black",
         tl.srt = 45,
         col = colorRampPalette(c("#2166AC", "white", "#B2182B"))(200),
         addCoef.col = "grey",
         number.cex = 0.65,
         diag = FALSE,
         mar = c(1, 1, 2, 1),
         main = "Feature Correlation (Patient-Level Means)")
dev.off()

print(
    sprintf("✓ Correlation matrix saved: %s", file.path(paths$output, "feature_correlation_matrix.png"))
)

# %% [code]
cat("\nAnalysis complete.\n")
