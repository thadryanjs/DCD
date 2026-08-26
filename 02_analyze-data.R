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
# who progressed to death vs those who did not. 
# 
# **Imputation Strategy**: Multiple Imputation by Chained Equations (MICE).
# We generate multiple imputed datasets (m=20) and pool the results using Rubin's Rules
# to ensure that the uncertainty of missing values is reflected in the confidence intervals.
#
# %% [code]
library(arrow)
library(tidyverse)
library(lme4)
library(lmerTest)
library(broom.mixed)
library(corrplot)
library(mice)

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
  obs = "observation",
  m_imputations = 20
)

# %% [markdown]
# ## Load Data
#
# %% [code]
df <- read_parquet(file.path(paths$data, "analytic-dataset.parquet"))

# %% [code]
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

# %% [code]
print(
    sprintf("Analyzing %d features across %d patients...", 
             length(numeric_features), 
             df %>% distinct(!!sym(config$id)) %>% nrow())
)

# %% [markdown]
# ## Multiple Imputation (MICE)
# We impute missing values across the canonical dataset.
#
# %% [code]
df_mice_with_id <- df %>% select(all_of(c(numeric_features, config$label, config$id)))
pred <- make.predictorMatrix(df_mice_with_id)
pred[, config$id] <- 0 
meth <- make.method(df_mice_with_id)
meth[config$id] <- ""

# %% [code]
print(sprintf("Starting MICE imputation (m=%d)...", config$m_imputations))

# %% [code]
imputed_data <- mice(df_mice_with_id, m = config$m_imputations, 
                     predictorMatrix = pred, method = meth, printFlag = FALSE)

# %% [code]
print("✓ MICE imputation complete.")

# %% [markdown]
# ## Pooled GLMM Analysis
# Fit: outcome ~ feature + (1 | patient_id)
#
# %% [code]
run_pooled_glmm <- function(feature, imputed_obj, label, id, original_df) {
  formula_str <- paste(label, "~", feature, "+ (1 |", id, ")")
  formula <- as.formula(formula_str)

  tryCatch({
    fits <- with(imputed_obj, 
                 glmer(formula, family = binomial,
                       control = glmerControl(optimizer = "bobyqa")))
    
    conv_status <- sapply(fits$analyses, function(m) m@optinfo$conv$convergence)
    if (any(conv_status != 0)) return(NULL)
    
    sing_status <- sapply(fits$analyses, function(m) is.singular(m))
    if (any(sing_status)) return(NULL)

    pooled <- pool(fits)
    summary_pooled <- summary(pooled, conf.int = TRUE)
    
    res <- summary_pooled %>% 
      filter(term == feature) %>%
      mutate(feature = feature,
             test = "Pooled GLMM (MICE)")

    first_imp <- complete(imputed_obj, 1)
    means <- first_imp %>%
      group_by(!!sym(label), !!sym(id)) %>%
      summarise(across(all_of(feature), ~mean(.x, na.rm = TRUE)), .groups = "drop") %>%
      group_by(!!sym(label)) %>%
      summarise(across(all_of(feature), ~mean(.x, na.rm = TRUE)), .groups = "drop") %>%
      pivot_wider(names_from = !!sym(label), values_from = all_of(feature),
                  names_prefix = "mean_")

    list(stats = res, means = means)
  }, error = function(e) {
    message(sprintf("Error analyzing %s: %s", feature, e$message))
    return(NULL)
  })
}

# %% [code]
results <- map(numeric_features, ~run_pooled_glmm(.x, imputed_data, config$label, config$id, df)) %>%
  compact() %>%
  map_dfr(~left_join(.x$stats, .x$means, by = "feature")) %>%
  mutate(p_adj = p.adjust(p_value, method = "fdr")) %>%
  rename(estimate = estimate,
         std_error = std.error,
         z_value = statistic,
         p_value = p.value,
         ci_low = `2.5 %`,
         ci_high = `97.5 %`) %>%
  mutate(mean_positive = `mean_1`,
         mean_negative = `mean_0`,
         diff = mean_positive - mean_negative) %>%
  select(feature, mean_positive, mean_negative, diff, p_value, p_adj, estimate, z_value, ci_low, ci_high, test) %>%
  arrange(p_value)

# %% [code]
print(
    sprintf("✓ Pooled GLMM completed: %d features analyzed", nrow(results))
)

# %% [markdown]
# ## Save Results
#
# %% [code]
write_csv(results, file.path(paths$data, "feature_analysis.csv"))

# %% [code]
print(
    sprintf("✓ Results saved: %s", file.path(paths$data, "feature_analysis.csv"))
)

# %% [markdown]
# ## 6. Mixed Effects Forest Plot
#
# %% [code]
plot_results <- results %>%
  mutate(OR = exp(estimate),
         lower = exp(ci_low),
         upper = exp(ci_high)) %>%
  arrange(OR) %>%
  mutate(feature = factor(feature, levels = feature))

ggplot(plot_results, aes(x = OR, y = feature)) +
  geom_vline(xintercept = 1, color = "red", linetype = "dashed") +
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2) +
  geom_point(aes(color = p_adj < 0.05), size = 2) +
  scale_x_log10() +
  scale_color_manual(values = c("grey70", "firebrick"), 
                     labels = c("p_adj >= 0.05", "p_adj < 0.05"), 
                     name = "Significance") +
  labs(title = "Pooled Mixed Effects Model: Odds Ratios (95% CI)",
       subtitle = "MICE Multiple Imputation (m=20) and Rubin's Rules pooling",
       x = "Odds Ratio (Log Scale)",
       y = "Clinical Feature") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8))

# %% [code]
ggsave(file.path(paths$output, "glmm_forest_plot_r.png"), 
       width = 10, height = 12, dpi = 300)

# %% [code]
print(
    sprintf("✓ GLMM Forest Plot saved: %s", file.path(paths$output, "glmm_forest_plot_r.png"))
)

# %% [markdown]
# ## 7. Correlation Heatmap (Patient Level)
#
# %% [code]
first_imp <- complete(imputed_data, 1)
df_patient <- first_imp %>%
  group_by(!!sym(config$id), !!sym(config$label)) %>%
  summarise(across(all_of(numeric_features), ~mean(.x, na.rm = TRUE)),
            .groups = "drop")

corr_mat <- df_patient %>%
  select(all_of(numeric_features)) %>%
  cor(use = "pairwise.complete.obs")

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

# %% [code]
print(
    sprintf("✓ Correlation matrix saved: %s", file.path(paths$output, "feature_correlation_matrix.png"))
)

# %% [code]
cat("\nAnalysis complete.\n")
