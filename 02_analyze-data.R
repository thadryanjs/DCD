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

# %% [markdown]
# # Mixed-Effects Analysis of Clinical Cases
# This script performs statistical testing to compare clinical features between patients 
# who progressed to death vs those who did not. 
# 
# **Imputation Strategy**: Multiple Imputation by Chained Equations (MICE).
# We generate multiple imputed datasets (m=5) and pool the results using Rubin's Rules
# to ensure that the uncertainty of missing values is reflected in the confidence intervals.
#
# %% [code]
library(tidyverse)
library(lme4)
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
  m_imputations = 5
)

# %% [markdown]
# ## Load Data
#
# %% [code]
df <- read_csv(file.path(paths$data, "analytic-dataset.csv"))

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
                     predictorMatrix = pred, method = meth, printFlag = TRUE)

# %% [code]
print("✓ MICE imputation complete.")

# %% [markdown]
# # Model Identifiability Check
# Check if outcome varies within patients to ensure GLMM is well-posed.
# %% [code]
prelim <- complete(imputed_data, 1)
within_var <- prelim %>%
  dplyr::group_by(.data[[config$id]]) %>%
  dplyr::summarise(n_levels = dplyr::n_distinct(.data[[config$label]]), .groups = "drop")

if (all(within_var$n_levels == 1)) {
  warning(
    "Outcome is CONSTANT within every patient. A (1 | patient_id) random ",
    "intercept is not identifiable here (expect singular fits and unstable ",
    "fixed effects). A patient-level GLM is the appropriate model for a ",
    "time-invariant outcome. Proceeding as requested, but treat results with care."
  )
}

# %% [markdown]
# # Pooled GLMM Analysis
# We use a binomial generalized linear mixed model with a patient-level random intercept
# to account for repeated measures within patients, pooled across imputations via Rubin's Rules.
#
# %% [code]
run_pooled_glmm <- function(feature, imputed_obj, label, id, original_df) {
  tryCatch({
    fits_list <- list()
    diag_list <- list()
    
    for (i in 1:imputed_obj$m) {
      di <- complete(imputed_obj, i)
      
      # Step 1: Data prep to fix 'list' coercion error
      # Extract columns as atomic vectors via unlist()
      d <- data.frame(
        y   = as.integer(unlist(di[[label]],   use.names = FALSE)),
        x   = as.numeric(scale(unlist(di[[feature]], use.names = FALSE))),
        grp = factor(   unlist(di[[id]],       use.names = FALSE))
      )
      stopifnot(!any(vapply(d, is.list, logical(1))))
      
      # Step 2: Fit GLMM
      fit <- tryCatch(
        {
          message(sprintf("Fitting GLMM for %s (Imputation %d)...", feature, i))
          glmer(
            y ~ x + (1 | grp),
            data    = d,
            family  = binomial,
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
          )
        },
        error = function(e) {
          # Fallback to nAGQ = 0 (Laplace approximation)
          tryCatch(
            {
              message(sprintf("Fallback to nAGQ=0 for %s (Imputation %d)...", feature, i))
              glmer(
                y ~ x + (1 | grp),
                data    = d,
                family  = binomial,
                nAGQ    = 0,
                control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
              )
            },
            error = function(e2) { 
              message(sprintf("glmer error [%s, imp %d]: %s", feature, i, e2$message))
              NULL 
            }
          )
        }
      )
      
      if (!is.null(fit)) {
        fits_list[[i]] <- fit
        # Step 3: Per-fit diagnostics
        diag_list[[i]] <- list(
          singular  = lme4::isSingular(fit),
          re_sd     = sqrt(as.numeric(lme4::VarCorr(fit)$grp)),
          converged = is.null(fit@optinfo$conv$lme4$messages)
        )
      }
    }
    
    fits <- Filter(Negate(is.null), fits_list)
    if (length(fits) < 2) return(NULL)   # Rubin's Rules need m >= 2
    
    # Step 4: Pool fixed effects with Rubin's Rules
    pooled <- summary(mice::pool(mice::as.mira(fits)), conf.int = TRUE)
    row_x  <- pooled[pooled$term == "x", ]
    
    # Aggregate diagnostics
    valid_diags <- Filter(Negate(is.null), diag_list)
    frac_singular <- mean(sapply(valid_diags, `[[`, "singular"))
    frac_nonconv  <- mean(!sapply(valid_diags, `[[`, "converged"))

    pooled_res <- tibble::tibble(
      feature       = feature,
      estimate      = row_x$estimate,
      std_error     = row_x$std.error,
      z_value       = row_x$statistic,
      p_value       = row_x$p.value,
      ci_low        = row_x[["2.5 %"]],
      ci_high       = row_x[["97.5 %"]],
      frac_singular = frac_singular,
      frac_nonconv  = frac_nonconv,
      test          = "Binomial GLMM (random patient intercept), MICE + Rubin's Rules"
    )
    
    # Means from first imputation for descriptive statistics
    first_imp <- complete(imputed_obj, 1)
    first_imp <- as.data.frame(first_imp)
    
    means <- first_imp %>%
      group_by(!!sym(label), !!sym(id)) %>%
      summarise(val = mean(!!sym(feature), na.rm = TRUE), .groups = "drop") %>%
      group_by(!!sym(label)) %>%
      summarise(val = mean(val, na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(names_from = !!sym(label), values_from = val,
                  names_prefix = "mean_") %>%
      mutate(feature = feature)

    list(stats = pooled_res, means = means)
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
  mutate(
    mean_positive = `mean_1`,
    mean_negative = `mean_0`,
    diff = mean_positive - mean_negative
  ) %>%
  select(feature, mean_positive, mean_negative, diff, p_value, p_adj, estimate, z_value, ci_low, ci_high, test, frac_singular, frac_nonconv) %>%
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
# # Mixed Effects Forest Plot
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
  labs(title = "Pooled Logistic Regression: Odds Ratios (95% CI)",
       subtitle = "Identifying High-Risk Cases for 'Send It' Decision (Patient-level means + MICE)",
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
# # Correlation Heatmap (Patient Level)
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
