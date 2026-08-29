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
# # Inferential Analysis: DCD Progression
#
# ## Statistical Framework: Patient-Level GLM
# The outcome (progression to death) is a patient-level constant; it does not vary across observations for a single individual. Consequently, a mixed-effects model (GLMM) with a patient random intercept is not identifiable, as there is no within-patient outcome variance to explain. 
#
# The **patient-level logistic regression (GLM)** is the proper inferential model for this data structure. By averaging observations per patient, we reduce the problem to a standard binary classification task, avoiding the boundary variance estimates and attenuated fixed effects typical of inappropriately specified mixed models on time-invariant outcomes.
#
# Missing values are handled using Multiple Imputation by Chained Equations (MICE) 
# with $m=5$ imputations. Each feature is fitted once per imputed dataset, and the 
# results are pooled using Rubin's rules. This allows the final confidence intervals 
# to account for the uncertainty introduced by the imputation process. 
#
# Both the "all observations" and "first-look only" analyses are pooled across 
# imputations for consistency. Features are scaled, so estimates are per standard deviation.
#
# %% [code]
library(tidyverse)
library(mice)

# %% [code]
label <- "progression_to_death"
id <- "alias_filled"
m <- 5
dir.create("output", showWarnings = FALSE)

df <- read_csv("data/processed/analytic-dataset.csv",
               guess_max = .Machine$integer.max, show_col_types = FALSE)

# %% [code]
print(sprintf("%d rows, %d patients", nrow(df), n_distinct(df[[id]])))

# %% [markdown]
# ## Patient-Level Frames
# Average to one row per patient, drop the id, keep only columns with at least two
# distinct values, then scale. The distinct-value filter matters: a constant column
# becomes all `NaN` under `scale()` and makes mice fail with an unhelpful error.
#
# %% [code]
to_patient <- function(d, name = "unnamed") {
  # Aggregation Receipt: Dimensions
  rows_before <- nrow(d)
  cols_before <- ncol(d)
  
  # Temporary dataframe to keep ID for sample check
  d_with_id <- d %>%
    select(-observation) %>%
    group_by(.data[[id]], .data[[label]]) %>%
    summarise(across(everything(), ~mean(.x, na.rm = TRUE)), .groups = "drop")
  
  rows_after <- nrow(d_with_id)
  cols_after <- ncol(d_with_id)
  print(sprintf("[%s] Aggregation: %d rows x %d cols -> %d rows x %d cols", 
                name, rows_before, cols_before, rows_after, cols_after))
  
  # Aggregation Receipt: Sample value check
  sample_patient <- unique(d[[id]])[1]
  sample_feats <- setdiff(names(d_with_id), c(id, label))[1:2]
  
  cat(sprintf("[%s] Sample mean check (Patient %s):\n", name, sample_patient))
  for (f in sample_feats) {
    raw_vals <- d %>% filter(.data[[id]] == sample_patient) %>% pull(!!sym(f))
    mean_val <- d_with_id %>% filter(.data[[id]] == sample_patient) %>% pull(!!sym(f))
    cat(sprintf("  %s: mean(%.3f, %.3f, ...) = %.3f\n", f, raw_vals[1], raw_vals[2], mean_val))
  }
  
  d_final <- d_with_id %>% select(-all_of(id))
  d_final <- d_final[sapply(d_final, function(x) n_distinct(x, na.rm = TRUE) >= 2)]
  d_final %>% mutate(across(-all_of(label), ~as.numeric(scale(.x))))
}

# %% [code]
all_obs <- to_patient(df, "All Observations")
first_obs <- to_patient(df %>% filter(observation == 1), "First Look Only")

# %% [code]
print(sprintf("All observations: %d x %d. First only: %d x %d.",
              nrow(all_obs), ncol(all_obs), nrow(first_obs), ncol(first_obs)))

# Scaling Receipt: Verify mean ~ 0, SD ~ 1 for selected features
receipt_feats <- setdiff(names(all_obs), label)[1:3]
cat("\n--- Scaling Receipt (All Observations) ---\n")
for (f in receipt_feats) {
  # Need original data to check "before"
  raw_vals <- df %>% 
    group_by(.data[[id]], .data[[label]]) %>% 
    summarise(m = mean(.data[[f]], na.rm = TRUE), .groups = "drop") %>% 
    pull(m)
  
  before_mean <- mean(raw_vals, na.rm = TRUE)
  before_sd <- sd(raw_vals, na.rm = TRUE)
  after_mean <- mean(all_obs[[f]], na.rm = TRUE)
  after_sd <- sd(all_obs[[f]], na.rm = TRUE)
  
  cat(sprintf("%s: Before[mean=%.3f, sd=%.3f] -> After[mean=%.3f, sd=%.3f]\n", 
              f, before_mean, before_sd, after_mean, after_sd))
}

# Feature Reconciliation: Compare R features against ML selection
if (file.exists("output/selected-features.csv")) {
  ml_feats <- read_csv("output/selected-features.csv", show_col_types = FALSE) %>% pull(feature)
  r_feats <- setdiff(names(all_obs), label)
  diff_set <- symmetric_difference <- setdiff(ml_feats, r_feats)
  diff_set_rev <- setdiff(r_feats, ml_feats)
  all_diffs <- unique(c(diff_set, diff_set_rev))
  
  cat("\n--- Feature Reconciliation Receipt ---\n")
  if (length(all_diffs) == 0) {
    cat("✓ Feature sets align perfectly between 01 and 02.\n")
  } else {
    cat(sprintf("⚠ Mismatch found! %d features differ.\n", length(all_diffs)))
    print(all_diffs)
  }
} else {
  cat("\n--- Feature Reconciliation Receipt ---\nWarning: output/selected-features.csv not found. Skipping.\n")
}

# %% [markdown]
# ## Fit
# Two loops: features outside, imputations inside, because a feature needs one fit per
# imputed dataset before those fits can be pooled. A fitted probability pinned at 0 or 1
# means the fit separated, which glm reports as a warning rather than an error, so it is
# checked directly and flagged.
#
# %% [code]
analyze <- function(d) {
  # **LOADBEARING** — Multiple Imputation (m=5) accounts for missing data uncertainty.
  # Seed is pinned to match ML pipeline for reproducibility.
  # Consumed by: `data/processed/feature_analysis.csv`
  imp <- mice(d, m = m, printFlag = FALSE, seed = 8675309)
  
  # Imputation Provenance: Log events (collinear columns, etc.)
  # If loggedEvents > 0, mice stripped columns; results may be biased.
  print(sprintf("mice logged events: %d", NROW(imp$loggedEvents)))
  if (NROW(imp$loggedEvents) > 0) {
    print("mice stripped columns due to collinearity/constant values:")
    print(imp$loggedEvents)
  }
  
  out <- NULL
  for (f in setdiff(names(d), label)) {
    fits <- list()
    separated <- FALSE
    for (i in 1:m) {
      fits[[i]] <- glm(as.formula(sprintf("%s ~ `%s`", label, f)),
                       complete(imp, i), family = binomial)
      p <- fitted(fits[[i]])
      separated <- separated || any(p < 1e-8 | p > 1 - 1e-8)
    }
    # Apply Rubin's Rules via pool()
    s <- summary(pool(as.mira(fits)), conf.int = TRUE)
    s <- s[s$term != "(Intercept)", ]
    
    # Capture stability details
    is_unstable <- separated || abs(s$estimate) > 10 || s$std.error > 5
    out <- rbind(out, data.frame(
      feature = f, estimate = s$estimate, std_error = s$std.error, p_value = s$p.value,
      ci_low = s[["2.5 %"]], ci_high = s[["97.5 %"]],
      unstable = is_unstable, separated = separated))
  }
  
  # Transparency: Print unstable fits
  unstable_fits <- out %>% filter(unstable)
  if (nrow(unstable_fits) > 0) {
    cat(sprintf("\nUnstable fits detected for %d features:\n", nrow(unstable_fits)))
    print(unstable_fits %>% select(feature, estimate, std_error, separated))
  }
  
  out %>% mutate(p_adj = p.adjust(p_value, "fdr")) %>% arrange(p_value)
}

# %% [code]
results <- analyze(all_obs)
first_results <- analyze(first_obs)

# %% [code]
print(sprintf("Significant (FDR, stable fits): %d of %d. Unstable: %d.",
              sum(results$p_adj < 0.05 & !results$unstable),
              nrow(results), sum(results$unstable)))

# %% [code]
print(head(results, 15))

# %% [code]
write_csv(results, "data/processed/feature_analysis.csv")
write_csv(first_results, "data/processed/first_look_analysis.csv")

# %% [markdown]
# ## Forest Plot
# Odds ratios per SD. Unstable fits are excluded: their intervals span orders of
# magnitude and would flatten everything else on a log axis.
#
# %% [code]
plot_df <- results %>%
  filter(!unstable) %>%
  mutate(or = exp(estimate), lo = exp(ci_low), hi = exp(ci_high), sig = p_adj < 0.05) %>%
  arrange(or) %>%
  mutate(feature = factor(feature, feature))

ggplot(plot_df, aes(or, feature)) +
  geom_vline(xintercept = 1, color = "red", linetype = "dashed") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.2) +
  geom_point(aes(color = sig), size = 2) +
  scale_x_log10() +
  scale_color_manual(values = c("grey70", "firebrick"), labels = c("ns", "FDR p < 0.05"), name = NULL) +
  labs(title = "DCD Progression: Unadjusted Marginal Odds Ratios (95% CI)",
       subtitle = sprintf("%d features; %d unstable fits excluded",
                          nrow(plot_df), sum(results$unstable)),
       x = "Unadjusted Marginal Odds Ratio per SD (log scale)", y = NULL) +
  theme_minimal()

ggsave("output/glmm_forest_plot_r.png", width = 10, height = 12, dpi = 300)

# %% [markdown]
# ## Limitations
#
# - **Multiple Testing.** We apply FDR correction (Benjamini-Hochberg) across all univariate fits. 
#   The resulting p_adj indicates the proportion of false discoveries expected among 
#   significant features.
# - **Inferential Nature.** Pooled Odds Ratios are unadjusted. They identify strong candidate 
#   drivers but do not account for confounders.
# - **No mixed-effects model.** The spec called for lme4 with person-level random
#   effects, but the outcome is time-invariant within patient, so a random intercept is
#   not identifiable. A mixed model here gives boundary variance estimates and
#   attenuated fixed effects; the patient-level GLM is the right model for this outcome.
# - **No time-to-event analysis.** Only whether the event occurred is recorded, not when,
#   so discrete-time survival with a patient frailty term is unavailable.
# - **Univariate screens.** Each feature is modelled alone with FDR correction across
#   features. The odds ratios are unadjusted, not independent effects.
# - **Equal patient weighting.** Averaging gives a patient with twelve observations the
#   same weight as one with two.
# - **Imputation ignores clustering** and uses every other column as a predictor. Check
#   the logged-events count above if the patient count is small.
# - **Feature correlation** is covered in `01_explore-data.py`; not duplicated here.
# - **Pin the mice version.** `pool` summary column names have changed across releases.
#
# %% [markdown]
# ## CSV Column Definitions
#
# The output `feature_analysis.csv` contains:
# - `estimate`: The log-odds ratio per standard deviation of the patient-level mean.
# - `p_adj`: FDR-adjusted p-value.
# - `unstable`: Flag for separated fits or extreme estimates/SEs.
#
# Note that in `04_analyze-model.py`, the `estimate` is exponentiated ($\exp(\beta)$) to 
# produce the Odds Ratio (OR) used in the forest plot.
#
# %% [code]
print("Analysis complete.")
