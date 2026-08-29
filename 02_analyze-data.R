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
# The outcome belongs to the patient, not the observation: every row for a patient
# carries the same label. A patient random intercept has no within-patient outcome
# variation to explain and is not identifiable, so we average each patient's
# observations and fit a patient-level logistic regression per feature, FDR corrected.
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
to_patient <- function(d) {
  d <- d %>%
    select(-observation) %>%
    group_by(.data[[id]], .data[[label]]) %>%
    summarise(across(everything(), ~mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    select(-all_of(id))
  d <- d[sapply(d, function(x) n_distinct(x, na.rm = TRUE) >= 2)]
  d %>% mutate(across(-all_of(label), ~as.numeric(scale(.x))))
}

# %% [code]
all_obs <- to_patient(df)
first_obs <- to_patient(df %>% filter(observation == 1))

# %% [code]
print(sprintf("All observations: %d x %d. First only: %d x %d.",
              nrow(all_obs), ncol(all_obs), nrow(first_obs), ncol(first_obs)))

# %% [markdown]
# ## Fit
# Two loops: features outside, imputations inside, because a feature needs one fit per
# imputed dataset before those fits can be pooled. A fitted probability pinned at 0 or 1
# means the fit separated, which glm reports as a warning rather than an error, so it is
# checked directly and flagged.
#
# %% [code]
analyze <- function(d) {
  imp <- mice(d, m = m, printFlag = FALSE, seed = 8675309)
  print(sprintf("mice logged events: %d", NROW(imp$loggedEvents)))
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
    s <- summary(pool(as.mira(fits)), conf.int = TRUE)
    s <- s[s$term != "(Intercept)", ]
    out <- rbind(out, data.frame(
      feature = f, estimate = s$estimate, std_error = s$std.error, p_value = s$p.value,
      ci_low = s[["2.5 %"]], ci_high = s[["97.5 %"]],
      unstable = separated || abs(s$estimate) > 10 || s$std.error > 5))
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
  labs(title = "DCD Progression: Patient-Level Odds Ratios (95% CI)",
       subtitle = sprintf("%d features; %d unstable fits excluded",
                          nrow(plot_df), sum(results$unstable)),
       x = "Odds ratio per SD (log scale)", y = NULL) +
  theme_minimal()

ggsave("output/glmm_forest_plot_r.png", width = 10, height = 12, dpi = 300)

# %% [markdown]
# ## Limitations
#
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
# %% [code]
print("Analysis complete.")
