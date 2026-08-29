# Finer Points — Tomorrow

Plumbing is verified clean (grouping, inner-CV routing, in-fold preprocessing). These are the judgment calls and loose ends.

## Resolve before anything goes in a table
- [ ] **Feature 4 direction flip.** Full RF says Protective, first-look says Risk. Check the SHAP *dependence* plot (not the median-split summary) for non-monotonicity — a U-shaped vital will flip sign under a median split by construction. Cross-reference the GLMM / first-look GLM sign as tiebreaker. If CIs cross zero, report as "direction not established," don't pick a side.
- [ ] **Singularity diagnostics.** Open `feature_analysis.csv`, read `frac_singular` / `frac_nonconv`. Non-trivial → that feature's OR/CI is untrustworthy; flag or drop from the forest plot. Decides whether the inferential headline is real.
- [ ] **Two-batch extraction sanity check.** Compare a couple of should-be-identical distributions across the two pulls (age, a stable vital's units/precision). If they line up, that's evidence the pulls agree and the footnote softens accordingly.

## Small code fixes
- [ ] Unify significance criterion — `04` colors by raw `p_value`, `02` by `p_adj`. Use FDR in both.
- [ ] Inner `GridSearchCV` `scoring="accuracy"` → AUC or average precision (contradicts `class_weight="balanced"` as-is).
- [ ] Outer folds 10 → repeated 5-fold, so folds aren't resting on ~3 positive patients.
- [ ] Trim RF/XGB grids — at this effective N the search is fitting selection noise.
- [ ] Verify row order == observation order within patient (or sort explicitly). "First look" silently depends on it.

## Framing / write-up
- [ ] Promote first-look to primary; full-dataset as optimistic upper bound, with the gap reported as a leakage diagnostic.
- [ ] State the 3-of-4 cross-design agreement as a robustness *result* — evidence against late-observation leakage, stronger than asserting leakage was controlled.
- [ ] Note SHAP is in-sample on training data and ranks are noisy at this N — "High agreement" = directional corroboration, not a precise ordering.
- [ ] Label the GLMM/GLM ORs as **unadjusted marginal** (univariate per-feature screen + FDR), not mutually adjusted effects.
- [ ] Describe the two imputation strategies separately (in-fold `IterativeImputer` for ML; MICE m=5 + Rubin for inference). Optionally pool the first-look GLM across imputations for consistency.
- [ ] Patient-constant outcome → say plainly that the patient-level GLM is the proper inferential model; show GLMM only if diagnostics are clean.

## If the go/no-go rule is a real deliverable
- [ ] Threshold from the clinical cost ratio (missed vs unnecessary send), not 0.5/F1.
- [ ] Calibration + net-benefit at that operating point; report in patient counts.
- [ ] Patient-level bootstrap for split/cutpoint stability before anything gets laminated.

## Batch only if already in the file
- [ ] Dead code: unused `train_test_split`, unused `scoring` dict, `df_analytic`, redundant imports in `03`.
- [ ] `04` re-derives the split to rebuild `x_train` — fragile across data/version changes.
- [x] MI block prints its value labeled `Accuracy=`.
- [ ] Outer/inner `train_idx`/`test_idx` shadowing in `03` (no active bug, but a landmine).
