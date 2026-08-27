
## 00_process-data.py
- [x] **Fix Concat Risk**: Used `how="diagonal"` in `pl.concat` to prevent crash on column mismatch.
- [x] **Robust Type Cast**: Implemented `safe_cast` helper for validated datetime and integer casting.

## 01_explore-data.py
- [x] **Refine Leakage Heuristic**: Replaced accuracy threshold with Mutual Information (MI) score.
- [x] **Fix Fill Order**: Implemented constant filling for leakage check to avoid masking patterns.
- [x] **Repeated Measures**: Added patient-level counts to class distribution report.

## 03_model.py
- [x] **Optimize CV**: Removed redundant `GridSearchCV` in final validation; now uses consensus parameters from nested CV.
- [x] **Model Persistence**: Save trained models as `.joblib` artifacts.

## 04_analyze-model.py
- [x] **Fix Directionality Analysis**: Replaced `np.corrcoef` with Median-split SHAP difference to handle non-linear effects.
- [x] **Remove Param Hardcoding**: Now loads models and parameters from artifacts.

## General Cleanup
- [x] **Sectioning**: Removed manual numbering (e.g., "## 1. ...") from all files.
- [x] **Terminology**: Removed all "Ophthalmology" references; updated to "Clinical Cases".
- [x] **Seed**: Updated all `random_state` to `8675309`.
- [x] **DRY Preprocessing**: Moved shared `ColumnTransformer` logic to `utils.py`.

## Adversarial Review (Technical Audit)

### Technical Debt & Risks
- [ ] **Suboptimal Tuning Metric**: `GridSearchCV` currently uses `scoring="accuracy"`. Given class imbalance (implied by `scale_pos_weight`), should switch to `roc_auc` or `f1`.
- [ ] **Unconventional Parameter Selection**: "Consensus parameters" (mode of best params across folds) used for final fit. Standard practice is re-tuning on full train set or using a hold-out.
- [ ] **Documentation Gap**: `README.md` is empty. Project relies on `spec.md` which is geared towards agent instructions rather than users.
- [ ] **Environmental Coupling**: Hardcoded `Path("data")` and `Path("output")` limits portability.
- [ ] **Seed Redundancy**: `8675309` hardcoded in 6+ locations; should be centralized.
- [ ] **Consensus Tree Leakage**: Surrogate tree fit on training data using in-sample predictions. Mimics RF overfit, not general logic. Needs OOF predictions and group-aware validation to handle repeated observations.
