# Updated Adversarial Code Review Report

## ✅ RESOLVED ISSUES

### 01_explore-data.py
- **Fixed**: O(N×M) missingness calculation $\rightarrow$ single scan `null_count()`.
- **Fixed**: Missing plot directory check $\rightarrow$ `mkdir(parents=True)`.
- **Fixed**: OOM risk $\rightarrow$ added memory warning for large matrices.
- **Fixed**: Duplicate parquet write removed.

### 02_model-cross-validation.py
- **Fixed**: Non-stratified CV $\rightarrow$ `StratifiedKFold`.
- **Fixed**: Data leakage risk $\rightarrow$ Added keyword-based leakage check.
- **Fixed**: Feature scaling $\rightarrow$ Added `StandardScaler` pipeline for Logistic Regression.
- **Fixed**: Class imbalance $\rightarrow$ Added `class_weight="balanced"`.
- **Fixed**: XGBoost defaults $\rightarrow$ Added `max_depth` regularization.
- **Fixed**: Reproducibility $\rightarrow$ Added `random_state` to all models.
- **Fixed**: Stratification failure risk $\rightarrow$ Added min-sample count check.

---

## 🚨 REMAINING ISSUES

### CRITICAL (Data Integrity/Correctness) - 00_process-dataset.py
- **Silent type cast failures**: `.cast(pl.Int64)` on string columns crashes on invalid data without context (Lines 167-189).
- **No validation for negative durations**: No check for negative values in time/duration columns after casting (After line 195).

### MAJOR (Correctness/Performance) - 00_process-dataset.py
- **Duplicate column name risk**: `clean_colnames()` regex strips special chars silently; e.g., `"A/B"` and `"A-B"` both become `"ab"` (Line 82).
- **Inefficient empty row check**: `pl.all_horizontal(pl.all().is_null())` scans all columns twice (Lines 95-96).
- **Silent Excel Read Failures**: `load_raw()` has no error handling for missing or corrupt files (Lines 11-12).

### MINOR (Maintainability & Portability)
| File | Issue | Location |
|------|-------|----------|
| All | Hardcoded absolute paths | `data_dir = Path("/home/thadryan/...")` |
| All | No `__main__` guard | Scripts execute immediately on import |
| All | Print-only output | No `logging` module for levels/timestamps |
| 02_model-cross-validation.py | Redundant import | `from sklearn.metrics import...` inside loop (Line 88) |

### STYLE VIOLATIONS (spec.md)
| Issue | Files | Spec Requirement |
|-------|-------|------------------|
| No `uv run` shebang | All | "run commands need to be written `uv run {whatever}`" |

---

## SUMMARY

| Severity | Remaining | Status |
|----------|-----------|---------|
| **Critical** | 2 | Only in `00_process-dataset.py` |
| **Major** | 3 | Only in `00_process-dataset.py` |
| **Minor** | 4 | Across all files |
| **Style** | 1 | Across all files |

**Next priority:** Fix `00_process-dataset.py` and resolve absolute paths.
