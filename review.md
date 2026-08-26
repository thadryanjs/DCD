# Code Review: `01_explore-data.py` & `02_analyze-data.R`

---

## CORRECTNESS (Bugs - Wrong Results)

### Python (`01_explore-data.py`)

| Issue | Location | Severity |
|-------|----------|----------|
| `.describe()` used instead of `.corr()` for correlation matrix | Line ~170 | **Critical** |
| Silent null handling: `if mean_pos is None: mean_pos = 0.0` | Line ~85 | High |
| Missingness heatmap warning but no actual downsampling logic | Line ~145 | High |
| Hardcoded absolute paths (`/home/thadryan/...`) | Line ~15 | Medium |

### R (`02_analyze-data.R`)

| Issue | Location | Severity |
|-------|----------|----------|
| No GLMM convergence status check | Line ~55 | **Critical** |
| No multiple testing correction (Bonferroni/FDR) | Line ~65 | **Critical** |
| Per-feature `drop_na()` creates inconsistent denominators | Line ~50 | High |
| R script skips Python's leakage detection step | N/A | High |
| Hardcoded relative paths (breaks if run from different CWD) | Line ~10 | Medium |

---

## EFFICIENCY (Performance - Slow/Memory)

### Python (`01_explore-data.py`)

| Issue | Location | Impact |
|-------|----------|--------|
| Per-column mean difference: 2N table scans | Line ~80-90 | High |
| Missingness comparison: 4N table scans | Line ~115-125 | High |
| `.to_dicts()` + Python loop for filtering | Line ~210 | Medium |
| `.row(0)` + Python loop for missing stats | Line ~105 | Medium |
| No lazy evaluation (all eager operations) | Throughout | Medium |
| Per-column schema access in list comprehension | Line ~75 | Low |

### R (`02_analyze-data.R`)

| Issue | Location | Impact |
|-------|----------|--------|
| `map()` + `glmer()` per feature (no parallelization) | Line ~65 | Medium |
| Patient-level aggregation then raw data for GLMM (inconsistent) | Line ~40, ~50 | Low |

---

## STYLE (Readability - Conventions)

### Python (`01_explore-data.py`)

| Issue | Location | Fix |
|-------|----------|-----|
| `import pandas as pd` mid-script (line ~220) | Line ~220 | Move to top |
| `import numpy as np` unused | Line ~10 | Remove |
| `sklearn` imports split across file | Lines ~20, ~220 | Consolidate at top |
| `pl.col()` verbose vs `from polars import col` | Throughout | Add `from polars import col` |
| Hardcoded absolute paths | Line ~15 | Use relative paths or env vars |

### R (`02_analyze-data.R`)

| Issue | Location | Fix |
|-------|----------|-----|
| `p_load()` hides missing dependencies | Line ~5 | Use explicit `library()` calls |
| Hardcoded relative paths | Line ~10 | Use `here::here()` or env vars |

---

## CROSS-LANGUAGE ISSUES

| Issue | Python | R |
|-------|--------|---|
| Hardcoded paths | ✅ | ✅ |
| Silent failures | ✅ | ✅ |
| Leakage detection | ✅ (skipped by R) | ❌ |

---

## REMEDIATION PLAN

### Phase 1: Correctness (Must Fix)
1. [ ] Python: Change `.describe()` to `.corr()` for correlation matrix
2. [ ] Python: Add actual downsampling when missingness matrix > threshold
3. [ ] R: Add GLMM convergence check
4. [ ] R: Add Bonferroni/FDR correction to p-values
5. [ ] R: Use consistent data source (aggregated or raw, not both)
6. [ ] R: Run leakage detection or document why skipped

### Phase 2: Efficiency (Should Fix)
1. [ ] Python: Replace per-column filters with single `group_by().mean()`
2. [ ] Python: Replace `.to_dicts()` + loop with `filter().get_column()`
3. [ ] Python: Add lazy evaluation wrapper
4. [ ] R: Consider parallel GLMM execution

### Phase 3: Style (Nice to Fix)
1. [ ] Python: Consolidate imports at top
2. [ ] Python: Remove unused `numpy` import
3. [ ] Python: Add `from polars import col`
4. [ ] Python: Fix hardcoded paths
5. [ ] R: Replace `p_load()` with explicit `library()` calls
6. [ ] R: Fix hardcoded paths

---

## NOTES

- **Spec compliance**: Type hints not required per `spec.md`
- **Jupytext**: Both scripts use percent format correctly
- **uv**: Python scripts should use `uv run` for execution
- **Data leakage**: `spec.md` requires `StratifiedGroupKFold` - verify downstream scripts comply