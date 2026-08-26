# Remaining Work: Data Pipeline and Modeling

## 00_process-data.py
- [ ] **Fix Concat Risk**: Use `how="diagonal"` in `pl.concat` to prevent crash on column mismatch.
- [ ] **Robust Type Cast**: Add error handling/validation for `str.to_datetime` and `cast(pl.Int64)`.

## 01_explore-data.py
- [ ] **Refine Leakage Heuristic**: Replace crude 95% accuracy threshold with a more robust method.
- [ ] **Fix Fill Order**: Move median filling after leak check to avoid masking patterns.

## 03_model.py
- [ ] **Optimize CV**: Remove redundant `GridSearchCV` in final test set validation. Use best params from nested CV.

## 04_analyze-model.py
- [ ] **Fix Directionality Analysis**: Replace `np.corrcoef` with a method that handles non-linear SHAP effects.
- [ ] **Remove Param Hardcoding**: Load RF hyperparameters from an artifact instead of hardcoding.



Manual nitpicks:
- Don't "manually" number sections no reason, just gets out of date, etc
- Does this account for the fact we have repeated measures?
    print("Class Distribution:")
    print(df.group_by("progression_to_death").agg(pl.len().alias("count")).sort("progression_to_death"))
- I don't see the visualization in this section: Missingness Heatmap
- Get the Ophthalmology stuff out of here - them models keep latching on to this, there is NOTHING related to it here I linked to another project for a TEMPLATE for structure. MUST GO.
- Use seed 8675309
    - Does MICE need it? Put it anywhere it's not clear it ISN'T needed
- Section called "Step 0", again non of this
- Pooled GLMM Analysis - this section doesn't use the transparent jupytext structure hardly at all PRINT all the stuff for me to read
- 6. Mixed Effects Forest Plot <- numbers again
- 7. Correlation Heatmap (Patient Level) - numbers and I don't see the plot
- df = pl.read_parquet(data_dir / "analytic-dataset.parquet") Are we using parquet or CSV? Whatever it is, be consistent
- Feature Importance (Random Forest) <- is this using the best params we established in teh grid or repeating grid? Why if so?
- Why are we seeing these again?
✓ Feature importance plot saved to output/rf_feature_importance.png

Evaluating Logistic Regression on Test Set...

Test Metrics for Logistic Regression:
  Accuracy  : 0.847
  Precision : 0.912
  Recall    : 0.799
  F1 Score  : 0.852
- Are fitting a whole new set of models in the explainability script? Why? Save them and load them

print(
    f"Feature set identified:\n"
    f"  Numeric: {len(numeric_cols)}\n"
    f"  Categorical: {len(categorical_cols)}"
)

Feature set identified:
  Numeric: 13
  Categorical: 8
numeric_transformer = Pipeline([
    ("imputer", IterativeImputer(random_state=42)),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

- Are we repeating this that's mostly the same as the analysis script?
6. GLMM Alignment (Forest Plot)
We load the pooled mixed-effects results from the R analysis to verify that the model’s drivers align with statistically unbiased clinical estimates.
- These need to be visible in the report, all the plots do
dir_df = pd.DataFrame(directions).sort_values("importance", ascending=False)
dir_df.to_csv(plots_dir / "rf_feature_directions.csv", index=False)
