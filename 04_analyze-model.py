# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model Explainability & Statistical Alignment
# This script uses SHAP (SHapley Additive exPlanations) to understand the directionality 
# of feature impacts for the Random Forest model and aligns these findings with the 
# GLMM results from the R analysis.
#
# %% [code]
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

# %% [code]
data_dir = Path("data/processed")
plots_dir = Path("output")
plots_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Load Data & Preprocessing Setup
# We maintain absolute consistency with the preprocessing used in the model training script.
#
# %% [code]
df = pl.read_parquet(data_dir / "analytic-dataset.parquet")

# %% [code]
print(f"Loaded analytic dataset: {df.shape}")

# %% [code]
id_cols = ["alias", "alias_filled", "observation"]
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "progression_to_death" and c not in id_cols]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and c != "progression_to_death"]

# %% [code]
print(
    f"Feature set identified:\n"
    f"  Numeric: {len(numeric_cols)}\n"
    f"  Categorical: {len(categorical_cols)}"
)

# %% [code]
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

# %% [markdown]
# ## 2. Data Splitting (Consistency Check)
# We use the same GroupShuffleSplit parameters as `03_model.py` to ensure we are explaining the same training set.
#
# %% [code]
groups = df["alias_filled"].to_numpy()
x_df = df.select(numeric_cols + categorical_cols).to_pandas()
y = df["progression_to_death"].to_numpy()

# %% [code]
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(x_df, y, groups))

# %% [code]
x_train, x_test = x_df.iloc[train_idx], x_df.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]

# %% [code]
print(
    f"""
Data split for SHAP analysis:
  Train size: {x_train.shape}
  Test size: {x_test.shape}
  Patients in train: {len(np.unique(groups_train))}
"""
)

# %% [markdown]
# ## 3. Fit Best Random Forest
# We re-fit the RF using the optimal hyperparameters identified during nested CV.
#
# %% [code]
# Hyperparams from CV: n_estimators=200, max_depth=20, min_samples_split=2
rf_pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, 
                                   random_state=42, class_weight="balanced"))
])

# %% [code]
rf_pipeline.fit(x_train, y_train)

# %% [code]
print("✓ Random Forest model fitted for explainability.")

# %% [markdown]
# ## 4. SHAP Analysis
# We transform the training data and use TreeExplainer to calculate feature contributions.
#
# %% [code]
x_train_transformed = rf_pipeline.named_steps['pre'].transform(x_train)
cat_features = rf_pipeline.named_steps['pre'].transformers_[1][1].get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(cat_features)

# %% [code]
x_train_transformed_df = pd.DataFrame(x_train_transformed, columns=all_feature_names)

# %% [code]
print(
    f"Transformed feature matrix created:\n"
    f"  Rows: {x_train_transformed_df.shape[0]}\n"
    f"  Cols: {x_train_transformed_df.shape[1]}"
)

# %% [code]
explainer = shap.TreeExplainer(rf_pipeline.named_steps['clf'])
shap_values = explainer.shap_values(x_train_transformed_df)

# %% [code]
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_class1 = shap_values[:, :, 1]
else:
    shap_values_class1 = shap_values

# %% [markdown]
# ## 5. SHAP Summary Visualization
#
# %% [code]
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_class1, x_train_transformed_df, show=False)
plt.title("SHAP Summary: Feature Impact on Positive Outcome (Death)")
plt.tight_layout()
plt.savefig(plots_dir / "rf_shap_summary.png")
plt.show()

# %% [code]
print(f"✓ SHAP summary plot saved to {plots_dir / 'rf_shap_summary.png'}")

# %% [markdown]
# ## 6. GLMM Alignment (Forest Plot)
# We load the pooled mixed-effects results from the R analysis to verify that the model's 
# drivers align with statistically unbiased clinical estimates.
#
# %% [code]
glmm_results_path = data_dir / "feature_analysis.csv"

# %% [code]
if glmm_results_path.exists():
    glmm_df = pd.read_csv(glmm_results_path)
    
    # Compute Odds Ratios from coefficients
    glmm_df['OR'] = np.exp(glmm_df['estimate'])
    glmm_df['lower_CI'] = np.exp(glmm_df['ci_low'])
    glmm_df['upper_CI'] = np.exp(glmm_df['ci_high'])
    
    plot_df = glmm_df.sort_values('OR')

    plt.figure(figsize=(10, 12))
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    colors = ['#d62728' if p < 0.05 else '#7f7f7f' for p in plot_df['p_value']]

    # Plotting error bars (grey)
    plt.errorbar(
        x=plot_df['OR'], 
        y=plot_df['feature'], 
        xerr=[plot_df['OR'] - plot_df['lower_CI'], plot_df['upper_CI'] - plot_df['OR']],
        fmt='none', 
        color='grey', 
        alpha=0.5,
        capsize=3
    )

    # Plotting points (colored by significance)
    plt.scatter(
        x=plot_df['OR'], 
        y=plot_df['feature'], 
        c=colors, 
        s=30, 
        edgecolors='black', 
        zorder=3
    )

    plt.xscale('log')
    plt.title("Mixed Effects Model: Odds Ratios (95% CI)\nRed = p < 0.05", fontsize=14)
    plt.xlabel("Odds Ratio (Log Scale)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "glmm_forest_plot.png")
    plt.show()
    
    print(f"✓ GLMM Forest plot saved to {plots_dir / 'glmm_forest_plot.png'}")
    
    print("\nTop Mixed Effects Odds Ratios:")
    print(plot_df[['feature', 'OR', 'p_value']].head(10).to_string(index=False))
else:
    print("ERROR: GLMM results not found. Please run 02_analyze-data.R first.")

# %% [markdown]
# ## 7. Feature Directionality Analysis
# We calculate the correlation between SHAP values and feature values to determine 
# if a feature is a "Risk Factor" or "Protective Factor".
#
# %% [code]
directions = []
for i, col in enumerate(all_feature_names):
    feat_vals = x_train_transformed_df[col].values
    s_vals = shap_values_class1[:, i]
    
    corr = np.corrcoef(feat_vals, s_vals)[0, 1]
    importance = np.abs(s_vals).mean()
    
    if np.isnan(corr): 
        direction = "Neutral/Non-linear"
    elif corr > 0: 
        direction = "Risk Factor (Increases Death Prob)"
    else: 
        direction = "Protective Factor (Decreases Death Prob)"
    
    directions.append({
        "feature": col,
        "importance": importance,
        "correlation": corr,
        "direction": direction
    })

# %% [code]
dir_df = pd.DataFrame(directions).sort_values("importance", ascending=False)
dir_df.to_csv(plots_dir / "rf_feature_directions.csv", index=False)

# %% [code]
print(f"✓ Feature directions saved to {plots_dir / 'rf_feature_directions.csv'}")

# %% [code]
print("\nTop Feature Directions:")
print(dir_df.head(20).to_string(index=False))
