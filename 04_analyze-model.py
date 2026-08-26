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
# # Model Explainability: SHAP Analysis for Random Forest
#
# ## Goal
# Use SHAP (SHapley Additive exPlanations) to understand not just which features are important, 
# but how they influence the model's prediction (directionality).

# %% [code]
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

data_dir = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data/processed")
plots_dir = Path("output")
plots_dir.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 1. Load Data & Setup Preprocessing

# %% [code]
df = pl.read_parquet(data_dir / "model-ready-dataset.parquet")
print(f"Loaded model-ready dataset: {df.shape}")


# %% [code]
id_cols = ["alias", "alias_filled", "observation"]
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "progression_to_death" and c not in id_cols]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and c != "progression_to_death"]


# %% [code]
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
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


# %% [code]
# Use same split as in CV script
groups = df["alias_filled"].to_numpy()
X_df = df.select(numeric_cols + categorical_cols).to_pandas()
y = df["progression_to_death"].to_numpy()

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(X_df, y, groups))

X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]

print(
    f"""
Data split for SHAP:
  Train size: {X_train.shape}
  Test size: {X_test.shape}
"""
)


# %% [markdown]
# ## 2. Fit Best Random Forest
# We re-fit the best RF parameters identified in the CV process.

# %% [code]
# Params from previous run: n_estimators=200, max_depth=20, min_samples_split=2
rf_pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, 
                                   random_state=42, class_weight="balanced"))
])

rf_pipeline.fit(X_train, y_train)
print("Random Forest model fitted.")


# %% [markdown]
# ## 3. SHAP Analysis

# %% [code]
# SHAP needs the transformed features
X_train_transformed = rf_pipeline.named_steps['pre'].transform(X_train)
cat_features = rf_pipeline.named_steps['pre'].transformers_[1][1].get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(cat_features)

# Create a DataFrame for SHAP
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=all_feature_names)
print(f"Transformed feature matrix shape: {X_train_transformed_df.shape}")


# %% [code]
# Initialize TreeExplainer
explainer = shap.TreeExplainer(rf_pipeline.named_steps['clf'])
shap_values = explainer.shap_values(X_train_transformed_df)

# For binary classification, shap_values is often a list [class0, class1]. We want class 1.
if isinstance(shap_values, list):
    # Some versions of SHAP return list for RF
    shap_values_class1 = shap_values[1]
elif len(shap_values.shape) == 3:
    # Some versions return (samples, features, classes)
    shap_values_class1 = shap_values[:, :, 1]
else:
    # Single array returned (already class 1)
    shap_values_class1 = shap_values


# %% [markdown]
# ## 5. Visualization & Directionality

# %% [code]
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_class1, X_train_transformed_df, show=False)
plt.title("SHAP Summary: Feature Impact on Class 1 (Positive Outcome)")
plt.tight_layout()
plt.savefig(plots_dir / "rf_shap_summary.png")
plt.show()


# %% [markdown]
# ## 6. Mixed Effects Forest Plot
# We leverage the results from the Generalized Linear Mixed Models (GLMM) conducted in the R analysis.
# This accounts for repeated observations per patient to provide unbiased Odds Ratios.

# %% [code]
# Load GLMM results from R analysis
glmm_results_path = data_dir / "feature_analysis.csv"
if glmm_results_path.exists():
    glmm_df = pd.read_csv(glmm_results_path)
    
    # Calculate Odds Ratios and CIs from coefficients
    glmm_df['OR'] = np.exp(glmm_df['estimate'])
    glmm_df['lower_CI'] = np.exp(glmm_df['ci_low'])
    glmm_df['upper_CI'] = np.exp(glmm_df['ci_high'])
    
    plot_df = glmm_df.sort_values('OR')

    plt.figure(figsize=(10, 12))
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    # Color by p-value
    colors = ['#d62728' if p < 0.05 else '#7f7f7f' for p in plot_df['p_value']]

    plt.errorbar(
        x=plot_df['OR'], 
        y=plot_df['feature'], 
        xerr=[plot_df['OR'] - plot_df['lower_CI'], plot_df['upper_CI'] - plot_df['OR']],
        fmt='o', 
        color=colors, 
        markersize=6, 
        capsize=3,
        markeredgecolor='black'
    )

    plt.xscale('log')
    plt.title("Mixed Effects Model: Odds Ratios (95% CI)\nRed = p < 0.05", fontsize=14)
    plt.xlabel("Odds Ratio (Log Scale)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "glmm_forest_plot.png")
    plt.show()
    
    print(f"GLMM Forest plot saved to {plots_dir / 'glmm_forest_plot.png'}")
    print("\nTop Mixed Effects Odds Ratios:")
    print(plot_df[['feature', 'OR', 'p_value']].head(10).to_string(index=False))
else:
    print("GLMM results not found. Please run 02_analyze-data.R first.")


# %% [code]
# Calculate mean absolute SHAP and sign of correlation to determine direction
# Direction = sign(correlation(shap_value, feature_value))
directions = []
for i, col in enumerate(all_feature_names):
    feat_vals = X_train_transformed_df[col].values
    s_vals = shap_values_class1[:, i]
    
    # Correlation between feature value and its SHAP value
    corr = np.corrcoef(feat_vals, s_vals)[0, 1]
    
    # Magnitude of impact
    importance = np.abs(s_vals).mean()
    
    direction = "Risk Factor (Increases Death Prob)" if corr > 0 else "Protective Factor (Decreases Death Prob)"
    if np.isnan(corr): direction = "Neutral/Non-linear"
    
    directions.append({
        "feature": col,
        "importance": importance,
        "correlation": corr,
        "direction": direction
    })

dir_df = pd.DataFrame(directions).sort_values("importance", ascending=False)
dir_df.to_csv(plots_dir / "rf_feature_directions.csv", index=False)


# %% [code]
print("\nTop Feature Directions:")
dir_df.head(20)
