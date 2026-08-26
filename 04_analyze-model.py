# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
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

id_cols = ["alias", "alias_filled", "observation"]
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "progression_to_death" and c not in id_cols]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and c != "progression_to_death"]

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

# Use same split as in CV script
groups = df["alias_filled"].to_numpy()
X_df = df.select(numeric_cols + categorical_cols).to_pandas()
y = df["progression_to_death"].to_numpy()

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(X_df, y, groups))

X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]

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

# %% [markdown]
# ## 3. SHAP Analysis

# %% [code]
# SHAP needs the transformed features
X_train_transformed = rf_pipeline.named_steps['pre'].transform(X_train)
cat_features = rf_pipeline.named_steps['pre'].transformers_[1][1].get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(cat_features)

# Create a DataFrame for SHAP
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=all_feature_names)

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
# ## 4. Visualization & Directionality

# %% [code]
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_class1, X_train_transformed_df, show=False)
plt.title("SHAP Summary: Feature Impact on Class 1 (Positive Outcome)")
plt.tight_layout()
plt.savefig(plots_dir / "rf_shap_summary.png")
plt.close()

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
    
    direction = "Positive (Helpful)" if corr > 0 else "Negative (Harmful)"
    if np.isnan(corr): direction = "Neutral/Non-linear"
    
    directions.append({
        "feature": col,
        "importance": importance,
        "correlation": corr,
        "direction": direction
    })

dir_df = pd.DataFrame(directions).sort_values("importance", ascending=False)
dir_df.to_csv(plots_dir / "rf_feature_directions.csv", index=False)

print("\nTop Feature Directions:")
print(dir_df.head(20).to_string(index=False))
