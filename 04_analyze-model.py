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
# pooled logistic regression results from the R analysis.
#
# %% [code]
import sys
import polars as pl
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib

from sklearn.model_selection import GroupShuffleSplit

# %% [code]
# Default to 'first_look', but allow override via command line
prefix = "first_look"
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg in ["full", "first_look"]:
        prefix = arg

print(f"Analyzing model prefix: {prefix}")
data_dir = Path("data/processed")
plots_dir = Path("output")
plots_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load Data & Preprocessing Setup
# We maintain absolute consistency with the preprocessing used in the model training script.
#
# %% [code]
df = pl.read_parquet(data_dir / "analytic-dataset.parquet")

if prefix == "first_look":
    df = df.filter(pl.col("observation") == 1)

print(f"Loaded analytic dataset: {df.shape}")

# %% [code]
id_cols = ["alias", "alias_filled", "observation"]
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "progression_to_death" and c not in id_cols]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and c != "progression_to_death" and c not in id_cols]

# %% [code]
print(
    f"Feature set identified:\n"
    f"  Numeric: {len(numeric_cols)}\n"
    f"  Categorical: {len(categorical_cols)}"
)

# %% [markdown]
# # Data Splitting (Consistency Check)
# We use the same GroupShuffleSplit parameters as `03_model.py` to ensure we are explaining the same training set.
#
# %% [code]
groups = df["alias_filled"].to_numpy()
x_df = df.select(numeric_cols + categorical_cols).to_pandas()
y = df["progression_to_death"].to_numpy()

# %% [code]
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=8675309)
train_idx, _ = next(gss.split(x_df, y, groups))

# %% [code]
x_train = x_df.iloc[train_idx]
y_train = y[train_idx]
groups_train = groups[train_idx]

# %% [code]
print(
    f"""
Data split for SHAP analysis:
  Train size: {x_train.shape}
  Patients in train: {len(np.unique(groups_train))}
"""
)

# %% [markdown]
# # Load Model
# We load the final Random Forest model from artifact.
#
# %% [code]
# Load the final Random Forest model from artifact
model_filename = f"{prefix}_random_forest_model.joblib"
model_path = data_dir / model_filename
if model_path.exists():
    rf_pipeline = joblib.load(model_path)
    print(f"✓ Loaded Random Forest model from {model_path}")
else:
    print(f"ERROR: RF model artifact {model_filename} not found. Please run 03_model.py first.")
    raise FileNotFoundError(f"Could not find {model_path}")

# %% [code]
print("✓ Random Forest model ready for explainability.")

# %% [markdown]
# # SHAP Analysis
# We transform the training data and use TreeExplainer to calculate feature contributions.
# SHAP is computed on the training split to provide a global explanation of the model's
# learned logic.
#
# %% [code]
x_train_transformed = rf_pipeline.named_steps['pre'].transform(x_train)
all_feature_names = rf_pipeline.named_steps['pre'].get_feature_names_out()

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
# # SHAP Summary Visualization
#
# %% [code]
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_class1, x_train_transformed_df, show=False)
plt.title(f"SHAP Summary: Feature Impact on Outcome ({prefix})")
plt.tight_layout()
plt.savefig(plots_dir / f"rf_shap_summary_{prefix}.png")
plt.close()

# %% [code]
print(f"✓ SHAP summary plot saved to {plots_dir / f'rf_shap_summary_{prefix}.png'}")

# %% [markdown]
# # Pooled Logistic Regression Alignment (Forest Plot)
# We load the results from the R analysis to verify that the model's
# drivers align with pooled clinical estimates.
#
# %% [code]
results_path = data_dir / ("first_look_analysis.csv" if prefix == "first_look" else "feature_analysis.csv")

# %% [code]
if results_path.exists():
    results_df = pd.read_csv(results_path)

    # Compute Odds Ratios from coefficients
    results_df['OR'] = np.exp(results_df['estimate'])
    results_df['lower_CI'] = np.exp(results_df['ci_low'])
    results_df['upper_CI'] = np.exp(results_df['ci_high'])

    plot_df = results_df.sort_values('OR')

    plt.figure(figsize=(10, 12))
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)

    colors = ['#d62728' if p < 0.05 else '#7f7f7f' for p in plot_df['p_adj']]

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
    plt.title("Pooled Logistic Regression: Odds Ratios (95% CI)\nRed = p_adj < 0.05", fontsize=14)
    plt.xlabel("Odds Ratio (Log Scale)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / f"glmm_forest_plot_{prefix}.png")
    plt.close()

    print(f"✓ Pooled Logistic Regression Forest plot saved to {plots_dir / f'glmm_forest_plot_{prefix}.png'}")

    print("\nTop Pooled Odds Ratios:")
    print(plot_df[['feature', 'OR', 'p_adj']].head(10).to_string(index=False))
else:
    print("ERROR: Results not found. Please run 02_analyze-data.R first.")

# %% [markdown]
# # Feature Directionality Analysis
# We calculate the correlation between SHAP values and feature values to determine
# if a feature is a "Risk Factor" or "Protective Factor".
#
# %% [code]
directions = []
for i, col in enumerate(all_feature_names):
    feat_vals = x_train_transformed_df[col].values
    s_vals = shap_values_class1[:, i]

    # Handle non-linear effects: compare SHAP values for high vs low feature values
    # We use the median as the split point
    median_val = np.median(feat_vals)
    high_mask = feat_vals > median_val
    low_mask = feat_vals <= median_val

    if not high_mask.any() or not low_mask.any():
        avg_shap_high = 0
        avg_shap_low = 0
    else:
        avg_shap_high = np.mean(s_vals[high_mask])
        avg_shap_low = np.mean(s_vals[low_mask])

    diff = avg_shap_high - avg_shap_low
    importance = np.abs(s_vals).mean()

    if np.abs(diff) < 1e-5:
        direction = "Neutral/Non-linear"
    elif diff > 0:
        direction = "Risk Factor (Increases Death Prob)"
    else:
        direction = "Protective Factor (Decreases Death Prob)"

    directions.append({
        "feature": col,
        "importance": importance,
        "diff_high_low": diff,
        "direction": direction
    })

# %% [code]
dir_df = pd.DataFrame(directions).sort_values("importance", ascending=False)
dir_df.to_csv(plots_dir / f"rf_feature_directions_{prefix}.csv", index=False)

print("\nTop Feature Directions:")
print(dir_df.head(10).to_string(index=False))
