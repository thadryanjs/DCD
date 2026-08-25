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
# # Data Exploration
# Exploratory analysis with detailed missingness patterns.
#
# %% [code]
import polars as pl
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

data_dir = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data")
processed_dir = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data/processed")

# %% [markdown]
# ## 1. Load Processed Dataset
#
# %% [code]
try:
    df = pl.read_parquet(processed_dir / "combined-dataset.parquet")
except FileNotFoundError:
    raise FileNotFoundError(
        "combined-dataset.parquet not found. Run 00_process-dataset.py first."
    )

print(f"Dataset loaded: {df.shape}")

# %% [markdown]
# ## 2. Class Distribution
#
# %% [code]
print("Class Distribution:")
print(df.group_by("label").agg(pl.len().alias("count")).sort("label"))

# %% [code]
print(
    f"""
Class Balance:
  Positive (1): {df.filter(pl.col("label") == 1).height} rows
  Negative (0): {df.filter(pl.col("label") == 0).height} rows
  Total: {df.height} rows
  Positive Rate: {df.filter(pl.col("label") == 1).height / df.height * 100:.1f}%
"""
)

# %% [markdown]
# ## 3. Numeric Features: Mean Differences by Class
#
# %% [code]
# Get numeric and categorical columns (exclude label and temporal/datetimes)
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "label"]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and not df.schema[c].is_temporal() and c != "label"]
candidate_cols = numeric_cols + categorical_cols

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Total candidate features: {len(candidate_cols)}")

# %% [code]
# Calculate absolute differences
diffs = []
for col in numeric_cols:
    mean_pos = df.filter(pl.col("label") == 1).select(pl.col(col).mean()).item()
    mean_neg = df.filter(pl.col("label") == 0).select(pl.col(col).mean()).item()
    # Handle null means (all null values in a class)
    if mean_pos is None:
        mean_pos = 0.0
    if mean_neg is None:
        mean_neg = 0.0
    diff = abs(mean_pos - mean_neg)
    diffs.append((col, mean_pos, mean_neg, diff))

# Sort by difference (descending)
diffs.sort(key=lambda x: x[3], reverse=True)

print("Mean differences by class (sorted by magnitude):")
for col, mean_pos, mean_neg, diff in diffs:
    print(f"  {col:30s}: Pos={mean_pos:10.3f}, Neg={mean_neg:10.3f}, Diff={diff:8.3f}")

# %% [markdown]
# ## 4. Missingness Analysis
# Deep dive into missing data patterns.
#
# %% [code]
# Missing count and percentage per column (efficient: single scan with null_count())
missing_stats = df.select([
    pl.col(c).null_count().alias(f"{c}_nulls") for c in df.columns
]).row(0)

columns = df.columns
missing_stats = [(col, missing_stats[i], missing_stats[i] / df.height * 100)
                 for i, col in enumerate(columns)]

# Sort by missing percentage (descending)
missing_stats.sort(key=lambda x: x[2], reverse=True)

print("Missingness by Column (sorted by % missing):")
print("-" * 60)
for col, count, pct in missing_stats:
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"{col:30s} |{bar}| {count:4d} ({pct:5.1f}%)")

# %% [code]
# Missingness by class (compare positive vs negative)
print("\nMissingness Comparison: Positive vs Negative Class")
print("=" * 70)
for col in df.columns:
    pos_missing = df.filter(pl.col("label") == 1, pl.col(col).is_null()).height
    neg_missing = df.filter(pl.col("label") == 0, pl.col(col).is_null()).height
    pos_total = df.filter(pl.col("label") == 1).height
    neg_total = df.filter(pl.col("label") == 0).height

    pos_pct = pos_missing / pos_total * 100 if pos_total > 0 else 0
    neg_pct = neg_missing / neg_total * 100 if neg_total > 0 else 0

    if pos_missing > 0 or neg_missing > 0:
        print(f"{col:30s}: Pos={pos_pct:5.1f}% ({pos_missing}/{pos_total}), Neg={neg_pct:5.1f}% ({neg_missing}/{neg_total})")

# %% [code]
# Missingness pattern matrix (which columns are missing together)
print("\nMissingness Pattern Matrix (correlation of missingness):")
print("Columns that share missing values tend to have correlated patterns.")

# Create binary missing indicator dataframe
missing_mask = df.select([pl.col(c).is_null().cast(pl.Int8).alias(f"{c}_miss") for c in df.columns])

# Calculate correlation matrix of missingness
if len(numeric_cols) > 1:
    miss_cols = [f"{c}_miss" for c in numeric_cols]
    corr_matrix = missing_mask.select(miss_cols).describe()
    print(corr_matrix)

# %% [markdown]
# ### Visualization: Missingness Heatmap
#
# %% [code]
# Ensure plots directory exists
plots_dir = Path("output")
plots_dir.mkdir(parents=True, exist_ok=True)

# Create binary missingness matrix in original spreadsheet order (rows x features)
miss_cols = [f"{c}_miss" for c in numeric_cols]
missing_matrix = missing_mask.select(miss_cols).to_numpy()

# OOM warning for large datasets
if missing_matrix.size > 10_000_000:
    mem_gb = missing_matrix.nbytes / (1024**3)
    print(f"WARNING: Missingness matrix is {mem_gb:.2f} GB. Consider downsampling.")

# Use a strict 2-color map to prevent grey interpolation
# 0: White (Present), 1: Black (Missing)
binary_cmap = ListedColormap(["white", "black"])

plt.figure(figsize=(14, 10))
# Use pcolormesh via sns.heatmap but remove 'interpolation' 
# which is for imshow, not pcolormesh.
# rasterized=True helps with vector rendering artifacts.
sns.heatmap(missing_matrix, 
            cmap=binary_cmap, 
            cbar=False, 
            xticklabels=numeric_cols, 
            rasterized=True)
plt.title("Missingness Heatmap (Original Spreadsheet Order)")
plt.ylabel("Row Index")
plt.xlabel("Feature")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(plots_dir / "missingness-heatmap.png", dpi=150)
plt.show()
# %% [code]
# Empirical Verification: Missingness Distribution (Top-Left vs Bottom-Right)
mid_row = df.height // 2
mid_col = len(numeric_cols) // 2

# 1. Quadrant Check
top_left = df.slice(0, mid_row).select(numeric_cols[:mid_col])
bottom_right = df.slice(mid_row).select(numeric_cols[mid_col:])

tl_nulls = top_left.null_count().sum_horizontal().item()
br_nulls = bottom_right.null_count().sum_horizontal().item()

print(f"Total Dataset Shape: {df.shape}")
print(f"Top-Left Quadrant (first {mid_row} rows, first {mid_col} cols) null pct: {tl_nulls / (top_left.height * top_left.width):.4f}")
print(f"Bottom-Right Quadrant (last {df.height - mid_row} rows, last {len(numeric_cols) - mid_col} cols) null pct: {br_nulls / (bottom_right.height * bottom_right.width):.4f}")

# 2. Feature Extremes (First 5 vs Last 5)
first_5_nulls = df.select(numeric_cols[:5]).null_count().sum_horizontal().item()
last_5_nulls = df.select(numeric_cols[-5:]).null_count().sum_horizontal().item()

print(f"\nFirst 5 columns null pct: {first_5_nulls / (df.height * 5):.4f}")
print(f"Last 5 columns null pct: {last_5_nulls / (df.height * 5):.4f}")

# 3. Sample Extremes (First 20% vs Last 20% rows)
row_chunk = int(df.height * 0.2)
first_20p_nulls = df.slice(0, row_chunk).select(numeric_cols).null_count().sum_horizontal().item()
last_20p_nulls = df.slice(df.height - row_chunk).select(numeric_cols).null_count().sum_horizontal().item()

print(f"\nFirst 20% of rows null pct: {first_20p_nulls / (row_chunk * len(numeric_cols)):.4f}")
print(f"Last 20% of rows null pct: {last_20p_nulls / (row_chunk * len(numeric_cols)):.4f}")

# %% [code]
# Filter by high-completeness threshold (90% present / 10% missing)
completeness_threshold = 0.90
# Completeness = non-nulls / total_rows
completeness_stats = df.select([
    (pl.col(c).count() / pl.len()).alias(c)
    for c in numeric_cols
])

# Create a map of feature name to completeness
comp_map = {col: val for col, val in zip(numeric_cols, completeness_stats.row(0))}
high_comp_cols = [col for col, val in comp_map.items() if val >= completeness_threshold]

print(f"Completeness threshold: {completeness_threshold*100:.0f}%")
print(f"Features passing threshold: {len(high_comp_cols)} / {len(numeric_cols)}")
print("\nRemaining features:")
for i, col in enumerate(high_comp_cols, 1):
    print(f"  {i:2d}. {col} ({comp_map[col]:.1%})")

# Create and save analytic dataset
# Drop raw 'alias' as it's redundant with 'alias_filled'
analytic_cols = [c for c in high_comp_cols if c != "alias"]
df_analytic = df.select(analytic_cols + ["label"])
output_analytic = processed_dir / "analytic-dataset.parquet"
df_analytic.write_parquet(output_analytic)
print(f"\nAnalytic dataset saved to: {output_analytic}")
print(f"Shape: {df_analytic.shape}")

# %% [code]
missing_per_row = missing_mask.select([pl.sum_horizontal([pl.col(f"{c}_miss") for c in numeric_cols]).alias("missing_count")])
missing_dist = missing_per_row.group_by("missing_count").agg(pl.len().alias("row_count")).sort("missing_count")

print("\nRows by Number of Missing Features:")
print(missing_dist)

# %% [code]
# Ensure plots directory exists
plots_dir = Path("output")
plots_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.bar(missing_dist["missing_count"], missing_dist["row_count"], color="steelblue")
plt.xlabel("Number of Missing Features")
plt.ylabel("Number of Rows")
plt.title("Distribution of Missingness Across Features")
plt.tight_layout()
plt.savefig(plots_dir / "missingness-distribution.png", dpi=150)
plt.show()
print(f"\nDistribution plot saved to {plots_dir / 'missingness-distribution.png'}")

# %% [code]
# Bar chart: features sorted by missingness (descending)
missingness_vals = [df[col].null_count() / df.height * 100 for col in numeric_cols]
sorted_pairs = sorted(zip(numeric_cols, missingness_vals), key=lambda x: x[1], reverse=True)
feature_names = [p[0] for p in sorted_pairs]
missing_pcts = [p[1] for p in sorted_pairs]

plt.figure(figsize=(14, 10))
plt.barh(range(len(feature_names)), missing_pcts, color="coral")
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel("Missing Percentage (%)")
plt.title("Feature Missingness (sorted descending)")
plt.gca().invert_yaxis()  # Highest missing at top
plt.tight_layout()
plt.savefig(plots_dir / "missingness-bar-chart.png", dpi=150)
plt.show()
print(f"\nBar chart saved to {plots_dir / 'missingness-bar-chart.png'}")

# %% [markdown]
# ## 5. Feature Selection
# Filter by missingness, low-variance, and high correlation.
#
# %% [code]
# 1. Define Exclusion Lists
# IDs or outcome-dependent features that must not be used in modeling.
leak_exclusion_list = [
    "dcd_nrp_total_pump_time", 
    "extubation_to_perfusion_warm_ischemic_time",
    "tod_to_perfusion",
    "sbp90_to_declaration",
    "warm_ischemic_time_agonal_phase_to_cooling",
    "did_patient_expire_within_timeframe"
]

# Technical identifiers and indices.
id_exclusion_list = [
    "alias", 
    "alias_filled", 
    "observation"
]

# Other features manually removed based on domain knowledge or noise.
manual_excludes = []

all_excludes = leak_exclusion_list + id_exclusion_list + manual_excludes
print(f"All excludes: {all_excludes}")

# 2. Calculate missingness for all candidates
missingness = df.select([
    pl.col(c).null_count().alias(f"{c}_miss") for c in candidate_cols
]).row(0)

missing_df = pl.DataFrame({
    "feature": candidate_cols,
    "missing_count": [missingness[i] for i in range(len(candidate_cols))],
    "missing_pct": [missingness[i] / df.height * 100 for i in range(len(candidate_cols))]
}).sort("missing_pct", descending=True)

print("Feature Missingness (highest first):")
print(missing_df)

# 3. Filter by missingness threshold (< 10%) AND not in any exclusion list
missingness_threshold = 0.10  # Remove features with >10% missing
survivors = [
    row["feature"] for row in missing_df.to_dicts() 
    if row["missing_pct"] <= missingness_threshold * 100 
    and row["feature"] not in all_excludes
]

print(f"\nMissingness threshold: {missingness_threshold * 100:.0f}%")
print(f"Total exclusions: {len(all_excludes)}")
print(f"Features passing initial filters: {len(survivors)}")
print(f"Features removed: {len(candidate_cols) - len(survivors)}")

# %% [code]
# Split survivors to apply variance filter only to numeric features
survivor_numeric = [c for c in survivors if df.schema[c].is_numeric()]
survivor_categorical = [c for c in survivors if not df.schema[c].is_numeric()]

# Calculate variance for numeric survivors
variances = df.select([pl.col(c).var().alias(f"{c}_var") for c in survivor_numeric]).row(0)

var_df = pl.DataFrame({
    "feature": survivor_numeric,
    "variance": [variances[i] for i in range(len(survivor_numeric))]
}).sort("variance")

print("\nNumeric Feature Variances (lowest first):")
print(var_df)

# Filter low-variance numeric features (threshold: 0.01)
variance_threshold = 0.01
high_var_numeric = [row["feature"] for row in var_df.filter(pl.col("variance") > variance_threshold).to_dicts()]

print(f"\nVariance threshold: {variance_threshold}")
print(f"Numeric features passing variance: {len(high_var_numeric)} / {len(survivor_numeric)}")

# Combine variance-filtered numeric and all categorical survivors
high_var_cols = high_var_numeric + survivor_categorical

# %% [code]
# Correlation analysis on remaining features
print("\nCorrelation Analysis (high correlations > 0.9):")
# Use only numeric features for correlation
numeric_high_var = [c for c in high_var_cols if df.schema[c].is_numeric()]

if len(numeric_high_var) > 1:
    corr_df = df.select(numeric_high_var).corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i, col1 in enumerate(numeric_high_var):
        for j, col2 in enumerate(numeric_high_var):
            if i < j:  # Upper triangle only
                corr_val = corr_df.select(pl.col(col1).gather(j)).item()
                if abs(corr_val) > 0.9:
                    high_corr_pairs.append((col1, col2, corr_val))

    if high_corr_pairs:
        print("Highly correlated feature pairs (|r| > 0.9) - consider dropping one:")
        for col1, col2, corr_val in high_corr_pairs:
            print(f"  {col1} <-> {col2}: {corr_val:.3f}")
    else:
        print("No highly correlated pairs found (threshold: |r| > 0.9)")
else:
    print("Not enough numeric features for correlation analysis.")

# %% [markdown]
# ### Leakage Detection: Predictive Power
# Remove features that are too predictive on their own (likely proxies for the label).
#
# %% [code]
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("\nChecking for Over-Predictive Features (Leakage)...")
leakage_threshold = 0.95
predictive_leaks = []
clean_features = []

# Use pandas for the leakage check to handle mixed types easily
import pandas as pd
from sklearn.preprocessing import LabelEncoder

X_all = df.select(high_var_cols).to_pandas()
y_all = df["label"].to_numpy()

for col in high_var_cols:
    series = X_all[col]
    
    if pd.api.types.is_numeric_dtype(series):
        # Numeric: Fill with median
        X_col = series.fillna(series.median() if not series.isna().all() else 0).values.reshape(-1, 1)
    else:
        # Categorical/Datetime: Fill with mode, then LabelEncode
        mode_val = series.mode()[0] if not series.mode().empty else "missing"
        filled_series = series.fillna(mode_val)
        le = LabelEncoder()
        X_col = le.fit_transform(filled_series.astype(str)).reshape(-1, 1)
    
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_col, y_all)
    acc = accuracy_score(y_all, clf.predict(X_col))
    
    if acc > leakage_threshold:
        predictive_leaks.append((col, acc))
    else:
        clean_features.append(col)

if predictive_leaks:
    print("Found over-predictive features (leaks):")
    for col, acc in predictive_leaks:
        print(f"  {col:30s}: Accuracy={acc:.1%}")
else:
    print("No over-predictive leaks found.")

selected_features = clean_features

# %% [code]
# Save final model-ready dataset
# We include 'alias_filled' for grouping during cross-validation to prevent patient-level leakage.
df_model = df.select(selected_features + ["alias_filled", "label"])
output_model = processed_dir / "model-ready-dataset.parquet"
df_model.write_parquet(output_model)

print(f"\\n{'='*60}")
print(f"FINAL FEATURE SET: {len(selected_features)} features")
print(f"{'='*60}")
print(f"Model-ready dataset saved to: {output_model}")
print(f"Shape: {df_model.shape}")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feat}")

# %% [markdown]
# ## 6. Feature Distribution Overview
#
# %% [code]
print("Feature Statistics (selected features only):")
print(df.select(selected_features).describe())

# %% [code]
plt.close("all")  # Free memory after plotting
