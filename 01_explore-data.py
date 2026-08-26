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
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
from polars import col
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# %% [code]
data_dir = Path("data")
processed_dir = Path("data/processed")

# %% [markdown]
# ## 1. Load Processed Dataset
#
# %% [code]
try:
    df = pl.read_parquet(processed_dir / "combined-dataset.parquet")
except FileNotFoundError:
    raise FileNotFoundError(
        "combined-dataset.parquet not found. Run 00_process-data.py first."
    )

print(f"Dataset loaded: {df.shape}")

# %% [markdown]
# ## 2. Class Distribution
#
# %% [code]
print("Class Distribution:")
print(df.group_by("progression_to_death").agg(pl.len().alias("count")).sort("progression_to_death"))

# %% [code]
print(
    f"""
Class Balance:
  Positive (1): {df.filter(col("progression_to_death") == 1).height} rows
  Negative (0): {df.filter(col("progression_to_death") == 0).height} rows
  Total: {df.height} rows
  Positive Rate: {df.filter(col("progression_to_death") == 1).height / df.height * 100:.1f}%
"""
)

# %% [markdown]
# ## 3. Numeric Features: Mean Differences by Class
#
# %% [code]
# Get numeric and categorical columns (exclude label and temporal/datetimes)
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "progression_to_death"]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and not df.schema[c].is_temporal() and c != "progression_to_death"]
candidate_cols = numeric_cols + categorical_cols

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Total candidate features: {len(candidate_cols)}")

# %% [code]
# Calculate mean differences by class efficiently
class_means = df.group_by("progression_to_death").mean()
pos_means = class_means.filter(col("progression_to_death") == 1).row(0)
neg_means = class_means.filter(col("progression_to_death") == 0).row(0)

col_names = class_means.columns[1:]
diffs = []
for i, c in enumerate(col_names):
    m_pos = pos_means[i+1] if pos_means[i+1] is not None else 0.0
    m_neg = neg_means[i+1] if neg_means[i+1] is not None else 0.0
    diffs.append((c, m_pos, m_neg, abs(m_pos - m_neg)))

# Sort by difference (descending)
diffs.sort(key=lambda x: x[3], reverse=True)

# %% [code]
print("Mean differences by class (sorted by magnitude):")
for c, m_pos, m_neg, diff in diffs:
    print(f"  {c:30s}: Pos={m_pos:10.3f}, Neg={m_neg:10.3f}, Diff={diff:8.3f}")

# %% [markdown]
# ## 4. Missingness Analysis
# Deep dive into missing data patterns.
#
# %% [code]
# Missing count and percentage per column (efficient: single scan with null_count())
missing_stats = df.select([
    col(c).null_count().alias(f"{c}_nulls") for c in df.columns
]).row(0)

columns = df.columns
missing_stats = [(c, missing_stats[i], missing_stats[i] / df.height * 100)
                 for i, c in enumerate(columns)]

# Sort by missing percentage (descending)
missing_stats.sort(key=lambda x: x[2], reverse=True)

# %% [code]
print("Missingness by Column (sorted by % missing):")
print("-" * 60)
for c, count, pct in missing_stats:
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"{c:30s} |{bar}| {count:4d} ({pct:5.1f}%)")

# %% [code]
print("\nMissingness Comparison: Positive vs Negative Class")
print("=" * 70)

# Single scan for all missingness counts by class
missing_by_class = df.group_by("progression_to_death").agg([
    col(c).null_count().alias(f"{c}_nulls") for c in df.columns
])

pos_row = missing_by_class.filter(col("progression_to_death") == 1).row(0)
neg_row = missing_by_class.filter(col("progression_to_death") == 0).row(0)
pos_total = df.filter(col("progression_to_death") == 1).height
neg_total = df.filter(col("progression_to_death") == 0).height

# %% [code]
for i, c in enumerate(df.columns):
    pos_missing = pos_row[i+1] if pos_row else 0
    neg_missing = neg_row[i+1] if neg_row else 0
    
    pos_pct = pos_missing / pos_total * 100 if pos_total > 0 else 0
    neg_pct = neg_missing / neg_total * 100 if neg_total > 0 else 0

    if pos_missing > 0 or neg_missing > 0:
        print(f"{c:30s}: Pos={pos_pct:5.1f}% ({pos_missing}/{pos_total}), Neg={neg_pct:5.1f}% ({neg_missing}/{neg_total})")

# %% [code]
print("\nMissingness Pattern Matrix (correlation of missingness):")
print("Columns that share missing values tend to have correlated patterns.")

# Create binary missing indicator dataframe
missing_mask = df.select([col(c).is_null().cast(pl.Int8).alias(f"{c}_miss") for c in df.columns])

# Calculate correlation matrix of missingness
if len(numeric_cols) > 1:
    miss_cols = [f"{c}_miss" for c in numeric_cols]
    corr_matrix = missing_mask.select(miss_cols).corr()
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
subset_mask = missing_mask.select(miss_cols)

# OOM prevention: Downsample if matrix is too large (> 10M elements)
if subset_mask.height * len(miss_cols) > 10_000_000:
    sample_size = 10_000_000 // len(miss_cols)
    print(f"WARNING: Dataset too large for heatmap ({subset_mask.height * len(miss_cols)} elements).")
    print(f"Downsampling to {sample_size} rows for visualization.")
    subset_mask = subset_mask.sample(n=sample_size)

missing_matrix = subset_mask.to_numpy()

# Use a strict 2-color map to prevent grey interpolation
# 0: White (Present), 1: Black (Missing)
binary_cmap = ListedColormap(["white", "black"])

# %% [code]
plt.figure(figsize=(14, 10))
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

# %% [code]
print(f"Total Dataset Shape: {df.shape}")
print(f"Top-Left Quadrant (first {mid_row} rows, first {mid_col} cols) null pct: {tl_nulls / (top_left.height * top_left.width):.4f}")
print(f"Bottom-Right Quadrant (last {df.height - mid_row} rows, last {len(numeric_cols) - mid_col} cols) null pct: {br_nulls / (bottom_right.height * bottom_right.width):.4f}")

# %% [code]
# 2. Feature Extremes (First 5 vs Last 5)
first_5_nulls = df.select(numeric_cols[:5]).null_count().sum_horizontal().item()
last_5_nulls = df.select(numeric_cols[-5:]).null_count().sum_horizontal().item()

print(f"\nFirst 5 columns null pct: {first_5_nulls / (df.height * 5):.4f}")
print(f"Last 5 columns null pct: {last_5_nulls / (df.height * 5):.4f}")

# %% [code]
# 3. Sample Extremes (First 20% vs Last 20% rows)
row_chunk = int(df.height * 0.2)
first_20p_nulls = df.slice(0, row_chunk).select(numeric_cols).null_count().sum_horizontal().item()
last_20p_nulls = df.slice(df.height - row_chunk).select(numeric_cols).null_count().sum_horizontal().item()

print(f"\nFirst 20% of rows null pct: {first_20p_nulls / (row_chunk * len(numeric_cols)):.4f}")
print(f"Last 20% of rows null pct: {last_20p_nulls / (row_chunk * len(numeric_cols)):.4f}")

# %% [code]
# Filter by high-completeness threshold (90% present / 10% missing)
completeness_threshold = 0.90
completeness_stats = df.select([
    (col(c).count() / pl.len()).alias(c)
    for c in numeric_cols
])

comp_map = {c: val for c, val in zip(numeric_cols, completeness_stats.row(0))}
high_comp_cols = [c for c, val in comp_map.items() if val >= completeness_threshold]

# %% [code]
print(f"Completeness threshold: {completeness_threshold*100:.0f}%")
print(f"Features passing threshold: {len(high_comp_cols)} / {len(numeric_cols)}")
print("\nRemaining features:")
for i, c in enumerate(high_comp_cols, 1):
    print(f"  {i:2d}. {c} ({comp_map[c]:.1%})")

# %% [code]
# Create and save analytic dataset
analytic_cols = [c for c in high_comp_cols if c != "alias"]
df_analytic = df.select(analytic_cols + ["progression_to_death"])
# Removed early save to analytic-dataset.parquet to prevent leakage in R analysis
print(f"\nInitial missingness filter applied. Shape: {df_analytic.shape}")

# %% [code]
missing_per_row = missing_mask.select([pl.sum_horizontal([col(f"{c}_miss") for c in numeric_cols]).alias("missing_count")])
missing_dist = missing_per_row.group_by("missing_count").agg(pl.len().alias("row_count")).sort("missing_count")

# %% [code]
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

# %% [code]
print(f"\nDistribution plot saved to {plots_dir / 'missingness-distribution.png'")

# %% [code]
# Bar chart: features sorted by missingness (descending)
missingness_vals = [df[c].null_count() / df.height * 100 for c in numeric_cols]
sorted_pairs = sorted(zip(numeric_cols, missingness_vals), key=lambda x: x[1], reverse=True)
feature_names = [p[0] for p in sorted_pairs]
missing_pcts = [p[1] for p in sorted_pairs]

# %% [code]
plt.figure(figsize=(14, 10))
plt.barh(range(len(feature_names)), missing_pcts, color="coral")
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel("Missing Percentage (%)")
plt.title("Feature Missingness (sorted descending)")
plt.gca().invert_yaxis()  # Highest missing at top
plt.tight_layout()
plt.savefig(plots_dir / "missingness-bar-chart.png", dpi=150)
plt.show()

# %% [code]
print(f"\nBar chart saved to {plots_dir / 'missingness-bar-chart.png'")

# %% [markdown]
# ## 5. Feature Selection
# Filter by missingness, low-variance, and high correlation.
#
# %% [code]
# 1. Define Exclusion Lists
leak_exclusion_list = [
    "dcd_nrp_total_pump_time", 
    "extubation_to_perfusion_warm_ischemic_time",
    "tod_to_perfusion",
    "sbp90_to_declaration",
    "warm_ischemic_time_agonal_phase_to_cooling",
    "did_patient_expire_within_timeframe"
]

id_exclusion_list = [
    "alias", 
    "alias_filled", 
    "observation"
]

manual_excludes = []
all_excludes = leak_exclusion_list + id_exclusion_list + manual_excludes
print(f"All excludes: {all_excludes}")

# %% [code]
# 2. Calculate missingness for all candidates
missingness = df.select([
    col(c).null_count().alias(f"{c}_miss") for c in candidate_cols
]).row(0)

missing_df = pl.DataFrame({
    "feature": candidate_cols,
    "missing_count": [missingness[i] for i in range(len(candidate_cols))],
    "missing_pct": [missingness[i] / df.height * 100 for i in range(len(candidate_cols))]
}).sort("missing_pct", descending=True)

# %% [code]
print("Feature Missingness (highest first):")
print(missing_df)

# %% [code]
# 3. Filter by missingness threshold (< 10%) AND not in any exclusion list
missingness_threshold = 0.10  # Remove features with >10% missing
survivors = missing_df.filter(
    (col("missing_pct") <= missingness_threshold * 100) & 
    (~col("feature").is_in(all_excludes))
).get_column("feature").to_list()

# %% [code]
print(f"\nMissingness threshold: {missingness_threshold * 100:.0f}%")
print(f"Total exclusions: {len(all_excludes)}")
print(f"Features passing initial filters: {len(survivors)}")
print(f"Features removed: {len(candidate_cols) - len(survivors)}")

# %% [code]
# Split survivors to apply variance filter only to numeric features
survivor_numeric = [c for c in survivors if df.schema[c].is_numeric()]
survivor_categorical = [c for c in survivors if not df.schema[c].is_numeric()]

# Calculate variance for numeric survivors
variances = df.select([col(c).var().alias(f"{c}_var") for c in survivor_numeric]).row(0)

var_df = pl.DataFrame({
    "feature": survivor_numeric,
    "variance": [variances[i] for i in range(len(survivor_numeric))]
}).sort("variance")

# %% [code]
print("\nNumeric Feature Variances (lowest first):")
print(var_df)

# %% [code]
# Filter low-variance numeric features (threshold: 0.01)
variance_threshold = 0.01
high_var_numeric = var_df.filter(col("variance") > variance_threshold).get_column("feature").to_list()

print(f"\nVariance threshold: {variance_threshold}")
print(f"Numeric features passing variance: {len(high_var_numeric)} / {len(survivor_numeric)}")

# %% [code]
# Combine variance-filtered numeric and all categorical survivors
high_var_cols = high_var_numeric + survivor_categorical

# %% [code]
# Correlation analysis on remaining features
print("\nCorrelation Analysis (high correlations > 0.9):")
numeric_high_var = [c for c in high_var_cols if df.schema[c].is_numeric()]

# %% [code]
if len(numeric_high_var) > 1:
    corr_df = df.select(numeric_high_var).corr()

    high_corr_pairs = []
    for i, c1 in enumerate(numeric_high_var):
        for j, c2 in enumerate(numeric_high_var):
            if i < j:
                corr_val = corr_df.select(col(c1).gather(j)).item()
                if abs(corr_val) > 0.9:
                    high_corr_pairs.append((c1, c2, corr_val))

    # %% [code]
    if high_corr_pairs:
        print("Highly correlated feature pairs (|r| > 0.9) - consider dropping one:")
        for c1, c2, corr_val in high_corr_pairs:
            print(f"  {c1} <-> {c2}: {corr_val:.3f}")
    else:
        print("No highly correlated pairs found (threshold: |r| > 0.9)")
else:
    print("Not enough numeric features for correlation analysis.")

# %% [markdown]
# ### Leakage Detection: Predictive Power
# Remove features that are too predictive on their own (likely proxies for the label).
#
# %% [code]
print("\nChecking for Over-Predictive Features (Leakage)...")
leakage_threshold = 0.95
predictive_leaks = []
clean_features = []

X_all = df.select(high_var_cols).to_pandas()
y_all = df["progression_to_death"].to_numpy()

# %% [code]
for c in high_var_cols:
    series = X_all[c]
    
    if pd.api.types.is_numeric_dtype(series):
        X_col = series.fillna(series.median() if not series.isna().all() else 0).values.reshape(-1, 1)
    else:
        mode_val = series.mode()[0] if not series.mode().empty else "missing"
        filled_series = series.fillna(mode_val)
        le = LabelEncoder()
        X_col = le.fit_transform(filled_series.astype(str)).reshape(-1, 1)
    
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_col, y_all)
    acc = accuracy_score(y_all, clf.predict(X_col))
    
    if acc > leakage_threshold:
        predictive_leaks.append((c, acc))
    else:
        clean_features.append(c)

# %% [code]
if predictive_leaks:
    print("Found over-predictive features (leaks):")
    for c, acc in predictive_leaks:
        print(f"  {c:30s}: Accuracy={acc:.1%}")
else:
    print("No over-predictive leaks found.")

selected_features = clean_features

# %% [code]
# Save final analytic dataset
# We include 'alias_filled' for grouping during cross-validation to prevent patient-level leakage.
df_model = df.select(selected_features + ["alias_filled", "progression_to_death"])
output_analytic = processed_dir / "analytic-dataset.parquet"
df_model.write_parquet(output_analytic)

# %% [code]
print(f"\\n{'='*60}")
print(f"FINAL FEATURE SET: {len(selected_features)} features")
print(f"{'='*60}")
print(f"Analytic dataset saved to: {output_analytic}")
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
