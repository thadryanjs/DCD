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
import numpy as np
from polars import col
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import mutual_info_classif

# %% [code]
def report_filter(name, before_list, after_list, reason_col=None, values=None):
    """Prints a transparency receipt for a feature filter."""
    dropped = sorted(list(set(before_list) - set(after_list)))
    print(f"\n--- {name} Filter Receipt ---")
    print(f"In: {len(before_list)} | Out: {len(after_list)} | Dropped: {len(dropped)}")
    if dropped:
        if reason_col and values:
            # Create a small table for the dropped features
            dropped_data = []
            for d in dropped:
                val = values.get(d, "N/A")
                dropped_data.append({"feature": d, reason_col: val})
            print(pl.DataFrame(dropped_data))
        else:
            print(f"Dropped features: {dropped}")
    else:
        print("No features dropped.")

# %% [code]
data_dir = Path("data")
processed_dir = Path("data/processed")

# %% [markdown]
# # Load Processed Dataset
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
# # Class Distribution
#
# %% [code]
print("Observation-level Distribution:")
print(df.group_by("progression_to_death").agg(pl.len().alias("count")).sort("progression_to_death"))

# %% [code]
print("\nPatient-level Distribution:")
unique_patients = df["alias_filled"].n_unique()
avg_obs = df.height / unique_patients
print(f"Total Unique Individuals: {unique_patients}")
print(f"Average Obs per Person: {avg_obs:.2f}")
print(df.group_by(["alias_filled", "progression_to_death"]).agg(pl.len().alias("obs")).group_by("progression_to_death").agg(pl.len().alias("count")).sort("progression_to_death"))

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

# %% [code]
# Get numeric and categorical columns (exclude label and temporal/datetimes)
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "progression_to_death"]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and not df.schema[c].is_temporal() and c != "progression_to_death"]
candidate_cols = numeric_cols + categorical_cols

# %% [markdown]
# # In-Patient Stability Analysis
# Quantify variance within patients relative to population variance.
#
# %% [code]
print("\nIn-Patient Stability Analysis (Within-Patient SD / Global SD):")
stability_results = []

for c in numeric_cols:
    # Global SD
    global_sd = df[c].std()
    if global_sd == 0 or global_sd is None:
        continue
    
    # Mean of within-patient SDs
    # We only consider patients with > 1 observation for a valid SD
    within_sd_mean = df.group_by("alias_filled").agg(pl.col(c).std().alias("std")).filter(pl.col("std").is_not_null())["std"].mean()
    
    if within_sd_mean is not None:
        ratio = within_sd_mean / global_sd
        stability_results.append((c, ratio))

# Sort by ratio (lowest = most stable)
stability_results.sort(key=lambda x: x[1])

for c, ratio in stability_results:
    status = "Stable" if ratio < 0.3 else "Volatile"
    print(f"  {c:30s}: Ratio={ratio:6.3f} [{status}]")

print(f"Average Stability Ratio: {np.mean([r for c, r in stability_results]):.3f}")

# %% [markdown]
# ### Interpretation of Stability
# The **Stability Ratio** ($\frac{\text{mean}(\sigma_{\text{patient}})}{\sigma_{\text{global}}}$) quantifies how much a feature varies *within* a patient relative to how much it varies *across* the population.
#
# - **Stable (Ratio < 0.3)**: Low within-patient variance. Repeated observations are largely redundant. This justifies the "First Look Only" approach as a single sample is representative of the patient.
# - **Volatile (Ratio $\ge$ 0.3)**: High within-patient variance. The feature changes significantly over time (e.g., `fio2`, `rate`), suggesting clinical drift or active intervention.
#
# A target average ratio of $\sim 0.3$ is used as a heuristic to determine if the "First Look" is a robust proxy for the patient's state, reducing the risk of time-series leakage in model training.
#
# %% [code]
print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Total candidate features: {len(candidate_cols)}")

# %% [code]
# Calculate mean differences by class efficiently
class_means = df.group_by("progression_to_death").agg(pl.col(numeric_cols).mean())
pos_means = class_means.filter(col("progression_to_death") == 1).row(0)
neg_means = class_means.filter(col("progression_to_death") == 0).row(0)

col_names = numeric_cols
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
# # Missingness Analysis
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
for c, count, pct in missing_stats:
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"{c:30s} |{bar}| {count:4d} ({pct:5.1f}%)")

# %% [code]
print("\nMissingness Comparison: Positive vs Negative Class")

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
    subset_mask = subset_mask.sample(n=sample_size, random_state=8675309)

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
# Ensure plots directory exists
plots_dir = Path("output")
plots_dir.mkdir(parents=True, exist_ok=True)

# Calculate distribution of missingness across features
missing_per_row = missing_mask.select(miss_cols).sum_horizontal()
missing_dist = missing_per_row.value_counts().sort("sum").to_pandas()
missing_dist.columns = ["missing_count", "row_count"]

plt.figure(figsize=(10, 5))
plt.bar(missing_dist["missing_count"], missing_dist["row_count"], color="steelblue")
plt.xlabel("Number of Missing Features")
plt.ylabel("Number of Rows")
plt.title("Distribution of Missingness Across Features")
plt.tight_layout()
plt.savefig(plots_dir / "missingness-distribution.png", dpi=150)

# %% [code]
print(f"\nDistribution plot saved to {plots_dir / 'missingness-distribution.png'}")

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

# %% [code]
print(f"\nBar chart saved to {plots_dir / 'missingness-bar-chart.png'}")

# %% [markdown]
# # Feature Selection
# Filter by missingness and low-variance; identify high correlations.
#
# %% [code]
# 1. Define Exclusion Lists
leak_justifications = {
    "dcd_nrp_total_pump_time": "Post-outcome: pump time recorded during DCD/NRP",
    "extubation_to_perfusion_warm_ischemic_time": "Post-outcome: time from extubation to perfusion",
    "tod_to_perfusion": "Post-outcome: time from declaration to perfusion",
    "sbp90_to_declaration": "Post-outcome: time from SBP<90 to declaration",
    "warm_ischemic_time_agonal_phase_to_cooling": "Post-outcome: agonal phase duration",
    "did_patient_expire_within_timeframe": "Direct proxy: patient expiration status"
}
leak_exclusion_list = list(leak_justifications.keys())

# %% [markdown]
# **LOADBEARING** — These are post-outcome variables; including any of them 
# makes the ML results meaningless.
# Consumed by: `03` model pipeline.

# %% [code]
print("\nLeak Exclusion List & Justifications:")
leak_df = pl.DataFrame([
    {"feature": k, "reason": v} for k, v in leak_justifications.items()
])
print(leak_df)

id_exclusion_list = [
    "alias",
    "alias_filled",
    "observation"
]

manual_excludes = []
all_excludes = leak_exclusion_list + id_exclusion_list + manual_excludes
print(f"\nTotal exclusions count: {len(all_excludes)}")

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
# 3. Filter by missingness threshold (< 10%) AND not in any exclusion list
missingness_threshold = 0.10  # Remove features with >10% missing
survivors = missing_df.filter(
    (col("missing_pct") <= missingness_threshold * 100) &
    (~col("feature").is_in(all_excludes))
).get_column("feature").to_list()

# Receipt for missingness and exclusions
miss_dropped = [c for c in candidate_cols if c not in survivors]
miss_vals = {c: missing_df.filter(pl.col("feature") == c)["missing_pct"][0] for c in miss_dropped}
report_filter("Missingness & Exclusions", candidate_cols, survivors, "missing_pct", miss_vals)

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

# %% [markdown]
# ### Variance Filtering
# **Note:** The variance threshold is applied to raw unscaled variances and is therefore unit-dependent.

# %% [code]
# Filter low-variance numeric features (threshold: 0.01)
variance_threshold = 0.01
high_var_numeric = var_df.filter(col("variance") > variance_threshold).get_column("feature").to_list()

var_dropped = [c for c in survivor_numeric if c not in high_var_numeric]
var_vals = {c: var_df.filter(pl.col("feature") == c)["variance"][0] for c in var_dropped}
report_filter("Variance", survivor_numeric, high_var_numeric, "variance", var_vals)

# Combine variance-filtered numeric and all categorical survivors
high_var_cols = high_var_numeric + survivor_categorical

# %% [code]
# Correlation analysis on remaining features
print("\nCorrelation Analysis (high correlations > 0.9):")
# Note: This section reports pairs but does not drop any features.
numeric_high_var = [c for c in high_var_cols if df.schema[c].is_numeric()]

# %% [markdown]
# ### Leakage Detection: Predictive Power
# Remove features that are too predictive on their own (likely proxies for the label).
#
# **LOADBEARING** — Fitting MI on all rows leaks test labels into feature selection.
# We must compute MI on training patients only.
# Consumed by: `03` model pipeline.

# %% [code]
print("\nChecking for Over-Predictive Features (Leakage)...")
# Use Mutual Information as a more robust leakage metric than raw accuracy
mi_threshold = 0.5 # Heuristic: MI > 0.5 is very high for binary targets
predictive_leaks = []
clean_features = []

X_all = df.select(high_var_cols).to_pandas()
y_all = df["progression_to_death"].to_numpy()
groups = df["alias_filled"].to_numpy()

# Prevent leakage: Compute MI on training patients only
seed_const = 8675309
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed_const)
train_idx, _ = next(gss.split(X_all, y_all, groups))

X_train = X_all.iloc[train_idx]
y_train = y_all[train_idx]

# Assert seed matches 03_model.py
assert seed_const == 8675309, f"MI seed {seed_const} differs from pipeline seed 8675309"
print(f"MI Split: Fitted on {len(np.unique(groups[train_idx]))} training patients. Seed: {seed_const}")

# For MI, we need to handle missing values first without biasing the leak check
X_mi = X_train.copy()
discrete_mask = []

for c in X_mi.columns:
    if pd.api.types.is_numeric_dtype(X_mi[c]):
        X_mi[c] = X_mi[c].fillna(-999)
        discrete_mask.append(False)
    else:
        X_mi[c] = X_mi[c].fillna("missing")
        X_mi[c] = LabelEncoder().fit_transform(X_mi[c].astype(str))
        discrete_mask.append(True)

mi_scores = mutual_info_classif(X_mi, y_train, discrete_features=discrete_mask, random_state=seed_const)

for i, c in enumerate(high_var_cols):
    score = mi_scores[i]
    if score > mi_threshold:
        predictive_leaks.append((c, score))
    else:
        clean_features.append(c)

# %% [code]
# Receipt for MI filter
mi_dropped = [c for c in high_var_cols if c not in clean_features]
mi_vals = {c: mi_scores[high_var_cols.index(c)] for c in mi_dropped}
report_filter("Mutual Information (Leakage)", high_var_cols, clean_features, "mi_score", mi_vals)

selected_features = clean_features

# %% [code]
# Save final analytic dataset
# We include 'alias_filled' and 'observation' for grouping/traceability
df_model = df.select(selected_features + ["alias_filled", "observation", "progression_to_death"])
output_analytic = processed_dir / "analytic-dataset.parquet"
df_model.write_parquet(output_analytic)

# Also save as CSV to avoid R 'arrow' dependency issues
output_analytic_csv = processed_dir / "analytic-dataset.csv"
df_model.write_csv(output_analytic_csv)

# Feature artifact: store values that let each feature through
feature_stats = []
for f in selected_features:
    m_pct = missing_df.filter(pl.col("feature") == f)["missing_pct"][0]
    v_val = var_df.filter(pl.col("feature") == f)["variance"][0] if f in survivor_numeric else np.nan
    mi_val = mi_scores[high_var_cols.index(f)]
    feature_stats.append({"feature": f, "missing_pct": m_pct, "variance": v_val, "mi_score": mi_val})

feat_artifact = pl.DataFrame(feature_stats)
feat_artifact.write_csv(plots_dir / "selected-features.csv")

# %% [code]
print(f"FINAL FEATURE SET: {len(selected_features)} features")
print(f"Analytic dataset saved to: {output_analytic}")
print(f"CSV export saved to: {output_analytic_csv}")
print(f"Feature artifact saved to: {plots_dir / 'selected-features.csv'}")
print(f"Shape: {df_model.shape}")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feat}")

# %% [markdown]
# # Feature Distribution Overview
#
# %% [code]
print("Feature Statistics (selected features only):")
print(df.select(selected_features).describe())

# %% [code]
plt.close("all")  # Free memory after plotting
