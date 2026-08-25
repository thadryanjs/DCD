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
# # Data Exploration: Ophthalmology Cases
# Exploratory analysis with detailed missingness patterns.
#
# %% [code]
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
# Get numeric columns (exclude label)
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "label"]

print(f"Numeric features: {numeric_cols}")

# %% [code]
print(f"Total features: {len(numeric_cols)}")

# %% [code]
# Calculate absolute differences
diffs = []
for col in numeric_cols:
    mean_pos = df.filter(pl.col("label") == 1).select(pl.col(col).mean()).item()
    mean_neg = df.filter(pl.col("label") == 0).select(pl.col(col).mean()).item()
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
plots_dir = Path("../plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# Create missingness matrix (rows = samples, cols = features)
missing_matrix = missing_mask.select(numeric_cols).to_numpy()

# OOM warning for large datasets
if missing_matrix.size > 10_000_000:
    mem_gb = missing_matrix.nbytes / (1024**3)
    print(f"WARNING: Missingness matrix is {mem_gb:.2f} GB. Consider downsampling.")

plt.figure(figsize=(12, 8))
sns.heatmap(missing_matrix.T, cmap="viridis", cbar_kws={"label": "Missing (1) / Present (0)"})
plt.title("Missingness Heatmap by Feature")
plt.xlabel("Sample Index")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(plots_dir / "missingness-heatmap.png", dpi=150)
plt.show()
print(f"\nHeatmap saved to {plots_dir / 'missingness-heatmap.png'}")

# %% [code]
# Missingness distribution: how many features are missing per row?
missing_per_row = missing_mask.select([pl.sum([pl.col(f"{c}_miss") for c in numeric_cols]).alias("missing_count")])
missing_dist = missing_per_row.group_by("missing_count").agg(pl.len().alias("row_count")).sort("missing_count")

print("\nRows by Number of Missing Features:")
print(missing_dist)

# %% [code]
# Ensure plots directory exists
plots_dir = Path("../plots")
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

# %% [markdown]
# ## 5. Feature Distribution Overview
#
# %% [code]
print("Feature Statistics (all data):")
print(df.select(numeric_cols).describe())

# %% [code]
plt.close("all")  # Free memory after plotting