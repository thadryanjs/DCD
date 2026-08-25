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
# # Data Tour: Ophthalmology Dataset
# Before we build a pipeline, we need to understand the raw state of the data. 
# This notebook is a "discovery" phase to identify what needs cleaning.
#
# %% [code]
import polars as pl
from pathlib import Path

DATA_DIR = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data")

df_pos_raw = pl.read_excel(DATA_DIR / "positive-cases.xlsx")
df_neg_raw = pl.read_excel(DATA_DIR / "negative-cases.xlsx")

print(f"Positive cases raw shape: {df_pos_raw.shape}")
print(f"Negative cases raw shape: {df_neg_raw.shape}")


# %% [markdown]
# ## 1. Inspecting Column Names
# Let's look at the raw column names to see if they are "machine-friendly".
#
# %% [code]
print("Positive Case Columns:")
print(df_pos_raw.columns)

print("\nNegative Case Columns:")
print(df_neg_raw.columns)


# %% [markdown]
# ### Observation: Dirty Names
# We see several issues:
# 1. Spaces (e.g., "O2 Sat ")
# 2. Slashes (e.g., "Case/ID")
# 3. Mixed Case
# 4. Trailing characters
# These will make coding difficult. We need a systematic `clean_colnames` function.
#
# %% [code]
# Find examples of names that need cleaning
dirty_cols = [c for c in df_pos_raw.columns if " " in c or "/" in c or not c.islower()]
print(f"Found {len(dirty_cols)} columns needing cleaning.")
print("Examples:", dirty_cols[:5])


# %% [markdown]
# ## 2. Evaluating Data Quality
# Are there completely empty rows? Are there "spacer" rows?
#
# %% [code]
# Check for completely null rows
pos_all_null = df_pos_raw.filter(pl.all_horizontal(pl.all().is_null())).height
neg_all_null = df_neg_raw.filter(pl.all_horizontal(pl.all().is_null())).height

print(f"Positive cases: {pos_all_null} completely empty rows.")
print(f"Negative cases: {neg_all_null} completely empty rows.")


# %% [markdown]
# Check for "essential" nulls (e.g., Age is missing).
#
# %% [code]
# We have to guess the column name since it might be "Age" or "age"
age_col = [c for c in df_pos_raw.columns if "age" in c.lower()][0]
null_age = df_pos_raw.filter(pl.col(age_col).is_null()).height
print(f"Column '{age_col}' has {null_age} null values in positive cases.")


# %% [markdown]
# ## 3. Semantic Redundancy
# Do we have columns that mean the same thing but have different names?
#
# %% [code]
# Look for BMI related columns
bmi_cols = [c for c in df_pos_raw.columns if "bmi" in c.lower()]
print(f"BMI related columns: {bmi_cols}")


# %% [markdown]
# ### Observation: Semantic Mismatch
# If we see both `bmi` and `bmicalc`, we have redundant information or inconsistent naming across files. 
# We will need a `semantic_map` to align these.
#
# %% [code]
# Check O2 Sat
o2_cols = [c for c in df_pos_raw.columns if "o2" in c.lower()]
print(f"O2 Sat related columns: {o2_cols}")


# %% [markdown]
# ## 4. The Concat Challenge
# What happens if we try to combine these datasets immediately?
#
# %% [code]
try:
    df_combined = pl.concat([df_pos_raw, df_neg_raw])
    print("Concat succeeded unexpectedly.")
except Exception as e:
    print(f"Concat failed as expected: {e}")


# %% [markdown]
# ### Observation: Schema Mismatch
# The failure is likely due to:
# 1. Different column names (one has "Age", other has "age").
# 2. Different data types for the same conceptual column.
#
# We need to align types (casting to String where they differ) before concatenating.
#
# %% [markdown]
# ## Summary of Needed Actions
# 1. **Clean Column Names**: Lowercase, replace spaces/slashes.
# 2. **Filter Noise**: Remove all-null rows.
# 3. **Deduplicate**: Ensure cases aren't repeated.
# 4. **Semantic Alignment**: Merge `bmicalc` -> `bmi`, etc.
# 5. **Type Alignment**: Cast mismatched types to String.
# 6. **Combine**: Concat into a single `df_all`.
#
# This will be implemented in `01_process-dataset.py`.
