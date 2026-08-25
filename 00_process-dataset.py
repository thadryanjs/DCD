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
# # Data Processing: Ophthalmology Cases
# This notebook is a full "proof" of the data pipeline. We begin with raw data,
# identify structural issues, implement fixes, and verify the results.
#
# %% [code]
import polars as pl
from pathlib import Path
import re

data_dir = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data")

def load_raw(filename):
    return pl.read_excel(data_dir / filename)

df_pos_raw = load_raw("positive-cases.xlsx")
df_neg_raw = load_raw("negative-cases.xlsx")

print(f"Raw Shapes -> Positive: {df_pos_raw.shape}, Negative: {df_neg_raw.shape}")

# %% [markdown]
# ## 1. Raw Data Exploration
# Before cleaning, we inspect the raw structure and a sample of the records.
#
# %% [code]
print("Positive Cases - Head:")
print(df_pos_raw.head())

# %% [code]
print("\nPositive Cases - Tail:")
print(df_pos_raw.tail())

# %% [code]
print("\n" + "="*40 + "\n")

# %% [code]
print("Negative Cases - Head:")
print(df_neg_raw.head())

# %% [code]
print("\nNegative Cases - Tail:")
print(df_neg_raw.tail())

# %% [code]
print("\n" + "="*40 + "\n")
print("Inspecting rows around index 19-23 (Negative Cases):")
print(df_neg_raw.slice(19, 5))


# %% [markdown]
# ### Longitudinal Data Inspection
# We inspect the raw structure around the transition point (rows 14-22). Note: polars `read_excel` auto-filtered the orange separator row that exists in the spreadsheet.
#
# %% [code]
print("Rows 14-22 (Negative Cases) - Transition from Person 1 to Person 2:")
print(df_neg_raw.slice(14, 9))


# %% [markdown]
# ### Data Structure Audit
# We examine how Alias, Age, and Date/Time relate to each other. This reveals the longitudinal pattern where measurements for the same subject share demographics.
#
# %% [code]
pl.Config.set_tbl_rows(100)
key_cols = [c for c in ["Alias", "Age", "Date/Time"] if c in df_neg_raw.columns]
print("Negative Cases - First 20 rows (Alias, Age, Date/Time):")
print(df_neg_raw.select(key_cols).slice(0, 20))


# %% [markdown]
# ### Raw Data Audit: Separator Detection
# We suspect the presence of "separator" rows—rows that aren't completely empty but don't contain case data (e.g., section headers).
#
# %% [code]
def find_separators(df, name):
    age_col = [c for c in df.columns if "age" in c.lower()]
    if not age_col:
        return None

    col = age_col[0]
    separators = df.filter(
        (~pl.all_horizontal(pl.all().is_null())) &
        (pl.col(col).is_null())
    )
    print(f"{name} - Found {separators.height} potential separator rows.")
    if separators.height > 0:
        print("Sample of separator rows (missing age, but not completely empty):")
        print(separators.head())
    return separators

# %% [code]
find_separators(df_pos_raw, "Positive Cases")

# %% [code]
print("\n" + "-"*20)
find_separators(df_neg_raw, "Negative Cases")

# %% [code]
def check_sparsity(df, name):
    row_nulls = df.select([pl.col(c).is_null() for c in df.columns]).sum_horizontal()
    threshold = int(len(df.columns) * 0.9)
    holes = df.filter(row_nulls > threshold)
    print(f"{name} - Found {holes.height} rows with >90% nulls (potential holes).")
    if holes.height > 0:
        print(holes.head())

# %% [code]
check_sparsity(df_pos_raw, "Positive Cases")

# %% [code]
print("\n" + "-"*20)
check_sparsity(df_neg_raw, "Negative Cases")


# %% [markdown]
# ## 2. Identifying Column Name Issues
# We inspect the raw columns to determine the cleaning requirements.
#
# %% [code]
print("Positive Case Raw Columns:")
print(df_pos_raw.columns)

print("\nNegative Case Raw Columns:")
print(df_neg_raw.columns)


# %% [markdown]
# ### Observation: Dirty Names
# Data shows:
# 1. Spaces ("O2 Sat ")
# 2. Slashes ("Case/ID")
# 3. Mixed Case ("Age")
#
# Action: Implement `clean_colnames` to normalize to lowercase, no spaces/slashes.
#
# %% [code]
def clean_colnames(df):
    """Systematically clean column names for consistency."""
    def _clean(name):
        name = name.strip()
        name = name.replace("/", "-")
        name = name.replace(" ", "_")
        name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
        return name.lower()

    return df.rename({col: _clean(col) for col in df.columns})

# Proof on toy data
demo_df = pl.DataFrame({"Case/ID": [1], "O2 Sat ": [98], "Age": [50]})
print("Toy Before:", demo_df.columns)
print("Toy After:", clean_colnames(demo_df).columns)


# %% [code]
df_pos = clean_colnames(df_pos_raw)
df_neg = clean_colnames(df_neg_raw)

print("Columns cleaned for both datasets.")
print("\nPositive Cases (Cleaned Names) Head:")
print(df_pos.head())


# %% [markdown]
# ## 3. Handling Null-Only Rows
# We check for rows that are completely empty (common in Excel exports).
#
# %% [code]
# Positive Before
pos_empty = df_pos.filter(pl.all_horizontal(pl.all().is_null())).height
print(f"Positive empty rows: {pos_empty}")

df_pos = df_pos.filter(~pl.all_horizontal(pl.all().is_null()))

# Negative Before
neg_empty = df_neg.filter(pl.all_horizontal(pl.all().is_null())).height
print(f"Negative empty rows: {neg_empty}")

df_neg = df_neg.filter(~pl.all_horizontal(pl.all().is_null()))

print(f"Null-filter complete. Shapes: Pos {df_pos.shape}, Neg {df_neg.shape}")
print("\nPositive Cases (Null-Filtered) Head:")
print(df_pos.head())


# %% [markdown]
# ## 4. Longitudinal Data Restoration
# As discovered in the Audit, the dataset uses an Excel merged-cell convention where the Alias is only present on the first row of a subject's records. 
# We must forward-fill the Alias to correctly associate measurements with their respective subjects.
#
# %% [code]
def restore_longitudinal_ids(df):
    if "alias" not in df.columns:
        return df
    
    # Forward fill the alias column to fill in the merged cell gaps
    df = df.with_columns(pl.col("alias").forward_fill())
    return df

# Proof on toy data
demo_long = pl.DataFrame({
    "alias": [1, None, None, 2, None],
    "val": [10, 11, 12, 20, 21]
})
print("Toy Before:\n", demo_long)
print("\nToy After:\n", restore_longitudinal_ids(demo_long))


# %% [code]
# Apply to datasets
df_pos = restore_longitudinal_ids(df_pos)
df_neg = restore_longitudinal_ids(df_neg)

print("\nPositive Cases (IDs Restored) Head:")
print(df_pos.select(["alias", "age"]).head(20))


# %% [markdown]
# ## 5. Deduplication (Exact Rows Only)
# We ensure cases are not duplicated. 
# NOTE: We only remove EXACT duplicates across all columns to preserve longitudinal history.
#
# %% [code]
def deduplicate_cases(df):
    return df.unique()

# Proof on toy data
demo_dup = pl.DataFrame({"id": [1, 1], "val": [10, 10]})
print("Toy Before:\n", demo_dup)
print("\nToy After:\n", deduplicate_cases(demo_dup))


# %% [code]
# Positive
print(f"Positive before dedup: {df_pos.height}")
df_pos = deduplicate_cases(df_pos)
print(f"Positive after dedup: {df_pos.height}")

# Negative
print(f"Negative before dedup: {df_neg.height}")
df_neg = deduplicate_cases(df_neg)
print(f"Negative after dedup: {df_neg.height}")

print("\nPositive Cases (Deduplicated) Head:")
print(df_pos.head())


# %% [markdown]
# ## 6. Semantic Alignment
# We identify and merge columns that represent the same feature.
#
# %% [code]
# Discovery: check for BMI and O2 variants
bmi_cols = [c for c in df_pos.columns if "bmi" in c]
o2_cols = [c for c in df_pos.columns if "o2" in c]
print(f"Found BMI cols: {bmi_cols}")
print(f"Found O2 cols: {o2_cols}")


# %% [markdown]
# ### Observation: Semantic Mismatch
# Variations like `bmicalc` and `o2sat` exist.
# Action: Implement `align_semantics` using a mapping dictionary.
#
# %% [code]
semantic_map = {
    "bmicalc": "bmi",
    "o2sat": "o2_sat",
    "o2_sat_": "o2_sat"
}

def align_semantics(df, mapping):
    return df.rename({k: v for k, v in mapping.items() if k in df.columns})

# Proof on toy data
demo_sem = pl.DataFrame({"bmicalc": [25], "o2sat": [95]})
print("Toy Before:", demo_sem.columns)
print("Toy After:", align_semantics(demo_sem, semantic_map).columns)


# %% [code]
df_pos = align_semantics(df_pos, semantic_map)
df_neg = align_semantics(df_neg, semantic_map)

print("Semantic alignment complete.")
print("\nPositive Cases (Aligned) Head:")
print(df_pos.head())


# %% [markdown]
# ## 7. Labeling
# Assign binary labels for class identification.
#
# %% [code]
df_pos = df_pos.with_columns(pl.lit(1).alias("label"))
df_neg = df_neg.with_columns(pl.lit(0).alias("label"))

print(f"Labels assigned. Pos sample: {df_pos.select('label').head(1).item()}, Neg sample: {df_neg.select('label').head(1).item()}")
print("\nPositive Cases (Labeled) Head:")
print(df_pos.head())


# %% [markdown]
# ## 8. Type Alignment & Final Combination
# We address schema mismatches by casting differing types to String.
#
# %% [code]
def align_types(df1, df2):
    """Cast columns to string if they have mismatched types."""
    s1, s2 = df1.schema, df2.schema
    mismatched = [col for col in s1.keys() if col in s2 and s1[col] != s2[col]]

    if mismatched:
        print(f"Mismatched types found: {mismatched}. Casting to string.")
        df1 = df1.with_columns([pl.col(col).cast(pl.String) for col in mismatched])
        df2 = df2.with_columns([pl.col(col).cast(pl.String) for col in mismatched])

    return df1, df2

# Proof on toy data
d1, d2 = pl.DataFrame({"a": [1]}), pl.DataFrame({"a": ["1"]})
print("Toy types before:", d1.schema["a"], d2.schema["a"])
d1_a, d2_a = align_types(d1, d2)
print("Toy types after:", d1_a.schema["a"], d2_a.schema["a"])


# %% [code]
df_pos, df_neg = align_types(df_pos, df_neg)

try:
    df_all = pl.concat([df_pos, df_neg])
    print("Concatenation successful!")
except Exception as e:
    print(f"Concat failed: {e}. Falling back to diagonal.")
    df_all = pl.concat([df_pos, df_neg], how="diagonal")

print(f"Final Dataset Shape: {df_all.shape}")
print("\nFinal Combined Dataset Head:")
print(df_all.head())
print("\nFinal Combined Dataset Tail:")
print(df_all.tail())


# %% [markdown]
# ## Final Verification
# Verify the distribution and the alignment of key features.
#
# %% [code]
print("Class Distribution:")
print(df_all.group_by("label").agg(pl.len().alias("count")))

features = ["label", "bmi", "o2_sat", "age"]
present_features = [f for f in features if f in df_all.columns]
print("\nFinal Aligned Sample:")
print(df_all.select(present_features).head(10))