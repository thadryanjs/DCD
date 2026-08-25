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

DATA_DIR = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data")

def load_raw(filename: str) -> pl.DataFrame:
    return pl.read_excel(DATA_DIR / filename)

df_pos_raw = load_raw("positive-cases.xlsx")
df_neg_raw = load_raw("negative-cases.xlsx")

print(f"Raw Shapes -> Positive: {df_pos_raw.shape}, Negative: {df_neg_raw.shape}")


# %% [markdown]
# ## 1. Identifying Column Name Issues
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
def clean_colnames(df: pl.DataFrame) -> pl.DataFrame:
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


# %% [markdown]
# ## 2. Handling Null-Only Rows
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


# %% [markdown]
# ## 3. Deduplication
# We ensure cases are not duplicated, ignoring the 'alias' column if it exists.
#
# %% [code]
def deduplicate_cases(df: pl.DataFrame) -> pl.DataFrame:
    if "alias" not in df.columns:
        return df
    subset = [c for c in df.columns if c != "alias"]
    return df.unique(subset=subset)

# Proof on toy data
demo_dup = pl.DataFrame({"id": [1, 1], "val": [10, 10], "alias": ["A", "B"]})
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


# %% [markdown]
# ## 4. Semantic Alignment
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

def align_semantics(df: pl.DataFrame, mapping: dict) -> pl.DataFrame:
    return df.rename({k: v for k, v in mapping.items() if k in df.columns})

# Proof on toy data
demo_sem = pl.DataFrame({"bmicalc": [25], "o2sat": [95]})
print("Toy Before:", demo_sem.columns)
print("Toy After:", align_semantics(demo_sem, semantic_map).columns)


# %% [code]
df_pos = align_semantics(df_pos, semantic_map)
df_neg = align_semantics(df_neg, semantic_map)

print("Semantic alignment complete.")


# %% [markdown]
# ## 5. Labeling
# Assign binary labels for class identification.
#
# %% [code]
df_pos = df_pos.with_columns(pl.lit(1).alias("label"))
df_neg = df_neg.with_columns(pl.lit(0).alias("label"))

print(f"Labels assigned. Pos sample: {df_pos.select('label').head(1).item()}, Neg sample: {df_neg.select('label').head(1).item()}")


# %% [markdown]
# ## 6. Type Alignment & Final Combination
# We address schema mismatches by casting differing types to String.
#
# %% [code]
def align_types(df1: pl.DataFrame, df2: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
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
