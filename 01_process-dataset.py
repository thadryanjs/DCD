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
# # Process Dataset: Ophthalmology Cases
# This notebook implements the cleaning pipeline identified during the Data Tour.
# We follow a "proof" approach: every transformation is demonstrated and verified.
#
# %% [code]
import polars as pl
from pathlib import Path
import re

DATA_DIR = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data")

def load_raw(filename: str) -> pl.DataFrame:
    return pl.read_excel(DATA_DIR / filename)

df_pos = load_raw("positive-cases.xlsx")
df_neg = load_raw("negative-cases.xlsx")

print(f"Loaded positive: {df_pos.shape}")
print(f"Loaded negative: {df_neg.shape}")


# %% [markdown]
# ## 1. Column Name Cleaning
# We need a consistent naming convention: lowercase, no spaces, no slashes.
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

# Demo
demo_df = pl.DataFrame({"Case/ID": [1], "O2 Sat ": [98], "Age": [50]})
print("Demo Before:")
print(demo_df.columns)

cleaned_demo = clean_colnames(demo_df)
print("\nDemo After:")
print(cleaned_demo.columns)


# %% [code]
# Apply to datasets
df_pos = clean_colnames(df_pos)
df_neg = clean_colnames(df_neg)

print("Positive columns cleaned.")
print("Negative columns cleaned.")


# %% [markdown]
# ## 2. Removing Null-Only Rows
# Some spreadsheets contain empty rows at the end or as spacers.
#
# %% [code]
# Positive Before
print(f"Positive rows before null-filter: {df_pos.height}")

df_pos = df_pos.filter(~pl.all_horizontal(pl.all().is_null()))

# Positive After
print(f"Positive rows after null-filter: {df_pos.height}")


# %% [code]
# Negative Before
print(f"Negative rows before null-filter: {df_neg.height}")

df_neg = df_neg.filter(~pl.all_horizontal(pl.all().is_null()))

# Negative After
print(f"Negative rows after null-filter: {df_neg.height}")


# %% [markdown]
# ## 3. Deduplication
# We check for duplicate records, ignoring the 'alias' column if it exists.
#
# %% [code]
def deduplicate_cases(df: pl.DataFrame) -> pl.DataFrame:
    if "alias" not in df.columns:
        return df
    
    subset = [c for c in df.columns if c != "alias"]
    return df.unique(subset=subset)

# Demo
demo_dup = pl.DataFrame({
    "id": [1, 1],
    "val": [10, 10],
    "alias": ["A", "B"]
})
print("Demo Before:\n", demo_dup)
print("\nDemo After:\n", deduplicate_cases(demo_dup))


# %% [code]
# Positive Before
print(f"Positive rows before deduplication: {df_pos.height}")

df_pos = deduplicate_cases(df_pos)

# Positive After
print(f"Positive rows after deduplication: {df_pos.height}")


# %% [code]
# Negative Before
print(f"Negative rows before deduplication: {df_neg.height}")

df_neg = deduplicate_cases(df_neg)

# Negative After
print(f"Negative rows after deduplication: {df_neg.height}")


# %% [markdown]
# ## 4. Semantic Alignment
# Align columns that represent the same feature but have different names.
#
# %% [code]
semantic_map = {
    "bmicalc": "bmi",
    "o2sat": "o2_sat",
    "o2_sat_": "o2_sat" 
}

def align_semantics(df: pl.DataFrame, mapping: dict) -> pl.DataFrame:
    return df.rename({k: v for k, v in mapping.items() if k in df.columns})

# Demo
demo_sem = pl.DataFrame({"bmicalc": [25], "o2sat": [95]})
print("Demo Before:\n", demo_sem.columns)

aligned_demo = align_semantics(demo_sem, semantic_map)
print("\nDemo After:\n", aligned_demo.columns)


# %% [code]
# Apply alignment
df_pos = align_semantics(df_pos, semantic_map)
df_neg = align_semantics(df_neg, semantic_map)

print("Semantic alignment complete.")


# %% [markdown]
# ## 5. Labeling
# Add binary labels for positive (1) and negative (0) cases.
#
# %% [code]
df_pos = df_pos.with_columns(pl.lit(1).alias("label"))
df_neg = df_neg.with_columns(pl.lit(0).alias("label"))

print("Labels added.")
print(f"Positive label sample: {df_pos.select('label').head(1).item()}")
print(f"Negative label sample: {df_neg.select('label').head(1).item()}")


# %% [markdown]
# ## 6. Type Alignment & Concatenation
# To avoid SchemaErrors, we cast columns to String if types differ between the two sets.
#
# %% [code]
def align_types(df1: pl.DataFrame, df2: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cast columns to string if they have mismatched types between two dataframes."""
    s1 = df1.schema
    s2 = df2.schema
    
    mismatched = []
    for col in s1.keys():
        if col in s2 and s1[col] != s2[col]:
            mismatched.append(col)
            
    if mismatched:
        print(f"Mismatched types found in: {mismatched}. Casting to string.")
        df1 = df1.with_columns([pl.col(col).cast(pl.String) for col in mismatched])
        df2 = df2.with_columns([pl.col(col).cast(pl.String) for col in mismatched])
        
    return df1, df2

# Demo
d1 = pl.DataFrame({"a": [1]})
d2 = pl.DataFrame({"a": ["1"]})
print("Demo types before:", d1.schema["a"], d2.schema["a"])

d1_a, d2_a = align_types(d1, d2)
print("Demo types after:", d1_a.schema["a"], d2_a.schema["a"])


# %% [code]
# Apply type alignment
df_pos, df_neg = align_types(df_pos, df_neg)

# Final Concatenation
try:
    df_all = pl.concat([df_pos, df_neg])
    print("Concatenation successful!")
except Exception as e:
    print(f"Concat failed: {e}. Falling back to diagonal.")
    df_all = pl.concat([df_pos, df_neg], how="diagonal")

print(f"Final Combined Dataset Shape: {df_all.shape}")


# %% [markdown]
# ## Final Verification
# Check the final distribution and a sample of the aligned features.
#
# %% [code]
print("Class Distribution:")
print(df_all.group_by("label").agg(pl.len().alias("count")))

# Sample of key aligned features
features = ["label", "bmi", "o2_sat", "age"]
present_features = [f for f in features if f in df_all.columns]
print("\nSample of key features:")
print(df_all.select(present_features).head(10))
