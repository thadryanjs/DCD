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
# # Dataset Exploration: Ophthalmology Cases
# Initial poke around the positive and negative case datasets.
#
# %% [code]
import polars as pl
from pathlib import Path
import re

DATA_DIR = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data")

def clean_colnames(df: pl.DataFrame) -> pl.DataFrame:
    """Systematically clean column names for consistency."""
    def _clean(name):
        name = name.strip()
        name = name.replace("/", "-")
        name = name.replace(" ", "_")
        name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
        return name.lower()

    return df.rename({col: _clean(col) for col in df.columns})

def load_case_data(filename: str):
    """Load xlsx case data using polars."""
    path = DATA_DIR / filename
    print(f"Loading {filename} from {path}...")
    return pl.read_excel(path)


# %% [code]
# Load positive cases
df_pos = load_case_data("positive-cases.xlsx")
df_pos = df_pos.filter(~pl.all_horizontal(pl.all().is_null()))
df_pos = clean_colnames(df_pos)

# Identify potential spacer rows (e.g., age is null)
null_age_pos = df_pos.filter(pl.col("age").is_null()).height
print(f"Positive cases: {df_pos.height} rows, {null_age_pos} have null age.")

if "alias" in df_pos.columns:
    orig_h = df_pos.height
    df_pos = df_pos.unique(subset=[c for c in df_pos.columns if c != "alias"])
    print(f"Positive cases: {orig_h} -> {df_pos.height} after deduplicating records (ignoring alias).")

print(f"Positive cases final shape: {df_pos.shape}")
print(df_pos.head())


# %% [code]
# Load negative cases
df_neg = load_case_data("negative-cases.xlsx")
df_neg = df_neg.filter(~pl.all_horizontal(pl.all().is_null()))
df_neg = clean_colnames(df_neg)

# Identify potential spacer rows (e.g., age is null)
null_age_neg = df_neg.filter(pl.col("age").is_null()).height
print(f"Negative cases: {df_neg.height} rows, {null_age_neg} have null age.")

if "alias" in df_neg.columns:
    orig_h = df_neg.height
    df_neg = df_neg.unique(subset=[c for c in df_neg.columns if c != "alias"])
    print(f"Negative cases: {orig_h} -> {df_neg.height} after deduplicating records (ignoring alias).")

print(f"Negative cases final shape: {df_neg.shape}")
print(df_neg.head())


# %% [markdown]
# ## Semantic Alignment
# Fix columns that are named differently but mean same thing.
#
# %% [code]
# Semantic map for alignment
semantic_map = {
    "bmicalc": "bmi",
    "o2sat": "o2_sat",
    "o2_sat_": "o2_sat" # Handle trailing space from "O2 Sat " -> "o2_sat_"
}

df_pos = df_pos.rename({k: v for k, v in semantic_map.items() if k in df_pos.columns})
df_neg = df_neg.rename({k: v for k, v in semantic_map.items() if k in df_neg.columns})

# Labeling
df_pos = df_pos.with_columns(pl.lit(1).alias("label"))
df_neg = df_neg.with_columns(pl.lit(0).alias("label"))

print("Aligned and labeled. Positive head:")
print(df_pos.select(["label", "bmi", "o2_sat"]).head() if "bmi" in df_pos.columns else "BMI not found")

print("\nAligned and labeled. Negative head:")
print(df_neg.select(["label", "bmi", "o2_sat"]).head() if "bmi" in df_neg.columns else "BMI not found")


# %% [code]
# Attempt concat. If type mismatch, find offending columns and cast to string.
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

df_pos, df_neg = align_types(df_pos, df_neg)

try:
    df_all = pl.concat([df_pos, df_neg])
    print(f"Combined dataset shape: {df_all.shape}")
except pl.exceptions.SchemaError as e:
    print(f"SchemaError during concat: {e}")
    print("Falling back to diagonal concat (fills missing/mismatched with nulls)")
    df_all = pl.concat([df_pos, df_neg], how="diagonal")
    print(f"Diagonal combined shape: {df_all.shape}")

print(df_all.group_by("label").agg(pl.len().alias("count")))
