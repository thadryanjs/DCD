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
# This notebook provides a guided tour of the combined positive and negative case datasets to verify structure, alignment, and data quality.
#
# %% [code]
import polars as pl
from pathlib import Path
import re

# Reuse processing logic from 00_process-dataset.py
DATA_DIR = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data")

def clean_colnames(df: pl.DataFrame) -> pl.DataFrame:
    def _clean(name):
        name = name.strip().replace("/", "-").replace(" ", "_")
        name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
        return name.lower()
    return df.rename({col: _clean(col) for col in df.columns})

def load_and_prep():
    df_pos = pl.read_excel(DATA_DIR / "positive-cases.xlsx").filter(~pl.all_horizontal(pl.all().is_null()))
    df_neg = pl.read_excel(DATA_DIR / "negative-cases.xlsx").filter(~pl.all_horizontal(pl.all().is_null()))
    
    df_pos = clean_colnames(df_pos)
    df_neg = clean_colnames(df_neg)
    
    semantic_map = {"bmicalc": "bmi", "o2sat": "o2_sat", "o2_sat_": "o2_sat"}
    df_pos = df_pos.rename({k: v for k, v in semantic_map.items() if k in df_pos.columns}).with_columns(pl.lit(1).alias("label"))
    df_neg = df_neg.rename({k: v for k, v in semantic_map.items() if k in df_neg.columns}).with_columns(pl.lit(0).alias("label"))
    
    # Align types for concatenation
    s1, s2 = df_pos.schema, df_neg.schema
    mismatched = [col for col in s1.keys() if col in s2 and s1[col] != s2[col]]
    df_pos = df_pos.with_columns([pl.col(col).cast(pl.String) for col in mismatched])
    df_neg = df_neg.with_columns([pl.col(col).cast(pl.String) for col in mismatched])
    
    return pl.concat([df_pos, df_neg])

df_all = load_and_prep()


# %% [markdown]
# ## 1. High Level Overview
# First, we check the dimensions and the class balance.
#
# %% [code]
print(f"Dataset Shape: {df_all.shape}")
print("\nClass Distribution:")
print(df_all.group_by("label").agg(pl.len().alias("count")))


# %% [markdown]
# ## 2. Column Taxonomy
# The dataset contains 49 columns. We can group them into logical categories to understand the feature space.
#
# %% [code]
categories = {
    "Demographics": ["alias", "age", "ageunit", "bmi", "sex"],
    "Institutional": ["hospital", "unit", "date-time"],
    "Clinical Vitals": ["ph", "pco2", "po2", "hco3", "be", "o2_sat", "fio2", "rate", "tv", "peep", "pip", "apnea"],
    "DCD Process": ["controlled_dcd", "enterordatetime", "dcd_nrp_intended", "was_heparin_administered_", "heparin_dose_units"],
    "Timeline": [col for col in df_all.columns if "date-time" in col or "time" in col or "to_" in col],
    "Outcome/Labels": ["label", "did_patient_expire_withintimeframe"]
}

print("Column Categories:")
for cat, cols in categories.items():
    present = [c for c in cols if c in df_all.columns]
    print(f"{cat} ({len(present)}): {present}")


# %% [markdown]
# ## 3. Data Quality & Missingness
# We examine how many nulls exist per column to identify "sparse" features.
#
# %% [code]
null_counts = df_all.null_count()
# Melt the null counts into a long format for easier reading
null_df = null_counts.unpivot(variable_name="column", value_name="null_count")
null_df = null_df.with_columns(
    (pl.col("null_count") / df_all.height * 100).alias("null_pct")
).sort("null_pct", descending=True)

print("Top 10 most sparse columns:")
print(null_df.head(10))


# %% [markdown]
# ## 4. Feature Samples
# Finally, we look at a few specific columns to verify the data types and values.
#
# %% [code]
sample_cols = ["age", "bmi", "o2_sat", "dcd_nrp_type", "label"]
print("Sample of key features:")
print(df_all.select(sample_cols).head(10))
