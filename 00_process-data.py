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
# # Data Processing
# A step-by-step proof of the data pipeline. We discover issues in the raw data and fix them immediately.
#
# %% [code]
import polars as pl
from pathlib import Path
import re

data_dir = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data")

def load_raw(filename):
    return pl.read_excel(data_dir / filename)

df_pos_raw = load_raw("positive-cases-Jennys-data-Edited.xlsx")
df_neg_raw = load_raw("negative-cases-ANR-Data.xlsx")

print(f"Loaded -> Positive: {df_pos_raw.shape}, Negative: {df_neg_raw.shape}")

# %% [markdown]
# ## 1. First Look: Raw Data
# Let's see what we're working with.
#
# %% [code]
print("Positive Cases - Head:")
print(df_pos_raw.head())

# %% [code]
print("\nPositive Cases - Tail:")
print(df_pos_raw.tail())

# %% [code]
print("\nNegative Cases - Head:")
print(df_neg_raw.head())

# %% [code]
print("\nNegative Cases - Tail:")
print(df_neg_raw.tail())


# %% [markdown]
# ## 2. Merged Cell IDs
# Looking at the negative cases, we see a pattern: `Alias=1` followed by nulls, then `Alias=2`.
# The Excel file used merged cells for the Alias column - polars reads these as nulls.
#
# %% [code]
print("Rows 14-22 (Negative Cases) - The pattern: 1, null, null, null, 2:")
print(df_neg_raw.slice(14, 9))

# %% [code]
print("\nFirst 20 rows (Alias, Age, Date/Time):")
pl.Config.set_tbl_rows(100)
key_cols = [c for c in ["Alias", "Age", "Date/Time"] if c in df_neg_raw.columns]
print(df_neg_raw.select(key_cols).slice(0, 20))


# %% [markdown]
# ### Fix: Forward-Fill the IDs
# Each ID propagates down until the next ID appears.
#
# %% [code]
def forward_populate_ids(df):
    """Forward-fill Alias: each ID propagates down until the next ID.
    Also creates an observation number for each row within each ID group."""
    alias_col = [c for c in df.columns if c.lower() == "alias"]
    if not alias_col:
        return df
    df = df.with_columns(pl.col(alias_col[0]).forward_fill())
    # Create observation number within each ID group
    df = df.with_columns(
        pl.col(alias_col[0]).alias("alias_filled"),
        pl.int_range(0, pl.len()).cum_count().over(alias_col[0]).alias("observation")
    )
    return df

# Proof on toy data
demo = pl.DataFrame({"alias": [1, None, None, 2, None], "val": [10, 11, 12, 20, 21]})
print("Before:\n", demo)
print("\nAfter:\n", forward_populate_ids(demo))

# %% [code]
df_pos = forward_populate_ids(df_pos_raw)
df_neg = forward_populate_ids(df_neg_raw)

print("\nPositive Cases (IDs Forward-Filled) - First 20 rows:")
print(df_pos.select(["alias_filled", "observation", "Age"]).head(20))


# %% [markdown]
# ## 3. Dirty Column Names
# Now let's check the column names.
#
# %% [code]
print("Positive Case Raw Columns:")
print(df_pos_raw.columns)

print("\nNegative Case Raw Columns:")
print(df_neg_raw.columns)


# %% [markdown]
# ### Fix: Clean Column Names
# Normalize to lowercase, replace spaces/slashes with underscores.
#
# %% [code]
def clean_colnames(df):
    """Clean column names: lowercase, no spaces/slashes."""
    def _clean(name):
        name = name.strip()
        name = name.replace("/", "_")
        name = name.replace(" ", "_")
        name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
        return name.lower()
    return df.rename({col: _clean(col) for col in df.columns})

# Proof on toy data
demo = pl.DataFrame({"Case/ID": [1], "O2 Sat ": [98], "Age": [50]})
print("Before:", demo.columns)
print("After:", clean_colnames(demo).columns)

# %% [code]
df_pos = clean_colnames(df_pos)
df_neg = clean_colnames(df_neg)

print("\nColumns cleaned.")
print("Positive Cases (Clean Names) Head:")
print(df_pos.head())


# %% [markdown]
# ## 4. Empty Rows
# Check for completely empty rows (common in Excel exports).
#
# %% [code]
pos_empty = df_pos.filter(pl.all_horizontal(pl.all().is_null())).height
print(f"Positive empty rows: {pos_empty}")

neg_empty = df_neg.filter(pl.all_horizontal(pl.all().is_null())).height
print(f"Negative empty rows: {neg_empty}")

df_pos = df_pos.filter(~pl.all_horizontal(pl.all().is_null()))
df_neg = df_neg.filter(~pl.all_horizontal(pl.all().is_null()))

print(f"\nFiltered. Shapes: Pos {df_pos.shape}, Neg {df_neg.shape}")


# %% [markdown]
# ## 5. Semantic Variations
# Check for columns that mean the same thing but have different names.
#
# %% [code]
# Check for potential semantic variations (BMI, O2, etc.)
bmi_cols = [c for c in df_pos.columns if "bmi" in c.lower()]
o2_cols = [c for c in df_pos.columns if "o2" in c.lower()]

print(f"Potential BMI columns: {bmi_cols}")
print(f"Potential O2 columns: {o2_cols}")

# Show first 5 rows of these columns to verify if they're the same thing
if bmi_cols or o2_cols:
    cols_to_check = bmi_cols + o2_cols
    print("\nFirst 5 rows of potential semantic variations:")
    print(df_pos.select(cols_to_check).head())


# %% [markdown]
# ### Observation: All Columns Are Distinct
# The potential variations are actually different clinical measurements:
# - `pco2` = partial pressure of CO2
# - `po2` = partial pressure of O2  
# - `o2_sat` = oxygen saturation
# - `fio2` = fraction of inspired O2
# - `baseline_vitals_o2_sat` = baseline O2 saturation reading
# - `date_time_o2_sat__90` = timestamp when O2 sat dropped below 90%
#
# No semantic alignment needed.
#
# %% [markdown]
# ## 6. Labeling
# Assign binary labels: 1 = progression to death (positive outcome).
#
# %% [code]
df_pos = df_pos.with_columns(pl.lit(1).alias("progression_to_death"))
df_neg = df_neg.with_columns(pl.lit(0).alias("progression_to_death"))

print(f"Labels: Pos={df_pos.select('progression_to_death').head(1).item()}, Neg={df_neg.select('progression_to_death').head(1).item()}")


# %% [markdown]
# ## 7. Column Check: Type Mismatches
# Find columns that exist in both datasets but have different types.
#
# %% [code]
# First, align column names that mean the same thing
semantic_map = {
    "bmicalc": "bmi",
    "o2sat": "o2_sat"
}

def align_semantics(df, mapping):
    return df.rename({k: v for k, v in mapping.items() if k in df.columns})

df_pos = align_semantics(df_pos, semantic_map)
df_neg = align_semantics(df_neg, semantic_map)

print("Semantic alignment complete (bmicalc->bmi, o2sat->o2_sat).")


# %% [code]
# Now check for type mismatches
common_cols = set(df_pos.columns) & set(df_neg.columns)
mismatched = [(col, df_pos.schema[col], df_neg.schema[col]) 
              for col in sorted(common_cols) if df_pos.schema[col] != df_neg.schema[col]]

if not mismatched:
    print("No type mismatches found.")
else:
    print(f"Found {len(mismatched)} type mismatches:\n")
    for i, (col, dtype1, dtype2) in enumerate(mismatched, 1):
        pos_sample = df_pos[col].head(3).to_list()
        neg_sample = df_neg[col].head(3).to_list()
        print(f"{i}. {col}")
        print(f"   Positive: {dtype1} -> {pos_sample}")
        print(f"   Negative: {dtype2} -> {neg_sample}")
        print()


# %% [markdown]
# ### Handling Mismatches
# Each mismatched column is handled bespoke based on its semantic meaning.
#

# %% [markdown]
# #### Datetime Columns (String → Datetime[ms])
# These columns contain timestamps stored as strings in the negative dataset.
# Convert using `str.to_datetime()` with millisecond precision to match positive dataset.
#
# %% [code]
# date_time_of_declaration_tod - Time of declaration of death
# All null in negative dataset
print(f"BEFORE (String): {df_neg['date_time_of_declaration_tod'].head(3).to_list()} (all null)")
df_neg = df_neg.with_columns(pl.col("date_time_of_declaration_tod").str.to_datetime(time_unit="ms"))
print(f"AFTER (Datetime): {df_neg['date_time_of_declaration_tod'].head(3).to_list()} (all null)")

# %% [code]
# date_time_of_pea_asystole - Time of PEA/asystole event
# All null in negative dataset
print(f"BEFORE (String): {df_neg['date_time_of_pea_asystole'].head(3).to_list()} (all null)")
df_neg = df_neg.with_columns(pl.col("date_time_of_pea_asystole").str.to_datetime(time_unit="ms"))
print(f"AFTER (Datetime): {df_neg['date_time_of_pea_asystole'].head(3).to_list()} (all null)")

# %% [code]
# date_time_of_perfusion - Time of perfusion start
# All null in negative dataset
print(f"BEFORE (String): {df_neg['date_time_of_perfusion'].head(3).to_list()} (all null)")
df_neg = df_neg.with_columns(pl.col("date_time_of_perfusion").str.to_datetime(time_unit="ms"))
print(f"AFTER (Datetime): {df_neg['date_time_of_perfusion'].head(3).to_list()} (all null)")

# %% [code]
# date_time_sbp__90 - Time when SBP dropped below 90
# Has actual data in negative dataset
print(f"BEFORE (String): {df_neg['date_time_sbp__90'].drop_nulls().head(3).to_list()}")
df_neg = df_neg.with_columns(pl.col("date_time_sbp__90").str.to_datetime(time_unit="ms"))
print(f"AFTER (Datetime): {df_neg['date_time_sbp__90'].drop_nulls().head(3).to_list()}")


# %% [markdown]
# #### Integer Columns (String → Int64)
# These columns contain numeric duration/time values stored as strings in negative dataset.
# Cast to Int64 to match positive dataset. **Note: All values are null in negative dataset**.
#
# %% [code]
# dcd_nrp_total_pump_time - Total pump time during DCD/NRP (minutes)
# All null in negative dataset (NRP only for positive cases)
print(f"BEFORE (String): {df_neg['dcd_nrp_total_pump_time'].head(3).to_list()} (all null)")
df_neg = df_neg.with_columns(pl.col("dcd_nrp_total_pump_time").cast(pl.Int64))
print(f"AFTER (Int64): {df_neg['dcd_nrp_total_pump_time'].head(3).to_list()} (all null)")

# %% [code]
# extubation_to_perfusion_warm_ischemic_time - Time from extubation to perfusion (minutes)
# All null in negative dataset
print(f"BEFORE (String): {df_neg['extubation_to_perfusion_warm_ischemic_time'].head(3).to_list()} (all null)")
df_neg = df_neg.with_columns(pl.col("extubation_to_perfusion_warm_ischemic_time").cast(pl.Int64))
print(f"AFTER (Int64): {df_neg['extubation_to_perfusion_warm_ischemic_time'].head(3).to_list()} (all null)")

# %% [code]
# sbp90_to_declaration - Time from SBP<90 to declaration (minutes)
# All null in negative dataset
print(f"BEFORE (String): {df_neg['sbp90_to_declaration'].head(3).to_list()} (all null)")
df_neg = df_neg.with_columns(pl.col("sbp90_to_declaration").cast(pl.Int64))
print(f"AFTER (Int64): {df_neg['sbp90_to_declaration'].head(3).to_list()} (all null)")

# %% [code]
# tod_to_perfusion - Time from declaration of death to perfusion (minutes)
# All null in negative dataset
print(f"BEFORE (String): {df_neg['tod_to_perfusion'].head(3).to_list()} (all null)")
df_neg = df_neg.with_columns(pl.col("tod_to_perfusion").cast(pl.Int64))
print(f"AFTER (Int64): {df_neg['tod_to_perfusion'].head(3).to_list()} (all null)")

# %% [code]
# warm_ischemic_time_agonal_phase_to_cooling - Warm ischemic time (minutes)
# All null in negative dataset
print(f"BEFORE (String): {df_neg['warm_ischemic_time_agonal_phase_to_cooling'].head(3).to_list()} (all null)")
df_neg = df_neg.with_columns(pl.col("warm_ischemic_time_agonal_phase_to_cooling").cast(pl.Int64))
print(f"AFTER (Int64): {df_neg['warm_ischemic_time_agonal_phase_to_cooling'].head(3).to_list()} (all null)")


# %% [code]
print("\nType alignment complete.")


# %% [code]
# Verify alignment
print("Checking for remaining mismatches...")
mismatched_after = [(col, df_pos.schema[col], df_neg.schema[col]) 
                    for col in common_cols if df_pos.schema[col] != df_neg.schema[col]]

if mismatched_after:
    print(f"Still have {len(mismatched_after)} mismatches:")
    for col, t1, t2 in mismatched_after:
        print(f"  {col}: Pos={t1}, Neg={t2}")
else:
    print("All types aligned!")


# %% [markdown]
# ## Combine Datasets
#
# %% [code]
print("Attempting concat...")
df_all = pl.concat([df_pos, df_neg])
print(f"Concatenation successful!")
print(f"Final Dataset Shape: {df_all.shape}")


# %% [markdown]
# ## Save Processed Dataset
# Save to parquet for downstream analysis.
#
# %% [code]
processed_dir = Path("data/processed")
processed_dir.mkdir(exist_ok=True)

output_path = processed_dir / "combined-dataset.parquet"
df_all.write_parquet(output_path)
print(f"Saved to: {output_path}")
print(f"Shape: {df_all.shape}")
print(f"Columns: {df_all.columns}")


# %% [markdown]
# ## End of Data Processing
# Final verification (class distribution, feature summary) moved to `01_analysis.py`.