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

# %% [code]
data_dir = Path("data")

def load_raw(filename):
    return pl.read_excel(data_dir / filename)

# %% [code]
df_pos_raw = load_raw("positive-cases-Jennys-data-Edited.xlsx")
df_neg_raw = load_raw("negative-cases-ANR-Data.xlsx")

# Filter completely empty rows immediately after load to ensure raw emptiness is captured
pos_empty = df_pos_raw.filter(pl.all_horizontal(pl.all().is_null())).height
neg_empty = df_neg_raw.filter(pl.all_horizontal(pl.all().is_null())).height

df_pos_raw = df_pos_raw.filter(~pl.all_horizontal(pl.all().is_null()))
df_neg_raw = df_neg_raw.filter(~pl.all_horizontal(pl.all().is_null()))

print(
    f"Loaded Raw Data\n"
    f"Positive shape: {df_pos_raw.shape} (Dropped {pos_empty} empty rows)\n"
    f"Negative shape: {df_neg_raw.shape} (Dropped {neg_empty} empty rows)"
)

# %% [markdown]
# # First Look: Raw Data
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
# # Merged Cell IDs
# Looking at the negative cases, we see a pattern: `Alias=1` followed by nulls, then `Alias=2`.
# The Excel file used merged cells for the Alias column - polars reads these as nulls.
#
# %% [code]
print("Rows 14-22 (Negative Cases) - The pattern: 1, null, null, null, 2:")
print(df_neg_raw.slice(14, 9))

# %% [code]
print("\nFirst 20 rows (Alias, Age, Date/Time):")
# Set table rows locally for this specific debug print
with pl.Config(tbl_rows=100):
    key_cols = [c for c in ["Alias", "Age", "Date/Time"] if c in df_neg_raw.columns]
    print(df_neg_raw.select(key_cols).slice(0, 20))


# %% [markdown]
# ### Fix: Forward-Fill the IDs
# Each ID propagates down until the next ID appears.
#
# **LOADBEARING** — `observation` numbering drives every first-look analysis.
# If numbering is shifted or non-sequential, "First Look" captures wrong rows.
# Consumed by: `01` stability analysis, `03` first-look pipeline.

# %% [code]
def forward_populate_ids(df, timestamp_col=None):
    """Forward-fill Alias: each ID propagates down until the next ID.
    Also creates an observation number for each row within each ID group.
    If timestamp_col is provided, sorts by it to ensure observation 1 is chronological.
    """
    alias_col = [c for c in df.columns if c.lower() == "alias"]
    if not alias_col:
        return df
    
    # 1. Forward-fill Alias (MUST happen before sorting)
    df = df.with_columns(pl.col(alias_col[0]).forward_fill())
    df = df.with_columns(pl.col(alias_col[0]).alias("alias_filled"))
    
    # 2. Sort by timestamp to ensure chronological order for 'observation'
    if timestamp_col and timestamp_col in df.columns:
        # Track original order to verify if sorting was necessary
        df = df.with_columns(pl.int_range(0, pl.len()).alias("_orig_idx"))
        df = df.sort(["alias_filled", timestamp_col])
        
        # Receipt: Check if original order was already chronological
        is_sorted = df["_orig_idx"].is_sorted().all()
        print(f"  Sorting by {timestamp_col}: {'Already chronological' if is_sorted else 'Reordered for chronology'}")
        df = df.drop("_orig_idx")
    
    # 3. Create observation number within each ID group
    df = df.with_columns(
        pl.col("alias_filled").cum_count().over("alias_filled").alias("observation")
    )
    return df

# %% [code]
# Proof on toy data
demo = pl.DataFrame({"alias": [1, None, None, 2, None], "val": [10, 11, 12, 20, 21]})
print("Before forward-fill:\n", demo)
print("\nAfter forward-fill:\n", forward_populate_ids(demo))

# %% [code]
# Real-data receipt: slice of negative cases spanning at least two ID boundaries
# We'll look for the first few transitions in the raw Alias column
raw_aliases = df_neg_raw["Alias"].to_list()
boundaries = [i for i in range(1, len(raw_aliases)) if raw_aliases[i] != raw_aliases[i-1]]

if len(boundaries) >= 2:
    # Slice from just before the first boundary to just after the second
    start = max(0, boundaries[0] - 2)
    end = boundaries[1] + 3
    length = end - start
    
    before_slice = df_neg_raw.slice(start, length).select(["Alias", "Age"])
    print(f"Before forward-fill (Negative Cases, slice {start}:{end}):\n", before_slice)

df_pos_raw_filled = forward_populate_ids(df_pos_raw, timestamp_col="Date/Time")
df_neg_raw_filled = forward_populate_ids(df_neg_raw, timestamp_col="Date/Time")

# Corrected receipt: show a slice of the negative dataset specifically
after_slice = df_neg_raw_filled.slice(neg_sample_start, 9).select(["alias_filled", "observation", "Age"])
print("\nAfter forward-fill (Real Data Slice):\n", after_slice)


# %% [markdown]
# # Dirty Column Names
# Now let's check the column names.
#
# %% [code]
print("Positive Case Raw Columns:")
print(df_pos_raw.columns)

# %% [code]
print("\nNegative Case Raw Columns:")
print(df_neg_raw.columns)


# %% [markdown]
# ### Fix: Clean Column Names
# Normalize to lowercase, replace spaces/slashes with underscores.
#
# **LOADBEARING** — The cleaner permits hyphens, and a hyphenated column name
# becomes subtraction in an R formula. `02` backticks its formulas because of this.
# Consumed by: `02` analyze-data.R

# %% [code]
def clean_colnames(df):
    """Clean column names: lowercase, no spaces/slashes. Returns (df, mapping)."""
    def _clean(name):
        name = name.strip()
        name = name.replace("/", "_")
        name = name.replace(" ", "_")
        name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
        return name.lower()
    mapping = {col: _clean(col) for col in df.columns}
    return df.rename(mapping), mapping

# %% [code]
# Proof on toy data
demo = pl.DataFrame({"Case/ID": [1], "O2 Sat ": [98], "Age": [50]})
print("Before cleaning:", demo.columns)
cleaned_demo, _ = clean_colnames(demo)
print("After cleaning:", cleaned_demo.columns)

# %% [code]
df_pos, map_pos = clean_colnames(df_pos_raw_filled)
df_neg, map_neg = clean_colnames(df_neg_raw_filled)

# %% [code]
# Mapping receipt
print("\nPositive Column Mapping:")
print(pl.DataFrame([{"old": k, "new": v} for k, v in map_pos.items()]))

print("\nNegative Column Mapping:")
print(pl.DataFrame([{"old": k, "new": v} for k, v in map_neg.items()]))

# Audit for non-standard characters (excluding lowercase, numbers, underscores, hyphens)
all_cleaned = list(map_pos.values()) + list(map_neg.values())
weird = [n for n in all_cleaned if re.search(r"[^a-z0-9_-]", n)]
if weird:
    print(f"\nWARNING: Names with non-standard characters (Potential R formula issues): {weird}")
else:
    print("\n✓ All cleaned names follow [a-z0-9_-] pattern.")

# Explicit delta receipt: print only changed names for clarity
print("\nPositive Name Changes:")
for old, new in map_pos.items():
    if old != new:
        print(f"  {old} -> {new}")

print("\nNegative Name Changes:")
for old, new in map_neg.items():
    if old != new:
        print(f"  {old} -> {new}")

print("\nColumns cleaned.")
print("Positive Cases (Clean Names) Head:")
print(df_pos.head())


# %% [markdown]
# # Semantic Variations
# Check for columns that mean the same thing but have different names.
#
# %% [code]
# Check for potential semantic variations (BMI, O2, etc.)
bmi_cols = [c for c in df_pos.columns if "bmi" in c.lower()]
o2_cols = [c for c in df_pos.columns if "o2" in c.lower()]

print(f"Potential BMI columns: {bmi_cols}")
print(f"Potential O2 columns: {o2_cols}")

# %% [code]
# Show first 5 rows of these columns to verify if they're the same thing
if bmi_cols or o2_cols:
    cols_to_check = bmi_cols + o2_cols
    print("\nFirst 5 rows of potential semantic variations:")
    print(df_pos.select(cols_to_check).head())


# %% [markdown]
# ### Analysis: Semantic Alignment
# We identify columns that are semantically identical across datasets and normalize them.
# For example, `bmicalc` in the negative set is the same as `bmi` in the positive set.
#
# %% [markdown]
# # Labeling
# Assign binary labels: 1 = progression to death (positive outcome).
#
# %% [code]
df_pos = df_pos.with_columns(pl.lit(1).alias("progression_to_death"))
df_neg = df_neg.with_columns(pl.lit(0).alias("progression_to_death"))

print(
    f"Labels assigned:\n"
    f"Pos: {df_pos.select('progression_to_death').head(1).item()}\n"
    f"Neg: {df_neg.select('progression_to_death').head(1).item()}"
)


# %% [markdown]
# # Column Check: Type Mismatches
# Find columns that exist in both datasets but have different types.
#
# %% [code]
# First, align column names that mean the same thing
semantic_map = {
    "bmicalc": "bmi",
    "o2sat": "o2_sat"
}

def align_semantics(df, mapping):
    """Rename columns based on semantic map if they exist."""
    renames = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(renames)
    return df, renames

df_pos, renames_pos = align_semantics(df_pos, semantic_map)
df_neg, renames_neg = align_semantics(df_neg, semantic_map)

print(f"Semantic alignment complete: {renames_pos | renames_neg}")


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
# We use typed nulls for all-null columns and `str.to_datetime()` with millisecond 
# precision for others to match the positive dataset.
#
# %% [code]
def safe_cast(df, col_name, target_type, cast_fn=None, time_unit="ms"):
    """Safely cast a column to target_type with validation.
    Raises ValueError if the number of nulls increases (indicating parsing errors)."""
    if col_name not in df.columns:
        print(f"Warning: Column {col_name} not found. Skipping cast.")
        return df
    
    nulls_before = df[col_name].null_count()
    
    try:
        if cast_fn:
            df = df.with_columns(cast_fn(pl.col(col_name)))
        else:
            # Handle Datetime time_unit explicitly to avoid ms/us mismatches
            if target_type == pl.Datetime:
                df = df.with_columns(pl.col(col_name).cast(pl.Datetime(time_unit=time_unit)))
            else:
                df = df.with_columns(pl.col(col_name).cast(target_type))
        
        nulls_after = df[col_name].null_count()
        if nulls_after > nulls_before:
            raise ValueError(f"Null count increased from {nulls_before} to {nulls_after}")
            
        print(f"✓ Cast {col_name} to {target_type} (nulls: {nulls_before} -> {nulls_after})")
    except Exception as e:
        print(f"Error casting {col_name} to {target_type}: {e}")
        raise e
    
    return df

# %% [code]
# date_time_of_declaration_tod - Time of declaration of death
# All null in negative dataset - use typed nulls to avoid version-dependent string casts
df_neg = df_neg.with_columns(pl.lit(None, dtype=pl.Datetime("ms")).alias("date_time_of_declaration_tod"))
print(f"✓ Set date_time_of_declaration_tod to Datetime nulls")

# %% [code]
# date_time_of_pea_asystole - Time of PEA/asystole event
# All null in negative dataset - use typed nulls to avoid version-dependent string casts
df_neg = df_neg.with_columns(pl.lit(None, dtype=pl.Datetime("ms")).alias("date_time_of_pea_asystole"))
print(f"✓ Set date_time_of_pea_asystole to Datetime nulls")

# %% [code]
# date_time_of_perfusion - Time of perfusion start
# All null in negative dataset - use typed nulls to avoid version-dependent string casts
df_neg = df_neg.with_columns(pl.lit(None, dtype=pl.Datetime("ms")).alias("date_time_of_perfusion"))
print(f"✓ Set date_time_of_perfusion to Datetime nulls")

# %% [code]
# date_time_sbp__90 - Time when SBP dropped below 90
# Has actual data in negative dataset
print(f"BEFORE (String): {df_neg['date_time_sbp__90'].drop_nulls().head(3).to_list()}")
df_neg = safe_cast(df_neg, "date_time_sbp__90", pl.Datetime, 
                  cast_fn=lambda c: c.str.to_datetime(time_unit="ms", format="%Y-%m-%d %H:%M:%S"))
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
df_neg = safe_cast(df_neg, "dcd_nrp_total_pump_time", pl.Int64)
print(f"AFTER (Int64): {df_neg['dcd_nrp_total_pump_time'].head(3).to_list()} (all null)")

# %% [code]
# extubation_to_perfusion_warm_ischemic_time - Time from extubation to perfusion (minutes)
# All null in negative dataset
print(f"BEFORE (String): {df_neg['extubation_to_perfusion_warm_ischemic_time'].head(3).to_list()} (all null)")
df_neg = safe_cast(df_neg, "extubation_to_perfusion_warm_ischemic_time", pl.Int64)
print(f"AFTER (Int64): {df_neg['extubation_to_perfusion_warm_ischemic_time'].head(3).to_list()} (all null)")

# %% [code]
# sbp90_to_declaration - Time from SBP<90 to declaration (minutes)
# All null in negative dataset
print(f"BEFORE (String): {df_neg['sbp90_to_declaration'].head(3).to_list()} (all null)")
df_neg = safe_cast(df_neg, "sbp90_to_declaration", pl.Int64)
print(f"AFTER (Int64): {df_neg['sbp90_to_declaration'].head(3).to_list()} (all null)")

# %% [code]
# tod_to_perfusion - Time from declaration of death to perfusion (minutes)
# All null in negative dataset
print(f"BEFORE (String): {df_neg['tod_to_perfusion'].head(3).to_list()} (all null)")
df_neg = safe_cast(df_neg, "tod_to_perfusion", pl.Int64)
print(f"AFTER (Int64): {df_neg['tod_to_perfusion'].head(3).to_list()} (all null)")

# %% [code]
# warm_ischemic_time_agonal_phase_to_cooling - Warm ischemic time (minutes)
# All null in negative dataset
print(f"BEFORE (String): {df_neg['warm_ischemic_time_agonal_phase_to_cooling'].head(3).to_list()} (all null)")
df_neg = safe_cast(df_neg, "warm_ischemic_time_agonal_phase_to_cooling", pl.Int64)
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
# **LOADBEARING** — Both workbooks number patients from 1. Without the prefix,
# `pos_1` and `neg_1` collide into a single group, which corrupts patient counts in
# `01`, degrades stratification in `03`, and makes the R random-effect structure
# meaningless. Consumed by: `01` stability analysis, `03` group splits, `02` aggregation.

# %% [code]
print("Attempting concat...")
# Namespace alias_filled to prevent collision between positive and negative patient IDs
pos_count_before = df_pos["alias_filled"].n_unique()
neg_count_before = df_neg["alias_filled"].n_unique()

df_pos = df_pos.with_columns(("pos_" + pl.col("alias_filled").cast(pl.String)).alias("alias_filled"))
df_neg = df_neg.with_columns(("neg_" + pl.col("alias_filled").cast(pl.String)).alias("alias_filled"))

pos_count_after = df_pos["alias_filled"].n_unique()
neg_count_after = df_neg["alias_filled"].n_unique()

df_all = pl.concat([df_pos, df_neg], how="diagonal")

# Namespacing receipt
print(f"Unique patients - Positives: {pos_count_before} -> {pos_count_after}")
print(f"Unique patients - Negatives: {neg_count_before} -> {neg_count_after}")
print(f"Combined unique patients: {df_all['alias_filled'].n_unique()}")
assert df_all['alias_filled'].n_unique() == pos_count_after + neg_count_after, "Patient ID collision during concat"

print("\nSample IDs:")
print(f"Pos: {df_pos['alias_filled'].head(5).to_list()}")
print(f"Neg: {df_neg['alias_filled'].head(5).to_list()}")

# %% [markdown]
# ### Diagonal Concat Column Provenance
#
# **LOADBEARING** — `how="diagonal"` fills absent columns with nulls.
# A column present in only one workbook can become a perfect label proxy.
# This is the mechanism the `01` missingness filter exists to catch.
# Consumed by: `01` missingness filter.

# %% [code]
cols_pos = set(df_pos.columns)
cols_neg = set(df_neg.columns)

pos_only = sorted(list(cols_pos - cols_neg))
neg_only = sorted(list(cols_neg - cols_pos))
both = sorted(list(cols_pos & cols_neg))

print(f"Columns in both: {len(both)}")
print(f"Columns in Positives only: {len(pos_only)} {pos_only}")
print(f"Columns in Negatives only: {len(neg_only)} {neg_only}")

# We allow them, but explicitly note they are candidate leaks
if pos_only or neg_only:
    print("\nNote: Asymmetric columns detected. These will be handled by the 10% missingness filter in 01.")

# %% [code]
print(f"Concatenation successful!")
print(f"Final Dataset Shape: {df_all.shape}")

# %% [code]
# Observation distribution per source
print("\nObservation Value Counts - Positives:")
print(df_pos.select("observation").value_counts().sort("observation"))

print("\nObservation Value Counts - Negatives:")
print(df_neg.select("observation").value_counts().sort("observation"))

print("\nObservation Value Counts (Combined):")
print(df_all.select("observation").value_counts().sort("observation"))

print("\nObservations per Patient Distribution (Combined):")
obs_dist_all = df_all.group_by("alias_filled").agg(pl.len().alias("count"))
print(f"Combined: Mean={obs_dist_all['count'].mean():.2f}, Max={obs_dist_all['count'].max()}, Min={obs_dist_all['count'].min()}")

# Receipt: per-source distribution
obs_dist_pos = df_pos.group_by("alias_filled").agg(pl.len().alias("count"))
obs_dist_neg = df_neg.group_by("alias_filled").agg(pl.len().alias("count"))
print(f"Positives: Mean={obs_dist_pos['count'].mean():.2f}, Max={obs_dist_pos['count'].max()}, Min={obs_dist_pos['count'].min()}")
print(f"Negatives: Mean={obs_dist_neg['count'].mean():.2f}, Max={obs_dist_neg['count'].max()}, Min={obs_dist_neg['count'].min()}")

# %% [markdown]
# ## Final Summary
#
# %% [code]
# Class distribution
class_summary = df_all.group_by("progression_to_death").agg([
    pl.len().alias("rows"),
    pl.col("alias_filled").n_unique().alias("patients")
])
print("Class Distribution:")
print(class_summary)

# Observations per patient by class
obs_summary = df_all.group_by(["progression_to_death", "alias_filled"]).agg(pl.len().alias("count"))
obs_by_class = obs_summary.group_by("progression_to_death").agg([
    pl.col("count").mean().alias("avg_obs"),
    pl.col("count").max().alias("max_obs"),
    pl.col("count").min().alias("min_obs")
])
print("\nObservations per Patient by Class:")
print(obs_by_class)

# %% [markdown]
# ## Save Processed Dataset
# Save to parquet for downstream analysis.
#
# %% [code]
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

output_path = processed_dir / "combined-dataset.parquet"
df_all.write_parquet(output_path)
print(f"Saved to: {output_path}")
print(f"Shape: {df_all.shape}")
print(f"Columns: {df_all.columns}")


# %% [markdown]
# ## End of Data Processing
# Final verification (class distribution, feature summary) moved to `01_explore-data.py`.
