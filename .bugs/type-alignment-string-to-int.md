# Type Alignment Bug: String to Int64 Casting

## Date
2025-02-XX

## Description
When aligning types between positive and negative datasets, the type mismatch handler incorrectly casted columns to `String` instead of casting the `String` column to match the target type.

## Root Cause
In the type alignment loop, the condition `pos_type == pl.String or neg_type == pl.String` would be true when EITHER type was String. However, the code always casted `df_neg` to String, which was wrong when:
- `pos_type = Int64` (target)
- `neg_type = String` (source)

The fix: cast `neg_type` (String) → `pos_type` (Int64), not both to String.

## Affected Columns
- `extubation_to_perfusion_warm_ischemic_time`
- `warm_ischemic_time_agonal_phase_to_cooling`
- `dcd_nrp_total_pump_time`
- `tod_to_perfusion`
- `sbp90_to_declaration`

All were `Int64` in positive dataset, `String` in negative dataset.

## Fix
Changed the type alignment logic from:
```python
elif pos_type == pl.String or neg_type == pl.String:
    df_neg = df_neg.with_columns(pl.col(col).cast(pl.String))
```

To:
```python
elif neg_type == pl.String:
    df_neg = df_neg.with_columns(pl.col(col).cast(pos_type))
```

This correctly casts the String column in `df_neg` to match the target type in `df_pos`.

## Status
✅ Fixed in `00_process-dataset.py`

## Related
- Pipeline now successfully concatenates positive and negative datasets
- 9 type mismatches handled: 4 datetime (String→Datetime), 5 integer (String→Int64)