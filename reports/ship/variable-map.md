# Variable map (internal)

What we call it in the deck/report mapped back to the original column name.
Final feature set: 17 features + label (source: `output/selected-features.csv`).

## Baseline numerics (12) — scaled per SD; tested individually in the GLM, used in all ML models

| What we call it | Original column |
|---|---|
| Patient age | `age` |
| Body mass index | `bmi` |
| Baseline blood pH | `ph` |
| Baseline carbon dioxide (PCO2) | `pco2` |
| Baseline oxygen (PO2) | `po2` |
| Baseline bicarbonate | `hco3` |
| Baseline base excess | `be` |
| Baseline oxygen saturation (lab panel) | `o2_sat` |
| Baseline heart rate | `baseline_vitals_heart_rate` |
| Baseline systolic blood pressure | `dcd_systolic_bp_baseline` |
| Baseline diastolic blood pressure | `dcd_diastolic_bp_baseline` |
| Baseline oxygen saturation (vitals) | `baseline_vitals_o2_sat` |

Note: two distinct oxygen-saturation columns exist — `o2_sat` (laboratory panel)
and `baseline_vitals_o2_sat` (vitals measurement). The latter is the one flagged
for fitted-probability separation in the GLM ("direction not established").

## Categoricals (5) — one-hot for ML; only sex and hospital entered the GLM

| What we call it | Original column | GLM treatment |
|---|---|---|
| Patient sex | `sex` | Single-term test |
| Hospital / care level | `hospital` | Pooled omnibus joint test |
| Age unit | `ageunit` | Constant in observed data — dropped from GLM |
| Controlled pathway flag | `controlled_dcd` | Constant in observed data — dropped from GLM |
| Perfusion intent flag | `dcd_nrp_intended` | Constant in observed data — dropped from GLM |

## Label

| What we call it | Original column |
|---|---|
| Outcome (progression vs not) | `progression_to_death` |

## Not used (excluded before modeling, 10 post-treatment variables)

Process timings, care unit, extubation site, and both heparin variables —
each with a documented reason in the book's leakage-exclusion section.
