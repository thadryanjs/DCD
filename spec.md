
## Project
- This is an inferential and predictive ML project predicting if an individual went on to an absorbing state (1) and didn't (0).
- There are multiple observations per subject so we used Mixed Effects Models for the classical statistical approach and group-aware stratification in the ML script.
- The code IS the report - we're using jupytext to write normal, git-friendly scripts and compile them to reader friendly pdf/html reports.
- This report used a previous report about Ophthalmology as a template and agents keep trying to stick name of it all over the place. It's a template. We're not cloning the name. This project is about DCD and that should be the only name reflected in naming conventions.
- Don't run the report preview - I usually have one going so this only invites confusion and mistakes.
- Don't execute code unless I say to - I usually run and selectively share output.
- We're using `uv`. This means run commands need to be written `uv run {whatever}` not just `{whatever}`. This applies to writing the justfile commands as well.
- The project was started in Python though we had to introduce one R script.
- We want to use a mixed-effects model (lme4 in R) for a more classical, inferential analysis.
    - The random effects would be person-level ids.
    - We want to note significant predictors.
- We want to produce a ten-fold cross-validated classifier for an ML approach
    - We want to see
        - Accuracy and AUC at each fold (as a boxplot)
        - Features importance
        - SHAP

## Data & ML Constraints
- **Patient-Level Leakage**: The dataset contains multiple observations per patient (`alias_filled`). Standard random splits cause severe data leakage (memorization of patients).
- **Requirement**: Always use `StratifiedGroupKFold` or `GroupShuffleSplit` with `groups=df["alias_filled"]` to ensure all observations for a single patient stay within the same fold.

## Style
- Use `just format` which calls `black` - sane defaults don't overthink it.
- Don't use ALL_CAPS for constants it's pointless and loud and in certain types of scripts it makes half the script uppercase.
- Don't use type hints unless there is a meaningful indication to add this complexity, ie
    - There is a type that impacts a calculation or could result in a bad transformation
    - Don't bother with it on something like "read_file" where passing number will result in a harmless error - type checking this is a marginal improvement over just getting a FileNotFound error.
- We're using `jupytext` to have clean scripts with good git diffs while also being able to compile to jupyter books. If you're an agent, ask about a skill for this. Some basics:
    - Any prints should occur at the end of a cell and start a new cell - so three smalls cells to instead of one larger one to print three things. It swill then read code/output/code/output/code/output not code/code/code/output/output/output.
    - Don't do this. Agents love it but it makes things worse not better if we're using jupyter because natural lightweight delineation happens with the cells already.
        `print("\nMissingness Comparison: Positive vs Negative Class")`
        `print("=" * 70)`

- All the imports go at the top! No exceptions!
- Don't change file names without asking me. If it's a slightly different name than expected it's because I did it on purpose.


## Spec: Transparency Pass

Goal: a reader should be able to scroll the compiled report and confirm every mutation
to the data by eye, without running anything or trusting a docstring. Anything that
cannot be confirmed visually gets an assertion instead, so failure is loud rather than
silent.

This spec adds to the existing project spec. It does not override it: prints still go at
the end of a cell, imports still go at the top, no separator prints, no ALL_CAPS.

---

### 1. The mutation receipt

Every cell that changes the shape, dtype, row count, column set, or values of a frame
follows the same three-beat pattern, split across cells so the compiled notebook reads
code / output / code / output.

```
# %% [markdown]
# ### What is about to happen and why

# %% [code]
# show the state that is about to change
print(<the specific thing, narrow enough to read>)

# %% [code]
# the mutation, and nothing else
df = <one operation>

# %% [code]
# show the same view again, plus what changed
print(<same view>)
print(f"rows {before} -> {after}, cols {before} -> {after}")
```

Rules:

- **One mutation per cell.** If a cell does two `with_columns` calls, split it.
- **The before and after views must be the same view.** Same columns, same slice, same
  ordering. A reader compares them line by line; changing the view defeats that.
- **Show the delta explicitly.** Never make the reader diff two tables in their head to
  find out that three rows disappeared.
- **Name what was removed, not just how many.** `Dropped 4 features` is not auditable.
  `Dropped 4 features: [a, b, c, d]` is.
- **Narrow the view.** A 60-column frame printed in full is not transparency, it is
  noise. Select the columns the mutation touches, plus the id.

### 2. Proof-on-toy-data stays, but is not sufficient

`00` already demonstrates `forward_populate_ids` and `clean_colnames` on toy frames.
Keep that. It shows intent. But a toy proof does not show what happened to the real
data, so every toy proof is followed by a real-data receipt showing the actual
before/after on a slice where the effect is visible — for a forward-fill, that means a
slice spanning an ID boundary, not the first 20 rows.

### 3. LOADBEARING tags

A step is load-bearing if changing it, removing it, or reordering it silently breaks a
correctness guarantee somewhere else in the pipeline. Not "important" — *silently*
breaking, and *elsewhere*.

Tag it in the markdown cell immediately above the code, on its own line:

```
# %% [markdown]
# ### Namespace patient IDs before concatenation
#
# **LOADBEARING** — Both workbooks number patients from 1. Without the prefix,
# `pos_1` and `neg_1` collide into a single group, which corrupts patient counts in
# `01`, degrades stratification in `03`, and makes the R random-effect structure
# meaningless. Consumed by: `01` stability analysis, `03` group splits, `02` aggregation.
```

Required parts, in this order:

1. `**LOADBEARING**` — exact spelling, bold, so a grep finds every one.
2. What breaks if this changes.
3. `Consumed by:` — the downstream files or steps that depend on it.

Do not tag things that fail loudly. A missing file raises `FileNotFoundError`; that is
self-reporting and does not need a tag. Tag the ones that produce plausible-looking
wrong numbers.

Target is roughly a dozen tags across the pipeline. If a file has ten, the bar has
slipped and the tag stops meaning anything.

## 4. Assertions for invisible guarantees

Some guarantees cannot be shown in a table because they are properties of a
relationship, not of a value. These get assertions with messages that state the
guarantee, placed in their own cell with a confirming print.

```
# %% [code]
overlap = set(groups_train) & set(groups_test)
assert not overlap, f"Patient leakage: {len(overlap)} patients in both splits: {sorted(overlap)[:5]}"
print(f"Confirmed: 0 patients shared between train and test ({len(set(groups_train))} / {len(set(groups_test))})")
```

The print matters as much as the assert. A silent assertion tells the reader nothing was
checked; a passing print tells them it was.

### 5. Artifacts for anything too long to print

Lists that are load-bearing but too long to read inline (the selected feature set, the
full column mapping) get written to `output/` as CSV *and* summarised inline with a
count plus the first few entries. The reader gets the summary; the auditor gets the file.

---

### Required additions by file

### `00_process-data.py`

- [ ] **Column name mapping receipt.** `clean_colnames` is proven on toy data but the
  real mapping is never shown. Print the full `{old: new}` dict for both workbooks,
  and separately print any cleaned name containing a character other than
  `[a-z0-9_]`.
  **Tag LOADBEARING:** the cleaner permits hyphens, and a hyphenated column name
  becomes subtraction in an R formula. `02` backticks its formulas because of this.
- [ ] **Forward-fill on real data at an ID boundary.** The current print shows the first
  20 rows of the positive file. Show a slice of the *negative* file (the one with merged
  cells) spanning at least two ID changes, columns `alias`, `alias_filled`,
  `observation`, before and after.
  **Tag LOADBEARING:** `observation` numbering drives every first-look analysis.
- [ ] **`observation` distribution, not just the minimum.** Currently asserts
  `min == 1`. Add: value counts of `observation`, and observations-per-patient
  distribution, per source file. This is also the input to the equal-weighting
  limitation in `02`.
- [ ] **Namespacing receipt.** Print `n_unique(alias_filled)` per file before, then
  after, then for the concatenated frame, and assert the combined count equals the sum.
  Show five sample IDs from each side.
  **Tag LOADBEARING** (text drafted in section 3 above).
- [ ] **Diagonal concat column provenance.** `how="diagonal"` fills absent columns with
  nulls, and a column present in one workbook only becomes a perfect label proxy. Print
  three lists: columns in positives only, negatives only, both. Assert the
  positives-only and negatives-only lists are empty *or* print them under a heading
  saying these are candidate leaks handled in `01`.
  **Tag LOADBEARING:** this is the mechanism the `01` missingness filter exists to catch.
- [ ] **Final class and patient summary.** Rows per class, patients per class,
  observations per patient by class. One cell, at the end.

### `01_explore-data.py`

- [ ] **Every filter reports its casualties by name.** For each of missingness,
  variance, mutual information, and the manual exclusion list: count in, count out, and
  the named list of what was dropped with the value that triggered it. A table, not a
  sentence.
- [ ] **`leak_exclusion_list` gets a per-column justification.** It is a hardcoded list
  of clinical judgements and it is the primary leak defence. Print each column beside a
  one-line reason.
  **Tag LOADBEARING:** these are post-outcome variables; including any of them makes the
  ML results meaningless.
- [ ] **Make the MI split visible.** Print that MI was fitted on training patients only,
  the patient count used, the seed, and that the seed matches `03`. Assert the seed
  constant is identical to the one in `03` rather than trusting that both say 8675309.
  **Tag LOADBEARING:** fitting this on all rows leaks test labels into feature selection.
- [ ] **Variance threshold caveat, inline.** The threshold applies to unscaled variances
  and is therefore unit-dependent. One markdown line where the threshold is set, so a
  reader does not assume it is scale-free.
- [ ] **Selected feature set as artifact.** Write `selected_features` to
  `output/selected-features.csv` with the missingness, variance, and MI value that let
  each one through. Print the count and first ten.
- [ ] **Correlation section honesty.** It reports pairs and drops nothing. State that
  in the markdown, so the reader does not assume a filter ran.

### `02_analyze-data.R`

- [ ] **Restore the dropped-feature print.** The 50-line rewrite made
  `d[sapply(d, ...)]` silent. It removes columns with fewer than two distinct values and
  currently names none of them. Print the names.
  **Tag LOADBEARING:** a dropped feature is absent from the forest plot with no trace,
  and the reader has no way to know it was ever considered.
- [ ] **Aggregation receipt.** Print row count and column count before and after
  `to_patient`, and for two or three named features show the raw observations for one
  patient beside that patient's computed mean. One patient is enough to confirm the
  aggregation does what the markdown says.
- [ ] **Scaling receipt.** For three named features, print mean and SD before and after
  scaling. Confirms the per-SD interpretation of the coefficients.
- [ ] **Feature set reconciliation.** Compare the R feature list against
  `output/selected-features.csv` from `01` and print the symmetric difference. Currently
  the two are derived independently and assumed to match.
  **Tag LOADBEARING:** `04` presents the GLM and the SHAP results as aligned; if the
  feature sets differ, that alignment is fictional.
- [ ] **Unstable fits named, not just counted.** Already printed as a count. Print the
  feature names with their estimate, SE, and separation flag.
- [ ] **State what the CSV columns mean.** `estimate` is a log-odds per SD of a patient
  mean. Three lines in the markdown, because `04` exponentiates these and a reader
  needs to know what the units were.

### `03_model.py`

- [ ] **Print the model matrix column list in full.** The exclusion of `alias_filled`
  from `categorical_cols` is the difference between a valid model and patient-identity
  leakage, and it is currently invisible. Print `numeric_cols + categorical_cols` and
  assert no element is in `id_cols`.
  **Tag LOADBEARING.**
- [ ] **Assert no patient spans train and test.** The header claims group-aware
  splitting prevents patient leakage. Nothing verifies it. Add the assertion and
  confirming print from section 4.
  **Tag LOADBEARING:** this is the pipeline's headline correctness claim.
- [ ] **Fold composition table.** Per outer fold: patient count, row count, and class
  balance in the held-out portion. This is what explains the low-AUC folds, and it turns
  the current guesswork into a printed fact.
- [ ] **Train and test class balance.** Patient counts and row counts per class for both
  splits. Currently only patient totals are printed, so a badly imbalanced test set is
  invisible.
- [ ] **`spw` printed.** It is computed from the training labels and silently injected
  into the XGBoost grid.
- [ ] **Assert the transformed width matches expectations.**
  `IterativeImputer` drops all-missing features, which would shift every feature name.
  After the final RF fit, assert
  `len(get_feature_names_out()) == len(feature_importances_)` and print both.
  **Tag LOADBEARING:** a mismatch mis-attributes every importance and every SHAP value.
- [ ] **Report median and range, not mean ± 2 SD.** Fold AUC is skewed and bounded at 1,
  so the current interval overshoots. Print median, min, max, IQR, and the contributing
  fold count.

### `04_analyze-model.py`

- [ ] **Assert the split matches `03`'s.** The split is reproduced by copy-pasted
  parameters. Compare the training patient set against what `03` used — simplest route
  is for `03` to write `output/train-patients-{prefix}.csv` and for `04` to assert set
  equality.
  **Tag LOADBEARING:** if the splits diverge, the SHAP values explain a model on data it
  was not trained on.
- [ ] **Assert feature name count matches the SHAP array width.** One line, before the
  loop that indexes `shap_values_class1[:, i]` by position.
- [ ] **Show the loaded model's identity.** Print the artifact path, its modification
  time, and the fitted hyperparameters, so the reader knows which run is being explained.
- [ ] **State the direction-test method and its blind spot.** The median split falls
  through to "Neutral/Non-linear" for one-hot features where the guard trips. Say so, and
  print how many features landed in that bucket for that reason.
- [ ] **Label the odds-ratio units on the forest plot.** Per SD of a patient mean, from a
  patient-level model — not per unit, and not from the same rows the RF was trained on.

---

## Acceptance checks

Runnable, and worth putting in the justfile:

- [ ] `grep -c 'LOADBEARING' *.py *.R` returns a number between 8 and 15. Zero means the
  pass did not happen; thirty means the tag is decoration.
- [ ] Every `LOADBEARING` line is followed within three lines by `Consumed by:`.
- [ ] No code line sits inside a `# %% [markdown]` cell in any file.
- [ ] Every `assert` in the pipeline has a message, and a `print` confirming the pass in
  the same cell.
- [ ] Every filter that removes rows or columns prints the names of what it removed.
- [ ] For each mutation cell, the before-print and after-print select the same columns.

The last two are eyeball checks, and they are the point of the exercise. The first four
can be scripted.

---

## Deliberately out of scope

- Rewriting logic. This pass adds visibility, not behaviour. If a receipt reveals a bug,
  that is a separate fix with its own review.
- Printing whole frames. Narrow views and artifacts, not `print(df)`.
- Docstrings. They describe intent; this spec is about evidence.

## Worktree workflow



```
# 1. review and merge PR in gh-dash (d, then m)

# 2. root catches up
git checkout main && git pull

# 3. per worktree with an open PR: rebase AND push
cd ../Dotfiles-agent2
git fetch && git rebase origin/main
git push --force-with-lease

cd ../Dotfiles-agent3
git fetch && git rebase origin/main
git push --force-with-lease

cd ~/Vaults/Projects/Dotfiles

# 4. next PR
```
