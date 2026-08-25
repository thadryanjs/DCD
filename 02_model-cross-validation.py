# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-Validation: Binary Classification
# 5-fold CV on Logistic Regression, Random Forest, XGBoost.
#
# %% [code]
import polars as pl
from pathlib import Path

# Third-party
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier

data_dir = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data/processed")

# %% [markdown]
# ## 1. Load Data
#
# %% [code]
df = pl.read_parquet(data_dir / "model-ready-dataset.parquet")
print(f"Loaded model-ready dataset: {df.shape}")
print("\nDataset Head:")
print(df.head())
print("\nColumns:")
print(df.columns)

# %% [markdown]
# ## 2. Train/Test Split
#
# %% [markdown]
# ### Data Leakage Check
# Verify no features directly encode the target.
#
# %% [code]
# Suspicious column patterns that may leak labels
leak_keywords = ["label", "outcome", "result", "diagnosis", "positive", "negative", "case"]

potential_leaks = [c for c in df.columns if any(kw in c.lower() for kw in leak_keywords)]

if potential_leaks:
    print(f"WARNING: Potential leakage columns found: {potential_leaks}")
    print("These should be removed before modeling.")
else:
    print("No obvious leakage columns detected.")

# %% [code]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# 1. Identify feature types and drop leakage
id_cols = ["alias", "alias_filled", "observation"]
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "label" and c not in id_cols]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and c != "label"]

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Dropped IDs: {id_cols}")

# 2. Define Preprocessing Transformers
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# Convert to numpy
X = df.select(numeric_cols + categorical_cols).to_numpy() # This is slightly wrong if we use ColumnTransformer
# Actually, if we use ColumnTransformer in the pipeline, we pass the DataFrame (as pandas or similar) 
# or just use the numpy array if indices are known.
# Best: Convert Polars DF to Pandas for sklearn compatibility with ColumnTransformer
X_df = df.select(numeric_cols + categorical_cols).to_pandas()
y = df["label"].to_numpy()

print(f"Full dataset: {X.shape}")

# %% [code]
# Split data
# We use 'alias_filled' as the group to ensure patients aren't split between train and test.
from sklearn.model_selection import GroupShuffleSplit

groups = df["alias_filled"].to_numpy()
X_df = df.select(numeric_cols + categorical_cols).to_pandas()
y = df["label"].to_numpy()

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(X_df, y, groups))

X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# %% [code]
# Verify stratification works
pos_count = y_train.sum()
print(f"Positive samples in train: {pos_count}")
if pos_count < 5:
    raise ValueError(f"Too few positive samples ({pos_count}) for 5-fold stratified CV")

# %% [markdown]
# ## 3. Define Models
#
# %% [code]
# Calculate scale_pos_weight for XGBoost
pos_count = (y_train == 1).sum()
neg_count = (y_train == 0).sum()
spw = neg_count / pos_count if pos_count > 0 else 1

models = {
    "Logistic Regression": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ]),
    "XGBoost": Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(n_estimators=100, max_depth=3, random_state=42, verbosity=0, scale_pos_weight=spw, eval_metric="logloss"))
    ]),
}

# %% [markdown]
# ## 4. 5-Fold Stratified Cross-Validation
#
# %% [code]
kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")

    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy", groups=groups_train)

    print(f"CV Fold scores: {scores}")
    print(f"CV Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# %% [markdown]
# ## 5. Detailed Metrics
#
# %% [code]
scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"{name} - Detailed Metrics")
    print(f"{'='*50}")

    results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, groups=groups_train)

    for metric in scoring.keys():
        scores = results[f"test_{metric}"]
        print(f"{metric:12s}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# %% [markdown]
# ## 6. Test Set Evaluation
#
# %% [code]
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"{name} - Test Set")
    print(f"{'='*50}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"precision: {precision_score(y_test, y_pred):.3f}")
    print(f"recall: {recall_score(y_test, y_pred):.3f}")
    print(f"f1: {f1_score(y_test, y_pred):.3f}")
