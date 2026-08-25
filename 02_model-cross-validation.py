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
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

data_dir = Path("/home/thadryan/Vaults/Projects/Work/Primary/DCD/Code/data/processed")

# %% [markdown]
# ## 1. Load Data
#
# %% [code]
df = pl.read_parquet(data_dir / "combined-dataset.parquet")
print(f"Loaded: {df.shape}")

# %% [markdown]
# ## 2. Train/Test Split
#
# %% [code]
# Get numeric columns (exclude label)
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "label"]
print(f"Features: {len(numeric_cols)}")

# %% [code]
# Convert to numpy
X = df.select(numeric_cols).to_numpy()
y = df["label"].to_numpy()

print(f"Full dataset: {X.shape}")

# %% [code]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# %% [markdown]
# ## 3. Define Models
#
# %% [code]
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
}

# %% [markdown]
# ## 4. 5-Fold Cross-Validation
#
# %% [code]
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
    
    print(f"CV Fold scores: {scores}")
    print(f"CV Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# %% [markdown]
# ## 5. Detailed Metrics
#
# %% [code]
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

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
    
    results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring)
    
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
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"precision: {precision_score(y_test, y_pred):.3f}")
    print(f"recall: {recall_score(y_test, y_pred):.3f}")
    print(f"f1: {f1_score(y_test, y_pred):.3f}")