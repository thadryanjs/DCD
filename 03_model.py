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
# # Model Evaluation: Binary Classification
# This script evaluates three classifier architectures (Logistic Regression, Random Forest, XGBoost) 
# to predict patient outcomes while strictly preventing data leakage.
#
# ## Strategy: Group-Aware Validation
# To prevent "patient-level leakage" (where the model memorizes a specific patient's characteristics), 
# we use Group-Aware Splitting. 
#
# - **Grouping Variable**: `alias_filled`
# - **Split**: `GroupShuffleSplit` (80% Train / 20% Test)
# - **CV**: 10-Fold `StratifiedGroupKFold` (Outer) / 5-Fold (Inner)
#
# This ensures that all observations from a single patient stay within the same fold.
#
# %% [code]
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, train_test_split, StratifiedGroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# %% [code]
data_dir = Path("data/processed")
plots_dir = Path("output")
plots_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load Data
#
# %% [code]
df = pl.read_parquet(data_dir / "analytic-dataset.parquet")

# %% [code]
print(f"Loaded model-ready dataset: {df.shape}")

# %% [markdown]
# # Feature Identification & Preprocessing
#
# %% [code]
id_cols = ["alias", "alias_filled", "observation"]
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "progression_to_death" and c not in id_cols]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and c != "progression_to_death"]

# %% [code]
print(
    f"Feature set identified:\n"
    f"  Numeric: {len(numeric_cols)}\n"
    f"  Categorical: {len(categorical_cols)}"
)

# %% [code]
from utils import get_preprocessor

# %% [code]
print(
    f"Feature set identified:\n"
    f"  Numeric: {len(numeric_cols)}\n"
    f"  Categorical: {len(categorical_cols)}"
)

# %% [code]
preprocessor = get_preprocessor(numeric_cols, categorical_cols)

# %% [markdown]
# # Group-Aware Train/Test Split
#
# %% [code]
groups = df["alias_filled"].to_numpy()
x_df = df.select(numeric_cols + categorical_cols).to_pandas()
y = df["progression_to_death"].to_numpy()

# %% [code]
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=8675309)
train_idx, test_idx = next(gss.split(x_df, y, groups))

# %% [code]
x_train, x_test = x_df.iloc[train_idx], x_df.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

# %% [code]
print(
    f"""
Split successfully executed (GroupShuffleSplit):
  Train shape: {x_train.shape}
  Test shape: {x_test.shape}
  Train patients: {len(np.unique(groups_train))}
  Test patients: {len(np.unique(groups_test))}
"""
)

# %% [markdown]
# # Model Architectures & Hyperparameter Grids
#
# %% [code]
pos_count = (y_train == 1).sum()
neg_count = (y_train == 0).sum()
spw = neg_count / pos_count if pos_count > 0 else 1

# %% [code]
pipelines = {
    "Logistic Regression": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=8675309))
    ]),
    "Random Forest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=8675309, class_weight="balanced"))
    ]),
    "XGBoost": Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(random_state=8675309, verbosity=0, eval_metric="logloss"))
    ]),
}

# %% [code]
param_grids = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100]
    },
    "Random Forest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7],
        "clf__n_estimators": [50, 100, 200],
        "clf__scale_pos_weight": [spw]
    }
}

# %% [markdown]
# # Nested Cross-Validation
# We use an outer loop for evaluation and an inner loop for tuning.
#
# %% [code]
kf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=8675309)
kf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=8675309)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "auc": "roc_auc",
}

# %% [code]
all_cv_results = []
model_summaries = {}
model_best_params = {name: [] for name in pipelines.keys()}

# %% [code]
for name in pipelines.keys():
    print(f"\nEvaluating Model: {name}")
    
    outer_metrics = {m: [] for m in scoring.keys()}
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf_outer.split(x_train, y_train, groups=groups_train)):
        x_tr_out, x_te_out = x_train.iloc[train_idx], x_train.iloc[test_idx]
        y_tr_out, y_te_out = y_train[train_idx], y_train[test_idx]
        gr_tr_out = groups_train[train_idx]
        
        grid = GridSearchCV(
            estimator=pipelines[name],
            param_grid=param_grids[name],
            cv=kf_inner,
            scoring="accuracy",
            n_jobs=-1
        )
        
        grid.fit(x_tr_out, y_tr_out, groups=gr_tr_out)
        model_best_params[name].append(grid.best_params_)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(x_te_out)
        y_prob = best_model.predict_proba(x_te_out)[:, 1]
        
        fold_results = {
            "accuracy": accuracy_score(y_te_out, y_pred),
            "precision": precision_score(y_te_out, y_pred, zero_division=0),
            "recall": recall_score(y_te_out, y_pred, zero_division=0),
            "f1": f1_score(y_te_out, y_pred, zero_division=0),
            "auc": roc_auc_score(y_te_out, y_prob),
        }
        
        for m, val in fold_results.items():
            outer_metrics[m].append(val)
            all_cv_results.append({
                "model": name,
                "fold": fold_idx + 1,
                "metric": m,
                "value": val
            })

    # Results for the model
    print(f"\nResults for {name}:")
    metrics_summary = {}
    for m, scores in outer_metrics.items():
        mean_s, std_s = np.mean(scores), np.std(scores)
        print(f"  {m:12s}: {mean_s:.3f} (+/- {std_s * 2:.3f})")
        metrics_summary[m] = (mean_s, std_s)
    model_summaries[name] = metrics_summary

# %% [code]
cv_results_df = pl.DataFrame(all_cv_results)
cv_results_path = plots_dir / "cv_metrics_per_fold.csv"
cv_results_df.write_csv(cv_results_path)

# %% [code]
print(f"✓ CV results saved to {cv_results_path}")

# %% [markdown]
# # Feature Importance (Random Forest)
# Fit a final Random Forest on the full training set to identify global predictors.
#
# %% [code]
print("\nFitting final Random Forest for global importance...")
rf_grid = GridSearchCV(
    estimator=pipelines["Random Forest"],
    param_grid=param_grids["Random Forest"],
    cv=kf_inner,
    scoring="accuracy",
    n_jobs=-1
)
rf_grid.fit(x_train, y_train, groups=groups_train)
best_rf = rf_grid.best_estimator_

# %% [code]
cat_features = best_rf.named_steps['pre'].transformers_[1][1].get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(cat_features)

# %% [code]
importances = best_rf.named_steps['clf'].feature_importances_
feat_imp_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)

# %% [code]
plt.figure(figsize=(12, 8))
sns.barplot(data=feat_imp_df.head(20), x='importance', y='feature', hue='feature', palette='viridis', legend=False)
plt.title("Top 20 Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig(plots_dir / "rf_feature_importance.png")
plt.show()

# %% [code]
print(f"✓ Feature importance plot saved to {plots_dir / 'rf_feature_importance.png'}")

# Final Test Set Validation
from collections import Counter
import joblib

# Save best params for explainability script
model_final_params = {}
saved_models = {}

for name in pipelines.keys():
    print(f"\nEvaluating {name} on Test Set...")
    
    # Determine consensus best parameters from nested CV folds
    params_list = model_best_params[name]
    consensus_params = {}
    for param_name in params_list[0].keys():
        values = [p[param_name] for p in params_list]
        consensus_params[param_name] = Counter(values).most_common(1)[0][0]
    
    model_final_params[name] = consensus_params
    print(f"  Using Consensus Params: {consensus_params}")
    
    # Fit model on full training set using consensus parameters
    best_model = pipelines[name].set_params(**consensus_params)
    best_model.fit(x_train, y_train)
    
    # Save model
    model_filename = f"{name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(best_model, data_dir / model_filename)
    saved_models[name] = model_filename
    
    y_pred = best_model.predict(x_test)

    print(
        f"""
Test Metrics for {name}:
  Accuracy  : {accuracy_score(y_test, y_pred):.3f}
  Precision : {precision_score(y_test, y_pred):.3f}
  Recall    : {recall_score(y_test, y_pred):.3f}
  F1 Score  : {f1_score(y_test, y_pred):.3f}
"""
    )

# Save parameters to artifact
import json
with open(data_dir / "model_params.json", "w") as f:
    json.dump(model_final_params, f)
print(f"✓ Model parameters saved to {data_dir / 'model_params.json'}")
print(f"✓ Models saved to {data_dir}")

# %% [markdown]
# # CV Performance Visualization
#
# %% [code]
results_df = pl.read_csv(cv_results_path).to_pandas()
acc_df = results_df[results_df["metric"] == "accuracy"]

# %% [code]
plt.figure(figsize=(10, 6))
sns.boxplot(data=acc_df, x="model", y="value", hue="model", palette="Set2", legend=False)
sns.stripplot(data=acc_df, x="model", y="value", color=".3", alpha=0.5)
plt.title("10-Fold Nested CV Accuracy Distribution")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "cv_accuracy_boxplot.png")
plt.show()

# %% [code]
auc_df = results_df[results_df["metric"] == "auc"]

# %% [code]
plt.figure(figsize=(10, 6))
sns.boxplot(data=auc_df, x="model", y="value", hue="model", palette="Set2", legend=False)
sns.stripplot(data=auc_df, x="model", y="value", color=".3", alpha=0.5)
plt.title("10-Fold Nested CV AUC Distribution")
plt.ylabel("AUC")
plt.xlabel("Model")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "cv_auc_boxplot.png")
plt.show()

# %% [code]
print(f"✓ Boxplots saved: cv_accuracy_boxplot.png, cv_auc_boxplot.png")

