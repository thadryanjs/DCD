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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import joblib

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
from utils import get_preprocessor

# %% [code]
preprocessor = get_preprocessor(numeric_cols, categorical_cols)

# %% [code]
print(
    f"Feature set identified:\n"
    f"  Numeric: {len(numeric_cols)}\n"
    f"  Categorical: {len(categorical_cols)}"
)

# %% [markdown]
# # Model Pipeline Definition
# We wrap the entire training and evaluation process to compare "Full Dataset" vs "First Look Only".
#
# %% [code]
def run_ml_pipeline(df, prefix="full"):
    print(f"\n{'='*40}\nRunning Pipeline: {prefix}\n{'='*40}")
    
    # Group-Aware Train/Test Split
    groups = df["alias_filled"].to_numpy()
    x_df = df.select(numeric_cols + categorical_cols).to_pandas()
    y = df["progression_to_death"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=8675309)
    train_idx, test_idx = next(gss.split(x_df, y, groups))

    x_train, x_test = x_df.iloc[train_idx], x_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    print(
        f"Split ({prefix}):\n"
        f"  Train shape: {x_train.shape}\n"
        f"  Test shape: {x_test.shape}\n"
        f"  Train patients: {len(np.unique(groups_train))}\n"
        f"  Test patients: {len(np.unique(groups_test))}"
    )

    # Model Architectures & Hyperparameter Grids
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    spw = neg_count / pos_count if pos_count > 0 else 1

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

    # Nested Cross-Validation
    kf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=8675309)
    kf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=8675309)

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
        "auc": "roc_auc",
    }

    all_cv_results = []
    
    for name in pipelines.keys():
        print(f"\nEvaluating Model: {name} ({prefix})")
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

        print(f"\nResults for {name} ({prefix}):")
        for m, scores in outer_metrics.items():
            mean_s, std_s = np.mean(scores), np.std(scores)
            print(f"  {m:12s}: {mean_s:.3f} (+/- {std_s * 2:.3f})")

    cv_results_df = pl.DataFrame(all_cv_results)
    cv_results_path = plots_dir / f"cv_metrics_per_fold_{prefix}.csv"
    cv_results_df.write_csv(cv_results_path)

    # Feature Importance (Random Forest)
    print(f"\nFitting final Random Forest ({prefix}) for global importance...")
    rf_grid = GridSearchCV(
        estimator=pipelines["Random Forest"],
        param_grid=param_grids["Random Forest"],
        cv=kf_inner,
        scoring="accuracy",
        n_jobs=-1
    )
    rf_grid.fit(x_train, y_train, groups=groups_train)
    best_rf = rf_grid.best_estimator_

    cat_features = best_rf.named_steps['pre'].transformers_[1][1].get_feature_names_out(categorical_cols)
    all_feature_names = numeric_cols + list(cat_features)

    importances = best_rf.named_steps['clf'].feature_importances_
    feat_imp_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=feat_imp_df.head(20), x='importance', y='feature', hue='feature', palette='viridis', legend=False)
    plt.title(f"Top 20 Feature Importances - Random Forest ({prefix})")
    plt.tight_layout()
    plt.savefig(plots_dir / f"rf_feature_importance_{prefix}.png")
    plt.close()

    # Final Test Set Validation
    model_final_params = {}
    
    for name in pipelines.keys():
        print(f"\nEvaluating {name} on Test Set ({prefix})...")
        grid = GridSearchCV(
            estimator=pipelines[name],
            param_grid=param_grids[name],
            cv=kf_inner,
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(x_train, y_train, groups=groups_train)
        best_model = grid.best_estimator_
        
        model_final_params[name] = grid.best_params_
        
        model_filename = f"{prefix}_{name.lower().replace(' ', '_')}_model.joblib"
        joblib.dump(best_model, data_dir / model_filename)
        
        y_pred = best_model.predict(x_test)
        print(
            f"Test Metrics for {name} ({prefix}):\n"
            f"  Accuracy  : {accuracy_score(y_test, y_pred):.3f}\n"
            f"  Precision : {precision_score(y_test, y_pred):.3f}\n"
            f"  Recall    : {recall_score(y_test, y_pred):.3f}\n"
            f"  F1 Score  : {f1_score(y_test, y_pred):.3f}"
        )

    with open(data_dir / f"model_params_{prefix}.json", "w") as f:
        json.dump(model_final_params, f)

    # CV Performance Visualization
    results_df = pl.read_csv(cv_results_path).to_pandas()
    for metric in ["accuracy", "auc"]:
        m_df = results_df[results_df["metric"] == metric]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=m_df, x="model", y="value", hue="model", palette="Set2", legend=False)
        sns.stripplot(data=m_df, x="model", y="value", color=".3", alpha=0.5)
        plt.title(f"10-Fold Nested CV {metric.upper()} Distribution ({prefix})")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Model")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plots_dir / f"cv_{metric}_boxplot_{prefix}.png")
        plt.close()

# %% [markdown]
# ## Full Dataset Analysis
# Evaluate models using all available observations.
#
# %% [code]
run_ml_pipeline(df, prefix="full")

# %% [markdown]
# ## First Look Only Analysis
# Evaluate models using only the first observation per patient to remove time-series bias.
#
# %% [code]
df_first_look = df.filter(pl.col("observation") == 1)
run_ml_pipeline(df_first_look, prefix="first_look")
