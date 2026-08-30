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
# - **CV**: 3x5 Repeated `StratifiedGroupKFold` (Outer) / 5-Fold (Inner)
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

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from utils import get_preprocessor

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
# For ML, we use in-fold imputation to avoid leakage. The `IterativeImputer` (a Bayesian Ridge 
# regression-based approach) is fit on the training folds of the outer CV and applied to the 
# test fold. This ensures that the imputation logic is derived only from the training data.
#
# %% [code]
id_cols = ["alias", "alias_filled", "observation"]
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "progression_to_death" and c not in id_cols]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and c != "progression_to_death" and c not in id_cols]

# %% [code]
# Preprocessor is now built inside run_ml_pipeline to avoid global scope issues
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
def run_ml_pipeline(df, numeric_cols, categorical_cols, prefix="full"):
    print(f"Running Pipeline: {prefix}")
    
    # Build preprocessor inside function to ensure consistency with arguments
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)

    # Group-Aware Train/Test Split
    groups = df["alias_filled"].to_numpy()
    x_df = df.select(numeric_cols + categorical_cols).to_pandas()
    y = df["progression_to_death"].to_numpy()

    # **LOADBEARING** — Using GroupShuffleSplit prevents patient-level leakage.
    # Seed pinned to match 01 feature selection split.
    # Consumed by: `04` analyze-model.py (reconstruction of training split for SHAP)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=8675309)
    train_idx, test_idx = next(gss.split(x_df, y, groups))

    x_train, x_test = x_df.iloc[train_idx], x_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    # Split Receipt
    train_patients = len(np.unique(groups_train))
    test_patients = len(np.unique(groups_test))
    
    # Class balance
    train_pos_rows = np.sum(y_train == 1)
    train_neg_rows = np.sum(y_train == 0)
    train_pos_pts = len(np.unique(groups_train[y_train == 1]))
    train_neg_pts = len(np.unique(groups_train[y_train == 0]))
    
    test_pos_rows = np.sum(y_test == 1)
    test_neg_rows = np.sum(y_test == 0)
    test_pos_pts = len(np.unique(groups_test[y_test == 1]))
    test_neg_pts = len(np.unique(groups_test[y_test == 0]))
    
    print(
        f"Split ({prefix}):\n"
        f"  Train shape: {x_train.shape} ({train_patients} patients)\n"
        f"    Class 1: {train_pos_rows} rows, {train_pos_pts} patients\n"
        f"    Class 0: {train_neg_rows} rows, {train_neg_pts} patients\n"
        f"  Test shape: {x_test.shape} ({test_patients} patients)\n"
        f"    Class 1: {test_pos_rows} rows, {test_pos_pts} patients\n"
        f"    Class 0: {test_neg_rows} rows, {test_neg_pts} patients\n"
        f"  Total patients: {train_patients + test_patients}"
    )

    # Model Architectures & Hyperparameter Grids
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    spw = float(neg_count / pos_count) if pos_count > 0 else 1.0
    print(f"Class Weighting: Pos={pos_count}, Neg={neg_count}, XGBoost spw={spw:.3f}")

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
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10],
            "clf__min_samples_split": [2, 5]
        },
        "XGBoost": {
            "clf__learning_rate": [0.1],
            "clf__max_depth": [3, 5],
            "clf__n_estimators": [100],
            "clf__scale_pos_weight": [spw]
        }
    }

    # Nested Cross-Validation
    kf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=8675309)
    kf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=8675309)

    # Pre-compute outer folds for transparency and consistency
    outer_folds = list(kf_outer.split(x_train, y_train, groups=groups_train))
    
    print("\n--- Outer Fold Composition (Held-out Portion) ---")
    fold_comp = []
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_folds):
        te_y = y_train[te_idx]
        te_gr = groups_train[te_idx]
        
        fold_comp.append({
            "fold": fold_idx + 1,
            "rows": len(te_idx),
            "patients": len(np.unique(te_gr)),
            "pos_rows": np.sum(te_y == 1),
            "pos_patients": len(np.unique(te_gr[te_y == 1])),
            "neg_rows": np.sum(te_y == 0),
            "neg_patients": len(np.unique(te_gr[te_y == 0])),
        })
    print(pd.DataFrame(fold_comp).to_string(index=False))
    print("-----------------------------------------------\n")

    metrics_to_track = ["accuracy", "precision", "recall", "f1", "auc"]

    all_cv_results = []
    
    for name in pipelines.keys():
        print(f"\nEvaluating Model: {name} ({prefix})")
        
        # 3x5 Repeated Outer CV
        for repeat in range(3):
            kf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=8675309 + repeat)
            repeat_metrics = {m: [] for m in metrics_to_track}
            
            for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(kf_outer.split(x_train, y_train, groups=groups_train)):
                x_tr_out, x_te_out = x_train.iloc[outer_train_idx], x_train.iloc[outer_test_idx]
                y_tr_out, y_te_out = y_train[outer_train_idx], y_train[outer_test_idx]
                gr_tr_out = groups_train[outer_train_idx]
                
                grid = GridSearchCV(
                    estimator=pipelines[name],
                    param_grid=param_grids[name],
                    cv=kf_inner,
                    scoring="roc_auc",
                    n_jobs=-1
                )
                
                grid.fit(x_tr_out, y_tr_out, groups=gr_tr_out)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(x_te_out)
                y_prob = best_model.predict_proba(x_te_out)[:, 1]
                
                # Compute metrics individually to prevent AUC failure from zeroing others
                acc = accuracy_score(y_te_out, y_pred)
                prec = precision_score(y_te_out, y_pred, zero_division=0)
                rec = recall_score(y_te_out, y_pred, zero_division=0)
                f1 = f1_score(y_te_out, y_pred, zero_division=0)
                
                try:
                    auc = roc_auc_score(y_te_out, y_prob)
                except ValueError:
                    auc = np.nan
                
                fold_results = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc,
                }
                
                for m, val in fold_results.items():
                    repeat_metrics[m].append(val)
                    all_cv_results.append({
                        "model": name,
                        "repeat": repeat + 1,
                        "fold": fold_idx + 1,
                        "metric": m,
                        "value": val
                    })

            # Print per-repeat summary
            print(f"  Repeat {repeat + 1}:")
            for m, scores in repeat_metrics.items():
                mean_s = np.nanmean(scores)
                print(f"    {m:12s}: {mean_s:.3f} [n={len(scores)}]")

        # Final summary across all repeats
        print(f"\nOverall Results for {name} ({prefix}):")
        model_results = pl.DataFrame(all_cv_results).filter(pl.col("model") == name)
        for m in metrics_to_track:
            scores = model_results.filter(pl.col("metric") == m)["value"].to_numpy()
            clean_scores = scores[~np.isnan(scores)]
            if len(clean_scores) == 0:
                print(f"  {m:12s}: NaN")
                continue
            
            median_s = np.median(clean_scores)
            min_s = np.min(clean_scores)
            max_s = np.max(clean_scores)
            iqr_s = np.percentile(clean_scores, 75) - np.percentile(clean_scores, 25)
            print(f"  {m:12s}: {median_s:.3f} [{min_s:.3f}, {max_s:.3f}] (IQR: {iqr_s:.3f}) [n={len(clean_scores)}]")

    cv_results_df = pl.DataFrame(all_cv_results)
    cv_results_path = plots_dir / f"cv_metrics_per_fold_{prefix}.csv"
    cv_results_df.write_csv(cv_results_path)

    # Feature Importance (Random Forest)
    print(f"\nFitting final Random Forest ({prefix}) for global importance...")
    rf_grid = GridSearchCV(
        estimator=pipelines["Random Forest"],
        param_grid=param_grids["Random Forest"],
        cv=kf_inner,
        scoring="roc_auc",
        n_jobs=-1
    )
    rf_grid.fit(x_train, y_train, groups=groups_train)
    best_rf = rf_grid.best_estimator_

    all_feature_names = best_rf.named_steps['pre'].get_feature_names_out()

    importances = best_rf.named_steps['clf'].feature_importances_
    
    # Transparency: Assert feature name count matches importance count
    assert len(all_feature_names) == len(importances), (
        f"Feature name count {len(all_feature_names)} != importance count {len(importances)}"
    )
    print(f"Confirmed: {len(all_feature_names)} features in, {len(importances)} importances out")
    
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
            scoring="roc_auc",
            n_jobs=-1
        )
        grid.fit(x_train, y_train, groups=groups_train)
        best_model = grid.best_estimator_
        
        model_final_params[name] = grid.best_params_
        
        model_filename = f"{prefix}_{name.lower().replace(' ', '_')}_model.joblib"
        joblib.dump(best_model, data_dir / model_filename)
        
        y_pred = best_model.predict(x_test)
        y_prob = best_model.predict_proba(x_test)[:, 1]
        print(
            f"Test Metrics for {name} ({prefix}):\n"
            f"  Accuracy  : {accuracy_score(y_test, y_pred):.3f}\n"
            f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.3f}\n"
            f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.3f}\n"
            f"  F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.3f}\n"
            f"  AUC       : {roc_auc_score(y_test, y_prob):.3f}"
        )

    with open(data_dir / f"model_params_{prefix}.json", "w") as f:
        json.dump(model_final_params, f)

    # CV Performance Visualization
    results_df = cv_results_df.to_pandas()
    for metric in ["accuracy", "auc"]:
        m_df = results_df[results_df["metric"] == metric]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=m_df, x="model", y="value", hue="model", palette="Set2", legend=False)
        sns.stripplot(data=m_df, x="model", y="value", color=".3", alpha=0.5)
        plt.title(f"3x5 Repeated Nested CV {metric.upper()} Distribution ({prefix})")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Model")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plots_dir / f"cv_{metric}_boxplot_{prefix}.png")
        plt.close()

# %% [markdown]
# ## First Look Only Analysis
# Evaluate models using only the first observation per patient to remove time-series bias.
# This is our primary analysis as it represents the most conservative estimate.
#
# %% [code]
df_first_look = df.filter(pl.col("observation") == 1)
run_ml_pipeline(df_first_look, numeric_cols, categorical_cols, prefix="first_look")

# %% [markdown]
# ## Full Dataset Analysis
# Evaluate models using all available observations.
# This serves as an optimistic upper bound; the performance gap between this and the
# "First Look" analysis is used as a diagnostic for late-observation leakage.
#
# %% [code]
run_ml_pipeline(df, numeric_cols, categorical_cols, prefix="full")
