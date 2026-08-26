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
# # Model Evaluation Report: Binary Classification
#
# ## Goal
# Evaluate three classifier architectures (Logistic Regression, Random Forest, XGBoost) to predict binary outcomes while strictly preventing data leakage.
#
# ## Strategy: Group-Aware Validation
# To prevent "patient-level leakage" (where the model memorizes a specific patient's characteristics rather than learning generalizable patterns), we use **Group-Aware Splitting**. 
#
# - **Grouping Variable**: `alias_filled`
# - **Split**: `GroupShuffleSplit` (80% Train / 20% Test)
# - **CV**: 10-Fold `StratifiedGroupKFold` (Outer) / 5-Fold (Inner)
#
# This ensures that all observations from a single patient stay within the same fold, providing a realistic estimate of generalizability to new patients.

# %% [code]
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Third-party
from sklearn.model_selection import cross_validate, train_test_split, StratifiedGroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import statsmodels.api as sm

data_dir = Path("data/processed")
plots_dir = Path("output")
plots_dir.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 1. Load Data

# %% [code]
df = pl.read_parquet(data_dir / "analytic-dataset.parquet")
print(f"Loaded model-ready dataset: {df.shape}")


# %% [markdown]
# ## 2. Train/Test Split & Preprocessing

# %% [code]
# Identify feature types
id_cols = ["alias", "alias_filled", "observation"]
numeric_cols = [c for c in df.columns if df.schema[c].is_numeric() and c != "progression_to_death" and c not in id_cols]
categorical_cols = [c for c in df.columns if not df.schema[c].is_numeric() and c != "progression_to_death"]

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")


# %% [code]
# Preprocessing Transformers
numeric_transformer = Pipeline([
    ("imputer", IterativeImputer(random_state=42)),
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


# %% [code]
# Split data with GroupShuffleSplit
groups = df["alias_filled"].to_numpy()
X_df = df.select(numeric_cols + categorical_cols).to_pandas()
y = df["progression_to_death"].to_numpy()

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(X_df, y, groups))

X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

print(
    f"""
Split successful:
  Train shape: {X_train.shape}
  Test shape: {X_test.shape}
  Train groups: {len(np.unique(groups_train))} unique patients
  Test groups: {len(np.unique(groups_test))} unique patients
"""
)


# %% [markdown]
# ## 3. Define Models and Hyperparameter Grids

# %% [code]
# Calculate scale_pos_weight for XGBoost
pos_count = (y_train == 1).sum()
neg_count = (y_train == 0).sum()
spw = neg_count / pos_count if pos_count > 0 else 1

pipelines = {
    "Logistic Regression": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ]),
    "XGBoost": Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"))
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


# %% [markdown]
# ## 4. Nested Cross-Validation: Parameter Tuning & Evaluation

# %% [code]
kf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
kf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
}

all_cv_results = []
model_summaries = {}

for name in pipelines.keys():
    print(f"\n{'='*60}\n Tuning & Evaluating: {name}\n{'='*60}")
    
    outer_metrics = {m: [] for m in scoring.keys()}
    
    # Manual Outer Loop to ensure groups are passed to inner GridSearchCV
    for fold_idx, (train_idx, test_idx) in enumerate(kf_outer.split(X_train, y_train, groups=groups_train)):
        X_tr_out, X_te_out = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr_out, y_te_out = y_train[train_idx], y_train[test_idx]
        gr_tr_out = groups_train[train_idx]
        
        # Inner Loop: Tuning
        grid = GridSearchCV(
            estimator=pipelines[name],
            param_grid=param_grids[name],
            cv=kf_inner,
            scoring="accuracy",
            n_jobs=-1
        )
        
        # Pass groups to inner CV via fit
        grid.fit(X_tr_out, y_tr_out, groups=gr_tr_out)
        best_model = grid.best_estimator_
        
        # Evaluate on outer test fold
        y_pred = best_model.predict(X_te_out)
        
        # Collect metrics
        fold_results = {
            "accuracy": accuracy_score(y_te_out, y_pred),
            "precision": precision_score(y_te_out, y_pred, zero_division=0),
            "recall": recall_score(y_te_out, y_pred, zero_division=0),
            "f1": f1_score(y_te_out, y_pred, zero_division=0),
        }
        
        for m, val in fold_results.items():
            outer_metrics[m].append(val)
            all_cv_results.append({
                "model": name,
                "fold": fold_idx + 1,
                "metric": m,
                "value": val
            })

    # Print summary for this model
    metrics_summary = {}
    for m, scores in outer_metrics.items():
        mean_s, std_s = np.mean(scores), np.std(scores)
        print(f"{m:12s}: {mean_s:.3f} (+/- {std_s * 2:.3f})")
        metrics_summary[m] = (mean_s, std_s)
    model_summaries[name] = metrics_summary


# %% [code]
# Save results
cv_results_df = pl.DataFrame(all_cv_results)
cv_results_path = plots_dir / "cv_metrics_per_fold.csv"
cv_results_df.write_csv(cv_results_path)
print(f"CV results saved to {cv_results_path}")


# %% [markdown]
# ## 5. Random Forest Feature Importance
# We fit the Random Forest using the best parameters found via a final GridSearch on the full training set to extract global feature importances.

# %% [code]
print("\nFitting final Random Forest for feature importance...")
rf_grid = GridSearchCV(
    estimator=pipelines["Random Forest"],
    param_grid=param_grids["Random Forest"],
    cv=kf_inner,
    scoring="accuracy",
    n_jobs=-1
)
rf_grid.fit(X_train, y_train, groups=groups_train)
best_rf = rf_grid.best_estimator_


# %% [code]
# Extract feature names from the preprocessor
cat_features = best_rf.named_steps['pre'].transformers_[1][1].get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(cat_features)

# Get importances
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
print(f"Feature importance plot saved to {plots_dir / 'rf_feature_importance.png'}")


# %% [markdown]
# ## 6. Model Comparisons & Findings
#
# ### Logistic Regression
# Logistic Regression provides a strong baseline. Nested CV allows us to see its stability when hyperparameters are tuned purely on training folds.
#
# ### Random Forest
# Random Forest handles non-linearities. Feature importance allows us to identify the most predictive variables.
#
# ### XGBoost
# XGBoost's sensitivity is now handled via the inner tuning loop, providing a fairer comparison.


# %% [markdown]
# ## 7. Final Test Set Validation

# %% [code]
for name in pipelines.keys():
    print(f"\n{'='*50}\n {name} - Test Set\n{'='*50}")
    
    # Tune on full train set then evaluate on test set
    final_grid = GridSearchCV(
        estimator=pipelines[name],
        param_grid=param_grids[name],
        cv=kf_inner,
        scoring="accuracy",
        n_jobs=-1
    )
    final_grid.fit(X_train, y_train, groups=groups_train)
    best_model = final_grid.best_estimator_
    
    y_pred = best_model.predict(X_test)
    print(
        f"""
Accuracy  : {accuracy_score(y_test, y_pred):.3f}
Precision : {precision_score(y_test, y_pred):.3f}
Recall    : {recall_score(y_test, y_pred):.3f}
F1 Score  : {f1_score(y_test, y_pred):.3f}
"""
    )


# %% [markdown]
# ## 8. CV Performance Visualization

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
print(f"Boxplot saved to {plots_dir / 'cv_accuracy_boxplot.png'}")
