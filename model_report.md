# Model Performance Report: Binary Classification

## Methodology
- **Splitting Strategy**: Group-aware split using `alias_filled` to prevent patient-level leakage.
- **Cross-Validation**: 10-Fold Stratified Group Cross-Validation.
- **Preprocessing**: `ColumnTransformer` with `SimpleImputer` (median/constant) and `StandardScaler`.

## CV Accuracy Distribution
![CV Accuracy Boxplot](output/cv_accuracy_boxplot.png)

---

### Logistic Regression
- **CV Accuracy**: 0.820 ($\pm 0.214$)
- **Test Accuracy**: 0.847
- **CV Precision**: 0.813 ($\pm 0.224$)
- **CV Recall**: 0.885 ($\pm 0.217$)
- **CV F1**: 0.844 ($\pm 0.192$)

### Random Forest
- **CV Accuracy**: 0.795 ($\pm 0.219$)
- **Test Accuracy**: 0.863
- **CV Precision**: 0.792 ($\pm 0.220$)
- **CV Recall**: 0.869 ($\pm 0.228$)
- **CV F1**: 0.823 ($\pm 0.191$)

### XGBoost
- **CV Accuracy**: 0.766 ($\pm 0.299$)
- **Test Accuracy**: 0.745
- **CV Precision**: 0.760 ($\pm 0.267$)
- **CV Recall**: 0.851 ($\pm 0.319$)
- **CV F1**: 0.798 ($\pm 0.276$)

---

## Final Assessment
Performance is now stable and realistic. High variance across folds suggests sensitivity to specific patient groups, but overall trends are consistent. Random Forest performs best on the held-out test set, while Logistic Regression provides the most stable and strong CV performance.
