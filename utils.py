from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer

def get_preprocessor(numeric_cols, categorical_cols):
    """Returns a standard preprocessor for the pipeline."""
    numeric_transformer = Pipeline([
        ("imputer", IterativeImputer(random_state=8675309)),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
