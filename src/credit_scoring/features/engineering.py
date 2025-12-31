"""Feature engineering for credit scoring.

This module provides feature transformations and preprocessing
for the German Credit dataset.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numerical and categorical features.

    Args:
        X: Input features DataFrame.

    Returns:
        Tuple of (numerical_features, categorical_features).
    """
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return numerical_features, categorical_features


def create_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Create a sklearn preprocessing pipeline.

    The preprocessor handles:
    - Numerical features: imputation + standardization
    - Categorical features: imputation + one-hot encoding

    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.

    Returns:
        Configured ColumnTransformer.
    """
    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations.

    Creates new features based on domain knowledge:
    - Credit utilization ratios
    - Age groups
    - Risk indicators

    Args:
        X: Input features DataFrame.

    Returns:
        DataFrame with engineered features.
    """
    X_new = X.copy()

    # Get column names (they may vary based on dataset version)
    cols = X_new.columns.tolist()

    # Create age groups if age column exists
    age_col = [c for c in cols if "age" in c.lower() or "Attribute13" in c]
    if age_col:
        age_col = age_col[0]
        X_new["age_group"] = pd.cut(
            X_new[age_col],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["young", "adult", "middle", "senior", "elderly"],
        )

    # Create credit amount groups if credit amount exists
    amount_col = [c for c in cols if "amount" in c.lower() or "Attribute5" in c]
    duration_col = [c for c in cols if "duration" in c.lower() or "Attribute2" in c]

    if amount_col and duration_col:
        amount_col = amount_col[0]
        duration_col = duration_col[0]

        # Monthly payment proxy
        X_new["monthly_payment"] = X_new[amount_col] / X_new[duration_col].replace(
            0, np.nan
        )

        # Credit amount per month of duration
        X_new["credit_intensity"] = np.log1p(X_new[amount_col]) / np.log1p(
            X_new[duration_col]
        )

    return X_new


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
    numerical_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    """Get feature names after preprocessing transformation.

    Args:
        preprocessor: Fitted ColumnTransformer.
        numerical_features: Original numerical feature names.
        categorical_features: Original categorical feature names.

    Returns:
        List of feature names after transformation.
    """
    feature_names = numerical_features.copy()

    # Get one-hot encoded feature names
    if categorical_features:
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = encoder.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_feature_names)

    return feature_names
