"""Script to run complete SHAP analysis on trained model."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

from credit_scoring.data.loader import load_german_credit
from credit_scoring.features.engineering import (
    create_preprocessor,
    get_feature_names_after_preprocessing,
    identify_feature_types,
)
from credit_scoring.interpretability.shap_analysis import generate_shap_report


def main():
    """Run SHAP analysis on the trained model."""
    print("=" * 60)
    print("SHAP INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n[*] Loading data...")
    X, y = load_german_credit(save_raw=False)

    # Identify feature types
    numerical_features, categorical_features = identify_feature_types(X)

    # Create preprocessor
    preprocessor = create_preprocessor(numerical_features, categorical_features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Transform data
    print("[*] Preprocessing data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    feature_names = get_feature_names_after_preprocessing(
        preprocessor, numerical_features, categorical_features
    )

    # Clean feature names (remove special characters for SHAP compatibility)
    clean_feature_names = [
        name.replace("<", "lt_").replace(">", "gt_").replace("=", "eq_")
        for name in feature_names
    ]

    # Train XGBoost model fresh for SHAP (avoids serialization issues)
    print("[*] Training XGBoost model for SHAP analysis...")
    classifier = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=2.33,
    )
    classifier.fit(X_train_transformed, y_train)

    # Output directory
    output_dir = Path(__file__).parents[3] / "reports" / "figures"

    # Use a sample for faster SHAP computation
    sample_size = min(500, len(X_test_transformed))
    X_sample = X_test_transformed[:sample_size]

    # Generate SHAP report
    importance_df = generate_shap_report(
        model=classifier,
        X=X_sample,
        feature_names=clean_feature_names,
        output_dir=output_dir,
        model_type="tree",
    )

    # Print top features
    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    print(importance_df.head(10).to_string(index=False))

    print("\n[OK] SHAP analysis complete!")
    print(f"[*] Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

