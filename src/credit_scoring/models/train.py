"""Model training module for credit scoring.

This module provides functions for training and comparing different
classification models for credit risk assessment, with Optuna optimization
and MLflow tracking.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Import gradient boosting models
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Import MLflow (optional)
try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from credit_scoring.data.loader import load_german_credit
from credit_scoring.features.engineering import (
    create_preprocessor,
    get_feature_names_after_preprocessing,
    identify_feature_types,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Constants
RANDOM_STATE = 42


def get_models() -> Dict[str, Any]:
    """Get dictionary of models to train.

    Returns:
        Dictionary mapping model names to model instances.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=2.33,
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
        )

    return models


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: Any,
    preprocessor: Any,
) -> Pipeline:
    """Train a single model with preprocessing pipeline.

    Args:
        X_train: Training features.
        y_train: Training target.
        model: Sklearn-compatible classifier.
        preprocessor: Fitted preprocessor.

    Returns:
        Fitted pipeline (preprocessor + model).
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def create_objective_xgb(X: pd.DataFrame, y: pd.Series, preprocessor: Any):
    """Create Optuna objective function for XGBoost.

    Args:
        X: Training features.
        y: Training target.
        preprocessor: Preprocessor to use.

    Returns:
        Objective function for Optuna.
    """
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = xgb.XGBClassifier(
            **params,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=2.33,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
        return scores.mean()

    return objective


def create_objective_lgb(X: pd.DataFrame, y: pd.Series, preprocessor: Any):
    """Create Optuna objective function for LightGBM.

    Args:
        X: Training features.
        y: Training target.
        preprocessor: Preprocessor to use.

    Returns:
        Objective function for Optuna.
    """
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = lgb.LGBMClassifier(
            **params,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
        return scores.mean()

    return objective


def optimize_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    n_trials: int = 30,
) -> Tuple[Dict[str, Any], float]:
    """Optimize hyperparameters using Optuna.

    Args:
        X: Features DataFrame.
        y: Target Series.
        model_type: Type of model ("xgboost" or "lightgbm").
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (best_params, best_score).
    """
    numerical_features, categorical_features = identify_feature_types(X)
    preprocessor = create_preprocessor(numerical_features, categorical_features)

    if model_type.lower() == "xgboost" and HAS_XGBOOST:
        objective = create_objective_xgb(X, y, preprocessor)
        study_name = "xgboost_optimization"
    elif model_type.lower() == "lightgbm" and HAS_LIGHTGBM:
        objective = create_objective_lgb(X, y, preprocessor)
        study_name = "lightgbm_optimization"
    else:
        raise ValueError(f"Model type '{model_type}' not supported or not installed.")

    print(f"\n[*] Optimizing {model_type} with Optuna ({n_trials} trials)...")

    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"   [OK] Best ROC-AUC (CV): {study.best_value:.4f}")

    # Log to MLflow if available
    if HAS_MLFLOW:
        try:
            from credit_scoring.tracking.experiment import log_optuna_study, setup_mlflow
            setup_mlflow()
            log_optuna_study(study_name, study.best_params, study.best_value, n_trials)
        except Exception as e:
            print(f"   [WARN] MLflow logging failed: {e}")

    return study.best_params, study.best_value


def train_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    n_trials: int = 30,
    test_size: float = 0.2,
) -> Tuple[Pipeline, pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
    """Train model with Optuna-optimized hyperparameters.

    Args:
        X: Features DataFrame.
        y: Target Series.
        model_type: Type of model to train.
        n_trials: Number of Optuna trials.
        test_size: Proportion for test set.

    Returns:
        Tuple of (pipeline, X_test, y_test, feature_names, best_params).
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Optimize hyperparameters
    best_params, _ = optimize_with_optuna(X_train, y_train, model_type, n_trials)

    # Create optimized model
    numerical_features, categorical_features = identify_feature_types(X)
    preprocessor = create_preprocessor(numerical_features, categorical_features)

    if model_type.lower() == "xgboost":
        model = xgb.XGBClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=2.33,
        )
    else:
        model = lgb.LGBMClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
        )

    # Train final model
    pipeline = train_model(X_train, y_train, model, preprocessor)

    # Get feature names
    preprocessor_fitted = preprocessor.fit(X_train)
    feature_names = get_feature_names_after_preprocessing(
        preprocessor_fitted, numerical_features, categorical_features
    )

    # Evaluate on test set
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"\n[OK] {model_type} trained with Optuna!")
    print(f"   Test ROC-AUC: {test_auc:.4f}")

    # Log to MLflow
    if HAS_MLFLOW:
        try:
            from credit_scoring.tracking.experiment import log_experiment, setup_mlflow
            setup_mlflow()
            log_experiment(
                model_name=f"{model_type}-optuna",
                pipeline=pipeline,
                params=best_params,
                metrics={"roc_auc": test_auc},
                X_sample=X_test.head(5),
                tags={"optimization": "optuna"},
            )
        except Exception as e:
            print(f"   [WARN] MLflow logging failed: {e}")

    return pipeline, X_test, y_test, feature_names, best_params


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Any],
    cv: int = 5,
) -> pd.DataFrame:
    """Evaluate multiple models using cross-validation.

    Args:
        X: Features DataFrame.
        y: Target Series.
        models: Dictionary of models to evaluate.
        cv: Number of cross-validation folds.

    Returns:
        DataFrame with model performance metrics.
    """
    numerical_features, categorical_features = identify_feature_types(X)
    preprocessor = create_preprocessor(numerical_features, categorical_features)

    results = []

    for name, model in models.items():
        print(f"\n[*] Training {name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        cv_scores_auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
        cv_scores_precision = cross_val_score(pipeline, X, y, cv=cv, scoring="precision")
        cv_scores_recall = cross_val_score(pipeline, X, y, cv=cv, scoring="recall")
        cv_scores_f1 = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")

        results.append({
            "Model": name,
            "ROC-AUC": cv_scores_auc.mean(),
            "ROC-AUC Std": cv_scores_auc.std(),
            "Precision": cv_scores_precision.mean(),
            "Recall": cv_scores_recall.mean(),
            "F1-Score": cv_scores_f1.mean(),
        })

        print(f"   [OK] ROC-AUC: {cv_scores_auc.mean():.4f} (+/- {cv_scores_auc.std():.4f})")

    return pd.DataFrame(results)


def train_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "XGBoost",
    test_size: float = 0.2,
) -> Tuple[Pipeline, pd.DataFrame, pd.Series, List[str]]:
    """Train the best model on the full dataset.

    Args:
        X: Features DataFrame.
        y: Target Series.
        model_name: Name of the model to train.
        test_size: Proportion of data for testing.

    Returns:
        Tuple of (fitted_pipeline, X_test, y_test, feature_names).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    numerical_features, categorical_features = identify_feature_types(X)
    preprocessor = create_preprocessor(numerical_features, categorical_features)

    models = get_models()
    if model_name not in models:
        raise ValueError(f"Model {model_name} not available. Choose from: {list(models.keys())}")

    model = models[model_name]
    pipeline = train_model(X_train, y_train, model, preprocessor)

    preprocessor.fit(X_train)
    feature_names = get_feature_names_after_preprocessing(
        preprocessor, numerical_features, categorical_features
    )

    print(f"\n[OK] {model_name} trained successfully!")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    return pipeline, X_test, y_test, feature_names


def save_model(pipeline: Pipeline, filepath: Path | str) -> None:
    """Save trained model to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"[SAVE] Model saved to: {filepath}")


def load_model(filepath: Path | str) -> Pipeline:
    """Load trained model from disk."""
    return joblib.load(filepath)


def main(use_optuna: bool = True, n_trials: int = 30):
    """Main training script.

    Args:
        use_optuna: Whether to use Optuna optimization.
        n_trials: Number of Optuna trials.
    """
    print("=" * 60)
    print("CREDIT SCORING MODEL TRAINING")
    print("=" * 60)

    # Load data
    print("\n[*] Loading German Credit Dataset...")
    X, y = load_german_credit()
    print(f"   Samples: {len(X)}, Features: {X.shape[1]}")

    model_dir = Path(__file__).parents[3] / "models"
    results_dir = Path(__file__).parents[3] / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)

    if use_optuna and HAS_XGBOOST:
        print("\n" + "=" * 60)
        print("OPTUNA HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)

        pipeline, X_test, y_test, feature_names, best_params = train_with_optuna(
            X, y, model_type="xgboost", n_trials=n_trials
        )

        save_model(pipeline, model_dir / "best_model_optuna.joblib")

        # Save params
        params_df = pd.DataFrame([best_params])
        params_df.to_csv(results_dir / "optuna_best_params.csv", index=False)
    else:
        # Fallback to regular training
        models = get_models()
        print(f"\n[*] Models to evaluate: {list(models.keys())}")

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 60)
        results = evaluate_models(X, y, models)
        print("\n" + results.to_string(index=False))

        best_model_name = "XGBoost" if HAS_XGBOOST else "LogisticRegression"
        pipeline, X_test, y_test, feature_names = train_best_model(X, y, best_model_name)

        save_model(pipeline, model_dir / "credit_scoring_model.joblib")
        results.to_csv(results_dir / "model_comparison.csv", index=False)

    print("\n" + "=" * 60)
    print("[OK] Training Complete!")
    print("=" * 60)

    return pipeline, X_test, y_test, feature_names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train credit scoring model")
    parser.add_argument("--no-optuna", action="store_true", help="Disable Optuna optimization")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    args = parser.parse_args()

    main(use_optuna=not args.no_optuna, n_trials=args.trials)

