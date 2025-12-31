"""MLflow experiment tracking utilities.

This module provides functions for tracking ML experiments,
logging parameters, metrics, and models with MLflow.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline


def setup_mlflow(
    experiment_name: str = "credit-scoring",
    tracking_uri: Optional[str] = None,
) -> str:
    """Configure MLflow tracking.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: URI for MLflow tracking server. Defaults to local.

    Returns:
        Experiment ID.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Use local mlruns directory with proper file: URI
        mlruns_dir = Path(__file__).parents[3] / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        # Use file: scheme with forward slashes (required by MLflow)
        path_str = str(mlruns_dir.absolute()).replace("\\", "/")
        mlflow.set_tracking_uri(f"file:///{path_str}")

    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    return experiment.experiment_id


def log_experiment(
    model_name: str,
    pipeline: Pipeline,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    X_sample: Any = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Log a complete experiment to MLflow.

    Args:
        model_name: Name of the model being trained.
        pipeline: Trained sklearn Pipeline.
        params: Hyperparameters used.
        metrics: Evaluation metrics.
        X_sample: Sample input for model signature.
        tags: Additional tags for the run.

    Returns:
        Run ID.
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log model
        if X_sample is not None:
            signature = infer_signature(X_sample, pipeline.predict(X_sample))
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
            )
        else:
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
            )

        print(f"[MLflow] Run logged: {run.info.run_id}")
        return run.info.run_id


def log_optuna_study(
    study_name: str,
    best_params: Dict[str, Any],
    best_value: float,
    n_trials: int,
) -> None:
    """Log Optuna study results to MLflow.

    Args:
        study_name: Name of the Optuna study.
        best_params: Best hyperparameters found.
        best_value: Best objective value.
        n_trials: Number of trials executed.
    """
    with mlflow.start_run(run_name=f"optuna-{study_name}"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_score", best_value)
        mlflow.log_metric("n_trials", n_trials)
        mlflow.set_tag("optimization", "optuna")

        print(f"[MLflow] Optuna study '{study_name}' logged")
