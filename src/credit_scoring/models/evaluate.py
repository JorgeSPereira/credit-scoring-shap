"""Model evaluation metrics for credit scoring.

This module provides specialized metrics for credit risk assessment,
including KS statistic and Gini coefficient.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate Kolmogorov-Smirnov statistic.

    KS statistic measures the maximum separation between the cumulative
    distribution of good and bad credit cases. Higher is better.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for positive class.

    Returns:
        KS statistic value (0 to 1).
    """
    # Sort by probability
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df = df.sort_values("y_proba", ascending=False).reset_index(drop=True)

    # Calculate cumulative distributions
    total_good = (df["y_true"] == 0).sum()
    total_bad = (df["y_true"] == 1).sum()

    df["cum_good"] = (df["y_true"] == 0).cumsum() / total_good
    df["cum_bad"] = (df["y_true"] == 1).cumsum() / total_bad

    # KS is the max difference
    ks_statistic = (df["cum_bad"] - df["cum_good"]).max()

    return ks_statistic


def calculate_gini(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate Gini coefficient.

    Gini = 2 * AUC - 1
    Measures the model's ability to rank credit applicants.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for positive class.

    Returns:
        Gini coefficient (-1 to 1, higher is better).
    """
    auc_score = roc_auc_score(y_true, y_proba)
    gini = 2 * auc_score - 1
    return gini


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """Comprehensive model evaluation.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities (optional, for AUC metrics).

    Returns:
        Dictionary of evaluation metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["ks_statistic"] = calculate_ks_statistic(y_true, y_proba)
        metrics["gini"] = calculate_gini(y_true, y_proba)

        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
        metrics["pr_auc"] = auc(recall_curve, precision_curve)

    return metrics


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> None:
    """Print formatted evaluation report.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities.
    """
    print("\n" + "=" * 50)
    print("Model Evaluation Report")
    print("=" * 50)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Good Credit", "Bad Credit"]))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"   TN: {cm[0, 0]:4d}  |  FP: {cm[0, 1]:4d}")
    print(f"   FN: {cm[1, 0]:4d}  |  TP: {cm[1, 1]:4d}")

    # Credit-specific metrics
    if y_proba is not None:
        metrics = evaluate_model(y_true, y_pred, y_proba)
        print("\nCredit Risk Metrics:")
        print(f"   ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"   PR-AUC:       {metrics['pr_auc']:.4f}")
        print(f"   KS Statistic: {metrics['ks_statistic']:.4f}")
        print(f"   Gini:         {metrics['gini']:.4f}")

    print("=" * 50)


def get_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find optimal classification threshold.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        metric: Metric to optimize ('f1', 'youden', 'precision', 'recall').

    Returns:
        Tuple of (optimal_threshold, metric_value).
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred)
        elif metric == "youden":
            # Youden's J statistic: sensitivity + specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            score = sensitivity + specificity - 1
        elif metric == "precision":
            score = precision_score(y_true, y_pred)
        elif metric == "recall":
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
