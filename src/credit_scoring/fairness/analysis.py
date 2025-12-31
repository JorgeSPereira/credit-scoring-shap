"""Fairness analysis module for credit scoring.

This module provides tools for analyzing and measuring fairness
in credit scoring models, focusing on detecting bias across
sensitive attributes like age and gender.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def create_sensitive_features(
    X: pd.DataFrame,
    age_column: str = "Attribute13",
    gender_column: str = "Attribute9",
) -> pd.DataFrame:
    """Create sensitive feature groups from raw data.

    Args:
        X: Input features DataFrame.
        age_column: Column name containing age.
        gender_column: Column name containing personal status (includes gender).

    Returns:
        DataFrame with sensitive feature groups.
    """
    sensitive = pd.DataFrame(index=X.index)

    # Age groups
    if age_column in X.columns:
        sensitive["age_group"] = pd.cut(
            X[age_column],
            bins=[0, 30, 50, 100],
            labels=["Young (<30)", "Middle (30-50)", "Senior (>50)"],
        )

    # Gender proxy from personal status
    # A91 = male divorced/separated
    # A92 = female divorced/separated/married
    # A93 = male single
    # A94 = male married/widowed
    # A95 = female single
    if gender_column in X.columns:
        gender_map = {
            "A91": "Male",
            "A92": "Female",
            "A93": "Male",
            "A94": "Male",
            "A95": "Female",
        }
        sensitive["gender"] = X[gender_column].map(gender_map).fillna("Unknown")

    return sensitive


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive fairness metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        sensitive_features: Sensitive attribute values.
        sample_weight: Optional sample weights.

    Returns:
        Dictionary containing fairness metrics.
    """
    # Create MetricFrame
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1_score": f1_score,
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "count": count,
    }

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight} if sample_weight else None,
    )

    # Compute disparity metrics
    results = {
        "by_group": metric_frame.by_group.to_dict(),
        "overall": metric_frame.overall.to_dict(),
        "difference": metric_frame.difference().to_dict(),
        "ratio": metric_frame.ratio().to_dict(),
        "demographic_parity_difference": demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        ),
        "demographic_parity_ratio": demographic_parity_ratio(
            y_true, y_pred, sensitive_features=sensitive_features
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        ),
        "equalized_odds_ratio": equalized_odds_ratio(
            y_true, y_pred, sensitive_features=sensitive_features
        ),
    }

    return results


def analyze_fairness(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    sensitive_columns: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Complete fairness analysis for a model.

    Args:
        model: Trained model with predict method.
        X: Input features.
        y_true: True labels.
        sensitive_columns: List of columns to analyze for fairness.

    Returns:
        Dictionary with fairness results for each sensitive attribute.
    """
    # Create sensitive features
    sensitive_df = create_sensitive_features(X)

    if sensitive_columns is None:
        sensitive_columns = sensitive_df.columns.tolist()

    # Get predictions
    y_pred = model.predict(X)

    results = {}

    for col in sensitive_columns:
        if col in sensitive_df.columns:
            print(f"\n[*] Analyzing fairness for: {col}")
            results[col] = compute_fairness_metrics(
                y_true=y_true.values,
                y_pred=y_pred,
                sensitive_features=sensitive_df[col],
            )

    return results


def plot_fairness_comparison(
    fairness_results: Dict[str, Dict],
    metric: str = "selection_rate",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot fairness comparison across groups.

    Args:
        fairness_results: Results from analyze_fairness.
        metric: Metric to visualize.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, len(fairness_results), figsize=(6 * len(fairness_results), 5))

    if len(fairness_results) == 1:
        axes = [axes]

    for ax, (attr, results) in zip(axes, fairness_results.items()):
        by_group = results["by_group"]

        if metric in by_group:
            data = by_group[metric]
            groups = list(data.keys())
            values = list(data.values())

            colors = plt.cm.RdYlGn([0.8 if v < 0.5 else 0.2 for v in values])
            bars = ax.bar(groups, values, color=colors, edgecolor="black")

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            # Add parity line
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Parity")

            ax.set_xlabel(attr, fontweight="bold")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"Fairness: {attr}", fontweight="bold")
            ax.set_ylim(0, 1)
            ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] Fairness plot saved to: {save_path}")

    return fig


def print_fairness_report(fairness_results: Dict[str, Dict]) -> None:
    """Print formatted fairness report.

    Args:
        fairness_results: Results from analyze_fairness.
    """
    print("\n" + "=" * 60)
    print("FAIRNESS ANALYSIS REPORT")
    print("=" * 60)

    for attr, results in fairness_results.items():
        print(f"\n{'=' * 40}")
        print(f"Sensitive Attribute: {attr}")
        print("=" * 40)

        print("\nDemographic Parity:")
        print(f"  Difference: {results['demographic_parity_difference']:.4f}")
        print(f"  Ratio: {results['demographic_parity_ratio']:.4f}")

        print("\nEqualized Odds:")
        print(f"  Difference: {results['equalized_odds_difference']:.4f}")
        print(f"  Ratio: {results['equalized_odds_ratio']:.4f}")

        print("\nSelection Rate by Group:")
        for group, rate in results["by_group"]["selection_rate"].items():
            print(f"  {group}: {rate:.4f}")

        # Fairness verdict
        dpd = abs(results["demographic_parity_difference"])
        if dpd < 0.1:
            verdict = "FAIR (DPD < 0.1)"
        elif dpd < 0.2:
            verdict = "MODERATE BIAS (0.1 <= DPD < 0.2)"
        else:
            verdict = "SIGNIFICANT BIAS (DPD >= 0.2)"

        print(f"\nVerdict: {verdict}")

    print("\n" + "=" * 60)
