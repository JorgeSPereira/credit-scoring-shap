"""Cost-sensitive evaluation for credit scoring models.

This module implements profit/cost analysis considering the asymmetric
costs of different prediction errors in credit decisions.
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


# Default cost matrix for credit scoring (in monetary units)
DEFAULT_COSTS = {
    "tn_profit": 50,      # Profit from lending to good customer
    "tp_profit": 100,     # Savings from rejecting bad customer
    "fp_cost": -200,      # Lost opportunity from rejecting good customer
    "fn_cost": -1000,     # Loss from lending to bad customer (default)
}


def calculate_profit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    costs: Dict[str, float] = None,
) -> Dict[str, float]:
    """Calculate total profit/loss from predictions.

    Args:
        y_true: True labels (0=good, 1=bad).
        y_pred: Predicted labels.
        costs: Cost matrix dictionary.

    Returns:
        Dictionary with profit breakdown and total.
    """
    if costs is None:
        costs = DEFAULT_COSTS

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    profit_breakdown = {
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "profit_tn": tn * costs["tn_profit"],
        "profit_tp": tp * costs["tp_profit"],
        "loss_fp": fp * abs(costs["fp_cost"]),
        "loss_fn": fn * abs(costs["fn_cost"]),
    }

    profit_breakdown["total_profit"] = (
        profit_breakdown["profit_tn"]
        + profit_breakdown["profit_tp"]
        - profit_breakdown["loss_fp"]
        - profit_breakdown["loss_fn"]
    )

    profit_breakdown["profit_per_customer"] = (
        profit_breakdown["total_profit"] / len(y_true)
    )

    return profit_breakdown


def find_optimal_threshold_by_profit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    costs: Dict[str, float] = None,
    thresholds: np.ndarray = None,
) -> Tuple[float, float, pd.DataFrame]:
    """Find threshold that maximizes profit.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities for class 1 (bad).
        costs: Cost matrix.
        thresholds: Thresholds to evaluate.

    Returns:
        Tuple of (optimal_threshold, max_profit, analysis_df).
    """
    if costs is None:
        costs = DEFAULT_COSTS
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    results = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        profit_info = calculate_profit(y_true, y_pred, costs)

        results.append({
            "threshold": thresh,
            "total_profit": profit_info["total_profit"],
            "profit_per_customer": profit_info["profit_per_customer"],
            "fn_count": profit_info["false_negatives"],
            "fp_count": profit_info["false_positives"],
            "approval_rate": (profit_info["true_negatives"] + profit_info["false_negatives"]) / len(y_true),
        })

    df = pd.DataFrame(results)
    best_idx = df["total_profit"].idxmax()
    optimal_threshold = df.loc[best_idx, "threshold"]
    max_profit = df.loc[best_idx, "total_profit"]

    return optimal_threshold, max_profit, df


def comprehensive_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    costs: Dict[str, float] = None,
) -> Dict[str, float]:
    """Comprehensive evaluation including ROC-AUC, PR-AUC, and profit metrics.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        model_name: Name for display.
        costs: Cost matrix.

    Returns:
        Dictionary with all metrics.
    """
    if costs is None:
        costs = DEFAULT_COSTS

    # Standard metrics
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    # Profit-based metrics
    opt_thresh, max_profit, _ = find_optimal_threshold_by_profit(
        y_true, y_proba, costs
    )

    # Metrics at default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    profit_default = calculate_profit(y_true, y_pred_default, costs)

    # Metrics at optimal threshold
    y_pred_optimal = (y_proba >= opt_thresh).astype(int)
    profit_optimal = calculate_profit(y_true, y_pred_optimal, costs)

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "optimal_threshold": opt_thresh,
        "profit_default_thresh": profit_default["total_profit"],
        "profit_optimal_thresh": profit_optimal["total_profit"],
        "profit_improvement": profit_optimal["total_profit"] - profit_default["total_profit"],
        "fn_at_optimal": profit_optimal["false_negatives"],
        "approval_rate_optimal": (profit_optimal["true_negatives"] + profit_optimal["false_negatives"]) / len(y_true),
    }


def plot_profit_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    costs: Dict[str, float] = None,
    save_path: str = None,
) -> plt.Figure:
    """Plot profit as a function of threshold.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        costs: Cost matrix.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    opt_thresh, max_profit, df = find_optimal_threshold_by_profit(
        y_true, y_proba, costs
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Profit vs Threshold
    ax1 = axes[0]
    ax1.plot(df["threshold"], df["total_profit"], "b-", linewidth=2, label="Total Profit")
    ax1.axvline(x=0.5, color="gray", linestyle="--", label="Default (0.5)")
    ax1.axvline(x=opt_thresh, color="red", linestyle="--", label=f"Optimal ({opt_thresh:.2f})")
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    ax1.scatter([opt_thresh], [max_profit], color="red", s=100, zorder=5)
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Total Profit (R$)", fontsize=12)
    ax1.set_title("Profit vs Decision Threshold", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Trade-off FN vs FP
    ax2 = axes[1]
    ax2.plot(df["threshold"], df["fn_count"], "r-", linewidth=2, label="False Negatives (Defaults)")
    ax2.plot(df["threshold"], df["fp_count"], "b-", linewidth=2, label="False Positives (Lost Sales)")
    ax2.axvline(x=opt_thresh, color="green", linestyle="--", label=f"Optimal ({opt_thresh:.2f})")

    ax2.set_xlabel("Threshold", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("FN vs FP Trade-off", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] Profit curve saved to: {save_path}")

    return fig


def print_cost_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    costs: Dict[str, float] = None,
) -> None:
    """Print a comprehensive cost analysis report.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        model_name: Name for display.
        costs: Cost matrix.
    """
    if costs is None:
        costs = DEFAULT_COSTS

    metrics = comprehensive_evaluation(y_true, y_proba, model_name, costs)

    print("=" * 60)
    print(f"COST-SENSITIVE EVALUATION: {model_name}")
    print("=" * 60)

    print("\n--- Cost Matrix ---")
    print(f"  TN Profit (good customer approved):  R$ {costs['tn_profit']:>6}")
    print(f"  TP Profit (bad customer rejected):   R$ {costs['tp_profit']:>6}")
    print(f"  FP Cost (good customer rejected):    R$ {costs['fp_cost']:>6}")
    print(f"  FN Cost (bad customer approved):     R$ {costs['fn_cost']:>6}")

    print("\n--- Performance Metrics ---")
    print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:   {metrics['pr_auc']:.4f}")

    print("\n--- Profit Analysis ---")
    print(f"  Threshold 0.50: R$ {metrics['profit_default_thresh']:>8,.0f}")
    print(f"  Threshold {metrics['optimal_threshold']:.2f}: R$ {metrics['profit_optimal_thresh']:>8,.0f}")
    print(f"  Improvement:    R$ {metrics['profit_improvement']:>8,.0f} ({metrics['profit_improvement']/abs(metrics['profit_default_thresh'])*100:+.1f}%)")

    print("\n--- Recommendation ---")
    print(f"  Use threshold {metrics['optimal_threshold']:.2f} to maximize profit")
    print(f"  Approval rate at optimal: {metrics['approval_rate_optimal']:.1%}")

    print("=" * 60)
