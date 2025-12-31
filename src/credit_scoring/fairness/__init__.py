"""Fairness analysis for credit scoring models."""

from credit_scoring.fairness.analysis import (
    analyze_fairness,
    compute_fairness_metrics,
    plot_fairness_comparison,
)

__all__ = ["analyze_fairness", "compute_fairness_metrics", "plot_fairness_comparison"]
