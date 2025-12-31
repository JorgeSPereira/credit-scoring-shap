"""SHAP interpretability analysis for credit scoring.

This module provides functions for generating SHAP explanations
and visualizations to interpret credit scoring model predictions.
"""

from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def create_shap_explainer(
    model: Any,
    X_background: np.ndarray,
    model_type: str = "tree",
) -> shap.Explainer:
    """Create appropriate SHAP explainer for the model.

    Args:
        model: Trained model (classifier only, not pipeline).
        X_background: Background data for SHAP (preprocessed).
        model_type: Type of model ('tree', 'linear', 'kernel').

    Returns:
        SHAP Explainer instance.
    """
    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X_background)
        else:
            # Kernel SHAP for any model (slower)
            explainer = shap.KernelExplainer(model.predict_proba, X_background[:100])
    except Exception as e:
        # Fallback to generic Explainer if TreeExplainer fails
        print(f"[WARN] TreeExplainer failed: {e}")
        print("[*] Falling back to generic Explainer...")
        # Use masker with background samples
        masker = shap.maskers.Independent(X_background[:100])
        explainer = shap.Explainer(model.predict_proba, masker)

    return explainer


def calculate_shap_values(
    explainer: shap.Explainer,
    X: np.ndarray,
) -> shap.Explanation:
    """Calculate SHAP values for given data.

    Args:
        explainer: SHAP Explainer instance.
        X: Data to explain (preprocessed).

    Returns:
        SHAP Explanation object.
    """
    shap_values = explainer(X)
    return shap_values


def plot_summary(
    shap_values: shap.Explanation,
    feature_names: List[str],
    save_path: Optional[Path] = None,
    max_display: int = 20,
) -> None:
    """Create SHAP summary plot (beeswarm).

    Shows the impact of each feature on model output across all samples.

    Args:
        shap_values: SHAP Explanation object.
        feature_names: List of feature names.
        save_path: Path to save the figure.
        max_display: Maximum number of features to display.
    """
    plt.figure(figsize=(12, 8))

    # Handle multi-output (binary classification)
    if len(shap_values.shape) == 3:
        # Use positive class (index 1)
        values_to_plot = shap_values[:, :, 1]
    else:
        values_to_plot = shap_values

    shap.summary_plot(
        values_to_plot.values,
        values_to_plot.data,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    plt.title("SHAP Summary Plot - Feature Impact on Default Prediction", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] Summary plot saved to: {save_path}")

    plt.close()


def plot_bar(
    shap_values: shap.Explanation,
    feature_names: List[str],
    save_path: Optional[Path] = None,
    max_display: int = 15,
) -> None:
    """Create SHAP bar plot (global feature importance).

    Args:
        shap_values: SHAP Explanation object.
        feature_names: List of feature names.
        save_path: Path to save the figure.
        max_display: Maximum number of features to display.
    """
    plt.figure(figsize=(10, 8))

    # Handle multi-output
    if len(shap_values.shape) == 3:
        values_to_plot = shap_values[:, :, 1]
    else:
        values_to_plot = shap_values

    shap.plots.bar(values_to_plot, max_display=max_display, show=False)

    plt.title("SHAP Feature Importance - Mean |SHAP Value|", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] Bar plot saved to: {save_path}")

    plt.close()


def plot_waterfall(
    shap_values: shap.Explanation,
    index: int,
    feature_names: List[str],
    save_path: Optional[Path] = None,
) -> None:
    """Create SHAP waterfall plot for a single prediction.

    Shows how each feature contributed to a specific prediction.

    Args:
        shap_values: SHAP Explanation object.
        index: Index of the sample to explain.
        feature_names: List of feature names.
        save_path: Path to save the figure.
    """
    plt.figure(figsize=(10, 8))

    # Handle multi-output
    if len(shap_values.shape) == 3:
        sample_shap = shap_values[index, :, 1]
    else:
        sample_shap = shap_values[index]

    shap.plots.waterfall(sample_shap, show=False)

    plt.title(f"SHAP Waterfall - Individual Prediction Explanation (Sample {index})", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] Waterfall plot saved to: {save_path}")

    plt.close()


def plot_force(
    explainer: shap.Explainer,
    shap_values: shap.Explanation,
    index: int,
    feature_names: List[str],
    save_path: Optional[Path] = None,
) -> None:
    """Create SHAP force plot for a single prediction.

    Args:
        explainer: SHAP Explainer instance.
        shap_values: SHAP Explanation object.
        index: Index of the sample to explain.
        feature_names: List of feature names.
        save_path: Path to save the figure.
    """
    # Handle multi-output
    if len(shap_values.shape) == 3:
        sample_shap = shap_values[index, :, 1].values
    else:
        sample_shap = shap_values[index].values

    # Create force plot
    force_plot = shap.force_plot(
        explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value,
        sample_shap,
        shap_values[index].data,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] Force plot saved to: {save_path}")

    plt.close()


def plot_dependence(
    shap_values: shap.Explanation,
    feature_idx: int,
    feature_names: List[str],
    save_path: Optional[Path] = None,
) -> None:
    """Create SHAP dependence plot for a feature.

    Shows the effect of a single feature across the dataset.

    Args:
        shap_values: SHAP Explanation object.
        feature_idx: Index of the feature to plot.
        feature_names: List of feature names.
        save_path: Path to save the figure.
    """
    plt.figure(figsize=(10, 6))

    # Handle multi-output
    if len(shap_values.shape) == 3:
        values_to_plot = shap_values[:, :, 1]
    else:
        values_to_plot = shap_values

    shap.dependence_plot(
        feature_idx,
        values_to_plot.values,
        values_to_plot.data,
        feature_names=feature_names,
        show=False,
    )

    plt.title(f"SHAP Dependence Plot - {feature_names[feature_idx]}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] Dependence plot saved to: {save_path}")

    plt.close()


def generate_shap_report(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    model_type: str = "tree",
) -> pd.DataFrame:
    """Generate complete SHAP analysis report.

    Creates all SHAP visualizations and returns feature importance ranking.

    Args:
        model: Trained model (classifier, not pipeline).
        X: Preprocessed data to analyze.
        feature_names: List of feature names.
        output_dir: Directory to save figures.
        model_type: Type of model for explainer selection.

    Returns:
        DataFrame with feature importance ranking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SHAP INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    # Create explainer
    print("\n[*] Creating SHAP explainer...")
    explainer = create_shap_explainer(model, X, model_type)

    # Calculate SHAP values
    print("[*] Calculating SHAP values...")
    shap_values = calculate_shap_values(explainer, X)

    # Generate plots
    print("\n[*] Generating visualizations...")

    # 1. Summary plot
    plot_summary(
        shap_values,
        feature_names,
        save_path=output_dir / "shap_summary.png",
    )

    # 2. Bar plot
    plot_bar(
        shap_values,
        feature_names,
        save_path=output_dir / "shap_importance.png",
    )

    # 3. Waterfall for first sample with default prediction
    # Find a sample predicted as default
    if len(shap_values.shape) == 3:
        mean_shap = np.abs(shap_values[:, :, 1].values).mean(axis=0)
    else:
        mean_shap = np.abs(shap_values.values).mean(axis=0)

    plot_waterfall(
        shap_values,
        index=0,
        feature_names=feature_names,
        save_path=output_dir / "shap_waterfall_sample.png",
    )

    # 4. Dependence plot for top feature
    top_feature_idx = np.argmax(mean_shap)
    plot_dependence(
        shap_values,
        feature_idx=top_feature_idx,
        feature_names=feature_names,
        save_path=output_dir / f"shap_dependence_{feature_names[top_feature_idx]}.png",
    )

    # Create importance ranking
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": mean_shap,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"\n[SAVE] Feature importance saved to: {output_dir / 'feature_importance.csv'}")

    print("\n[OK] SHAP analysis complete!")
    print(f"   Figures saved to: {output_dir}")

    return importance_df
