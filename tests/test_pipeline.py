"""Tests for the credit scoring pipeline."""

import pytest
import numpy as np
import pandas as pd

from credit_scoring.data.loader import load_german_credit, get_feature_info
from credit_scoring.features.engineering import (
    identify_feature_types,
    create_preprocessor,
)
from credit_scoring.models.evaluate import (
    calculate_ks_statistic,
    calculate_gini,
    evaluate_model,
)


class TestDataLoader:
    """Tests for data loading module."""

    def test_load_german_credit_returns_correct_types(self):
        """Test that loader returns DataFrame and Series."""
        X, y = load_german_credit(save_raw=False)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_load_german_credit_correct_shape(self):
        """Test that dataset has expected dimensions."""
        X, y = load_german_credit(save_raw=False)

        assert X.shape[0] == 1000  # 1000 samples
        assert X.shape[1] == 20  # 20 features
        assert len(y) == 1000

    def test_target_is_binary(self):
        """Test that target variable is binary."""
        _, y = load_german_credit(save_raw=False)

        unique_values = y.unique()
        assert len(unique_values) == 2
        assert set(unique_values) == {0, 1}

    def test_get_feature_info_returns_dict(self):
        """Test that feature info returns dictionary."""
        info = get_feature_info()

        assert isinstance(info, dict)
        assert len(info) == 20


class TestFeatureEngineering:
    """Tests for feature engineering module."""

    def test_identify_feature_types(self):
        """Test feature type identification."""
        X, _ = load_german_credit(save_raw=False)
        numerical, categorical = identify_feature_types(X)

        assert isinstance(numerical, list)
        assert isinstance(categorical, list)
        assert len(numerical) + len(categorical) == X.shape[1]

    def test_create_preprocessor(self):
        """Test preprocessor creation."""
        X, _ = load_german_credit(save_raw=False)
        numerical, categorical = identify_feature_types(X)

        preprocessor = create_preprocessor(numerical, categorical)

        assert preprocessor is not None

    def test_preprocessor_transforms_data(self):
        """Test that preprocessor transforms data correctly."""
        X, _ = load_german_credit(save_raw=False)
        numerical, categorical = identify_feature_types(X)

        preprocessor = create_preprocessor(numerical, categorical)
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == X.shape[0]
        assert not np.isnan(X_transformed).any()


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""

    def test_calculate_ks_statistic(self):
        """Test KS statistic calculation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        ks = calculate_ks_statistic(y_true, y_proba)

        assert 0 <= ks <= 1
        assert ks > 0.5  # Good separation

    def test_calculate_gini(self):
        """Test Gini coefficient calculation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        gini = calculate_gini(y_true, y_proba)

        assert -1 <= gini <= 1
        assert gini > 0.5  # Good model

    def test_evaluate_model_returns_all_metrics(self):
        """Test that evaluate_model returns all expected metrics."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.6, 0.4, 0.7, 0.8])

        metrics = evaluate_model(y_true, y_pred, y_proba)

        expected_keys = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "ks_statistic",
            "gini",
            "pr_auc",
        ]

        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
