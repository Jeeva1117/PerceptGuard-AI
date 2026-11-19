"""Unit tests for baseline evaluation metrics (MAE, RMSE, multi-horizon summary)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.forecasting.baseline_metrics import (
    MetricCalculationError,
    calculate_mae,
    calculate_metrics_summary,
    calculate_rmse,
    evaluate_predictions_by_horizon,
)


class TestBaselineMetrics:
    """Tests for regression metrics and multi-horizon evaluation."""

    def test_mae_exact_values(self):
        """MAE must compute exact mean absolute difference."""
        y_true = [1.0, 0.8, 0.6]
        y_pred = [0.9, 0.8, 0.4]
        # diffs: [0.1, 0.0, 0.2] -> mean = 0.1
        mae = calculate_mae(y_true, y_pred)
        assert pytest.approx(mae, abs=1e-6) == 0.1

    def test_rmse_exact_values(self):
        """RMSE must compute exact root mean squared difference."""
        y_true = [1.0, 0.5]
        y_pred = [0.6, 0.5]
        # sq diffs: [0.16, 0.0] -> mean = 0.08 -> sqrt = 0.2828427
        rmse = calculate_rmse(y_true, y_pred)
        assert pytest.approx(rmse, abs=1e-6) == math.sqrt(0.08)

    def test_mae_identical_arrays_zero(self):
        """Identical predictions must result in zero error."""
        y = [0.25, 0.5, 0.75, 1.0]
        assert calculate_mae(y, y) == 0.0

    def test_rmse_identical_arrays_zero(self):
        """Identical predictions must result in zero RMSE."""
        y = [0.25, 0.5, 0.75, 1.0]
        assert calculate_rmse(y, y) == 0.0

    def test_empty_array_error(self):
        """Empty input arrays must raise MetricCalculationError."""
        with pytest.raises(MetricCalculationError, match="empty input arrays"):
            calculate_mae([], [])

    def test_length_mismatch_error(self):
        """Unequal length arrays must raise MetricCalculationError."""
        with pytest.raises(MetricCalculationError, match="Array length mismatch"):
            calculate_mae([1.0, 0.5], [1.0])

    def test_nan_detection_y_true(self):
        """NaN values in y_true must be detected and rejected."""
        with pytest.raises(MetricCalculationError, match="y_true contains non-finite"):
            calculate_mae([1.0, np.nan], [0.9, 0.8])

    def test_nan_detection_y_pred(self):
        """NaN values in y_pred must be detected and rejected."""
        with pytest.raises(MetricCalculationError, match="y_pred contains non-finite"):
            calculate_rmse([1.0, 0.8], [0.9, np.nan])

    def test_inf_detection(self):
        """Infinity in inputs must raise MetricCalculationError."""
        with pytest.raises(MetricCalculationError, match="non-finite"):
            calculate_mae([1.0, np.inf], [0.9, 0.8])

    def test_calculate_metrics_summary(self):
        """Summary function must return expected keys and rounded statistics."""
        y_true = np.array([0.9, 0.8, 0.7])
        y_pred = np.array([0.85, 0.8, 0.65])
        summary = calculate_metrics_summary(y_true, y_pred, horizon=3, strategy="persistence")

        assert summary["sample_count"] == 3
        assert summary["horizon"] == 3
        assert summary["strategy"] == "persistence"
        assert "mae" in summary
        assert "rmse" in summary
        assert "mean_actual" in summary
        assert "mean_predicted" in summary
        assert summary["mae"] == pytest.approx(0.033333, abs=1e-5)

    def test_evaluate_predictions_empty(self):
        """Evaluating an empty list of predictions must return status EMPTY."""
        res = evaluate_predictions_by_horizon([])
        assert res["status"] == "EMPTY"

    def test_evaluate_predictions_with_valid_targets(self):
        """Multi-horizon predictions with ground-truth targets must calculate metrics."""
        preds = [
            {
                "baseline_strategy": "persistence",
                "target_horizon": 1,
                "predicted_reliability": 0.85,
                "actual_reliability": 0.90,
            },
            {
                "baseline_strategy": "persistence",
                "target_horizon": 1,
                "predicted_reliability": 0.75,
                "actual_reliability": 0.80,
            },
            {
                "baseline_strategy": "persistence",
                "target_horizon": 3,
                "predicted_reliability": 0.70,
                "actual_reliability": 0.60,
            },
        ]
        res = evaluate_predictions_by_horizon(preds, horizons=[1, 3])
        assert res["status"] == "PASS"
        h1 = res["strategies"]["persistence"]["h1"]
        assert h1["status"] == "PASS"
        assert h1["valid_pairs"] == 2
        assert h1["mae"] == pytest.approx(0.05, abs=1e-5)

        h3 = res["strategies"]["persistence"]["h3"]
        assert h3["status"] == "PASS"
        assert h3["valid_pairs"] == 1
        assert h3["mae"] == pytest.approx(0.10, abs=1e-5)

    def test_evaluate_predictions_missing_targets(self):
        """Predictions lacking ground truth must report NOT AVAILABLE honestly."""
        preds = [
            {
                "baseline_strategy": "persistence",
                "target_horizon": 1,
                "predicted_reliability": 0.85,
                "actual_reliability": None,
            },
            {
                "baseline_strategy": "mean",
                "target_horizon": 1,
                "predicted_reliability": 0.70,
                "actual_reliability": None,
            },
        ]
        res = evaluate_predictions_by_horizon(preds, horizons=[1])
        assert res["status"] == "PASS"
        h1_pers = res["strategies"]["persistence"]["h1"]
        assert h1_pers["status"] == "NOT AVAILABLE"
        assert h1_pers["valid_pairs"] == 0
        assert h1_pers["mae"] is None
        assert h1_pers["rmse"] is None
