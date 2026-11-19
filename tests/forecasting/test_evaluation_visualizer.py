"""Tests for ForecastingEvaluationVisualizer (Day 6 - Task 6).

Validates:
1. Headless Matplotlib execution (non-interactive Agg backend)
2. Actual vs Predicted reliability chart generation
3. MAE by Horizon bar chart generation
4. RMSE by Horizon bar chart generation
5. Prediction Error Distribution histogram generation
6. Baseline vs LSTM comparison grouped bar chart generation
7. Residual analysis scatter plot generation
8. Batch generation of all diagnostic plots
9. Graceful exception and edge case handling (empty inputs, single point)
"""

from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.forecasting.evaluation_visualizer import (
    ForecastingEvaluationVisualizer,
    VisualizationError,
)
from src.forecasting.model_evaluator import (
    AlignedEvaluationPair,
    HorizonEvaluationMetrics,
)


@pytest.fixture
def sample_pairs() -> List[AlignedEvaluationPair]:
    """Fixture providing sample aligned pairs across horizons 1, 3, 5."""
    pairs = []
    # Horizon 1
    pairs.append(AlignedEvaluationPair("p1", "o1", "d", "v", 10, 11, 1, 0.85, 0.82, "test"))
    pairs.append(AlignedEvaluationPair("p2", "o2", "d", "v", 11, 12, 1, 0.78, 0.80, "test"))
    # Horizon 3
    pairs.append(AlignedEvaluationPair("p3", "o3", "d", "v", 10, 13, 3, 0.75, 0.70, "test"))
    pairs.append(AlignedEvaluationPair("p4", "o4", "d", "v", 11, 14, 3, 0.72, 0.74, "test"))
    # Horizon 5
    pairs.append(AlignedEvaluationPair("p5", "o5", "d", "v", 10, 15, 5, 0.65, 0.60, "test"))
    pairs.append(AlignedEvaluationPair("p6", "o6", "d", "v", 11, 16, 5, 0.62, 0.65, "test"))
    return pairs


@pytest.fixture
def sample_metrics() -> Dict[str, HorizonEvaluationMetrics]:
    """Fixture providing sample horizon evaluation metrics."""
    return {
        "h1": HorizonEvaluationMetrics(
            horizon=1, sample_count=2, mae=0.025, rmse=0.026, bias=0.005,
            median_absolute_error=0.025, error_std=0.025, mape=3.1,
            underprediction_count=1, overprediction_count=1, exact_prediction_count=0,
            min_error=-0.02, max_error=0.03
        ),
        "h3": HorizonEvaluationMetrics(
            horizon=3, sample_count=2, mae=0.035, rmse=0.037, bias=0.015,
            median_absolute_error=0.035, error_std=0.035, mape=4.8,
            underprediction_count=1, overprediction_count=1, exact_prediction_count=0,
            min_error=-0.02, max_error=0.05
        ),
        "h5": HorizonEvaluationMetrics(
            horizon=5, sample_count=2, mae=0.040, rmse=0.042, bias=0.010,
            median_absolute_error=0.040, error_std=0.040, mape=6.3,
            underprediction_count=1, overprediction_count=1, exact_prediction_count=0,
            min_error=-0.03, max_error=0.05
        ),
    }


@pytest.fixture
def sample_comparison_records() -> List[Dict[str, Any]]:
    """Fixture providing sample baseline comparison records."""
    return [
        {"model_name": "LSTMForecaster", "horizon": 1, "mae": 0.025, "rmse": 0.026},
        {"model_name": "Baseline_Persistence", "horizon": 1, "mae": 0.050, "rmse": 0.065},
        {"model_name": "Baseline_Mean", "horizon": 1, "mae": 0.070, "rmse": 0.085},
        {"model_name": "LSTMForecaster", "horizon": 3, "mae": 0.035, "rmse": 0.037},
        {"model_name": "Baseline_Persistence", "horizon": 3, "mae": 0.080, "rmse": 0.095},
        {"model_name": "Baseline_Mean", "horizon": 3, "mae": 0.090, "rmse": 0.105},
        {"model_name": "LSTMForecaster", "horizon": 5, "mae": 0.040, "rmse": 0.042},
        {"model_name": "Baseline_Persistence", "horizon": 5, "mae": 0.110, "rmse": 0.125},
        {"model_name": "Baseline_Mean", "horizon": 5, "mae": 0.115, "rmse": 0.130},
    ]


# ============================================================================
# 1. Initialization Tests
# ============================================================================

def test_visualizer_initialization():
    """Test default and custom visualizer initialization."""
    vis_default = ForecastingEvaluationVisualizer()
    assert vis_default.dpi == 300
    assert vis_default.figsize_standard == (10, 6)

    vis_custom = ForecastingEvaluationVisualizer(dpi=150, figsize_standard=(8, 5))
    assert vis_custom.dpi == 150
    assert vis_custom.figsize_standard == (8, 5)


# ============================================================================
# 2. Individual Plot Generation Tests
# ============================================================================

def test_plot_actual_vs_predicted(tmp_path, sample_pairs):
    """Test actual vs predicted scatter plot generation."""
    vis = ForecastingEvaluationVisualizer(dpi=100)
    out_file = tmp_path / "actual_vs_pred.png"
    result = vis.plot_actual_vs_predicted(sample_pairs, out_file)

    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_actual_vs_predicted_empty(tmp_path):
    """Test actual vs predicted raises VisualizationError on empty pairs."""
    vis = ForecastingEvaluationVisualizer()
    out_file = tmp_path / "dummy.png"
    with pytest.raises(VisualizationError, match="pairs list is empty"):
        vis.plot_actual_vs_predicted([], out_file)


def test_plot_mae_by_horizon(tmp_path, sample_metrics):
    """Test MAE by horizon bar chart generation."""
    vis = ForecastingEvaluationVisualizer(dpi=100)
    out_file = tmp_path / "mae_horizon.png"
    result = vis.plot_mae_by_horizon(sample_metrics, out_file)

    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_mae_by_horizon_empty(tmp_path):
    """Test MAE plot raises VisualizationError on empty metrics."""
    vis = ForecastingEvaluationVisualizer()
    out_file = tmp_path / "dummy.png"
    with pytest.raises(VisualizationError, match="metrics dictionary is empty"):
        vis.plot_mae_by_horizon({}, out_file)


def test_plot_rmse_by_horizon(tmp_path, sample_metrics):
    """Test RMSE by horizon bar chart generation."""
    vis = ForecastingEvaluationVisualizer(dpi=100)
    out_file = tmp_path / "rmse_horizon.png"
    result = vis.plot_rmse_by_horizon(sample_metrics, out_file)

    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_rmse_by_horizon_empty(tmp_path):
    """Test RMSE plot raises VisualizationError on empty metrics."""
    vis = ForecastingEvaluationVisualizer()
    out_file = tmp_path / "dummy.png"
    with pytest.raises(VisualizationError, match="metrics dictionary is empty"):
        vis.plot_rmse_by_horizon({}, out_file)


def test_plot_error_distribution(tmp_path, sample_pairs):
    """Test error distribution histogram generation."""
    vis = ForecastingEvaluationVisualizer(dpi=100)
    out_file = tmp_path / "err_dist.png"
    result = vis.plot_error_distribution(sample_pairs, out_file)

    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_error_distribution_empty(tmp_path):
    """Test error distribution raises VisualizationError on empty pairs."""
    vis = ForecastingEvaluationVisualizer()
    out_file = tmp_path / "dummy.png"
    with pytest.raises(VisualizationError, match="pairs list is empty"):
        vis.plot_error_distribution([], out_file)


def test_plot_baseline_comparison(tmp_path, sample_comparison_records):
    """Test baseline comparison grouped bar chart generation."""
    vis = ForecastingEvaluationVisualizer(dpi=100)
    out_file = tmp_path / "baseline_comp.png"
    result = vis.plot_baseline_comparison(sample_comparison_records, out_file)

    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_baseline_comparison_empty(tmp_path):
    """Test baseline comparison raises VisualizationError on empty records."""
    vis = ForecastingEvaluationVisualizer()
    out_file = tmp_path / "dummy.png"
    with pytest.raises(VisualizationError, match="records list is empty"):
        vis.plot_baseline_comparison([], out_file)


def test_plot_residuals(tmp_path, sample_pairs):
    """Test residual scatter plot generation."""
    vis = ForecastingEvaluationVisualizer(dpi=100)
    out_file = tmp_path / "residuals.png"
    result = vis.plot_residuals(sample_pairs, out_file)

    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_residuals_empty(tmp_path):
    """Test residuals plot raises VisualizationError on empty pairs."""
    vis = ForecastingEvaluationVisualizer()
    out_file = tmp_path / "dummy.png"
    with pytest.raises(VisualizationError, match="pairs list is empty"):
        vis.plot_residuals([], out_file)


# ============================================================================
# 3. Batch Plot Suite Tests
# ============================================================================

def test_generate_all_plots(tmp_path, sample_pairs, sample_metrics, sample_comparison_records):
    """Test generating full suite of 6 diagnostic plots."""
    vis = ForecastingEvaluationVisualizer(dpi=100)
    plot_dir = tmp_path / "all_plots"

    plots = vis.generate_all_plots(
        pairs=sample_pairs,
        metrics=sample_metrics,
        comparison_records=sample_comparison_records,
        output_dir=plot_dir,
    )

    assert len(plots) == 6
    assert "actual_vs_predicted" in plots
    assert "mae_by_horizon" in plots
    assert "rmse_by_horizon" in plots
    assert "error_distribution" in plots
    assert "residual_analysis" in plots
    assert "baseline_comparison" in plots

    for name, p in plots.items():
        assert p.exists()
        assert p.stat().st_size > 1000


def test_generate_all_plots_no_baselines(tmp_path, sample_pairs, sample_metrics):
    """Test generating plots when no baseline comparisons are provided."""
    vis = ForecastingEvaluationVisualizer(dpi=100)
    plot_dir = tmp_path / "no_baselines"

    plots = vis.generate_all_plots(
        pairs=sample_pairs,
        metrics=sample_metrics,
        comparison_records=None,
        output_dir=plot_dir,
    )

    # 5 plots generated (excluding baseline comparison)
    assert len(plots) == 5
    assert "baseline_comparison" not in plots


def test_generate_all_plots_empty_pairs(tmp_path, sample_metrics):
    """Test that empty pairs returns empty dict gracefully without crash."""
    vis = ForecastingEvaluationVisualizer(dpi=100)
    plot_dir = tmp_path / "empty_pairs"

    plots = vis.generate_all_plots(
        pairs=[],
        metrics=sample_metrics,
        output_dir=plot_dir,
    )

    assert plots == {}
