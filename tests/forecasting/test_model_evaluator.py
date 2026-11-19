"""Tests for ForecastingModelEvaluator (Day 6 - Task 6).

Validates:
1. Target integrity (rejection of heuristic/YOLO confidence ground truth)
2. Ground-truth loading and schema validation
3. Strict temporal alignment by (dataset_id, video_id, target_frame_index, horizon)
4. Split isolation (evaluating strictly on test split)
5. Metric computation (MAE, RMSE, Bias, Median AE, Error Std, MAPE, under/over counts)
6. Baseline model comparison (LSTM vs Persistence, Mean, Trend)
7. Error distribution and breakdown analysis
8. Artifact persistence and schema conformance
9. CLI execution and graceful handling when ground-truth is unavailable
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.forecasting.model_evaluator import (
    AlignedEvaluationPair,
    ForecastingModelEvaluator,
    GroundTruthObservation,
    HorizonEvaluationMetrics,
    ModelEvaluationError,
    main,
)


@pytest.fixture
def sample_ground_truth_dict() -> Dict[str, Any]:
    """Fixture providing valid ground-truth observations across horizons."""
    return {
        "metadata": {
            "version": "1.0",
            "dataset_id": "synthetic_eval_ds",
            "target_definition": "future_object_detection_reliability",
            "target_source": "human_verified_annotation",
            "target_labels_available": True,
            "total_observations": 6,
        },
        "observations": [
            {
                "observation_id": "gt_001",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "target_frame_index": 31,
                "horizon": 1,
                "ground_truth_reliability": 0.85,
                "split": "test",
                "metadata": {"lighting": "normal"},
            },
            {
                "observation_id": "gt_002",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "target_frame_index": 33,
                "horizon": 3,
                "ground_truth_reliability": 0.75,
                "split": "test",
                "metadata": {"lighting": "normal"},
            },
            {
                "observation_id": "gt_003",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "target_frame_index": 35,
                "horizon": 5,
                "ground_truth_reliability": 0.65,
                "split": "test",
                "metadata": {"lighting": "normal"},
            },
            {
                "observation_id": "gt_004",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "target_frame_index": 32,
                "horizon": 1,
                "ground_truth_reliability": 0.90,
                "split": "test",
                "metadata": {"lighting": "bright"},
            },
            {
                "observation_id": "gt_005",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "target_frame_index": 34,
                "horizon": 3,
                "ground_truth_reliability": 0.80,
                "split": "test",
                "metadata": {"lighting": "bright"},
            },
            {
                "observation_id": "gt_006",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "target_frame_index": 36,
                "horizon": 5,
                "ground_truth_reliability": 0.70,
                "split": "test",
                "metadata": {"lighting": "bright"},
            },
        ],
    }


@pytest.fixture
def sample_predictions_dict() -> Dict[str, Any]:
    """Fixture providing multi-horizon predictions matching sample ground truth."""
    return {
        "metadata": {
            "model_name": "LSTMForecaster",
            "model_version": "1.0",
            "split": "test",
            "horizons": [1, 3, 5],
            "total_predictions": 6,
        },
        "predictions": [
            {
                "prediction_id": "pred_001",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "source_frame_index": 30,
                "target_frame_index": 31,
                "horizon": 1,
                "predicted_reliability": 0.82,
                "split": "test",
                "confidence_interval": [0.77, 0.87],
            },
            {
                "prediction_id": "pred_002",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "source_frame_index": 30,
                "target_frame_index": 33,
                "horizon": 3,
                "predicted_reliability": 0.71,
                "split": "test",
                "confidence_interval": [0.65, 0.77],
            },
            {
                "prediction_id": "pred_003",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "source_frame_index": 30,
                "target_frame_index": 35,
                "horizon": 5,
                "predicted_reliability": 0.60,
                "split": "test",
                "confidence_interval": [0.52, 0.68],
            },
            {
                "prediction_id": "pred_004",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "source_frame_index": 31,
                "target_frame_index": 32,
                "horizon": 1,
                "predicted_reliability": 0.92,
                "split": "test",
                "confidence_interval": [0.87, 0.97],
            },
            {
                "prediction_id": "pred_005",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "source_frame_index": 31,
                "target_frame_index": 34,
                "horizon": 3,
                "predicted_reliability": 0.83,
                "split": "test",
                "confidence_interval": [0.77, 0.89],
            },
            {
                "prediction_id": "pred_006",
                "dataset_id": "synthetic_eval_ds",
                "video_id": "test_video_01",
                "source_frame_index": 31,
                "target_frame_index": 36,
                "horizon": 5,
                "predicted_reliability": 0.68,
                "split": "test",
                "confidence_interval": [0.61, 0.75],
            },
        ],
    }


@pytest.fixture
def sample_baseline_results() -> Dict[str, Any]:
    """Fixture providing baseline results structure."""
    return {
        "metadata": {"task": "baseline_evaluation", "version": "1.0"},
        "strategies": {
            "persistence": {
                "h1": {"mae": 0.050, "rmse": 0.065, "sample_count": 2},
                "h3": {"mae": 0.080, "rmse": 0.095, "sample_count": 2},
                "h5": {"mae": 0.110, "rmse": 0.125, "sample_count": 2},
            },
            "mean": {
                "h1": {"mae": 0.070, "rmse": 0.085, "sample_count": 2},
                "h3": {"mae": 0.090, "rmse": 0.105, "sample_count": 2},
                "h5": {"mae": 0.115, "rmse": 0.130, "sample_count": 2},
            },
            "trend": {
                "h1": {"mae": 0.060, "rmse": 0.075, "sample_count": 2},
                "h3": {"mae": 0.085, "rmse": 0.100, "sample_count": 2},
                "h5": {"mae": 0.105, "rmse": 0.120, "sample_count": 2},
            },
        },
    }


# ============================================================================
# 1. Dataclass Unit Tests
# ============================================================================

def test_ground_truth_observation_dataclass():
    """Test GroundTruthObservation initialization and dictionary conversion."""
    obs = GroundTruthObservation(
        observation_id="gt_1",
        dataset_id="ds_1",
        video_id="vid_1",
        target_frame_index=15,
        horizon=3,
        ground_truth_reliability=0.88,
        split="test",
        metadata={"scenario": "highway"},
    )
    assert obs.observation_id == "gt_1"
    assert obs.ground_truth_reliability == 0.88
    d = obs.to_dict()
    assert d["video_id"] == "vid_1"
    assert d["horizon"] == 3


def test_aligned_evaluation_pair_error_computation():
    """Test error, absolute error, and percentage error calculations."""
    pair = AlignedEvaluationPair(
        prediction_id="p1",
        observation_id="o1",
        dataset_id="ds1",
        video_id="v1",
        source_frame_index=10,
        target_frame_index=11,
        horizon=1,
        predicted_reliability=0.80,
        ground_truth_reliability=0.75,
        split="test",
    )
    assert pytest.approx(pair.error, 1e-6) == 0.05
    assert pytest.approx(pair.absolute_error, 1e-6) == 0.05
    assert pytest.approx(pair.percentage_error, 1e-4) == (0.05 / 0.75) * 100.0


def test_aligned_evaluation_pair_zero_ground_truth():
    """Test percentage error handling when ground-truth reliability is zero."""
    pair = AlignedEvaluationPair(
        prediction_id="p1",
        observation_id="o1",
        dataset_id="ds1",
        video_id="v1",
        source_frame_index=10,
        target_frame_index=11,
        horizon=1,
        predicted_reliability=0.10,
        ground_truth_reliability=0.0,
        split="test",
    )
    assert pair.percentage_error == 0.0


def test_horizon_evaluation_metrics_dataclass():
    """Test HorizonEvaluationMetrics initialization and dictionary conversion."""
    m = HorizonEvaluationMetrics(
        horizon=1,
        sample_count=100,
        mae=0.045,
        rmse=0.062,
        bias=-0.012,
        median_absolute_error=0.038,
        error_std=0.060,
        mape=5.5,
        underprediction_count=60,
        overprediction_count=38,
        exact_prediction_count=2,
        min_error=-0.15,
        max_error=0.12,
    )
    d = m.to_dict()
    assert d["horizon"] == 1
    assert d["sample_count"] == 100
    assert d["mae"] == 0.045
    assert d["rmse"] == 0.062


# ============================================================================
# 2. Target Integrity & Ground Truth Validation Tests
# ============================================================================

def test_target_integrity_rejects_yolo_confidence(tmp_path):
    """Verify that evaluator strictly rejects YOLO confidence as ground truth."""
    evaluator = ForecastingModelEvaluator()
    invalid_gt = {
        "metadata": {
            "target_definition": "yolo_detection_confidence",
            "target_labels_available": True,
        },
        "observations": [
            {
                "dataset_id": "d1",
                "video_id": "v1",
                "target_frame_index": 1,
                "horizon": 1,
                "ground_truth_reliability": 0.8,
            }
        ],
    }
    gt_file = tmp_path / "invalid_gt.json"
    with open(gt_file, "w") as f:
        json.dump(invalid_gt, f)

    with pytest.raises(ModelEvaluationError, match="CRITICAL TARGET INTEGRITY VIOLATION"):
        evaluator.load_ground_truth(gt_file)


def test_target_integrity_rejects_heuristic_score(tmp_path):
    """Verify that evaluator rejects heuristic proxy indicators."""
    evaluator = ForecastingModelEvaluator()
    invalid_gt = {
        "metadata": {
            "target_definition": "heuristic_stability_proxy",
            "target_labels_available": True,
        },
        "observations": [],
    }
    gt_file = tmp_path / "invalid_gt2.json"
    with open(gt_file, "w") as f:
        json.dump(invalid_gt, f)

    with pytest.raises(ModelEvaluationError, match="CRITICAL TARGET INTEGRITY VIOLATION"):
        evaluator.load_ground_truth(gt_file)


def test_target_validation_missing_flag(sample_ground_truth_dict):
    """Test validation when target_labels_available is False."""
    sample_ground_truth_dict["metadata"]["target_labels_available"] = False
    evaluator = ForecastingModelEvaluator()
    res = evaluator.validate_ground_truth_contract(sample_ground_truth_dict)
    assert res["valid"] is False
    assert res["status"] == "NOT_AVAILABLE"


def test_target_validation_out_of_bounds_value(sample_ground_truth_dict):
    """Test validation error when ground truth reliability is outside [0, 1]."""
    sample_ground_truth_dict["observations"][0]["ground_truth_reliability"] = 1.5
    evaluator = ForecastingModelEvaluator()
    res = evaluator.validate_ground_truth_contract(sample_ground_truth_dict)
    assert res["valid"] is False
    assert "outside valid bounded range" in res["message"]


def test_target_validation_negative_value(sample_ground_truth_dict):
    """Test validation error when ground truth reliability is negative."""
    sample_ground_truth_dict["observations"][0]["ground_truth_reliability"] = -0.1
    evaluator = ForecastingModelEvaluator()
    res = evaluator.validate_ground_truth_contract(sample_ground_truth_dict)
    assert res["valid"] is False


def test_target_validation_missing_required_keys(sample_ground_truth_dict):
    """Test validation error when an observation is missing required keys."""
    del sample_ground_truth_dict["observations"][0]["target_frame_index"]
    evaluator = ForecastingModelEvaluator()
    res = evaluator.validate_ground_truth_contract(sample_ground_truth_dict)
    assert res["valid"] is False
    assert "missing required keys" in res["message"]


# ============================================================================
# 3. Temporal Alignment & Split Filtering Tests
# ============================================================================

def test_temporal_alignment_exact_match(sample_predictions_dict, sample_ground_truth_dict):
    """Test exact temporal matching by (dataset_id, video_id, target_frame_index, horizon)."""
    evaluator = ForecastingModelEvaluator()
    preds = evaluator.load_predictions(sample_predictions_dict)
    gt = evaluator.load_ground_truth(sample_ground_truth_dict)

    pairs, excluded = evaluator.align_predictions_and_ground_truth(preds, gt, split="test")

    assert len(pairs) == 6
    assert len(excluded) == 0

    p0 = pairs[0]
    assert p0.horizon == 1
    assert p0.target_frame_index == 31
    assert p0.predicted_reliability == 0.82
    assert p0.ground_truth_reliability == 0.85
    assert pytest.approx(p0.error, 1e-6) == -0.03


def test_alignment_handles_unmatched_records(sample_predictions_dict, sample_ground_truth_dict):
    """Test that unmatched predictions and ground-truth observations are recorded in excluded."""
    sample_predictions_dict["predictions"].append({
        "prediction_id": "pred_orphan",
        "dataset_id": "synthetic_eval_ds",
        "video_id": "test_video_01",
        "source_frame_index": 50,
        "target_frame_index": 51,
        "horizon": 1,
        "predicted_reliability": 0.88,
        "split": "test",
    })
    sample_ground_truth_dict["observations"].append({
        "observation_id": "gt_orphan",
        "dataset_id": "synthetic_eval_ds",
        "video_id": "test_video_01",
        "target_frame_index": 99,
        "horizon": 1,
        "ground_truth_reliability": 0.70,
        "split": "test",
    })

    evaluator = ForecastingModelEvaluator()
    preds = evaluator.load_predictions(sample_predictions_dict)
    gt = evaluator.load_ground_truth(sample_ground_truth_dict)

    pairs, excluded = evaluator.align_predictions_and_ground_truth(preds, gt, split="test")

    assert len(pairs) == 6
    assert len(excluded) == 2
    reasons = [e["reason"] for e in excluded]
    assert "UNMATCHED_PREDICTION" in reasons
    assert "UNMATCHED_GROUND_TRUTH" in reasons


def test_alignment_split_filtering(sample_predictions_dict, sample_ground_truth_dict):
    """Test filtering by split (e.g., test only)."""
    sample_predictions_dict["predictions"][0]["split"] = "train"
    sample_ground_truth_dict["observations"][0]["split"] = "train"

    evaluator = ForecastingModelEvaluator()
    preds = evaluator.load_predictions(sample_predictions_dict)
    gt = evaluator.load_ground_truth(sample_ground_truth_dict)

    pairs, _ = evaluator.align_predictions_and_ground_truth(preds, gt, split="test")
    assert len(pairs) == 5

    pairs_all, _ = evaluator.align_predictions_and_ground_truth(preds, gt, split=None)
    assert len(pairs_all) == 6


def test_alignment_duplicate_prediction_handling(sample_predictions_dict, sample_ground_truth_dict):
    """Test duplicate predictions are detected and excluded."""
    dup = dict(sample_predictions_dict["predictions"][0])
    dup["prediction_id"] = "pred_dup"
    sample_predictions_dict["predictions"].append(dup)

    evaluator = ForecastingModelEvaluator()
    preds = evaluator.load_predictions(sample_predictions_dict)
    gt = evaluator.load_ground_truth(sample_ground_truth_dict)

    pairs, excluded = evaluator.align_predictions_and_ground_truth(preds, gt)
    assert len(pairs) == 6
    dup_excluded = [e for e in excluded if e["reason"] == "DUPLICATE_PREDICTION"]
    assert len(dup_excluded) == 1


# ============================================================================
# 4. Metric Computation Tests
# ============================================================================

def test_compute_horizon_metrics_known_values():
    """Verify MAE, RMSE, Bias, and statistics against known hand-calculated values."""
    evaluator = ForecastingModelEvaluator()

    pairs = [
        AlignedEvaluationPair("p1", "o1", "d", "v", 1, 2, 1, 0.80, 0.75, "test"),
        AlignedEvaluationPair("p2", "o2", "d", "v", 2, 3, 1, 0.70, 0.70, "test"),
        AlignedEvaluationPair("p3", "o3", "d", "v", 3, 4, 1, 0.90, 0.80, "test"),
    ]

    m = evaluator.compute_horizon_metrics(pairs, horizon=1)
    assert m.horizon == 1
    assert m.sample_count == 3
    assert pytest.approx(m.mae, 1e-6) == 0.05
    assert pytest.approx(m.rmse, abs=1e-4) == 0.06455
    assert pytest.approx(m.bias, 1e-6) == 0.05
    assert pytest.approx(m.median_absolute_error, 1e-6) == 0.05
    assert m.overprediction_count == 2
    assert m.underprediction_count == 0
    assert m.exact_prediction_count == 1
    assert pytest.approx(m.min_error, 1e-6) == 0.00
    assert pytest.approx(m.max_error, 1e-6) == 0.10


def test_compute_all_horizon_metrics(sample_predictions_dict, sample_ground_truth_dict):
    """Test computing metrics across all requested horizons (1, 3, 5)."""
    evaluator = ForecastingModelEvaluator(horizons=[1, 3, 5])
    preds = evaluator.load_predictions(sample_predictions_dict)
    gt = evaluator.load_ground_truth(sample_ground_truth_dict)
    pairs, _ = evaluator.align_predictions_and_ground_truth(preds, gt)

    metrics = evaluator.compute_all_horizon_metrics(pairs)

    assert "h1" in metrics
    assert "h3" in metrics
    assert "h5" in metrics
    assert metrics["h1"].sample_count == 2
    assert metrics["h3"].sample_count == 2
    assert metrics["h5"].sample_count == 2


def test_compute_horizon_metrics_empty_pairs():
    """Test that computing metrics on empty list raises ModelEvaluationError."""
    evaluator = ForecastingModelEvaluator()
    with pytest.raises(ModelEvaluationError, match="No pairs available"):
        evaluator.compute_horizon_metrics([], horizon=1)


# ============================================================================
# 5. Baseline Comparison Tests
# ============================================================================

def test_compare_against_baselines(sample_baseline_results):
    """Test baseline comparison computation and relative improvement."""
    evaluator = ForecastingModelEvaluator()

    lstm_metrics = {
        "h1": HorizonEvaluationMetrics(1, 2, mae=0.030, rmse=0.040, bias=0.0, median_absolute_error=0.03, error_std=0.01, mape=3.0, underprediction_count=1, overprediction_count=1, exact_prediction_count=0, min_error=-0.03, max_error=0.03),
        "h3": HorizonEvaluationMetrics(3, 2, mae=0.040, rmse=0.050, bias=0.0, median_absolute_error=0.04, error_std=0.01, mape=4.0, underprediction_count=1, overprediction_count=1, exact_prediction_count=0, min_error=-0.04, max_error=0.04),
        "h5": HorizonEvaluationMetrics(5, 2, mae=0.050, rmse=0.060, bias=0.0, median_absolute_error=0.05, error_std=0.01, mape=5.0, underprediction_count=1, overprediction_count=1, exact_prediction_count=0, min_error=-0.05, max_error=0.05),
    }

    comparisons = evaluator.compare_against_baselines(lstm_metrics, sample_baseline_results)

    assert len(comparisons) == 12

    pers_h1 = next(c for c in comparisons if c["model_name"] == "Baseline_Persistence" and c["horizon"] == 1)
    assert pytest.approx(pers_h1["mae"], 1e-4) == 0.050
    assert pytest.approx(pers_h1["improvement_over_persistence_mae_pct"], 1e-2) == 40.0


def test_compare_against_baselines_none_provided():
    """Test baseline comparison when no baseline results are available."""
    evaluator = ForecastingModelEvaluator()
    lstm_metrics = {
        "h1": HorizonEvaluationMetrics(1, 2, mae=0.030, rmse=0.040, bias=0.0, median_absolute_error=0.03, error_std=0.01, mape=3.0, underprediction_count=1, overprediction_count=1, exact_prediction_count=0, min_error=-0.03, max_error=0.03),
    }
    comparisons = evaluator.compare_against_baselines(lstm_metrics, baseline_results_path=None)
    assert len(comparisons) == 1
    assert comparisons[0]["model_name"] == "LSTMForecaster"


# ============================================================================
# 6. Error Analysis Tests
# ============================================================================

def test_perform_error_analysis(sample_predictions_dict, sample_ground_truth_dict):
    """Test statistical error analysis and distribution calculations."""
    evaluator = ForecastingModelEvaluator()
    preds = evaluator.load_predictions(sample_predictions_dict)
    gt = evaluator.load_ground_truth(sample_ground_truth_dict)
    pairs, _ = evaluator.align_predictions_and_ground_truth(preds, gt)

    analysis = evaluator.perform_error_analysis(pairs)

    assert "overall_error_distribution" in analysis
    dist = analysis["overall_error_distribution"]
    assert dist["sample_count"] == 6
    assert "mean" in dist
    assert "std" in dist
    assert "percentiles" in dist
    assert "p50" in dist["percentiles"]

    assert "per_horizon_breakdown" in analysis
    assert "h1" in analysis["per_horizon_breakdown"]
    assert "h3" in analysis["per_horizon_breakdown"]
    assert "h5" in analysis["per_horizon_breakdown"]

    assert "worst_error_cases" in analysis


def test_perform_error_analysis_empty():
    """Test error analysis on empty list returns empty status dictionary."""
    evaluator = ForecastingModelEvaluator()
    res = evaluator.perform_error_analysis([])
    assert res["status"] == "EMPTY"


# ============================================================================
# 7. Artifact Persistence Tests
# ============================================================================

def test_save_evaluation_artifacts(tmp_path, sample_predictions_dict, sample_ground_truth_dict, sample_baseline_results):
    """Test that all 6 required evaluation artifacts are created with valid schemas."""
    evaluator = ForecastingModelEvaluator()
    preds = evaluator.load_predictions(sample_predictions_dict)
    gt = evaluator.load_ground_truth(sample_ground_truth_dict)
    gt_val = evaluator.validate_ground_truth_contract(sample_ground_truth_dict)

    pairs, excluded = evaluator.align_predictions_and_ground_truth(preds, gt)
    metrics = evaluator.compute_all_horizon_metrics(pairs)
    comparisons = evaluator.compare_against_baselines(metrics, sample_baseline_results)
    error_analysis = evaluator.perform_error_analysis(pairs)

    out_dir = tmp_path / "eval_out"
    artifacts = evaluator.save_evaluation_artifacts(
        output_dir=out_dir,
        gt_validation=gt_val,
        metrics=metrics,
        comparisons=comparisons,
        error_analysis=error_analysis,
        excluded_records=excluded,
        aligned_pairs=pairs,
        split="test",
    )

    assert Path(artifacts["evaluation_results"]).exists()
    assert Path(artifacts["evaluation_summary"]).exists()
    assert Path(artifacts["baseline_comparison"]).exists()
    assert Path(artifacts["error_analysis"]).exists()
    assert Path(artifacts["excluded_records"]).exists()
    assert Path(artifacts["horizon_metrics_h1"]).exists()
    assert Path(artifacts["horizon_metrics_h3"]).exists()
    assert Path(artifacts["horizon_metrics_h5"]).exists()

    with open(artifacts["evaluation_summary"], "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert summary_data["evaluation_status"] == "COMPLETED"
    assert summary_data["total_aligned_pairs"] == 6
    assert summary_data["metrics"]["h1"]["sample_count"] == 2


# ============================================================================
# 8. CLI & Missing Ground-Truth Execution Tests
# ============================================================================

def test_evaluator_cli_real_data_reports_not_available():
    """Verify that CLI exits cleanly when evaluating on missing or invalid real ground truth."""
    test_args = [
        "--ground-truth", "artifacts/day5_features/temporal_sequences.json",
    ]
    ret = main(test_args)
    assert ret in (0, 2)


def test_evaluator_cli_synthetic_data_full_run(tmp_path, sample_predictions_dict, sample_ground_truth_dict):
    """Verify CLI end-to-end run on synthetic ground truth and predictions."""
    preds_file = tmp_path / "preds.json"
    gt_file = tmp_path / "gt.json"
    out_dir = tmp_path / "cli_out"

    with open(preds_file, "w") as f:
        json.dump(sample_predictions_dict, f)
    with open(gt_file, "w") as f:
        json.dump(sample_ground_truth_dict, f)

    test_args = [
        "--predictions", str(preds_file),
        "--ground-truth", str(gt_file),
        "--output", str(out_dir),
        "--split", "test",
        "--generate-plots",
    ]

    ret = main(test_args)
    assert ret == 0
    assert (out_dir / "evaluation_summary.json").exists()
    assert (out_dir / "plots" / "actual_vs_predicted_reliability.png").exists()
    assert (out_dir / "plots" / "mae_by_horizon.png").exists()



# ============================================================================
# 9. Additional Edge Case & Robustness Tests
# ============================================================================

def test_evaluator_unsupported_horizon():
    """Test that evaluator raises ModelEvaluationError for unsupported horizons."""
    with pytest.raises(ModelEvaluationError, match="Unsupported horizon"):
        ForecastingModelEvaluator(horizons=[2, 4])


def test_evaluator_custom_supported_horizons():
    """Test initializing evaluator with a subset of supported horizons."""
    ev = ForecastingModelEvaluator(horizons=[1, 5])
    assert ev.horizons == [1, 5]


def test_evaluator_load_predictions_missing_file(tmp_path):
    """Test loading predictions from non-existent file raises ModelEvaluationError."""
    ev = ForecastingModelEvaluator()
    missing_file = tmp_path / "does_not_exist.json"
    with pytest.raises(ModelEvaluationError, match="Predictions source not found"):
        ev.load_predictions(missing_file)


def test_evaluator_load_predictions_missing_key(tmp_path):
    """Test loading predictions from dict missing required keys."""
    ev = ForecastingModelEvaluator()
    invalid_data = [{"dataset_id": "d1"}]  # missing predicted_reliability
    with pytest.raises(ModelEvaluationError, match="missing 'predicted_reliability'"):
        ev.load_predictions(invalid_data)


def test_evaluator_load_ground_truth_missing_file(tmp_path):
    """Test loading ground truth from non-existent file raises ModelEvaluationError."""
    ev = ForecastingModelEvaluator()
    missing_file = tmp_path / "does_not_exist_gt.json"
    with pytest.raises(ModelEvaluationError, match="Ground-truth source not found"):
        ev.load_ground_truth(missing_file)


def test_evaluator_load_ground_truth_non_finite_reliability():
    """Test loading ground truth with non-finite reliability raises error."""
    ev = ForecastingModelEvaluator()
    bad_data = [
        {"dataset_id": "d1", "video_id": "v1", "target_frame_index": 1, "horizon": 1, "ground_truth_reliability": float("nan")}
    ]
    with pytest.raises(ModelEvaluationError, match="non-finite reliability"):
        ev.load_ground_truth(bad_data)


def test_alignment_non_finite_prediction():
    """Test that non-finite predictions are excluded rather than crashing alignment."""
    ev = ForecastingModelEvaluator()
    preds = [{
        "prediction_id": "p_nan",
        "dataset_id": "ds",
        "video_id": "vid",
        "target_frame_index": 10,
        "horizon": 1,
        "predicted_reliability": float("nan"),
        "split": "test",
    }]
    gt = [
        GroundTruthObservation("gt_1", "ds", "vid", 10, 1, 0.8, "test")
    ]
    pairs, excluded = ev.align_predictions_and_ground_truth(preds, gt)
    assert len(pairs) == 0
    assert any(e["reason"] == "NON_FINITE_VALUE" for e in excluded)


def test_directionality_all_underpredictions():
    """Test metric computation when all forecasts underpredict."""
    ev = ForecastingModelEvaluator()
    pairs = [
        AlignedEvaluationPair("p1", "o1", "d", "v", 1, 2, 1, 0.70, 0.80, "test"),
        AlignedEvaluationPair("p2", "o2", "d", "v", 2, 3, 1, 0.60, 0.75, "test"),
    ]
    m = ev.compute_horizon_metrics(pairs, horizon=1)
    assert m.underprediction_count == 2
    assert m.overprediction_count == 0
    assert m.bias < 0.0


def test_directionality_all_overpredictions():
    """Test metric computation when all forecasts overpredict."""
    ev = ForecastingModelEvaluator()
    pairs = [
        AlignedEvaluationPair("p1", "o1", "d", "v", 1, 2, 1, 0.85, 0.80, "test"),
        AlignedEvaluationPair("p2", "o2", "d", "v", 2, 3, 1, 0.80, 0.75, "test"),
    ]
    m = ev.compute_horizon_metrics(pairs, horizon=1)
    assert m.underprediction_count == 0
    assert m.overprediction_count == 2
    assert m.bias > 0.0


def test_error_analysis_worst_cases_sorted():
    """Verify worst error cases in perform_error_analysis are sorted by absolute error descending."""
    ev = ForecastingModelEvaluator()
    pairs = [
        AlignedEvaluationPair("p1", "o1", "d", "v", 1, 2, 1, 0.81, 0.80, "test"),  # err 0.01
        AlignedEvaluationPair("p2", "o2", "d", "v", 2, 3, 1, 0.95, 0.75, "test"),  # err 0.20
        AlignedEvaluationPair("p3", "o3", "d", "v", 3, 4, 1, 0.70, 0.75, "test"),  # err 0.05
    ]
    analysis = ev.perform_error_analysis(pairs)
    worst = analysis["worst_error_cases"]
    assert len(worst) == 3
    assert worst[0]["prediction_id"] == "p2"
    assert worst[1]["prediction_id"] == "p3"
    assert worst[2]["prediction_id"] == "p1"


def test_evaluator_cli_missing_predictions_file(tmp_path, sample_ground_truth_dict):
    """Test CLI behavior when prediction file is missing."""
    gt_file = tmp_path / "valid_gt.json"
    with open(gt_file, "w") as f:
        json.dump(sample_ground_truth_dict, f)

    test_args = [
        "--predictions", str(tmp_path / "non_existent_preds.json"),
        "--ground-truth", str(gt_file),
    ]
    ret = main(test_args)
    assert ret == 1
