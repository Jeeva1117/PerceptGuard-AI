"""Unit and integration tests for BaselineForecaster (Persistence, Mean, Trend)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from src.forecasting.baseline_data_loader import (
    BaselineDataLoader,
    BaselineDataset,
    SequenceDatasetRecord,
)
from src.forecasting.baseline_forecaster import (
    BaselineForecaster,
    BaselineForecastingError,
    PredictionRecord,
)


class TestBaselineForecaster:
    """Tests for baseline forecasting strategies and multi-horizon contracts."""

    @pytest.fixture
    def real_day5_dir(self) -> Path:
        """Path to real Day 5 sequence directory."""
        p = Path("datasets/processed/driving_sample/features/sequences/run_official")
        if not p.exists():
            pytest.skip("Real Day 5 sequence directory not found")
        return p

    def test_init_defaults(self):
        """Default initialization must configure persistence and horizons 1, 3, 5."""
        bf = BaselineForecaster()
        assert bf.strategy == "persistence"
        assert bf.horizons == [1, 3, 5]
        assert bf.trend_window == 5
        assert bf.target_range == (0.0, 1.0)
        assert bf.training_mean is None
        assert bf.is_fitted is False

    def test_init_invalid_strategy(self):
        """Unsupported strategy name must raise BaselineForecastingError."""
        with pytest.raises(BaselineForecastingError, match="Unsupported baseline strategy"):
            BaselineForecaster(strategy="neural_net")

    def test_persistence_prediction_with_history(self):
        """Strategy A (Persistence) must forecast y_hat_{t+h} = y_t."""
        bf = BaselineForecaster(strategy="persistence")
        rec = SequenceDatasetRecord(
            sequence_id="seq_01",
            dataset_id="test_ds",
            sequence_length=5,
            feature_count=10,
            features=np.zeros((5, 10)),
            start_frame_id="f1",
            end_frame_id="f5",
            start_source_frame_index=0,
            end_source_frame_index=20,
            start_timestamp_seconds=0.0,
            end_timestamp_seconds=0.8,
            split="test",
            horizons={"h1": {"target_frame_id": "f6"}},
            target_labels={1: 0.88},
            target_labels_available=True,
        )

        hist_targets = [0.70, 0.75, 0.80, 0.82, 0.85]
        pred = bf.predict_single(rec, horizon=1, strategy="persistence", historical_target_series=hist_targets)

        assert isinstance(pred, PredictionRecord)
        assert pred.baseline_strategy == "persistence"
        assert pred.predicted_reliability == 0.85
        assert pred.actual_reliability == 0.88
        assert pred.prediction_status == "VALID"
        assert pred.target_horizon == 1
        assert pred.target_frame_id == "f6"

    def test_historical_mean_training_only(self):
        """Strategy B (Historical Mean) must calculate mean strictly from train split."""
        recs = [
            # 2 train records with targets: 0.80 and 0.90 -> mean = 0.85
            SequenceDatasetRecord(
                sequence_id="s1", dataset_id="d", sequence_length=5, feature_count=2,
                features=np.zeros((5, 2)), start_frame_id="f1", end_frame_id="f5",
                start_source_frame_index=0, end_source_frame_index=4,
                start_timestamp_seconds=0.0, end_timestamp_seconds=0.4, split="train",
                horizons={}, target_labels={1: 0.80}, target_labels_available=True,
            ),
            SequenceDatasetRecord(
                sequence_id="s2", dataset_id="d", sequence_length=5, feature_count=2,
                features=np.zeros((5, 2)), start_frame_id="f6", end_frame_id="f10",
                start_source_frame_index=5, end_source_frame_index=9,
                start_timestamp_seconds=0.5, end_timestamp_seconds=0.9, split="train",
                horizons={}, target_labels={1: 0.90}, target_labels_available=True,
            ),
            # 1 test record with target 0.10 -> must NOT affect training mean!
            SequenceDatasetRecord(
                sequence_id="s3", dataset_id="d", sequence_length=5, feature_count=2,
                features=np.zeros((5, 2)), start_frame_id="f11", end_frame_id="f15",
                start_source_frame_index=10, end_source_frame_index=14,
                start_timestamp_seconds=1.0, end_timestamp_seconds=1.4, split="test",
                horizons={}, target_labels={1: 0.10}, target_labels_available=True,
            ),
        ]
        ds = BaselineDataset("d", 5, 2, ["f1", "f2"], recs, np.zeros((3, 5, 2)), True)

        bf = BaselineForecaster(strategy="mean")
        bf.fit(ds)

        assert bf.is_fitted is True
        assert bf.training_sample_count == 2
        assert pytest.approx(bf.training_mean, abs=1e-5) == 0.85

        # Predicting test record must yield training mean 0.85
        pred_test = bf.predict_single(recs[2], horizon=1, strategy="mean")
        assert pred_test.predicted_reliability == 0.85
        assert pred_test.actual_reliability == 0.10

    def test_linear_trend_prediction_and_extrapolation(self):
        """Strategy C (Trend) must fit slope and extrapolate ahead."""
        bf = BaselineForecaster(strategy="trend", trend_window=5)
        rec = SequenceDatasetRecord(
            sequence_id="s1", dataset_id="d", sequence_length=5, feature_count=2,
            features=np.zeros((5, 2)), start_frame_id="f1", end_frame_id="f5",
            start_source_frame_index=0, end_source_frame_index=4,
            start_timestamp_seconds=0.0, end_timestamp_seconds=0.4, split="test",
            horizons={"h1": {"target_frame_id": "f6"}, "h3": {"target_frame_id": "f8"}},
            target_labels={1: None, 3: None}, target_labels_available=False,
        )

        # Perfect linear slope +0.02 per step: [0.70, 0.72, 0.74, 0.76, 0.78]
        hist = [0.70, 0.72, 0.74, 0.76, 0.78]
        # h=1 -> 0.78 + 0.02 * 1 = 0.80
        pred_h1 = bf.predict_single(rec, horizon=1, strategy="trend", historical_target_series=hist)
        assert pred_h1.predicted_reliability == pytest.approx(0.80, abs=1e-5)

        # h=3 -> 0.78 + 0.02 * 3 = 0.84
        pred_h3 = bf.predict_single(rec, horizon=3, strategy="trend", historical_target_series=hist)
        assert pred_h3.predicted_reliability == pytest.approx(0.84, abs=1e-5)

    def test_linear_trend_clipping_bounds(self):
        """Trend extrapolations outside [0.0, 1.0] must be clipped."""
        bf = BaselineForecaster(strategy="trend", target_range=(0.0, 1.0))
        rec = SequenceDatasetRecord(
            sequence_id="s1", dataset_id="d", sequence_length=5, feature_count=2,
            features=np.zeros((5, 2)), start_frame_id="f1", end_frame_id="f5",
            start_source_frame_index=0, end_source_frame_index=4,
            start_timestamp_seconds=0.0, end_timestamp_seconds=0.4, split="test",
            horizons={}, target_labels={}, target_labels_available=False,
        )

        # Steep upward trend -> exceeds 1.0
        hist_up = [0.80, 0.90, 0.95]
        pred_up = bf.predict_single(rec, horizon=5, strategy="trend", historical_target_series=hist_up)
        assert pred_up.predicted_reliability == 1.0

        # Steep downward trend -> falls below 0.0
        hist_down = [0.20, 0.10, 0.05]
        pred_down = bf.predict_single(rec, horizon=5, strategy="trend", historical_target_series=hist_down)
        assert pred_down.predicted_reliability == 0.0

    def test_linear_trend_insufficient_points(self):
        """Trend with fewer than 2 valid points must report NOT AVAILABLE."""
        bf = BaselineForecaster(strategy="trend")
        rec = SequenceDatasetRecord(
            sequence_id="s1", dataset_id="d", sequence_length=5, feature_count=2,
            features=np.zeros((5, 2)), start_frame_id="f1", end_frame_id="f5",
            start_source_frame_index=0, end_source_frame_index=4,
            start_timestamp_seconds=0.0, end_timestamp_seconds=0.4, split="test",
            horizons={}, target_labels={}, target_labels_available=False,
        )
        pred = bf.predict_single(rec, horizon=1, strategy="trend", historical_target_series=[0.8])
        assert pred.predicted_reliability is None
        assert "NOT AVAILABLE" in pred.prediction_status

    def test_prediction_record_schema(self):
        """PredictionRecord must serialize to dict containing all required fields."""
        rec = PredictionRecord(
            dataset_id="driving_sample",
            sequence_id="seq_001",
            input_start_frame="frame_001.jpg",
            input_end_frame="frame_005.jpg",
            target_frame_id="frame_006.jpg",
            target_horizon=1,
            baseline_strategy="persistence",
            predicted_reliability=0.85,
            actual_reliability=0.90,
            prediction_status="VALID",
        )
        d = rec.to_dict()
        assert d["dataset_id"] == "driving_sample"
        assert d["sequence_id"] == "seq_001"
        assert d["target_horizon"] == 1
        assert d["baseline_strategy"] == "persistence"
        assert d["predicted_reliability"] == 0.85
        assert d["actual_reliability"] == 0.90
        assert d["prediction_status"] == "VALID"

    def test_predict_real_day5_dataset_target_honesty(self, real_day5_dir):
        """On real Day 5 data, forecaster must honestly report targets NOT AVAILABLE."""
        loader = BaselineDataLoader(real_day5_dir)
        dataset = loader.load_sequence_dataset(sequence_length=5)

        forecaster = BaselineForecaster(strategy="all", horizons=[1, 3, 5])
        forecaster.fit(dataset.filter_by_split("train"))

        # Training mean should be None because real data has no ground truth targets
        assert forecaster.training_mean is None

        preds = forecaster.predict_dataset(dataset)
        assert len(preds) == len(dataset) * 3 * 3  # 138 seqs * 3 strats * 3 horizons = 1242
        # All predictions must report NOT AVAILABLE
        assert all("NOT AVAILABLE" in p.prediction_status for p in preds)
        assert all(p.predicted_reliability is None for p in preds)

    def test_cli_execution_help(self):
        """CLI --help must return 0."""
        cmd = [sys.executable, "-m", "src.forecasting.baseline_forecaster", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Baseline Perception Reliability Forecaster" in res.stdout

    def test_cli_execution_real_data(self, real_day5_dir):
        """CLI on real Day 5 sequences must execute and return 0 with status PASS."""
        cmd = [
            sys.executable,
            "-m",
            "src.forecasting.baseline_forecaster",
            "--input",
            str(real_day5_dir),
            "--strategy",
            "persistence",
            "--horizons",
            "1",
            "3",
            "5",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "BASELINE MODEL STATUS: [PASS" in res.stdout
        assert "Ground-truth reliability labels are NOT AVAILABLE" in res.stdout

    def test_cli_execution_invalid_directory(self, tmp_path):
        """CLI with invalid input directory must return exit code 1."""
        missing = tmp_path / "does_not_exist"
        cmd = [
            sys.executable,
            "-m",
            "src.forecasting.baseline_forecaster",
            "--input",
            str(missing),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        assert "[FAIL]" in res.stdout
