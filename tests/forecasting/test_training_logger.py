"""Unit Tests for Training Logger Module.

Tests structured telemetry logging, metadata persistence, JSON summary generation,
and experiment artifact directory structuring.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.forecasting.forecasting_config import ForecastingConfig
from src.forecasting.training_logger import TrainingLogger


class TestTrainingLogger:
    """Test suite for TrainingLogger."""

    def test_training_logger_init(self, tmp_path: Path) -> None:
        """Verify experiment directory creation and metadata file generation."""
        exp_dir = tmp_path / "experiments"
        logger = TrainingLogger(experiment_dir=exp_dir, run_id="test_run_01")

        assert (exp_dir / "test_run_01").exists()
        meta_file = exp_dir / "test_run_01" / "metadata.json"
        assert meta_file.exists()

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["run_id"] == "test_run_01"
        assert "python_version" in meta
        assert "torch_version" in meta
        assert "cpu_count" in meta

    def test_training_logger_save_config(self, tmp_path: Path) -> None:
        """Verify configuration serialization to YAML."""
        logger = TrainingLogger(experiment_dir=tmp_path, run_id="run_cfg")
        config = ForecastingConfig()
        logger.save_config(config)

        cfg_file = tmp_path / "run_cfg" / "experiment_config.yaml"
        assert cfg_file.exists()
        assert cfg_file.stat().st_size > 0

    def test_training_logger_log_epoch(self, tmp_path: Path) -> None:
        """Verify logging epoch telemetry records to JSON."""
        logger = TrainingLogger(experiment_dir=tmp_path, run_id="run_epoch")

        logger.log_epoch(
            epoch=1,
            train_loss=0.082,
            val_loss=0.075,
            learning_rate=0.001,
            epoch_duration_seconds=0.45,
            batch_count=10,
            gradient_norm=0.85,
            early_stopped=False,
        )

        logger.log_epoch(
            epoch=2,
            train_loss=0.065,
            val_loss=0.058,
            learning_rate=0.001,
            epoch_duration_seconds=0.42,
            batch_count=10,
            gradient_norm=0.72,
            early_stopped=False,
        )

        history = logger.get_history()
        assert len(history) == 2
        assert history[0]["epoch"] == 1
        assert history[0]["val_loss"] == pytest.approx(0.075)
        assert history[1]["epoch"] == 2
        assert history[1]["val_loss"] == pytest.approx(0.058)

        # Verify disk persistence
        hist_file = tmp_path / "run_epoch" / "training_history.json"
        assert hist_file.exists()
        with open(hist_file, "r", encoding="utf-8") as f:
            disk_history = json.load(f)
        assert len(disk_history) == 2

    def test_training_logger_save_summary(self, tmp_path: Path) -> None:
        """Verify summary JSON generation upon experiment completion."""
        logger = TrainingLogger(experiment_dir=tmp_path, run_id="run_summary")
        summary_path = logger.save_summary(
            total_epochs=15,
            best_epoch=12,
            best_val_loss=0.035,
            total_duration_seconds=12.4,
            status="EARLY_STOPPED",
            termination_reason="Patience exhausted",
        )

        assert summary_path.exists()
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_epochs"] == 15
        assert data["best_epoch"] == 12
        assert data["best_val_loss"] == pytest.approx(0.035)
        assert data["status"] == "EARLY_STOPPED"
        assert data["termination_reason"] == "Patience exhausted"

    def test_training_logger_log_error(self, tmp_path: Path) -> None:
        """Verify recording runtime errors to errors.json."""
        logger = TrainingLogger(experiment_dir=tmp_path, run_id="run_err")
        logger.log_error("Non-finite loss detected", context={"loss": "nan", "batch": 4})

        err_file = tmp_path / "run_err" / "errors.json"
        assert err_file.exists()
        with open(err_file, "r", encoding="utf-8") as f:
            errs = json.load(f)
        assert len(errs) == 1
        assert "Non-finite loss" in errs[0]["message"]
        assert errs[0]["context"]["batch"] == 4
