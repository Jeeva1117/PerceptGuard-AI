"""Unit and Integration Tests for LSTM Training CLI.

Tests CLI argument parsing, target readiness reporting, configuration overrides,
and safe execution stopping when ground-truth labels are unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pytest

from src.forecasting.train_lstm import parse_args, run_training_pipeline


class TestTrainLSTMCLI:
    """Test suite for LSTM training CLI."""

    def test_cli_help_parser(self) -> None:
        """Verify CLI argument parser definition and defaults."""
        args = parse_args([])
        assert args.config == "config/forecasting_config.yaml"
        assert args.device == "cpu"
        assert args.sequence_length == 5
        assert args.verbose is False

    def test_cli_argument_overrides(self) -> None:
        """Verify custom argument overrides."""
        args = parse_args([
            "--config", "config/custom.yaml",
            "--epochs", "25",
            "--batch-size", "16",
            "--learning-rate", "0.0005",
            "--device", "cpu",
            "--sequence-length", "10",
            "--verbose",
        ])
        assert args.config == "config/custom.yaml"
        assert args.epochs == 25
        assert args.batch_size == 16
        assert args.learning_rate == 0.0005
        assert args.sequence_length == 10
        assert args.verbose is True

    def test_cli_missing_config_file(self, tmp_path: Path) -> None:
        """Verify exit code 1 when configuration file does not exist."""
        args = parse_args(["--config", str(tmp_path / "nonexistent.yaml")])
        ret = run_training_pipeline(args)
        assert ret == 1

    def test_cli_real_day5_target_readiness_gate(self, capsys: pytest.CaptureFixture) -> None:
        """Verify CLI honestly halts before Epoch 1 when running on real Day 5 data."""
        args = parse_args([
            "--config", "config/forecasting_config.yaml",
            "--sequence-length", "5",
        ])
        ret = run_training_pipeline(args)
        assert ret == 0  # Clean diagnostic halt

        captured = capsys.readouterr().out
        assert "SUPERVISED TARGET READINESS GATE VERIFICATION" in captured
        assert "NOT AVAILABLE" in captured
        assert "YOLO confidence scores are detection probabilities, NOT future reliability labels" in captured
        assert "Halting training loop before Epoch 1" in captured

    def test_cli_real_day5_all_sequence_lengths(self, capsys: pytest.CaptureFixture) -> None:
        """Verify CLI correctly loads and inspects L=10 on real Day 5 data."""
        args = parse_args([
            "--config", "config/forecasting_config.yaml",
            "--sequence-length", "10",
        ])
        ret = run_training_pipeline(args)
        assert ret == 0

        captured = capsys.readouterr().out
        assert "Sequence Length: L = 10" in captured
        assert "Train=96, Val=13, Test=14" in captured
        assert "NOT AVAILABLE" in captured

    def test_cli_labeled_data_dry_run(self, capsys: pytest.CaptureFixture) -> None:
        """Verify CLI successfully passes target gate and completes dry run with labeled metadata."""
        meta_file = "datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json"
        if not Path(meta_file).exists():
            pytest.skip("Labeled metadata artifact not found on disk")

        args = parse_args([
            "--config", "config/forecasting_config.yaml",
            "--metadata-file", meta_file,
            "--sequence-length", "5",
            "--dry-run",
        ])
        ret = run_training_pipeline(args)
        assert ret == 0

        captured = capsys.readouterr().out
        assert "Using custom metadata file" in captured
        assert "DRY-RUN VERIFICATION SUCCESSFUL" in captured
        assert "Trainable Parameters: 7907" in captured
        assert "No training epochs executed" in captured

    def test_cli_supervised_training_execution(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Verify CLI successfully executes supervised training on labeled metadata for 2 epochs."""
        meta_file = "datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json"
        if not Path(meta_file).exists():
            pytest.skip("Labeled metadata artifact not found on disk")

        out_dir = tmp_path / "exp_cli_test"
        args = parse_args([
            "--config", "config/forecasting_config.yaml",
            "--metadata-file", meta_file,
            "--sequence-length", "5",
            "--epochs", "2",
            "--batch-size", "8",
            "--output", str(out_dir),
            "--loss", "mse",
            "--optimizer", "adamw",
            "--seed", "42",
            "--verbose",
        ])
        ret = run_training_pipeline(args)
        assert ret == 0

        captured = capsys.readouterr().out
        assert "STARTING SUPERVISED TRAINING (2 EPOCHS)" in captured
        assert "Training completed with status: COMPLETED" in captured
        assert "Best Epoch:" in captured

        # Check that artifacts exist
        assert (out_dir / "official_training_experiment" / "training_summary.json").exists() or \
               any(out_dir.glob("**/training_summary.json"))
