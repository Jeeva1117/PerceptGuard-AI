"""Unit and integration tests for ForecastingEnvironmentVerifier."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch

from src.forecasting.forecasting_config import ForecastingConfig
from src.forecasting.forecasting_environment import (
    EnvironmentVerificationResult,
    ForecastingEnvironmentVerifier,
)


class TestForecastingEnvironmentVerifier:
    """Tests for forecasting environment verification and hardware constraints."""

    def test_verifier_init(self):
        """Verifier must initialize with default or specified configuration."""
        verifier = ForecastingEnvironmentVerifier()
        assert verifier.config is not None
        assert verifier.config.runtime.device == "cpu"

    def test_verify_all_on_active_system(self):
        """Verification on current development environment must pass."""
        verifier = ForecastingEnvironmentVerifier()
        result = verifier.verify_all()
        assert isinstance(result, EnvironmentVerificationResult)
        assert result.validation_status == "PASS"
        assert result.python_compatible is True
        assert result.torch_available is True
        assert result.cpu_only_verified is True
        assert result.config_valid is True
        assert result.checkpoint_dir_ready is True
        assert result.experiment_dir_ready is True

    def test_pytorch_cpu_configuration(self):
        """CPU threads must be bounded and torch.get_num_threads() must reflect target."""
        verifier = ForecastingEnvironmentVerifier()
        configured, actual = verifier.configure_cpu_threads(2)
        assert configured == 2
        assert actual == 2
        assert torch.get_num_threads() == 2

    def test_reproducibility_seed(self):
        """Setting seed must be deterministic across random, numpy, torch."""
        verifier = ForecastingEnvironmentVerifier()
        success = verifier.set_reproducibility(12345)
        assert success is True

        r1 = np.random.rand()
        t1 = torch.rand(1).item()

        # Reset and check matching values
        verifier.set_reproducibility(12345)
        r2 = np.random.rand()
        t2 = torch.rand(1).item()

        assert r1 == r2
        assert t1 == t2

    def test_discover_day5_artifacts_real(self):
        """Artifact discovery must find the driving_sample Day 5 sequence outputs."""
        verifier = ForecastingEnvironmentVerifier()
        found, run_path, tensors = verifier.discover_day5_artifacts()
        assert found is True
        assert run_path is not None
        assert "L5" in tensors
        assert "L10" in tensors
        assert "L15" in tensors
        assert "L20" in tensors
        assert all(Path(p).exists() for p in tensors.values())

    def test_discover_day5_artifacts_missing(self, tmp_path):
        """Artifact discovery must return False on empty directory."""
        verifier = ForecastingEnvironmentVerifier()
        found, run_path, tensors = verifier.discover_day5_artifacts(base_dir=tmp_path)
        assert found is False
        assert run_path is None
        assert len(tensors) == 0

    def test_checkpoint_and_experiment_directories_created(self, tmp_path):
        """Verifier must ensure checkpoint and experiment directories exist and are writable."""
        custom_ckpt = tmp_path / "models" / "forecasting"
        custom_exp = tmp_path / "experiments" / "forecasting"

        custom_cfg = tmp_path / "custom_config.yaml"
        cfg = ForecastingConfig()
        cfg_dict = cfg.to_dict()
        cfg_dict["artifacts"]["checkpoint_directory"] = str(custom_ckpt)
        cfg_dict["artifacts"]["experiment_directory"] = str(custom_exp)

        import yaml
        with open(custom_cfg, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f)

        verifier = ForecastingEnvironmentVerifier(config_path=custom_cfg)
        result = verifier.verify_all()
        assert custom_ckpt.exists()
        assert custom_exp.exists()
        assert result.checkpoint_dir_ready is True
        assert result.experiment_dir_ready is True

    def test_supervised_readiness_honesty(self):
        """When Day 5 artifacts lack ground-truth target labels, report NOT AVAILABLE."""
        verifier = ForecastingEnvironmentVerifier()
        result = verifier.verify_all()
        # In current Day 5 outputs, target_labels_available is False
        assert result.ground_truth_labels_available is False
        assert result.supervised_readiness == "NOT AVAILABLE"
        # Crucial: Unrelated environment validation must still PASS
        assert result.validation_status == "PASS"

    def test_cli_execution_success(self):
        """python -m src.forecasting.forecasting_environment must execute and return 0."""
        cmd = [sys.executable, "-m", "src.forecasting.forecasting_environment"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "OVERALL ENVIRONMENT STATUS: [PASS]" in res.stdout
        assert "Supervised Training Ready  : NOT AVAILABLE" in res.stdout
        assert "YOLO confidence alone is NOT ground-truth reliability" in res.stdout

    def test_cli_execution_custom_empty_dataset(self, tmp_path):
        """CLI execution pointing to empty directory reports warning but passes environment check."""
        cmd = [
            sys.executable,
            "-m",
            "src.forecasting.forecasting_environment",
            "--dataset-dir",
            str(tmp_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "OVERALL ENVIRONMENT STATUS: [PASS]" in res.stdout
        assert "Day 5 sequence artifacts not found" in res.stdout
