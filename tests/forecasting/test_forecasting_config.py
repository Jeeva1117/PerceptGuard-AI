"""Unit and integration tests for ForecastingConfig and related schemas."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import pytest
import yaml

from src.forecasting.forecasting_config import (
    ArtifactsConfig,
    DataConfig,
    EvaluationConfig,
    ForecastingConfig,
    ForecastingConfigError,
    ModelConfig,
    ProjectConfig,
    RuntimeConfig,
    TrainingConfig,
)


class TestForecastingConfig:
    """Tests for forecasting configuration loading, serialization, and validation."""

    def test_default_config_valid(self):
        """Default configuration must pass validation."""
        config = ForecastingConfig()
        is_valid, errors = config.validate(raise_on_error=False)
        assert is_valid is True
        assert len(errors) == 0

    def test_from_yaml_valid(self):
        """Standard config/forecasting_config.yaml must load and validate."""
        config_path = Path("config/forecasting_config.yaml")
        assert config_path.exists()
        config = ForecastingConfig.from_yaml(config_path, validate=True)
        assert config.project.name == "PerceptGuard AI"
        assert config.runtime.device == "cpu"
        assert config.runtime.num_threads == 2
        assert config.data.sequence_lengths == [5, 10, 15, 20]
        assert config.data.forecast_horizons == [1, 3, 5]
        assert config.model.architecture == "lstm"
        assert config.model.hidden_size == 32
        assert config.training.learning_rate == 0.001
        assert "mae" in config.evaluation.metrics

    def test_to_dict_and_from_dict_roundtrip(self):
        """Converting to dict and reconstructing must preserve all settings."""
        orig = ForecastingConfig()
        d = orig.to_dict()
        assert isinstance(d, dict)
        assert d["project"]["name"] == "PerceptGuard AI"
        assert d["data"]["batch_size"] == 8

        reconstructed = ForecastingConfig.from_dict(d, validate=True)
        assert reconstructed == orig

    def test_save_yaml_and_reload(self, tmp_path):
        """Saving to a YAML file and reloading must yield identical configuration."""
        target_file = tmp_path / "saved_config.yaml"
        config = ForecastingConfig(
            project=ProjectConfig(name="CustomProject", random_seed=123),
            runtime=RuntimeConfig(num_threads=4),
            model=ModelConfig(architecture="gru", hidden_size=64),
        )
        config.save_yaml(target_file)
        assert target_file.exists()

        reloaded = ForecastingConfig.from_yaml(target_file, validate=True)
        assert reloaded.project.name == "CustomProject"
        assert reloaded.project.random_seed == 123
        assert reloaded.runtime.num_threads == 4
        assert reloaded.model.architecture == "gru"
        assert reloaded.model.hidden_size == 64

    def test_project_config_validation(self):
        """ProjectConfig must reject invalid names, stages, and seeds."""
        p1 = ProjectConfig(name="")
        assert len(p1.validate()) > 0

        p2 = ProjectConfig(stage="")
        assert len(p2.validate()) > 0

        p3 = ProjectConfig(random_seed=-1)
        assert len(p3.validate()) > 0

    def test_runtime_config_device_validation(self):
        """RuntimeConfig must reject unsupported devices."""
        r_bad = RuntimeConfig(device="tpu")
        errs = r_bad.validate()
        assert any("device must be 'cpu' or 'cuda'" in e for e in errs)

    def test_runtime_config_threads_boundary(self):
        """RuntimeConfig must enforce num_threads >= 1."""
        r_zero = RuntimeConfig(num_threads=0)
        assert any("num_threads must be an integer >= 1" in e for e in r_zero.validate())

        r_neg = RuntimeConfig(num_threads=-2)
        assert any("num_threads must be an integer >= 1" in e for e in r_neg.validate())

    def test_data_config_validation(self):
        """DataConfig must validate sequence lengths, horizons, batch size, and workers."""
        d1 = DataConfig(sequence_lengths=[])
        assert any("sequence_lengths must be a non-empty list" in e for e in d1.validate())

        d2 = DataConfig(sequence_lengths=[5, -1])
        assert any("entries must be positive integers" in e for e in d2.validate())

        d3 = DataConfig(forecast_horizons=[])
        assert any("forecast_horizons must be a non-empty list" in e for e in d3.validate())

        d4 = DataConfig(batch_size=0)
        assert any("batch_size must be an integer >= 1" in e for e in d4.validate())

        d5 = DataConfig(num_workers=-1)
        assert any("num_workers must be an integer >= 0" in e for e in d5.validate())

    def test_model_config_architecture(self):
        """ModelConfig must reject unsupported architectures."""
        m_bad = ModelConfig(architecture="transformer")
        assert any("architecture must be 'lstm' or 'gru'" in e for e in m_bad.validate())

        m_lstm = ModelConfig(architecture="lstm")
        assert len(m_lstm.validate([1, 3, 5])) == 0

        m_gru = ModelConfig(architecture="gru")
        assert len(m_gru.validate([1, 3, 5])) == 0

    def test_model_config_input_size(self):
        """ModelConfig input_size must be 'auto' or a positive integer."""
        m_auto = ModelConfig(input_size="auto")
        assert len(m_auto.validate([1, 3, 5])) == 0

        m_int = ModelConfig(input_size=27)
        assert len(m_int.validate([1, 3, 5])) == 0

        m_bad = ModelConfig(input_size=0)
        assert any("input_size must be 'auto' or a positive integer" in e for e in m_bad.validate([1, 3, 5]))

    def test_model_config_hidden_size_boundary(self):
        """ModelConfig hidden_size must be >= 4."""
        m_low = ModelConfig(hidden_size=2)
        assert any("hidden_size must be an integer >= 4" in e for e in m_low.validate([1, 3, 5]))

    def test_model_config_dropout_boundary(self):
        """ModelConfig dropout must be in [0.0, 1.0)."""
        m_neg = ModelConfig(dropout=-0.1)
        assert any("dropout must be in [0.0, 1.0)" in e for e in m_neg.validate([1, 3, 5]))

        m_one = ModelConfig(dropout=1.0)
        assert any("dropout must be in [0.0, 1.0)" in e for e in m_one.validate([1, 3, 5]))

        m_valid = ModelConfig(dropout=0.2)
        assert len(m_valid.validate([1, 3, 5])) == 0

    def test_model_config_output_horizons_subset(self):
        """ModelConfig output_horizons must match or be subset of data horizons."""
        m_bad = ModelConfig(output_horizons=[1, 7])
        errs = m_bad.validate(data_horizons=[1, 3, 5])
        assert any("not found in data.forecast_horizons" in e for e in errs)

    def test_training_config_boundaries(self):
        """TrainingConfig must enforce valid bounds on optimization parameters."""
        t1 = TrainingConfig(learning_rate=0.0)
        assert any("learning_rate must be > 0.0" in e for e in t1.validate())

        t2 = TrainingConfig(weight_decay=-0.01)
        assert any("weight_decay must be >= 0.0" in e for e in t2.validate())

        t3 = TrainingConfig(max_epochs=0)
        assert any("max_epochs must be an integer >= 1" in e for e in t3.validate())

        t4 = TrainingConfig(early_stopping_patience=0)
        assert any("early_stopping_patience must be an integer >= 1" in e for e in t4.validate())

    def test_evaluation_config_metrics(self):
        """EvaluationConfig must validate metric names."""
        e_bad = EvaluationConfig(metrics=["invalid_metric"])
        assert any("not a recognized metric" in err for err in e_bad.validate())

        e_empty = EvaluationConfig(metrics=[])
        assert any("must be a non-empty list" in err for err in e_empty.validate())

        e_ok = EvaluationConfig(metrics=["mae", "rmse", "mse"])
        assert len(e_ok.validate()) == 0

    def test_raise_on_error_exception(self):
        """ForecastingConfig.validate(raise_on_error=True) must raise ForecastingConfigError."""
        bad_config = ForecastingConfig(
            project=ProjectConfig(name=""),
            runtime=RuntimeConfig(num_threads=-1),
        )
        with pytest.raises(ForecastingConfigError) as exc_info:
            bad_config.validate(raise_on_error=True)
        assert "validation failed" in str(exc_info.value)

    def test_cli_execution_valid(self):
        """CLI validator on standard config must return exit code 0."""
        cmd = [sys.executable, "-m", "src.forecasting.forecasting_config", "--validate"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "[PASS] Configuration is VALID" in res.stdout

    def test_cli_execution_invalid_file(self, tmp_path):
        """CLI validator on missing file must return exit code 1."""
        missing = tmp_path / "nonexistent.yaml"
        cmd = [sys.executable, "-m", "src.forecasting.forecasting_config", "--config", str(missing), "--validate"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        assert "[FAIL]" in res.stdout
