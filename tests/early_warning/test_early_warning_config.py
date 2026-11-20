"""Tests for Early Warning Configuration Module."""

from pathlib import Path
import tempfile
import pytest
import yaml

from src.early_warning.early_warning_config import (
    EarlyWarningConfig,
    EarlyWarningConfigError,
    ForecastHorizonsConfig,
    ProjectConfig,
    RiskClassificationConfig,
    RuntimeConfig,
    load_early_warning_config,
)


def test_default_config_validity():
    """Verify default early warning configuration is completely valid."""
    config = EarlyWarningConfig()
    errors = config.validate()
    assert len(errors) == 0
    assert config.project.name == "PerceptGuard AI"
    assert config.runtime.device == "cpu"
    assert config.forecast_horizons.supported == (1, 3, 5)
    assert config.risk_classification.levels == ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_load_default_yaml_file():
    """Verify loading from the checked-in config/early_warning_config.yaml."""
    yaml_path = Path("config/early_warning_config.yaml")
    assert yaml_path.exists(), "config/early_warning_config.yaml must exist"
    config = load_early_warning_config(yaml_path)
    assert config.project.stage == "early_warning"
    assert config.runtime.max_memory_rss_mb == 512
    assert config.forecast_horizons.default_horizon == 3


def test_config_to_dict_and_roundtrip():
    """Verify dictionary serialization and deserialization."""
    config = EarlyWarningConfig()
    data = config.to_dict()
    assert isinstance(data, dict)
    assert "project" in data
    assert "risk_classification" in data

    recreated = EarlyWarningConfig.from_dict(data)
    assert recreated == config


def test_invalid_runtime_device():
    """Reject non-cpu runtime device (e.g. cuda is prohibited under CPU-only constraint)."""
    with pytest.raises(EarlyWarningConfigError) as exc_info:
        EarlyWarningConfig.from_dict({"runtime": {"device": "cuda"}})
    assert "runtime.device must be 'cpu'" in str(exc_info.value)


def test_invalid_runtime_threads():
    """Reject invalid thread counts."""
    with pytest.raises(EarlyWarningConfigError) as exc_info:
        EarlyWarningConfig.from_dict({"runtime": {"num_threads": 0}})
    assert "runtime.num_threads must be a positive integer" in str(exc_info.value)


def test_invalid_forecast_horizons():
    """Reject empty or non-positive horizons."""
    with pytest.raises(EarlyWarningConfigError) as exc_info:
        EarlyWarningConfig.from_dict({"forecast_horizons": {"supported": []}})
    assert "forecast_horizons.supported must be a non-empty sequence" in str(exc_info.value)

    with pytest.raises(EarlyWarningConfigError) as exc_info:
        EarlyWarningConfig.from_dict({"forecast_horizons": {"supported": [1, -3]}})
    assert "must be positive integers" in str(exc_info.value)


def test_invalid_horizon_weights():
    """Reject weights that do not sum to 1.0."""
    with pytest.raises(EarlyWarningConfigError) as exc_info:
        EarlyWarningConfig.from_dict(
            {"forecast_horizons": {"weights": {"h1": 0.1, "h3": 0.1, "h5": 0.1}}}
        )
    assert "forecast_horizons.weights must sum approximately to 1.0" in str(exc_info.value)


def test_invalid_threshold_monotonicity():
    """Reject non-monotonic risk thresholds."""
    with pytest.raises(EarlyWarningConfigError) as exc_info:
        EarlyWarningConfig.from_dict(
            {
                "risk_classification": {
                    "provisional_thresholds": {
                        "low_upper": 0.70,
                        "medium_upper": 0.40,
                        "high_upper": 0.85,
                    }
                }
            }
        )
    assert "provisional thresholds must satisfy 0.0 < low < med < high < 1.0" in str(exc_info.value)


def test_invalid_logging_level():
    """Reject invalid logging verbosity string."""
    with pytest.raises(EarlyWarningConfigError) as exc_info:
        EarlyWarningConfig.from_dict({"logging": {"level": "SUPER_VERBOSE"}})
    assert "logging.level must be one of" in str(exc_info.value)


def test_from_yaml_missing_file():
    """Raise FileNotFoundError when config file does not exist."""
    with pytest.raises(FileNotFoundError):
        EarlyWarningConfig.from_yaml("non_existent_path.yaml")
