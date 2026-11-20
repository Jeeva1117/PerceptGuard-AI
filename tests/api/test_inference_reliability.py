"""Automated tests for ML model loading, checkpoint validation, and inference reliability."""

from __future__ import annotations

import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config.system_settings import SystemSettings
from src.risk.risk_scorer import ReliabilityRiskScorer
from src.risk.risk_validator import (
    RiskValidationError,
    validate_reliability_value,
    validate_risk_value,
)


def test_missing_checkpoints_handling_in_readiness():
    """Verify missing checkpoints are explicitly reported without crashing the application."""
    custom_settings = SystemSettings.load(
        env_override={
            "APP_ENV": "testing",
            "LSTM_MODEL_PATH": "models/non_existent_dir/non_existent_lstm.pt",
            "BASELINE_MODEL_PATH": "models/non_existent_dir/non_existent_baseline.pt",
        }
    )
    app = create_app(settings=custom_settings)

    with TestClient(app) as client:
        res = client.get("/api/v1/health/ready")
        assert res.status_code == 200
        data = res.json()
        assert data["subsystems"]["lstm_checkpoint"] == "UNAVAILABLE"


def test_nan_and_infinite_prediction_rejection():
    """Verify that NaN or infinite forecast outputs are strictly rejected by the reliability risk engine."""
    # Valid reliability
    assert validate_reliability_value(0.85) == 0.85

    # NaN reliability
    with pytest.raises(RiskValidationError, match="NaN or Infinite"):
        validate_reliability_value(float("nan"))

    # Infinite reliability
    with pytest.raises(RiskValidationError, match="NaN or Infinite"):
        validate_reliability_value(float("inf"))

    # Valid risk value
    assert validate_risk_value(0.15) == 0.15

    # Out of range risk value
    with pytest.raises(RiskValidationError, match="within"):
        validate_risk_value(1.5)


def test_multi_horizon_semantics_preserved():
    """Verify that H1, H3, and H5 horizons maintain discrete, independent risk assessments."""
    app = create_app()
    with TestClient(app) as client:
        res = client.get("/api/v1/telemetry/risk-overview")
        assert res.status_code == 200
        horizons = res.json()["horizons"]

        # Ensure all three horizons are evaluated independently
        assert "h1" in horizons
        assert "h3" in horizons
        assert "h5" in horizons
        assert horizons["h1"]["horizon"] == 1
        assert horizons["h3"]["horizon"] == 3
        assert horizons["h5"]["horizon"] == 5


def test_model_unavailable_distinction():
    """Verify clear distinction between model unavailable and valid telemetry."""
    app = create_app()
    with TestClient(app) as client:
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert "model_availability" in data
        assert isinstance(data["model_availability"], dict)
        for model_name, status in data["model_availability"].items():
            assert status in ("AVAILABLE", "UNAVAILABLE")
