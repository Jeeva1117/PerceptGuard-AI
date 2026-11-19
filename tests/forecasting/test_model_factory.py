"""Unit tests for the forecasting model factory and dynamic dimension resolution."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.forecasting.forecasting_config import ForecastingConfig, ModelConfig
from src.forecasting.forecasting_data_contract import ModelOutputSpec
from src.forecasting.lstm_forecaster import LSTMForecaster
from src.forecasting.model_factory import (
    ModelFactoryError,
    build_forecasting_model,
    resolve_input_dimension,
)


class TestModelFactory:
    """Tests for build_forecasting_model factory function and resolution logic."""

    @pytest.fixture
    def real_manifest_dir(self) -> Path:
        """Path to real Day 5 sequence directory with feature_manifest.json."""
        p = Path("datasets/processed/driving_sample/features/sequences/run_official")
        if not p.exists():
            pytest.skip("Real Day 5 sequence directory not found")
        return p

    def test_build_model_default(self, real_manifest_dir):
        """Factory must resolve input_size=27 from manifest and return LSTMForecaster."""
        model = build_forecasting_model(artifacts_dir=real_manifest_dir)
        assert isinstance(model, LSTMForecaster)
        assert model.input_size == 27
        assert model.hidden_size == 32
        assert model.num_layers == 1
        assert model.output_size == 3
        assert model.output_horizons == [1, 3, 5]

    def test_build_model_explicit_input_size(self):
        """Explicit input_size must override configuration and manifest resolution."""
        model = build_forecasting_model(input_size=15)
        assert model.input_size == 15

    def test_resolve_input_dimension_explicit(self):
        """Explicit input dimension returns directly without disk lookup."""
        cfg = ForecastingConfig()
        dim = resolve_input_dimension(cfg, explicit_input_size=42)
        assert dim == 42

    def test_resolve_input_dimension_invalid_explicit(self):
        """Negative or zero explicit input size raises ModelFactoryError."""
        cfg = ForecastingConfig()
        with pytest.raises(ModelFactoryError, match="must be a positive integer"):
            resolve_input_dimension(cfg, explicit_input_size=-5)

    def test_resolve_input_dimension_from_manifest(self, real_manifest_dir):
        """Resolves 27 features directly from Day 5 feature manifest."""
        cfg = ForecastingConfig()
        dim = resolve_input_dimension(cfg, artifacts_dir=real_manifest_dir)
        assert dim == 27

    def test_resolve_input_dimension_missing_manifest_error(self, tmp_path):
        """When manifest does not exist and input_size is auto, raises ModelFactoryError."""
        cfg = ForecastingConfig(
            model=ModelConfig(input_size="auto"),
            data=type(ForecastingConfig().data)(sequence_directory=str(tmp_path)),
        )
        with pytest.raises(ModelFactoryError, match="feature_manifest.json was not found"):
            resolve_input_dimension(cfg, artifacts_dir=tmp_path / "empty")

    def test_build_model_custom_configuration(self):
        """Factory applies custom model hyperparameters accurately."""
        cfg = ForecastingConfig(
            model=ModelConfig(
                architecture="lstm",
                input_size=10,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                output_horizons=[1, 3, 5],
            )
        )
        model = build_forecasting_model(config=cfg)
        assert model.input_size == 10
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.dropout_rate == 0.2

    def test_build_model_gru_unsupported(self):
        """Attempting to build GRU architecture in Task 3 raises ModelFactoryError."""
        cfg = ForecastingConfig(
            model=ModelConfig(architecture="gru", input_size=10)
        )
        with pytest.raises(ModelFactoryError, match="Architecture 'gru' is not supported in Day 6 Task 3"):
            build_forecasting_model(config=cfg)

    def test_build_model_unsupported_architecture(self):
        """Attempting to build unrecognized architecture raises ModelFactoryError."""
        # Bypass frozen dataclass validation to test factory defense
        cfg = ForecastingConfig(
            model=ModelConfig(input_size=10)
        )
        object.__setattr__(cfg.model, "architecture", "transformer")
        with pytest.raises(ModelFactoryError, match="Unsupported architecture 'transformer'"):
            build_forecasting_model(config=cfg)

    def test_build_model_target_contract_matching(self):
        """Matching target contract must be accepted by the factory."""
        contract = ModelOutputSpec(horizons=[1, 3, 5])
        model = build_forecasting_model(input_size=10, target_contract=contract)
        assert model.output_horizons == [1, 3, 5]

    def test_build_model_target_contract_mismatch(self):
        """Mismatched target contract horizons must be rejected."""
        contract = ModelOutputSpec(horizons=[1, 2, 4])
        with pytest.raises(ModelFactoryError, match="Target contract horizons"):
            build_forecasting_model(input_size=10, target_contract=contract)

    def test_build_model_device_cpu(self):
        """Model must be placed on CPU device."""
        model = build_forecasting_model(input_size=10, device="cpu")
        dev = next(model.parameters()).device
        assert dev.type == "cpu"
