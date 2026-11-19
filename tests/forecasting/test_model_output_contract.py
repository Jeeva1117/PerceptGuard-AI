"""Tests for model output contracts, multi-horizon mappings, and real Day 5 tensor compatibility."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from src.forecasting.forecasting_data_contract import ModelInputSpec, ModelOutputSpec
from src.forecasting.lstm_forecaster import (
    LSTMForecaster,
    ModelArchitectureError,
    create_forecasting_loss,
    validate_loss_inputs,
)
from src.forecasting.model_factory import build_forecasting_model


class TestModelOutputContract:
    """Tests for multi-horizon contract enforcement and real tensor compatibility."""

    @pytest.fixture
    def real_day5_dir(self) -> Path:
        """Path to real Day 5 sequence directory."""
        p = Path("datasets/processed/driving_sample/features/sequences/run_official")
        if not p.exists():
            pytest.skip("Real Day 5 sequence directory not found")
        return p

    def test_multi_horizon_column_mapping(self):
        """Model must strictly preserve column-to-horizon ordering: 0->h1, 1->h3, 2->h5."""
        model = LSTMForecaster(input_size=10, output_horizons=[1, 3, 5])
        assert model.get_column_for_horizon(1) == 0
        assert model.get_column_for_horizon(3) == 1
        assert model.get_column_for_horizon(5) == 2

    def test_get_column_for_invalid_horizon_raises(self):
        """Querying column index for non-existent horizon raises ModelArchitectureError."""
        model = LSTMForecaster(input_size=10, output_horizons=[1, 3, 5])
        with pytest.raises(ModelArchitectureError, match="not found in model output_horizons"):
            model.get_column_for_horizon(2)

        with pytest.raises(ModelArchitectureError, match="not found in model output_horizons"):
            model.get_column_for_horizon(10)

    def test_output_shape_matches_horizons_length(self):
        """Output dimension must dynamically match length of configured horizons."""
        model = LSTMForecaster(input_size=10, output_horizons=[1, 2, 3, 5])
        assert model.output_size == 4
        x = torch.randn(4, 5, 10)
        out = model(x)
        assert out.shape == (4, 4)

    def test_real_day5_tensor_l5_compatibility(self, real_day5_dir):
        """Real Day 5 sequences_L5.npy tensor must pass forward execution cleanly."""
        tensor_file = real_day5_dir / "sequences_L5.npy"
        arr = np.load(tensor_file)
        assert arr.shape == (138, 5, 27)

        model = build_forecasting_model(artifacts_dir=real_day5_dir)
        tensor_t = torch.from_numpy(arr).float()

        # Test single sample batch
        out1 = model(tensor_t[:1])
        assert out1.shape == (1, 3)
        assert torch.isfinite(out1).all()

        # Test batch size 8
        out8 = model(tensor_t[:8])
        assert out8.shape == (8, 3)
        assert torch.isfinite(out8).all()

        # Test complete dataset in one forward pass
        out_all = model(tensor_t)
        assert out_all.shape == (138, 3)
        assert torch.isfinite(out_all).all()

    def test_real_day5_tensor_l10_compatibility(self, real_day5_dir):
        """Real Day 5 sequences_L10.npy tensor must pass forward execution cleanly."""
        arr = np.load(real_day5_dir / "sequences_L10.npy")
        assert arr.shape == (123, 10, 27)

        model = build_forecasting_model(artifacts_dir=real_day5_dir)
        tensor_t = torch.from_numpy(arr).float()
        out = model(tensor_t)
        assert out.shape == (123, 3)
        assert torch.isfinite(out).all()

    def test_real_day5_tensor_l15_compatibility(self, real_day5_dir):
        """Real Day 5 sequences_L15.npy tensor must pass forward execution cleanly."""
        arr = np.load(real_day5_dir / "sequences_L15.npy")
        assert arr.shape == (108, 15, 27)

        model = build_forecasting_model(artifacts_dir=real_day5_dir)
        tensor_t = torch.from_numpy(arr).float()
        out = model(tensor_t)
        assert out.shape == (108, 3)
        assert torch.isfinite(out).all()

    def test_real_day5_tensor_l20_compatibility(self, real_day5_dir):
        """Real Day 5 sequences_L20.npy tensor must pass forward execution cleanly."""
        arr = np.load(real_day5_dir / "sequences_L20.npy")
        assert arr.shape == (93, 20, 27)

        model = build_forecasting_model(artifacts_dir=real_day5_dir)
        tensor_t = torch.from_numpy(arr).float()
        out = model(tensor_t)
        assert out.shape == (93, 3)
        assert torch.isfinite(out).all()

    def test_synthetic_optimization_step(self):
        """Simulate a single synthetic optimization step to verify loss and backprop."""
        model = LSTMForecaster(input_size=10, hidden_size=16, output_horizons=[1, 3, 5])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = create_forecasting_loss("mse")

        x = torch.randn(8, 5, 10)
        target = torch.rand(8, 3)

        # Record initial weights of fc layer
        initial_weight = model.fc.weight.clone()

        model.train()
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, target)
        loss.backward()
        optimizer.step()

        # Weights must be updated
        assert not torch.equal(model.fc.weight, initial_weight)
        assert loss.item() > 0.0

    def test_target_readiness_gate_real_dataset(self, real_day5_dir):
        """Real dataset forward pass executes without requiring fabricated real labels."""
        model = build_forecasting_model(artifacts_dir=real_day5_dir)
        arr = np.load(real_day5_dir / "sequences_L5.npy")
        x = torch.from_numpy(arr[:4]).float()

        model.eval()
        with torch.no_grad():
            preds = model(x)

        assert preds.shape == (4, 3)
        # Predictions are generated by the model architecture, but no false loss is calculated
        # against non-existent real-world ground truth labels
