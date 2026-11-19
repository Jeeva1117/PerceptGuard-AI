"""Unit and integration tests for LSTMForecaster architecture, forward pass, and gradients."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.forecasting.lstm_forecaster import (
    LSTMForecaster,
    ModelArchitectureError,
    benchmark_cpu_latency,
    create_forecasting_loss,
    validate_loss_inputs,
)


class TestLSTMForecaster:
    """Tests for the LSTM recurrent forecasting architecture."""

    def test_init_defaults(self):
        """Default initialization must configure LSTM layers, output size, and mappings."""
        model = LSTMForecaster(input_size=27)
        assert model.input_size == 27
        assert model.hidden_size == 32
        assert model.num_layers == 1
        assert model.dropout_rate == 0.0
        assert model.output_horizons == [1, 3, 5]
        assert model.output_size == 3
        assert model.horizon_to_index == {1: 0, 3: 1, 5: 2}
        assert model.target_type == "regression"
        assert model.activation_name == "sigmoid"
        assert model.batch_first is True

    def test_init_invalid_parameters(self):
        """Model must reject invalid input size, hidden size, layers, and dropout."""
        with pytest.raises(ModelArchitectureError, match="input_size must be a positive integer"):
            LSTMForecaster(input_size=0)

        with pytest.raises(ModelArchitectureError, match="hidden_size must be an integer >= 4"):
            LSTMForecaster(input_size=10, hidden_size=2)

        with pytest.raises(ModelArchitectureError, match="num_layers must be an integer >= 1"):
            LSTMForecaster(input_size=10, num_layers=0)

        with pytest.raises(ModelArchitectureError, match="dropout must be in"):
            LSTMForecaster(input_size=10, dropout=-0.1)

        with pytest.raises(ModelArchitectureError, match="dropout must be in"):
            LSTMForecaster(input_size=10, dropout=1.0)

        with pytest.raises(ModelArchitectureError, match="output_horizons must be non-empty"):
            LSTMForecaster(input_size=10, output_horizons=[])

        with pytest.raises(ModelArchitectureError, match="target_type must be"):
            LSTMForecaster(input_size=10, target_type="clustering")

        with pytest.raises(ModelArchitectureError, match="activation must be"):
            LSTMForecaster(input_size=10, activation="softmax")

    def test_forward_single_sample_batch(self):
        """Single-sample batch (1, L, F) must produce output of shape (1, 3)."""
        model = LSTMForecaster(input_size=10, hidden_size=16, output_horizons=[1, 3, 5])
        x = torch.randn(1, 5, 10)
        out = model(x)
        assert out.shape == (1, 3)
        assert torch.isfinite(out).all()

    def test_forward_multi_sample_batch(self):
        """Multi-sample batch (batch_size=8) must produce output of shape (8, 3)."""
        model = LSTMForecaster(input_size=27, hidden_size=32, output_horizons=[1, 3, 5])
        x = torch.randn(8, 10, 27)
        out = model(x)
        assert out.shape == (8, 3)
        assert torch.isfinite(out).all()

    def test_forward_different_sequence_lengths(self):
        """Model must support variable sequence lengths L in {5, 10, 15, 20}."""
        model = LSTMForecaster(input_size=12, hidden_size=16)
        for seq_len in [5, 10, 15, 20]:
            x = torch.randn(4, seq_len, 12)
            out = model(x)
            assert out.shape == (4, 3)

    def test_forward_output_range_sigmoid(self):
        """Sigmoid activation must bound all predictions strictly within [0.0, 1.0]."""
        model = LSTMForecaster(input_size=8, activation="sigmoid")
        x = torch.randn(16, 5, 8) * 10.0  # Large inputs to test saturation
        out = model(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_forward_identity_activation(self):
        """Identity activation allows unbounded linear output."""
        model = LSTMForecaster(input_size=8, activation="none")
        x = torch.randn(4, 5, 8)
        out = model(x)
        assert out.shape == (4, 3)

    def test_forward_invalid_ndim(self):
        """Non-3D tensors must be rejected."""
        model = LSTMForecaster(input_size=10)
        with pytest.raises(ModelArchitectureError, match="must be 3-dimensional"):
            model(torch.randn(10, 5))

        with pytest.raises(ModelArchitectureError, match="must be 3-dimensional"):
            model(torch.randn(2, 5, 10, 2))

    def test_forward_empty_batch(self):
        """Batch size 0 must be rejected."""
        model = LSTMForecaster(input_size=10)
        with pytest.raises(ModelArchitectureError, match="batch_size=0"):
            model(torch.empty(0, 5, 10))

    def test_forward_feature_count_mismatch(self):
        """Feature dimension mismatch must be rejected with helpful error."""
        model = LSTMForecaster(input_size=27)
        with pytest.raises(ModelArchitectureError, match="Input feature count"):
            model(torch.randn(4, 5, 15))

    def test_forward_nan_detection(self):
        """Tensors containing NaN must raise ModelArchitectureError."""
        model = LSTMForecaster(input_size=10)
        x = torch.randn(4, 5, 10)
        x[0, 2, 3] = float("nan")
        with pytest.raises(ModelArchitectureError, match="contains non-finite"):
            model(x)

    def test_forward_inf_detection(self):
        """Tensors containing Inf must raise ModelArchitectureError."""
        model = LSTMForecaster(input_size=10)
        x = torch.randn(4, 5, 10)
        x[1, 0, 0] = float("inf")
        with pytest.raises(ModelArchitectureError, match="contains non-finite"):
            model(x)

    def test_gradient_propagation(self):
        """Backpropagation must compute non-zero gradients on all model parameters."""
        model = LSTMForecaster(input_size=10, hidden_size=16, output_horizons=[1, 3, 5])
        x = torch.randn(4, 5, 10, requires_grad=True)
        target = torch.rand(4, 3)  # Targets in [0, 1]

        preds = model(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(preds, target)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
            assert torch.count_nonzero(param.grad) > 0, f"Parameter {name} has all-zero gradient"

    def test_deterministic_initialization(self):
        """Fixed seed must produce identical parameter initializations."""
        torch.manual_seed(42)
        m1 = LSTMForecaster(input_size=8, hidden_size=16)

        torch.manual_seed(42)
        m2 = LSTMForecaster(input_size=8, hidden_size=16)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_model_summary(self):
        """Model summary must report parameter counts and memory estimates accurately."""
        model = LSTMForecaster(input_size=27, hidden_size=32, num_layers=1, output_horizons=[1, 3, 5])
        summary = model.get_model_summary()

        assert summary["architecture"] == "LSTMForecaster"
        assert summary["input_size"] == 27
        assert summary["hidden_size"] == 32
        assert summary["num_layers"] == 1
        assert summary["output_size"] == 3
        # LSTM params for 1 layer: 4 * ((27 + 32 + 1) * 32) = 4 * (60 * 32) = 7680
        # Linear head params: (32 * 3) + 3 = 96 + 3 = 99
        # Total = 7680 + 99 = 7779
        # Wait, let's check bias in LSTM: weight_ih has shape (128, 27), weight_hh (128, 32), bias_ih (128), bias_hh (128)
        # Total = 128*27 + 128*32 + 128 + 128 = 3456 + 4096 + 256 = 7808 + 99 = 7907
        assert summary["total_parameters"] == 7907
        assert summary["trainable_parameters"] == 7907
        assert summary["non_trainable_parameters"] == 0
        assert summary["estimated_param_memory_kb"] == pytest.approx(30.89, abs=0.1)

    def test_forward_batch_sizes_scaling(self):
        """Model must smoothly handle varying batch sizes: 2, 16, 32."""
        model = LSTMForecaster(input_size=10, hidden_size=16)
        for bs in [2, 16, 32]:
            x = torch.randn(bs, 5, 10)
            out = model(x)
            assert out.shape == (bs, 3)
            assert torch.isfinite(out).all()

    def test_loss_interface(self):
        """Loss factory and input validation must function properly."""
        mse = create_forecasting_loss("mse")
        l1 = create_forecasting_loss("mae")
        smooth_l1 = create_forecasting_loss("smooth_l1")
        assert isinstance(mse, nn.MSELoss)
        assert isinstance(l1, nn.L1Loss)
        assert isinstance(smooth_l1, nn.SmoothL1Loss)

        p = torch.tensor([[0.8, 0.7, 0.6]])
        t = torch.tensor([[0.9, 0.7, 0.4]])
        p_v, t_v = validate_loss_inputs(p, t)
        loss = mse(p_v, t_v)
        assert loss.item() > 0.0

        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_loss_inputs(p, torch.tensor([[0.9, 0.7]]))

        with pytest.raises(ValueError, match="Unsupported forecasting loss"):
            create_forecasting_loss("unsupported_loss")

    def test_cpu_latency_benchmark(self):
        """Latency benchmark must record positive timings and throughput."""
        model = LSTMForecaster(input_size=10, hidden_size=16)
        bench = benchmark_cpu_latency(model, batch_size=4, sequence_length=5, warmup_runs=2, benchmark_runs=10)

        assert bench["batch_size"] == 4
        assert bench["sequence_length"] == 5
        assert bench["mean_latency_ms"] > 0.0
        assert bench["median_latency_ms"] > 0.0
        assert bench["throughput_samples_per_sec"] > 0.0

    def test_cli_execution_summary(self):
        """CLI invocation with --summary and --benchmark must execute with return code 0."""
        cmd = [sys.executable, "-m", "src.forecasting.lstm_forecaster", "--summary", "--benchmark"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Advanced LSTM Forecasting Architecture" in res.stdout
        assert "Trainable Parameters       : 7,907" in res.stdout
        assert "CPU Forward Throughput" in res.stdout
