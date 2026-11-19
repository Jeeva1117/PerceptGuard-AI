"""Advanced LSTM Forecaster Module for PerceptGuard AI.

Implements a production-grade, CPU-optimized recurrent neural network (LSTM)
for multi-horizon perception reliability forecasting (horizons t+1, t+3, t+5).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.forecasting.forecasting_config import ForecastingConfig


class ModelArchitectureError(ValueError):
    """Raised when model inputs, dimensions, or configuration fail validation."""
    pass


class LSTMForecaster(nn.Module):
    """Deep recurrent neural network for multi-horizon perception reliability forecasting.
    
    Processes historical temporal feature sequences of shape:
        (batch_size, sequence_length, feature_count)
    and predicts future reliability scores of shape:
        (batch_size, number_of_horizons)
    where output columns correspond strictly to configured horizons (e.g. [1, 3, 5]).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_horizons: Sequence[int] = (1, 3, 5),
        target_type: str = "regression",
        activation: str = "sigmoid",
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        # 1. Parameter Validation
        if not isinstance(input_size, int) or input_size <= 0:
            raise ModelArchitectureError(f"input_size must be a positive integer, got: {input_size}")
        if not isinstance(hidden_size, int) or hidden_size < 4:
            raise ModelArchitectureError(f"hidden_size must be an integer >= 4, got: {hidden_size}")
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ModelArchitectureError(f"num_layers must be an integer >= 1, got: {num_layers}")
        if not isinstance(dropout, (int, float)) or not (0.0 <= float(dropout) < 1.0):
            raise ModelArchitectureError(f"dropout must be in [0.0, 1.0), got: {dropout}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = float(dropout)
        self.output_horizons = list(output_horizons)
        if not self.output_horizons or any(not isinstance(h, int) or h <= 0 for h in self.output_horizons):
            raise ModelArchitectureError(f"output_horizons must be non-empty list of positive integers, got: {output_horizons}")

        self.output_size = len(self.output_horizons)
        self.target_type = str(target_type).lower().strip()
        if self.target_type not in ["regression", "classification"]:
            raise ModelArchitectureError(f"target_type must be 'regression' or 'classification', got: {target_type}")

        self.activation_name = str(activation).lower().strip()
        if self.activation_name not in ["sigmoid", "none", "identity"]:
            raise ModelArchitectureError(f"activation must be 'sigmoid', 'none', or 'identity', got: {activation}")

        self.batch_first = bool(batch_first)
        if not self.batch_first:
            raise ModelArchitectureError("Only batch_first=True is supported for production deployment")

        # Create horizon-to-column index mapping
        self.horizon_to_index: Dict[int, int] = {
            h: idx for idx, h in enumerate(self.output_horizons)
        }

        # 2. Recurrent Temporal Encoder (LSTM)
        # Note: PyTorch drops dropout if num_layers == 1 with a warning, so we enforce 0.0 for 1 layer
        effective_dropout = self.dropout_rate if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=effective_dropout,
            bidirectional=False,  # Strictly unidirectional (no future leakage)
        )

        # 3. Forecasting Head
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # 4. Activation Layer
        if self.activation_name == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()

        # 5. Parameter Initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier/Glorot initialization for weights and zero biases."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                # Forget gate bias initialization to 1.0 (improves gradient flow)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def get_column_for_horizon(self, horizon: int) -> int:
        """Return output column index for a given forecast horizon."""
        if horizon not in self.horizon_to_index:
            raise ModelArchitectureError(
                f"Horizon {horizon} not found in model output_horizons: {self.output_horizons}"
            )
        return self.horizon_to_index[horizon]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through LSTM temporal encoder and forecasting head.
        
        Args:
            x: Input feature sequence tensor of shape (batch_size, sequence_length, feature_count)
            
        Returns:
            Forecast tensor of shape (batch_size, output_size)
        """
        if not isinstance(x, torch.Tensor):
            raise ModelArchitectureError(f"Input must be a torch.Tensor, got: {type(x).__name__}")

        if x.ndim != 3:
            raise ModelArchitectureError(
                f"Input tensor must be 3-dimensional (batch_size, sequence_length, feature_count), "
                f"got ndim={x.ndim} with shape {list(x.shape)}"
            )

        batch_size, seq_len, feature_count = x.shape

        if batch_size == 0:
            raise ModelArchitectureError("Cannot perform forward pass on empty batch (batch_size=0)")

        if seq_len == 0:
            raise ModelArchitectureError("Sequence length must be > 0")

        if feature_count != self.input_size:
            raise ModelArchitectureError(
                f"Input feature count ({feature_count}) does not match model input_size ({self.input_size})"
            )

        if not torch.isfinite(x).all():
            raise ModelArchitectureError("Input tensor contains non-finite numbers (NaN or Inf)")

        # Ensure float32 dtype
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # 1. Temporal Recurrent Encoding
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)

        # 2. Extract Final Temporal Representation (Last historical observation step)
        last_representation = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # 3. Project Through Forecasting Head
        logits = self.fc(last_representation)  # (batch_size, output_size)

        # 4. Activation (e.g. Sigmoid for continuous score in [0.0, 1.0])
        predictions = self.act(logits)  # (batch_size, output_size)

        return predictions

    def get_model_summary(self) -> Dict[str, Any]:
        """Generate architectural summary and parameter memory audit."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + non_trainable

        # 4 bytes per float32 parameter
        mem_kb = round((total * 4) / 1024, 2)
        device = next(self.parameters()).device.type

        return {
            "architecture": "LSTMForecaster",
            "device": device,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "batch_first": self.batch_first,
            "output_horizons": self.output_horizons,
            "output_size": self.output_size,
            "horizon_mapping": self.horizon_to_index,
            "target_type": self.target_type,
            "activation": self.activation_name,
            "expected_input_shape": f"(batch_size, sequence_length, {self.input_size})",
            "expected_output_shape": f"(batch_size, {self.output_size})",
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
            "total_parameters": total,
            "estimated_param_memory_kb": mem_kb,
        }


def create_forecasting_loss(
    loss_name: str = "mse",
    reduction: str = "mean",
) -> nn.Module:
    """Factory creating compatible loss functions for continuous reliability forecasting.
    
    Args:
        loss_name: 'mse' for Mean Squared Error, 'mae' or 'l1' for Mean Absolute Error.
        reduction: 'mean', 'sum', or 'none'.
        
    Returns:
        Configured PyTorch loss module.
    """
    name = str(loss_name).lower().strip()
    if name == "mse":
        return nn.MSELoss(reduction=reduction)
    elif name in ["mae", "l1"]:
        return nn.L1Loss(reduction=reduction)
    elif name == "smooth_l1":
        return nn.SmoothL1Loss(reduction=reduction)
    else:
        raise ValueError(f"Unsupported forecasting loss '{loss_name}'. Supported: ['mse', 'mae', 'l1', 'smooth_l1']")


def validate_loss_inputs(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate prediction and target tensor shapes, types, and values for loss computation.
    
    Args:
        predictions: Model predictions (batch_size, output_size).
        targets: Ground-truth targets (batch_size, output_size).
        
    Returns:
        Tuple of validated float32 tensors.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError("Both predictions and targets must be torch.Tensor instances")

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch in loss computation: predictions {list(predictions.shape)} != targets {list(targets.shape)}"
        )

    if not torch.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite numbers (NaN or Inf)")

    if not torch.isfinite(targets).all():
        raise ValueError("Targets contain non-finite numbers (NaN or Inf)")

    return predictions.to(torch.float32), targets.to(torch.float32)


def benchmark_cpu_latency(
    model: LSTMForecaster,
    batch_size: int = 8,
    sequence_length: int = 5,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> Dict[str, Any]:
    """Benchmark model-only CPU forward-pass execution latency.
    
    Args:
        model: LSTMForecaster instance.
        batch_size: Number of sequences per batch.
        sequence_length: Sequence window length L.
        warmup_runs: Number of un-timed warm-up forward passes.
        benchmark_runs: Number of timed benchmark iterations.
        
    Returns:
        Dictionary containing latency statistics and throughput.
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(batch_size, sequence_length, model.input_size, dtype=torch.float32, device=device)

    # Warm-up phase
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies_arr))
    median_lat = float(np.median(latencies_arr))
    min_lat = float(np.min(latencies_arr))
    max_lat = float(np.max(latencies_arr))
    std_lat = float(np.std(latencies_arr))

    # Throughput (samples / sec)
    throughput = (batch_size / (mean_lat / 1000.0)) if mean_lat > 0 else 0.0

    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "feature_count": model.input_size,
        "device": device.type,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "mean_latency_ms": round(mean_lat, 4),
        "median_latency_ms": round(median_lat, 4),
        "min_latency_ms": round(min_lat, 4),
        "max_latency_ms": round(max_lat, 4),
        "std_latency_ms": round(std_lat, 4),
        "throughput_samples_per_sec": round(throughput, 2),
    }


def main() -> int:
    """CLI entrypoint for LSTM model inspection and CPU benchmarking."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI — LSTM Forecasting Architecture Inspector"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/forecasting_config.yaml",
        help="Path to forecasting configuration YAML",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Display model architecture summary and parameter analysis",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run CPU forward-pass latency benchmark",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=27,
        help="Feature count (default: 27 from Day 5 feature manifest)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="LSTM hidden dimension (default: 32)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of LSTM layers (default: 1)",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Forecast horizons (default: 1 3 5)",
    )

    args = parser.parse_args()

    model = LSTMForecaster(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_horizons=args.horizons,
    )

    summary = model.get_model_summary()

    print("=" * 70)
    print(" PerceptGuard AI — Advanced LSTM Forecasting Architecture")
    print("=" * 70)
    print(f"Architecture Name          : {summary['architecture']}")
    print(f"Device                     : {summary['device']}")
    print(f"Input Feature Dimension (F): {summary['input_size']}")
    print(f"LSTM Hidden Dimension (H)  : {summary['hidden_size']}")
    print(f"LSTM Layer Count           : {summary['num_layers']}")
    print(f"Dropout Rate               : {summary['dropout']}")
    print(f"Batch First Input          : {summary['batch_first']}")
    print(f"Target Forecast Horizons   : {summary['output_horizons']}")
    print(f"Output Horizon Dimension   : {summary['output_size']}")
    print(f"Horizon-to-Column Mapping  : {summary['horizon_mapping']}")
    print(f"Target Type & Activation   : {summary['target_type']} ({summary['activation']})")
    print("-" * 70)
    print(f"Expected Input Shape       : {summary['expected_input_shape']}")
    print(f"Expected Output Shape      : {summary['expected_output_shape']}")
    print(f"Trainable Parameters       : {summary['trainable_parameters']:,}")
    print(f"Non-Trainable Parameters   : {summary['non_trainable_parameters']}")
    print(f"Total Parameters           : {summary['total_parameters']:,}")
    print(f"Parameter Memory (Est.)    : {summary['estimated_param_memory_kb']} KB")
    print("=" * 70)

    if args.benchmark:
        print("\n--- RUNNING CPU LATENCY BENCHMARK ---")
        bench = benchmark_cpu_latency(model, batch_size=8, sequence_length=5)
        print(f"Benchmark Iterations       : {bench['benchmark_runs']} (Warm-up: {bench['warmup_runs']})")
        print(f"Batch Size / Seq Length    : {bench['batch_size']} / {bench['sequence_length']}")
        print(f"Mean Forward Latency       : {bench['mean_latency_ms']} ms")
        print(f"Median Forward Latency     : {bench['median_latency_ms']} ms")
        print(f"Min / Max Latency          : {bench['min_latency_ms']} ms / {bench['max_latency_ms']} ms")
        print(f"CPU Forward Throughput     : {bench['throughput_samples_per_sec']} samples / sec")
        print("-" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
