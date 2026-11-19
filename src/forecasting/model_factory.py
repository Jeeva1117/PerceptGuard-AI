"""Model Factory Module for PerceptGuard AI Forecasting.

Constructs, configures, and validates deep learning forecasting models
(LSTMForecaster) using configuration objects, dynamic feature manifests,
and target contract specifications.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from src.forecasting.forecasting_config import ForecastingConfig
from src.forecasting.forecasting_data_contract import ModelInputSpec, ModelOutputSpec
from src.forecasting.lstm_forecaster import LSTMForecaster, ModelArchitectureError


class ModelFactoryError(ValueError):
    """Raised when model factory construction or contract resolution fails."""
    pass


def resolve_input_dimension(
    config: ForecastingConfig,
    explicit_input_size: Optional[int] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> int:
    """Resolve input feature dimension from arguments, config, or feature manifest.
    
    Args:
        config: ForecastingConfig instance.
        explicit_input_size: Optional explicit feature dimension override.
        artifacts_dir: Optional path to Day 5 sequence directory with feature_manifest.json.
        
    Returns:
        Resolved integer feature count.
    """
    if explicit_input_size is not None:
        if isinstance(explicit_input_size, int) and explicit_input_size > 0:
            return explicit_input_size
        raise ModelFactoryError(f"explicit_input_size must be a positive integer, got: {explicit_input_size}")

    cfg_input = config.model.input_size
    if isinstance(cfg_input, int) and cfg_input > 0:
        return cfg_input

    # If 'auto', inspect feature manifest
    if artifacts_dir is not None:
        candidate_dirs = [Path(artifacts_dir)]
    else:
        candidate_dirs = [
            Path(config.data.sequence_directory) / "driving_sample" / "features" / "sequences" / "run_official",
            Path("datasets/processed/driving_sample/features/sequences/run_official"),
        ]

    for cand in candidate_dirs:
        manifest_file = cand / "feature_manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                feat_count = data.get("total_features") or len(data.get("ordered_feature_names", []))
                if feat_count > 0:
                    return int(feat_count)
            except Exception:
                continue

    raise ModelFactoryError(
        "Cannot resolve model input_size 'auto': feature_manifest.json was not found "
        "and no explicit input_size was provided"
    )


def build_forecasting_model(
    config: Optional[ForecastingConfig] = None,
    input_size: Optional[int] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    target_contract: Optional[Union[ModelOutputSpec, Any]] = None,
    device: Optional[str] = None,
) -> LSTMForecaster:
    """Factory creating validated forecasting models configured for CPU execution.
    
    Args:
        config: Optional ForecastingConfig. If None, loads from config/forecasting_config.yaml.
        input_size: Optional explicit input feature dimension override.
        artifacts_dir: Optional directory to search for feature_manifest.json when input_size='auto'.
        target_contract: Optional ModelOutputSpec for output contract validation.
        device: Device to place model on (mandated 'cpu' for Day 6 Task 3).
        
    Returns:
        Configured and initialized LSTMForecaster module.
    """
    if config is None:
        cfg_path = Path("config/forecasting_config.yaml")
        if cfg_path.exists():
            config = ForecastingConfig.from_yaml(cfg_path, validate=True)
        else:
            config = ForecastingConfig()

    # 1. Architecture Validation
    arch = str(config.model.architecture).lower().strip()
    if arch != "lstm":
        if arch == "gru":
            raise ModelFactoryError(
                "Architecture 'gru' is not supported in Day 6 Task 3. "
                "Only 'lstm' is implemented for this task."
            )
        raise ModelFactoryError(
            f"Unsupported architecture '{config.model.architecture}'. Only 'lstm' is supported."
        )

    # 2. Dynamic Input Size Resolution
    resolved_input_size = resolve_input_dimension(
        config=config,
        explicit_input_size=input_size,
        artifacts_dir=artifacts_dir,
    )

    # 3. Output Horizons & Contract Validation
    horizons = list(config.model.output_horizons)
    if target_contract is not None:
        contract_horizons = getattr(target_contract, "horizons", None)
        if contract_horizons is not None and list(contract_horizons) != horizons:
            raise ModelFactoryError(
                f"Target contract horizons {contract_horizons} do not match model horizons {horizons}"
            )

    # 4. Device Configuration
    target_device_str = device or config.runtime.device or "cpu"
    if str(target_device_str).lower() == "cuda":
        if not torch.cuda.is_available():
            raise ModelFactoryError("Device 'cuda' requested, but CUDA is not available on this system")
    target_device = torch.device(target_device_str)

    # 5. Model Instantiation
    model = LSTMForecaster(
        input_size=resolved_input_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        output_horizons=horizons,
        target_type=config.model.target_type,
        activation=config.model.activation,
        batch_first=config.model.batch_first,
    )

    model.to(target_device)
    return model
