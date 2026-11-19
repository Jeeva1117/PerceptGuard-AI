"""Forecasting Configuration Module for PerceptGuard AI.

Defines strongly-typed, CPU-first configuration schemas and validation
for deep learning temporal forecasting (LSTM/GRU).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


class ForecastingConfigError(ValueError):
    """Raised when forecasting configuration validation fails."""
    pass


@dataclass(frozen=True)
class ProjectConfig:
    """Project-level metadata configuration."""
    name: str = "PerceptGuard AI"
    stage: str = "forecasting"
    random_seed: int = 42

    def validate(self) -> List[str]:
        errors = []
        if not self.name or not isinstance(self.name, str):
            errors.append("project.name must be a non-empty string")
        if not self.stage or not isinstance(self.stage, str):
            errors.append("project.stage must be a non-empty string")
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            errors.append("project.random_seed must be a non-negative integer")
        return errors


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime execution parameters tailored for CPU-first execution."""
    device: str = "cpu"
    num_threads: int = 2
    deterministic: bool = True

    def validate(self) -> List[str]:
        errors = []
        valid_devices = ["cpu"]
        # Allow checking cuda request
        dev_str = str(self.device).lower().strip()
        if dev_str not in ["cpu", "cuda"]:
            errors.append(f"runtime.device must be 'cpu' or 'cuda', got: {self.device}")
        elif dev_str == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append("runtime.device is set to 'cuda', but CUDA is not available on this system")
            except ImportError:
                errors.append("runtime.device is set to 'cuda', but PyTorch is not installed")
        if not isinstance(self.num_threads, int) or self.num_threads < 1:
            errors.append("runtime.num_threads must be an integer >= 1")
        if not isinstance(self.deterministic, bool):
            errors.append("runtime.deterministic must be a boolean")
        return errors


@dataclass(frozen=True)
class DataConfig:
    """Dataset and sequence ingestion configuration."""
    sequence_directory: str = "datasets/processed"
    sequence_lengths: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 3, 5])
    batch_size: int = 8
    num_workers: int = 0

    def validate(self) -> List[str]:
        errors = []
        if not self.sequence_directory or not isinstance(self.sequence_directory, str):
            errors.append("data.sequence_directory must be a non-empty string")
        if not isinstance(self.sequence_lengths, list) or len(self.sequence_lengths) == 0:
            errors.append("data.sequence_lengths must be a non-empty list of positive integers")
        else:
            for l in self.sequence_lengths:
                if not isinstance(l, int) or l <= 0:
                    errors.append(f"data.sequence_lengths entries must be positive integers, got: {l}")
        if not isinstance(self.forecast_horizons, list) or len(self.forecast_horizons) == 0:
            errors.append("data.forecast_horizons must be a non-empty list of positive integers")
        else:
            for h in self.forecast_horizons:
                if not isinstance(h, int) or h <= 0:
                    errors.append(f"data.forecast_horizons entries must be positive integers, got: {h}")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            errors.append("data.batch_size must be an integer >= 1")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            errors.append("data.num_workers must be an integer >= 0")
        return errors


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = "lstm"
    input_size: Union[str, int] = "auto"
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    output_horizons: List[int] = field(default_factory=lambda: [1, 3, 5])
    output_size: Optional[int] = None
    target_type: str = "regression"
    batch_first: bool = True
    activation: str = "sigmoid"

    def validate(self, data_horizons: Optional[List[int]] = None) -> List[str]:
        errors = []
        arch = str(self.architecture).lower().strip()
        if arch not in ["lstm", "gru"]:
            errors.append(f"model.architecture must be 'lstm' or 'gru', got: {self.architecture}")
        if self.input_size != "auto":
            if not isinstance(self.input_size, int) or self.input_size <= 0:
                errors.append(f"model.input_size must be 'auto' or a positive integer, got: {self.input_size}")
        if not isinstance(self.hidden_size, int) or self.hidden_size < 4:
            errors.append(f"model.hidden_size must be an integer >= 4, got: {self.hidden_size}")
        if not isinstance(self.num_layers, int) or self.num_layers < 1:
            errors.append(f"model.num_layers must be an integer >= 1, got: {self.num_layers}")
        if not isinstance(self.dropout, (int, float)) or not (0.0 <= float(self.dropout) < 1.0):
            errors.append(f"model.dropout must be in [0.0, 1.0), got: {self.dropout}")
        if not isinstance(self.output_horizons, list) or len(self.output_horizons) == 0:
            errors.append("model.output_horizons must be a non-empty list of positive integers")
        else:
            for h in self.output_horizons:
                if not isinstance(h, int) or h <= 0:
                    errors.append(f"model.output_horizons entries must be positive integers, got: {h}")
            if data_horizons is not None:
                for h in self.output_horizons:
                    if h not in data_horizons:
                        errors.append(f"model.output_horizons entry {h} not found in data.forecast_horizons {data_horizons}")

        if self.output_size is not None:
            if not isinstance(self.output_size, int) or self.output_size <= 0:
                errors.append(f"model.output_size must be a positive integer, got: {self.output_size}")
            elif self.output_size != len(self.output_horizons):
                errors.append(
                    f"model.output_size ({self.output_size}) does not match len(model.output_horizons) ({len(self.output_horizons)})"
                )

        target = str(self.target_type).lower().strip()
        if target not in ["regression", "classification"]:
            errors.append(f"model.target_type must be 'regression' or 'classification', got: {self.target_type}")

        if not isinstance(self.batch_first, bool):
            errors.append("model.batch_first must be a boolean")

        act = str(self.activation).lower().strip()
        if act not in ["sigmoid", "none", "identity"]:
            errors.append(f"model.activation must be 'sigmoid', 'none', or 'identity', got: {self.activation}")

        return errors


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameter and optimization configuration."""
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 50
    early_stopping_patience: int = 8

    def validate(self) -> List[str]:
        errors = []
        if not isinstance(self.learning_rate, (int, float)) or float(self.learning_rate) <= 0.0:
            errors.append(f"training.learning_rate must be > 0.0, got: {self.learning_rate}")
        if not isinstance(self.weight_decay, (int, float)) or float(self.weight_decay) < 0.0:
            errors.append(f"training.weight_decay must be >= 0.0, got: {self.weight_decay}")
        if not isinstance(self.max_epochs, int) or self.max_epochs < 1:
            errors.append(f"training.max_epochs must be an integer >= 1, got: {self.max_epochs}")
        if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience < 1:
            errors.append(f"training.early_stopping_patience must be an integer >= 1, got: {self.early_stopping_patience}")
        return errors


@dataclass(frozen=True)
class EvaluationConfig:
    """Forecasting evaluation metrics configuration."""
    metrics: List[str] = field(default_factory=lambda: ["mae", "rmse"])

    def validate(self) -> List[str]:
        errors = []
        valid_metrics = {"mae", "rmse", "mse", "mape", "r2"}
        if not isinstance(self.metrics, list) or len(self.metrics) == 0:
            errors.append("evaluation.metrics must be a non-empty list of strings")
        else:
            for m in self.metrics:
                if not isinstance(m, str) or m.lower().strip() not in valid_metrics:
                    errors.append(f"evaluation.metrics entry '{m}' is not a recognized metric (valid: {sorted(valid_metrics)})")
        return errors


@dataclass(frozen=True)
class ArtifactsConfig:
    """Artifact and checkpoint persistence directories."""
    checkpoint_directory: str = "models/forecasting"
    experiment_directory: str = "experiments/forecasting"

    def validate(self) -> List[str]:
        errors = []
        if not self.checkpoint_directory or not isinstance(self.checkpoint_directory, str):
            errors.append("artifacts.checkpoint_directory must be a non-empty string")
        if not self.experiment_directory or not isinstance(self.experiment_directory, str):
            errors.append("artifacts.experiment_directory must be a non-empty string")
        return errors


@dataclass(frozen=True)
class ForecastingConfig:
    """Top-level immutable forecasting configuration container."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)

    def validate(self, raise_on_error: bool = False) -> Tuple[bool, List[str]]:
        """Validate all configuration sections.
        
        Args:
            raise_on_error: If True, raises ForecastingConfigError when invalid.
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors: List[str] = []
        errors.extend(self.project.validate())
        errors.extend(self.runtime.validate())
        errors.extend(self.data.validate())
        errors.extend(self.model.validate(data_horizons=self.data.forecast_horizons))
        errors.extend(self.training.validate())
        errors.extend(self.evaluation.validate())
        errors.extend(self.artifacts.validate())

        is_valid = len(errors) == 0
        if not is_valid and raise_on_error:
            raise ForecastingConfigError(
                f"Forecasting configuration validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {err}" for err in errors)
            )
        return is_valid, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration hierarchy to dictionary."""
        return asdict(self)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], validate: bool = True) -> ForecastingConfig:
        """Construct ForecastingConfig from a dictionary."""
        if not isinstance(data, dict):
            raise ForecastingConfigError(f"Expected dict for configuration data, got: {type(data).__name__}")

        project_data = data.get("project", {})
        runtime_data = data.get("runtime", {})
        data_data = data.get("data", {})
        model_data = data.get("model", {})
        training_data = data.get("training", {})
        evaluation_data = data.get("evaluation", {})
        artifacts_data = data.get("artifacts", {})

        config = cls(
            project=ProjectConfig(**project_data),
            runtime=RuntimeConfig(**runtime_data),
            data=DataConfig(**data_data),
            model=ModelConfig(**model_data),
            training=TrainingConfig(**training_data),
            evaluation=EvaluationConfig(**evaluation_data),
            artifacts=ArtifactsConfig(**artifacts_data),
        )

        if validate:
            config.validate(raise_on_error=True)

        return config

    @classmethod
    def from_yaml(cls, path: Union[str, Path], validate: bool = True) -> ForecastingConfig:
        """Load and parse configuration from a YAML file."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        if not yaml_path.is_file():
            raise ValueError(f"Configuration path is not a file: {yaml_path}")

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except Exception as e:
            raise ForecastingConfigError(f"Failed to parse YAML file {yaml_path}: {e}") from e

        if content is None:
            content = {}
        elif not isinstance(content, dict):
            raise ForecastingConfigError(f"YAML configuration root must be a mapping, got {type(content).__name__}")

        return cls.from_dict(content, validate=validate)


def main() -> int:
    """CLI entrypoint for forecasting configuration validation."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI Forecasting Configuration Validator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/forecasting_config.yaml",
        help="Path to forecasting configuration YAML file",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Perform strict configuration validation",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    print("=" * 65)
    print(" PerceptGuard AI — Forecasting Configuration Validator")
    print("=" * 65)
    print(f"Target Configuration Path : {config_path.resolve()}")

    if not config_path.exists():
        print(f"[FAIL] Configuration file does not exist: {config_path}")
        return 1

    try:
        config = ForecastingConfig.from_yaml(config_path, validate=False)
        is_valid, errors = config.validate(raise_on_error=False)

        print(f"Project Name              : {config.project.name}")
        print(f"Stage                     : {config.project.stage}")
        print(f"Random Seed               : {config.project.random_seed}")
        print(f"Device                    : {config.runtime.device}")
        print(f"Threads                   : {config.runtime.num_threads}")
        print(f"Sequence Lengths          : {config.data.sequence_lengths}")
        print(f"Forecast Horizons         : {config.data.forecast_horizons}")
        print(f"Batch Size                : {config.data.batch_size}")
        print(f"Model Architecture        : {config.model.architecture}")
        print(f"Hidden Size / Layers      : {config.model.hidden_size} / {config.model.num_layers}")
        print(f"Output Horizons           : {config.model.output_horizons}")
        print(f"Learning Rate / Epochs    : {config.training.learning_rate} / {config.training.max_epochs}")
        print(f"Evaluation Metrics        : {config.evaluation.metrics}")
        print(f"Checkpoint Directory      : {config.artifacts.checkpoint_directory}")
        print(f"Experiment Directory      : {config.artifacts.experiment_directory}")
        print("-" * 65)

        if is_valid:
            print("[PASS] Configuration is VALID and strictly adheres to schema.")
            print("=" * 65)
            return 0
        else:
            print(f"[FAIL] Configuration validation FAILED with {len(errors)} error(s):")
            for err in errors:
                print(f"  * {err}")
            print("=" * 65)
            return 1

    except Exception as exc:
        print(f"[ERROR] Exception during configuration loading: {exc}")
        print("=" * 65)
        return 2


if __name__ == "__main__":
    sys.exit(main())
