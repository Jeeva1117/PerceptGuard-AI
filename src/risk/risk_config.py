"""Configuration module for PerceptGuard AI Reliability Risk Scoring Engine.

Defines strongly-typed, CPU-first configuration schemas and validation
for transforming multi-horizon reliability forecasts into quantified risk scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import yaml


class RiskConfigError(ValueError):
    """Raised when risk configuration validation fails."""
    pass


@dataclass(frozen=True)
class TransformationConfig:
    """Configuration for mapping predicted reliability into a derived risk score."""
    method: str = "linear_inverse"  # risk = 1.0 - reliability
    clip_output: bool = True
    min_risk: float = 0.0
    max_risk: float = 1.0

    def validate(self) -> List[str]:
        errors = []
        if self.method not in ["linear_inverse"]:
            errors.append(f"transformation.method must be 'linear_inverse', got: {self.method}")
        if self.min_risk < 0.0 or self.max_risk > 1.0 or self.min_risk >= self.max_risk:
            errors.append(
                f"transformation bounds must satisfy 0.0 <= min_risk < max_risk <= 1.0, "
                f"got: min={self.min_risk}, max={self.max_risk}"
            )
        return errors


@dataclass(frozen=True)
class AggregationConfig:
    """Configuration for combining multi-horizon risk scores."""
    strategy: str = "weighted_linear"  # "weighted_linear", "worst_case", "primary_horizon", "none"
    primary_horizon: int = 3
    horizon_weights: Dict[int, float] = field(
        default_factory=lambda: {1: 0.50, 3: 0.35, 5: 0.15}
    )
    missing_horizon_policy: str = "renormalize"  # "renormalize", "fail", "ignore"
    min_required_horizons: int = 1

    def validate(self, supported_horizons: Sequence[int] = (1, 3, 5)) -> List[str]:
        errors = []
        valid_strategies = ["weighted_linear", "worst_case", "primary_horizon", "none"]
        if self.strategy not in valid_strategies:
            errors.append(f"aggregation.strategy must be one of {valid_strategies}, got: {self.strategy}")

        if self.primary_horizon not in supported_horizons:
            errors.append(
                f"aggregation.primary_horizon ({self.primary_horizon}) must be in supported horizons: {supported_horizons}"
            )

        if self.strategy == "weighted_linear":
            for h, w in self.horizon_weights.items():
                if h not in supported_horizons:
                    errors.append(f"aggregation.horizon_weights contains unsupported horizon: {h}")
                if w < 0.0:
                    errors.append(f"aggregation.horizon_weights[{h}] must be non-negative, got: {w}")
            total_weight = sum(self.horizon_weights.values())
            if total_weight <= 0.0:
                errors.append(f"aggregation.horizon_weights must sum to a positive value, got: {total_weight}")

        valid_policies = ["renormalize", "fail", "ignore"]
        if self.missing_horizon_policy not in valid_policies:
            errors.append(f"aggregation.missing_horizon_policy must be one of {valid_policies}, got: {self.missing_horizon_policy}")

        if self.min_required_horizons < 1 or self.min_required_horizons > len(supported_horizons):
            errors.append(
                f"aggregation.min_required_horizons must be between 1 and {len(supported_horizons)}, "
                f"got: {self.min_required_horizons}"
            )
        return errors


@dataclass(frozen=True)
class TemporalFeatureConfig:
    """Configuration for temporal trend and degradation feature extraction."""
    history_window_size: int = 5
    min_history_points: int = 2
    degradation_slope_threshold: float = -0.10
    detect_volatility: bool = True
    volatility_window_size: int = 5

    def validate(self) -> List[str]:
        errors = []
        if self.history_window_size < 2:
            errors.append(f"temporal.history_window_size must be >= 2, got: {self.history_window_size}")
        if self.min_history_points < 2 or self.min_history_points > self.history_window_size:
            errors.append(
                f"temporal.min_history_points must be between 2 and {self.history_window_size}, "
                f"got: {self.min_history_points}"
            )
        if self.degradation_slope_threshold >= 0.0:
            errors.append(
                f"temporal.degradation_slope_threshold must be negative, got: {self.degradation_slope_threshold}"
            )
        return errors


@dataclass(frozen=True)
class RiskConfig:
    """Master configuration container for PerceptGuard AI Risk Scoring Engine."""
    supported_horizons: Tuple[int, ...] = (1, 3, 5)
    transformation: TransformationConfig = field(default_factory=TransformationConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    temporal: TemporalFeatureConfig = field(default_factory=TemporalFeatureConfig)
    precision_decimals: int = 6

    def validate(self) -> List[str]:
        errors = []
        if not self.supported_horizons:
            errors.append("supported_horizons must not be empty")
        for h in self.supported_horizons:
            if not isinstance(h, int) or h <= 0:
                errors.append(f"supported_horizons elements must be positive integers, got: {h}")

        errors.extend(self.transformation.validate())
        errors.extend(self.aggregation.validate(supported_horizons=self.supported_horizons))
        errors.extend(self.temporal.validate())

        if self.precision_decimals < 1 or self.precision_decimals > 12:
            errors.append(f"precision_decimals must be between 1 and 12, got: {self.precision_decimals}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RiskConfig:
        """Instantiate RiskConfig from dictionary with validation."""
        data = data or {}

        horizons = tuple(data.get("supported_horizons", [1, 3, 5]))

        # Transformation
        t_data = data.get("transformation", {})
        transformation = TransformationConfig(
            method=t_data.get("method", "linear_inverse"),
            clip_output=t_data.get("clip_output", True),
            min_risk=t_data.get("min_risk", 0.0),
            max_risk=t_data.get("max_risk", 1.0),
        )

        # Aggregation
        a_data = data.get("aggregation", {})
        raw_weights = a_data.get("horizon_weights", {1: 0.50, 3: 0.35, 5: 0.15})
        # Ensure keys are integers
        horizon_weights = {int(k): float(v) for k, v in raw_weights.items()}

        aggregation = AggregationConfig(
            strategy=a_data.get("strategy", "weighted_linear"),
            primary_horizon=a_data.get("primary_horizon", 3),
            horizon_weights=horizon_weights,
            missing_horizon_policy=a_data.get("missing_horizon_policy", "renormalize"),
            min_required_horizons=a_data.get("min_required_horizons", 1),
        )

        # Temporal
        temp_data = data.get("temporal", {})
        temporal = TemporalFeatureConfig(
            history_window_size=temp_data.get("history_window_size", 5),
            min_history_points=temp_data.get("min_history_points", 2),
            degradation_slope_threshold=temp_data.get("degradation_slope_threshold", -0.10),
            detect_volatility=temp_data.get("detect_volatility", True),
            volatility_window_size=temp_data.get("volatility_window_size", 5),
        )

        precision = data.get("precision_decimals", 6)

        config = cls(
            supported_horizons=horizons,
            transformation=transformation,
            aggregation=aggregation,
            temporal=temporal,
            precision_decimals=precision,
        )

        errors = config.validate()
        if errors:
            raise RiskConfigError(f"Risk configuration validation failed with {len(errors)} error(s):\n - " + "\n - ".join(errors))

        return config

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> RiskConfig:
        """Load and validate configuration from YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
