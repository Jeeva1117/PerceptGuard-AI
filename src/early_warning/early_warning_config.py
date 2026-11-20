"""Early Warning Configuration Module for PerceptGuard AI.

Defines strongly-typed, CPU-first configuration schemas and validation
for the predictive early warning and perception risk assessment system.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml


class EarlyWarningConfigError(ValueError):
    """Raised when early warning configuration validation fails."""
    pass


@dataclass(frozen=True)
class ProjectConfig:
    """Project-level metadata configuration."""
    name: str = "PerceptGuard AI"
    stage: str = "early_warning"
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
    """Runtime execution parameters tailored for CPU-only execution."""
    device: str = "cpu"
    num_threads: int = 2
    deterministic: bool = True
    max_memory_rss_mb: int = 512

    def validate(self) -> List[str]:
        errors = []
        dev_str = str(self.device).lower().strip()
        if dev_str not in ["cpu"]:
            errors.append(f"runtime.device must be 'cpu' (CUDA is not supported), got: {self.device}")
        if not isinstance(self.num_threads, int) or self.num_threads < 1:
            errors.append(f"runtime.num_threads must be a positive integer, got: {self.num_threads}")
        if not isinstance(self.max_memory_rss_mb, int) or self.max_memory_rss_mb < 64:
            errors.append(f"runtime.max_memory_rss_mb must be at least 64 MB, got: {self.max_memory_rss_mb}")
        return errors


@dataclass(frozen=True)
class ForecastHorizonsConfig:
    """Configuration for consumed multi-horizon forecasts."""
    supported: Tuple[int, ...] = (1, 3, 5)
    default_horizon: int = 3
    weights: Dict[str, float] = field(
        default_factory=lambda: {"h1": 0.50, "h3": 0.35, "h5": 0.15}
    )

    def validate(self) -> List[str]:
        errors = []
        if not self.supported:
            errors.append("forecast_horizons.supported must be a non-empty sequence")
        for h in self.supported:
            if not isinstance(h, int) or h <= 0:
                errors.append(f"forecast_horizons elements must be positive integers, got: {h}")
        if self.default_horizon not in self.supported:
            errors.append(
                f"forecast_horizons.default_horizon ({self.default_horizon}) must be in supported horizons: {self.supported}"
            )
        # Check weights sum approximately to 1.0
        weight_sum = sum(self.weights.values())
        if not (0.95 <= weight_sum <= 1.05):
            errors.append(f"forecast_horizons.weights must sum approximately to 1.0, got: {weight_sum:.4f}")
        return errors


@dataclass(frozen=True)
class RiskClassificationConfig:
    """Configuration for risk levels and classification bounds."""
    levels: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    provisional_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "low_upper": 0.30,
            "medium_upper": 0.60,
            "high_upper": 0.85,
        }
    )
    boundary_policy: str = "closed_right"
    missing_data_level: str = "UNKNOWN"

    def validate(self) -> List[str]:
        errors = []
        expected_levels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if set(self.levels) != expected_levels:
            errors.append(f"risk_classification.levels must contain exactly {expected_levels}, got: {self.levels}")
        low = self.provisional_thresholds.get("low_upper", 0.0)
        med = self.provisional_thresholds.get("medium_upper", 0.0)
        high = self.provisional_thresholds.get("high_upper", 0.0)
        if not (0.0 < low < med < high < 1.0):
            errors.append(
                f"risk_classification provisional thresholds must satisfy 0.0 < low < med < high < 1.0, "
                f"got: low={low}, med={med}, high={high}"
            )
        if self.boundary_policy not in ["closed_right", "closed_left"]:
            errors.append(f"risk_classification.boundary_policy must be 'closed_right' or 'closed_left', got: {self.boundary_policy}")
        return errors


@dataclass(frozen=True)
class TemporalMonitoringConfig:
    """Configuration for sliding window temporal risk monitoring and debouncing."""
    sliding_window_size: int = 5
    min_consecutive_alerts: int = 2
    alert_debounce_frames: int = 3
    degradation_slope_threshold: float = -0.10
    trend_history_length: int = 5

    def validate(self) -> List[str]:
        errors = []
        if self.sliding_window_size < 1:
            errors.append(f"temporal_monitoring.sliding_window_size must be >= 1, got: {self.sliding_window_size}")
        if self.min_consecutive_alerts < 1:
            errors.append(f"temporal_monitoring.min_consecutive_alerts must be >= 1, got: {self.min_consecutive_alerts}")
        if self.alert_debounce_frames < 0:
            errors.append(f"temporal_monitoring.alert_debounce_frames must be >= 0, got: {self.alert_debounce_frames}")
        if self.degradation_slope_threshold >= 0.0:
            errors.append(f"temporal_monitoring.degradation_slope_threshold must be negative, got: {self.degradation_slope_threshold}")
        return errors


@dataclass(frozen=True)
class AlertGenerationConfig:
    """Configuration for alert suppression, cooldown, and notification routing."""
    cooldown_frames: int = 5
    suppress_redundant: bool = True
    enable_critical_escalation: bool = True
    max_active_alerts: int = 10
    notification_channels: Tuple[str, ...] = ("log", "dashboard_event")

    def validate(self) -> List[str]:
        errors = []
        if self.cooldown_frames < 0:
            errors.append(f"alert_generation.cooldown_frames must be >= 0, got: {self.cooldown_frames}")
        if self.max_active_alerts < 1:
            errors.append(f"alert_generation.max_active_alerts must be >= 1, got: {self.max_active_alerts}")
        if not self.notification_channels:
            errors.append("alert_generation.notification_channels must not be empty")
        return errors


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for prediction ingestion validation and freshness."""
    prediction_freshness_max_ms: float = 1000.0
    allow_missing_horizons: bool = True
    require_timestamp: bool = True
    validation_mode: str = "strict"

    def validate(self) -> List[str]:
        errors = []
        if self.prediction_freshness_max_ms <= 0.0:
            errors.append(f"data_ingestion.prediction_freshness_max_ms must be > 0.0, got: {self.prediction_freshness_max_ms}")
        if self.validation_mode not in ["strict", "permissive"]:
            errors.append(f"data_ingestion.validation_mode must be 'strict' or 'permissive', got: {self.validation_mode}")
        return errors


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and telemetry parameters."""
    level: str = "INFO"
    log_predictions: bool = True
    log_risk_scores: bool = True
    log_warnings: bool = True

    def validate(self) -> List[str]:
        errors = []
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            errors.append(f"logging.level must be one of {valid_levels}, got: {self.level}")
        return errors


@dataclass(frozen=True)
class EarlyWarningConfig:
    """Master configuration container for PerceptGuard AI Early Warning System."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    forecast_horizons: ForecastHorizonsConfig = field(default_factory=ForecastHorizonsConfig)
    risk_classification: RiskClassificationConfig = field(default_factory=RiskClassificationConfig)
    temporal_monitoring: TemporalMonitoringConfig = field(default_factory=TemporalMonitoringConfig)
    alert_generation: AlertGenerationConfig = field(default_factory=AlertGenerationConfig)
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> List[str]:
        """Aggregate validation errors across all configuration sub-sections."""
        errors = []
        errors.extend(self.project.validate())
        errors.extend(self.runtime.validate())
        errors.extend(self.forecast_horizons.validate())
        errors.extend(self.risk_classification.validate())
        errors.extend(self.temporal_monitoring.validate())
        errors.extend(self.alert_generation.validate())
        errors.extend(self.data_ingestion.validate())
        errors.extend(self.logging.validate())
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a nested dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EarlyWarningConfig:
        """Instantiate strongly-typed EarlyWarningConfig from raw dictionary."""
        data = data or {}

        # Project
        proj_dict = data.get("project", {})
        project = ProjectConfig(
            name=proj_dict.get("name", "PerceptGuard AI"),
            stage=proj_dict.get("stage", "early_warning"),
            random_seed=proj_dict.get("random_seed", 42),
        )

        # Runtime
        rt_dict = data.get("runtime", {})
        runtime = RuntimeConfig(
            device=rt_dict.get("device", "cpu"),
            num_threads=rt_dict.get("num_threads", 2),
            deterministic=rt_dict.get("deterministic", True),
            max_memory_rss_mb=rt_dict.get("max_memory_rss_mb", 512),
        )

        # Forecast Horizons
        fh_dict = data.get("forecast_horizons", {})
        supported = tuple(fh_dict.get("supported", [1, 3, 5]))
        default_horizon = fh_dict.get("default_horizon", 3)
        weights = fh_dict.get("weights", {"h1": 0.50, "h3": 0.35, "h5": 0.15})
        forecast_horizons = ForecastHorizonsConfig(
            supported=supported,
            default_horizon=default_horizon,
            weights=weights,
        )

        # Risk Classification
        rc_dict = data.get("risk_classification", {})
        levels = tuple(rc_dict.get("levels", ["LOW", "MEDIUM", "HIGH", "CRITICAL"]))
        provisional_thresholds = rc_dict.get(
            "provisional_thresholds",
            {"low_upper": 0.30, "medium_upper": 0.60, "high_upper": 0.85},
        )
        boundary_policy = rc_dict.get("boundary_policy", "closed_right")
        missing_data_level = rc_dict.get("missing_data_level", "UNKNOWN")
        risk_classification = RiskClassificationConfig(
            levels=levels,
            provisional_thresholds=provisional_thresholds,
            boundary_policy=boundary_policy,
            missing_data_level=missing_data_level,
        )

        # Temporal Monitoring
        tm_dict = data.get("temporal_monitoring", {})
        temporal_monitoring = TemporalMonitoringConfig(
            sliding_window_size=tm_dict.get("sliding_window_size", 5),
            min_consecutive_alerts=tm_dict.get("min_consecutive_alerts", 2),
            alert_debounce_frames=tm_dict.get("alert_debounce_frames", 3),
            degradation_slope_threshold=tm_dict.get("degradation_slope_threshold", -0.10),
            trend_history_length=tm_dict.get("trend_history_length", 5),
        )

        # Alert Generation
        ag_dict = data.get("alert_generation", {})
        alert_generation = AlertGenerationConfig(
            cooldown_frames=ag_dict.get("cooldown_frames", 5),
            suppress_redundant=ag_dict.get("suppress_redundant", True),
            enable_critical_escalation=ag_dict.get("enable_critical_escalation", True),
            max_active_alerts=ag_dict.get("max_active_alerts", 10),
            notification_channels=tuple(ag_dict.get("notification_channels", ["log", "dashboard_event"])),
        )

        # Data Ingestion
        di_dict = data.get("data_ingestion", {})
        data_ingestion = DataIngestionConfig(
            prediction_freshness_max_ms=di_dict.get("prediction_freshness_max_ms", 1000.0),
            allow_missing_horizons=di_dict.get("allow_missing_horizons", True),
            require_timestamp=di_dict.get("require_timestamp", True),
            validation_mode=di_dict.get("validation_mode", "strict"),
        )

        # Logging
        log_dict = data.get("logging", {})
        logging = LoggingConfig(
            level=log_dict.get("level", "INFO"),
            log_predictions=log_dict.get("log_predictions", True),
            log_risk_scores=log_dict.get("log_risk_scores", True),
            log_warnings=log_dict.get("log_warnings", True),
        )

        config = cls(
            project=project,
            runtime=runtime,
            forecast_horizons=forecast_horizons,
            risk_classification=risk_classification,
            temporal_monitoring=temporal_monitoring,
            alert_generation=alert_generation,
            data_ingestion=data_ingestion,
            logging=logging,
        )

        errors = config.validate()
        if errors:
            raise EarlyWarningConfigError(f"Configuration validation failed with {len(errors)} error(s):\n - " + "\n - ".join(errors))

        return config

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> EarlyWarningConfig:
        """Load and validate configuration from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


def load_early_warning_config(config_path: Optional[Union[str, Path]] = None) -> EarlyWarningConfig:
    """Load early warning configuration from given path or default location."""
    if config_path is not None:
        return EarlyWarningConfig.from_yaml(config_path)

    default_path = Path("config/early_warning_config.yaml")
    if default_path.exists():
        return EarlyWarningConfig.from_yaml(default_path)

    # Return default initialized config
    return EarlyWarningConfig()
