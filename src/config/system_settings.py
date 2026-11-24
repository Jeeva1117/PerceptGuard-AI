"""Centralized, strongly typed, environment-aware configuration engine for PerceptGuard AI.

Supports DEVELOPMENT, TESTING, and PRODUCTION profiles with comprehensive
field validation, secrets masking, environment variable overrides, and 12-factor design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dotenv

from src.config.environment import EnvironmentType
from src.config.secrets import SecretStr, redact_sensitive_dict
from src.config.validator import (
    ConfigurationError,
    validate_cors_origins,
    validate_device,
    validate_log_level,
    validate_numeric_range,
    validate_port,
    validate_url,
)


# Project Root Directory (c:/Users/acer/Desktop/PerceptGuard-AI)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _parse_bool(val: Any, name: str = "Boolean field") -> bool:
    """Parse boolean value from boolean, numeric, or string representation."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in ("true", "1", "yes", "y", "on", "t"):
            return True
        if cleaned in ("false", "0", "no", "n", "off", "f", ""):
            return False
    raise ConfigurationError(f"{name} must be a valid boolean, got {val!r}")


def _parse_int(val: Any, name: str = "Integer field") -> int:
    """Parse integer value with clear error message."""
    try:
        return int(val)
    except (ValueError, TypeError):
        raise ConfigurationError(f"{name} must be an integer, got {val!r}")


def _parse_float(val: Any, name: str = "Float field") -> float:
    """Parse float value with clear error message."""
    try:
        return float(val)
    except (ValueError, TypeError):
        raise ConfigurationError(f"{name} must be a floating-point number, got {val!r}")


def _parse_list(val: Any) -> List[str]:
    """Parse comma-separated string or list into a list of strings."""
    if isinstance(val, list):
        return [str(item).strip() for item in val if str(item).strip()]
    if isinstance(val, str):
        return [item.strip() for item in val.split(",") if item.strip()]
    return []


@dataclass
class AppConfig:
    """Application metadata and general runtime configuration."""

    app_name: str = "PerceptGuard AI"
    version: str = "1.0.0-preview"
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    debug: bool = True
    log_level: str = "INFO"


@dataclass
class APIConfig:
    """FastAPI / Backend REST API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    prefix: str = "/api/v1"
    reload: bool = False
    workers: int = 1
    cors_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ]
    )
    timeout_seconds: float = 30.0


@dataclass
class SecurityConfig:
    """Security credentials, tokens, allowed hosts, and protection rules."""

    secret_key: SecretStr = field(
        default_factory=lambda: SecretStr("dev-insecure-secret-key-perceptguard-2026")
    )
    allowed_hosts: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 120
    enable_csrf: bool = False


@dataclass
class LoggingConfig:
    """Central logging configuration."""

    log_level: str = "INFO"
    json_format: bool = False
    log_file: Optional[Path] = None
    log_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")


@dataclass
class ModelConfig:
    """Machine learning model checkpoints and inference configurations."""

    device: str = "cpu"
    yolo_model: str = "yolov8n.pt"
    lstm_model_path: Path = field(
        default_factory=lambda: BASE_DIR / "models" / "forecasting" / "lstm" / "model.pt"
    )
    baseline_model_path: Path = field(
        default_factory=lambda: BASE_DIR / "models" / "forecasting" / "baseline" / "model.pt"
    )
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45


@dataclass
class VideoProcessingConfig:
    """Video ingestion, tracking, and temporal forecasting configuration."""

    tracker_type: str = "bytetrack"
    historical_sequence_length: int = 10
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 3, 5])
    reliability_iou_correct_threshold: float = 0.50
    reliability_iou_high_threshold: float = 0.70


@dataclass
class DashboardConfig:
    """Streamlit frontend dashboard configuration."""

    host: str = "localhost"
    port: int = 8501
    api_base_url: str = "http://localhost:8000"
    auto_refresh_seconds: int = 5
    telemetry_freshness_ttl_seconds: int = 30
    calibration_status_label: str = "PROVISIONAL CALIBRATION"
    api_request_timeout_seconds: float = 10.0


@dataclass
class SystemSettings:
    """Unified system configuration root containing all subsystems."""

    app: AppConfig = field(default_factory=AppConfig)
    api: APIConfig = field(default_factory=APIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    video_processing: VideoProcessingConfig = field(default_factory=VideoProcessingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    # Core system directory anchors
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    datasets_dir: Path = field(default_factory=lambda: BASE_DIR / "datasets")
    raw_data_dir: Path = field(default_factory=lambda: BASE_DIR / "datasets" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: BASE_DIR / "datasets" / "processed")
    models_dir: Path = field(default_factory=lambda: BASE_DIR / "models")
    outputs_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs")
    config_dir: Path = field(default_factory=lambda: BASE_DIR / "config")
    yolo_config_dir: Path = field(default_factory=lambda: BASE_DIR / ".yolo")

    def __post_init__(self) -> None:
        """Validate entire configuration state upon initialization."""
        self.validate()

    @property
    def environment(self) -> EnvironmentType:
        """Shorthand access to runtime environment profile."""
        return self.app.environment

    @property
    def debug(self) -> bool:
        """Shorthand access to debug flag."""
        return self.app.debug

    @property
    def device(self) -> str:
        """Shorthand access to compute device."""
        return self.model.device

    def validate(self) -> None:
        """Execute cross-component validation rules against the loaded settings."""
        # Validate log level
        self.app.log_level = validate_log_level(self.app.log_level)
        self.logging.log_level = self.app.log_level

        # Validate ports
        validate_port(self.api.port, "API port")
        validate_port(self.dashboard.port, "Dashboard port")

        # Validate URLs
        validate_url(self.dashboard.api_base_url, "API base URL")

        # Validate device
        self.model.device = validate_device(self.model.device)

        # Validate numerical bounds
        validate_numeric_range(
            self.model.confidence_threshold, 0.0, 1.0, "Confidence threshold", inclusive_min=False
        )
        validate_numeric_range(
            self.model.iou_threshold, 0.0, 1.0, "IoU threshold", inclusive_min=False
        )
        validate_numeric_range(
            self.video_processing.historical_sequence_length,
            1,
            1000,
            "Historical sequence length",
        )
        validate_numeric_range(
            self.security.rate_limit_per_minute, 1, 100000, "Rate limit per minute"
        )
        validate_numeric_range(
            self.dashboard.telemetry_freshness_ttl_seconds,
            1,
            3600,
            "Telemetry freshness TTL seconds",
        )
        validate_numeric_range(
            self.dashboard.api_request_timeout_seconds,
            0.1,
            300.0,
            "Dashboard API request timeout seconds",
        )

        # Validate forecast horizons
        if not self.video_processing.forecast_horizons:
            raise ConfigurationError("Forecast horizons list cannot be empty.")
        for h in self.video_processing.forecast_horizons:
            if not isinstance(h, int) or h <= 0:
                raise ConfigurationError(f"Forecast horizon must be a positive integer, got {h}")

        # Validate CORS origins
        self.api.cors_origins = validate_cors_origins(
            self.api.cors_origins, self.app.environment
        )

        # Strict Production Guardrails
        if self.app.environment == EnvironmentType.PRODUCTION:
            if self.app.debug:
                raise ConfigurationError(
                    "DEBUG mode cannot be True in production environment (must be False)."
                )

            # Secret key strength check
            raw_secret = self.security.secret_key.get_secret_value()
            if (
                not raw_secret
                or "dev-insecure" in raw_secret.lower()
                or "secret" == raw_secret.lower()
                or len(raw_secret) < 16
            ):
                raise ConfigurationError(
                    "SECRET_KEY must be a secure, non-default string of at least 16 characters in production."
                )

            # Allowed hosts check
            if "*" in self.security.allowed_hosts:
                raise ConfigurationError(
                    "Wildcard allowed_hosts '*' is strictly prohibited in production. Explicit hostnames or domains are required."
                )

    def to_dict(self, redact_secrets: bool = True) -> Dict[str, Any]:
        """Convert settings tree to a dictionary, optionally masking sensitive values."""
        raw_dict = {
            "app": asdict(self.app),
            "api": asdict(self.api),
            "security": {
                "secret_key": (
                    str(self.security.secret_key)
                    if redact_secrets
                    else self.security.secret_key.get_secret_value()
                ),
                "allowed_hosts": list(self.security.allowed_hosts),
                "rate_limit_per_minute": self.security.rate_limit_per_minute,
                "enable_csrf": self.security.enable_csrf,
            },
            "logging": {
                "log_level": self.logging.log_level,
                "json_format": self.logging.json_format,
                "log_file": str(self.logging.log_file) if self.logging.log_file else None,
                "log_dir": str(self.logging.log_dir),
            },
            "model": {
                "device": self.model.device,
                "yolo_model": self.model.yolo_model,
                "lstm_model_path": str(self.model.lstm_model_path),
                "baseline_model_path": str(self.model.baseline_model_path),
                "confidence_threshold": self.model.confidence_threshold,
                "iou_threshold": self.model.iou_threshold,
            },
            "video_processing": {
                "tracker_type": self.video_processing.tracker_type,
                "historical_sequence_length": self.video_processing.historical_sequence_length,
                "forecast_horizons": list(self.video_processing.forecast_horizons),
                "reliability_iou_correct_threshold": self.video_processing.reliability_iou_correct_threshold,
                "reliability_iou_high_threshold": self.video_processing.reliability_iou_high_threshold,
            },
            "dashboard": asdict(self.dashboard),
            "paths": {
                "base_dir": str(self.base_dir),
                "datasets_dir": str(self.datasets_dir),
                "raw_data_dir": str(self.raw_data_dir),
                "processed_data_dir": str(self.processed_data_dir),
                "models_dir": str(self.models_dir),
                "outputs_dir": str(self.outputs_dir),
                "config_dir": str(self.config_dir),
                "yolo_config_dir": str(self.yolo_config_dir),
            },
        }
        if redact_secrets:
            return redact_sensitive_dict(raw_dict)
        return raw_dict

    @classmethod
    def load(
        cls,
        env_file: Optional[Union[str, Path]] = None,
        env_override: Optional[Dict[str, str]] = None,
    ) -> SystemSettings:
        """Load configuration from environment variables, optional .env file, and explicit overrides."""
        # 1. Load from .env file if specified or if default .env exists
        if env_file:
            env_path = Path(env_file)
            if env_path.exists():
                dotenv.load_dotenv(dotenv_path=env_path, override=True)
        else:
            default_env = BASE_DIR / ".env"
            if default_env.exists():
                dotenv.load_dotenv(dotenv_path=default_env, override=False)

        # Build merged environment map
        source_env = dict(os.environ)
        if env_override:
            source_env.update(env_override)

        # Parse environment profile
        raw_env_str = source_env.get("APP_ENV", source_env.get("ENVIRONMENT", "development"))
        try:
            env_type = EnvironmentType.from_str(raw_env_str)
        except ValueError as e:
            raise ConfigurationError(str(e)) from e

        # AppConfig
        debug_default = env_type != EnvironmentType.PRODUCTION
        raw_debug = source_env.get("DEBUG")
        debug_val = _parse_bool(raw_debug, "DEBUG") if raw_debug is not None else debug_default
        log_level_val = source_env.get("LOG_LEVEL", "INFO")
        app_cfg = AppConfig(
            app_name=source_env.get("APP_NAME", "PerceptGuard AI"),
            version=source_env.get("APP_VERSION", "1.0.0-preview"),
            environment=env_type,
            debug=debug_val,
            log_level=log_level_val,
        )

        # APIConfig
        api_port_val = _parse_int(source_env.get("API_PORT", source_env.get("PORT", "8000")), "API port")
        api_reload_raw = source_env.get("API_RELOAD")
        api_reload_val = _parse_bool(api_reload_raw, "API_RELOAD") if api_reload_raw is not None else False
        cors_raw = source_env.get("CORS_ORIGINS")
        if cors_raw is not None:
            cors_list = _parse_list(cors_raw)
        else:
            cors_list = [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:8501",
                "http://127.0.0.1:8501",
            ]

        api_cfg = APIConfig(
            host=source_env.get("API_HOST", source_env.get("HOST", "0.0.0.0")),
            port=api_port_val,
            prefix=source_env.get("API_PREFIX", "/api/v1"),
            reload=api_reload_val,
            workers=_parse_int(source_env.get("API_WORKERS", "1"), "API workers"),
            cors_origins=cors_list,
            timeout_seconds=_parse_float(source_env.get("API_TIMEOUT_SECONDS", "30.0"), "API timeout"),
        )

        # SecurityConfig
        secret_key_str = source_env.get("SECRET_KEY", "dev-insecure-secret-key-perceptguard-2026")
        allowed_hosts_raw = source_env.get("ALLOWED_HOSTS")
        allowed_hosts_list = _parse_list(allowed_hosts_raw) if allowed_hosts_raw else ["*"]
        sec_cfg = SecurityConfig(
            secret_key=SecretStr(secret_key_str),
            allowed_hosts=allowed_hosts_list,
            rate_limit_per_minute=_parse_int(
                source_env.get("RATE_LIMIT_PER_MINUTE", "120"), "Rate limit per minute"
            ),
            enable_csrf=_parse_bool(source_env.get("ENABLE_CSRF", "false"), "ENABLE_CSRF"),
        )

        # LoggingConfig
        log_cfg = LoggingConfig(
            log_level=log_level_val,
            json_format=_parse_bool(source_env.get("LOG_JSON", "false"), "LOG_JSON"),
            log_file=Path(source_env["LOG_FILE"]) if "LOG_FILE" in source_env else None,
            log_dir=Path(source_env.get("LOG_DIR", str(BASE_DIR / "logs"))),
        )

        # ModelConfig
        default_yolo = "yolo11n.pt" if (BASE_DIR / "models" / "yolo" / "yolo11n.pt").exists() else "yolov8n.pt"
        model_cfg = ModelConfig(
            device=source_env.get("DEVICE", "cpu"),
            yolo_model=source_env.get("YOLO_MODEL", default_yolo),
            lstm_model_path=Path(
                source_env.get(
                    "LSTM_MODEL_PATH",
                    str(BASE_DIR / "models" / "forecasting" / "lstm" / "model.pt"),
                )
            ),
            baseline_model_path=Path(
                source_env.get(
                    "BASELINE_MODEL_PATH",
                    str(BASE_DIR / "models" / "forecasting" / "baseline" / "model.pt"),
                )
            ),
            confidence_threshold=_parse_float(
                source_env.get("CONFIDENCE_THRESHOLD", "0.35"), "Confidence threshold"
            ),
            iou_threshold=_parse_float(source_env.get("IOU_THRESHOLD", "0.45"), "IoU threshold"),
        )

        # VideoProcessingConfig
        horizons_raw = source_env.get("FORECAST_HORIZONS", "1,3,5")
        try:
            horizons = [int(h.strip()) for h in horizons_raw.split(",") if h.strip()]
        except ValueError:
            raise ConfigurationError(f"FORECAST_HORIZONS must be comma-separated integers, got {horizons_raw!r}")

        video_cfg = VideoProcessingConfig(
            tracker_type=source_env.get("TRACKER_TYPE", "bytetrack"),
            historical_sequence_length=_parse_int(
                source_env.get("HISTORICAL_SEQUENCE_LENGTH", "10"), "Historical sequence length"
            ),
            forecast_horizons=horizons,
            reliability_iou_correct_threshold=_parse_float(
                source_env.get("RELIABILITY_IOU_CORRECT_THRESHOLD", "0.50"),
                "Reliability IoU correct threshold",
            ),
            reliability_iou_high_threshold=_parse_float(
                source_env.get("RELIABILITY_IOU_HIGH_THRESHOLD", "0.70"),
                "Reliability IoU high threshold",
            ),
        )

        # DashboardConfig
        dash_port_val = _parse_int(
            source_env.get("STREAMLIT_PORT", source_env.get("DASHBOARD_PORT", "8501")),
            "Dashboard port",
        )
        dash_cfg = DashboardConfig(
            host=source_env.get("STREAMLIT_HOST", source_env.get("DASHBOARD_HOST", "localhost")),
            port=dash_port_val,
            api_base_url=source_env.get("API_BASE_URL", "http://localhost:8000"),
            auto_refresh_seconds=_parse_int(
                source_env.get("DASHBOARD_AUTO_REFRESH_SECONDS", "5"), "Auto refresh seconds"
            ),
            telemetry_freshness_ttl_seconds=_parse_int(
                source_env.get("TELEMETRY_FRESHNESS_TTL_SECONDS", "30"), "Telemetry freshness TTL"
            ),
            calibration_status_label=source_env.get(
                "CALIBRATION_STATUS_LABEL", "PROVISIONAL CALIBRATION"
            ),
            api_request_timeout_seconds=_parse_float(
                source_env.get(
                    "DASHBOARD_REQUEST_TIMEOUT_SECONDS",
                    source_env.get("API_REQUEST_TIMEOUT_SECONDS", "10.0"),
                ),
                "Dashboard API request timeout",
            ),
        )

        # Assemble and validate
        return cls(
            app=app_cfg,
            api=api_cfg,
            security=sec_cfg,
            logging=log_cfg,
            model=model_cfg,
            video_processing=video_cfg,
            dashboard=dash_cfg,
        )


# Global Singleton Cache
_settings_instance: Optional[SystemSettings] = None


def get_settings() -> SystemSettings:
    """Access the application-wide cached configuration singleton."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = SystemSettings.load()
    return _settings_instance


def reload_settings(
    env_file: Optional[Union[str, Path]] = None,
    env_override: Optional[Dict[str, str]] = None,
) -> SystemSettings:
    """Reload and update the application configuration singleton."""
    global _settings_instance
    _settings_instance = SystemSettings.load(env_file=env_file, env_override=env_override)
    return _settings_instance
