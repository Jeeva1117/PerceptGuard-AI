"""PerceptGuard AI — Configuration Package.

Provides environment classification, secrets handling, constraint validation,
and centralized strongly typed system settings.
"""

from src.config.environment import EnvironmentType
from src.config.secrets import SecretStr, redact_sensitive_dict
from src.config.system_settings import (
    APIConfig,
    AppConfig,
    DashboardConfig,
    LoggingConfig,
    ModelConfig,
    SecurityConfig,
    SystemSettings,
    VideoProcessingConfig,
    get_settings,
    reload_settings,
)
from src.config.validator import (
    ConfigurationError,
    validate_cors_origins,
    validate_device,
    validate_log_level,
    validate_numeric_range,
    validate_port,
    validate_url,
)

__all__ = [
    "EnvironmentType",
    "SecretStr",
    "redact_sensitive_dict",
    "ConfigurationError",
    "validate_port",
    "validate_numeric_range",
    "validate_cors_origins",
    "validate_log_level",
    "validate_device",
    "validate_url",
    "AppConfig",
    "APIConfig",
    "SecurityConfig",
    "LoggingConfig",
    "ModelConfig",
    "VideoProcessingConfig",
    "DashboardConfig",
    "SystemSettings",
    "get_settings",
    "reload_settings",
]
