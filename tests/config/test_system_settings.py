"""Comprehensive test suite for PerceptGuard AI configuration hardening.

Validates the 12 required areas:
1. Default development configuration
2. Explicit testing configuration
3. Production configuration validation & security guardrails
4. Missing / insecure settings rejection in production
5. Invalid environment name rejection
6. Invalid URL validation for CORS and API endpoints
7. Invalid numeric setting validation (ports, bounds, thresholds)
8. Secret masking and redaction utilities
9. CORS configuration validation
10. Frontend dashboard configuration parameters
11. Model path and compute device validation
12. Backward compatibility with config.settings and config
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.config.environment import EnvironmentType
from src.config.secrets import SecretStr, redact_sensitive_dict
from src.config.system_settings import (
    SystemSettings,
    get_settings,
    reload_settings,
)
from src.config.validator import (
    ConfigurationError,
    validate_cors_origins,
    validate_device,
    validate_numeric_range,
    validate_port,
    validate_url,
)


# ==============================================================================
# AREA 1: Default Development Configuration
# ==============================================================================

def test_default_development_config():
    """Verify that default settings instantiate cleanly in development mode."""
    settings = SystemSettings.load(env_override={"APP_ENV": "development"})

    assert settings.app.environment == EnvironmentType.DEVELOPMENT
    assert settings.app.debug is True
    assert settings.api.port == 8000
    assert settings.dashboard.port == 8501
    assert settings.video_processing.forecast_horizons == [1, 3, 5]
    assert settings.video_processing.historical_sequence_length == 10
    assert settings.model.device in ("cpu", "cuda")
    assert settings.base_dir.exists()
    assert settings.config_dir.exists()


# ==============================================================================
# AREA 2: Explicit Testing Configuration
# ==============================================================================

def test_explicit_testing_config():
    """Verify that testing environment profile is properly initialized."""
    settings = SystemSettings.load(
        env_override={
            "APP_ENV": "testing",
            "DEBUG": "false",
            "LOG_LEVEL": "DEBUG",
        }
    )

    assert settings.app.environment == EnvironmentType.TESTING
    assert settings.app.debug is False
    assert settings.app.log_level == "DEBUG"
    assert settings.logging.log_level == "DEBUG"


# ==============================================================================
# AREA 3: Production Configuration Validation & Security Guardrails
# ==============================================================================

def test_production_valid_configuration():
    """Verify that a compliant production configuration validates successfully."""
    valid_prod_env = {
        "APP_ENV": "production",
        "DEBUG": "false",
        "SECRET_KEY": "a-very-strong-production-secret-key-32chars",
        "CORS_ORIGINS": "https://dashboard.perceptguard.com",
        "ALLOWED_HOSTS": "api.perceptguard.com",
    }
    settings = SystemSettings.load(env_override=valid_prod_env)

    assert settings.app.environment == EnvironmentType.PRODUCTION
    assert settings.app.debug is False
    assert settings.security.secret_key.get_secret_value() == "a-very-strong-production-secret-key-32chars"
    assert settings.api.cors_origins == ["https://dashboard.perceptguard.com"]
    assert settings.security.allowed_hosts == ["api.perceptguard.com"]


def test_production_rejects_debug_true():
    """Verify that production mode strictly forbids DEBUG=True."""
    with pytest.raises(ConfigurationError, match="DEBUG mode cannot be True in production"):
        SystemSettings.load(
            env_override={
                "APP_ENV": "production",
                "DEBUG": "true",
                "SECRET_KEY": "a-very-strong-production-secret-key-32chars",
                "CORS_ORIGINS": "https://dashboard.perceptguard.com",
                "ALLOWED_HOSTS": "api.perceptguard.com",
            }
        )


def test_production_rejects_wildcard_cors():
    """Verify that production mode strictly forbids wildcard '*' CORS origin."""
    with pytest.raises(ConfigurationError, match=r"Wildcard CORS origin '\*' is strictly prohibited"):
        SystemSettings.load(
            env_override={
                "APP_ENV": "production",
                "DEBUG": "false",
                "SECRET_KEY": "a-very-strong-production-secret-key-32chars",
                "CORS_ORIGINS": "*",
                "ALLOWED_HOSTS": "api.perceptguard.com",
            }
        )


def test_production_rejects_default_secret_key():
    """Verify that production mode rejects default or insecure SECRET_KEY."""
    with pytest.raises(ConfigurationError, match="SECRET_KEY must be a secure, non-default string"):
        SystemSettings.load(
            env_override={
                "APP_ENV": "production",
                "DEBUG": "false",
                "SECRET_KEY": "dev-insecure-secret-key-perceptguard-2026",
                "CORS_ORIGINS": "https://dashboard.perceptguard.com",
                "ALLOWED_HOSTS": "api.perceptguard.com",
            }
        )


# ==============================================================================
# AREA 4: Missing Required Settings in Production
# ==============================================================================

def test_production_rejects_short_secret_key():
    """Verify that short secret keys (<16 chars) are rejected in production."""
    with pytest.raises(ConfigurationError, match="at least 16 characters"):
        SystemSettings.load(
            env_override={
                "APP_ENV": "production",
                "DEBUG": "false",
                "SECRET_KEY": "short-secret",
                "CORS_ORIGINS": "https://dashboard.perceptguard.com",
                "ALLOWED_HOSTS": "api.perceptguard.com",
            }
        )


def test_production_rejects_wildcard_allowed_hosts():
    """Verify that wildcard allowed_hosts '*' is forbidden in production."""
    with pytest.raises(ConfigurationError, match=r"Wildcard allowed_hosts '\*' is strictly prohibited"):
        SystemSettings.load(
            env_override={
                "APP_ENV": "production",
                "DEBUG": "false",
                "SECRET_KEY": "a-very-strong-production-secret-key-32chars",
                "CORS_ORIGINS": "https://dashboard.perceptguard.com",
                "ALLOWED_HOSTS": "*",
            }
        )


# ==============================================================================
# AREA 5: Invalid Environment Names
# ==============================================================================

@pytest.mark.parametrize("invalid_env", ["staging", "invalid", "qa", "sandbox", ""])
def test_invalid_environment_names(invalid_env: str):
    """Verify that unrecognized or non-standard environment names raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        SystemSettings.load(env_override={"APP_ENV": invalid_env})


# ==============================================================================
# AREA 6: Invalid URLs in CORS Origins and API URLs
# ==============================================================================

def test_invalid_api_base_url():
    """Verify that malformed API base URLs fail validation."""
    with pytest.raises(ConfigurationError, match="Invalid API base URL"):
        SystemSettings.load(env_override={"API_BASE_URL": "not-a-valid-url"})


def test_invalid_cors_origin_scheme():
    """Verify that CORS origins without http:// or https:// fail validation."""
    with pytest.raises(ConfigurationError, match="Must include http:// or https:// scheme"):
        validate_cors_origins(["ftp://malicious.org"], EnvironmentType.DEVELOPMENT)

    with pytest.raises(ConfigurationError, match="Must include http:// or https:// scheme"):
        validate_cors_origins(["localhost:8501"], EnvironmentType.DEVELOPMENT)


# ==============================================================================
# AREA 7: Invalid Numeric Settings
# ==============================================================================

@pytest.mark.parametrize("invalid_port", [0, -1, 65536, 70000])
def test_invalid_ports(invalid_port: int):
    """Verify that out-of-range ports raise ConfigurationError."""
    with pytest.raises(ConfigurationError, match="must be an integer between 1 and 65535"):
        validate_port(invalid_port)


def test_invalid_confidence_threshold():
    """Verify that non-standard confidence thresholds are rejected."""
    with pytest.raises(ConfigurationError):
        SystemSettings.load(env_override={"CONFIDENCE_THRESHOLD": "1.5"})

    with pytest.raises(ConfigurationError):
        SystemSettings.load(env_override={"CONFIDENCE_THRESHOLD": "-0.1"})


def test_invalid_sequence_length():
    """Verify that zero or negative sequence lengths are rejected."""
    with pytest.raises(ConfigurationError):
        SystemSettings.load(env_override={"HISTORICAL_SEQUENCE_LENGTH": "0"})


# ==============================================================================
# AREA 8: Secret Redaction & Masking
# ==============================================================================

def test_secret_string_masking():
    """Verify that SecretStr masks itself in str and repr."""
    raw = "super-confidential-token-12345"
    sec = SecretStr(raw)

    assert str(sec) == "**********"
    assert repr(sec) == "SecretStr('**********')"
    assert sec.get_secret_value() == raw
    assert len(sec) == len(raw)
    assert bool(sec) is True


def test_recursive_dictionary_redaction():
    """Verify that redact_sensitive_dict scrubs sensitive keys and SecretStr instances."""
    data = {
        "app_name": "PerceptGuard",
        "api_key": "raw-api-key",
        "secret_token": "raw-token",
        "nested": {
            "password": "my-password",
            "safe_val": 42,
        },
        "custom_secret": SecretStr("my-custom-secret"),
    }
    redacted = redact_sensitive_dict(data)

    assert redacted["app_name"] == "PerceptGuard"
    assert redacted["api_key"] == "**********"
    assert redacted["secret_token"] == "**********"
    assert redacted["nested"]["password"] == "**********"
    assert redacted["nested"]["safe_val"] == 42
    assert redacted["custom_secret"] == "**********"


def test_settings_to_dict_redaction():
    """Verify that SystemSettings.to_dict masks credentials by default."""
    settings = SystemSettings.load(
        env_override={"SECRET_KEY": "my-development-secret-key-1234"}
    )
    dump = settings.to_dict(redact_secrets=True)
    assert dump["security"]["secret_key"] == "**********"

    unmasked_dump = settings.to_dict(redact_secrets=False)
    assert unmasked_dump["security"]["secret_key"] == "my-development-secret-key-1234"


# ==============================================================================
# AREA 9: CORS Configuration Validation
# ==============================================================================

def test_cors_list_parsing():
    """Verify comma-separated CORS string parsing."""
    settings = SystemSettings.load(
        env_override={
            "CORS_ORIGINS": "http://localhost:3000, https://app.example.com",
        }
    )
    assert settings.api.cors_origins == ["http://localhost:3000", "https://app.example.com"]


# ==============================================================================
# AREA 10: Frontend API Configuration Parameters
# ==============================================================================

def test_dashboard_settings_parsing():
    """Verify parsing and validation of dashboard configuration parameters."""
    settings = SystemSettings.load(
        env_override={
            "STREAMLIT_HOST": "127.0.0.1",
            "STREAMLIT_PORT": "8502",
            "API_BASE_URL": "http://127.0.0.1:8000",
            "DASHBOARD_REQUEST_TIMEOUT_SECONDS": "18.5",
            "CALIBRATION_STATUS_LABEL": "VERIFIED PRODUCTION",
            "TELEMETRY_FRESHNESS_TTL_SECONDS": "45",
        }
    )
    assert settings.dashboard.host == "127.0.0.1"
    assert settings.dashboard.port == 8502
    assert settings.dashboard.api_base_url == "http://127.0.0.1:8000"
    assert settings.dashboard.api_request_timeout_seconds == 18.5
    assert settings.dashboard.calibration_status_label == "VERIFIED PRODUCTION"
    assert settings.dashboard.telemetry_freshness_ttl_seconds == 45


# ==============================================================================
# AREA 11: Model Path and Device Validation
# ==============================================================================

def test_device_validation():
    """Verify valid compute devices ('cpu', 'cuda') and rejection of invalid ones."""
    assert validate_device("cpu") == "cpu"
    assert validate_device("cuda") == "cuda"
    assert validate_device("CPU") == "cpu"
    assert validate_device("CUDA") == "cuda"

    with pytest.raises(ConfigurationError, match="Unsupported device"):
        validate_device("tpu")


def test_forecast_horizons_parsing():
    """Verify multi-horizon parsing and validation."""
    settings = SystemSettings.load(env_override={"FORECAST_HORIZONS": "1, 3, 5, 10"})
    assert settings.video_processing.forecast_horizons == [1, 3, 5, 10]

    with pytest.raises(ConfigurationError):
        SystemSettings.load(env_override={"FORECAST_HORIZONS": "1, invalid, 5"})


# ==============================================================================
# AREA 12: Backward Compatibility with config.settings and config
# ==============================================================================

def test_backward_compatibility_with_config_settings():
    """Verify that existing legacy imports from config.settings continue to work identically."""
    import config.settings as legacy_settings
    from config.settings import (
        BASE_DIR,
        DEVICE,
        DEFAULT_YOLO_MODEL,
        SEQUENCE_LENGTH,
        FORECAST_HORIZONS,
        DEFAULT_CONF_THRESHOLD,
        DEFAULT_IOU_THRESHOLD,
        TRACKER_TYPE,
        RELIABILITY_IOU_CORRECT_THRESHOLD,
        RELIABILITY_IOU_HIGH_THRESHOLD,
    )

    current_settings = get_settings()

    assert BASE_DIR == current_settings.base_dir
    assert DEVICE == current_settings.model.device
    assert DEFAULT_YOLO_MODEL == current_settings.model.yolo_model
    assert SEQUENCE_LENGTH == current_settings.video_processing.historical_sequence_length
    assert FORECAST_HORIZONS == current_settings.video_processing.forecast_horizons
    assert DEFAULT_CONF_THRESHOLD == current_settings.model.confidence_threshold
    assert DEFAULT_IOU_THRESHOLD == current_settings.model.iou_threshold
    assert TRACKER_TYPE == current_settings.video_processing.tracker_type
    assert RELIABILITY_IOU_CORRECT_THRESHOLD == current_settings.video_processing.reliability_iou_correct_threshold
    assert RELIABILITY_IOU_HIGH_THRESHOLD == current_settings.video_processing.reliability_iou_high_threshold
    assert isinstance(BASE_DIR, Path)


def test_reload_settings_singleton():
    """Verify that reload_settings updates the global singleton."""
    s1 = get_settings()
    s2 = reload_settings(env_override={"APP_NAME": "PerceptGuard Custom"})

    assert s2.app.app_name == "PerceptGuard Custom"
    assert get_settings().app.app_name == "PerceptGuard Custom"

    # Reset back to default
    reload_settings(env_override={"APP_NAME": "PerceptGuard AI"})
    assert get_settings().app.app_name == "PerceptGuard AI"
