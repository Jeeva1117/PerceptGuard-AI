"""Validation utilities and constraint rules for configuration parameters."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence
from urllib.parse import urlparse

from src.config.environment import EnvironmentType


class ConfigurationError(ValueError):
    """Raised when environment or application configuration fails validation."""
    pass


def validate_port(port: int, name: str = "Port") -> int:
    """Ensure port is within valid TCP port range [1, 65535]."""
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise ConfigurationError(f"{name} must be an integer between 1 and 65535, got {port}")
    return port


def validate_numeric_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
) -> float:
    """Ensure numerical setting is within designated bounds."""
    val = float(value)
    min_ok = val >= min_val if inclusive_min else val > min_val
    max_ok = val <= max_val if inclusive_max else val < max_val
    if not (min_ok and max_ok):
        op_min = ">=" if inclusive_min else ">"
        op_max = "<=" if inclusive_max else "<"
        raise ConfigurationError(
            f"{name} must satisfy {min_val} {op_min} value {op_max} {max_val}, got {val}"
        )
    return val


def validate_cors_origins(origins: Sequence[str], env: EnvironmentType) -> List[str]:
    """Validate CORS origins for security and syntax compliance."""
    if not origins:
        if env == EnvironmentType.PRODUCTION:
            raise ConfigurationError("CORS origins cannot be empty in production.")
        return ["http://localhost:8501", "http://127.0.0.1:8501"]

    cleaned: List[str] = []
    for origin in origins:
        org = origin.strip()
        if not org:
            continue
        if org == "*":
            if env == EnvironmentType.PRODUCTION:
                raise ConfigurationError(
                    "Wildcard CORS origin '*' is strictly prohibited in production."
                )
            cleaned.append(org)
            continue

        parsed = urlparse(org)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            raise ConfigurationError(
                f"Invalid CORS origin URL '{org}'. Must include http:// or https:// scheme."
            )
        cleaned.append(org)

    if not cleaned and env == EnvironmentType.PRODUCTION:
        raise ConfigurationError("No valid CORS origins configured for production.")
    return cleaned


def validate_log_level(level: str) -> str:
    """Validate log level string."""
    allowed = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}
    lvl = level.strip().upper()
    if lvl not in allowed:
        raise ConfigurationError(f"Invalid log level '{level}'. Must be one of: {sorted(allowed)}")
    return "WARNING" if lvl == "WARN" else lvl


def validate_device(device: str) -> str:
    """Validate device selection ('cpu' or 'cuda')."""
    dev = device.strip().lower()
    if dev not in ("cpu", "cuda"):
        raise ConfigurationError(f"Unsupported device '{device}'. Must be 'cpu' or 'cuda'.")
    return dev


def validate_url(url: str, name: str = "URL") -> str:
    """Ensure URL string contains valid scheme and network location."""
    if not url or not isinstance(url, str):
        raise ConfigurationError(f"{name} must be a non-empty string URL.")
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ConfigurationError(
            f"Invalid {name} '{url}'. Must include valid scheme (e.g. http://) and host."
        )
    return url.strip()
