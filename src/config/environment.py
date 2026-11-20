"""Environment enumeration and resolution for PerceptGuard AI."""

from __future__ import annotations

from enum import Enum


class EnvironmentType(str, Enum):
    """Supported runtime deployment environment profiles."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

    @classmethod
    def from_str(cls, value: str) -> EnvironmentType:
        """Parse and validate environment string case-insensitively."""
        if not value or not isinstance(value, str):
            raise ValueError(f"Environment name must be a non-empty string, got {value!r}")
        cleaned = value.strip().lower()
        # Aliases
        if cleaned in ("dev", "local", "development"):
            return cls.DEVELOPMENT
        elif cleaned in ("test", "testing", "pytest"):
            return cls.TESTING
        elif cleaned in ("prod", "production", "live"):
            return cls.PRODUCTION
        else:
            allowed = [e.value for e in cls]
            raise ValueError(
                f"Invalid environment name '{value}'. Must be one of: {allowed}"
            )
