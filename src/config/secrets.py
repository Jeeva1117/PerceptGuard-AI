"""Secrets management and redaction utilities for PerceptGuard AI.

Provides SecretStr to prevent accidental leakage of sensitive values
in logs, exceptions, prints, and serialized representations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Union


class SecretStr:
    """Container for sensitive string values that masks itself in all string representations."""

    def __init__(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"SecretStr value must be a string, got {type(value).__name__}")
        self._secret_value = value

    def get_secret_value(self) -> str:
        """Retrieve the raw unmasked secret value."""
        return self._secret_value

    def __repr__(self) -> str:
        return "SecretStr('**********')"

    def __str__(self) -> str:
        return "**********"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SecretStr):
            return self._secret_value == other._secret_value
        return False

    def __hash__(self) -> int:
        return hash(self._secret_value)

    def __bool__(self) -> bool:
        return bool(self._secret_value)

    def __len__(self) -> int:
        return len(self._secret_value)


SENSITIVE_KEY_SUBSTRINGS: Set[str] = {
    "secret",
    "password",
    "token",
    "api_key",
    "apikey",
    "credential",
    "private_key",
    "auth",
}


def redact_sensitive_dict(data: Dict[str, Any], mask: str = "**********") -> Dict[str, Any]:
    """Recursively redact dictionary entries whose keys indicate sensitive information."""
    redacted: Dict[str, Any] = {}
    for key, value in data.items():
        key_lower = str(key).lower()
        if any(sub in key_lower for sub in SENSITIVE_KEY_SUBSTRINGS):
            redacted[key] = mask
        elif isinstance(value, SecretStr):
            redacted[key] = mask
        elif isinstance(value, dict):
            redacted[key] = redact_sensitive_dict(value, mask=mask)
        elif isinstance(value, list):
            redacted[key] = [
                redact_sensitive_dict(item, mask=mask) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            redacted[key] = value
    return redacted
