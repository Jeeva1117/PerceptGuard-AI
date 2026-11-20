"""Validation utilities for PerceptGuard AI Risk Scoring Engine.

Ensures numerical safety, range conformance, identifier preservation,
and horizon validity for all input forecasts and derived risk scores.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union


class RiskValidationError(ValueError):
    """Raised when risk score validation or input transformation checks fail."""
    pass


def validate_reliability_value(value: Any) -> float:
    """Validate that predicted reliability is a finite float in [0.0, 1.0]."""
    if value is None:
        raise RiskValidationError("Predicted reliability cannot be None.")
    try:
        val = float(value)
    except (TypeError, ValueError) as err:
        raise RiskValidationError(f"Invalid reliability value type: {type(value).__name__}") from err

    if math.isnan(val) or math.isinf(val):
        raise RiskValidationError("Predicted reliability cannot be NaN or Infinite.")

    if not (0.0 <= val <= 1.0):
        raise RiskValidationError(f"Predicted reliability must be strictly within [0.0, 1.0], got: {val}")

    return val


def validate_risk_value(value: Any, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Validate that derived risk score is a finite float within [min_val, max_val]."""
    if value is None:
        raise RiskValidationError("Risk score cannot be None.")
    try:
        val = float(value)
    except (TypeError, ValueError) as err:
        raise RiskValidationError(f"Invalid risk score value type: {type(value).__name__}") from err

    if math.isnan(val) or math.isinf(val):
        raise RiskValidationError("Risk score cannot be NaN or Infinite.")

    if not (min_val <= val <= max_val):
        raise RiskValidationError(f"Risk score must be within [{min_val}, {max_val}], got: {val}")

    return val


def validate_horizon(horizon: Any, supported_horizons: Sequence[int] = (1, 3, 5)) -> int:
    """Validate that horizon is an integer belonging to supported horizons."""
    if horizon is None:
        raise RiskValidationError("Horizon cannot be None.")
    try:
        h = int(horizon)
    except (TypeError, ValueError) as err:
        raise RiskValidationError(f"Invalid horizon type: {type(horizon).__name__}") from err

    if h not in supported_horizons:
        raise RiskValidationError(f"Horizon {h} is not supported. Must be one of: {list(supported_horizons)}")

    return h


def validate_aggregation_weights(
    weights: Dict[int, float], available_horizons: Sequence[int]
) -> Dict[int, float]:
    """Validate and filter weights for available horizons, ensuring non-negative values."""
    if not weights:
        raise RiskValidationError("Horizon weights dictionary cannot be empty.")

    filtered_weights: Dict[int, float] = {}
    for h in available_horizons:
        if h in weights:
            w = float(weights[h])
            if w < 0.0:
                raise RiskValidationError(f"Weight for horizon {h} must be non-negative, got: {w}")
            filtered_weights[h] = w

    if not filtered_weights:
        raise RiskValidationError(f"None of the available horizons {available_horizons} exist in configured weights {weights}")

    total_weight = sum(filtered_weights.values())
    if total_weight <= 0.0:
        raise RiskValidationError(f"Sum of available horizon weights must be positive, got: {total_weight}")

    return filtered_weights
