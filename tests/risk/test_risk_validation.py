"""Tests for input validation and error handling in Risk Scoring Engine."""

import math
import pytest

from src.risk.risk_scorer import ReliabilityRiskScorer
from src.risk.risk_validator import (
    RiskValidationError,
    validate_aggregation_weights,
    validate_horizon,
    validate_reliability_value,
    validate_risk_value,
)


def test_reject_none_reliability():
    """Reject None reliability input."""
    with pytest.raises(RiskValidationError):
        validate_reliability_value(None)


def test_reject_nan_reliability():
    """Reject NaN reliability input."""
    with pytest.raises(RiskValidationError) as exc_info:
        validate_reliability_value(float("nan"))
    assert "cannot be NaN" in str(exc_info.value)


def test_reject_inf_reliability():
    """Reject Infinite reliability input."""
    with pytest.raises(RiskValidationError) as exc_info:
        validate_reliability_value(float("inf"))
    assert "cannot be NaN or Infinite" in str(exc_info.value)


def test_reject_negative_reliability():
    """Reject negative reliability input."""
    with pytest.raises(RiskValidationError) as exc_info:
        validate_reliability_value(-0.01)
    assert "must be strictly within [0.0, 1.0]" in str(exc_info.value)


def test_reject_greater_than_one_reliability():
    """Reject reliability values exceeding 1.0."""
    with pytest.raises(RiskValidationError) as exc_info:
        validate_reliability_value(1.0001)
    assert "must be strictly within [0.0, 1.0]" in str(exc_info.value)


def test_reject_invalid_horizon():
    """Reject non-integer or unsupported horizon."""
    with pytest.raises(RiskValidationError):
        validate_horizon("not_an_int")

    with pytest.raises(RiskValidationError) as exc_info:
        validate_horizon(2, supported_horizons=(1, 3, 5))
    assert "Horizon 2 is not supported" in str(exc_info.value)


def test_reject_invalid_aggregation_weights():
    """Reject negative weights or weights without available horizons."""
    # Negative weight
    with pytest.raises(RiskValidationError) as exc_info:
        validate_aggregation_weights({1: -0.5, 3: 0.5}, available_horizons=[1, 3])
    assert "must be non-negative" in str(exc_info.value)

    # Empty intersection
    with pytest.raises(RiskValidationError) as exc_info:
        validate_aggregation_weights({1: 0.5}, available_horizons=[3, 5])
    assert "None of the available horizons" in str(exc_info.value)


def test_scorer_rejects_nan_in_transformation():
    """Scorer must fail immediately when asked to transform NaN."""
    scorer = ReliabilityRiskScorer()
    with pytest.raises(RiskValidationError):
        scorer.transform_reliability_to_risk(float("nan"))
