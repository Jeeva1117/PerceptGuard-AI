"""Tests for basic reliability-to-risk transformation."""

import pytest

from src.risk.risk_config import RiskConfig, TransformationConfig
from src.risk.risk_scorer import ReliabilityRiskScorer
from src.risk.risk_validator import RiskValidationError


def test_boundary_transformations():
    """Verify exact transformation at mathematical boundaries (0.0, 1.0)."""
    scorer = ReliabilityRiskScorer()

    # Reliability 1.0 (perfect perception) -> Risk 0.0 (no failure risk)
    assert scorer.transform_reliability_to_risk(1.0) == 0.0

    # Reliability 0.0 (complete failure) -> Risk 1.0 (maximum failure risk)
    assert scorer.transform_reliability_to_risk(0.0) == 1.0


def test_midpoint_and_intermediate_transformations():
    """Verify transformation at midpoint and intermediate values."""
    scorer = ReliabilityRiskScorer()

    # 0.5 -> 0.5
    assert scorer.transform_reliability_to_risk(0.5) == 0.5

    # 0.8 -> 0.2
    assert scorer.transform_reliability_to_risk(0.8) == 0.2

    # 0.25 -> 0.75
    assert scorer.transform_reliability_to_risk(0.25) == 0.75

    # Day 9 typical values (e.g. 0.014574 -> 0.985426)
    assert scorer.transform_reliability_to_risk(0.014574) == 0.985426


def test_inversion_direction_guarantee():
    """Ensure that higher reliability strictly maps to lower risk (monotonic inverse)."""
    scorer = ReliabilityRiskScorer()
    reliabilities = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    risks = [scorer.transform_reliability_to_risk(r) for r in reliabilities]

    for i in range(len(risks) - 1):
        assert risks[i] > risks[i + 1], f"Monotonic inverse failed at {reliabilities[i]} vs {reliabilities[i+1]}"


def test_custom_clipping_bounds():
    """Verify custom min/max risk clipping boundaries."""
    config = RiskConfig(
        transformation=TransformationConfig(min_risk=0.05, max_risk=0.95)
    )
    scorer = ReliabilityRiskScorer(config)

    # 1.0 would normally be 0.0 -> clipped to 0.05
    assert scorer.transform_reliability_to_risk(1.0) == 0.05

    # 0.0 would normally be 1.0 -> clipped to 0.95
    assert scorer.transform_reliability_to_risk(0.0) == 0.95
