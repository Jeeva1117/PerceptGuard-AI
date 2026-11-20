"""Tests for multi-horizon risk score aggregation strategies."""

import pytest

from src.early_warning.early_warning_data_contract import EarlyWarningInputRecord
from src.risk.risk_config import AggregationConfig, RiskConfig
from src.risk.risk_scorer import ReliabilityRiskScorer, RiskScoringError


@pytest.fixture
def sample_records():
    """Create sample multi-horizon predictions with risks: H1=0.10, H3=0.30, H5=0.60."""
    return [
        EarlyWarningInputRecord(
            prediction_id="seq_001_H1",
            sequence_id="seq_001",
            source_frame_id="frame_000005.jpg",
            target_frame_id="frame_000006.jpg",
            horizon=1,
            predicted_reliability=0.90,  # Risk = 0.10
        ),
        EarlyWarningInputRecord(
            prediction_id="seq_001_H3",
            sequence_id="seq_001",
            source_frame_id="frame_000005.jpg",
            target_frame_id="frame_000008.jpg",
            horizon=3,
            predicted_reliability=0.70,  # Risk = 0.30
        ),
        EarlyWarningInputRecord(
            prediction_id="seq_001_H5",
            sequence_id="seq_001",
            source_frame_id="frame_000005.jpg",
            target_frame_id="frame_000010.jpg",
            horizon=5,
            predicted_reliability=0.40,  # Risk = 0.60
        ),
    ]


def test_weighted_linear_aggregation(sample_records):
    """Verify weighted linear aggregation: 0.50*0.10 + 0.35*0.30 + 0.15*0.60 = 0.245."""
    config = RiskConfig(
        aggregation=AggregationConfig(
            strategy="weighted_linear",
            horizon_weights={1: 0.50, 3: 0.35, 5: 0.15},
        )
    )
    scorer = ReliabilityRiskScorer(config)
    agg = scorer.score_multi_horizon(sample_records)

    expected = 0.50 * 0.10 + 0.35 * 0.30 + 0.15 * 0.60
    assert abs(agg.aggregated_risk_score - expected) < 1e-5
    assert agg.aggregation_strategy == "weighted_linear"


def test_worst_case_aggregation(sample_records):
    """Verify worst-case (maximum risk) strategy returns max(0.10, 0.30, 0.60) = 0.60."""
    config = RiskConfig(aggregation=AggregationConfig(strategy="worst_case"))
    scorer = ReliabilityRiskScorer(config)
    agg = scorer.score_multi_horizon(sample_records)

    assert agg.aggregated_risk_score == 0.60
    assert agg.aggregation_strategy == "worst_case"


def test_primary_horizon_aggregation(sample_records):
    """Verify primary horizon strategy returns risk of primary horizon (e.g. H3 = 0.30)."""
    config = RiskConfig(
        aggregation=AggregationConfig(strategy="primary_horizon", primary_horizon=3)
    )
    scorer = ReliabilityRiskScorer(config)
    agg = scorer.score_multi_horizon(sample_records)

    assert agg.aggregated_risk_score == 0.30
    assert agg.aggregation_strategy == "primary_horizon"


def test_missing_horizon_renormalization(sample_records):
    """Verify weight renormalization when a horizon (e.g. H5) is missing."""
    # Only H1 (0.10) and H3 (0.30) available; configured weights 0.50 and 0.35
    records_h1_h3 = sample_records[:2]
    config = RiskConfig(
        aggregation=AggregationConfig(
            strategy="weighted_linear",
            horizon_weights={1: 0.50, 3: 0.35, 5: 0.15},
            missing_horizon_policy="renormalize",
        )
    )
    scorer = ReliabilityRiskScorer(config)
    agg = scorer.score_multi_horizon(records_h1_h3)

    # Renormalized weights: w1 = 0.50/0.85, w3 = 0.35/0.85
    w1 = 0.50 / 0.85
    w3 = 0.35 / 0.85
    expected = w1 * 0.10 + w3 * 0.30
    assert abs(agg.aggregated_risk_score - expected) < 1e-5
    assert set(agg.weights_applied.keys()) == {1, 3}


def test_empty_records_rejection():
    """Reject empty sequence records."""
    scorer = ReliabilityRiskScorer()
    with pytest.raises(RiskScoringError):
        scorer.score_multi_horizon([])
