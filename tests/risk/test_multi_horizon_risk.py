"""Tests for multi-horizon risk scoring behavior."""

import pytest

from src.early_warning.early_warning_data_contract import EarlyWarningInputRecord
from src.risk.risk_config import RiskConfig
from src.risk.risk_scorer import ReliabilityRiskScorer, RiskScoringError
from src.risk.risk_validator import RiskValidationError


@pytest.fixture
def sample_records():
    """Create sample multi-horizon predictions for sequence seq_001."""
    return [
        EarlyWarningInputRecord(
            prediction_id="seq_001_H1",
            sequence_id="seq_001",
            source_frame_id="frame_000005.jpg",
            target_frame_id="frame_000006.jpg",
            horizon=1,
            predicted_reliability=0.90,
            source_frame_index=24,
            target_frame_index=30,
            target_timestamp_seconds=1.0,
        ),
        EarlyWarningInputRecord(
            prediction_id="seq_001_H3",
            sequence_id="seq_001",
            source_frame_id="frame_000005.jpg",
            target_frame_id="frame_000008.jpg",
            horizon=3,
            predicted_reliability=0.70,
            source_frame_index=24,
            target_frame_index=42,
            target_timestamp_seconds=1.4,
        ),
        EarlyWarningInputRecord(
            prediction_id="seq_001_H5",
            sequence_id="seq_001",
            source_frame_id="frame_000005.jpg",
            target_frame_id="frame_000010.jpg",
            horizon=5,
            predicted_reliability=0.40,
            source_frame_index=24,
            target_frame_index=54,
            target_timestamp_seconds=1.8,
        ),
    ]


def test_single_horizon_risk_scoring(sample_records):
    """Verify single horizon risk scoring across H1, H3, H5."""
    scorer = ReliabilityRiskScorer()

    # H1
    h1_risk = scorer.score_single_horizon(sample_records[0])
    assert h1_risk.horizon == 1
    assert h1_risk.predicted_reliability == 0.90
    assert h1_risk.risk_score == 0.10
    assert h1_risk.prediction_id == "seq_001_H1"

    # H3
    h3_risk = scorer.score_single_horizon(sample_records[1])
    assert h3_risk.horizon == 3
    assert h3_risk.predicted_reliability == 0.70
    assert h3_risk.risk_score == 0.30

    # H5
    h5_risk = scorer.score_single_horizon(sample_records[2])
    assert h5_risk.horizon == 5
    assert h5_risk.predicted_reliability == 0.40
    assert h5_risk.risk_score == 0.60


def test_multi_horizon_scoring_and_details(sample_records):
    """Verify scoring a full sequence with H1, H3, H5."""
    scorer = ReliabilityRiskScorer()
    agg = scorer.score_multi_horizon(sample_records)

    assert agg.sequence_id == "seq_001"
    assert agg.horizon_risk_scores[1] == 0.10
    assert agg.horizon_risk_scores[3] == 0.30
    assert agg.horizon_risk_scores[5] == 0.60
    assert agg.horizon_reliabilities[1] == 0.90
    assert agg.horizon_reliabilities[3] == 0.70
    assert agg.horizon_reliabilities[5] == 0.40
    assert agg.is_calibrated_probability is False


def test_sequence_id_mismatch_rejection(sample_records):
    """Reject multi-horizon scoring if records belong to differing sequences."""
    sample_records[2].sequence_id = "seq_DIFFERENT"
    scorer = ReliabilityRiskScorer()
    with pytest.raises(RiskScoringError) as exc_info:
        scorer.score_multi_horizon(sample_records)
    assert "mismatched sequence IDs" in str(exc_info.value)


def test_unsupported_horizon_rejection(sample_records):
    """Reject single horizon scoring if horizon is not in {1, 3, 5}."""
    sample_records[0].horizon = 7
    scorer = ReliabilityRiskScorer()
    with pytest.raises(RiskValidationError) as exc_info:
        scorer.score_single_horizon(sample_records[0])
    assert "Horizon 7 is not supported" in str(exc_info.value)
