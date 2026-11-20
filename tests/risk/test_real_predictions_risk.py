"""Tests verifying Risk Scoring Engine compatibility with authentic Day 9 predictions."""

from pathlib import Path
import pytest

from src.early_warning.early_warning_data_contract import EarlyWarningDataContract
from src.risk.risk_config import RiskConfig
from src.risk.risk_scorer import ReliabilityRiskScorer


def test_score_all_day9_predictions():
    """Verify that all 414 Day 9 predictions are scored without error."""
    predictions_path = Path("experiments/forecasting/multi_horizon/official_multi_horizon_run/predictions.json")
    if not predictions_path.exists():
        pytest.skip("Day 9 predictions file not found")

    contract = EarlyWarningDataContract()
    records, summary = contract.load_and_validate_file(predictions_path)
    assert len(records) == 414

    scorer = ReliabilityRiskScorer()

    # 1. Test single-horizon scoring for all records
    for rec in records:
        h_risk = scorer.score_single_horizon(rec)
        assert 0.0 <= h_risk.risk_score <= 1.0
        assert h_risk.horizon in [1, 3, 5]
        # Mathematical invariant: risk + reliability == 1.0
        assert abs((h_risk.risk_score + h_risk.predicted_reliability) - 1.0) < 1e-4

    # 2. Test multi-horizon scoring grouped by sequence
    grouped = EarlyWarningDataContract.group_by_sequence(records)
    assert len(grouped) == 138

    history = []
    for seq_id, horizons_map in grouped.items():
        seq_records = list(horizons_map.values())
        agg_risk = scorer.score_multi_horizon(seq_records, history=history)

        assert agg_risk.sequence_id == seq_id
        assert 0.0 <= agg_risk.aggregated_risk_score <= 1.0
        assert set(agg_risk.horizon_risk_scores.keys()) == {1, 3, 5}
        assert agg_risk.is_calibrated_probability is False

        # Add to retrospective history
        history.append(seq_records[0])


def test_early_warning_engine_interface():
    """Verify ReliabilityRiskScorer implements IRiskAssessmentEngine interface."""
    predictions_path = Path("experiments/forecasting/multi_horizon/official_multi_horizon_run/predictions.json")
    if not predictions_path.exists():
        pytest.skip("Day 9 predictions file not found")

    contract = EarlyWarningDataContract()
    records, _ = contract.load_and_validate_file(predictions_path)
    grouped = EarlyWarningDataContract.group_by_sequence(records)

    scorer = ReliabilityRiskScorer()
    first_seq_records = list(list(grouped.values())[0].values())

    risk_score = scorer.evaluate_risk(first_seq_records)
    assert risk_score.sequence_id == first_seq_records[0].sequence_id
    assert 0.0 <= risk_score.overall_risk_score <= 1.0
    assert "H1" in risk_score.horizon_risk_scores
    assert "H3" in risk_score.horizon_risk_scores
    assert "H5" in risk_score.horizon_risk_scores
