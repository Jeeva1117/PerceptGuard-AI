"""Tests for Early Warning Data Contract Module."""

import math
from pathlib import Path
import pytest

from src.early_warning.early_warning_config import EarlyWarningConfig
from src.early_warning.early_warning_data_contract import (
    EarlyWarningDataContract,
    EarlyWarningDataContractError,
    EarlyWarningInputRecord,
)
from src.early_warning.interfaces import RiskLevel, RiskScore


@pytest.fixture
def valid_record_dict():
    """Sample valid prediction record dictionary matching Day 9 output schema."""
    return {
        "dataset_id": "driving_sample",
        "video_id": "driving_sample",
        "sequence_id": "seq_driving_sample_L5_000001",
        "source_frame_id": "frame_000005.jpg",
        "target_frame_id": "frame_000006.jpg",
        "source_frame_index": 24,
        "target_frame_index": 30,
        "target_timestamp_seconds": 1.0,
        "horizon": 1,
        "predicted_reliability": 0.01457,
        "model_name": "LSTMForecaster",
        "checkpoint_id": "official_lstm_training",
        "inference_time_ms": 1.25,
        "prediction_status": "success",
    }


def test_valid_record_parsing(valid_record_dict):
    """Verify clean instantiation and field derivation."""
    record = EarlyWarningInputRecord.from_dict(valid_record_dict)
    assert record.sequence_id == "seq_driving_sample_L5_000001"
    assert record.horizon == 1
    assert record.predicted_reliability == 0.01457
    assert record.prediction_id == "seq_driving_sample_L5_000001_H1"
    errors = record.validate()
    assert len(errors) == 0


def test_missing_mandatory_fields():
    """Verify missing mandatory fields raise EarlyWarningDataContractError."""
    with pytest.raises(EarlyWarningDataContractError):
        EarlyWarningInputRecord.from_dict({"horizon": 1})  # missing predicted_reliability

    with pytest.raises(EarlyWarningDataContractError):
        EarlyWarningInputRecord.from_dict({"predicted_reliability": 0.5})  # missing horizon


def test_horizon_validation(valid_record_dict):
    """Reject unsupported horizons outside {1, 3, 5}."""
    valid_record_dict["horizon"] = 7
    record = EarlyWarningInputRecord.from_dict(valid_record_dict)
    errors = record.validate(allowed_horizons=(1, 3, 5))
    assert any("horizon 7 not in allowed horizons" in e for e in errors)


def test_reliability_range_validation(valid_record_dict):
    """Reject negative, greater than 1.0, NaN, or Inf reliability values."""
    # Negative
    valid_record_dict["predicted_reliability"] = -0.05
    record = EarlyWarningInputRecord.from_dict(valid_record_dict)
    assert any("must be in [0.0, 1.0]" in e for e in record.validate())

    # Greater than 1.0
    valid_record_dict["predicted_reliability"] = 1.05
    record = EarlyWarningInputRecord.from_dict(valid_record_dict)
    assert any("must be in [0.0, 1.0]" in e for e in record.validate())

    # NaN
    valid_record_dict["predicted_reliability"] = float("nan")
    record = EarlyWarningInputRecord.from_dict(valid_record_dict)
    assert any("cannot be NaN or Inf" in e for e in record.validate())

    # Inf
    valid_record_dict["predicted_reliability"] = float("inf")
    record = EarlyWarningInputRecord.from_dict(valid_record_dict)
    assert any("cannot be NaN or Inf" in e for e in record.validate())


def test_frame_index_chronological_ordering(valid_record_dict):
    """Reject records where target frame index is before or equal to source frame index."""
    valid_record_dict["source_frame_index"] = 30
    valid_record_dict["target_frame_index"] = 24  # backward in time
    record = EarlyWarningInputRecord.from_dict(valid_record_dict)
    errors = record.validate()
    assert any("must be strictly greater than source_frame_index" in e for e in errors)


def test_day9_horizon_prediction_record_compatibility():
    """Verify seamless ingestion of real Day 9 prediction outputs."""
    predictions_file = Path("experiments/forecasting/multi_horizon/official_multi_horizon_run/predictions.json")
    if predictions_file.exists():
        contract = EarlyWarningDataContract()
        records, summary = contract.load_and_validate_file(predictions_file)
        assert len(records) == 414, f"Expected 414 Day 9 predictions, got {len(records)}"
        assert summary.validation_passed is True
        assert summary.invalid_records == 0
        assert summary.horizons_present == [1, 3, 5]
        assert summary.sequences_count == 138

        # Test grouping by sequence
        grouped = EarlyWarningDataContract.group_by_sequence(records)
        assert len(grouped) == 138
        first_seq = list(grouped.keys())[0]
        assert set(grouped[first_seq].keys()) == {1, 3, 5}


def test_batch_validation_summary(valid_record_dict):
    """Verify batch summary calculations."""
    rec1 = EarlyWarningInputRecord.from_dict(valid_record_dict)
    
    invalid_dict = dict(valid_record_dict)
    invalid_dict["predicted_reliability"] = 2.5
    invalid_dict["prediction_id"] = "bad_rec"
    rec2 = EarlyWarningInputRecord.from_dict(invalid_dict)

    contract = EarlyWarningDataContract()
    summary = contract.validate_batch([rec1, rec2])

    assert summary.total_records == 2
    assert summary.valid_records == 1
    assert summary.invalid_records == 1
    assert summary.validation_passed is False
    assert "bad_rec" in summary.errors_by_record


def test_risk_level_enum():
    """Verify RiskLevel enum and string parsing."""
    assert RiskLevel.from_string("low") == RiskLevel.LOW
    assert RiskLevel.from_string("MEDIUM") == RiskLevel.MEDIUM
    assert RiskLevel.from_string("high") == RiskLevel.HIGH
    assert RiskLevel.from_string("Critical") == RiskLevel.CRITICAL
    assert RiskLevel.from_string("UNKNOWN_VALUE") == RiskLevel.UNKNOWN

    assert RiskLevel.LOW.severity_rank < RiskLevel.MEDIUM.severity_rank < RiskLevel.HIGH.severity_rank < RiskLevel.CRITICAL.severity_rank
