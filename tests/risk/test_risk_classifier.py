"""Unit and integration tests for RiskClassifier (Day 10 Task 3).

Verifies deterministic risk classification across tiers, boundary conditions,
independent horizons, invalid/missing/stale input safety, and boundary policies.
"""

import math
import pytest

from src.early_warning.interfaces import RiskLevel
from src.risk.risk_classifier import ClassificationResult, RiskClassifier
from src.risk.risk_config import ClassificationConfig, RiskConfig, ThresholdProfile
from src.risk.risk_models import AggregatedRiskRecord, HorizonRiskRecord
from src.risk.risk_validator import RiskValidationError


class TestRiskClassifierBaselines:
    """Test standard tier classification under default thresholds (0.30, 0.60, 0.85)."""

    @pytest.fixture
    def classifier(self):
        return RiskClassifier()

    def test_low_tier(self, classifier):
        assert classifier.classify_score(0.15) == RiskLevel.LOW
        assert classifier.classify_score(0.0) == RiskLevel.LOW

    def test_medium_tier(self, classifier):
        assert classifier.classify_score(0.45) == RiskLevel.MEDIUM
        assert classifier.classify_score(0.55) == RiskLevel.MEDIUM

    def test_high_tier(self, classifier):
        assert classifier.classify_score(0.70) == RiskLevel.HIGH
        assert classifier.classify_score(0.80) == RiskLevel.HIGH

    def test_critical_tier(self, classifier):
        assert classifier.classify_score(0.90) == RiskLevel.CRITICAL
        assert classifier.classify_score(1.0) == RiskLevel.CRITICAL


class TestBoundaryPolicies:
    """Test exact boundary transitions under closed_right and closed_left policies."""

    def test_closed_right_boundaries(self):
        # Default: closed_right (S <= threshold belongs to lower tier)
        # Low: [0.0, 0.30], Med: (0.30, 0.60], High: (0.60, 0.85], Crit: (0.85, 1.0]
        classifier = RiskClassifier(boundary_policy="closed_right")

        # Boundary 0.30
        assert classifier.classify_score(0.2999) == RiskLevel.LOW
        assert classifier.classify_score(0.3000) == RiskLevel.LOW
        assert classifier.classify_score(0.3001) == RiskLevel.MEDIUM

        # Boundary 0.60
        assert classifier.classify_score(0.5999) == RiskLevel.MEDIUM
        assert classifier.classify_score(0.6000) == RiskLevel.MEDIUM
        assert classifier.classify_score(0.6001) == RiskLevel.HIGH

        # Boundary 0.85
        assert classifier.classify_score(0.8499) == RiskLevel.HIGH
        assert classifier.classify_score(0.8500) == RiskLevel.HIGH
        assert classifier.classify_score(0.8501) == RiskLevel.CRITICAL

    def test_closed_left_boundaries(self):
        # closed_left: S < threshold belongs to lower tier, S >= threshold belongs to higher tier
        # Low: [0.0, 0.30), Med: [0.30, 0.60), High: [0.60, 0.85), Crit: [0.85, 1.0]
        classifier = RiskClassifier(boundary_policy="closed_left")

        # Boundary 0.30
        assert classifier.classify_score(0.2999) == RiskLevel.LOW
        assert classifier.classify_score(0.3000) == RiskLevel.MEDIUM
        assert classifier.classify_score(0.3001) == RiskLevel.MEDIUM

        # Boundary 0.60
        assert classifier.classify_score(0.5999) == RiskLevel.MEDIUM
        assert classifier.classify_score(0.6000) == RiskLevel.HIGH
        assert classifier.classify_score(0.6001) == RiskLevel.HIGH

        # Boundary 0.85
        assert classifier.classify_score(0.8499) == RiskLevel.HIGH
        assert classifier.classify_score(0.8500) == RiskLevel.CRITICAL
        assert classifier.classify_score(0.8501) == RiskLevel.CRITICAL


class TestHorizonSpecificThresholds:
    """Test horizon-specific threshold profiles (H1 vs H3 vs H5)."""

    def test_horizon_threshold_profiles(self):
        # Configure stricter threshold for immediate horizon H1 (0.20, 0.50, 0.75)
        # and more relaxed for distant horizon H5 (0.40, 0.70, 0.90)
        config = ClassificationConfig(
            default_thresholds=ThresholdProfile(0.30, 0.60, 0.85),
            horizon_thresholds={
                1: ThresholdProfile(0.20, 0.50, 0.75, horizon=1, threshold_version="h1-strict-v1"),
                5: ThresholdProfile(0.40, 0.70, 0.90, horizon=5, threshold_version="h5-relaxed-v1"),
            },
        )
        classifier = RiskClassifier(config=config)

        # Score = 0.25
        # Under H1: > 0.20 -> MEDIUM
        assert classifier.classify_score(0.25, horizon=1) == RiskLevel.MEDIUM
        # Under H3 (default): <= 0.30 -> LOW
        assert classifier.classify_score(0.25, horizon=3) == RiskLevel.LOW
        # Under H5: <= 0.40 -> LOW
        assert classifier.classify_score(0.25, horizon=5) == RiskLevel.LOW

        # Score = 0.78
        # Under H1: > 0.75 -> CRITICAL
        assert classifier.classify_score(0.78, horizon=1) == RiskLevel.CRITICAL
        # Under H3 (default): <= 0.85 -> HIGH
        assert classifier.classify_score(0.78, horizon=3) == RiskLevel.HIGH
        # Under H5: <= 0.90 -> HIGH
        assert classifier.classify_score(0.78, horizon=5) == RiskLevel.HIGH


class TestSafetyAndInvalidInputs:
    """Test error handling and safety guarantees (never silently fail or return LOW)."""

    @pytest.fixture
    def classifier(self):
        return RiskClassifier()

    def test_reject_nan(self, classifier):
        with pytest.raises(RiskValidationError, match="NaN or Infinite"):
            classifier.classify_score(float("nan"))

    def test_reject_inf(self, classifier):
        with pytest.raises(RiskValidationError, match="NaN or Infinite"):
            classifier.classify_score(float("inf"))

    def test_reject_negative(self, classifier):
        with pytest.raises(RiskValidationError, match=r"within \[0.0, 1.0\]"):
            classifier.classify_score(-0.01)

    def test_reject_greater_than_one(self, classifier):
        with pytest.raises(RiskValidationError, match=r"within \[0.0, 1.0\]"):
            classifier.classify_score(1.01)

    def test_missing_data_maps_to_unknown(self, classifier):
        """CRITICAL: missing/stale data MUST NEVER map to LOW."""
        res_missing = classifier.classify_missing(sequence_id="seq_miss", horizon=1, reason="No telemetry")
        assert res_missing.risk_level == RiskLevel.UNKNOWN
        assert res_missing.risk_level != RiskLevel.LOW
        assert res_missing.is_missing is True

    def test_stale_data_maps_to_unknown(self, classifier):
        res_stale = classifier.classify_stale(
            sequence_id="seq_stale",
            horizon=3,
            stale_score=0.10,  # Even if numerical score was low, stale data must be UNKNOWN
            staleness_seconds=5.0,
        )
        assert res_stale.risk_level == RiskLevel.UNKNOWN
        assert res_stale.risk_level != RiskLevel.LOW
        assert res_stale.is_stale is True


class TestRecordClassification:
    """Test classification of HorizonRiskRecord and AggregatedRiskRecord."""

    @pytest.fixture
    def classifier(self):
        return RiskClassifier()

    def test_classify_horizon_record(self, classifier):
        record = HorizonRiskRecord(
            prediction_id="pred_h3_1",
            sequence_id="seq_test_001",
            source_frame_id="frame_0010",
            target_frame_id="frame_0013",
            horizon=3,
            predicted_reliability=0.28,
            risk_score=0.72,
            target_timestamp_seconds=0.4333,
        )
        result = classifier.classify_horizon_record(record)
        assert isinstance(result, ClassificationResult)
        assert result.sequence_id == "seq_test_001"
        assert result.horizon == 3
        assert result.risk_score == 0.72
        assert result.risk_level == RiskLevel.HIGH
        assert result.prediction_id == "pred_h3_1"
        assert result.source_frame_id == "frame_0010"
        assert result.target_frame_id == "frame_0013"
        assert result.target_timestamp_seconds == 0.4333
        assert result.calibration_status == "PROVISIONAL"

    def test_classify_invalid_horizon_record(self, classifier):
        invalid_record = HorizonRiskRecord(
            prediction_id="pred_invalid_1",
            sequence_id="seq_test_002",
            source_frame_id="frame_0010",
            target_frame_id="frame_0011",
            horizon=1,
            predicted_reliability=0.85,
            risk_score=0.15,
        )
        # Mark as invalid via attribute
        invalid_record.is_valid = False
        invalid_record.validation_notes = "Missing sensor frame"

        result = classifier.classify_horizon_record(invalid_record)
        assert result.risk_level == RiskLevel.UNKNOWN
        assert result.is_missing is True
        assert "Missing sensor frame" in result.notes

    def test_classify_aggregated_record(self, classifier):
        agg_record = AggregatedRiskRecord(
            sequence_id="seq_agg_001",
            source_frame_id="frame_0010",
            aggregated_risk_score=0.92,
            aggregation_strategy="exponential",
            primary_horizon=1,
            horizon_risk_scores={1: 0.90, 3: 0.92, 5: 0.94},
            horizon_reliabilities={1: 0.10, 3: 0.08, 5: 0.06},
            weights_applied={1: 0.5396, 3: 0.2973, 5: 0.1631},
        )
        result = classifier.classify_aggregated_record(agg_record)
        assert result.sequence_id == "seq_agg_001"
        assert result.horizon == 1
        assert result.risk_score == 0.92
        assert result.risk_level == RiskLevel.CRITICAL

    def test_batch_classification(self, classifier):
        records = [
            HorizonRiskRecord("p1", "s1", "f1", "f2", 1, 0.90, 0.10),
            HorizonRiskRecord("p2", "s1", "f1", "f4", 3, 0.55, 0.45),
            HorizonRiskRecord("p3", "s1", "f1", "f6", 5, 0.12, 0.88),
        ]
        results = classifier.classify_batch(records)
        assert len(results) == 3
        assert results[0].risk_level == RiskLevel.LOW
        assert results[1].risk_level == RiskLevel.MEDIUM
        assert results[2].risk_level == RiskLevel.CRITICAL
