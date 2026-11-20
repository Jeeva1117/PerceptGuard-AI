"""Unit and integration tests for ThresholdCalibrator (Day 10 Task 3).

Verifies strict split isolation, metric calculation, candidate grid optimization,
single-class zero-inflation detection, and real validation dataset calibration.
"""

from pathlib import Path
import pytest

from src.early_warning.interfaces import RiskLevel
from src.risk.risk_classifier import RiskClassifier
from src.risk.risk_config import ThresholdProfile
from src.risk.threshold_calibrator import (
    CalibrationError,
    CalibrationReport,
    CalibrationSample,
    ThresholdCalibrator,
    ThresholdCandidateEvaluation,
)


class TestThresholdCalibratorBaselines:
    """Test fundamental validation and split isolation."""

    def test_strict_split_isolation_rejects_test(self):
        calibrator = ThresholdCalibrator(split_name="test")
        samples = [CalibrationSample("s1", 1, 0.8, 0.2, True)]
        with pytest.raises(CalibrationError, match="scientific methodology.*Test split"):
            calibrator.calibrate(samples)

    def test_strict_split_isolation_rejects_train(self):
        calibrator = ThresholdCalibrator(split_name="train")
        samples = [CalibrationSample("s1", 1, 0.8, 0.2, True)]
        with pytest.raises(CalibrationError, match="scientific methodology.*Training split"):
            calibrator.calibrate(samples)

    def test_empty_samples_rejected(self):
        calibrator = ThresholdCalibrator(split_name="val")
        with pytest.raises(CalibrationError, match="empty"):
            calibrator.calibrate([])

    def test_invalid_risk_score_in_sample(self):
        with pytest.raises(Exception):
            CalibrationSample("s1", 1, 1.5, 0.2, True)


class TestBalancedCalibration:
    """Test calibration behavior on a synthetic balanced validation distribution."""

    @pytest.fixture
    def balanced_samples(self):
        # 10 samples: 5 failure events (risk high), 5 normal events (risk low)
        samples = []
        # Normal (ground truth reliability = 0.9, risk < 0.3)
        for i in range(5):
            samples.append(
                CalibrationSample(
                    sequence_id=f"norm_{i}",
                    horizon=1,
                    risk_score=0.10 + i * 0.04,  # 0.10 to 0.26
                    ground_truth_reliability=0.90,
                    is_ground_truth_failure=False,
                )
            )
        # Failures (ground truth reliability = 0.1, risk > 0.6)
        for i in range(5):
            samples.append(
                CalibrationSample(
                    sequence_id=f"fail_{i}",
                    horizon=1,
                    risk_score=0.65 + i * 0.06,  # 0.65 to 0.89
                    ground_truth_reliability=0.10,
                    is_ground_truth_failure=True,
                )
            )
        return samples

    def test_candidate_evaluation(self, balanced_samples):
        calibrator = ThresholdCalibrator(split_name="val")
        eval_mid = calibrator.evaluate_threshold(0.50, balanced_samples)
        # All 5 failures should have risk > 0.5 (TP=5)
        # All 5 normal should have risk <= 0.5 (TN=5)
        assert eval_mid.true_positives == 5
        assert eval_mid.false_positives == 0
        assert eval_mid.true_negatives == 5
        assert eval_mid.false_negatives == 0
        assert eval_mid.precision == 1.0
        assert eval_mid.recall == 1.0
        assert eval_mid.f1_score == 1.0
        assert eval_mid.accuracy == 1.0

    def test_balanced_calibrate_yields_calibrated_status(self, balanced_samples):
        calibrator = ThresholdCalibrator(split_name="val")
        report = calibrator.calibrate(balanced_samples)
        assert report.calibration_status == "CALIBRATED"
        assert report.sample_count == 10
        assert "recommended_thresholds" in report.to_dict()
        profile = report.to_threshold_profile()
        assert profile.calibration_status == "CALIBRATED"
        assert profile.low_upper < profile.medium_upper < profile.high_upper


class TestSingleClassZeroInflation:
    """Test calibration when ground-truth failure rate is 100% (zero-inflation).

    In our real dataset, 100% of validation ground-truth reliability targets are 0.0.
    The calibrator must detect this and assign PROVISIONAL status.
    """

    @pytest.fixture
    def zero_inflated_samples(self):
        # 18 samples, all failures (ground-truth reliability = 0.0)
        samples = []
        for i in range(18):
            samples.append(
                CalibrationSample(
                    sequence_id=f"val_seq_{i}",
                    horizon=1,
                    risk_score=0.96 + (i % 5) * 0.005,  # 0.96 to 0.98
                    ground_truth_reliability=0.0,
                    is_ground_truth_failure=True,
                )
            )
        return samples

    def test_detects_single_class_zero_inflation(self, zero_inflated_samples):
        calibrator = ThresholdCalibrator(split_name="val")
        report = calibrator.calibrate(zero_inflated_samples)

        assert report.calibration_status == "PROVISIONAL"
        assert any("single-class" in w.lower() or "provisional" in w.lower() for w in report.warnings)
        profile = report.to_threshold_profile()
        assert profile.calibration_status == "PROVISIONAL"
        # Must retain sensible monotonic thresholds
        assert profile.low_upper < profile.medium_upper < profile.high_upper


class TestRealDatasetValidationCalibration:
    """Integration test: calibrating on the actual PerceptGuard AI validation dataset."""

    def test_load_real_validation_samples_and_calibrate(self):
        labeled_meta_path = Path("datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json")
        predictions_path = Path("experiments/forecasting/multi_horizon/official_multi_horizon_run/predictions.json")

        if not (labeled_meta_path.exists() and predictions_path.exists()):
            pytest.skip("Dataset or predictions artifacts not found")

        calibrator = ThresholdCalibrator(split_name="val")
        samples = calibrator.load_samples_from_dataset(
            labeled_metadata_path=labeled_meta_path,
            predictions_path=predictions_path,
            split="val",
        )

        # 18 sequences * 3 horizons = 54 validation pairs
        assert len(samples) == 54
        assert all(s.horizon in (1, 3, 5) for s in samples)

        # Run calibration
        report = calibrator.calibrate(samples)
        assert isinstance(report, CalibrationReport)
        assert report.sample_count == 54
        assert report.calibration_status == "PROVISIONAL"

        # Check distribution statistics
        assert report.risk_score_distribution["count"] == 54
        assert report.risk_score_distribution["min"] >= 0.90
        assert report.risk_score_distribution["max"] <= 1.0

        # Build profile and plug into classifier
        profile = report.to_threshold_profile()
        classifier = RiskClassifier()
        # Verify classification works seamlessly with the calibrated profile
        res = classifier.classify_score(0.95, profile=profile)
        assert res in (RiskLevel.HIGH, RiskLevel.CRITICAL)
