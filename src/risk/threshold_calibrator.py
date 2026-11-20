"""Threshold Calibration Engine for PerceptGuard AI.

Analyzes validation dataset risk score distributions and ground-truth reliability
outcomes to evaluate candidate cutoffs, compute classification metrics, and derive
empirically grounded or provisional threshold profiles under strict leakage prevention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.early_warning.early_warning_data_contract import EarlyWarningInputRecord
from src.risk.risk_config import ThresholdProfile
from src.risk.risk_scorer import ReliabilityRiskScorer
from src.risk.risk_validator import RiskValidationError, validate_risk_value


class CalibrationError(ValueError):
    """Raised when calibration data is invalid, missing, or violates split integrity."""
    pass


@dataclass
class CalibrationSample:
    """Paired risk score and ground-truth reliability outcome for calibration."""
    sequence_id: str
    horizon: int
    risk_score: float
    ground_truth_reliability: float
    is_ground_truth_failure: bool  # True if ground truth reliability < failure_threshold (e.g. < 0.50)

    def __post_init__(self):
        validate_risk_value(self.risk_score)
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdCandidateEvaluation:
    """Performance evaluation of a candidate threshold cutoff."""
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationReport:
    """Comprehensive threshold calibration analysis and diagnostic findings."""
    dataset_id: str
    split_name: str
    sample_count: int
    horizon: Optional[int]
    calibration_status: str  # "CALIBRATED", "PROVISIONAL", "NOT_CALIBRATED", "BLOCKED"
    risk_score_distribution: Dict[str, float]
    ground_truth_distribution: Dict[str, Any]
    recommended_thresholds: Dict[str, float]
    candidate_evaluations: List[ThresholdCandidateEvaluation]
    limitation_statement: str
    warnings: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["candidate_evaluations"] = [e.to_dict() for e in self.candidate_evaluations]
        return d

    def to_threshold_profile(self) -> ThresholdProfile:
        """Convert report to deployable ThresholdProfile."""
        th = self.recommended_thresholds
        return ThresholdProfile(
            low_upper=th.get("low_upper", 0.30),
            medium_upper=th.get("medium_upper", 0.60),
            high_upper=th.get("high_upper", 0.85),
            horizon=self.horizon,
            calibration_status=self.calibration_status,
            calibration_dataset_id=self.dataset_id,
            threshold_version="1.0.0-provisional" if self.calibration_status == "PROVISIONAL" else "1.0.0-calibrated",
            notes=self.limitation_statement,
        )


class ThresholdCalibrator:
    """Empirical threshold calibration engine for perception failure risk classification.

    Strictly uses validation split data to prevent data leakage into test sets.
    Under single-clip zero-inflation where failure outcomes are degenerate,
    honestly flags thresholds as PROVISIONAL rather than claiming production calibration.
    """

    def __init__(
        self,
        split_name: str = "val",
        failure_threshold: float = 0.50,
        scorer: Optional[ReliabilityRiskScorer] = None,
    ):
        self.split_name = split_name
        self.failure_threshold = failure_threshold
        self.scorer = scorer or ReliabilityRiskScorer()

    def build_samples_from_validation_manifest(
        self,
        predictions: Sequence[Union[EarlyWarningInputRecord, Dict[str, Any]]],
        labeled_metadata_path: Union[str, Path],
        target_split: str = "val",
    ) -> List[CalibrationSample]:
        """Align prediction risk scores with ground-truth reliability labels on validation split.

        Parameters
        ----------
        predictions : Sequence
            Input predictions across horizons.
        labeled_metadata_path : Union[str, Path]
            Path to sequence_metadata_labeled.json containing ground-truth targets.
        target_split : str
            Must be 'val'. Calibration on 'test' or 'train' is strictly forbidden.
        """
        # Critical Leakage Gate: Test/Train data cannot be used for calibration!
        if target_split.lower() in ["test", "testing"]:
            raise CalibrationError(
                "Violation of scientific methodology: Test split cannot be used for threshold calibration!"
            )
        if target_split.lower() in ["train", "training"]:
            raise CalibrationError(
                "Violation of scientific methodology: Training split cannot be used for threshold calibration!"
            )
        if target_split.lower() != "val":
            raise CalibrationError(f"Calibration split must be 'val', got '{target_split}'")

        p = Path(labeled_metadata_path)
        if not p.exists():
            raise FileNotFoundError(f"Labeled metadata file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Build map of (sequence_id, horizon) -> ground_truth_label
        gt_map: Dict[Tuple[str, int], float] = {}
        for seq in metadata.get("sequences", []):
            if seq.get("split") == target_split and seq.get("input_sequence_length") == 5:
                seq_id = seq["sequence_id"]
                horizons_dict = seq.get("horizons", {})
                for h_key, h_data in horizons_dict.items():
                    if h_data.get("target_label_available"):
                        h_val = int(h_data.get("horizon", 1))
                        gt_map[(seq_id, h_val)] = float(h_data["target_label"])

        samples: List[CalibrationSample] = []
        for raw_pred in predictions:
            if isinstance(raw_pred, dict):
                rec = EarlyWarningInputRecord.from_dict(raw_pred)
            else:
                rec = raw_pred

            key = (rec.sequence_id, rec.horizon)
            if key in gt_map:
                gt_reliability = gt_map[key]
                risk = self.scorer.transform_reliability_to_risk(rec.predicted_reliability)
                is_failure = (gt_reliability < self.failure_threshold)

                samples.append(
                    CalibrationSample(
                        sequence_id=rec.sequence_id,
                        horizon=rec.horizon,
                        risk_score=risk,
                        ground_truth_reliability=gt_reliability,
                        is_ground_truth_failure=is_failure,
                    )
                )

        return samples

    def load_samples_from_dataset(
        self,
        labeled_metadata_path: Union[str, Path],
        predictions_path: Union[str, Path],
        split: str = "val",
    ) -> List[CalibrationSample]:
        """Load and align validation samples from disk artifacts."""
        p_path = Path(predictions_path)
        if not p_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {p_path}")

        with open(p_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        preds = raw_data.get("predictions", raw_data) if isinstance(raw_data, dict) else raw_data
        return self.build_samples_from_validation_manifest(
            predictions=preds,
            labeled_metadata_path=labeled_metadata_path,
            target_split=split,
        )

    def compute_distribution_stats(self, values: Sequence[float]) -> Dict[str, float]:
        """Compute comprehensive statistical metrics over a sequence of float values."""
        if not values:
            return {
                "count": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "std": 0.0,
            }

        arr = np.array(values, dtype=np.float64)
        return {
            "count": float(len(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "std": float(np.std(arr)),
        }

    def evaluate_candidate_threshold(
        self, samples: Sequence[CalibrationSample], threshold: float
    ) -> ThresholdCandidateEvaluation:
        """Evaluate binary classification metrics for a candidate risk threshold.

        Predicts Failure if risk_score >= threshold.
        Ground truth failure is is_ground_truth_failure == True.
        """
        tp = sum(1 for s in samples if s.risk_score >= threshold and s.is_ground_truth_failure)
        fp = sum(1 for s in samples if s.risk_score >= threshold and not s.is_ground_truth_failure)
        fn = sum(1 for s in samples if s.risk_score < threshold and s.is_ground_truth_failure)
        tn = sum(1 for s in samples if s.risk_score < threshold and not s.is_ground_truth_failure)

        total = len(samples)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return ThresholdCandidateEvaluation(
            threshold=threshold,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            accuracy=float(accuracy),
        )

    # Alias for convenience
    evaluate_threshold = lambda self, threshold, samples: self.evaluate_candidate_threshold(samples, threshold)

    def calibrate(
        self,
        samples: Sequence[CalibrationSample],
        dataset_id: str = "driving_sample",
        horizon: Optional[int] = None,
        candidate_grid: Optional[Sequence[float]] = None,
    ) -> CalibrationReport:
        """Execute threshold calibration over validation samples.

        Parameters
        ----------
        samples : Sequence[CalibrationSample]
            Paired validation samples.
        dataset_id : str
            Dataset source identifier.
        horizon : Optional[int]
            Horizon filter (1, 3, 5) or None for all horizons.
        candidate_grid : Optional[Sequence[float]]
            Grid of threshold cutoffs to evaluate.
        """
        # Strict validation of split and input sample count
        if self.split_name.lower() in ("test", "testing"):
            raise CalibrationError(
                "Violation of scientific methodology: Test split cannot be used for threshold calibration!"
            )
        if self.split_name.lower() in ("train", "training"):
            raise CalibrationError(
                "Violation of scientific methodology: Training split cannot be used for threshold calibration!"
            )
        if self.split_name.lower() != "val":
            raise CalibrationError(f"Calibration split must be 'val', got '{self.split_name}'")

        if len(samples) == 0:
            raise CalibrationError("Calibration sample set is empty")

        if horizon is not None:
            filtered_samples = [s for s in samples if s.horizon == horizon]
        else:
            filtered_samples = list(samples)

        sample_count = len(filtered_samples)
        if sample_count == 0:
            raise CalibrationError(f"No calibration samples match horizon {horizon}")

        risk_scores = [s.risk_score for s in filtered_samples]
        gt_reliabilities = [s.ground_truth_reliability for s in filtered_samples]
        failures_count = sum(1 for s in filtered_samples if s.is_ground_truth_failure)

        risk_dist = self.compute_distribution_stats(risk_scores)
        zeros_count = sum(1 for r in gt_reliabilities if r == 0.0)
        non_zeros_count = len(gt_reliabilities) - zeros_count
        failure_rate = failures_count / sample_count

        # Evaluate candidate grid
        if candidate_grid is None:
            candidate_grid = [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]

        evaluations: List[ThresholdCandidateEvaluation] = [
            self.evaluate_candidate_threshold(filtered_samples, t) for t in candidate_grid
        ]

        # Check for degenerate zero-inflation (single-class targets)
        is_single_class = (zeros_count == sample_count or non_zeros_count == 0)

        warnings: List[str] = []
        if is_single_class:
            calibration_status = "PROVISIONAL"
            warning_msg = (
                "Single-class validation ground truth detected: 100% failure outcomes (y=0.0). "
                "Empirical multi-class optimization is degenerate. Status is PROVISIONAL."
            )
            warnings.append(warning_msg)
            limitation = (
                "Validation ground-truth targets are 100% zero (single-class ground truth) "
                "due to zero-inflation on the single driving clip. Statistical multi-class optimization "
                "is degenerate; recommended thresholds are classified as PROVISIONAL and must be "
                "calibrated against multi-video BDD100K data."
            )
            recommended_thresholds = {
                "low_upper": 0.30,
                "medium_upper": 0.60,
                "high_upper": 0.85,
            }
        else:
            calibration_status = "CALIBRATED"
            limitation = "Thresholds calibrated against multi-class validation data."
            # Select best F1 threshold for high risk
            best_eval = max(evaluations, key=lambda e: (e.f1_score, e.recall))
            high_th = best_eval.threshold
            med_th = round(max(0.10, high_th * 0.70), 2)
            low_th = round(max(0.05, high_th * 0.35), 2)
            recommended_thresholds = {
                "low_upper": low_th,
                "medium_upper": med_th,
                "high_upper": high_th,
            }

        return CalibrationReport(
            dataset_id=dataset_id,
            split_name="val",
            sample_count=sample_count,
            horizon=horizon,
            calibration_status=calibration_status,
            risk_score_distribution=risk_dist,
            ground_truth_distribution={
                "zeros_count": zeros_count,
                "non_zeros_count": non_zeros_count,
                "failure_rate": failure_rate,
            },
            recommended_thresholds=recommended_thresholds,
            candidate_evaluations=evaluations,
            limitation_statement=limitation,
            warnings=warnings,
        )

    def create_threshold_profile(self, report: CalibrationReport) -> ThresholdProfile:
        """Convert a CalibrationReport into a deployable ThresholdProfile."""
        return report.to_threshold_profile()
