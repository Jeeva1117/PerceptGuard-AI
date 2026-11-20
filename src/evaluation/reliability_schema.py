"""PerceptGuard-AI: Canonical Annotation Schema & Reliability Labeling Specifications.

Senior Computer Vision & ML Reliability Architect-grade specifications, data models,
and contracts for object detection reliability forecasting.

Key Guarantees:
- Normalized, versioned canonical annotation schema (Schema v1.0.0).
- Pure mathematical Intersection over Union (IoU) calculation engine.
- Deterministic class-aware Hungarian and greedy matching specifications.
- Clear demarcation of True Positives, False Positives (unmatched & duplicates),
  and False Negatives (missed ground truth).
- Continuous frame-level reliability formulation R_t in [0.0, 1.0] with documented
  harmonic mean (F1-score) and composite score baselines.
- Rigorous handling of zero-prediction and zero-ground-truth edge cases.
- Multi-horizon temporal target formulation Y_t = [R_{t+1}, R_{t+3}, R_{t+5}]^T.
- Strict contract validation ensuring compatibility with Days 5 & 6 temporal sequences.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

SCHEMA_VERSION = "1.0.0"
RELIABILITY_SCHEMA_VERSION = SCHEMA_VERSION

# ==============================================================================
# ENUMS
# ==============================================================================

class MatchingOutcome(str, Enum):
    """Categorical classification of object detection evaluation outcomes."""
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    IGNORED_CROWD = "ignored_crowd"


class TargetStatus(str, Enum):
    """Status indicator for temporal forecast horizon targets."""
    VALID = "valid"
    GAP_DETECTED = "gap_detected"
    OUT_OF_BOUNDS = "out_of_bounds"
    EMBARGO_PURGED = "embargo_purged"

# ==============================================================================
# DATA MODELS: CANONICAL ANNOTATION & EVALUATION
# ==============================================================================

@dataclass
class CanonicalBoundingBox:
    """Canonical axis-aligned bounding box representation [x_min, y_min, x_max, y_max]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        self.x_min = float(self.x_min)
        self.y_min = float(self.y_min)
        self.x_max = float(self.x_max)
        self.y_max = float(self.y_max)

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.x_min, self.y_min, self.x_max, self.y_max))

    def is_valid(self, canvas_width: float = 640.0, canvas_height: float = 360.0, min_area: float = 1.0) -> bool:
        """Check if coordinates strictly conform to canvas geometric boundaries."""
        if not self.is_finite:
            return False
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            return False
        if self.x_min < 0.0 or self.y_min < 0.0:
            return False
        if self.x_max > canvas_width or self.y_max > canvas_height:
            return False
        if self.area < min_area:
            return False
        return True

    def to_list(self) -> List[float]:
        return [round(self.x_min, 2), round(self.y_min, 2), round(self.x_max, 2), round(self.y_max, 2)]

    def to_dict(self) -> Dict[str, float]:
        return {
            "x_min": round(self.x_min, 2),
            "y_min": round(self.y_min, 2),
            "x_max": round(self.x_max, 2),
            "y_max": round(self.y_max, 2)
        }


@dataclass
class ObjectReliabilityRecord:
    """Evaluation record for a single object-level detection or missed ground truth."""
    frame_id: str
    extracted_frame_id: int
    source_frame_index: int
    timestamp_seconds: float
    detection_index: Optional[int]
    ground_truth_id: Optional[str]
    track_id: Optional[str]
    predicted_class: Optional[str]
    ground_truth_class: Optional[str]
    confidence: Optional[float]
    iou: float
    outcome: MatchingOutcome
    object_correctness: float  # 1.0 for True Positive, 0.0 for False Positive

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "extracted_frame_id": self.extracted_frame_id,
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 4),
            "detection_index": self.detection_index,
            "ground_truth_id": self.ground_truth_id,
            "track_id": self.track_id,
            "predicted_class": self.predicted_class,
            "ground_truth_class": self.ground_truth_class,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "iou": round(self.iou, 4),
            "outcome": self.outcome.value,
            "object_correctness": self.object_correctness
        }


@dataclass
class FrameReliabilityRecord:
    """Comprehensive reliability evaluation for a single video frame."""
    frame_id: str
    extracted_frame_id: int
    source_frame_index: int
    timestamp_seconds: float
    total_predictions: int
    total_ground_truth: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    mean_matched_iou: float
    composite_reliability: float
    primary_reliability: float  # Default: F1-score
    status: str = "valid"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "extracted_frame_id": self.extracted_frame_id,
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 4),
            "total_predictions": self.total_predictions,
            "total_ground_truth": self.total_ground_truth,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "mean_matched_iou": round(self.mean_matched_iou, 4),
            "composite_reliability": round(self.composite_reliability, 4),
            "primary_reliability": round(self.primary_reliability, 4),
            "status": self.status
        }


@dataclass
class HorizonTargetSpec:
    """Forecasting target specification for a single horizon step."""
    horizon_step: int
    target_frame_id: Optional[str]
    target_source_frame_index: Optional[int]
    target_timestamp_seconds: Optional[float]
    target_reliability: Optional[float]
    target_status: TargetStatus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_step": self.horizon_step,
            "target_frame_id": self.target_frame_id,
            "target_source_frame_index": self.target_source_frame_index,
            "target_timestamp_seconds": round(self.target_timestamp_seconds, 4) if self.target_timestamp_seconds is not None else None,
            "target_reliability": round(self.target_reliability, 4) if self.target_reliability is not None else None,
            "target_status": self.target_status.value
        }


@dataclass
class TemporalTargetRecord:
    """Multi-horizon target specification aligned with an input sequence."""
    sequence_id: str
    dataset_id: str
    source_frame_id: str
    source_extracted_frame_id: int
    source_timestamp_seconds: float
    split: str
    horizons: Dict[str, HorizonTargetSpec] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "dataset_id": self.dataset_id,
            "source_frame_id": self.source_frame_id,
            "source_extracted_frame_id": self.source_extracted_frame_id,
            "source_timestamp_seconds": round(self.source_timestamp_seconds, 4),
            "split": self.split,
            "horizons": {k: v.to_dict() for k, v in self.horizons.items()}
        }

# ==============================================================================
# MATHEMATICAL UTILITIES & METRICS
# ==============================================================================

def calculate_iou(box1: Union[CanonicalBoundingBox, Sequence[float]], box2: Union[CanonicalBoundingBox, Sequence[float]]) -> float:
    """Compute exact Intersection over Union (IoU) between two bounding boxes.
    
    Supports CanonicalBoundingBox or 4-element sequence [x1, y1, x2, y2].
    """
    if isinstance(box1, CanonicalBoundingBox):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.x_min, box1.y_min, box1.x_max, box1.y_max
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])

    if isinstance(box2, CanonicalBoundingBox):
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.x_min, box2.y_min, box2.x_max, box2.y_max
    else:
        b2_x1, b2_y1, b2_x2, b2_y2 = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])

    # Compute intersection rectangle
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, b1_x2 - b1_x1) * max(0.0, b1_y2 - b1_y1)
    area2 = max(0.0, b2_x2 - b2_x1) * max(0.0, b2_y2 - b2_y1)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    return min(1.0, max(0.0, inter_area / union_area))


def compute_frame_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    total_predictions: int,
    total_ground_truth: int,
) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and F1-score with documented edge case handling.
    
    Edge Case Policies:
    - Both empty (|B| == 0 and |G| == 0): Precision=1.0, Recall=1.0, F1=1.0.
    - Predictions only (|B| > 0 and |G| == 0): Precision=0.0, Recall=0.0, F1=0.0 (Pure false alarms).
    - Ground truth only (|B| == 0 and |G| > 0): Precision=0.0, Recall=0.0, F1=0.0 (Complete misses).
    - Zero matches (TP == 0): Precision=0.0, Recall=0.0, F1=0.0.
    """
    if total_predictions == 0 and total_ground_truth == 0:
        return 1.0, 1.0, 1.0

    if total_predictions > 0 and total_ground_truth == 0:
        return 0.0, 0.0, 0.0

    if total_predictions == 0 and total_ground_truth > 0:
        return 0.0, 0.0, 0.0

    if true_positives == 0:
        return 0.0, 0.0, 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return min(1.0, max(0.0, precision)), min(1.0, max(0.0, recall)), min(1.0, max(0.0, f1))


def compute_composite_reliability(
    f1_score: float,
    mean_matched_iou: float,
    alpha: float = 0.70,
) -> float:
    """Compute composite reliability score combining detection harmonic mean and spatial IoU.
    
    R_t = alpha * F1 + (1 - alpha) * Mean_IoU
    """
    alpha = max(0.0, min(1.0, alpha))
    composite = alpha * f1_score + (1.0 - alpha) * mean_matched_iou
    return min(1.0, max(0.0, composite))

# ==============================================================================
# DATA CONTRACT VALIDATOR
# ==============================================================================

class ReliabilityDataContract:
    """Verifies that evaluation schemas strictly adhere to PerceptGuard data contracts."""

    @staticmethod
    def validate_ground_truth_schema(gt_dict: Dict[str, Any]) -> bool:
        """Verify ground-truth annotations JSON structure."""
        required_top = {"schema_version", "sequence_id", "dataset_id", "frames"}
        if not required_top.issubset(gt_dict.keys()):
            return False
        if not isinstance(gt_dict["frames"], list) or len(gt_dict["frames"]) == 0:
            return False
        f0 = gt_dict["frames"][0]
        required_frame = {"frame_index", "extracted_frame_id", "frame_filename", "annotations"}
        return required_frame.issubset(f0.keys())

    @staticmethod
    def validate_detection_schema(det_dict: Dict[str, Any]) -> bool:
        """Verify YOLO detection results JSON structure."""
        required_top = {"schema_version", "dataset_id", "total_frames", "frame_results"}
        if not required_top.issubset(det_dict.keys()):
            return False
        if not isinstance(det_dict["frame_results"], list) or len(det_dict["frame_results"]) == 0:
            return False
        fr0 = det_dict["frame_results"][0]
        required_frame = {"image_filename", "extracted_frame_id", "detections"}
        return required_frame.issubset(fr0.keys())

    @staticmethod
    def validate_frame_reliability_schema(frame_dict: Dict[str, Any]) -> bool:
        """Verify FrameReliabilityRecord schema compliance."""
        required = {
            "frame_id", "extracted_frame_id", "timestamp_seconds",
            "true_positives", "false_positives", "false_negatives",
            "precision", "recall", "f1_score", "primary_reliability"
        }
        if not required.issubset(frame_dict.keys()):
            return False
        # Value bounds verification
        f1 = frame_dict.get("f1_score", -1.0)
        return 0.0 <= f1 <= 1.0

    @staticmethod
    def validate_temporal_target_schema(target_dict: Dict[str, Any]) -> bool:
        """Verify TemporalTargetRecord schema compliance."""
        required = {"sequence_id", "source_frame_id", "split", "horizons"}
        if not required.issubset(target_dict.keys()):
            return False
        horizons = target_dict.get("horizons", {})
        for h in ("h1", "h3", "h5"):
            if h in horizons:
                h_rec = horizons[h]
                if not {"horizon_step", "target_status"}.issubset(h_rec.keys()):
                    return False
        return True
