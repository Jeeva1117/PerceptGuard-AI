"""PerceptGuard AI: Advanced YOLO Detection Metrics & Perception Analysis Engine.

Senior Staff-level, production-grade metrics and analysis engine for autonomous
driving computer vision reliability evaluation.

Key Capabilities:
- Ingests and validates versioned Task 4 batch detection results (Schema v1.0).
- Global detection volume, frame completion, and distribution statistics.
- Parametric confidence profiling: mean, median, min, max, standard deviation,
  histogram distributions, and low-confidence percentage.
- Semantic class-wise metrics breakdown: frequencies, class distributions, and
  per-class confidence bounds.
- Frame-level evaluation: timestamps, detection volumes, latency telemetry,
  and heuristic reliability indicators.
- Sequence temporal consistency analysis: frame-to-frame delta variations,
  volatility metrics, and temporal stability scoring without tracking assumptions.
- Rigorous ground-truth evaluation: IoU computation, greedy bipartite matching,
  TP/FP/FN accounting, precision, recall, and F1-score when ground-truth annotations
  are provided. Explicitly reports unavailable status when annotations are absent.
- Transparent heuristic detection reliability indicator: combines confidence,
  temporal stability, and processing health with documented formulas.
- Versioned multi-file atomic JSON persistence (Schema v1.0).
- Pure analytical decoupling: zero model weights, zero PyTorch inference passes.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger("PerceptGuard.DetectionMetrics")


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class DetectionMetricsError(Exception):
    """Base exception for all detection metrics and analysis errors."""
    pass


class InvalidDetectionDataError(DetectionMetricsError):
    """Raised when detection JSON fails structural or semantic validation."""
    pass


class UnsupportedSchemaError(DetectionMetricsError):
    """Raised when input detection JSON schema version is unsupported."""
    pass


class GroundTruthEvaluationError(DetectionMetricsError):
    """Raised when ground-truth annotations are malformed or incompatible."""
    pass


class MetricsPersistenceError(DetectionMetricsError):
    """Raised when persisting metrics artifacts to disk fails."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass
class ClassMetricRecord:
    """Statistical evaluation metrics for an individual object class."""
    class_id: int
    class_name: str
    detection_count: int
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    percentage_of_total_detections: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "detection_count": int(self.detection_count),
            "mean_confidence": round(float(self.mean_confidence), 4),
            "min_confidence": round(float(self.min_confidence), 4),
            "max_confidence": round(float(self.max_confidence), 4),
            "percentage_of_total_detections": round(float(self.percentage_of_total_detections), 2),
        }


@dataclass
class FrameMetricRecord:
    """Per-frame metrics, temporal deltas, and heuristic reliability telemetry."""
    frame_identifier: str
    extracted_frame_id: Optional[int] = None
    source_frame_index: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    detection_count: int = 0
    mean_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    confidence_std_dev: Optional[float] = None
    processing_duration_ms: float = 0.0
    processing_status: str = "success"
    heuristic_reliability: float = 0.0
    detection_count_delta: Optional[int] = None
    mean_confidence_delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_identifier": self.frame_identifier,
            "extracted_frame_id": self.extracted_frame_id,
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": (
                round(float(self.timestamp_seconds), 4)
                if self.timestamp_seconds is not None else None
            ),
            "detection_count": int(self.detection_count),
            "mean_confidence": (
                round(float(self.mean_confidence), 4)
                if self.mean_confidence is not None else None
            ),
            "max_confidence": (
                round(float(self.max_confidence), 4)
                if self.max_confidence is not None else None
            ),
            "min_confidence": (
                round(float(self.min_confidence), 4)
                if self.min_confidence is not None else None
            ),
            "confidence_std_dev": (
                round(float(self.confidence_std_dev), 4)
                if self.confidence_std_dev is not None else None
            ),
            "processing_duration_ms": round(float(self.processing_duration_ms), 2),
            "processing_status": self.processing_status,
            "heuristic_reliability": round(float(self.heuristic_reliability), 4),
            "detection_count_delta": self.detection_count_delta,
            "mean_confidence_delta": (
                round(float(self.mean_confidence_delta), 4)
                if self.mean_confidence_delta is not None else None
            ),
        }


@dataclass
class TemporalConsistencyMetrics:
    """Metrics quantifying sequence-level frequency stability across chronological frames."""
    mean_detection_count_variation: float
    max_detection_count_variation: int
    detection_count_variance: float
    mean_confidence_variation: float
    stability_score: float
    consecutive_zero_detection_max: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_detection_count_variation": round(float(self.mean_detection_count_variation), 4),
            "max_detection_count_variation": int(self.max_detection_count_variation),
            "detection_count_variance": round(float(self.detection_count_variance), 4),
            "mean_confidence_variation": round(float(self.mean_confidence_variation), 4),
            "stability_score": round(float(self.stability_score), 4),
            "consecutive_zero_detection_max": int(self.consecutive_zero_detection_max),
        }


@dataclass
class GroundTruthEvaluationResult:
    """Outcomes from matching predictions against labeled ground-truth annotations."""
    status: str  # "available" or "unavailable"
    iou_threshold: float = 0.50
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mean_matched_iou: Optional[float] = None
    class_evaluations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    disclaimer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "iou_threshold": round(float(self.iou_threshold), 4),
            "true_positives": int(self.true_positives),
            "false_positives": int(self.false_positives),
            "false_negatives": int(self.false_negatives),
            "precision": round(float(self.precision), 4) if self.precision is not None else None,
            "recall": round(float(self.recall), 4) if self.recall is not None else None,
            "f1_score": round(float(self.f1_score), 4) if self.f1_score is not None else None,
            "mean_matched_iou": (
                round(float(self.mean_matched_iou), 4)
                if self.mean_matched_iou is not None else None
            ),
            "class_evaluations": self.class_evaluations,
            "disclaimer": self.disclaimer,
        }


@dataclass
class MetricsAnalysisSummary:
    """High-level summary of detection metrics and perception stability."""
    schema_version: str = "1.0"
    run_id: str = ""
    dataset_id: str = ""
    detection_run_id: str = ""
    total_frames_processed: int = 0
    successful_frames: int = 0
    failed_frames: int = 0
    frames_with_detections: int = 0
    frames_without_detections: int = 0
    total_detections: int = 0
    average_detections_per_frame: float = 0.0
    min_detections_per_frame: int = 0
    max_detections_per_frame: int = 0
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    confidence_std_dev: float = 0.0
    low_confidence_count: int = 0
    low_confidence_percentage: float = 0.0
    temporal_stability_score: float = 0.0
    mean_heuristic_reliability: float = 0.0
    ground_truth_status: str = "unavailable"
    final_status: str = "PASS"  # "PASS", "WARNING", "FAIL"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "detection_run_id": self.detection_run_id,
            "total_frames_processed": int(self.total_frames_processed),
            "successful_frames": int(self.successful_frames),
            "failed_frames": int(self.failed_frames),
            "frames_with_detections": int(self.frames_with_detections),
            "frames_without_detections": int(self.frames_without_detections),
            "total_detections": int(self.total_detections),
            "average_detections_per_frame": round(float(self.average_detections_per_frame), 2),
            "min_detections_per_frame": int(self.min_detections_per_frame),
            "max_detections_per_frame": int(self.max_detections_per_frame),
            "mean_confidence": round(float(self.mean_confidence), 4),
            "median_confidence": round(float(self.median_confidence), 4),
            "min_confidence": round(float(self.min_confidence), 4),
            "max_confidence": round(float(self.max_confidence), 4),
            "confidence_std_dev": round(float(self.confidence_std_dev), 4),
            "low_confidence_count": int(self.low_confidence_count),
            "low_confidence_percentage": round(float(self.low_confidence_percentage), 2),
            "temporal_stability_score": round(float(self.temporal_stability_score), 4),
            "mean_heuristic_reliability": round(float(self.mean_heuristic_reliability), 4),
            "ground_truth_status": self.ground_truth_status,
            "final_status": self.final_status,
            "generated_at": self.generated_at,
        }


# ==============================================================================
# Geometric & Matching Utilities
# ==============================================================================

def compute_box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate Intersection over Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


# ==============================================================================
# DetectionMetrics Engine
# ==============================================================================

class DetectionMetrics:
    """Production-grade metrics and analysis engine for object detection pipelines."""

    SUPPORTED_SCHEMA_VERSIONS: Set[str] = {"1.0"}

    def __init__(
        self,
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        project_root: Optional[Path] = None,
    ) -> None:
        """Initialize DetectionMetrics engine.
        
        Args:
            confidence_threshold: Cutoff for categorizing low-confidence detections.
            iou_threshold: Minimum IoU overlap required for ground-truth matching.
            project_root: Absolute root path of PerceptGuard AI repository.
        """
        self.confidence_threshold = max(0.0, min(confidence_threshold, 1.0))
        self.iou_threshold = max(0.01, min(iou_threshold, 1.0))
        self.project_root = project_root or _project_root

    # --------------------------------------------------------------------------
    # Detection JSON Loading & Validation
    # --------------------------------------------------------------------------

    def load_and_validate_input(
        self,
        input_file: Union[str, Path],
    ) -> Dict[str, Any]:
        """Load and structurally validate Task 4 detection JSON file.
        
        Args:
            input_file: Path to detection_results.json.
            
        Returns:
            Validated detection data dictionary.
            
        Raises:
            InvalidDetectionDataError: On missing file, invalid JSON, or missing schema fields.
            UnsupportedSchemaError: If schema_version is unsupported.
        """
        path = Path(input_file)
        if not path.is_file():
            raise InvalidDetectionDataError(f"Detection results file not found: '{path}'")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise InvalidDetectionDataError(f"Malformed JSON syntax in '{path}': {e}") from e
        except Exception as e:
            raise InvalidDetectionDataError(f"Failed to read file '{path}': {e}") from e

        if not isinstance(data, dict):
            raise InvalidDetectionDataError(f"Expected root JSON object in '{path}', got {type(data).__name__}")

        schema_ver = data.get("schema_version")
        if not schema_ver:
            raise InvalidDetectionDataError(f"Missing required 'schema_version' key in '{path}'")
        if str(schema_ver) not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedSchemaError(
                f"Unsupported schema version '{schema_ver}' in '{path}'. "
                f"Supported: {sorted(self.SUPPORTED_SCHEMA_VERSIONS)}"
            )

        for required_key in ["run_id", "dataset_id", "frame_results"]:
            if required_key not in data:
                raise InvalidDetectionDataError(f"Missing required field '{required_key}' in '{path}'")

        if not isinstance(data["frame_results"], list):
            raise InvalidDetectionDataError(
                f"'frame_results' must be a list in '{path}', got {type(data['frame_results']).__name__}"
            )

        return data

    # --------------------------------------------------------------------------
    # Statistical & Confidence Analysis
    # --------------------------------------------------------------------------

    def compute_detection_statistics(
        self,
        frame_results: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute dataset-level detection counts, frame completions, and volume metrics."""
        total_frames = len(frame_results)
        successful_frames = 0
        failed_frames = 0
        frames_with_detections = 0
        frames_without_detections = 0
        counts_per_frame: List[int] = []

        for frame in frame_results:
            status = frame.get("processing_status", "success")
            det_count = int(frame.get("detection_count", 0))

            if status == "success":
                successful_frames += 1
                counts_per_frame.append(det_count)
                if det_count > 0:
                    frames_with_detections += 1
                else:
                    frames_without_detections += 1
            else:
                failed_frames += 1

        total_detections = sum(counts_per_frame)
        avg_det = (total_detections / successful_frames) if successful_frames > 0 else 0.0
        min_det = min(counts_per_frame) if counts_per_frame else 0
        max_det = max(counts_per_frame) if counts_per_frame else 0

        return {
            "total_frames_processed": total_frames,
            "successful_frames": successful_frames,
            "failed_frames": failed_frames,
            "frames_with_detections": frames_with_detections,
            "frames_without_detections": frames_without_detections,
            "total_detections": total_detections,
            "average_detections_per_frame": round(avg_det, 2),
            "min_detections_per_frame": min_det,
            "max_detections_per_frame": max_det,
        }

    def compute_confidence_analysis(
        self,
        frame_results: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate statistical distributions and confidence metrics across all detections."""
        confidences: List[float] = []

        for frame in frame_results:
            if frame.get("processing_status") != "success":
                continue
            for det in frame.get("detections", []):
                conf = det.get("confidence")
                if conf is not None:
                    try:
                        c_val = float(conf)
                        if math.isfinite(c_val) and 0.0 <= c_val <= 1.0:
                            confidences.append(c_val)
                    except (ValueError, TypeError):
                        continue

        if not confidences:
            return {
                "mean_confidence": 0.0,
                "median_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "confidence_std_dev": 0.0,
                "confidence_distribution": {
                    "0.00-0.20": 0,
                    "0.20-0.40": 0,
                    "0.40-0.60": 0,
                    "0.60-0.80": 0,
                    "0.80-1.00": 0,
                },
                "low_confidence_threshold": self.confidence_threshold,
                "low_confidence_count": 0,
                "low_confidence_percentage": 0.0,
            }

        mean_c = statistics.mean(confidences)
        median_c = statistics.median(confidences)
        min_c = min(confidences)
        max_c = max(confidences)
        std_c = statistics.stdev(confidences) if len(confidences) > 1 else 0.0

        # Histogram distribution across 5 standard intervals
        bins = {"0.00-0.20": 0, "0.20-0.40": 0, "0.40-0.60": 0, "0.60-0.80": 0, "0.80-1.00": 0}
        low_conf_count = 0

        for c in confidences:
            if c < self.confidence_threshold:
                low_conf_count += 1

            if c < 0.20:
                bins["0.00-0.20"] += 1
            elif c < 0.40:
                bins["0.20-0.40"] += 1
            elif c < 0.60:
                bins["0.40-0.60"] += 1
            elif c < 0.80:
                bins["0.60-0.80"] += 1
            else:
                bins["0.80-1.00"] += 1

        low_conf_pct = (low_conf_count / len(confidences)) * 100.0

        return {
            "mean_confidence": round(mean_c, 4),
            "median_confidence": round(median_c, 4),
            "min_confidence": round(min_c, 4),
            "max_confidence": round(max_c, 4),
            "confidence_std_dev": round(std_c, 4),
            "confidence_distribution": bins,
            "low_confidence_threshold": self.confidence_threshold,
            "low_confidence_count": low_conf_count,
            "low_confidence_percentage": round(low_conf_pct, 2),
        }

    # --------------------------------------------------------------------------
    # Class-Wise Breakdown
    # --------------------------------------------------------------------------

    def compute_class_metrics(
        self,
        frame_results: Sequence[Dict[str, Any]],
    ) -> Dict[str, ClassMetricRecord]:
        """Compute per-class detection counts, frequency shares, and confidence ranges."""
        class_buckets: Dict[str, Dict[str, Any]] = {}
        total_valid_detections = 0

        for frame in frame_results:
            if frame.get("processing_status") != "success":
                continue
            for det in frame.get("detections", []):
                cid = det.get("class_id", -1)
                cname = str(det.get("class_name", f"class_{cid}")).strip()
                conf = det.get("confidence")

                try:
                    conf_val = float(conf)
                    if not (math.isfinite(conf_val) and 0.0 <= conf_val <= 1.0):
                        continue
                except (ValueError, TypeError):
                    continue

                if cname not in class_buckets:
                    class_buckets[cname] = {
                        "class_id": cid,
                        "class_name": cname,
                        "confidences": [],
                    }
                class_buckets[cname]["confidences"].append(conf_val)
                total_valid_detections += 1

        results: Dict[str, ClassMetricRecord] = {}
        for cname, data in class_buckets.items():
            confs = data["confidences"]
            count = len(confs)
            mean_c = statistics.mean(confs) if confs else 0.0
            min_c = min(confs) if confs else 0.0
            max_c = max(confs) if confs else 0.0
            pct = (count / total_valid_detections * 100.0) if total_valid_detections > 0 else 0.0

            results[cname] = ClassMetricRecord(
                class_id=data["class_id"],
                class_name=cname,
                detection_count=count,
                mean_confidence=mean_c,
                min_confidence=min_c,
                max_confidence=max_c,
                percentage_of_total_detections=pct,
            )

        return results

    # --------------------------------------------------------------------------
    # Frame-Level & Temporal Consistency Analysis
    # --------------------------------------------------------------------------

    def compute_frame_and_temporal_metrics(
        self,
        frame_results: Sequence[Dict[str, Any]],
    ) -> Tuple[List[FrameMetricRecord], TemporalConsistencyMetrics]:
        """Evaluate frame-level metrics, rolling differences, and temporal stability."""
        frame_records: List[FrameMetricRecord] = []
        count_diffs: List[int] = []
        conf_diffs: List[float] = []

        prev_count: Optional[int] = None
        prev_mean_conf: Optional[float] = None
        current_zero_streak = 0
        max_zero_streak = 0

        for frame in frame_results:
            filename = frame.get("image_filename", "unknown_frame")
            ext_id = frame.get("extracted_frame_id")
            src_idx = frame.get("source_frame_index")
            ts = frame.get("timestamp_seconds")
            status = frame.get("processing_status", "success")
            dur_ms = float(frame.get("inference_time_ms", 0.0))

            det_count = int(frame.get("detection_count", 0))
            detections = frame.get("detections", [])

            # Extract valid frame confidences
            frame_confs: List[float] = []
            for d in detections:
                c = d.get("confidence")
                if c is not None:
                    try:
                        c_val = float(c)
                        if math.isfinite(c_val) and 0.0 <= c_val <= 1.0:
                            frame_confs.append(c_val)
                    except (ValueError, TypeError):
                        continue

            if frame_confs:
                mean_conf = statistics.mean(frame_confs)
                max_conf = max(frame_confs)
                min_conf = min(frame_confs)
                std_conf = statistics.stdev(frame_confs) if len(frame_confs) > 1 else 0.0
                current_zero_streak = 0
            else:
                mean_conf = None
                max_conf = None
                min_conf = None
                std_conf = None
                if status == "success":
                    current_zero_streak += 1
                    if current_zero_streak > max_zero_streak:
                        max_zero_streak = current_zero_streak

            # Temporal deltas
            count_delta: Optional[int] = None
            conf_delta: Optional[float] = None

            if prev_count is not None:
                count_delta = det_count - prev_count
                count_diffs.append(abs(count_delta))

            if prev_mean_conf is not None and mean_conf is not None:
                conf_delta = mean_conf - prev_mean_conf
                conf_diffs.append(abs(conf_delta))

            # Heuristic detection reliability indicator (0.0 to 1.0)
            # Combines: base confidence, low-confidence penalty, and execution health
            # Note: Explicitly documented as a heuristic signal, NOT ground-truth accuracy.
            reliability = self._calculate_heuristic_reliability(
                status=status,
                det_count=det_count,
                mean_conf=mean_conf,
                confidences=frame_confs,
                count_delta=count_delta,
            )

            record = FrameMetricRecord(
                frame_identifier=filename,
                extracted_frame_id=ext_id,
                source_frame_index=src_idx,
                timestamp_seconds=ts,
                detection_count=det_count,
                mean_confidence=mean_conf,
                max_confidence=max_conf,
                min_confidence=min_conf,
                confidence_std_dev=std_conf,
                processing_duration_ms=dur_ms,
                processing_status=status,
                heuristic_reliability=reliability,
                detection_count_delta=count_delta,
                mean_confidence_delta=conf_delta,
            )
            frame_records.append(record)

            # Update lag references
            prev_count = det_count
            prev_mean_conf = mean_conf

        # Aggregate temporal consistency metrics
        mean_cnt_var = statistics.mean(count_diffs) if count_diffs else 0.0
        max_cnt_var = max(count_diffs) if count_diffs else 0
        cnt_variance = statistics.variance(count_diffs) if len(count_diffs) > 1 else 0.0
        mean_cnf_var = statistics.mean(conf_diffs) if conf_diffs else 0.0

        # Stability score: Normalized [0, 1] penalizing large consecutive fluctuations
        # S = max(0.0, 1.0 - (mean_count_diff / max(1.0, mean_detection_volume)))
        counts_list = [f.detection_count for f in frame_records if f.processing_status == "success"]
        avg_vol = statistics.mean(counts_list) if counts_list else 1.0
        stability = max(0.0, min(1.0, 1.0 - (mean_cnt_var / max(1.0, avg_vol + 1.0))))

        temporal_metrics = TemporalConsistencyMetrics(
            mean_detection_count_variation=mean_cnt_var,
            max_detection_count_variation=max_cnt_var,
            detection_count_variance=cnt_variance,
            mean_confidence_variation=mean_cnf_var,
            stability_score=stability,
            consecutive_zero_detection_max=max_zero_streak,
        )

        return frame_records, temporal_metrics

    def _calculate_heuristic_reliability(
        self,
        status: str,
        det_count: int,
        mean_conf: Optional[float],
        confidences: List[float],
        count_delta: Optional[int],
    ) -> float:
        """Calculate heuristic frame-level perception reliability indicator [0.0, 1.0].
        
        Formula:
          R = 0.0 if status == 'failed'
          If zero detections and successful:
             R = 0.85 (neutral clean frame baseline)
          If detections exist:
             R_base = mean_confidence
             penalty_low_conf = (count(conf < threshold) / det_count) * 0.25
             penalty_delta = min(0.20, abs(count_delta) * 0.05) if count_delta else 0.0
             R = max(0.05, min(1.0, R_base - penalty_low_conf - penalty_delta))
        """
        if status != "success":
            return 0.0

        if det_count == 0 or mean_conf is None:
            return 0.85  # Clean frame without detections

        r_base = mean_conf
        low_count = sum(1 for c in confidences if c < self.confidence_threshold)
        penalty_low = (low_count / max(1, det_count)) * 0.25
        penalty_delta = (
            min(0.20, abs(count_delta) * 0.05) if count_delta is not None else 0.0
        )

        return max(0.05, min(1.0, r_base - penalty_low - penalty_delta))

    # --------------------------------------------------------------------------
    # Ground-Truth Evaluation
    # --------------------------------------------------------------------------

    def evaluate_ground_truth(
        self,
        frame_results: Sequence[Dict[str, Any]],
        ground_truth_file: Optional[Union[str, Path]] = None,
    ) -> GroundTruthEvaluationResult:
        """Evaluate detection predictions against labeled ground-truth annotations.
        
        If ground_truth_file is None or does not exist, explicitly marks status as 'unavailable'.
        If provided, executes greedy bipartite IoU matching at self.iou_threshold.
        """
        if ground_truth_file is None:
            return GroundTruthEvaluationResult(
                status="unavailable",
                iou_threshold=self.iou_threshold,
                disclaimer=(
                    "No ground-truth annotations provided in dataset. "
                    "Precision, recall, F1-score, and mAP cannot be evaluated."
                ),
            )

        gt_path = Path(ground_truth_file)
        if not gt_path.is_file():
            return GroundTruthEvaluationResult(
                status="unavailable",
                iou_threshold=self.iou_threshold,
                disclaimer=(
                    f"Ground-truth file not found at '{gt_path}'. "
                    "Precision, recall, F1-score, and mAP cannot be evaluated."
                ),
            )

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except Exception as e:
            raise GroundTruthEvaluationError(f"Failed to parse ground-truth JSON '{gt_path}': {e}") from e

        # Index ground truth by image_filename
        gt_by_frame: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(gt_data, dict) and "frames" in gt_data:
            for gtf in gt_data["frames"]:
                fname = gtf.get("image_filename")
                if fname:
                    gt_by_frame[fname] = gtf.get("annotations", [])
        elif isinstance(gt_data, list):
            for gtf in gt_data:
                fname = gtf.get("image_filename")
                if fname:
                    gt_by_frame[fname] = gtf.get("annotations", [])
        else:
            raise GroundTruthEvaluationError(f"Unsupported ground-truth format in '{gt_path}'")

        total_tp = 0
        total_fp = 0
        total_fn = 0
        matched_ious: List[float] = []
        class_metrics: Dict[str, Dict[str, int]] = {}

        for frame in frame_results:
            fname = frame.get("image_filename", "")
            preds = frame.get("detections", [])
            gts = deepcopy(gt_by_frame.get(fname, []))

            # Sort predictions by confidence descending for greedy bipartite matching
            sorted_preds = sorted(preds, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
            gt_matched = [False] * len(gts)

            for pred in sorted_preds:
                p_box = pred.get("bbox_xyxy", [])
                p_cls = pred.get("class_name", "")
                best_iou = 0.0
                best_gt_idx = -1

                for idx, gt in enumerate(gts):
                    if gt_matched[idx]:
                        continue
                    g_cls = gt.get("class_name", "")
                    if p_cls.lower() != g_cls.lower():
                        continue
                    g_box = gt.get("bbox_xyxy", [])
                    iou = compute_box_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    total_tp += 1
                    gt_matched[best_gt_idx] = True
                    matched_ious.append(best_iou)
                    if p_cls not in class_metrics:
                        class_metrics[p_cls] = {"tp": 0, "fp": 0, "fn": 0}
                    class_metrics[p_cls]["tp"] += 1
                else:
                    total_fp += 1
                    if p_cls not in class_metrics:
                        class_metrics[p_cls] = {"tp": 0, "fp": 0, "fn": 0}
                    class_metrics[p_cls]["fp"] += 1

            # Remaining unmatched ground truth are False Negatives
            for idx, matched in enumerate(gt_matched):
                if not matched:
                    total_fn += 1
                    g_cls = gts[idx].get("class_name", "")
                    if g_cls not in class_metrics:
                        class_metrics[g_cls] = {"tp": 0, "fp": 0, "fn": 0}
                    class_metrics[g_cls]["fn"] += 1

        precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        mean_iou = statistics.mean(matched_ious) if matched_ious else 0.0

        # Compute per-class precision, recall, F1
        per_class_summary: Dict[str, Dict[str, Any]] = {}
        for cname, counts in class_metrics.items():
            c_tp = counts["tp"]
            c_fp = counts["fp"]
            c_fn = counts["fn"]
            c_prec = (c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
            c_rec = (c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 0.0
            c_f1 = (2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0
            per_class_summary[cname] = {
                "true_positives": c_tp,
                "false_positives": c_fp,
                "false_negatives": c_fn,
                "precision": round(c_prec, 4),
                "recall": round(c_rec, 4),
                "f1_score": round(c_f1, 4),
            }

        return GroundTruthEvaluationResult(
            status="available",
            iou_threshold=self.iou_threshold,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            mean_matched_iou=round(mean_iou, 4),
            class_evaluations=per_class_summary,
            disclaimer="Evaluation performed against verified ground-truth annotations.",
        )

    # --------------------------------------------------------------------------
    # Full Execution Pipeline
    # --------------------------------------------------------------------------

    def analyze_detections(
        self,
        detection_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        ground_truth_file: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        overwrite: bool = True,
    ) -> Tuple[MetricsAnalysisSummary, Path]:
        """Execute complete metrics evaluation and persist versioned JSON outputs.
        
        Args:
            detection_file: Path to detection_results.json.
            output_dir: Destination directory (auto-resolved under datasets/processed/ if None).
            ground_truth_file: Optional path to ground-truth annotations.
            max_frames: Optional frame limit for partial smoke evaluation.
            overwrite: Whether to overwrite existing destination directory.
            
        Returns:
            Tuple of (MetricsAnalysisSummary, output_directory_path).
            
        Raises:
            MetricsPersistenceError: If output dir exists and overwrite is False.
        """
        # 1. Load detection data
        data = self.load_and_validate_input(detection_file)
        detection_run_id = str(data.get("run_id", "unknown_detection_run"))
        dataset_id = str(data.get("dataset_id", "unknown_dataset"))
        all_frames: List[Dict[str, Any]] = data["frame_results"]
        total_discovered = len(all_frames)

        # 2. Frame slicing if max_frames is set
        frames_to_analyze = all_frames[:max_frames] if (max_frames and max_frames > 0) else all_frames
        total_attempted = len(frames_to_analyze)
        is_partial_run = total_attempted < total_discovered

        # 3. Resolve output destination
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_uuid = uuid.uuid4().hex[:6]
        run_id = f"metrics_{timestamp_slug}_{run_uuid}"

        if output_dir:
            out_root = Path(output_dir).resolve()
        else:
            out_root = (
                self.project_root
                / "datasets"
                / "processed"
                / dataset_id
                / "analysis"
                / run_id
            ).resolve()

        if out_root.exists() and not overwrite:
            raise MetricsPersistenceError(
                f"Analysis directory '{out_root}' already exists and overwrite=False"
            )

        out_root.mkdir(parents=True, exist_ok=True)

        # 4. Compute metrics across all dimensions
        det_stats = self.compute_detection_statistics(frames_to_analyze)
        conf_analysis = self.compute_confidence_analysis(frames_to_analyze)
        class_metrics = self.compute_class_metrics(frames_to_analyze)
        frame_records, temporal_metrics = self.compute_frame_and_temporal_metrics(frames_to_analyze)
        gt_result = self.evaluate_ground_truth(frames_to_analyze, ground_truth_file=ground_truth_file)

        # Heuristic reliability mean across successful frames
        reliabilities = [
            f.heuristic_reliability for f in frame_records if f.processing_status == "success"
        ]
        mean_reliability = statistics.mean(reliabilities) if reliabilities else 0.0

        # Operational status assignment
        if det_stats["failed_frames"] == 0 and not is_partial_run:
            final_status = "PASS"
        elif det_stats["successful_frames"] > 0:
            final_status = "WARNING"
        else:
            final_status = "FAIL"

        summary = MetricsAnalysisSummary(
            schema_version="1.0",
            run_id=run_id,
            dataset_id=dataset_id,
            detection_run_id=detection_run_id,
            total_frames_processed=det_stats["total_frames_processed"],
            successful_frames=det_stats["successful_frames"],
            failed_frames=det_stats["failed_frames"],
            frames_with_detections=det_stats["frames_with_detections"],
            frames_without_detections=det_stats["frames_without_detections"],
            total_detections=det_stats["total_detections"],
            average_detections_per_frame=det_stats["average_detections_per_frame"],
            min_detections_per_frame=det_stats["min_detections_per_frame"],
            max_detections_per_frame=det_stats["max_detections_per_frame"],
            mean_confidence=conf_analysis["mean_confidence"],
            median_confidence=conf_analysis["median_confidence"],
            min_confidence=conf_analysis["min_confidence"],
            max_confidence=conf_analysis["max_confidence"],
            confidence_std_dev=conf_analysis["confidence_std_dev"],
            low_confidence_count=conf_analysis["low_confidence_count"],
            low_confidence_percentage=conf_analysis["low_confidence_percentage"],
            temporal_stability_score=temporal_metrics.stability_score,
            mean_heuristic_reliability=mean_reliability,
            ground_truth_status=gt_result.status,
            final_status=final_status,
        )

        # 5. Persist versioned JSON artifacts
        self._write_analysis_artifacts(
            out_root=out_root,
            summary=summary,
            det_stats=det_stats,
            conf_analysis=conf_analysis,
            class_metrics=class_metrics,
            frame_records=frame_records,
            temporal_metrics=temporal_metrics,
            gt_result=gt_result,
            source_file=Path(detection_file).resolve(),
            source_metadata=data.get("effective_configuration"),
        )

        return summary, out_root

    # --------------------------------------------------------------------------
    # Output Persistence (Atomic writing)
    # --------------------------------------------------------------------------

    def _write_analysis_artifacts(
        self,
        out_root: Path,
        summary: MetricsAnalysisSummary,
        det_stats: Dict[str, Any],
        conf_analysis: Dict[str, Any],
        class_metrics: Dict[str, ClassMetricRecord],
        frame_records: List[FrameMetricRecord],
        temporal_metrics: TemporalConsistencyMetrics,
        gt_result: GroundTruthEvaluationResult,
        source_file: Path,
        source_metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Atomically dump all 5 versioned analysis JSON artifacts."""
        # 1. detection_metrics.json (Global stats, confidence, temporal, heuristic)
        metrics_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "source_detection_run_id": summary.detection_run_id,
            "detection_statistics": det_stats,
            "confidence_analysis": conf_analysis,
            "temporal_consistency": temporal_metrics.to_dict(),
            "heuristic_reliability": {
                "mean_reliability_score": round(summary.mean_heuristic_reliability, 4),
                "formula_documentation": (
                    "R = mean_confidence - (low_conf_pct * 0.25) - (delta_penalty). "
                    "Neutral baseline 0.85 for successful zero-detection frames."
                ),
                "disclaimer": (
                    "Heuristic signal based on detection volume stability and softmax confidence. "
                    "Not ground-truth precision/recall accuracy."
                ),
            },
            "ground_truth_evaluation": gt_result.to_dict(),
            "generated_at": summary.generated_at,
        }
        self._atomic_json_dump(out_root / "detection_metrics.json", metrics_payload)

        # 2. class_metrics.json
        class_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "total_classes_detected": len(class_metrics),
            "classes": {cname: cdata.to_dict() for cname, cdata in class_metrics.items()},
            "generated_at": summary.generated_at,
        }
        self._atomic_json_dump(out_root / "class_metrics.json", class_payload)

        # 3. frame_metrics.json
        frame_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "total_frames": len(frame_records),
            "frame_evaluations": [f.to_dict() for f in frame_records],
            "generated_at": summary.generated_at,
        }
        self._atomic_json_dump(out_root / "frame_metrics.json", frame_payload)

        # 4. analysis_summary.json
        self._atomic_json_dump(out_root / "analysis_summary.json", summary.to_dict())

        # 5. analysis_metadata.json
        meta_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "source_detection_run_id": summary.detection_run_id,
            "source_detection_file": str(source_file),
            "analysis_parameters": {
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
            },
            "metric_definitions": {
                "detection_statistics": "Counts of frames, volume, and bounds.",
                "confidence_analysis": "Softmax probability distribution across COCO classes.",
                "temporal_consistency": "Frame-to-frame delta volatility across chronological camera frames.",
                "ground_truth_evaluation": (
                    "Bipartite IoU matching against annotated ground truth. "
                    "Reported as 'unavailable' when labels are absent."
                ),
            },
            "source_model_configuration": source_metadata or {},
            "software_versions": {
                "python": platform.python_version(),
            },
            "environment": {
                "os": platform.platform(),
                "processor": platform.processor(),
            },
            "generated_at": summary.generated_at,
        }
        self._atomic_json_dump(out_root / "analysis_metadata.json", meta_payload)

    @staticmethod
    def _atomic_json_dump(target_path: Path, payload: Dict[str, Any]) -> None:
        """Serialize dictionary to JSON atomically using a temporary file."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_path.with_name(f".tmp_{uuid.uuid4().hex}_{target_path.name}")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(temp_file, target_path)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise MetricsPersistenceError(f"Failed to atomically persist JSON to '{target_path}': {e}") from e


# ==============================================================================
# CLI Entrypoint & Formatted Reporting
# ==============================================================================

def main(cli_args: Optional[List[str]] = None) -> int:
    """CLI runner for DetectionMetrics."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Advanced YOLO Detection Metrics & Perception Analysis Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to detection_results.json generated by Task 4.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for metrics and analysis artifacts.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.30,
        help="Threshold below which detections are categorized as low confidence.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="Minimum IoU overlap required for ground-truth matching.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Optional path to ground-truth annotations JSON.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to evaluate (for partial smoke analysis).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination directory if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG diagnostic logging.",
    )

    args = parser.parse_args(cli_args)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    engine = DetectionMetrics(
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        project_root=_project_root,
    )

    try:
        summary, out_root = engine.analyze_detections(
            detection_file=args.input,
            output_dir=args.output_dir,
            ground_truth_file=args.ground_truth,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"\nFATAL: Detection metrics evaluation aborted: {e}", file=sys.stderr)
        return 1

    # Formatted terminal output
    print("=" * 60)
    print("PERCEPTGUARD AI – DETECTION METRICS & ANALYSIS")
    print("=" * 60)
    print(f"Dataset:                  {summary.dataset_id}")
    print(f"Source detection run:     {summary.detection_run_id}")
    print(f"Input detection file:     {Path(args.input).resolve()}")
    print(f"Output directory:         {out_root}")
    print()
    print(f"Frames processed:         {summary.total_frames_processed}")
    print(f"Successful frames:        {summary.successful_frames}")
    print(f"Failed frames:            {summary.failed_frames}")
    print(f"Frames with detections:   {summary.frames_with_detections}")
    print(f"Frames without detections:{summary.frames_without_detections}")
    print(f"Total detections:         {summary.total_detections}")
    print(f"Average detections/frame: {summary.average_detections_per_frame:.2f}")
    print(f"Detections range:         [{summary.min_detections_per_frame}, {summary.max_detections_per_frame}]")
    print()
    print(f"Mean confidence:          {summary.mean_confidence:.4f}")
    print(f"Median confidence:        {summary.median_confidence:.4f}")
    print(f"Confidence std dev:       {summary.confidence_std_dev:.4f}")
    print(f"Confidence bounds:        [{summary.min_confidence:.4f}, {summary.max_confidence:.4f}]")
    print(f"Low-confidence detections:{summary.low_confidence_count} ({summary.low_confidence_percentage:.1f}%)")
    print()
    print(f"Temporal stability score: {summary.temporal_stability_score:.4f}")
    print(f"Mean heuristic reliability:{summary.mean_heuristic_reliability:.4f}")
    print(f"Ground-truth status:      {summary.ground_truth_status}")
    print()
    print(f"Final evaluation status:  {summary.final_status}")
    print("=" * 60)

    return 0 if summary.final_status in ("PASS", "WARNING") else 1


if __name__ == "__main__":
    sys.exit(main())

