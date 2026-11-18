"""PerceptGuard AI: Detection Feature Extractor.

Extracts detection-level and frame-level perception features from existing YOLO
detection artifacts (Schema v1.0) without re-running model inference.

Features extracted:
1. Detection-level Identity: dataset_id, frame_id, source_frame_index, timestamp_seconds,
   detection_index, class_id, class_name.
2. Confidence Features: confidence, confidence_category (low, medium, high).
3. Bounding Box Geometry: x_min, y_min, x_max, y_max, width, height, area, center_x, center_y.
4. Normalized Geometry: normalized_center_x, normalized_center_y, normalized_width,
   normalized_height, normalized_bbox_area (bounded by image dimensions).
5. Frame-level Detection Aggregations: detection_count, unique_class_count, mean_confidence,
   minimum_confidence, maximum_confidence, confidence_std, total_bbox_area, mean_bbox_area,
   maximum_bbox_area, mean_bbox_width, mean_bbox_height, and class_wise_detection_counts.

Zero-detection frames are represented with explicit null values for unavailable statistics,
preserving complete temporal frame sequences.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.features.feature_config import FeatureConfig, FeatureConfigError
from src.features.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    validate_input_artifacts,
)

logger = logging.getLogger("PerceptGuard.DetectionFeatureExtractor")


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class DetectionFeatureExtractionError(Exception):
    """Base exception for detection feature extraction errors."""
    pass


class InputArtifactValidationError(DetectionFeatureExtractionError):
    """Raised when the input detection artifact fails structural schema validation."""
    pass


class InvalidDetectionRecordError(DetectionFeatureExtractionError):
    """Raised when detection records violate geometric or numerical invariants."""
    pass


class OutputDirectoryExistsError(DetectionFeatureExtractionError):
    """Raised when target output directory already exists and overwrite is False."""
    pass


# ==============================================================================
# Data Contracts & Records
# ==============================================================================

@dataclass
class DetectionFeatureRecord:
    """Strongly typed feature vector for an individual object detection."""
    dataset_id: str
    frame_id: str
    source_frame_index: Optional[int]
    timestamp_seconds: Optional[float]
    detection_index: int
    class_id: int
    class_name: str
    confidence: float
    confidence_category: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    bbox_center_x: float
    bbox_center_y: float
    normalized_center_x: Optional[float]
    normalized_center_y: Optional[float]
    normalized_width: Optional[float]
    normalized_height: Optional[float]
    normalized_bbox_area: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameDetectionFeatureRecord:
    """Strongly typed aggregated detection features for a single video frame."""
    frame_id: str
    source_frame_index: Optional[int]
    timestamp_seconds: Optional[float]
    image_width: Optional[int]
    image_height: Optional[int]
    detection_count: int
    unique_class_count: int
    mean_confidence: Optional[float]
    minimum_confidence: Optional[float]
    maximum_confidence: Optional[float]
    confidence_std: Optional[float]
    total_bbox_area: Optional[float]
    mean_bbox_area: Optional[float]
    maximum_bbox_area: Optional[float]
    mean_bbox_width: Optional[float]
    mean_bbox_height: Optional[float]
    class_wise_detection_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionExtractionErrorRecord:
    """Diagnostic record capturing validation rejections during extraction."""
    frame_id: str
    detection_index: Optional[int]
    error_type: str
    message: str
    raw_data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionFeatureSummary:
    """High-level execution summary and dataset telemetry."""
    schema_version: str
    run_id: str
    dataset_id: str
    source_detection_artifact: str
    extracted_at: str
    is_partial: bool
    processed_frame_count: int
    valid_detection_count: int
    invalid_detection_count: int
    zero_detection_frame_count: int
    extraction_duration_seconds: float
    validation_status: str
    feature_counts: Dict[str, int]
    class_distribution: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Helper Functions
# ==============================================================================

def categorize_confidence(conf: float) -> str:
    """Assign deterministic confidence tier: high (>=0.75), medium (>=0.40), low (<0.40)."""
    if conf >= 0.75:
        return "high"
    elif conf >= 0.40:
        return "medium"
    return "low"


def compute_std(values: Sequence[float], mean_val: float) -> float:
    """Compute population standard deviation sqrt(sum((x - mean)^2) / N)."""
    if not values or len(values) <= 1:
        return 0.0
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


# ==============================================================================
# Detection Feature Extractor Engine
# ==============================================================================

class DetectionFeatureExtractor:
    """Production detection feature extractor operating on YOLO inference artifacts."""

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        decimal_places: Optional[int] = None,
    ) -> None:
        """Initialize extractor with configuration parameters.

        Args:
            config: Optional FeatureConfig instance. If None, default configuration is loaded.
            decimal_places: Optional decimal precision override. Defaults to config.decimal_places.
        """
        if config is None:
            try:
                self.config = FeatureConfig.from_yaml()
            except Exception:
                self.config = FeatureConfig()
        else:
            self.config = config

        self.decimal_places = (
            decimal_places if decimal_places is not None else self.config.decimal_places
        )

    def extract_from_file(
        self,
        input_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        overwrite: bool = False,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Load detection artifact from JSON file and extract features.

        Args:
            input_file: Path to detection results JSON.
            output_dir: Destination directory. If None, resolves to default feature hierarchy.
            max_frames: Maximum number of frames to process.
            start_frame: Start frame sequence index (inclusive).
            end_frame: End frame sequence index (inclusive).
            overwrite: If True, allows overwriting existing output directory.
            strict: If True, raises exceptions on artifact validation warnings.

        Returns:
            Dictionary containing output directory and summary telemetry.
        """
        t_start = time.perf_counter()
        in_path = Path(input_file).resolve()

        if not in_path.exists():
            raise FileNotFoundError(f"Detection input artifact not found: '{in_path}'")
        if in_path.stat().st_size == 0:
            raise InputArtifactValidationError(f"Detection input artifact is empty: '{in_path}'")

        try:
            with open(in_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as ex:
            raise InputArtifactValidationError(f"Failed to parse JSON in '{in_path}': {ex}") from ex

        # Schema Validation
        val_res = validate_input_artifacts(detection_file=in_path, strict=strict)
        if not val_res.is_valid and strict:
            raise InputArtifactValidationError(
                f"Artifact validation failed for '{in_path}': {val_res.errors}"
            )

        # Extract features in-memory
        (
            det_records,
            frame_records,
            summary,
            errors,
            raw_metadata,
        ) = self.extract_from_dict(
            raw_data=raw_data,
            source_artifact=str(in_path),
            max_frames=max_frames,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        duration = round(time.perf_counter() - t_start, 4)
        summary.extraction_duration_seconds = duration

        # Resolve output directory
        target_dir = self._resolve_output_dir(
            output_dir=output_dir,
            dataset_id=summary.dataset_id,
            run_id=summary.run_id,
        )

        if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
            raise OutputDirectoryExistsError(
                f"Output directory '{target_dir}' exists and is not empty. Pass overwrite=True to replace."
            )

        target_dir.mkdir(parents=True, exist_ok=True)

        # Persist artifacts atomically
        self._persist_artifacts(
            target_dir=target_dir,
            det_records=det_records,
            frame_records=frame_records,
            summary=summary,
            errors=errors,
            raw_metadata=raw_metadata,
        )

        logger.info(
            f"Extraction completed successfully in {duration}s. "
            f"Frames: {summary.processed_frame_count}, Valid Detections: {summary.valid_detection_count}, "
            f"Output: {target_dir}"
        )

        return {
            "output_dir": str(target_dir),
            "run_id": summary.run_id,
            "summary": summary.to_dict(),
            "valid_detection_count": summary.valid_detection_count,
            "processed_frame_count": summary.processed_frame_count,
            "duration_seconds": duration,
        }

    def extract_from_dict(
        self,
        raw_data: Dict[str, Any],
        source_artifact: str = "in_memory_dict",
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> Tuple[
        List[DetectionFeatureRecord],
        List[FrameDetectionFeatureRecord],
        DetectionFeatureSummary,
        List[DetectionExtractionErrorRecord],
        Dict[str, Any],
    ]:
        """Extract detection features from pre-parsed detection JSON dictionary.

        Returns:
            Tuple of (detection_records, frame_records, summary, errors, raw_metadata).
        """
        t_start = time.perf_counter()
        errors: List[DetectionExtractionErrorRecord] = []

        # 1. Dataset & Run Identification
        dataset_id = str(raw_data.get("dataset_id") or "unknown_dataset")
        run_id = f"det_features_{dataset_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # 2. Extract and Normalize Frame Array
        frames_raw = self._normalize_frame_list(raw_data)
        if not frames_raw and not raw_data.get("detections"):
            errors.append(
                DetectionExtractionErrorRecord(
                    frame_id="ROOT",
                    detection_index=None,
                    error_type="NO_FRAMES_FOUND",
                    message="Detection dictionary contains no valid frame results or detections.",
                )
            )

        # 3. Check for Duplicate Frame Identifiers & Validate Ordering
        seen_frame_ids: Set[str] = set()
        normalized_frames: List[Dict[str, Any]] = []

        for f_entry in frames_raw:
            fid = str(
                f_entry.get("frame_id")
                or f_entry.get("image_filename")
                or f_entry.get("extracted_frame_id")
                or f"frame_{len(normalized_frames):06d}"
            )
            if fid in seen_frame_ids:
                errors.append(
                    DetectionExtractionErrorRecord(
                        frame_id=fid,
                        detection_index=None,
                        error_type="DUPLICATE_FRAME_ID",
                        message=f"Duplicate frame identifier encountered: '{fid}'. Skipping duplicate.",
                    )
                )
                continue
            seen_frame_ids.add(fid)
            f_entry["_resolved_frame_id"] = fid
            normalized_frames.append(f_entry)

        # 4. Filter by Sequence Boundaries (start_frame, end_frame, max_frames)
        is_partial = False
        if start_frame is not None or end_frame is not None or max_frames is not None:
            is_partial = True

        filtered_frames = normalized_frames
        if start_frame is not None:
            filtered_frames = [
                f for f in filtered_frames
                if f.get("source_frame_index", f.get("extracted_frame_id", 0)) >= start_frame
            ]
        if end_frame is not None:
            filtered_frames = [
                f for f in filtered_frames
                if f.get("source_frame_index", f.get("extracted_frame_id", 0)) <= end_frame
            ]
        if max_frames is not None and max_frames > 0:
            filtered_frames = filtered_frames[:max_frames]

        # 5. Iterative Extraction
        det_records: List[DetectionFeatureRecord] = []
        frame_records: List[FrameDetectionFeatureRecord] = []
        class_distribution: Counter[str] = Counter()

        zero_detection_frames = 0
        valid_detection_count = 0
        invalid_detection_count = 0

        for frame in filtered_frames:
            frame_id = frame["_resolved_frame_id"]
            src_idx = frame.get("source_frame_index")
            ts_sec = frame.get("timestamp_seconds")
            img_w = frame.get("image_width")
            img_h = frame.get("image_height")

            # Validate image dimensions
            valid_dims = (
                isinstance(img_w, (int, float))
                and isinstance(img_h, (int, float))
                and img_w > 0
                and img_h > 0
            )
            if not valid_dims and (img_w is not None or img_h is not None):
                errors.append(
                    DetectionExtractionErrorRecord(
                        frame_id=frame_id,
                        detection_index=None,
                        error_type="INVALID_IMAGE_DIMENSIONS",
                        message=f"Invalid image dimensions: width={img_w}, height={img_h}",
                    )
                )

            raw_detections = frame.get("detections", [])
            if not isinstance(raw_detections, list):
                errors.append(
                    DetectionExtractionErrorRecord(
                        frame_id=frame_id,
                        detection_index=None,
                        error_type="MALFORMED_DETECTIONS_FIELD",
                        message=f"Detections field is not a list in frame '{frame_id}'",
                    )
                )
                raw_detections = []

            frame_valid_dets: List[DetectionFeatureRecord] = []
            seen_det_indices: Set[int] = set()

            for d_idx, d_raw in enumerate(raw_detections):
                if not isinstance(d_raw, dict):
                    invalid_detection_count += 1
                    errors.append(
                        DetectionExtractionErrorRecord(
                            frame_id=frame_id,
                            detection_index=d_idx,
                            error_type="MALFORMED_DETECTION_OBJECT",
                            message=f"Detection entry at index {d_idx} is not a dictionary.",
                            raw_data=d_raw,
                        )
                    )
                    continue

                det_index = d_raw.get("detection_index", d_idx)
                if det_index in seen_det_indices:
                    errors.append(
                        DetectionExtractionErrorRecord(
                            frame_id=frame_id,
                            detection_index=det_index,
                            error_type="DUPLICATE_DETECTION_INDEX",
                            message=f"Duplicate detection_index {det_index} in frame '{frame_id}'",
                        )
                    )
                seen_det_indices.add(det_index)

                # Validate Detection Record
                is_valid, err_msg, err_type = self._validate_detection_raw(d_raw)
                if not is_valid:
                    invalid_detection_count += 1
                    errors.append(
                        DetectionExtractionErrorRecord(
                            frame_id=frame_id,
                            detection_index=det_index,
                            error_type=err_type,
                            message=err_msg,
                            raw_data=d_raw,
                        )
                    )
                    continue

                # Extract Detection Features
                rec = self._extract_single_detection(
                    d_raw=d_raw,
                    dataset_id=dataset_id,
                    frame_id=frame_id,
                    source_frame_index=src_idx,
                    timestamp_seconds=ts_sec,
                    detection_index=det_index,
                    img_w=float(img_w) if valid_dims else None,
                    img_h=float(img_h) if valid_dims else None,
                )
                frame_valid_dets.append(rec)
                det_records.append(rec)
                class_distribution[rec.class_name] += 1
                valid_detection_count += 1

            # Build Frame-level Aggregation Record
            f_record = self._build_frame_record(
                frame_id=frame_id,
                source_frame_index=src_idx,
                timestamp_seconds=ts_sec,
                image_width=int(img_w) if valid_dims else None,
                image_height=int(img_h) if valid_dims else None,
                valid_detections=frame_valid_dets,
            )
            frame_records.append(f_record)

            if len(frame_valid_dets) == 0:
                zero_detection_frames += 1

        duration = round(time.perf_counter() - t_start, 4)
        val_status = "PASS" if invalid_detection_count == 0 else "WARNING"

        summary = DetectionFeatureSummary(
            schema_version=FEATURE_SCHEMA_VERSION,
            run_id=run_id,
            dataset_id=dataset_id,
            source_detection_artifact=source_artifact,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            is_partial=is_partial,
            processed_frame_count=len(filtered_frames),
            valid_detection_count=valid_detection_count,
            invalid_detection_count=invalid_detection_count,
            zero_detection_frame_count=zero_detection_frames,
            extraction_duration_seconds=duration,
            validation_status=val_status,
            feature_counts={
                "detection_level_features": 22,
                "frame_level_features": 16,
            },
            class_distribution=dict(class_distribution),
        )

        raw_metadata = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "source_detection_artifact": source_artifact,
            "extracted_at": summary.extracted_at,
            "is_partial": is_partial,
            "configuration": self.config.to_dict(),
            "feature_dictionary": self._get_feature_dictionary(),
            "execution_environment": {
                "python_version": sys.version,
                "platform": sys.platform,
            },
        }

        return det_records, frame_records, summary, errors, raw_metadata

    def _validate_detection_raw(
        self, d_raw: Dict[str, Any]
    ) -> Tuple[bool, str, str]:
        """Verify that raw detection payload satisfies coordinate and confidence invariants."""
        # 1. Check Class Information
        class_id = d_raw.get("class_id")
        class_name = d_raw.get("class_name")
        if class_id is None or not isinstance(class_id, int) or class_id < 0:
            return False, f"Invalid or missing class_id: {class_id}", "INVALID_CLASS_ID"
        if not class_name or not isinstance(class_name, str) or not class_name.strip():
            return False, f"Invalid or empty class_name: {class_name}", "INVALID_CLASS_NAME"

        # 2. Check Confidence
        conf = d_raw.get("confidence")
        if conf is None or not isinstance(conf, (int, float)):
            return False, f"Missing or non-numeric confidence: {conf}", "INVALID_CONFIDENCE"
        if math.isnan(conf) or math.isinf(conf):
            return False, f"Non-finite confidence: {conf}", "NON_FINITE_CONFIDENCE"
        if not (0.0 <= conf <= 1.0):
            return False, f"Confidence {conf} out of range [0.0, 1.0]", "CONFIDENCE_OUT_OF_RANGE"

        # 3. Check Bounding Box Coordinates
        bbox = d_raw.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False, f"Bounding box must be 4-element sequence, got: {bbox}", "INVALID_BBOX_STRUCTURE"

        x_min, y_min, x_max, y_max = bbox
        for val in [x_min, y_min, x_max, y_max]:
            if not isinstance(val, (int, float)):
                return False, f"Non-numeric bbox coordinate: {val}", "NON_NUMERIC_COORDINATE"
            if math.isnan(val) or math.isinf(val):
                return False, f"Non-finite bbox coordinate: {val}", "NON_FINITE_COORDINATE"

        if x_min < 0.0 or y_min < 0.0:
            return False, f"Negative bbox coordinate: ({x_min}, {y_min})", "NEGATIVE_COORDINATES"
        if x_max <= x_min:
            return False, f"Inverted or zero width bbox: x_min={x_min}, x_max={x_max}", "INVERTED_X_COORDINATES"
        if y_max <= y_min:
            return False, f"Inverted or zero height bbox: y_min={y_min}, y_max={y_max}", "INVERTED_Y_COORDINATES"

        area = (x_max - x_min) * (y_max - y_min)
        if area <= 0.0:
            return False, f"Zero or negative bbox area: {area}", "ZERO_AREA_BBOX"

        return True, "", ""

    def _extract_single_detection(
        self,
        d_raw: Dict[str, Any],
        dataset_id: str,
        frame_id: str,
        source_frame_index: Optional[int],
        timestamp_seconds: Optional[float],
        detection_index: int,
        img_w: Optional[float],
        img_h: Optional[float],
    ) -> DetectionFeatureRecord:
        """Extract individual detection record applying geometric and normalization math."""
        dec = self.decimal_places

        class_id = int(d_raw["class_id"])
        class_name = str(d_raw["class_name"]).strip()
        confidence = round(float(d_raw["confidence"]), dec)
        conf_cat = categorize_confidence(confidence)

        x_min, y_min, x_max, y_max = [float(v) for v in d_raw["bbox_xyxy"]]

        bbox_width = round(x_max - x_min, dec)
        bbox_height = round(y_max - y_min, dec)
        bbox_area = round(bbox_width * bbox_height, dec)
        bbox_center_x = round((x_min + x_max) / 2.0, dec)
        bbox_center_y = round((y_min + y_max) / 2.0, dec)

        x_min_r = round(x_min, dec)
        y_min_r = round(y_min, dec)
        x_max_r = round(x_max, dec)
        y_max_r = round(y_max, dec)

        if img_w is not None and img_h is not None and img_w > 0 and img_h > 0:
            norm_cx = round(bbox_center_x / img_w, dec)
            norm_cy = round(bbox_center_y / img_h, dec)
            norm_w = round(bbox_width / img_w, dec)
            norm_h = round(bbox_height / img_h, dec)
            norm_area = round(bbox_area / (img_w * img_h), dec)
        else:
            norm_cx = None
            norm_cy = None
            norm_w = None
            norm_h = None
            norm_area = None

        return DetectionFeatureRecord(
            dataset_id=dataset_id,
            frame_id=frame_id,
            source_frame_index=source_frame_index,
            timestamp_seconds=round(timestamp_seconds, dec) if timestamp_seconds is not None else None,
            detection_index=detection_index,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            confidence_category=conf_cat,
            x_min=x_min_r,
            y_min=y_min_r,
            x_max=x_max_r,
            y_max=y_max_r,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            bbox_area=bbox_area,
            bbox_center_x=bbox_center_x,
            bbox_center_y=bbox_center_y,
            normalized_center_x=norm_cx,
            normalized_center_y=norm_cy,
            normalized_width=norm_w,
            normalized_height=norm_h,
            normalized_bbox_area=norm_area,
        )

    def _build_frame_record(
        self,
        frame_id: str,
        source_frame_index: Optional[int],
        timestamp_seconds: Optional[float],
        image_width: Optional[int],
        image_height: Optional[int],
        valid_detections: List[DetectionFeatureRecord],
    ) -> FrameDetectionFeatureRecord:
        """Calculate aggregated frame-level statistics or explicit null values."""
        dec = self.decimal_places

        if not valid_detections:
            # Policy for zero-detection frame: explicit nulls for statistics
            return FrameDetectionFeatureRecord(
                frame_id=frame_id,
                source_frame_index=source_frame_index,
                timestamp_seconds=round(timestamp_seconds, dec) if timestamp_seconds is not None else None,
                image_width=image_width,
                image_height=image_height,
                detection_count=0,
                unique_class_count=0,
                mean_confidence=None,
                minimum_confidence=None,
                maximum_confidence=None,
                confidence_std=None,
                total_bbox_area=None,
                mean_bbox_area=None,
                maximum_bbox_area=None,
                mean_bbox_width=None,
                mean_bbox_height=None,
                class_wise_detection_counts={},
            )

        n = len(valid_detections)
        confs = [d.confidence for d in valid_detections]
        areas = [d.bbox_area for d in valid_detections]
        widths = [d.bbox_width for d in valid_detections]
        heights = [d.bbox_height for d in valid_detections]
        classes = [d.class_name for d in valid_detections]

        mean_c = sum(confs) / n
        std_c = compute_std(confs, mean_c)
        tot_area = sum(areas)
        mean_area = tot_area / n
        mean_w = sum(widths) / n
        mean_h = sum(heights) / n

        return FrameDetectionFeatureRecord(
            frame_id=frame_id,
            source_frame_index=source_frame_index,
            timestamp_seconds=round(timestamp_seconds, dec) if timestamp_seconds is not None else None,
            image_width=image_width,
            image_height=image_height,
            detection_count=n,
            unique_class_count=len(set(classes)),
            mean_confidence=round(mean_c, dec),
            minimum_confidence=round(min(confs), dec),
            maximum_confidence=round(max(confs), dec),
            confidence_std=round(std_c, dec),
            total_bbox_area=round(tot_area, dec),
            mean_bbox_area=round(mean_area, dec),
            maximum_bbox_area=round(max(areas), dec),
            mean_bbox_width=round(mean_w, dec),
            mean_bbox_height=round(mean_h, dec),
            class_wise_detection_counts=dict(Counter(classes)),
        )

    def _normalize_frame_list(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize varied JSON detection representations into standard frame list."""
        if "frame_results" in raw_data and isinstance(raw_data["frame_results"], list):
            return raw_data["frame_results"]
        if "frames" in raw_data and isinstance(raw_data["frames"], list):
            return raw_data["frames"]

        # Support Single-Image Detection Format
        if "image" in raw_data and "result" in raw_data:
            img_dict = raw_data.get("image", {})
            res_dict = raw_data.get("result", {})
            single_frame = {
                "image_filename": img_dict.get("filename", "single_image.jpg"),
                "extracted_frame_id": 1,
                "source_frame_index": 0,
                "timestamp_seconds": 0.0,
                "image_width": img_dict.get("width"),
                "image_height": img_dict.get("height"),
                "detection_count": res_dict.get("detection_count", len(res_dict.get("detections", []))),
                "detections": res_dict.get("detections", []),
            }
            return [single_frame]

        return []

    def _resolve_output_dir(
        self,
        output_dir: Optional[Union[str, Path]],
        dataset_id: str,
        run_id: str,
    ) -> Path:
        """Resolve output directory hierarchy."""
        if output_dir:
            return Path(output_dir).resolve()

        base_out = Path(self.config.output_dir)
        if not base_out.is_absolute():
            base_out = _project_root / base_out

        # Output pattern: <base_out>/detection/<run_id>/
        return base_out / "detection" / run_id

    def _persist_artifacts(
        self,
        target_dir: Path,
        det_records: List[DetectionFeatureRecord],
        frame_records: List[FrameDetectionFeatureRecord],
        summary: DetectionFeatureSummary,
        errors: List[DetectionExtractionErrorRecord],
        raw_metadata: Dict[str, Any],
    ) -> None:
        """Write all 5 versioned output artifacts atomically."""
        # 1. detection_features.json
        det_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "source_detection_artifact": summary.source_detection_artifact,
            "total_records": len(det_records),
            "records": [r.to_dict() for r in det_records],
        }
        with open(target_dir / "detection_features.json", "w", encoding="utf-8") as f:
            json.dump(det_payload, f, indent=2)

        # 2. frame_detection_features.json
        frame_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "source_detection_artifact": summary.source_detection_artifact,
            "total_frames": len(frame_records),
            "frames": [f.to_dict() for f in frame_records],
        }
        with open(target_dir / "frame_detection_features.json", "w", encoding="utf-8") as f:
            json.dump(frame_payload, f, indent=2)

        # 3. detection_feature_summary.json
        with open(target_dir / "detection_feature_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        # 4. feature_metadata.json
        with open(target_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
            json.dump(raw_metadata, f, indent=2)

        # 5. feature_errors.json
        err_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "error_count": len(errors),
            "errors": [e.to_dict() for e in errors],
        }
        with open(target_dir / "feature_errors.json", "w", encoding="utf-8") as f:
            json.dump(err_payload, f, indent=2)

    def _get_feature_dictionary(self) -> Dict[str, Dict[str, str]]:
        """Return standardized registry of extracted feature names and types."""
        return {
            "dataset_id": {"type": "string", "description": "Origin dataset identifier."},
            "frame_id": {"type": "string", "description": "Identifier of the host video frame."},
            "source_frame_index": {"type": "int", "description": "Sequential chronological frame index."},
            "timestamp_seconds": {"type": "float", "description": "Frame elapsed time in seconds."},
            "detection_index": {"type": "int", "description": "Zero-indexed rank of detection in frame."},
            "class_id": {"type": "int", "description": "COCO semantic class index."},
            "class_name": {"type": "string", "description": "Semantic category label."},
            "confidence": {"type": "float", "description": "Raw YOLO detection confidence score in [0.0, 1.0]."},
            "confidence_category": {"type": "string", "description": "Categorical confidence tier (high, medium, low)."},
            "x_min": {"type": "float", "description": "Top-left X pixel boundary."},
            "y_min": {"type": "float", "description": "Top-left Y pixel boundary."},
            "x_max": {"type": "float", "description": "Bottom-right X pixel boundary."},
            "y_max": {"type": "float", "description": "Bottom-right Y pixel boundary."},
            "bbox_width": {"type": "float", "description": "Bounding box horizontal span (x_max - x_min)."},
            "bbox_height": {"type": "float", "description": "Bounding box vertical span (y_max - y_min)."},
            "bbox_area": {"type": "float", "description": "Bounding box area in square pixels."},
            "bbox_center_x": {"type": "float", "description": "Horizontal center coordinate (x_min + x_max) / 2."},
            "bbox_center_y": {"type": "float", "description": "Vertical center coordinate (y_min + y_max) / 2."},
            "normalized_center_x": {"type": "float", "description": "Center X divided by image width."},
            "normalized_center_y": {"type": "float", "description": "Center Y divided by image height."},
            "normalized_width": {"type": "float", "description": "Bounding box width divided by image width."},
            "normalized_height": {"type": "float", "description": "Bounding box height divided by image height."},
            "normalized_bbox_area": {"type": "float", "description": "Bounding box area divided by image resolution."},
            "detection_count": {"type": "int", "description": "Total valid detections in host frame."},
            "unique_class_count": {"type": "int", "description": "Number of distinct classes detected in frame."},
            "mean_confidence": {"type": "float", "description": "Average detection confidence across frame detections."},
            "minimum_confidence": {"type": "float", "description": "Lowest detection confidence in frame."},
            "maximum_confidence": {"type": "float", "description": "Highest detection confidence in frame."},
            "confidence_std": {"type": "float", "description": "Population standard deviation of detection confidence."},
            "total_bbox_area": {"type": "float", "description": "Cumulative bounding box area across frame detections."},
            "mean_bbox_area": {"type": "float", "description": "Mean bounding box area across frame detections."},
            "maximum_bbox_area": {"type": "float", "description": "Maximum bounding box area in frame."},
            "mean_bbox_width": {"type": "float", "description": "Average bounding box width in frame."},
            "mean_bbox_height": {"type": "float", "description": "Average bounding box height in frame."},
            "class_wise_detection_counts": {"type": "dict", "description": "Mapping from class name to detection count."},
        }


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """CLI for Detection Feature Extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Detection Feature Extractor CLI."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input detection_results.json file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Optional path to feature_config.yaml file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom destination directory for feature outputs.",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Starting source frame index (inclusive).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Ending source frame index (inclusive).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose / debug diagnostic output.",
    )

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    cfg = FeatureConfig.from_yaml(args.config) if args.config else None
    extractor = DetectionFeatureExtractor(config=cfg)

    try:
        result = extractor.extract_from_file(
            input_file=args.input,
            output_dir=args.output,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overwrite=args.overwrite,
        )

        sum_dict = result["summary"]
        print("\n" + "=" * 70)
        print("PERCEPTGUARD AI — DETECTION FEATURE EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"  Dataset ID            : {sum_dict['dataset_id']}")
        print(f"  Run ID                : {sum_dict['run_id']}")
        print(f"  Status                : [{sum_dict['validation_status']}]")
        print(f"  Is Partial Run        : {sum_dict['is_partial']}")
        print(f"  Processed Frames      : {sum_dict['processed_frame_count']}")
        print(f"  Valid Detections      : {sum_dict['valid_detection_count']}")
        print(f"  Invalid Detections    : {sum_dict['invalid_detection_count']}")
        print(f"  Zero-Detection Frames : {sum_dict['zero_detection_frame_count']}")
        print(f"  Duration              : {sum_dict['extraction_duration_seconds']}s")
        print(f"  Output Directory      : {result['output_dir']}")
        print("=" * 70 + "\n")
        return 0

    except Exception as ex:
        logger.error(f"Detection feature extraction failed: {ex}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
