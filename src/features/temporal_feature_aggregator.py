"""PerceptGuard AI: Temporal Feature Aggregator.

Day 5 Task 5: Modular, mathematically verified, CPU-efficient multi-source
temporal feature aggregation and trend analysis across sequential video frames.

Responsibilities:
- Ingest pre-computed detection features (Task 2), motion features (Task 3),
  and visual quality features (Task 4) with zero model re-inference.
- Multi-source alignment across dataset_id and frame_id with chronological
  sequence sorting and dimensional consistency validation.
- Fuse 27 base frame-level metrics across detection, motion, and visual quality.
- Trailing past-looking temporal window aggregations (e.g., 3, 5, 10 frames)
  with strict zero future-frame leakage guarantees.
- Numerical aggregate statistics (mean, min, max, std, median, first, last, diff)
  with explicit observation coverage tracking.
- Ordinary Least Squares (OLS) linear trend extraction for confidence, blur,
  brightness, contrast, displacement, and speed metrics.
- Atomic persistence of Schema v1.0 multi-file artifacts.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

import numpy as np

from src.features.feature_config import FeatureConfig
from src.features.feature_schema import FEATURE_SCHEMA_VERSION

logger = logging.getLogger("perceptguard.features.temporal")

# Default temporal window sizes and alignment
DEFAULT_WINDOW_SIZES: Tuple[int, ...] = (3, 5, 10)
DEFAULT_WINDOW_ALIGNMENT: str = "trailing"
DEFAULT_MIN_OBSERVATIONS_FOR_TREND: int = 2

# Base frame metrics to aggregate across temporal windows
NUMERICAL_FEATURE_KEYS: Tuple[str, ...] = (
    # Detection features (Task 2)
    "detection_count",
    "unique_class_count",
    "mean_confidence",
    "minimum_confidence",
    "maximum_confidence",
    "confidence_std",
    "total_bbox_area",
    "mean_bbox_area",
    "mean_bbox_width",
    "mean_bbox_height",
    # Motion features (Task 3)
    "observed_track_count",
    "moving_track_count",
    "stationary_track_count",
    "mean_displacement",
    "maximum_displacement",
    "mean_speed",
    "mean_velocity_x",
    "mean_velocity_y",
    "motion_feature_coverage",
    # Visual quality features (Task 4)
    "blur_score",
    "mean_brightness",
    "median_brightness",
    "brightness_std",
    "contrast_std",
    "contrast_range",
    "contrast_percentile_range",
    "image_aspect_ratio",
)

# Metric fields tracked for linear trend slope calculation
TREND_FEATURE_MAPPINGS: Dict[str, str] = {
    "confidence_trend": "mean_confidence",
    "blur_trend": "blur_score",
    "brightness_trend": "mean_brightness",
    "contrast_trend": "contrast_std",
    "displacement_trend": "mean_displacement",
    "speed_trend": "mean_speed",
}


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class TemporalFeatureAggregationError(Exception):
    """Base exception for all temporal feature aggregation errors."""
    pass


class InputArtifactError(TemporalFeatureAggregationError):
    """Raised when an input artifact is malformed, missing, or unreadable."""
    pass


class MissingSourceArtifactError(InputArtifactError):
    """Raised when required source artifacts are missing or not provided."""
    pass


class DatasetIdMismatchError(InputArtifactError):
    """Raised when provided feature sources originate from different datasets."""
    pass


class FrameIdMismatchError(InputArtifactError):
    """Raised when input sources have no overlapping frame identifiers."""
    pass


class DimensionMismatchError(InputArtifactError):
    """Raised when input sources report conflicting image dimensions for a frame."""
    pass


class DuplicateFrameIdError(InputArtifactError):
    """Raised when duplicate frame identifiers are encountered within a source."""
    pass


class OutputDirectoryExistsError(TemporalFeatureAggregationError):
    """Raised when destination directory exists and overwrite=False."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass
class SourceAvailabilityRecord:
    """Flag indicators of which feature modalities are present for a frame."""
    detection: bool
    motion: bool
    visual_quality: bool

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


@dataclass
class FeatureCoverageRecord:
    """Summary of data coverage and observation completeness for a frame."""
    source_coverage: float
    total_sources_available: int
    valid_numeric_feature_count: int
    total_numeric_feature_count: int
    numeric_feature_coverage: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WindowFeatureStatistics:
    """Statistical distribution of a numerical feature within a temporal window."""
    mean: Optional[float]
    min: Optional[float]
    max: Optional[float]
    std: Optional[float]
    median: Optional[float]
    first: Optional[float]
    last: Optional[float]
    diff: Optional[float]
    valid_count: int
    missing_count: int
    coverage_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalTrendRecord:
    """Ordinary Least Squares linear trend over valid chronological observations."""
    slope: Optional[float]
    status: str  # "valid", "constant", "insufficient_data", "no_data"
    direction: str  # "increasing", "decreasing", "flat", "none"
    method: str = "ordinary_least_squares"
    observation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameTemporalRecord:
    """Unified multi-source temporal feature vector for an individual video frame."""
    schema_version: str
    dataset_id: str
    frame_id: str
    source_frame_index: Optional[int]
    timestamp_seconds: Optional[float]
    image_width: Optional[int]
    image_height: Optional[int]
    source_availability: SourceAvailabilityRecord
    feature_coverage: FeatureCoverageRecord
    detection_features: Dict[str, Any]
    motion_features: Dict[str, Any]
    visual_quality_features: Dict[str, Any]
    temporal_windows: Dict[str, Any]
    temporal_trends: Dict[str, Any]
    processing_status: str  # "success", "partial"

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to deeply serialized dictionary."""
        d = asdict(self)
        d["source_availability"] = self.source_availability.to_dict()
        d["feature_coverage"] = self.feature_coverage.to_dict()
        return d


@dataclass
class TemporalAggregationSummary:
    """Telemetry and distribution summary across the aggregated dataset."""
    schema_version: str
    run_id: str
    dataset_id: str
    extracted_at: str
    is_partial: bool
    processed_frame_count: int
    successfully_aligned_frame_count: int
    partially_aligned_frame_count: int
    missing_source_counts: Dict[str, int]
    invalid_record_count: int
    configured_window_sizes: List[int]
    window_alignment: str
    extraction_duration_seconds: float
    validation_status: str
    feature_counts: Dict[str, int]
    coverage_summary: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalAggregationErrorRecord:
    """Audit entry recording alignment anomalies, schema violations, or mismatches."""
    frame_id: str
    error_type: str
    message: str
    source_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Helper Mathematical Functions
# ==============================================================================

def extract_frame_index_from_id(frame_id: str) -> Optional[int]:
    """Parse numeric sequence index from frame ID (e.g. frame_000042.jpg -> 42)."""
    match = re.search(r"\d+", str(frame_id))
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None


def calculate_ols_trend(
    chronological_observations: List[Tuple[int, float]],
    decimal_places: int = 4,
    min_observations: int = 2,
) -> TemporalTrendRecord:
    """Calculate Ordinary Least Squares (OLS) linear slope across valid observations.

    Args:
        chronological_observations: List of (step_index, value) tuples where value is non-null.
        decimal_places: Precision for rounding computed slope.
        min_observations: Minimum valid points required to compute a slope.

    Returns:
        TemporalTrendRecord describing slope, status, direction, and valid counts.
    """
    k = len(chronological_observations)
    if k == 0:
        return TemporalTrendRecord(
            slope=None,
            status="no_data",
            direction="none",
            method="ordinary_least_squares",
            observation_count=0,
        )

    if k < min_observations:
        return TemporalTrendRecord(
            slope=None,
            status="insufficient_data",
            direction="none",
            method="ordinary_least_squares",
            observation_count=k,
        )

    xs = [obs[0] for obs in chronological_observations]
    ys = [obs[1] for obs in chronological_observations]

    # Check for constant values
    min_y = min(ys)
    max_y = max(ys)
    if math.isclose(min_y, max_y, abs_tol=1e-9):
        return TemporalTrendRecord(
            slope=0.0,
            status="constant",
            direction="flat",
            method="ordinary_least_squares",
            observation_count=k,
        )

    mean_x = float(sum(xs)) / k
    mean_y = float(sum(ys)) / k

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)

    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return TemporalTrendRecord(
            slope=None,
            status="insufficient_data",
            direction="none",
            method="ordinary_least_squares",
            observation_count=k,
        )

    raw_slope = numerator / denominator
    rounded_slope = round(raw_slope, decimal_places)

    if rounded_slope > 1e-4:
        direction = "increasing"
    elif rounded_slope < -1e-4:
        direction = "decreasing"
    else:
        direction = "flat"

    return TemporalTrendRecord(
        slope=rounded_slope,
        status="valid",
        direction=direction,
        method="ordinary_least_squares",
        observation_count=k,
    )


# ==============================================================================
# Temporal Feature Aggregator Engine
# ==============================================================================

class TemporalFeatureAggregator:
    """Production temporal feature aggregator unifying multi-modal perception streams."""

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        window_sizes: Optional[Sequence[int]] = None,
        window_alignment: Optional[str] = None,
        decimal_places: Optional[int] = None,
        min_observations_for_trend: Optional[int] = None,
        allow_partial: bool = False,
    ) -> None:
        """Initialize the Temporal Feature Aggregator.

        Args:
            config: Optional FeatureConfig instance.
            window_sizes: Sequence of positive window lengths (e.g. [3, 5, 10]).
            window_alignment: Temporal alignment mode ("trailing", "centered", "future").
            decimal_places: Precision for rounding output floating-point values.
            min_observations_for_trend: Minimum valid observations required for trend.
            allow_partial: If True, allow processing when some sources are missing.
        """
        self.config = config or FeatureConfig()

        raw_ws = window_sizes or getattr(self.config, "temporal_window_sizes", DEFAULT_WINDOW_SIZES)
        self.window_sizes = tuple(sorted(list(set(int(w) for w in raw_ws if int(w) > 0))))
        if not self.window_sizes:
            self.window_sizes = DEFAULT_WINDOW_SIZES

        raw_align = window_alignment or getattr(self.config, "temporal_window_alignment", DEFAULT_WINDOW_ALIGNMENT)
        self.window_alignment = str(raw_align).strip().lower()
        if self.window_alignment not in {"trailing", "centered", "future"}:
            self.window_alignment = "trailing"

        self.decimal_places = (
            decimal_places
            if decimal_places is not None
            else getattr(self.config, "decimal_places", 4)
        )

        self.min_observations_for_trend = (
            min_observations_for_trend
            if min_observations_for_trend is not None
            else getattr(self.config, "min_observations_for_trend", DEFAULT_MIN_OBSERVATIONS_FOR_TREND)
        )

        self.allow_partial = allow_partial

    # --------------------------------------------------------------------------
    # Main Public Execution Methods
    # --------------------------------------------------------------------------

    def aggregate(
        self,
        detection_source: Optional[Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]] = None,
        motion_source: Optional[Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]] = None,
        visual_source: Optional[Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        dataset_id: Optional[str] = None,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Aggregate detection, motion, and visual quality features across time.

        Args:
            detection_source: File path, dir path, or parsed data for detection features.
            motion_source: File path, dir path, or parsed data for motion features.
            visual_source: File path, dir path, or parsed data for visual quality features.
            output_dir: Destination directory for persisted feature artifacts.
            dataset_id: Optional explicit dataset identifier override.
            max_frames: Maximum number of chronological frames to process.
            start_frame: Start source frame index (inclusive).
            end_frame: End source frame index (inclusive).
            overwrite: If True, replace contents of existing output directory.

        Returns:
            Dictionary containing output directory, summary telemetry, and feature records.
        """
        t_start = time.perf_counter()
        errors: List[TemporalAggregationErrorRecord] = []

        # 1. Ingest and normalize inputs
        det_map, det_ds, det_dims, det_path = self._load_source(detection_source, "detection")
        mot_map, mot_ds, mot_dims, mot_path = self._load_source(motion_source, "motion")
        vq_map, vq_ds, vq_dims, vq_path = self._load_source(visual_source, "visual_quality")

        active_sources = {
            "detection": det_map is not None,
            "motion": mot_map is not None,
            "visual_quality": vq_map is not None,
        }

        if not any(active_sources.values()):
            raise MissingSourceArtifactError("No valid input feature sources provided.")

        if not self.allow_partial and not all(active_sources.values()):
            missing = [k for k, v in active_sources.items() if not v]
            raise MissingSourceArtifactError(
                f"Missing required feature sources: {missing}. Set allow_partial=True to permit partial runs."
            )

        # 2. Dataset identity validation
        inferred_ds = dataset_id or det_ds or mot_ds or vq_ds or "unknown_dataset"
        detected_datasets = {
            k: ds for k, (ds, active) in {
                "detection": (det_ds, active_sources["detection"]),
                "motion": (mot_ds, active_sources["motion"]),
                "visual_quality": (vq_ds, active_sources["visual_quality"]),
            }.items() if active and ds is not None
        }

        unique_datasets = set(detected_datasets.values())
        if len(unique_datasets) > 1:
            raise DatasetIdMismatchError(
                f"Conflicting dataset identifiers detected: {detected_datasets}."
            )

        run_id = f"temporal_{inferred_ds}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # 3. Identify and sort all frame IDs
        all_frame_ids: Set[str] = set()
        if det_map:
            all_frame_ids.update(det_map.keys())
        if mot_map:
            all_frame_ids.update(mot_map.keys())
        if vq_map:
            all_frame_ids.update(vq_map.keys())

        if not all_frame_ids:
            raise FrameIdMismatchError("No frames found across any provided feature sources.")

        # Check frame overlap across active sources
        active_id_sets = [
            set(m.keys()) for m in [det_map, mot_map, vq_map] if m is not None
        ]
        if len(active_id_sets) > 1:
            common_ids = set.intersection(*active_id_sets)
            if not common_ids:
                raise FrameIdMismatchError(
                    "Zero overlapping frame identifiers found across active feature sources."
                )

        # Natural alphanumeric + chronological sorting
        sorted_frame_ids = sorted(
            list(all_frame_ids),
            key=lambda fid: self._get_sort_key(fid, det_map, mot_map, vq_map)
        )

        # 4. Filter frames based on start_frame, end_frame, max_frames
        is_partial_run = not all(active_sources.values())
        filtered_frame_ids: List[str] = []

        for fid in sorted_frame_ids:
            src_idx = self._resolve_source_frame_index(fid, det_map, mot_map, vq_map)
            if start_frame is not None and src_idx is not None and src_idx < start_frame:
                continue
            if end_frame is not None and src_idx is not None and src_idx > end_frame:
                continue
            filtered_frame_ids.append(fid)

        if start_frame is not None or end_frame is not None:
            is_partial_run = True

        if max_frames is not None and max_frames > 0 and len(filtered_frame_ids) > max_frames:
            filtered_frame_ids = filtered_frame_ids[:max_frames]
            is_partial_run = True

        # 5. Extract unified base frame records
        base_frame_records: List[Dict[str, Any]] = []
        for fid in filtered_frame_ids:
            base_rec, frame_errors = self._fuse_frame(
                fid=fid,
                dataset_id=inferred_ds,
                det_map=det_map,
                mot_map=mot_map,
                vq_map=vq_map,
            )
            base_frame_records.append(base_rec)
            errors.extend(frame_errors)

        # 6. Compute temporal window aggregations and trends
        temporal_records: List[FrameTemporalRecord] = self._compute_temporal_windows(
            base_records=base_frame_records,
            window_sizes=self.window_sizes,
            window_alignment=self.window_alignment,
            min_observations_for_trend=self.min_observations_for_trend,
            decimal_places=self.decimal_places,
        )

        duration = round(time.perf_counter() - t_start, 4)

        # 7. Build summary telemetry
        summary = self._build_summary(
            run_id=run_id,
            dataset_id=inferred_ds,
            records=temporal_records,
            errors=errors,
            active_sources=active_sources,
            window_sizes=self.window_sizes,
            window_alignment=self.window_alignment,
            is_partial=is_partial_run,
            duration=duration,
        )

        # 8. Resolve destination and persist artifacts
        target_dir = self._resolve_output_dir(output_dir, inferred_ds, run_id)
        if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
            raise OutputDirectoryExistsError(
                f"Output directory '{target_dir}' exists and is not empty. Pass overwrite=True to replace."
            )
        target_dir.mkdir(parents=True, exist_ok=True)

        raw_metadata = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": run_id,
            "dataset_id": inferred_ds,
            "extracted_at": summary.extracted_at,
            "source_artifact_paths": {
                "detection": str(det_path) if det_path else None,
                "motion": str(mot_path) if mot_path else None,
                "visual_quality": str(vq_path) if vq_path else None,
            },
            "active_sources": active_sources,
            "configured_window_sizes": list(self.window_sizes),
            "window_alignment": self.window_alignment,
            "min_observations_for_trend": self.min_observations_for_trend,
            "decimal_places": self.decimal_places,
            "is_partial": is_partial_run,
            "feature_dictionary": self._get_feature_dictionary(),
            "hardware_execution": {
                "cpu_only": True,
                "device": "cpu",
                "no_cuda_dependency": True,
            },
            "execution_environment": {
                "python_version": sys.version,
                "platform": sys.platform,
            },
        }

        self._persist_artifacts(
            target_dir=target_dir,
            records=temporal_records,
            summary=summary,
            errors=errors,
            raw_metadata=raw_metadata,
        )

        logger.info(
            f"Temporal feature aggregation complete in {duration}s. "
            f"Frames: {len(temporal_records)}, Aligned: {summary.successfully_aligned_frame_count}, "
            f"Partial: {summary.partially_aligned_frame_count}, Output: {target_dir}"
        )

        return {
            "output_dir": str(target_dir),
            "run_id": run_id,
            "summary": summary,
            "summary_dict": summary.to_dict(),
            "features": temporal_records,
            "records": temporal_records,
            "errors": errors,
            "processed_frame_count": summary.processed_frame_count,
            "successfully_aligned_frame_count": summary.successfully_aligned_frame_count,
            "partially_aligned_frame_count": summary.partially_aligned_frame_count,
            "duration_seconds": duration,
        }

    # --------------------------------------------------------------------------
    # Ingestion & Loading
    # --------------------------------------------------------------------------

    def _load_source(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]],
        source_type: str,
    ) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[str], Optional[Tuple[int, int]], Optional[Path]]:
        """Load and normalize feature records from path, directory, or dictionary."""
        if source is None:
            return None, None, None, None

        source_path: Optional[Path] = None
        raw_data: Any = None

        if isinstance(source, (str, Path)):
            p = Path(source).resolve()
            if not p.exists():
                raise InputArtifactError(f"Input source path does not exist: '{p}'")

            source_path = p
            if p.is_dir():
                target_file = self._find_source_file_in_dir(p, source_type)
                if target_file is None:
                    raise InputArtifactError(
                        f"Could not find valid {source_type} feature JSON in directory '{p}'"
                    )
                source_path = target_file

            if source_path.stat().st_size == 0:
                raise InputArtifactError(f"Input source artifact is empty (0 bytes): '{source_path}'")

            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except Exception as ex:
                raise InputArtifactError(f"Failed to parse JSON in '{source_path}': {ex}") from ex
        else:
            raw_data = source

        records_map, dataset_id, dimensions = self._normalize_records(raw_data, source_type)
        return records_map, dataset_id, dimensions, source_path

    def _find_source_file_in_dir(self, directory: Path, source_type: str) -> Optional[Path]:
        """Auto-detect appropriate frame-level feature JSON within a directory."""
        candidates = {
            "detection": ["frame_detection_features.json", "detection_features.json"],
            "motion": ["frame_motion_features.json", "motion_features.json"],
            "visual_quality": ["visual_quality_features.json"],
        }.get(source_type, [])

        for candidate_name in candidates:
            cand = directory / candidate_name
            if cand.exists() and cand.is_file():
                return cand

        # Fallback to search recursively for candidates
        for candidate_name in candidates:
            matches = list(directory.rglob(candidate_name))
            if matches:
                return matches[0]

        return None

    def _normalize_records(
        self,
        raw_data: Any,
        source_type: str,
    ) -> Tuple[Dict[str, Dict[str, Any]], Optional[str], Optional[Tuple[int, int]]]:
        """Convert varied JSON schema layouts into a standardized frame_id index."""
        records_list: List[Dict[str, Any]] = []
        dataset_id: Optional[str] = None
        dimensions: Optional[Tuple[int, int]] = None

        if isinstance(raw_data, dict):
            dataset_id = raw_data.get("dataset_id")
            if "frames" in raw_data and isinstance(raw_data["frames"], list):
                records_list = raw_data["frames"]
            elif "features" in raw_data and isinstance(raw_data["features"], list):
                records_list = raw_data["features"]
            elif "detections" in raw_data and isinstance(raw_data["detections"], list):
                records_list = self._aggregate_detections_by_frame(raw_data["detections"])
            elif "motions" in raw_data and isinstance(raw_data["motions"], list):
                records_list = self._aggregate_motions_by_frame(raw_data["motions"])
            else:
                records_list = []
        elif isinstance(raw_data, list):
            records_list = raw_data

        seen_fids: Set[str] = set()
        records_map: Dict[str, Dict[str, Any]] = {}

        for item in records_list:
            if not isinstance(item, dict):
                continue
            fid = item.get("frame_id")
            if not fid:
                continue
            fid = str(fid).strip()

            if fid in seen_fids:
                raise DuplicateFrameIdError(
                    f"Duplicate frame_id '{fid}' encountered in {source_type} source artifact."
                )
            seen_fids.add(fid)

            # Check dimensions if present
            w = item.get("image_width")
            h = item.get("image_height")
            if dimensions is None and isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                dimensions = (w, h)

            records_map[fid] = item

        return records_map, dataset_id, dimensions

    def _aggregate_detections_by_frame(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synthesize frame-level aggregates if provided raw detection-level array."""
        frame_groups: Dict[str, List[Dict[str, Any]]] = {}
        for d in detections:
            fid = d.get("frame_id")
            if fid:
                frame_groups.setdefault(str(fid), []).append(d)

        frames: List[Dict[str, Any]] = []
        dec = self.decimal_places

        for fid, group in frame_groups.items():
            first = group[0]
            confs = [d["confidence"] for d in group if "confidence" in d and d["confidence"] is not None]
            areas = [d["bbox_area"] for d in group if "bbox_area" in d and d["bbox_area"] is not None]
            widths = [d["bbox_width"] for d in group if "bbox_width" in d and d["bbox_width"] is not None]
            heights = [d["bbox_height"] for d in group if "bbox_height" in d and d["bbox_height"] is not None]

            frames.append({
                "frame_id": fid,
                "source_frame_index": first.get("source_frame_index"),
                "timestamp_seconds": first.get("timestamp_seconds"),
                "image_width": first.get("image_width"),
                "image_height": first.get("image_height"),
                "detection_count": len(group),
                "unique_class_count": len(set(d.get("class_name") for d in group if d.get("class_name"))),
                "mean_confidence": round(float(np.mean(confs)), dec) if confs else None,
                "minimum_confidence": round(float(min(confs)), dec) if confs else None,
                "maximum_confidence": round(float(max(confs)), dec) if confs else None,
                "confidence_std": round(float(np.std(confs)), dec) if confs else None,
                "total_bbox_area": round(float(sum(areas)), dec) if areas else None,
                "mean_bbox_area": round(float(np.mean(areas)), dec) if areas else None,
                "mean_bbox_width": round(float(np.mean(widths)), dec) if widths else None,
                "mean_bbox_height": round(float(np.mean(heights)), dec) if heights else None,
            })
        return frames

    def _aggregate_motions_by_frame(self, motions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synthesize frame-level aggregates if provided raw observation-level array."""
        frame_groups: Dict[str, List[Dict[str, Any]]] = {}
        for m in motions:
            fid = m.get("frame_id")
            if fid:
                frame_groups.setdefault(str(fid), []).append(m)

        frames: List[Dict[str, Any]] = []
        dec = self.decimal_places

        for fid, group in frame_groups.items():
            first = group[0]
            disps = [m["displacement"] for m in group if "displacement" in m and m["displacement"] is not None]
            speeds = [m["speed"] for m in group if "speed" in m and m["speed"] is not None]
            vxs = [m["velocity_x"] for m in group if "velocity_x" in m and m["velocity_x"] is not None]
            vys = [m["velocity_y"] for m in group if "velocity_y" in m and m["velocity_y"] is not None]

            moving = sum(1 for d in disps if d > 1.0)
            stationary = sum(1 for d in disps if d <= 1.0)

            frames.append({
                "frame_id": fid,
                "source_frame_index": first.get("source_frame_index"),
                "timestamp_seconds": first.get("timestamp_seconds"),
                "observed_track_count": len(group),
                "moving_track_count": moving,
                "stationary_track_count": stationary,
                "mean_displacement": round(float(np.mean(disps)), dec) if disps else None,
                "maximum_displacement": round(float(max(disps)), dec) if disps else None,
                "mean_speed": round(float(np.mean(speeds)), dec) if speeds else None,
                "mean_velocity_x": round(float(np.mean(vxs)), dec) if vxs else None,
                "mean_velocity_y": round(float(np.mean(vys)), dec) if vys else None,
                "motion_feature_coverage": round(len(disps) / len(group), dec) if group else 0.0,
            })
        return frames

    # --------------------------------------------------------------------------
    # Multi-Source Frame Fusion
    # --------------------------------------------------------------------------

    def _get_sort_key(
        self,
        fid: str,
        det_map: Optional[Dict[str, Any]],
        mot_map: Optional[Dict[str, Any]],
        vq_map: Optional[Dict[str, Any]],
    ) -> Tuple[int, str]:
        """Determine chronological sequence position for sorting frames."""
        idx = self._resolve_source_frame_index(fid, det_map, mot_map, vq_map)
        if idx is not None:
            return idx, fid
        parsed = extract_frame_index_from_id(fid)
        return parsed if parsed is not None else 999999999, fid

    def _resolve_source_frame_index(
        self,
        fid: str,
        det_map: Optional[Dict[str, Any]],
        mot_map: Optional[Dict[str, Any]],
        vq_map: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """Resolve source_frame_index with priority: detection -> motion -> visual -> parsed."""
        for m in [det_map, mot_map, vq_map]:
            if m and fid in m:
                val = m[fid].get("source_frame_index")
                if isinstance(val, int):
                    return val
        return extract_frame_index_from_id(fid)

    def _fuse_frame(
        self,
        fid: str,
        dataset_id: str,
        det_map: Optional[Dict[str, Any]],
        mot_map: Optional[Dict[str, Any]],
        vq_map: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[TemporalAggregationErrorRecord]]:
        """Combine available metrics from detection, motion, and visual quality."""
        frame_errors: List[TemporalAggregationErrorRecord] = []
        dec = self.decimal_places

        det_rec = det_map.get(fid) if det_map else None
        mot_rec = mot_map.get(fid) if mot_map else None
        vq_rec = vq_map.get(fid) if vq_map else None

        avail_det = det_rec is not None
        avail_mot = mot_rec is not None
        avail_vq = vq_rec is not None

        # Resolve metadata
        src_idx = None
        ts = None
        dims: List[Tuple[int, int]] = []

        for rec in [det_rec, mot_rec, vq_rec]:
            if rec:
                if src_idx is None and "source_frame_index" in rec and rec["source_frame_index"] is not None:
                    src_idx = int(rec["source_frame_index"])
                if ts is None and "timestamp_seconds" in rec and rec["timestamp_seconds"] is not None:
                    ts = round(float(rec["timestamp_seconds"]), dec)
                w = rec.get("image_width")
                h = rec.get("image_height")
                if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                    dims.append((w, h))

        # Check dimension consistency
        image_w, image_h = dims[0] if dims else (None, None)
        if len(dims) > 1:
            unique_dims = set(dims)
            if len(unique_dims) > 1:
                frame_errors.append(
                    TemporalAggregationErrorRecord(
                        frame_id=fid,
                        error_type="DIMENSION_MISMATCH",
                        message=f"Conflicting image dimensions across sources: {unique_dims}",
                    )
                )

        # 1. Detection Features (Task 2)
        det_features: Dict[str, Any] = {
            "detection_count": det_rec.get("detection_count", 0) if avail_det else None,
            "unique_class_count": det_rec.get("unique_class_count", 0) if avail_det else None,
            "mean_confidence": det_rec.get("mean_confidence") if avail_det else None,
            "minimum_confidence": det_rec.get("minimum_confidence") if avail_det else None,
            "maximum_confidence": det_rec.get("maximum_confidence") if avail_det else None,
            "confidence_std": det_rec.get("confidence_std") if avail_det else None,
            "total_bbox_area": det_rec.get("total_bbox_area") if avail_det else None,
            "mean_bbox_area": det_rec.get("mean_bbox_area") if avail_det else None,
            "mean_bbox_width": det_rec.get("mean_bbox_width") if avail_det else None,
            "mean_bbox_height": det_rec.get("mean_bbox_height") if avail_det else None,
        }

        # 2. Motion Features (Task 3)
        mot_features: Dict[str, Any] = {
            "observed_track_count": mot_rec.get("observed_track_count", 0) if avail_mot else None,
            "moving_track_count": mot_rec.get("moving_track_count", 0) if avail_mot else None,
            "stationary_track_count": mot_rec.get("stationary_track_count", 0) if avail_mot else None,
            "mean_displacement": mot_rec.get("mean_displacement") if avail_mot else None,
            "maximum_displacement": mot_rec.get("maximum_displacement") if avail_mot else None,
            "mean_speed": mot_rec.get("mean_speed") if avail_mot else None,
            "mean_velocity_x": mot_rec.get("mean_velocity_x") if avail_mot else None,
            "mean_velocity_y": mot_rec.get("mean_velocity_y") if avail_mot else None,
            "motion_feature_coverage": mot_rec.get("motion_feature_coverage") if avail_mot else None,
        }

        # 3. Visual Quality Features (Task 4)
        vq_features: Dict[str, Any] = {
            "blur_score": vq_rec.get("blur_score") if avail_vq else None,
            "mean_brightness": vq_rec.get("mean_brightness") if avail_vq else None,
            "median_brightness": vq_rec.get("median_brightness") if avail_vq else None,
            "brightness_std": vq_rec.get("brightness_std") if avail_vq else None,
            "contrast_std": vq_rec.get("contrast_std") if avail_vq else None,
            "contrast_range": vq_rec.get("contrast_range") if avail_vq else None,
            "contrast_percentile_range": vq_rec.get("contrast_percentile_range") if avail_vq else None,
            "image_aspect_ratio": vq_rec.get("image_aspect_ratio") if avail_vq else None,
        }

        # Source availability & coverage telemetry
        src_avail = SourceAvailabilityRecord(
            detection=avail_det,
            motion=avail_mot,
            visual_quality=avail_vq,
        )
        total_sources = sum([avail_det, avail_mot, avail_vq])
        src_cov = round(total_sources / 3.0, dec)

        # Count non-null numeric metrics across all 27 keys
        combined_values = {**det_features, **mot_features, **vq_features}
        valid_nums = sum(
            1 for k in NUMERICAL_FEATURE_KEYS
            if combined_values.get(k) is not None and not math.isnan(combined_values[k])
        )
        num_cov = round(valid_nums / len(NUMERICAL_FEATURE_KEYS), dec)

        feat_cov = FeatureCoverageRecord(
            source_coverage=src_cov,
            total_sources_available=total_sources,
            valid_numeric_feature_count=valid_nums,
            total_numeric_feature_count=len(NUMERICAL_FEATURE_KEYS),
            numeric_feature_coverage=num_cov,
        )

        status = "success" if total_sources == 3 else "partial"

        base_rec = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "frame_id": fid,
            "source_frame_index": src_idx,
            "timestamp_seconds": ts,
            "image_width": image_w,
            "image_height": image_h,
            "source_availability": src_avail,
            "feature_coverage": feat_cov,
            "detection_features": det_features,
            "motion_features": mot_features,
            "visual_quality_features": vq_features,
            "processing_status": status,
        }

        return base_rec, frame_errors

    # --------------------------------------------------------------------------
    # Temporal Window Aggregation & Trends
    # --------------------------------------------------------------------------

    def _compute_temporal_windows(
        self,
        base_records: List[Dict[str, Any]],
        window_sizes: Sequence[int],
        window_alignment: str,
        min_observations_for_trend: int,
        decimal_places: int,
    ) -> List[FrameTemporalRecord]:
        """Compute rolling aggregate statistics and linear trends across chronological frames."""
        N = len(base_records)
        output_records: List[FrameTemporalRecord] = []

        # Flattened lookup list of numeric values per frame for fast vector slicing
        frame_numerics: List[Dict[str, Optional[float]]] = []
        for r in base_records:
            numerics: Dict[str, Optional[float]] = {}
            for group_key in ["detection_features", "motion_features", "visual_quality_features"]:
                for k, v in r[group_key].items():
                    if k in NUMERICAL_FEATURE_KEYS:
                        numerics[k] = float(v) if v is not None and not math.isnan(v) else None
            frame_numerics.append(numerics)

        for i, rec in enumerate(base_records):
            windows_dict: Dict[str, Any] = {}
            primary_trends: Dict[str, Any] = {}

            for W in window_sizes:
                # Trailing Window Slice (STRICT: no future frames)
                if window_alignment == "trailing":
                    start_idx = max(0, i - W + 1)
                    end_idx = i  # inclusive
                elif window_alignment == "centered":
                    half = W // 2
                    start_idx = max(0, i - half)
                    end_idx = min(N - 1, i + half)
                elif window_alignment == "future":
                    start_idx = i
                    end_idx = min(N - 1, i + W - 1)
                else:
                    start_idx = max(0, i - W + 1)
                    end_idx = i

                slice_indices = list(range(start_idx, end_idx + 1))
                contributing_count = len(slice_indices)
                start_frame_id = base_records[start_idx]["frame_id"]
                end_frame_id = base_records[end_idx]["frame_id"]

                # 1. Feature Statistics
                feature_stats: Dict[str, Any] = {}
                coverage_ratios: List[float] = []

                for k in NUMERICAL_FEATURE_KEYS:
                    window_values = [frame_numerics[j][k] for j in slice_indices]
                    stat_obj = self._compute_feature_stats(window_values, decimal_places)
                    feature_stats[k] = stat_obj.to_dict()
                    coverage_ratios.append(stat_obj.coverage_ratio)

                mean_win_cov = round(float(np.mean(coverage_ratios)), decimal_places) if coverage_ratios else 0.0

                # 2. Window Trends
                window_trends: Dict[str, Any] = {}
                for trend_name, metric_key in TREND_FEATURE_MAPPINGS.items():
                    # Collect chronological valid pairs (relative_step, value)
                    valid_pairs: List[Tuple[int, float]] = []
                    for step_idx, j in enumerate(slice_indices):
                        val = frame_numerics[j][metric_key]
                        if val is not None and not math.isnan(val):
                            valid_pairs.append((step_idx, val))

                    trend_rec = calculate_ols_trend(
                        chronological_observations=valid_pairs,
                        decimal_places=decimal_places,
                        min_observations=min_observations_for_trend,
                    )
                    window_trends[trend_name] = trend_rec.to_dict()

                w_key = f"w{W}"
                windows_dict[w_key] = {
                    "window_size": W,
                    "window_alignment": window_alignment,
                    "window_start_frame": start_frame_id,
                    "window_end_frame": end_frame_id,
                    "contributing_frame_count": contributing_count,
                    "feature_coverage": mean_win_cov,
                    "feature_statistics": feature_stats,
                    "trends": window_trends,
                }

                # Default primary trends correspond to W=5 (or smallest window >= 3)
                if not primary_trends or W == 5 or (W == window_sizes[0] and not primary_trends):
                    primary_trends = window_trends

            frame_obj = FrameTemporalRecord(
                schema_version=rec["schema_version"],
                dataset_id=rec["dataset_id"],
                frame_id=rec["frame_id"],
                source_frame_index=rec["source_frame_index"],
                timestamp_seconds=rec["timestamp_seconds"],
                image_width=rec["image_width"],
                image_height=rec["image_height"],
                source_availability=rec["source_availability"],
                feature_coverage=rec["feature_coverage"],
                detection_features=rec["detection_features"],
                motion_features=rec["motion_features"],
                visual_quality_features=rec["visual_quality_features"],
                temporal_windows=windows_dict,
                temporal_trends=primary_trends,
                processing_status=rec["processing_status"],
            )
            output_records.append(frame_obj)

        return output_records

    def _compute_feature_stats(
        self,
        values: List[Optional[float]],
        decimal_places: int,
    ) -> WindowFeatureStatistics:
        """Compute summary statistics for a feature sequence within a window."""
        valid_vals = [v for v in values if v is not None and not math.isnan(v)]
        total = len(values)
        valid_count = len(valid_vals)
        missing_count = total - valid_count
        cov = round(valid_count / total, decimal_places) if total > 0 else 0.0

        if valid_count == 0:
            return WindowFeatureStatistics(
                mean=None,
                min=None,
                max=None,
                std=None,
                median=None,
                first=None,
                last=None,
                diff=None,
                valid_count=0,
                missing_count=missing_count,
                coverage_ratio=0.0,
            )

        mean_val = round(float(np.mean(valid_vals)), decimal_places)
        min_val = round(float(min(valid_vals)), decimal_places)
        max_val = round(float(max(valid_vals)), decimal_places)
        std_val = round(float(np.std(valid_vals)), decimal_places)
        median_val = round(float(np.median(valid_vals)), decimal_places)
        first_val = round(valid_vals[0], decimal_places)
        last_val = round(valid_vals[-1], decimal_places)
        diff_val = round(last_val - first_val, decimal_places)

        return WindowFeatureStatistics(
            mean=mean_val,
            min=min_val,
            max=max_val,
            std=std_val,
            median=median_val,
            first=first_val,
            last=last_val,
            diff=diff_val,
            valid_count=valid_count,
            missing_count=missing_count,
            coverage_ratio=cov,
        )

    # --------------------------------------------------------------------------
    # Summary & Output Persistence
    # --------------------------------------------------------------------------

    def _build_summary(
        self,
        run_id: str,
        dataset_id: str,
        records: List[FrameTemporalRecord],
        errors: List[TemporalAggregationErrorRecord],
        active_sources: Dict[str, bool],
        window_sizes: Sequence[int],
        window_alignment: str,
        is_partial: bool,
        duration: float,
    ) -> TemporalAggregationSummary:
        """Construct comprehensive dataset-level aggregation telemetry."""
        total_frames = len(records)
        aligned_count = sum(1 for r in records if r.processing_status == "success")
        partial_count = total_frames - aligned_count

        missing_counts = {
            "detection": sum(1 for r in records if not r.source_availability.detection),
            "motion": sum(1 for r in records if not r.source_availability.motion),
            "visual_quality": sum(1 for r in records if not r.source_availability.visual_quality),
        }

        avg_source_cov = (
            round(float(np.mean([r.feature_coverage.source_coverage for r in records])), self.decimal_places)
            if records else 0.0
        )
        avg_num_cov = (
            round(float(np.mean([r.feature_coverage.numeric_feature_coverage for r in records])), self.decimal_places)
            if records else 0.0
        )

        val_status = "PASS" if len(errors) == 0 and partial_count == 0 else "WARNING"

        return TemporalAggregationSummary(
            schema_version=FEATURE_SCHEMA_VERSION,
            run_id=run_id,
            dataset_id=dataset_id,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            is_partial=is_partial,
            processed_frame_count=total_frames,
            successfully_aligned_frame_count=aligned_count,
            partially_aligned_frame_count=partial_count,
            missing_source_counts=missing_counts,
            invalid_record_count=len(errors),
            configured_window_sizes=list(window_sizes),
            window_alignment=window_alignment,
            extraction_duration_seconds=duration,
            validation_status=val_status,
            feature_counts={
                "base_numerical_features": len(NUMERICAL_FEATURE_KEYS),
                "trend_features": len(TREND_FEATURE_MAPPINGS),
                "window_count": len(window_sizes),
                "statistics_per_feature": 8,
            },
            coverage_summary={
                "average_source_coverage": avg_source_cov,
                "average_numeric_feature_coverage": avg_num_cov,
            },
        )

    def _resolve_output_dir(
        self,
        output_dir: Optional[Union[str, Path]],
        dataset_id: str,
        run_id: str,
    ) -> Path:
        """Resolve target directory for temporal feature artifacts."""
        if output_dir:
            return Path(output_dir).resolve()
        return (
            Path("datasets")
            / "processed"
            / dataset_id
            / "features"
            / "temporal"
            / run_id
        ).resolve()

    def _persist_artifacts(
        self,
        target_dir: Path,
        records: List[FrameTemporalRecord],
        summary: TemporalAggregationSummary,
        errors: List[TemporalAggregationErrorRecord],
        raw_metadata: Dict[str, Any],
    ) -> None:
        """Atomically persist all Schema v1.0 output artifacts to target directory."""
        # 1. temporal_features.json (Full nested representation)
        temporal_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_type": "temporal_aggregated",
            "dataset_id": summary.dataset_id,
            "run_id": summary.run_id,
            "generated_at": summary.extracted_at,
            "total_frames": len(records),
            "features": [r.to_dict() for r in records],
        }
        self._write_json_atomically(target_dir / "temporal_features.json", temporal_payload)

        # 2. aggregated_frame_features.json (Flattened sequence-friendly representation)
        flat_records = [self._flatten_frame_record(r) for r in records]
        aggregated_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "dataset_id": summary.dataset_id,
            "run_id": summary.run_id,
            "total_frames": len(records),
            "frames": flat_records,
        }
        self._write_json_atomically(target_dir / "aggregated_frame_features.json", aggregated_payload)

        # 3. temporal_feature_summary.json
        self._write_json_atomically(target_dir / "temporal_feature_summary.json", summary.to_dict())

        # 4. feature_metadata.json
        self._write_json_atomically(target_dir / "feature_metadata.json", raw_metadata)

        # 5. feature_errors.json
        errors_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "total_errors": len(errors),
            "errors": [e.to_dict() for e in errors],
        }
        self._write_json_atomically(target_dir / "feature_errors.json", errors_payload)

    def _flatten_frame_record(self, record: FrameTemporalRecord) -> Dict[str, Any]:
        """Convert nested frame record into a flat dictionary suitable for tabular/tensor ingestion."""
        flat: Dict[str, Any] = {
            "frame_id": record.frame_id,
            "source_frame_index": record.source_frame_index,
            "timestamp_seconds": record.timestamp_seconds,
            "source_coverage": record.feature_coverage.source_coverage,
            "numeric_feature_coverage": record.feature_coverage.numeric_feature_coverage,
        }

        # Base features with clean prefix
        for k, v in record.detection_features.items():
            flat[f"det_{k}"] = v
        for k, v in record.motion_features.items():
            flat[f"mot_{k}"] = v
        for k, v in record.visual_quality_features.items():
            flat[f"vq_{k}"] = v

        # Temporal window statistics
        for w_key, w_data in record.temporal_windows.items():
            for feat_key, stat_obj in w_data.get("feature_statistics", {}).items():
                for stat_name in ["mean", "std", "min", "max", "diff"]:
                    flat[f"{w_key}_{feat_key}_{stat_name}"] = stat_obj.get(stat_name)
            for trend_name, trend_obj in w_data.get("trends", {}).items():
                flat[f"{w_key}_{trend_name}_slope"] = trend_obj.get("slope")

        return flat

    def _write_json_atomically(self, target_path: Path, data: Any) -> None:
        """Write JSON payload to temporary file before atomic filesystem rename."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(target_path)

    def _get_feature_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """Return schema registry of base, window, and trend feature definitions."""
        d: Dict[str, Dict[str, Any]] = {
            "detection_count": {"type": "int", "description": "Number of detected objects in the frame."},
            "mean_confidence": {"type": "float", "description": "Average detector confidence across detections."},
            "observed_track_count": {"type": "int", "description": "Total tracked objects present in the frame."},
            "mean_displacement": {"type": "float", "description": "Mean Euclidean pixel displacement between frames."},
            "blur_score": {"type": "float", "description": "Laplacian variance representing optical edge sharpness."},
            "mean_brightness": {"type": "float", "description": "Average grayscale pixel luminance [0, 255]."},
            "contrast_std": {"type": "float", "description": "RMS contrast: standard deviation of pixel intensities."},
        }
        for stat in ["mean", "min", "max", "std", "median", "first", "last", "diff"]:
            d[f"w<N>_<feature>_{stat}"] = {
                "type": "float",
                "description": f"Rolling trailing window {stat} metric for specified feature across <N> frames.",
            }
        for trend in TREND_FEATURE_MAPPINGS.keys():
            d[f"w<N>_{trend}"] = {
                "type": "dict",
                "description": f"Ordinary Least Squares linear slope for {trend} across trailing <N> frames.",
            }
        return d


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """CLI for PerceptGuard AI Temporal Feature Aggregator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Temporal Feature Aggregator CLI."
    )
    parser.add_argument(
        "--detection",
        "-d",
        type=str,
        default=None,
        help="Path to detection features JSON artifact or run directory.",
    )
    parser.add_argument(
        "--motion",
        "-m",
        type=str,
        default=None,
        help="Path to motion features JSON artifact or run directory.",
    )
    parser.add_argument(
        "--visual",
        "-v",
        type=str,
        default=None,
        help="Path to visual quality features JSON artifact or run directory.",
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
        help="Custom destination directory for output feature artifacts.",
    )
    parser.add_argument(
        "--window-sizes",
        "-w",
        type=str,
        default="3,5,10",
        help="Comma-separated temporal window lengths (e.g. 3,5,10).",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Optional dataset identifier override.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of chronological frames to evaluate.",
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
        "--allow-partial",
        action="store_true",
        help="Permit execution when one or two input feature sources are missing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target directory if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose diagnostic telemetry and stack traces.",
    )

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    cfg = FeatureConfig.from_yaml(args.config) if args.config else None

    # Parse window sizes
    try:
        w_sizes = [int(x.strip()) for x in args.window_sizes.split(",") if x.strip()]
    except Exception as ex:
        logger.error(f"Invalid window sizes format '{args.window_sizes}': {ex}")
        return 1

    aggregator = TemporalFeatureAggregator(
        config=cfg,
        window_sizes=w_sizes,
        allow_partial=args.allow_partial,
    )

    try:
        result = aggregator.aggregate(
            detection_source=args.detection,
            motion_source=args.motion,
            visual_source=args.visual,
            output_dir=args.output,
            dataset_id=args.dataset_id,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overwrite=args.overwrite,
        )

        sum_dict = result["summary"].to_dict() if hasattr(result["summary"], "to_dict") else result["summary"]
        print("\n" + "=" * 75)
        print("PERCEPTGUARD AI — TEMPORAL FEATURE AGGREGATION COMPLETE")
        print("=" * 75)
        print(f"  Dataset ID            : {sum_dict['dataset_id']}")
        print(f"  Run ID                : {sum_dict['run_id']}")
        print(f"  Status                : [{sum_dict['validation_status']}]")
        print(f"  Is Partial Run        : {sum_dict['is_partial']}")
        print(f"  Processed Frames      : {sum_dict['processed_frame_count']}")
        print(f"  Fully Aligned Frames  : {sum_dict['successfully_aligned_frame_count']}")
        print(f"  Partially Aligned     : {sum_dict['partially_aligned_frame_count']}")
        print(f"  Window Sizes          : {sum_dict['configured_window_sizes']} ({sum_dict['window_alignment']})")
        print(f"  Avg Source Coverage   : {sum_dict['coverage_summary']['average_source_coverage']}")
        print(f"  Avg Numeric Coverage  : {sum_dict['coverage_summary']['average_numeric_feature_coverage']}")
        print(f"  Duration              : {sum_dict['extraction_duration_seconds']}s")
        print(f"  Output Directory      : {result['output_dir']}")
        print("=" * 75 + "\n")
        return 0

    except Exception as ex:
        logger.error(f"Temporal feature aggregation failed: {ex}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
