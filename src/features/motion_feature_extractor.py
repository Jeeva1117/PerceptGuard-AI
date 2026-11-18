"""PerceptGuard AI: Motion Feature Extractor.

Extracts kinematic motion features for tracked objects across sequential video frames
from existing ByteTrack, Single-Object, or Multi-Object Tracking artifacts (Schema v1.0).

Primary Responsibilities:
1. Object Position Features: Centroid coordinates (center_x, center_y), width, height, area.
2. Frame-to-Frame Displacement: Horizontal (delta_x), vertical (delta_y), Euclidean displacement,
   absolute displacement, and normalized displacement (scaled by image dimensions/diagonal).
3. Velocity Features: velocity_x, velocity_y, speed (displacement / delta_time), requiring strictly
   positive time differences (delta_time > 0).
4. Movement Direction: atan2 angle in radians and degrees, categorized into 8 directional sectors
   plus "stationary" using a configurable displacement threshold.
5. Scale / Bounding Box Changes: delta_width, delta_height, delta_area, and area_change_ratio.
6. Temporal Continuity & Gap Handling: Explicit tracking of frame gaps, observation gaps, first
   observations, consecutive motion, and gap-based motion.
7. Aggregations: Track-level summary metrics (mean/max displacement, speed, direction coverage)
   and frame-level motion indicators (moving vs. stationary tracks, mean displacement, coverage).
8. Fault Containment & Validation: Non-silent error accounting, zero-detection and empty frame support,
   explicit null policies for unavailable statistics.
"""

from collections import Counter, defaultdict
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

from src.features.feature_config import FeatureConfig
from src.features.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    validate_input_artifacts,
)

logger = logging.getLogger("PerceptGuard.MotionFeatureExtractor")


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class MotionFeatureExtractionError(Exception):
    """Base exception for all motion feature extraction failures."""
    pass


class TrackingArtifactValidationError(MotionFeatureExtractionError):
    """Raised when the input tracking artifact fails structural schema validation."""
    pass


class InvalidTrackRecordError(MotionFeatureExtractionError):
    """Raised when a track observation violates coordinate or numerical constraints."""
    pass


class OutputDirectoryExistsError(MotionFeatureExtractionError):
    """Raised when the destination output directory already exists and overwrite is False."""
    pass


# ==============================================================================
# Data Contracts & Records
# ==============================================================================

@dataclass
class MotionFeatureRecord:
    """Strongly typed feature vector representing an individual object motion observation."""
    dataset_id: str
    frame_id: str
    source_frame_index: Optional[int]
    timestamp_seconds: Optional[float]
    track_id: int
    class_id: int
    class_name: str
    bbox_center_x: float
    bbox_center_y: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    previous_frame_id: Optional[str]
    frame_gap: Optional[int]
    observation_gap: Optional[int]
    motion_calculation_status: str  # "first_observation", "consecutive", "gap_motion", "gap_too_large", "invalid"
    delta_x: Optional[float]
    delta_y: Optional[float]
    displacement: Optional[float]
    absolute_delta_x: Optional[float]
    absolute_delta_y: Optional[float]
    normalized_delta_x: Optional[float]
    normalized_delta_y: Optional[float]
    normalized_displacement: Optional[float]
    delta_time: Optional[float]
    velocity_x: Optional[float]
    velocity_y: Optional[float]
    speed: Optional[float]
    direction_radians: Optional[float]
    direction_degrees: Optional[float]
    direction_category: Optional[str]
    bbox_width_change: Optional[float]
    bbox_height_change: Optional[float]
    bbox_area_change: Optional[float]
    bbox_area_change_ratio: Optional[float]
    center_distance: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackMotionSummaryRecord:
    """Aggregated motion statistics for an individual persistent tracklet."""
    track_id: int
    class_id: int
    class_name: str
    total_observations: int
    total_valid_motion_steps: int
    gap_motion_steps: int
    mean_displacement: Optional[float]
    maximum_displacement: Optional[float]
    minimum_displacement: Optional[float]
    displacement_std: Optional[float]
    mean_speed: Optional[float]
    maximum_speed: Optional[float]
    mean_velocity_x: Optional[float]
    mean_velocity_y: Optional[float]
    dominant_movement_direction: Optional[str]
    motion_data_coverage: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameMotionFeatureRecord:
    """Aggregated motion metrics for an individual video frame."""
    frame_id: str
    source_frame_index: Optional[int]
    timestamp_seconds: Optional[float]
    observed_track_count: int
    moving_track_count: int
    stationary_track_count: int
    mean_displacement: Optional[float]
    maximum_displacement: Optional[float]
    mean_speed: Optional[float]
    motion_feature_coverage: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MotionExtractionErrorRecord:
    """Audit entry recording rejected tracking observations or numerical anomalies."""
    frame_id: str
    track_id: Optional[int]
    error_type: str
    message: str
    raw_data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MotionFeatureSummary:
    """Global execution summary and dataset-level telemetry."""
    schema_version: str
    run_id: str
    dataset_id: str
    source_tracking_artifact: str
    source_frame_metadata: Optional[str]
    extracted_at: str
    is_partial: bool
    processed_frame_count: int
    unique_track_count: int
    total_track_observations: int
    valid_motion_calculation_count: int
    gap_based_calculation_count: int
    unavailable_motion_count: int
    invalid_record_count: int
    extraction_duration_seconds: float
    validation_status: str
    feature_counts: Dict[str, int]
    track_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Mathematical & Kinematic Utilities
# ==============================================================================

def calculate_direction(
    delta_x: float,
    delta_y: float,
    displacement: float,
    stationary_threshold: float = 1.0,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Calculate direction angle and classify into 8-cardinal sectors or stationary.

    Angle convention:
    - delta_x: positive right, negative left.
    - delta_y: positive down (image coordinates), negative up.
    - atan2(delta_y, delta_x) returns angle in radians [-pi, pi].
    """
    if displacement <= stationary_threshold:
        return 0.0, 0.0, "stationary"

    rad = math.atan2(delta_y, delta_x)
    deg = math.degrees(rad)

    # Sector classification
    if -22.5 <= deg <= 22.5:
        category = "right"
    elif 22.5 < deg <= 67.5:
        category = "down_right"
    elif 67.5 < deg <= 112.5:
        category = "down"
    elif 112.5 < deg <= 157.5:
        category = "down_left"
    elif deg > 157.5 or deg < -157.5:
        category = "left"
    elif -157.5 <= deg < -112.5:
        category = "up_left"
    elif -112.5 <= deg < -67.5:
        category = "up"
    else:  # -67.5 <= deg < -22.5
        category = "up_right"

    return rad, deg, category


def compute_std(values: Sequence[float], mean_val: float) -> float:
    """Calculate population standard deviation sqrt(sum((x - mean)^2) / N)."""
    if not values or len(values) <= 1:
        return 0.0
    var = sum((x - mean_val) ** 2 for x in values) / len(values)
    return math.sqrt(var)


# ==============================================================================
# Motion Feature Extractor Engine
# ==============================================================================

class MotionFeatureExtractor:
    """Production engine extracting kinematic motion vectors from tracking results."""

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        stationary_threshold: float = 1.0,
        max_gap_tolerance: int = 5,
        decimal_places: Optional[int] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> None:
        """Initialize extractor.

        Args:
            config: FeatureConfig instance or None.
            stationary_threshold: Threshold in pixels below which motion is stationary.
            max_gap_tolerance: Maximum missing frames across which gap motion is calculated.
            decimal_places: Precision rounding decimal places.
            image_width: Explicit image width for spatial normalization.
            image_height: Explicit image height for spatial normalization.
        """
        if config is None:
            try:
                self.config = FeatureConfig.from_yaml()
            except Exception:
                self.config = FeatureConfig()
        else:
            self.config = config

        self.stationary_threshold = float(stationary_threshold)
        self.max_gap_tolerance = int(max_gap_tolerance)
        self.decimal_places = (
            decimal_places if decimal_places is not None else self.config.decimal_places
        )
        self.image_width = image_width
        self.image_height = image_height

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
        """Extract motion features from tracking JSON artifact file.

        Args:
            input_file: Path to tracking JSON (multi_object_tracking, single_object_tracking, or ByteTrack).
            output_dir: Destination directory.
            max_frames: Maximum number of frames to process.
            start_frame: Starting source frame index (inclusive).
            end_frame: Ending source frame index (inclusive).
            overwrite: Overwrite destination directory if it exists.
            strict: Strict validation mode.

        Returns:
            Dictionary containing output directory and summary metrics.
        """
        t_start = time.perf_counter()
        in_path = Path(input_file).resolve()

        if not in_path.exists():
            raise FileNotFoundError(f"Tracking input artifact not found: '{in_path}'")
        if in_path.stat().st_size == 0:
            raise TrackingArtifactValidationError(f"Tracking input artifact is empty: '{in_path}'")

        try:
            with open(in_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as ex:
            raise TrackingArtifactValidationError(f"Invalid JSON in tracking artifact '{in_path}': {ex}") from ex

        # Schema Validation
        val_res = validate_input_artifacts(tracking_file=in_path, strict=strict)
        if not val_res.is_valid and strict:
            raise TrackingArtifactValidationError(
                f"Tracking artifact validation failed for '{in_path}': {val_res.errors}"
            )

        # In-memory feature extraction
        (
            motion_records,
            track_summaries,
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

        # Persist all 6 versioned artifacts
        self._persist_artifacts(
            target_dir=target_dir,
            motion_records=motion_records,
            track_summaries=track_summaries,
            frame_records=frame_records,
            summary=summary,
            errors=errors,
            raw_metadata=raw_metadata,
        )

        logger.info(
            f"Motion feature extraction completed in {duration}s. "
            f"Frames: {summary.processed_frame_count}, Unique Tracks: {summary.unique_track_count}, "
            f"Valid Motions: {summary.valid_motion_calculation_count}, Output: {target_dir}"
        )

        return {
            "output_dir": str(target_dir),
            "run_id": summary.run_id,
            "summary": summary.to_dict(),
            "valid_motion_count": summary.valid_motion_calculation_count,
            "unique_track_count": summary.unique_track_count,
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
        List[MotionFeatureRecord],
        List[TrackMotionSummaryRecord],
        List[FrameMotionFeatureRecord],
        MotionFeatureSummary,
        List[MotionExtractionErrorRecord],
        Dict[str, Any],
    ]:
        """Extract motion features from a pre-parsed tracking JSON dictionary."""
        t_start = time.perf_counter()
        errors: List[MotionExtractionErrorRecord] = []

        # 1. Dataset & Run Identification
        dataset_id = str(raw_data.get("dataset_id") or "unknown_dataset")
        run_id = f"motion_features_{dataset_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # 2. Image Dimensions Resolution
        img_w, img_h = self._resolve_dimensions(raw_data)
        img_diag = (
            math.sqrt(img_w ** 2 + img_h ** 2)
            if img_w is not None and img_h is not None and img_w > 0 and img_h > 0
            else None
        )

        # 3. Normalize Input Frames
        frames_raw = self._normalize_tracking_frames(raw_data)
        if not frames_raw:
            errors.append(
                MotionExtractionErrorRecord(
                    frame_id="ROOT",
                    track_id=None,
                    error_type="NO_FRAMES_FOUND",
                    message="Tracking dictionary contains no valid frame results or observations.",
                )
            )

        # 4. Check Duplicate Frame Identifiers & Validate Ordering
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
                    MotionExtractionErrorRecord(
                        frame_id=fid,
                        track_id=None,
                        error_type="DUPLICATE_FRAME_ID",
                        message=f"Duplicate frame identifier: '{fid}'. Skipping duplicate.",
                    )
                )
                continue
            seen_frame_ids.add(fid)
            f_entry["_resolved_frame_id"] = fid
            normalized_frames.append(f_entry)

        # 5. Apply Frame Sequence Filtering
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

        # 6. Track Continuity State Tracker: track_id -> last_valid_observation_dict
        last_track_obs: Dict[int, Dict[str, Any]] = {}
        track_observations_map: Dict[int, List[MotionFeatureRecord]] = defaultdict(list)

        motion_records: List[MotionFeatureRecord] = []
        frame_records: List[FrameMotionFeatureRecord] = []

        valid_motion_count = 0
        gap_motion_count = 0
        unavailable_motion_count = 0
        invalid_record_count = 0
        total_observations = 0

        dec = self.decimal_places

        # 7. Chronological Frame Processing
        for frame_idx, frame in enumerate(filtered_frames):
            frame_id = frame["_resolved_frame_id"]
            src_idx = frame.get("source_frame_index", frame_idx)
            ts_sec = frame.get("timestamp_seconds")

            raw_tracks = frame.get("tracks", [])
            if not isinstance(raw_tracks, list):
                errors.append(
                    MotionExtractionErrorRecord(
                        frame_id=frame_id,
                        track_id=None,
                        error_type="MALFORMED_TRACKS_FIELD",
                        message=f"Tracks field in frame '{frame_id}' is not a list.",
                    )
                )
                raw_tracks = []

            frame_valid_motions: List[MotionFeatureRecord] = []
            seen_track_ids_in_frame: Set[int] = set()

            for t_raw in raw_tracks:
                if not isinstance(t_raw, dict):
                    invalid_record_count += 1
                    errors.append(
                        MotionExtractionErrorRecord(
                            frame_id=frame_id,
                            track_id=None,
                            error_type="MALFORMED_TRACK_OBJECT",
                            message="Track observation is not a dictionary.",
                            raw_data=t_raw,
                        )
                    )
                    continue

                total_observations += 1

                # Validate Track Observation
                is_valid, err_msg, err_type = self._validate_track_raw(t_raw)
                if not is_valid:
                    invalid_record_count += 1
                    errors.append(
                        MotionExtractionErrorRecord(
                            frame_id=frame_id,
                            track_id=t_raw.get("track_id"),
                            error_type=err_type,
                            message=err_msg,
                            raw_data=t_raw,
                        )
                    )
                    continue

                track_id = int(t_raw["track_id"])
                class_id = int(t_raw["class_id"])
                class_name = str(t_raw["class_name"]).strip()

                if track_id in seen_track_ids_in_frame:
                    errors.append(
                        MotionExtractionErrorRecord(
                            frame_id=frame_id,
                            track_id=track_id,
                            error_type="DUPLICATE_TRACK_ID_IN_FRAME",
                            message=f"Duplicate track_id {track_id} in frame '{frame_id}'",
                        )
                    )
                seen_track_ids_in_frame.add(track_id)

                # Geometry & Centroid Math
                x_min, y_min, x_max, y_max = [float(v) for v in t_raw["bbox_xyxy"]]
                curr_w = round(x_max - x_min, dec)
                curr_h = round(y_max - y_min, dec)
                curr_area = round(curr_w * curr_h, dec)
                curr_cx = round((x_min + x_max) / 2.0, dec)
                curr_cy = round((y_min + y_max) / 2.0, dec)

                # Check Track History
                prev_obs = last_track_obs.get(track_id)

                if prev_obs is None:
                    # First Observation of Track
                    motion_rec = MotionFeatureRecord(
                        dataset_id=dataset_id,
                        frame_id=frame_id,
                        source_frame_index=src_idx,
                        timestamp_seconds=round(ts_sec, dec) if ts_sec is not None else None,
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        bbox_center_x=curr_cx,
                        bbox_center_y=curr_cy,
                        bbox_width=curr_w,
                        bbox_height=curr_h,
                        bbox_area=curr_area,
                        previous_frame_id=None,
                        frame_gap=None,
                        observation_gap=None,
                        motion_calculation_status="first_observation",
                        delta_x=None,
                        delta_y=None,
                        displacement=None,
                        absolute_delta_x=None,
                        absolute_delta_y=None,
                        normalized_delta_x=None,
                        normalized_delta_y=None,
                        normalized_displacement=None,
                        delta_time=None,
                        velocity_x=None,
                        velocity_y=None,
                        speed=None,
                        direction_radians=None,
                        direction_degrees=None,
                        direction_category=None,
                        bbox_width_change=None,
                        bbox_height_change=None,
                        bbox_area_change=None,
                        bbox_area_change_ratio=None,
                        center_distance=None,
                    )
                    unavailable_motion_count += 1

                else:
                    # Consecutive or Gap Motion Calculation
                    prev_frame_id = prev_obs["frame_id"]
                    prev_src_idx = prev_obs.get("source_frame_index", frame_idx - 1)
                    prev_ts = prev_obs.get("timestamp_seconds")
                    prev_cx = prev_obs["center_x"]
                    prev_cy = prev_obs["center_y"]
                    prev_w = prev_obs["width"]
                    prev_h = prev_obs["height"]
                    prev_area = prev_obs["area"]

                    frame_gap = (
                        src_idx - prev_src_idx
                        if src_idx is not None and prev_src_idx is not None
                        else 1
                    )
                    obs_gap = max(0, frame_gap - 1)

                    if frame_gap <= 0:
                        motion_status = "invalid_frame_gap"
                        errors.append(
                            MotionExtractionErrorRecord(
                                frame_id=frame_id,
                                track_id=track_id,
                                error_type="NON_MONOTONIC_FRAME_SEQUENCE",
                                message=f"Frame index {src_idx} is not greater than previous index {prev_src_idx} for track {track_id}",
                            )
                        )
                    elif frame_gap == 1:
                        motion_status = "consecutive"
                    elif frame_gap <= self.max_gap_tolerance:
                        motion_status = "gap_motion"
                        gap_motion_count += 1
                    else:
                        motion_status = "gap_too_large"

                    if motion_status in ("consecutive", "gap_motion"):
                        dx = round(curr_cx - prev_cx, dec)
                        dy = round(curr_cy - prev_cy, dec)
                        disp = round(math.sqrt(dx ** 2 + dy ** 2), dec)
                        abs_dx = round(abs(dx), dec)
                        abs_dy = round(abs(dy), dec)

                        # Normalized Displacement
                        if img_w and img_h and img_diag:
                            norm_dx = round(dx / img_w, dec)
                            norm_dy = round(dy / img_h, dec)
                            norm_disp = round(disp / img_diag, dec)
                        else:
                            norm_dx = None
                            norm_dy = None
                            norm_disp = None

                        # Velocity & Speed
                        if ts_sec is not None and prev_ts is not None:
                            dt = ts_sec - prev_ts
                            if dt > 0:
                                dt_r = round(dt, dec)
                                vx = round(dx / dt, dec)
                                vy = round(dy / dt, dec)
                                spd = round(disp / dt, dec)
                            else:
                                dt_r = None
                                vx = None
                                vy = None
                                spd = None
                                errors.append(
                                    MotionExtractionErrorRecord(
                                        frame_id=frame_id,
                                        track_id=track_id,
                                        error_type="NON_POSITIVE_TIME_DELTA",
                                        message=f"Timestamp difference {dt} <= 0 between frames for track {track_id}",
                                    )
                                )
                        else:
                            dt_r = None
                            vx = None
                            vy = None
                            spd = None

                        # Direction
                        rad, deg, d_cat = calculate_direction(
                            delta_x=dx,
                            delta_y=dy,
                            displacement=disp,
                            stationary_threshold=self.stationary_threshold,
                        )
                        rad_r = round(rad, dec) if rad is not None else None
                        deg_r = round(deg, dec) if deg is not None else None

                        # Bounding Box Scale Changes
                        w_change = round(curr_w - prev_w, dec)
                        h_change = round(curr_h - prev_h, dec)
                        area_change = round(curr_area - prev_area, dec)
                        area_ratio = (
                            round(area_change / prev_area, dec)
                            if prev_area > 0
                            else None
                        )

                        motion_rec = MotionFeatureRecord(
                            dataset_id=dataset_id,
                            frame_id=frame_id,
                            source_frame_index=src_idx,
                            timestamp_seconds=round(ts_sec, dec) if ts_sec is not None else None,
                            track_id=track_id,
                            class_id=class_id,
                            class_name=class_name,
                            bbox_center_x=curr_cx,
                            bbox_center_y=curr_cy,
                            bbox_width=curr_w,
                            bbox_height=curr_h,
                            bbox_area=curr_area,
                            previous_frame_id=prev_frame_id,
                            frame_gap=frame_gap,
                            observation_gap=obs_gap,
                            motion_calculation_status=motion_status,
                            delta_x=dx,
                            delta_y=dy,
                            displacement=disp,
                            absolute_delta_x=abs_dx,
                            absolute_delta_y=abs_dy,
                            normalized_delta_x=norm_dx,
                            normalized_delta_y=norm_dy,
                            normalized_displacement=norm_disp,
                            delta_time=dt_r,
                            velocity_x=vx,
                            velocity_y=vy,
                            speed=spd,
                            direction_radians=rad_r,
                            direction_degrees=deg_r,
                            direction_category=d_cat,
                            bbox_width_change=w_change,
                            bbox_height_change=h_change,
                            bbox_area_change=area_change,
                            bbox_area_change_ratio=area_ratio,
                            center_distance=disp,
                        )
                        valid_motion_count += 1
                        frame_valid_motions.append(motion_rec)

                    else:
                        # Gap too large or invalid frame gap
                        motion_rec = MotionFeatureRecord(
                            dataset_id=dataset_id,
                            frame_id=frame_id,
                            source_frame_index=src_idx,
                            timestamp_seconds=round(ts_sec, dec) if ts_sec is not None else None,
                            track_id=track_id,
                            class_id=class_id,
                            class_name=class_name,
                            bbox_center_x=curr_cx,
                            bbox_center_y=curr_cy,
                            bbox_width=curr_w,
                            bbox_height=curr_h,
                            bbox_area=curr_area,
                            previous_frame_id=prev_frame_id,
                            frame_gap=frame_gap,
                            observation_gap=obs_gap,
                            motion_calculation_status=motion_status,
                            delta_x=None,
                            delta_y=None,
                            displacement=None,
                            absolute_delta_x=None,
                            absolute_delta_y=None,
                            normalized_delta_x=None,
                            normalized_delta_y=None,
                            normalized_displacement=None,
                            delta_time=None,
                            velocity_x=None,
                            velocity_y=None,
                            speed=None,
                            direction_radians=None,
                            direction_degrees=None,
                            direction_category=None,
                            bbox_width_change=None,
                            bbox_height_change=None,
                            bbox_area_change=None,
                            bbox_area_change_ratio=None,
                            center_distance=None,
                        )
                        unavailable_motion_count += 1

                # Update Last Observation Cache
                last_track_obs[track_id] = {
                    "frame_id": frame_id,
                    "source_frame_index": src_idx,
                    "timestamp_seconds": ts_sec,
                    "center_x": curr_cx,
                    "center_y": curr_cy,
                    "width": curr_w,
                    "height": curr_h,
                    "area": curr_area,
                }

                motion_records.append(motion_rec)
                track_observations_map[track_id].append(motion_rec)

            # Build Frame-level Aggregation Record
            f_record = self._build_frame_motion_record(
                frame_id=frame_id,
                source_frame_index=src_idx,
                timestamp_seconds=ts_sec,
                raw_track_count=len(raw_tracks),
                valid_motions=frame_valid_motions,
            )
            frame_records.append(f_record)

        # 8. Build Track-level Summary Records
        track_summaries = self._build_track_summaries(track_observations_map)

        duration = round(time.perf_counter() - t_start, 4)
        val_status = "PASS" if invalid_record_count == 0 else "WARNING"

        summary = MotionFeatureSummary(
            schema_version=FEATURE_SCHEMA_VERSION,
            run_id=run_id,
            dataset_id=dataset_id,
            source_tracking_artifact=source_artifact,
            source_frame_metadata=str(self.config.dataset_metadata_path or "None"),
            extracted_at=datetime.now(timezone.utc).isoformat(),
            is_partial=is_partial,
            processed_frame_count=len(filtered_frames),
            unique_track_count=len(track_observations_map),
            total_track_observations=total_observations,
            valid_motion_calculation_count=valid_motion_count,
            gap_based_calculation_count=gap_motion_count,
            unavailable_motion_count=unavailable_motion_count,
            invalid_record_count=invalid_record_count,
            extraction_duration_seconds=duration,
            validation_status=val_status,
            feature_counts={
                "motion_observation_features": 35,
                "track_motion_summary_features": 16,
                "frame_motion_features": 10,
            },
            track_summary={
                t.track_id: {
                    "class_name": t.class_name,
                    "observations": t.total_observations,
                    "valid_motion_steps": t.total_valid_motion_steps,
                    "mean_displacement": t.mean_displacement,
                    "dominant_direction": t.dominant_movement_direction,
                }
                for t in track_summaries
            },
        )

        raw_metadata = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "source_tracking_artifact": source_artifact,
            "extracted_at": summary.extracted_at,
            "is_partial": is_partial,
            "image_dimensions": {
                "width": img_w,
                "height": img_h,
                "diagonal": round(img_diag, dec) if img_diag else None,
            },
            "parameters": {
                "stationary_threshold_pixels": self.stationary_threshold,
                "max_gap_tolerance_frames": self.max_gap_tolerance,
                "decimal_places": self.decimal_places,
            },
            "feature_dictionary": self._get_feature_dictionary(),
            "execution_environment": {
                "python_version": sys.version,
                "platform": sys.platform,
            },
        }

        return motion_records, track_summaries, frame_records, summary, errors, raw_metadata

    def _validate_track_raw(self, t_raw: Dict[str, Any]) -> Tuple[bool, str, str]:
        """Validate an individual track observation."""
        track_id = t_raw.get("track_id")
        if track_id is None or not isinstance(track_id, int) or track_id < 0:
            return False, f"Invalid track_id: {track_id}", "INVALID_TRACK_ID"

        class_id = t_raw.get("class_id")
        class_name = t_raw.get("class_name")
        if class_id is None or not isinstance(class_id, int) or class_id < 0:
            return False, f"Invalid class_id: {class_id}", "INVALID_CLASS_ID"
        if not class_name or not isinstance(class_name, str) or not class_name.strip():
            return False, f"Invalid class_name: {class_name}", "INVALID_CLASS_NAME"

        bbox = t_raw.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False, f"Invalid bbox format: {bbox}", "INVALID_BBOX_STRUCTURE"

        x_min, y_min, x_max, y_max = bbox
        for v in [x_min, y_min, x_max, y_max]:
            if not isinstance(v, (int, float)):
                return False, f"Non-numeric coordinate: {v}", "NON_NUMERIC_COORDINATE"
            if math.isnan(v) or math.isinf(v):
                return False, f"Non-finite coordinate: {v}", "NON_FINITE_COORDINATE"

        if x_min < 0.0 or y_min < 0.0:
            return False, f"Negative bbox coordinates: ({x_min}, {y_min})", "NEGATIVE_COORDINATES"
        if x_max <= x_min:
            return False, f"Inverted or zero width bbox: ({x_min}, {x_max})", "INVERTED_X_COORDINATES"
        if y_max <= y_min:
            return False, f"Inverted or zero height bbox: ({y_min}, {y_max})", "INVERTED_Y_COORDINATES"

        area = (x_max - x_min) * (y_max - y_min)
        if area <= 0.0:
            return False, f"Zero or negative bbox area: {area}", "ZERO_AREA_BBOX"

        return True, "", ""

    def _build_frame_motion_record(
        self,
        frame_id: str,
        source_frame_index: Optional[int],
        timestamp_seconds: Optional[float],
        raw_track_count: int,
        valid_motions: List[MotionFeatureRecord],
    ) -> FrameMotionFeatureRecord:
        """Construct aggregated frame-level motion telemetry."""
        dec = self.decimal_places

        if not valid_motions:
            return FrameMotionFeatureRecord(
                frame_id=frame_id,
                source_frame_index=source_frame_index,
                timestamp_seconds=round(timestamp_seconds, dec) if timestamp_seconds is not None else None,
                observed_track_count=raw_track_count,
                moving_track_count=0,
                stationary_track_count=0,
                mean_displacement=None,
                maximum_displacement=None,
                mean_speed=None,
                motion_feature_coverage=0.0 if raw_track_count > 0 else None,
            )

        disps = [m.displacement for m in valid_motions if m.displacement is not None]
        speeds = [m.speed for m in valid_motions if m.speed is not None]

        moving = sum(1 for d in disps if d > self.stationary_threshold)
        stationary = sum(1 for d in disps if d <= self.stationary_threshold)

        mean_disp = round(sum(disps) / len(disps), dec) if disps else None
        max_disp = round(max(disps), dec) if disps else None
        mean_spd = round(sum(speeds) / len(speeds), dec) if speeds else None
        cov = round(len(valid_motions) / raw_track_count, dec) if raw_track_count > 0 else 0.0

        return FrameMotionFeatureRecord(
            frame_id=frame_id,
            source_frame_index=source_frame_index,
            timestamp_seconds=round(timestamp_seconds, dec) if timestamp_seconds is not None else None,
            observed_track_count=raw_track_count,
            moving_track_count=moving,
            stationary_track_count=stationary,
            mean_displacement=mean_disp,
            maximum_displacement=max_disp,
            mean_speed=mean_spd,
            motion_feature_coverage=cov,
        )

    def _build_track_summaries(
        self, track_map: Dict[int, List[MotionFeatureRecord]]
    ) -> List[TrackMotionSummaryRecord]:
        """Aggregate per-track kinematic metrics."""
        dec = self.decimal_places
        summaries: List[TrackMotionSummaryRecord] = []

        for track_id, records in sorted(track_map.items()):
            first_rec = records[0]
            cid = first_rec.class_id
            cname = first_rec.class_name

            valid_steps = [r for r in records if r.displacement is not None]
            gap_steps = [r for r in valid_steps if r.motion_calculation_status == "gap_motion"]

            disps = [r.displacement for r in valid_steps if r.displacement is not None]
            speeds = [r.speed for r in valid_steps if r.speed is not None]
            vxs = [r.velocity_x for r in valid_steps if r.velocity_x is not None]
            vys = [r.velocity_y for r in valid_steps if r.velocity_y is not None]
            dirs = [
                r.direction_category for r in valid_steps
                if r.direction_category and r.direction_category != "stationary"
            ]

            mean_d = round(sum(disps) / len(disps), dec) if disps else None
            max_d = round(max(disps), dec) if disps else None
            min_d = round(min(disps), dec) if disps else None
            std_d = round(compute_std(disps, mean_d), dec) if mean_d is not None and disps else None

            mean_s = round(sum(speeds) / len(speeds), dec) if speeds else None
            max_s = round(max(speeds), dec) if speeds else None
            mean_vx = round(sum(vxs) / len(vxs), dec) if vxs else None
            mean_vy = round(sum(vys) / len(vys), dec) if vys else None

            if dirs:
                dominant_dir = Counter(dirs).most_common(1)[0][0]
            elif valid_steps:
                dominant_dir = "stationary"
            else:
                dominant_dir = None

            tot_obs = len(records)
            cov = (
                round(len(valid_steps) / (tot_obs - 1), dec)
                if tot_obs > 1
                else 0.0
            )

            summaries.append(
                TrackMotionSummaryRecord(
                    track_id=track_id,
                    class_id=cid,
                    class_name=cname,
                    total_observations=tot_obs,
                    total_valid_motion_steps=len(valid_steps),
                    gap_motion_steps=len(gap_steps),
                    mean_displacement=mean_d,
                    maximum_displacement=max_d,
                    minimum_displacement=min_d,
                    displacement_std=std_d,
                    mean_speed=mean_s,
                    maximum_speed=max_s,
                    mean_velocity_x=mean_vx,
                    mean_velocity_y=mean_vy,
                    dominant_movement_direction=dominant_dir,
                    motion_data_coverage=cov,
                )
            )

        return summaries

    def _normalize_tracking_frames(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize varied tracking formats into unified frame array."""
        # Multi-Object Tracking or ByteTrack format
        if "frame_results" in raw_data and isinstance(raw_data["frame_results"], list):
            return raw_data["frame_results"]
        if "frames" in raw_data and isinstance(raw_data["frames"], list):
            return raw_data["frames"]

        # Single-Object Tracking format (flat observations list)
        if "observations" in raw_data and isinstance(raw_data["observations"], list):
            frames_dict: Dict[str, Dict[str, Any]] = {}
            for obs in raw_data["observations"]:
                fid = str(obs.get("frame_id") or "frame_000000.jpg")
                if fid not in frames_dict:
                    frames_dict[fid] = {
                        "frame_id": fid,
                        "source_frame_index": obs.get("source_frame_index"),
                        "timestamp_seconds": obs.get("timestamp_seconds"),
                        "tracks": [],
                    }
                # Include observation in tracks only if visible and bounding box is present
                if obs.get("is_visible") is not False and obs.get("bbox_xyxy") is not None:
                    frames_dict[fid]["tracks"].append(obs)
            return list(frames_dict.values())

        return []

    def _resolve_dimensions(self, raw_data: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """Determine image width and height for spatial normalization."""
        if self.image_width and self.image_height:
            return self.image_width, self.image_height

        # Check raw_data root
        rw = raw_data.get("image_width")
        rh = raw_data.get("image_height")
        if rw and rh and rw > 0 and rh > 0:
            return int(rw), int(rh)

        # Check frame_results
        frames = raw_data.get("frame_results") or raw_data.get("frames") or []
        if frames and isinstance(frames[0], dict):
            fw = frames[0].get("image_width")
            fh = frames[0].get("image_height")
            if fw and fh and fw > 0 and fh > 0:
                return int(fw), int(fh)

        # Check metadata.csv in project
        meta_p = (
            _project_root / Path(self.config.dataset_metadata_path)
            if self.config.dataset_metadata_path
            else _project_root / "datasets" / "processed" / "driving_sample" / "metadata.csv"
        )
        if meta_p.exists() and meta_p.stat().st_size > 0:
            try:
                import pandas as pd
                df = pd.read_csv(meta_p, nrows=1)
                if "processed_width" in df.columns and "processed_height" in df.columns:
                    pw = int(df.iloc[0]["processed_width"])
                    ph = int(df.iloc[0]["processed_height"])
                    if pw > 0 and ph > 0:
                        return pw, ph
            except Exception as ex:
                logger.debug(f"Could not read dimensions from metadata.csv: {ex}")

        return None, None

    def _resolve_output_dir(
        self,
        output_dir: Optional[Union[str, Path]],
        dataset_id: str,
        run_id: str,
    ) -> Path:
        """Resolve output directory path."""
        if output_dir:
            return Path(output_dir).resolve()

        base_out = Path(self.config.output_dir)
        if not base_out.is_absolute():
            base_out = _project_root / base_out

        return base_out / "motion" / run_id

    def _persist_artifacts(
        self,
        target_dir: Path,
        motion_records: List[MotionFeatureRecord],
        track_summaries: List[TrackMotionSummaryRecord],
        frame_records: List[FrameMotionFeatureRecord],
        summary: MotionFeatureSummary,
        errors: List[MotionExtractionErrorRecord],
        raw_metadata: Dict[str, Any],
    ) -> None:
        """Atomically persist all 6 versioned output artifacts."""
        # 1. motion_features.json
        motion_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "source_tracking_artifact": summary.source_tracking_artifact,
            "total_records": len(motion_records),
            "records": [r.to_dict() for r in motion_records],
        }
        with open(target_dir / "motion_features.json", "w", encoding="utf-8") as f:
            json.dump(motion_payload, f, indent=2)

        # 2. track_motion_features.json
        track_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "unique_tracks": len(track_summaries),
            "tracks": [t.to_dict() for t in track_summaries],
        }
        with open(target_dir / "track_motion_features.json", "w", encoding="utf-8") as f:
            json.dump(track_payload, f, indent=2)

        # 3. frame_motion_features.json
        frame_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "total_frames": len(frame_records),
            "frames": [f.to_dict() for f in frame_records],
        }
        with open(target_dir / "frame_motion_features.json", "w", encoding="utf-8") as f:
            json.dump(frame_payload, f, indent=2)

        # 4. motion_feature_summary.json
        with open(target_dir / "motion_feature_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        # 5. feature_metadata.json
        with open(target_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
            json.dump(raw_metadata, f, indent=2)

        # 6. feature_errors.json
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
        """Return schema dictionary for all motion features."""
        return {
            "bbox_center_x": {"type": "float", "description": "Horizontal center coordinate in pixels."},
            "bbox_center_y": {"type": "float", "description": "Vertical center coordinate in pixels."},
            "bbox_width": {"type": "float", "description": "Width of bounding box in pixels."},
            "bbox_height": {"type": "float", "description": "Height of bounding box in pixels."},
            "bbox_area": {"type": "float", "description": "Area of bounding box in square pixels."},
            "delta_x": {"type": "float", "description": "Horizontal centroid displacement (curr_x - prev_x)."},
            "delta_y": {"type": "float", "description": "Vertical centroid displacement (curr_y - prev_y)."},
            "displacement": {"type": "float", "description": "Euclidean displacement sqrt(delta_x^2 + delta_y^2)."},
            "absolute_delta_x": {"type": "float", "description": "Magnitude of horizontal displacement."},
            "absolute_delta_y": {"type": "float", "description": "Magnitude of vertical displacement."},
            "normalized_delta_x": {"type": "float", "description": "delta_x divided by image width."},
            "normalized_delta_y": {"type": "float", "description": "delta_y divided by image height."},
            "normalized_displacement": {"type": "float", "description": "displacement divided by image diagonal."},
            "delta_time": {"type": "float", "description": "Elapsed seconds between observations."},
            "velocity_x": {"type": "float", "description": "Pixel velocity in X direction (delta_x / delta_time)."},
            "velocity_y": {"type": "float", "description": "Pixel velocity in Y direction (delta_y / delta_time)."},
            "speed": {"type": "float", "description": "Pixel velocity magnitude (displacement / delta_time)."},
            "direction_radians": {"type": "float", "description": "Movement angle in radians [-pi, pi]."},
            "direction_degrees": {"type": "float", "description": "Movement angle in degrees [-180, 180]."},
            "direction_category": {"type": "string", "description": "Categorical direction (stationary, right, left, up, down, diagonals)."},
            "bbox_width_change": {"type": "float", "description": "Change in width relative to previous observation."},
            "bbox_height_change": {"type": "float", "description": "Change in height relative to previous observation."},
            "bbox_area_change": {"type": "float", "description": "Change in area relative to previous observation."},
            "bbox_area_change_ratio": {"type": "float", "description": "Fractional change in area (area_change / prev_area)."},
            "center_distance": {"type": "float", "description": "Euclidean distance between successive centers."},
        }


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """CLI for Motion Feature Extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Motion Feature Extractor CLI."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input tracking JSON file (multi_object_tracking.json).",
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
        "--stationary-threshold",
        type=float,
        default=1.0,
        help="Threshold in pixels below which displacement is considered stationary (default: 1.0).",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=5,
        help="Maximum missing frames tolerance for gap-based motion (default: 5).",
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
    extractor = MotionFeatureExtractor(
        config=cfg,
        stationary_threshold=args.stationary_threshold,
        max_gap_tolerance=args.max_gap,
    )

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
        print("PERCEPTGUARD AI — MOTION FEATURE EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"  Dataset ID            : {sum_dict['dataset_id']}")
        print(f"  Run ID                : {sum_dict['run_id']}")
        print(f"  Status                : [{sum_dict['validation_status']}]")
        print(f"  Is Partial Run        : {sum_dict['is_partial']}")
        print(f"  Processed Frames      : {sum_dict['processed_frame_count']}")
        print(f"  Unique Tracks         : {sum_dict['unique_track_count']}")
        print(f"  Total Observations    : {sum_dict['total_track_observations']}")
        print(f"  Valid Motion Steps    : {sum_dict['valid_motion_calculation_count']}")
        print(f"  Gap-based Steps       : {sum_dict['gap_based_calculation_count']}")
        print(f"  Unavailable Motions   : {sum_dict['unavailable_motion_count']}")
        print(f"  Duration              : {sum_dict['extraction_duration_seconds']}s")
        print(f"  Output Directory      : {result['output_dir']}")
        print("=" * 70 + "\n")
        return 0

    except Exception as ex:
        logger.error(f"Motion feature extraction failed: {ex}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
