"""PerceptGuard AI: Production Multi-Object Tracking (MOT) Architecture.

Senior Staff-level, production-grade Multi-Object Tracking system integrating
YOLOv8/11 object detection with ByteTrack state estimation. Tracks multiple dynamic
and stationary obstacles across sequential video frames with persistent identity
management, formal lifecycle tracking, kinematic trajectory analysis, and
multi-file Schema v1.0 persistence.

Key Architectural Guarantees:
- Pure CPU execution (Intel Core i3 compatible, 8 GB RAM bounded).
- Strict identity preservation: preserves ByteTrack-assigned track IDs without arbitrary fabrication.
- Deterministic 5-state lifecycle management: NEW, TRACKING, TEMPORARILY_LOST, REACQUIRED, TERMINATED.
- Versatile sequential input ingestion: supports image directories, video files (.mp4), and precomputed detection JSON.
- Comprehensive per-track historical trajectories: centroid, dimensions, Euclidean displacement, confidence profiles.
- Versioned Schema v1.0 Multi-File Persistence:
  - multi_object_tracking.json
  - track_histories.json
  - tracking_summary.json
  - metadata.json
  - errors.json
- Full session isolation: complete reset() capability for multi-sequence pipelines.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np

from src.detection.single_image_detector import SingleImageDetector
from src.detection.yolo_config import YOLOConfig
from src.tracking.bytetrack_engine import ByteTrackEngine, TrackRecord
from src.tracking.tracker_config import TrackerConfig, TrackerConfigError
from src.tracking.tracking_pipeline import _atomic_write_json, _natural_sort_key

logger = logging.getLogger("PerceptGuard.MultiObjectTracker")


# ==============================================================================
# Domain Enumerations & Lifecycle States
# ==============================================================================


class TrackLifecycleState(str, Enum):
    """Formal lifecycle states for multi-object tracking."""

    NEW = "NEW"
    TRACKING = "TRACKING"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    REACQUIRED = "REACQUIRED"
    TERMINATED = "TERMINATED"
    UNKNOWN = "UNKNOWN"


# ==============================================================================
# Domain Exceptions
# ==============================================================================


class MultiObjectTrackingError(Exception):
    """Base exception for all multi-object tracking operations."""

    pass


class MOTInputNotFoundError(MultiObjectTrackingError, FileNotFoundError):
    """Raised when the specified input sequence, directory, or video cannot be found."""

    pass


class MOTConfigurationError(MultiObjectTrackingError):
    """Raised when tracker or detector configuration is invalid or incompatible."""

    pass


class MOTInferenceError(MultiObjectTrackingError):
    """Raised when detection or tracking inference encounters an unrecoverable error."""

    pass


# ==============================================================================
# Data Contracts: TrackedObject, TrackHistory, FrameMOTRecord
# ==============================================================================


@dataclass
class TrackedObject:
    """Detailed observation record for a single tracked object within a specific video frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    center_x: float
    center_y: float
    width: float
    height: float
    frame_id: str
    source_frame_index: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    displacement_pixels: Optional[float] = None
    velocity_pixels_per_second: Optional[float] = None
    lifecycle_state: str = "TRACKING"

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracked object observation to a JSON-serializable dictionary."""
        return {
            "track_id": int(self.track_id),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [round(float(c), 2) for c in self.bbox_xyxy],
            "center_x": round(float(self.center_x), 2),
            "center_y": round(float(self.center_y), 2),
            "width": round(float(self.width), 2),
            "height": round(float(self.height), 2),
            "frame_id": str(self.frame_id),
            "source_frame_index": int(self.source_frame_index) if self.source_frame_index is not None else None,
            "timestamp_seconds": round(float(self.timestamp_seconds), 4) if self.timestamp_seconds is not None else None,
            "displacement_pixels": round(float(self.displacement_pixels), 2) if self.displacement_pixels is not None else None,
            "velocity_pixels_per_second": round(float(self.velocity_pixels_per_second), 2) if self.velocity_pixels_per_second is not None else None,
            "lifecycle_state": str(self.lifecycle_state),
        }


@dataclass
class TrackHistory:
    """Historical trajectory and kinematic lifecycle record for a persistent track identity."""

    track_id: int
    class_id: int
    class_name: str
    first_observed_frame: str = ""
    last_observed_frame: str = ""
    first_timestamp_seconds: Optional[float] = None
    last_timestamp_seconds: Optional[float] = None
    first_frame_sequence_index: int = 0
    last_observed_frame_sequence_index: int = 0
    total_observed_frames: int = 0
    observed_duration_seconds: Optional[float] = None
    last_known_bbox_xyxy: List[float] = field(default_factory=list)
    last_center_x: Optional[float] = None
    last_center_y: Optional[float] = None
    last_timestamp: Optional[float] = None
    cumulative_displacement_pixels: float = 0.0
    mean_displacement_pixels: Optional[float] = None
    mean_confidence: float = 0.0
    confidence_std_dev: float = 0.0
    reacquisition_count: int = 0
    longest_lost_interval_frames: int = 0
    consecutive_lost_frames: int = 0
    lifecycle_state: str = "NEW"
    observations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert track history to dictionary."""
        return {
            "track_id": int(self.track_id),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "first_observed_frame": self.first_observed_frame,
            "last_observed_frame": self.last_observed_frame,
            "first_timestamp_seconds": round(float(self.first_timestamp_seconds), 4) if self.first_timestamp_seconds is not None else None,
            "last_timestamp_seconds": round(float(self.last_timestamp_seconds), 4) if self.last_timestamp_seconds is not None else None,
            "total_observed_frames": int(self.total_observed_frames),
            "observed_duration_seconds": round(float(self.observed_duration_seconds), 2) if self.observed_duration_seconds is not None else None,
            "last_known_bbox_xyxy": [round(float(c), 2) for c in self.last_known_bbox_xyxy] if self.last_known_bbox_xyxy else [],
            "cumulative_displacement_pixels": round(float(self.cumulative_displacement_pixels), 2),
            "mean_displacement_pixels": round(float(self.mean_displacement_pixels), 2) if self.mean_displacement_pixels is not None else None,
            "mean_confidence": round(float(self.mean_confidence), 4),
            "confidence_std_dev": round(float(self.confidence_std_dev), 4),
            "reacquisition_count": int(self.reacquisition_count),
            "longest_lost_interval_frames": int(self.longest_lost_interval_frames),
            "lifecycle_state": str(self.lifecycle_state),
            "observations": self.observations,
        }


@dataclass
class FrameMOTRecord:
    """Frame-level multi-object tracking record capturing active tracks, counts, and latency."""

    frame_id: str
    source_frame_index: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    detection_count: int = 0
    active_track_count: int = 0
    active_tracks: List[TrackedObject] = field(default_factory=list)
    inference_duration_ms: Optional[float] = None
    tracking_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    processing_status: str = "success"

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame MOT record to dictionary."""
        return {
            "frame_id": self.frame_id,
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": round(float(self.timestamp_seconds), 4) if self.timestamp_seconds is not None else None,
            "detection_count": int(self.detection_count),
            "active_track_count": int(self.active_track_count),
            "inference_duration_ms": round(float(self.inference_duration_ms), 2) if self.inference_duration_ms is not None else None,
            "tracking_duration_ms": round(float(self.tracking_duration_ms), 3),
            "total_duration_ms": round(float(self.total_duration_ms), 2),
            "processing_status": self.processing_status,
            "tracks": [t.to_dict() for t in self.active_tracks],
        }


@dataclass
class MultiObjectTrackingSummary:
    """Global multi-object tracking operational summary and aggregate statistics."""

    schema_version: str = "1.0"
    dataset_id: str = "driving_sample"
    run_id: str = ""
    input_source: str = ""
    model_name: str = "yolo11n"
    tracker_type: str = "bytetrack"
    total_frames_processed: int = 0
    successful_frames: int = 0
    failed_frames: int = 0
    empty_frames: int = 0
    total_detections: int = 0
    total_track_observations: int = 0
    total_unique_tracks: int = 0
    max_simultaneous_active_tracks: int = 0
    avg_objects_per_frame: float = 0.0
    avg_active_tracks_per_frame: float = 0.0
    per_class_track_counts: Dict[str, int] = field(default_factory=dict)
    track_duration_distribution: Dict[str, Any] = field(default_factory=dict)
    mean_confidence_overall: float = 0.0
    mean_processing_latency_ms: float = 0.0
    total_pipeline_duration_seconds: float = 0.0
    status: str = "PASS"  # PASS, WARNING, FAIL

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)


# ==============================================================================
# Multi-Object Tracker Architecture
# ==============================================================================


class MultiObjectTracker:
    """Production Multi-Object Tracking Engine for PerceptGuard AI.

    Associates YOLO detections across sequential frames using ByteTrack, manages per-object
    lifecycle states, maintains continuous trajectories, and generates Schema v1.0 outputs.
    """

    def __init__(
        self,
        config: Optional[Union[TrackerConfig, Dict[str, Any], Path, str]] = None,
        detector: Optional[SingleImageDetector] = None,
        max_lost_frames: int = 30,
        frame_rate: int = 30,
        dataset_id: str = "driving_sample",
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize MultiObjectTracker.

        Args:
            config: Tracker configuration instance, dict, or path.
            detector: Optional pre-initialized SingleImageDetector.
            max_lost_frames: Consecutive missing frames before track TERMINATION.
            frame_rate: Frame rate for motion prediction (default: 30 fps).
            dataset_id: Target dataset identifier.
            project_root: Repository root path.
        """
        self.project_root = Path(project_root).resolve() if project_root else _project_root
        self.dataset_id = dataset_id
        self.max_lost_frames = max(1, int(max_lost_frames))
        self.frame_rate = max(1, int(frame_rate))

        # Tracker configuration
        self.tracker_config = (
            TrackerConfig.from_yaml() if config is None else ByteTrackEngine._resolve_config(config)
        )
        self.bytetrack_engine = ByteTrackEngine(config=self.tracker_config, frame_rate=self.frame_rate)

        # Object detector (lazy or injected)
        self._detector: Optional[SingleImageDetector] = detector

        # Session state
        self.histories: Dict[int, TrackHistory] = {}
        self.frame_records: List[FrameMOTRecord] = []
        self.error_records: List[Dict[str, Any]] = []
        self._current_sequence_index = 0

    @property
    def detector(self) -> SingleImageDetector:
        """Lazily initialize SingleImageDetector if not provided upfront."""
        if self._detector is None:
            self._detector = SingleImageDetector(project_root=self.project_root)
        return self._detector

    @property
    def total_unique_tracks(self) -> int:
        """Number of unique persistent track IDs discovered in current session."""
        return len(self.histories)

    # --------------------------------------------------------------------------
    # Frame-Level Tracking
    # --------------------------------------------------------------------------

    def track_detections(
        self,
        detections: Sequence[Any],
        frame_id: str,
        source_frame_index: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
    ) -> FrameMOTRecord:
        """Update tracker with pre-computed detection bounding boxes for one chronological frame.

        Args:
            detections: List of detections (dicts with bbox_xyxy, confidence, class_id/class_name).
            frame_id: Frame identifier (e.g. 'frame_000001.jpg').
            source_frame_index: Optional source video frame index.
            timestamp_seconds: Optional physical timestamp in seconds.
            inference_time_ms: Optional inference latency of the detection pass.

        Returns:
            FrameMOTRecord containing active tracks and timing information.
        """
        start_tracking_t = time.perf_counter()
        self._current_sequence_index += 1
        seq_idx = self._current_sequence_index

        # Update ByteTrack engine
        active_track_records: List[TrackRecord] = self.bytetrack_engine.update(
            detections=detections,
            frame_id=frame_id,
        )
        tracking_duration_ms = (time.perf_counter() - start_tracking_t) * 1000.0

        active_track_ids_this_frame: Set[int] = set()
        active_tracked_objects: List[TrackedObject] = []

        # 1. Process observed tracks
        for rec in active_track_records:
            t_id = rec.track_id
            active_track_ids_this_frame.add(t_id)

            bbox = [float(c) for c in rec.bbox_xyxy]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            # Determine lifecycle state and kinematic displacement
            displacement: Optional[float] = None
            velocity_px_s: Optional[float] = None
            lifecycle_state = TrackLifecycleState.NEW

            if t_id not in self.histories:
                # NEW track
                lifecycle_state = TrackLifecycleState.NEW
                hist = TrackHistory(
                    track_id=t_id,
                    class_id=rec.class_id,
                    class_name=rec.class_name,
                    first_observed_frame=frame_id,
                    last_observed_frame=frame_id,
                    first_timestamp_seconds=timestamp_seconds,
                    last_timestamp_seconds=timestamp_seconds,
                    first_frame_sequence_index=seq_idx,
                    last_observed_frame_sequence_index=seq_idx,
                    total_observed_frames=1,
                    last_known_bbox_xyxy=bbox,
                    last_center_x=cx,
                    last_center_y=cy,
                    last_timestamp=timestamp_seconds,
                    lifecycle_state=lifecycle_state.value,
                )
                self.histories[t_id] = hist
            else:
                hist = self.histories[t_id]
                prev_cx = hist.last_center_x
                prev_cy = hist.last_center_y
                prev_ts = hist.last_timestamp

                # Check if observed consecutively or reacquired
                if seq_idx == hist.last_observed_frame_sequence_index + 1:
                    lifecycle_state = TrackLifecycleState.TRACKING
                else:
                    lifecycle_state = TrackLifecycleState.REACQUIRED
                    hist.reacquisition_count += 1

                # Kinematic displacement
                if prev_cx is not None and prev_cy is not None:
                    displacement = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                    hist.cumulative_displacement_pixels += displacement

                    if timestamp_seconds is not None and prev_ts is not None:
                        dt = timestamp_seconds - prev_ts
                        if dt > 0.0:
                            velocity_px_s = displacement / dt

                # Update history fields
                hist.total_observed_frames += 1
                hist.last_observed_frame = frame_id
                hist.last_timestamp_seconds = timestamp_seconds
                hist.last_observed_frame_sequence_index = seq_idx
                hist.last_known_bbox_xyxy = bbox
                hist.last_center_x = cx
                hist.last_center_y = cy
                hist.last_timestamp = timestamp_seconds
                hist.consecutive_lost_frames = 0
                hist.lifecycle_state = lifecycle_state.value

            # Record observation in history
            obs_entry = {
                "frame_id": frame_id,
                "sequence_index": seq_idx,
                "source_frame_index": source_frame_index,
                "timestamp_seconds": timestamp_seconds,
                "bbox_xyxy": [round(c, 2) for c in bbox],
                "center": [round(cx, 2), round(cy, 2)],
                "width": round(w, 2),
                "height": round(h, 2),
                "confidence": round(float(rec.confidence), 4),
                "displacement_pixels": round(displacement, 2) if displacement is not None else None,
                "velocity_pixels_per_second": round(velocity_px_s, 2) if velocity_px_s is not None else None,
                "state": lifecycle_state.value,
            }
            hist.observations.append(obs_entry)

            tracked_obj = TrackedObject(
                track_id=t_id,
                class_id=rec.class_id,
                class_name=rec.class_name,
                confidence=rec.confidence,
                bbox_xyxy=bbox,
                center_x=cx,
                center_y=cy,
                width=w,
                height=h,
                frame_id=frame_id,
                source_frame_index=source_frame_index,
                timestamp_seconds=timestamp_seconds,
                displacement_pixels=displacement,
                velocity_pixels_per_second=velocity_px_s,
                lifecycle_state=lifecycle_state.value,
            )
            active_tracked_objects.append(tracked_obj)

        # 2. Update status for unobserved existing tracks
        for t_id, hist in self.histories.items():
            if t_id not in active_track_ids_this_frame:
                if hist.lifecycle_state != TrackLifecycleState.TERMINATED.value:
                    hist.consecutive_lost_frames += 1
                    if hist.consecutive_lost_frames > hist.longest_lost_interval_frames:
                        hist.longest_lost_interval_frames = hist.consecutive_lost_frames

                    if hist.consecutive_lost_frames > self.max_lost_frames:
                        hist.lifecycle_state = TrackLifecycleState.TERMINATED.value
                    else:
                        hist.lifecycle_state = TrackLifecycleState.TEMPORARILY_LOST.value

        # Build Frame Record
        total_time_ms = tracking_duration_ms + (inference_time_ms or 0.0)
        status = "empty_frame" if not active_tracked_objects and not detections else "success"

        frame_record = FrameMOTRecord(
            frame_id=frame_id,
            source_frame_index=source_frame_index,
            timestamp_seconds=timestamp_seconds,
            detection_count=len(detections),
            active_track_count=len(active_tracked_objects),
            active_tracks=active_tracked_objects,
            inference_duration_ms=inference_time_ms,
            tracking_duration_ms=tracking_duration_ms,
            total_duration_ms=total_time_ms,
            processing_status=status,
        )
        self.frame_records.append(frame_record)
        return frame_record

    def track_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        frame_id: Optional[str] = None,
        source_frame_index: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
    ) -> FrameMOTRecord:
        """Run YOLO object detection on an image input and track resulting detections.

        Args:
            image_input: Path to image file or decoded BGR numpy image array.
            frame_id: Frame identifier string.
            source_frame_index: Optional physical video frame index.
            timestamp_seconds: Optional physical timestamp.

        Returns:
            FrameMOTRecord for the processed frame.
        """
        f_name = frame_id or (Path(image_input).name if isinstance(image_input, (str, Path)) else f"frame_{self._current_sequence_index + 1:06d}.jpg")

        # 1. Detection Pass
        t_det_start = time.perf_counter()
        try:
            if isinstance(image_input, np.ndarray):
                # Direct prediction on array
                res = self.detector.loader.predict_single_frame(image_input)
                boxes = getattr(res, "boxes", None)
                names = getattr(res, "names", {})
                detections: List[Dict[str, Any]] = []
                if boxes is not None and len(boxes) > 0:
                    xyxy_arr = boxes.xyxy.cpu().numpy()
                    conf_arr = boxes.conf.cpu().numpy()
                    cls_arr = boxes.cls.cpu().numpy()
                    for idx in range(len(boxes)):
                        c_id = int(cls_arr[idx])
                        detections.append({
                            "detection_index": idx,
                            "class_id": c_id,
                            "class_name": names.get(c_id, str(c_id)),
                            "confidence": float(conf_arr[idx]),
                            "bbox_xyxy": [float(x) for x in xyxy_arr[idx]],
                        })
            else:
                # Path detection
                det_res = self.detector.detect(image_input, save_json=False)
                detections = [
                    {
                        "detection_index": d.detection_index,
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox_xyxy": d.bbox_xyxy,
                    }
                    for d in det_res.detections
                ]
            infer_ms = (time.perf_counter() - t_det_start) * 1000.0
        except Exception as e:
            logger.error("Detection error on frame %s: %s", f_name, e)
            self.error_records.append({
                "frame_id": f_name,
                "stage": "detection",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            # Return empty frame record marked with error status
            frame_rec = FrameMOTRecord(
                frame_id=f_name,
                source_frame_index=source_frame_index,
                timestamp_seconds=timestamp_seconds,
                detection_count=0,
                active_track_count=0,
                active_tracks=[],
                inference_duration_ms=0.0,
                tracking_duration_ms=0.0,
                total_duration_ms=0.0,
                processing_status="error",
            )
            self.frame_records.append(frame_rec)
            return frame_rec

        # 2. Tracking Pass
        return self.track_detections(
            detections=detections,
            frame_id=f_name,
            source_frame_index=source_frame_index,
            timestamp_seconds=timestamp_seconds,
            inference_time_ms=infer_ms,
        )

    # --------------------------------------------------------------------------
    # Sequence Ingestion & Execution Loop
    # --------------------------------------------------------------------------

    def run(
        self,
        input_source: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        overwrite: bool = False,
    ) -> Tuple[MultiObjectTrackingSummary, List[FrameMOTRecord], Dict[int, TrackHistory]]:
        """Run sequential multi-object tracking across an input sequence.

        Supported input sources:
        1. Pre-computed batch detection JSON (e.g. detection_results.json).
        2. Directory of sequential frame image files.
        3. Video file (.mp4, .avi, .mov).

        Args:
            input_source: Path to detection JSON, frame directory, or video.
            output_dir: Output destination directory for tracking artifacts.
            max_frames: Frame limit cap (flags partial run).
            start_frame: Start frame sequence index.
            end_frame: End frame sequence index.
            overwrite: Whether to overwrite existing destination.

        Returns:
            Tuple of (MultiObjectTrackingSummary, List[FrameMOTRecord], Dict[int, TrackHistory]).
        """
        start_total_t = time.perf_counter()
        p = Path(input_source)
        if not p.is_absolute():
            p = (self.project_root / p).resolve()

        if not p.exists():
            raise MOTInputNotFoundError(f"Input source does not exist: {p}")

        # Reset session
        self.reset()

        # Determine Input Mode
        if p.is_file() and p.suffix.lower() == ".json":
            summary = self._run_from_detection_json(
                json_path=p,
                max_frames=max_frames,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        elif p.is_dir():
            summary = self._run_from_frame_directory(
                frame_dir=p,
                max_frames=max_frames,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        elif p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            summary = self._run_from_video_file(
                video_path=p,
                max_frames=max_frames,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        else:
            raise MOTConfigurationError(
                f"Unsupported input source format: {p}. Expected .json detection file, image folder, or .mp4 video."
            )

        summary.total_pipeline_duration_seconds = round(time.perf_counter() - start_total_t, 2)

        # Output Persistence
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"mot_{self.dataset_id}_{timestamp_slug}_{os.urandom(3).hex()}"
        summary.run_id = run_id

        if output_dir is not None:
            resolved_out = Path(output_dir).resolve()
        else:
            resolved_out = (
                self.project_root
                / "datasets"
                / "processed"
                / self.dataset_id
                / "tracking"
                / "multi_object"
                / run_id
            ).resolve()

        if resolved_out.exists() and not overwrite:
            raise FileExistsError(
                f"Output directory exists: {resolved_out}. Pass overwrite=True to replace."
            )

        self._persist_artifacts(output_dir=resolved_out, summary=summary, input_source=str(p))
        return summary, self.frame_records, self.histories

    # --------------------------------------------------------------------------
    # Specialized Ingestion Drivers
    # --------------------------------------------------------------------------

    def _run_from_detection_json(
        self,
        json_path: Path,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> MultiObjectTrackingSummary:
        """Sequential ingestion from pre-computed detection results JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_id = data.get("dataset_id", self.dataset_id)
        model_name = data.get("model_name", "yolo11n")
        raw_frames = data.get("frame_results", [])
        if not raw_frames:
            raise MultiObjectTrackingError(f"No 'frame_results' found in detection JSON: {json_path}")

        # Natural sort & deduplication
        seen_fids = set()
        deduped = []
        for f_item in sorted(raw_frames, key=lambda x: _natural_sort_key(x.get("image_filename", x.get("frame_id", "")))):
            fid = f_item.get("image_filename", f_item.get("frame_id", ""))
            if fid not in seen_fids:
                seen_fids.add(fid)
                deduped.append(f_item)
            else:
                logger.warning("Skipping duplicate frame entry: %s", fid)

        total_discovered = len(deduped)
        s_idx = start_frame if start_frame is not None else 0
        e_idx = end_frame if end_frame is not None else total_discovered
        sliced = deduped[s_idx:e_idx]
        if max_frames is not None and max_frames > 0:
            sliced = sliced[:max_frames]

        is_smoke = len(sliced) < total_discovered

        for f_item in sliced:
            f_name = f_item.get("image_filename", f_item.get("frame_id", ""))
            ext_id = f_item.get("extracted_frame_id")
            src_idx = f_item.get("source_frame_index")
            ts = f_item.get("timestamp_seconds")
            infer_ms = f_item.get("inference_time_ms")
            dets = f_item.get("detections", [])

            self.track_detections(
                detections=dets,
                frame_id=f_name,
                source_frame_index=ext_id if ext_id is not None else src_idx,
                timestamp_seconds=ts,
                inference_time_ms=infer_ms,
            )

        return self.compute_summary(
            dataset_id=dataset_id,
            input_source=str(json_path),
            model_name=model_name,
            is_smoke=is_smoke,
        )

    def _run_from_frame_directory(
        self,
        frame_dir: Path,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> MultiObjectTrackingSummary:
        """Sequential ingestion and inference on a directory of frame images."""
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = sorted(
            [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts],
            key=lambda p: _natural_sort_key(p.name),
        )
        if not image_files:
            raise MOTInputNotFoundError(f"No valid image files found in directory: {frame_dir}")

        # Attempt to load metadata.csv from dataset root
        metadata_map = self._load_metadata_csv_if_available(frame_dir)

        total_discovered = len(image_files)
        s_idx = start_frame if start_frame is not None else 0
        e_idx = end_frame if end_frame is not None else total_discovered
        sliced = image_files[s_idx:e_idx]
        if max_frames is not None and max_frames > 0:
            sliced = sliced[:max_frames]

        is_smoke = len(sliced) < total_discovered

        for img_p in sliced:
            f_name = img_p.name
            src_idx, ts = metadata_map.get(f_name, (None, None))
            self.track_image(
                image_input=img_p,
                frame_id=f_name,
                source_frame_index=src_idx,
                timestamp_seconds=ts,
            )

        return self.compute_summary(
            dataset_id=self.dataset_id,
            input_source=str(frame_dir),
            model_name=self.detector.loader.model_info.model_name if hasattr(self.detector.loader, "model_info") else "yolo11n",
            is_smoke=is_smoke,
        )

    def _run_from_video_file(
        self,
        video_path: Path,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> MultiObjectTrackingSummary:
        """Sequential frame extraction and tracking from video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise MOTInputNotFoundError(f"OpenCV failed to open video file: {video_path}")

        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        current_idx = 0
        processed_count = 0
        is_smoke = False

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if start_frame is not None and current_idx < start_frame:
                    current_idx += 1
                    continue

                if end_frame is not None and current_idx >= end_frame:
                    is_smoke = True
                    break

                if max_frames is not None and processed_count >= max_frames:
                    is_smoke = True
                    break

                f_name = f"frame_{current_idx + 1:06d}.jpg"
                ts = current_idx / fps

                self.track_image(
                    image_input=frame_bgr,
                    frame_id=f_name,
                    source_frame_index=current_idx,
                    timestamp_seconds=ts,
                )
                current_idx += 1
                processed_count += 1
        finally:
            cap.release()

        return self.compute_summary(
            dataset_id=self.dataset_id,
            input_source=str(video_path),
            model_name="yolo11n",
            is_smoke=is_smoke,
        )

    def _load_metadata_csv_if_available(self, frame_dir: Path) -> Dict[str, Tuple[int, float]]:
        """Look for metadata.csv in adjacent directories and map image_filename -> (src_idx, ts)."""
        mapping: Dict[str, Tuple[int, float]] = {}
        candidate_paths = [
            frame_dir.parent / "metadata.csv",
            self.project_root / "datasets" / "processed" / self.dataset_id / "metadata.csv",
        ]
        csv_file = next((p for p in candidate_paths if p.exists() and p.is_file()), None)
        if csv_file is None:
            return mapping

        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("preprocessed_image_filename") or row.get("original_image_filename")
                    if fname:
                        try:
                            s_idx = int(row.get("source_frame_index") or row.get("extracted_frame_id", 0))
                            ts = float(row.get("timestamp_seconds", 0.0))
                            mapping[fname] = (s_idx, ts)
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            logger.warning("Could not parse metadata.csv at %s: %s", csv_file, e)

        return mapping

    # --------------------------------------------------------------------------
    # Analytics & Summary Formulation
    # --------------------------------------------------------------------------

    def compute_summary(
        self,
        dataset_id: str,
        input_source: str,
        model_name: str = "yolo11n",
        is_smoke: bool = False,
    ) -> MultiObjectTrackingSummary:
        """Formulate multi-object tracking operational analytics."""
        total_frames = len(self.frame_records)
        successful_frames = sum(1 for f in self.frame_records if f.processing_status == "success")
        failed_frames = sum(1 for f in self.frame_records if f.processing_status == "error")
        empty_frames = sum(1 for f in self.frame_records if f.processing_status == "empty_frame")

        total_detections = sum(f.detection_count for f in self.frame_records)
        total_track_obs = sum(f.active_track_count for f in self.frame_records)
        unique_tracks = len(self.histories)

        max_simultaneous = max((f.active_track_count for f in self.frame_records), default=0)
        avg_objects = (total_detections / total_frames) if total_frames > 0 else 0.0
        avg_active_tracks = (total_track_obs / total_frames) if total_frames > 0 else 0.0

        # Class breakdown
        per_class_counts: Dict[str, int] = {}
        for h in self.histories.values():
            per_class_counts[h.class_name] = per_class_counts.get(h.class_name, 0) + 1

        # Track durations and confidence statistics
        all_confidences: List[float] = []
        durations: List[int] = []

        for h in self.histories.values():
            durations.append(h.total_observed_frames)
            obs_confs = [o["confidence"] for o in h.observations if "confidence" in o]
            all_confidences.extend(obs_confs)

            if obs_confs:
                h.mean_confidence = round(float(np.mean(obs_confs)), 4)
                h.confidence_std_dev = round(float(np.std(obs_confs)), 4)

            # Observed duration
            if h.first_timestamp_seconds is not None and h.last_timestamp_seconds is not None:
                h.observed_duration_seconds = round(h.last_timestamp_seconds - h.first_timestamp_seconds, 2)

            # Displacements
            disps = [o["displacement_pixels"] for o in h.observations if o.get("displacement_pixels") is not None]
            if disps:
                h.mean_displacement_pixels = round(float(np.mean(disps)), 2)

        duration_dist: Dict[str, Any] = {
            "min_frames": int(min(durations)) if durations else 0,
            "max_frames": int(max(durations)) if durations else 0,
            "mean_frames": round(float(np.mean(durations)), 2) if durations else 0.0,
            "median_frames": round(float(np.median(durations)), 2) if durations else 0.0,
        }

        mean_conf_overall = round(float(np.mean(all_confidences)), 4) if all_confidences else 0.0

        latencies = [f.total_duration_ms for f in self.frame_records]
        mean_latency = round(float(np.mean(latencies)), 3) if latencies else 0.0

        status = "WARNING" if (is_smoke or failed_frames > 0) else "PASS"

        return MultiObjectTrackingSummary(
            schema_version="1.0",
            dataset_id=dataset_id,
            run_id="",
            input_source=input_source,
            model_name=model_name,
            tracker_type="bytetrack",
            total_frames_processed=total_frames,
            successful_frames=successful_frames,
            failed_frames=failed_frames,
            empty_frames=empty_frames,
            total_detections=total_detections,
            total_track_observations=total_track_obs,
            total_unique_tracks=unique_tracks,
            max_simultaneous_active_tracks=max_simultaneous,
            avg_objects_per_frame=round(avg_objects, 2),
            avg_active_tracks_per_frame=round(avg_active_tracks, 2),
            per_class_track_counts=per_class_counts,
            track_duration_distribution=duration_dist,
            mean_confidence_overall=mean_conf_overall,
            mean_processing_latency_ms=mean_latency,
            total_pipeline_duration_seconds=0.0,
            status=status,
        )

    # --------------------------------------------------------------------------
    # Output Persistence (Schema v1.0)
    # --------------------------------------------------------------------------

    def _persist_artifacts(
        self,
        output_dir: Path,
        summary: MultiObjectTrackingSummary,
        input_source: str,
    ) -> None:
        """Write all 5 required Schema v1.0 artifacts to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. multi_object_tracking.json (Full sequential frame tracks)
        mot_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "input_source": input_source,
            "model_name": summary.model_name,
            "tracker_type": summary.tracker_type,
            "total_frames": summary.total_frames_processed,
            "unique_tracks": summary.total_unique_tracks,
            "frame_results": [f.to_dict() for f in self.frame_records],
        }
        _atomic_write_json(output_dir / "multi_object_tracking.json", mot_payload)

        # 2. track_histories.json (Per-track continuous trajectories)
        hist_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "total_unique_tracks": summary.total_unique_tracks,
            "tracks": {str(t_id): hist.to_dict() for t_id, hist in self.histories.items()},
        }
        _atomic_write_json(output_dir / "track_histories.json", hist_payload)

        # 3. tracking_summary.json (Operational summary)
        _atomic_write_json(output_dir / "tracking_summary.json", summary.to_dict())

        # 4. metadata.json (Runtime configuration & telemetry)
        meta_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_source": input_source,
            "tracker_configuration": self.tracker_config.to_dict(),
            "environment": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
                "cpu_count": os.cpu_count() or 1,
            },
        }
        _atomic_write_json(output_dir / "metadata.json", meta_payload)

        # 5. errors.json (Error diagnostics catalog)
        errors_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "error_count": len(self.error_records),
            "errors": self.error_records,
        }
        _atomic_write_json(output_dir / "errors.json", errors_payload)

    # --------------------------------------------------------------------------
    # Session Management
    # --------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal ByteTrack engine, clear histories, and restart sequence state."""
        self.bytetrack_engine.reset()
        self.histories.clear()
        self.frame_records.clear()
        self.error_records.clear()
        self._current_sequence_index = 0
        logger.info("MultiObjectTracker state reset successfully.")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line interface for MultiObjectTracker."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Production Multi-Object Tracking (MOT) Pipeline"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to detection JSON, frame directory, or .mp4 video file",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="driving_sample",
        help="Dataset identifier (default: driving_sample)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to tracker_config.yaml",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination directory for tracking artifacts",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=None,
        help="Maximum frames to process (useful for smoke tests)",
    )
    parser.add_argument(
        "--start-frame",
        "-s",
        type=int,
        default=None,
        help="Start frame sequence index",
    )
    parser.add_argument(
        "--end-frame",
        "-e",
        type=int,
        default=None,
        help="End frame sequence index",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logging",
    )

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(name)s: %(message)s")

    try:
        # Default input resolution if not provided
        if args.input is None:
            # Check default processed frames or batch detection run
            default_frames = _project_root / "datasets" / "processed" / args.dataset / "preprocessed_frames"
            default_dets = _project_root / "datasets" / "processed" / args.dataset / "detections" / "batch_runs"
            if default_dets.exists():
                runs = sorted([p for p in default_dets.iterdir() if p.is_dir() and p.name.startswith("run_")])
                if runs and (runs[-1] / "detection_results.json").exists():
                    target_input = runs[-1] / "detection_results.json"
                elif default_frames.exists():
                    target_input = default_frames
                else:
                    raise MOTInputNotFoundError(f"No default detection or frame source in {args.dataset}")
            elif default_frames.exists():
                target_input = default_frames
            else:
                raise MOTInputNotFoundError(f"No input specified and default dataset paths not found for {args.dataset}")
        else:
            target_input = Path(args.input)

        tracker = MultiObjectTracker(
            config=args.config,
            dataset_id=args.dataset,
        )

        print(f"[INFO] Initializing Multi-Object Tracking on input: '{target_input}'...")
        summary, _, _ = tracker.run(
            input_source=target_input,
            output_dir=args.output,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overwrite=args.overwrite,
        )

        print("-" * 65)
        print("PERCEPTGUARD AI — MULTI-OBJECT TRACKING COMPLETE")
        print("-" * 65)
        print(f"  Run Identifier           : {summary.run_id}")
        print(f"  Dataset / Model          : {summary.dataset_id} / {summary.model_name}")
        print(f"  Processed Frames         : {summary.total_frames_processed} (Success: {summary.successful_frames})")
        print(f"  Unique Tracks Discovered : {summary.total_unique_tracks}")
        print(f"  Max Simultaneous Tracks  : {summary.max_simultaneous_active_tracks}")
        print(f"  Total Track Observations : {summary.total_track_observations}")
        print(f"  Avg Tracks per Frame     : {summary.avg_active_tracks_per_frame:.2f}")
        print(f"  Per-Class Track Counts   : {summary.per_class_track_counts}")
        print(f"  Track Duration (Frames)  : min={summary.track_duration_distribution.get('min_frames', 0)}, max={summary.track_duration_distribution.get('max_frames', 0)}, mean={summary.track_duration_distribution.get('mean_frames', 0.0):.1f}")
        print(f"  Mean Overall Confidence  : {summary.mean_confidence_overall:.4f}")
        print(f"  Mean Processing Latency  : {summary.mean_processing_latency_ms:.2f} ms / frame")
        print(f"  Total Duration           : {summary.total_pipeline_duration_seconds:.2f} s")
        print(f"  Run Status               : [{summary.status}]")
        print("-" * 65)
        return 0

    except (MOTInputNotFoundError, FileNotFoundError) as e:
        print(f"[ERROR] Input not found: {e}", file=sys.stderr)
        return 2
    except FileExistsError as e:
        print(f"[ERROR] Destination conflict: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[FATAL] Multi-object tracking failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
