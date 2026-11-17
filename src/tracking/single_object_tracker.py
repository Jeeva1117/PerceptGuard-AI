"""PerceptGuard AI: Production Single-Object Tracking Architecture.

Senior Staff-level, production-grade single-object tracking module built upon
PerceptGuard AI's ByteTrack infrastructure. Follows a selected target object's
persistent tracking identity across sequential video frames, tracking state transitions,
spatial displacement, and visibility metrics.

Key Architectural Guarantees:
- Reuses existing ByteTrack tracking outputs and engine; zero duplicate YOLO re-inference.
- Formal tracking state machine: INITIALIZED -> TRACKING <-> TEMPORARILY_LOST -> REACQUIRED / TERMINATED.
- Deterministic target selection policies: by explicit Track ID, by class with disambiguation rules.
- True kinematic pixel displacement tracking (without uncalibrated metric speed claims).
- Versioned Multi-File Persistence (Schema v1.0): single_object_tracking, target_history, summary, metadata.
- Rigorous visibility accounting clearly separated from detection confidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from src.tracking.bytetrack_engine import ByteTrackEngine, TrackRecord
from src.tracking.tracker_config import TrackerConfig, TrackerConfigError
from src.tracking.tracking_pipeline import _atomic_write_json, _natural_sort_key

logger = logging.getLogger("PerceptGuard.SingleObjectTracker")

# ==============================================================================
# Domain Enumerations & State Machine
# ==============================================================================


class SingleObjectTrackingState(str, Enum):
    """Explicit state machine states governing target object tracking."""

    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZED = "INITIALIZED"
    TRACKING = "TRACKING"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    REACQUIRED = "REACQUIRED"
    TERMINATED = "TERMINATED"


class TargetSelectionRule(str, Enum):
    """Disambiguation policy when multiple candidate objects match a target class."""

    EXPLICIT_ID = "explicit_id"
    HIGHEST_CONFIDENCE = "highest_confidence"
    LARGEST_AREA = "largest_area"
    FIRST_DETECTED = "first_detected"


# ==============================================================================
# Domain Exceptions
# ==============================================================================


class SingleObjectTrackingError(Exception):
    """Base exception for single object tracking operations."""

    pass


class TargetNotFoundError(SingleObjectTrackingError):
    """Raised when the specified target ID or target class cannot be found in candidates."""

    pass


class AmbiguousTargetSelectionError(SingleObjectTrackingError):
    """Raised when multiple candidate objects exist and no disambiguation rule is provided."""

    pass


class InvalidTargetIdError(SingleObjectTrackingError):
    """Raised when target ID is malformed, non-positive, or of incorrect type."""

    pass


# ==============================================================================
# Data Contracts: TargetObservation & SingleObjectTrackingSummary
# ==============================================================================


@dataclass
class TargetObservation:
    """Observation record of the selected target object at a specific video frame."""

    frame_id: str
    source_frame_index: Optional[int]
    timestamp_seconds: Optional[float]
    track_id: int
    class_id: int
    class_name: str
    confidence: Optional[float]
    bbox_xyxy: Optional[List[float]]
    bbox_center_x: Optional[float]
    bbox_center_y: Optional[float]
    bbox_width: Optional[float]
    bbox_height: Optional[float]
    frame_displacement_pixels: Optional[float]
    velocity_pixels_per_second: Optional[float]
    tracking_state: str
    is_visible: bool
    consecutive_lost_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to JSON-serializable dictionary."""
        return {
            "frame_id": self.frame_id,
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "track_id": int(self.track_id),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4) if self.confidence is not None else None,
            "bbox_xyxy": [round(float(c), 2) for c in self.bbox_xyxy] if self.bbox_xyxy is not None else None,
            "bbox_center_x": round(float(self.bbox_center_x), 2) if self.bbox_center_x is not None else None,
            "bbox_center_y": round(float(self.bbox_center_y), 2) if self.bbox_center_y is not None else None,
            "bbox_width": round(float(self.bbox_width), 2) if self.bbox_width is not None else None,
            "bbox_height": round(float(self.bbox_height), 2) if self.bbox_height is not None else None,
            "frame_displacement_pixels": round(float(self.frame_displacement_pixels), 2) if self.frame_displacement_pixels is not None else None,
            "velocity_pixels_per_second": round(float(self.velocity_pixels_per_second), 2) if self.velocity_pixels_per_second is not None else None,
            "tracking_state": str(self.tracking_state),
            "is_visible": bool(self.is_visible),
            "consecutive_lost_frames": int(self.consecutive_lost_frames),
        }


@dataclass
class SingleObjectTrackingSummary:
    """Aggregate summary capturing target tracking continuity and spatial kinematics."""

    schema_version: str = "1.0"
    run_id: str = ""
    dataset_id: str = "driving_sample"
    target_id: int = 0
    target_class: str = ""
    selection_method: str = "explicit_id"
    selection_rule: str = "highest_confidence"
    total_frames_processed: int = 0
    visible_frame_count: int = 0
    missing_frame_count: int = 0
    visibility_percentage: float = 0.0
    reacquisition_count: int = 0
    longest_consecutive_missing_frames: int = 0
    first_observed_frame: Optional[str] = None
    last_observed_frame: Optional[str] = None
    tracking_duration_seconds: Optional[float] = None
    mean_confidence: Optional[float] = None
    confidence_std_dev: Optional[float] = None
    mean_displacement_pixels: Optional[float] = None
    cumulative_displacement_pixels: float = 0.0
    final_state: str = "TERMINATED"
    mean_processing_latency_ms: float = 0.0
    total_pipeline_duration_seconds: float = 0.0
    status: str = "PASS"  # "PASS", "WARNING", "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to JSON dictionary."""
        return asdict(self)


# ==============================================================================
# Single Object Tracker Architecture
# ==============================================================================


class SingleObjectTracker:
    """Dedicated single-object tracking architecture for following a target across video frames.

    Consumes sequential frame tracklets from ByteTrack, isolates the designated target ID,
    maintains a state machine with occlusion/reacquisition detection, and computes
    kinematic pixel displacements.
    """

    def __init__(
        self,
        target_id: Optional[int] = None,
        target_class: Optional[str] = None,
        selection_rule: Union[str, TargetSelectionRule] = TargetSelectionRule.HIGHEST_CONFIDENCE,
        max_lost_frames: int = 30,
        config: Optional[Union[TrackerConfig, Dict[str, Any], Path, str]] = None,
        dataset_id: str = "driving_sample",
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize SingleObjectTracker.

        Args:
            target_id: Specific integer tracking ID to follow (if known upfront).
            target_class: Semantic class name to target (e.g. 'car', 'traffic light').
            selection_rule: Policy to disambiguate multiple objects of target_class.
            max_lost_frames: Maximum frames a lost target is tolerated before TERMINATED.
            config: Tracker configuration instance, dict, or path.
            dataset_id: Dataset identifier.
            project_root: Repository root path.
        """
        if target_id is not None:
            if isinstance(target_id, bool) or not isinstance(target_id, int) or target_id <= 0:
                raise InvalidTargetIdError(
                    f"target_id must be a positive integer, got {type(target_id).__name__}: {target_id}"
                )

        self.target_id: Optional[int] = int(target_id) if target_id is not None else None
        self.target_class: Optional[str] = str(target_class).strip().lower() if target_class else None
        self.selection_rule = (
            selection_rule if isinstance(selection_rule, TargetSelectionRule)
            else TargetSelectionRule(str(selection_rule).lower())
        )
        self.max_lost_frames = max(1, int(max_lost_frames))
        self.dataset_id = dataset_id
        self.project_root = Path(project_root).resolve() if project_root else _project_root
        self.config = TrackerConfig.from_yaml() if config is None else ByteTrackEngine._resolve_config(config)

        # Internal State Machine
        self.state: SingleObjectTrackingState = SingleObjectTrackingState.UNINITIALIZED
        self.target_class_id: Optional[int] = None
        self.target_class_name: Optional[str] = None

        self.history: List[TargetObservation] = []
        self._last_visible_obs: Optional[TargetObservation] = None
        self._consecutive_lost: int = 0
        self._longest_lost: int = 0
        self._reacquisitions: int = 0
        self._cumulative_displacement: float = 0.0

    # --------------------------------------------------------------------------
    # Target Selection Logic
    # --------------------------------------------------------------------------

    def select_target_from_candidates(
        self,
        candidates: Sequence[Any],
        frame_id: Optional[str] = None,
    ) -> int:
        """Disambiguate and select a single target track from available candidates.

        Args:
            candidates: Sequence of TrackRecord objects or track dictionaries.
            frame_id: Source frame identifier for diagnostic messages.

        Returns:
            The selected integer tracking ID.

        Raises:
            TargetNotFoundError: If target_id or target_class is not among candidates.
            AmbiguousTargetSelectionError: If candidates are ambiguous and no valid rule applies.
        """
        if not candidates:
            raise TargetNotFoundError(f"No candidate tracks present in frame {frame_id or 'unknown'}.")

        # 1. Explicit Target ID match
        if self.target_id is not None:
            matching = [c for c in candidates if self._get_track_id(c) == self.target_id]
            if not matching:
                raise TargetNotFoundError(
                    f"Target ID {self.target_id} not found in frame {frame_id or 'unknown'} candidates."
                )
            selected = matching[0]
            self.target_class_id = self._get_class_id(selected)
            self.target_class_name = self._get_class_name(selected)
            return self.target_id

        # 2. Class-based matching
        if self.target_class is not None:
            class_matches = [
                c for c in candidates
                if str(self._get_class_name(c)).lower() == self.target_class
                or str(self._get_class_id(c)) == self.target_class
            ]
            if not class_matches:
                raise TargetNotFoundError(
                    f"No objects of class '{self.target_class}' found in frame {frame_id or 'unknown'}."
                )
            if len(class_matches) == 1:
                selected = class_matches[0]
            else:
                selected = self._apply_disambiguation_rule(class_matches)

            self.target_id = self._get_track_id(selected)
            self.target_class_id = self._get_class_id(selected)
            self.target_class_name = self._get_class_name(selected)
            return self.target_id

        # 3. Neither specified: select from all candidates by rule
        selected = self._apply_disambiguation_rule(list(candidates))
        self.target_id = self._get_track_id(selected)
        self.target_class_id = self._get_class_id(selected)
        self.target_class_name = self._get_class_name(selected)
        return self.target_id

    def _apply_disambiguation_rule(self, candidates: List[Any]) -> Any:
        """Apply explicit selection rule to pick one candidate from multiple choices."""
        if self.selection_rule == TargetSelectionRule.HIGHEST_CONFIDENCE:
            return max(candidates, key=lambda c: self._get_confidence(c))
        elif self.selection_rule == TargetSelectionRule.LARGEST_AREA:
            return max(candidates, key=lambda c: self._get_area(c))
        elif self.selection_rule == TargetSelectionRule.FIRST_DETECTED:
            return candidates[0]
        else:
            raise AmbiguousTargetSelectionError(
                f"Unknown or unsupported selection rule: '{self.selection_rule}'"
            )

    # --------------------------------------------------------------------------
    # Field Extraction Helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _get_track_id(record: Any) -> int:
        if isinstance(record, dict):
            val = record.get("track_id", 0)
        else:
            val = getattr(record, "track_id", 0)
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _get_class_id(record: Any) -> int:
        if isinstance(record, dict):
            val = record.get("class_id", 0)
        else:
            val = getattr(record, "class_id", 0)
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _get_class_name(record: Any) -> str:
        if isinstance(record, dict):
            return str(record.get("class_name", str(record.get("class_id", "unknown"))))
        return str(getattr(record, "class_name", str(getattr(record, "class_id", "unknown"))))

    @staticmethod
    def _get_confidence(record: Any) -> float:
        if isinstance(record, dict):
            val = record.get("confidence", 0.0)
        else:
            val = getattr(record, "confidence", 0.0)
        try:
            return float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _get_bbox(record: Any) -> List[float]:
        if isinstance(record, dict):
            raw = record.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        else:
            raw = getattr(record, "bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        if not raw or len(raw) < 4:
            return [0.0, 0.0, 0.0, 0.0]
        try:
            return [float(x) for x in raw[:4]]
        except (ValueError, TypeError):
            return [0.0, 0.0, 0.0, 0.0]

    @classmethod
    def _get_area(cls, record: Any) -> float:
        bbox = cls._get_bbox(record)
        w = max(0.0, bbox[2] - bbox[0])
        h = max(0.0, bbox[3] - bbox[1])
        return w * h

    # --------------------------------------------------------------------------
    # Sequential Frame Processing & State Machine
    # --------------------------------------------------------------------------

    def step(
        self,
        frame_tracks: Sequence[Any],
        frame_id: str,
        source_frame_index: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
    ) -> TargetObservation:
        """Process tracking records for one chronological frame and update target history.

        Args:
            frame_tracks: List of active TrackRecord or track dicts for this frame.
            frame_id: Frame identifier (e.g. 'frame_000001.jpg').
            source_frame_index: Optional physical video frame index.
            timestamp_seconds: Optional physical video elapsed timestamp.

        Returns:
            TargetObservation capturing current target status and position.
        """
        # Auto-select target if not yet bound
        if self.target_id is None:
            if frame_tracks:
                try:
                    self.select_target_from_candidates(frame_tracks, frame_id=frame_id)
                except TargetNotFoundError:
                    pass

        # Look for target in current frame's tracks
        matched_record: Optional[Any] = None
        if self.target_id is not None and frame_tracks:
            for t in frame_tracks:
                if self._get_track_id(t) == self.target_id:
                    matched_record = t
                    break

        # Case A: Target is VISIBLE in current frame
        if matched_record is not None:
            bbox = self._get_bbox(matched_record)
            conf = self._get_confidence(matched_record)
            cls_id = self._get_class_id(matched_record)
            cls_name = self._get_class_name(matched_record)

            self.target_class_id = cls_id
            self.target_class_name = cls_name

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            # Compute displacement from last visible observation
            displacement: Optional[float] = None
            velocity_px_s: Optional[float] = None

            if self._last_visible_obs is not None:
                prev_cx = self._last_visible_obs.bbox_center_x
                prev_cy = self._last_visible_obs.bbox_center_y
                if prev_cx is not None and prev_cy is not None:
                    displacement = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                    self._cumulative_displacement += displacement

                    # Velocity if timestamps are available and advancing
                    if (
                        timestamp_seconds is not None
                        and self._last_visible_obs.timestamp_seconds is not None
                    ):
                        dt = timestamp_seconds - self._last_visible_obs.timestamp_seconds
                        if dt > 0.0:
                            velocity_px_s = displacement / dt

            # Update State Machine
            if self.state in (SingleObjectTrackingState.UNINITIALIZED, SingleObjectTrackingState.TERMINATED):
                new_state = SingleObjectTrackingState.INITIALIZED
            elif self.state == SingleObjectTrackingState.TEMPORARILY_LOST:
                new_state = SingleObjectTrackingState.REACQUIRED
                self._reacquisitions += 1
            else:
                new_state = SingleObjectTrackingState.TRACKING

            self.state = new_state
            self._consecutive_lost = 0

            obs = TargetObservation(
                frame_id=frame_id,
                source_frame_index=source_frame_index,
                timestamp_seconds=timestamp_seconds,
                track_id=self.target_id,
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox_xyxy=[x1, y1, x2, y2],
                bbox_center_x=cx,
                bbox_center_y=cy,
                bbox_width=w,
                bbox_height=h,
                frame_displacement_pixels=displacement,
                velocity_pixels_per_second=velocity_px_s,
                tracking_state=self.state.value,
                is_visible=True,
                consecutive_lost_frames=0,
            )
            self._last_visible_obs = obs
            self.history.append(obs)
            return obs

        # Case B: Target is MISSING in current frame
        else:
            # If target has never been initialized
            if self.state == SingleObjectTrackingState.UNINITIALIZED:
                obs = TargetObservation(
                    frame_id=frame_id,
                    source_frame_index=source_frame_index,
                    timestamp_seconds=timestamp_seconds,
                    track_id=self.target_id or 0,
                    class_id=self.target_class_id or 0,
                    class_name=self.target_class_name or (self.target_class or "unknown"),
                    confidence=None,
                    bbox_xyxy=None,
                    bbox_center_x=None,
                    bbox_center_y=None,
                    bbox_width=None,
                    bbox_height=None,
                    frame_displacement_pixels=None,
                    velocity_pixels_per_second=None,
                    tracking_state=SingleObjectTrackingState.UNINITIALIZED.value,
                    is_visible=False,
                    consecutive_lost_frames=0,
                )
                self.history.append(obs)
                return obs

            # Target was previously seen: handle loss/termination
            self._consecutive_lost += 1
            if self._consecutive_lost > self._longest_lost:
                self._longest_lost = self._consecutive_lost

            if self._consecutive_lost > self.max_lost_frames:
                self.state = SingleObjectTrackingState.TERMINATED
            else:
                self.state = SingleObjectTrackingState.TEMPORARILY_LOST

            obs = TargetObservation(
                frame_id=frame_id,
                source_frame_index=source_frame_index,
                timestamp_seconds=timestamp_seconds,
                track_id=self.target_id or 0,
                class_id=self.target_class_id or 0,
                class_name=self.target_class_name or (self.target_class or "unknown"),
                confidence=None,
                bbox_xyxy=None,
                bbox_center_x=None,
                bbox_center_y=None,
                bbox_width=None,
                bbox_height=None,
                frame_displacement_pixels=None,
                velocity_pixels_per_second=None,
                tracking_state=self.state.value,
                is_visible=False,
                consecutive_lost_frames=self._consecutive_lost,
            )
            self.history.append(obs)
            return obs

    # --------------------------------------------------------------------------
    # Multi-Frame Sequence Ingestion
    # --------------------------------------------------------------------------

    def run_from_tracking_results(
        self,
        tracking_results_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        overwrite: bool = False,
    ) -> Tuple[SingleObjectTrackingSummary, List[TargetObservation]]:
        """Run single-object tracking by consuming an existing `tracking_results.json` run.

        Args:
            tracking_results_file: Path to `tracking_results.json` from a previous ByteTrack run.
            output_dir: Destination folder for single object tracking artifacts.
            max_frames: Optional frame limit.
            start_frame: Start index.
            end_frame: End index.
            overwrite: If True, allows replacing existing output directory.

        Returns:
            Tuple of (SingleObjectTrackingSummary, List[TargetObservation]).
        """
        start_t = time.perf_counter()
        p = Path(tracking_results_file)
        if not p.is_absolute():
            p = (self.project_root / p).resolve()

        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Tracking results file not found at: {p}")

        with open(p, "r", encoding="utf-8") as f:
            batch_data = json.load(f)

        dataset_id = batch_data.get("dataset_id", self.dataset_id)
        raw_frames = batch_data.get("frame_results", [])
        if not raw_frames:
            raise SingleObjectTrackingError(f"No 'frame_results' found in: {p}")

        # Natural sort & deduplicate duplicate frame identifiers
        seen_fids = set()
        deduped_frames = []
        for f_item in sorted(raw_frames, key=lambda x: _natural_sort_key(x.get("image_filename", x.get("frame_id", "")))):
            fid = f_item.get("image_filename", f_item.get("frame_id", ""))
            if fid not in seen_fids:
                seen_fids.add(fid)
                deduped_frames.append(f_item)
            else:
                logger.warning("Ignoring duplicate frame identifier: %s", fid)
        sorted_frames = deduped_frames
        total_discovered = len(sorted_frames)

        s_idx = start_frame if start_frame is not None else 0
        e_idx = end_frame if end_frame is not None else total_discovered
        sliced_frames = sorted_frames[s_idx:e_idx]

        if max_frames is not None and max_frames > 0:
            sliced_frames = sliced_frames[:max_frames]

        is_smoke = len(sliced_frames) < total_discovered

        # Reset tracker state
        self.reset()

        # Target auto-selection on first frame containing candidates
        if self.target_id is None:
            for f_item in sliced_frames:
                tracks = f_item.get("tracks", [])
                if tracks:
                    try:
                        self.select_target_from_candidates(tracks, frame_id=f_item.get("image_filename", ""))
                        break
                    except TargetNotFoundError:
                        continue
            if self.target_id is None:
                raise TargetNotFoundError(
                    f"Unable to auto-select target from sequence: target_class='{self.target_class}' was never detected."
                )

        step_latencies: List[float] = []

        for f_item in sliced_frames:
            s_time = time.perf_counter()
            f_name = f_item.get("image_filename", f_item.get("frame_id", ""))
            ext_id = f_item.get("extracted_frame_id")
            src_idx = f_item.get("source_frame_index")
            ts = f_item.get("timestamp_seconds")
            tracks = f_item.get("tracks", [])

            self.step(
                frame_tracks=tracks,
                frame_id=f_name,
                source_frame_index=ext_id if ext_id is not None else src_idx,
                timestamp_seconds=ts,
            )
            step_latencies.append((time.perf_counter() - s_time) * 1000.0)

        total_proc_sec = time.perf_counter() - start_t
        summary = self.compute_summary(
            dataset_id=dataset_id,
            is_smoke=is_smoke,
            mean_step_latency_ms=float(np.mean(step_latencies)) if step_latencies else 0.0,
            total_duration_sec=total_proc_sec,
        )

        # Output Persistence
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"single_target_{self.target_id}_{timestamp_slug}_{os.urandom(3).hex()}"
        summary.run_id = run_id

        if output_dir is not None:
            resolved_out = Path(output_dir).resolve()
        else:
            resolved_out = (
                self.project_root
                / "datasets"
                / "processed"
                / dataset_id
                / "tracking"
                / "single_object"
                / run_id
            ).resolve()

        if resolved_out.exists() and not overwrite:
            raise FileExistsError(
                f"Output directory exists: {resolved_out}. Pass overwrite=True to replace."
            )

        self._persist_artifacts(
            output_dir=resolved_out,
            summary=summary,
            source_file=str(p),
        )

        return summary, self.history

    # --------------------------------------------------------------------------
    # Analytics & Summary Formulation
    # --------------------------------------------------------------------------

    def compute_summary(
        self,
        dataset_id: str,
        is_smoke: bool = False,
        mean_step_latency_ms: float = 0.0,
        total_duration_sec: float = 0.0,
    ) -> SingleObjectTrackingSummary:
        """Calculate statistical quality indicators and tracking continuity metrics."""
        total_frames = len(self.history)
        visible_obs = [o for o in self.history if o.is_visible]
        visible_count = len(visible_obs)
        missing_count = total_frames - visible_count

        vis_pct = (visible_count / total_frames * 100.0) if total_frames > 0 else 0.0

        first_frame = visible_obs[0].frame_id if visible_obs else None
        last_frame = visible_obs[-1].frame_id if visible_obs else None

        # Tracking duration
        duration_s: Optional[float] = None
        if visible_obs and visible_obs[0].timestamp_seconds is not None and visible_obs[-1].timestamp_seconds is not None:
            duration_s = round(visible_obs[-1].timestamp_seconds - visible_obs[0].timestamp_seconds, 2)

        # Confidence statistics
        confidences = [o.confidence for o in visible_obs if o.confidence is not None]
        mean_conf = round(float(np.mean(confidences)), 4) if confidences else None
        std_conf = round(float(np.std(confidences)), 4) if confidences else None

        # Displacement statistics
        displacements = [o.frame_displacement_pixels for o in visible_obs if o.frame_displacement_pixels is not None]
        mean_disp = round(float(np.mean(displacements)), 2) if displacements else None

        if self.target_id is not None and self.target_class is None:
            selection_method = "explicit_id"
        elif self.target_class is not None:
            selection_method = f"class_{self.target_class}"
        else:
            selection_method = "auto_selected"

        status = "WARNING" if is_smoke else "PASS"

        return SingleObjectTrackingSummary(
            schema_version="1.0",
            run_id="",
            dataset_id=dataset_id,
            target_id=self.target_id or 0,
            target_class=self.target_class_name or (self.target_class or "unknown"),
            selection_method=selection_method,
            selection_rule=self.selection_rule.value,
            total_frames_processed=total_frames,
            visible_frame_count=visible_count,
            missing_frame_count=missing_count,
            visibility_percentage=round(vis_pct, 2),
            reacquisition_count=self._reacquisitions,
            longest_consecutive_missing_frames=self._longest_lost,
            first_observed_frame=first_frame,
            last_observed_frame=last_frame,
            tracking_duration_seconds=duration_s,
            mean_confidence=mean_conf,
            confidence_std_dev=std_conf,
            mean_displacement_pixels=mean_disp,
            cumulative_displacement_pixels=round(self._cumulative_displacement, 2),
            final_state=self.state.value,
            mean_processing_latency_ms=round(mean_step_latency_ms, 3),
            total_pipeline_duration_seconds=round(total_duration_sec, 2),
            status=status,
        )

    # --------------------------------------------------------------------------
    # Output Persistence (Schema v1.0)
    # --------------------------------------------------------------------------

    def _persist_artifacts(
        self,
        output_dir: Path,
        summary: SingleObjectTrackingSummary,
        source_file: str,
    ) -> None:
        """Write single_object_tracking, target_history, summary, and metadata JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. single_object_tracking.json (Complete sequence observations)
        tracking_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "target_id": summary.target_id,
            "target_class": summary.target_class,
            "total_frames": summary.total_frames_processed,
            "observations": [o.to_dict() for o in self.history],
        }
        _atomic_write_json(output_dir / "single_object_tracking.json", tracking_payload)

        # 2. target_history.json (Visible observations only with positions)
        visible_observations = [o.to_dict() for o in self.history if o.is_visible]
        history_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "target_id": summary.target_id,
            "target_class": summary.target_class,
            "visible_count": len(visible_observations),
            "trajectory": visible_observations,
        }
        _atomic_write_json(output_dir / "target_history.json", history_payload)

        # 3. tracking_summary.json
        _atomic_write_json(output_dir / "tracking_summary.json", summary.to_dict())

        # 4. tracking_metadata.json
        metadata_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_selection": {
                "target_id": summary.target_id,
                "target_class": summary.target_class,
                "selection_rule": summary.selection_rule,
            },
            "tracker_configuration": self.config.to_dict(),
            "source_tracking_file": source_file,
            "environment": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
                "cpu_count": os.cpu_count() or 1,
            },
        }
        _atomic_write_json(output_dir / "tracking_metadata.json", metadata_payload)

    def reset(self) -> None:
        """Reset internal tracking history and state machine."""
        self.state = SingleObjectTrackingState.UNINITIALIZED
        self.history.clear()
        self._last_visible_obs = None
        self._consecutive_lost = 0
        self._longest_lost = 0
        self._reacquisitions = 0
        self._cumulative_displacement = 0.0


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line interface for SingleObjectTracker."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Single Object Tracking Pipeline"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to tracking_results.json from previous ByteTrack run",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="driving_sample",
        help="Dataset identifier (default: driving_sample)",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        default=None,
        help="Target tracking ID to follow (e.g. 1, 2, 3)",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default=None,
        help="Target class name (e.g. 'car', 'traffic light', 'truck')",
    )
    parser.add_argument(
        "--selection-rule",
        type=str,
        default="highest_confidence",
        choices=["highest_confidence", "largest_area", "first_detected"],
        help="Rule to disambiguate multiple candidate objects of target-class",
    )
    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=30,
        help="Maximum frames a lost target is retained before TERMINATED (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Destination directory for single object tracking artifacts",
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
        help="Start frame index",
    )
    parser.add_argument(
        "--end-frame",
        "-e",
        type=int,
        default=None,
        help="End frame index",
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

    # Mutual exclusivity & validation
    if args.target_id is not None and args.target_class is not None:
        print("[ERROR] Arguments --target-id and --target-class are mutually exclusive. Specify only one.", file=sys.stderr)
        return 1

    if args.target_id is not None and args.target_id <= 0:
        print(f"[ERROR] target-id must be a positive integer, got {args.target_id}.", file=sys.stderr)
        return 1

    try:
        # Resolve input file
        if args.input is None:
            default_dir = (
                _project_root
                / "datasets"
                / "processed"
                / args.dataset
                / "tracking"
            )
            if default_dir.exists():
                runs = sorted([p for p in default_dir.iterdir() if p.is_dir() and p.name.startswith("track_")])
                if runs:
                    target_file = runs[-1] / "tracking_results.json"
                else:
                    raise FileNotFoundError(f"No tracking runs found in {default_dir}")
            else:
                raise FileNotFoundError(f"Tracking directory not found: {default_dir}")
        else:
            target_file = Path(args.input)

        tracker = SingleObjectTracker(
            target_id=args.target_id,
            target_class=args.target_class,
            selection_rule=args.selection_rule,
            max_lost_frames=args.max_lost_frames,
            dataset_id=args.dataset,
        )

        print(f"[INFO] Ingesting tracking results from: '{target_file}'...")
        summary, _ = tracker.run_from_tracking_results(
            tracking_results_file=target_file,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overwrite=args.overwrite,
        )

        print("-" * 65)
        print("PERCEPTGUARD AI — SINGLE OBJECT TRACKING COMPLETE")
        print("-" * 65)
        print(f"  Run Identifier           : {summary.run_id}")
        print(f"  Target Track ID          : {summary.target_id} ({summary.target_class})")
        print(f"  Selection Method         : {summary.selection_method} (Rule: {summary.selection_rule})")
        print(f"  Processed Frames         : {summary.total_frames_processed}")
        print(f"  Target Visible Frames    : {summary.visible_frame_count} ({summary.visibility_percentage:.1f}%)")
        print(f"  Target Missing Frames    : {summary.missing_frame_count}")
        print(f"  Reacquisitions Observed  : {summary.reacquisition_count}")
        print(f"  Max Consecutive Missing  : {summary.longest_consecutive_missing_frames} frames")
        print(f"  First / Last Frame       : {summary.first_observed_frame} -> {summary.last_observed_frame}")
        if summary.tracking_duration_seconds is not None:
            print(f"  Tracking Duration        : {summary.tracking_duration_seconds:.2f} s")
        if summary.mean_confidence is not None:
            print(f"  Mean Confidence          : {summary.mean_confidence:.4f} (std: {summary.confidence_std_dev or 0.0:.4f})")
        print(f"  Cumulative Displacement  : {summary.cumulative_displacement_pixels:.2f} px")
        print(f"  Final State              : [{summary.final_state}]")
        print(f"  Run Status               : [{summary.status}]")
        print("-" * 65)
        return 0

    except (FileNotFoundError, TargetNotFoundError) as e:
        print(f"[ERROR] Target or input not found: {e}", file=sys.stderr)
        return 2
    except FileExistsError as e:
        print(f"[ERROR] Destination conflict: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[FATAL] Single object tracking failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
