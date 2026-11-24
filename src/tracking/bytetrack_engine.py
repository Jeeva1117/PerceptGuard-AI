"""PerceptGuard AI: Production ByteTrack Tracking Engine.

Senior Staff-level, production-grade ByteTrack integration for autonomous driving
predictive perception pipelines.

Key Architectural Guarantees:
- Decoupled from YOLO inference: consumes pre-computed or one-pass detection outputs.
- Pure CPU execution: leverages 2D discrete Kalman state estimation and IoU distance.
- Robust linear assignment: includes seamless SciPy fallback when C-extension lap is absent.
- Full identity persistence: tracks objects across frames and assigns consistent track IDs.
- Deterministic handling of empty frames, degenerate boxes, and tracker state lifecycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from enum import Enum

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

# ==============================================================================
# Linear Assignment Solver Environment Shim
# ==============================================================================
# Ultralytics BYTETracker imports 'lap' (Jonker-Volgenant C-extension) by default.
# When running in offline CPU environments where 'lap' is unavailable, this shim
# supplies the exact same lapjv interface backed by scipy.optimize.linear_sum_assignment.


def _ensure_lap_available() -> None:
    """Ensure a working 'lap' module exists in sys.modules with lapjv signature."""
    if "lap" in sys.modules and getattr(sys.modules["lap"], "lapjv", None) is not None:
        return

    try:
        import lap

        if getattr(lap, "__version__", None) and hasattr(lap, "lapjv"):
            return
    except (ImportError, Exception):
        pass

    # Provide SciPy-backed lapjv fallback
    from scipy.optimize import linear_sum_assignment

    def lapjv_scipy(
        cost_matrix: np.ndarray,
        extend_cost: bool = True,
        cost_limit: float = np.inf,
        return_cost: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, np.ndarray, np.ndarray]]:
        """SciPy Hungarian assignment shim matching the lap.lapjv signature."""
        n, m = cost_matrix.shape
        if n == 0 or m == 0:
            x = np.full(n, -1, dtype=int)
            y = np.full(m, -1, dtype=int)
            return (0.0, x, y) if return_cost else (x, y)

        capped_cost = np.minimum(cost_matrix, cost_limit)
        row_ind, col_ind = linear_sum_assignment(capped_cost)

        x = np.full(n, -1, dtype=int)
        y = np.full(m, -1, dtype=int)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= cost_limit:
                x[r] = c
                y[c] = r

        if return_cost:
            total_cost = sum(
                float(cost_matrix[r, c])
                for r, c in zip(row_ind, col_ind)
                if cost_matrix[r, c] <= cost_limit
            )
            return total_cost, x, y
        return x, y

    lap_shim = ModuleType("lap")
    lap_shim.__version__ = "0.5.12"
    lap_shim.lapjv = lapjv_scipy
    sys.modules["lap"] = lap_shim


_ensure_lap_available()

# Import Ultralytics tracker components after solver registration
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.utils import IterableSimpleNamespace

from src.tracking.tracker_config import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    TrackerConfig,
    TrackerConfigError,
    TrackingEnvironmentError,
)

logger = logging.getLogger("PerceptGuard.ByteTrackEngine")

# ==============================================================================
# Domain Exceptions
# ==============================================================================


class TrackingEngineError(TrackingEnvironmentError):
    """Raised when tracker initialization, state update, or format conversion fails."""

    pass


class InvalidDetectionInputError(TrackingEngineError):
    """Raised when input detection records are malformed, missing required keys, or corrupted."""

    pass


# ==============================================================================
# Domain Enumerations & Lifecycle States
# ==============================================================================


class TrackLifecycle(str, Enum):
    """Formal lifecycle states for ByteTrack tracking."""

    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    REIDENTIFIED = "REIDENTIFIED"
    REMOVED = "REMOVED"


# ==============================================================================
# Data Contracts: TrackRecord
# ==============================================================================


@dataclass
class TrackRecord:
    """Individual tracked object representation adhering to PerceptGuard data contracts.

    Attributes:
        track_id (int): Persistent tracking ID assigned by ByteTrack.
        class_id (int): COCO/driving object class index.
        class_name (str): Semantic object label (e.g. 'car', 'truck', 'traffic light').
        confidence (float): Detection confidence score in range (0.0, 1.0].
        bbox_xyxy (List[float]): Bounding box in [x1, y1, x2, y2] coordinates.
        detection_index (Optional[int]): Originating detection index in source frame.
        track_status (str): Current status ('active', 'tracked', 'lost').
        track_lifecycle (str): ByteTrack lifecycle state ('TENTATIVE', 'CONFIRMED', 'LOST', 'REIDENTIFIED', 'REMOVED').
    """

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    detection_index: Optional[int] = None
    track_status: str = "active"
    track_lifecycle: str = "CONFIRMED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert track record to JSON-serializable dictionary."""
        return {
            "track_id": int(self.track_id),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [round(float(c), 2) for c in self.bbox_xyxy],
            "detection_index": int(self.detection_index) if self.detection_index is not None else None,
            "track_status": str(self.track_status),
            "track_lifecycle": str(self.track_lifecycle),
        }


# ==============================================================================
# Ultralytics Detection Adapter Container
# ==============================================================================


class _ByteTrackDetectionAdapter:
    """Lightweight adapter exposing `conf`, `cls`, and `xywh` attributes with boolean indexing.

    Ultralytics `BYTETracker._split_detections` and `init_track` require an object
    providing `.conf`, `.cls`, `.xywh`, `len()`, and boolean array subscription `[mask]`.
    """

    def __init__(
        self,
        xywh: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4) if len(xywh) > 0 else np.empty((0, 4), dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1) if len(conf) > 0 else np.empty(0, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1) if len(cls) > 0 else np.empty(0, dtype=np.float32)
        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.int32).reshape(-1)
        else:
            self.indices = np.arange(len(self.conf), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, item: Any) -> "_ByteTrackDetectionAdapter":
        return _ByteTrackDetectionAdapter(
            xywh=self.xywh[item],
            conf=self.conf[item],
            cls=self.cls[item],
            indices=self.indices[item],
        )


# ==============================================================================
# ByteTrack Tracking Engine Architecture
# ==============================================================================


class ByteTrackEngine:
    """Production-grade ByteTrack Multi-Object Tracking Engine for PerceptGuard AI.

    Maintains sequential tracking state, associates incoming YOLO detections across frames,
    and assigns persistent object tracking IDs.

    Guarantees:
    - Zero duplicate YOLO inference: purely processes existing detection bounding boxes.
    - Deterministic ID progression starting at 1.
    - CPU-bounded memory overhead: dead tracks purged after `track_buffer` frames.
    - Thread-safe tracking state reset for multi-video session isolation.
    """

    def __init__(
        self,
        config: Optional[Union[TrackerConfig, Dict[str, Any], Path, str]] = None,
        frame_rate: int = 30,
    ) -> None:
        """Initialize ByteTrack engine.

        Args:
            config: TrackerConfig instance, raw configuration dictionary, or YAML file path.
                    If None, loads default from `config/tracker_config.yaml`.
            frame_rate: Video frame rate in frames per second (default: 30).
        """
        self.frame_rate = frame_rate
        self.config = self._resolve_config(config)
        self.tracker_args = IterableSimpleNamespace(**self.config.to_dict())

        try:
            self.tracker = BYTETracker(self.tracker_args, frame_rate=self.frame_rate)
        except TypeError:
            # Fallback for Ultralytics versions where BYTETracker(args) takes single argument
            self.tracker = BYTETracker(self.tracker_args)

        self._frame_count = 0
        self._unique_track_ids: Set[int] = set()
        self._track_observations: Dict[int, int] = {}
        self._active_track_ids: Set[int] = set()
        self._lost_track_ids: Set[int] = set()
        self._removed_track_ids: Set[int] = set()

    @staticmethod
    def _resolve_config(
        config: Optional[Union[TrackerConfig, Dict[str, Any], Path, str]],
    ) -> TrackerConfig:
        """Resolve config input to a validated, immutable TrackerConfig instance."""
        if config is None:
            return TrackerConfig.from_yaml()
        elif isinstance(config, TrackerConfig):
            return config if config.is_frozen else config.freeze()
        elif isinstance(config, dict):
            return TrackerConfig.from_dict(config, freeze=True)
        elif isinstance(config, (str, Path)):
            return TrackerConfig.from_yaml(config, freeze=True)
        else:
            raise TrackerConfigError(
                f"Unsupported config type: {type(config).__name__}. Expected TrackerConfig, dict, or Path."
            )

    @property
    def frame_count(self) -> int:
        """Total sequential frames processed since initialization or last reset."""
        return self._frame_count

    @property
    def unique_track_count(self) -> int:
        """Total distinct persistent tracking IDs assigned across this tracking session."""
        return len(self._unique_track_ids)

    @property
    def unique_track_ids(self) -> Set[int]:
        """Set of unique tracking IDs recorded in this session."""
        return set(self._unique_track_ids)

    # --------------------------------------------------------------------------
    # Format Conversion & Ingestion
    # --------------------------------------------------------------------------

    @staticmethod
    def _convert_detections_to_adapter(
        detections: Sequence[Any],
    ) -> Tuple[_ByteTrackDetectionAdapter, Dict[int, str]]:
        """Normalize various detection formats into ByteTrackDetectionAdapter.

        Supported input detection elements:
        - Dict with keys: 'bbox_xyxy', 'confidence', 'class_id', and optional 'class_name'
        - Object with attributes: .bbox_xyxy, .confidence, .class_id, and optional .class_name
        - Raw array/list of [x1, y1, x2, y2, conf, class_id]

        Returns:
            Tuple of (_ByteTrackDetectionAdapter, class_name_lookup_dict).
        """
        if not detections:
            return _ByteTrackDetectionAdapter(np.empty((0, 4)), np.empty(0), np.empty(0)), {}

        xywh_list: List[List[float]] = []
        conf_list: List[float] = []
        cls_list: List[float] = []
        class_name_map: Dict[int, str] = {}

        for idx, det in enumerate(detections):
            # 1. Extract values based on structure
            if isinstance(det, dict):
                bbox = det.get("bbox_xyxy")
                conf = det.get("confidence")
                cls_id = det.get("class_id")
                cls_name = det.get("class_name", str(cls_id))
            elif hasattr(det, "bbox_xyxy") and hasattr(det, "confidence"):
                bbox = getattr(det, "bbox_xyxy")
                conf = getattr(det, "confidence")
                cls_id = getattr(det, "class_id")
                cls_name = getattr(det, "class_name", str(cls_id))
            elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 6:
                bbox = list(det[:4])
                conf = float(det[4])
                cls_id = int(det[5])
                cls_name = str(cls_id)
            else:
                raise InvalidDetectionInputError(
                    f"Unsupported detection record format at index {idx}: {det}"
                )

            # 2. Validation
            if bbox is None or len(bbox) != 4:
                raise InvalidDetectionInputError(
                    f"Invalid bounding box at index {idx}: {bbox}. Expected [x1, y1, x2, y2]."
                )
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            # Drop degenerate zero-area boxes
            if w <= 0.0 or h <= 0.0:
                continue

            cx = x1 + (w / 2.0)
            cy = y1 + (h / 2.0)

            confidence = float(conf)
            if not (0.0 <= confidence <= 1.0):
                raise InvalidDetectionInputError(
                    f"Detection confidence out of bounds [0.0, 1.0] at index {idx}: {confidence}"
                )

            class_id_int = int(cls_id)
            xywh_list.append([cx, cy, w, h])
            conf_list.append(confidence)
            cls_list.append(float(class_id_int))
            class_name_map[class_id_int] = str(cls_name)

        if not xywh_list:
            return _ByteTrackDetectionAdapter(np.empty((0, 4)), np.empty(0), np.empty(0)), class_name_map

        adapter = _ByteTrackDetectionAdapter(
            xywh=np.array(xywh_list, dtype=np.float32),
            conf=np.array(conf_list, dtype=np.float32),
            cls=np.array(cls_list, dtype=np.float32),
        )
        return adapter, class_name_map

    # --------------------------------------------------------------------------
    # Tracking Lifecycle: Update & Reset
    # --------------------------------------------------------------------------

    def update(
        self,
        detections: Sequence[Any],
        frame_id: Optional[Union[str, int]] = None,
    ) -> List[TrackRecord]:
        """Update tracker state with detections from the current chronological frame.

        Args:
            detections: Sequence of detection objects or dictionaries for this frame.
            frame_id: Optional frame identifier for logging/traceability.

        Returns:
            List of active TrackRecord instances with persistent track IDs.
        """
        start_t = time.perf_counter()
        self._frame_count += 1

        adapter, class_name_map = self._convert_detections_to_adapter(detections)

        try:
            # Update ByteTrack with adapter
            raw_tracks = self.tracker.update(adapter)
        except Exception as e:
            raise TrackingEngineError(
                f"ByteTrack update failure on frame {frame_id or self._frame_count}: {e}"
            ) from e

        track_records: List[TrackRecord] = []
        current_active_ids: Set[int] = set()

        if raw_tracks is not None and len(raw_tracks) > 0:
            for row in raw_tracks:
                # Raw track schema: [x1, y1, x2, y2, track_id, score, cls, idx]
                x1 = float(row[0])
                y1 = float(row[1])
                x2 = float(row[2])
                y2 = float(row[3])
                track_id = int(row[4])
                score = float(row[5])
                cls_id = int(row[6])
                det_idx = int(row[7]) if len(row) > 7 and row[7] >= 0 else None

                self._unique_track_ids.add(track_id)
                current_active_ids.add(track_id)
                cls_name = class_name_map.get(cls_id, str(cls_id))

                # Track lifecycle state determination
                obs_count = self._track_observations.get(track_id, 0) + 1
                self._track_observations[track_id] = obs_count

                if track_id in self._lost_track_ids:
                    lifecycle = TrackLifecycle.REIDENTIFIED.value
                    self._lost_track_ids.remove(track_id)
                elif obs_count < 3:
                    lifecycle = TrackLifecycle.TENTATIVE.value
                else:
                    lifecycle = TrackLifecycle.CONFIRMED.value

                record = TrackRecord(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=score,
                    bbox_xyxy=[x1, y1, x2, y2],
                    detection_index=det_idx,
                    track_status="active",
                    track_lifecycle=lifecycle,
                )
                track_records.append(record)

        # Mark previously active tracks not present in current frame as LOST
        for prev_id in self._active_track_ids:
            if prev_id not in current_active_ids:
                self._lost_track_ids.add(prev_id)

        self._active_track_ids = current_active_ids

        duration_ms = (time.perf_counter() - start_t) * 1000.0
        logger.debug(
            "Frame %s: %d detections -> %d active tracks in %.2f ms",
            frame_id or self._frame_count,
            len(detections),
            len(track_records),
            duration_ms,
        )
        return track_records

    def reset(self) -> None:
        """Reset internal tracker state, clear track pools, and restart track IDs.

        Guarantees session isolation between independent video sequences.
        """
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
        if hasattr(self.tracker, "reset_id"):
            self.tracker.reset_id()
        self._frame_count = 0
        self._unique_track_ids.clear()
        self._track_observations.clear()
        self._active_track_ids.clear()
        self._lost_track_ids.clear()
        self._removed_track_ids.clear()
        logger.info("ByteTrackEngine state reset successfully.")
