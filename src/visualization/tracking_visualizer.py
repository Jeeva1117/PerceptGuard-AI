"""PerceptGuard AI: Production Multi-Object Tracking Visualization Engine.

Senior Staff-level, production-grade tracking visualization engine for autonomous driving
perception pipelines. Converts sequential multi-object tracking results into annotated
video frames and optional video sequences with persistent object identities, deterministic
track colors, historical movement trajectories, centroid anchors, and frame telemetry overlays.

Key Architectural Guarantees:
- Decoupled from model passes: zero duplicate YOLO inference, zero ByteTrack re-runs.
- Deterministic, mathematically reproducible track colors: track ID color is constant across all frames and runs.
- Trajectory visualization: connects valid historical centroid observations with bounded length.
- High-contrast, non-clipping bounding boxes and label banners (e.g. 'ID: 12 | Car | 0.91').
- Memory-bounded sequential execution: processes one frame at a time (< 250 MB RSS).
- Clean handling of zero-tracking frames without failure.
- Optional OpenCV VideoWriter integration with decode validation and codec fallback.
- Versioned Schema v1.0 multi-file persistence:
  - annotated_frames/
  - annotated_video.mp4 (optional)
  - visualization_summary.json
  - visualization_metadata.json
  - errors.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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

from src.tracking.tracking_pipeline import _atomic_write_json, _natural_sort_key

logger = logging.getLogger("PerceptGuard.TrackingVisualizer")


# ==============================================================================
# Domain Exceptions
# ==============================================================================


class TrackingVisualizationError(Exception):
    """Base exception for all tracking visualization failures."""

    pass


class TrackingJSONValidationError(TrackingVisualizationError):
    """Raised when tracking results JSON fails structural or semantic validation."""

    pass


class TrackingSourceFrameNotFoundError(TrackingVisualizationError, FileNotFoundError):
    """Raised when the source image corresponding to a tracking frame cannot be resolved."""

    pass


class TrackingSourceFrameDecodeError(TrackingVisualizationError):
    """Raised when the source image file is corrupted or cannot be decoded."""

    pass


class TrackingVideoGenerationError(TrackingVisualizationError):
    """Raised when VideoWriter fails to open or encode video output."""

    pass


# ==============================================================================
# Deterministic Track Color Strategy
# ==============================================================================


def get_deterministic_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a mathematically deterministic BGR color for any integer tracking ID.

    Uses golden-ratio hue spacing in HSV space to guarantee:
    1. The same track ID always receives the exact same color across all frames.
    2. Different track IDs receive visually distinct, high-contrast colors.
    """
    golden_ratio_conjugate = 0.618033988749895
    h = ((int(abs(track_id)) * golden_ratio_conjugate) + 0.1) % 1.0
    h_deg = h * 360.0
    s = 0.85
    v = 0.95
    c = s * v
    x = c * (1.0 - abs((h_deg / 60.0) % 2 - 1.0))
    m = v - c

    if 0.0 <= h_deg < 60.0:
        r_p, g_p, b_p = c, x, 0.0
    elif 60.0 <= h_deg < 120.0:
        r_p, g_p, b_p = x, c, 0.0
    elif 120.0 <= h_deg < 180.0:
        r_p, g_p, b_p = 0.0, c, x
    elif 180.0 <= h_deg < 240.0:
        r_p, g_p, b_p = 0.0, x, c
    elif 240.0 <= h_deg < 300.0:
        r_p, g_p, b_p = x, 0.0, c
    else:
        r_p, g_p, b_p = c, 0.0, x

    b = int(round((b_p + m) * 255.0))
    g = int(round((g_p + m) * 255.0))
    r = int(round((r_p + m) * 255.0))
    return (b, g, r)


# ==============================================================================
# Data Contracts: Annotation, Frame Record & Summary
# ==============================================================================


@dataclass(frozen=True)
class TrackAnnotation:
    """Immutable data contract representing a validated visual track annotation."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    center_x: float
    center_y: float
    color_bgr: Tuple[int, int, int]
    lifecycle_state: Optional[str] = None


@dataclass
class FrameTrackingVisRecord:
    """Visualization outcome and metrics for an individual frame."""

    frame_id: str
    source_image_path: str
    output_image_path: Optional[str] = None
    extracted_frame_id: Optional[int] = None
    source_frame_index: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    track_count: int = 0
    tracks_rendered: int = 0
    status: str = "success"  # "success", "empty_frame", "failed"
    error_message: Optional[str] = None
    visualization_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame record to JSON dictionary."""
        return {
            "frame_id": self.frame_id,
            "source_image_path": self.source_image_path,
            "output_image_path": self.output_image_path,
            "extracted_frame_id": self.extracted_frame_id,
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": round(float(self.timestamp_seconds), 4) if self.timestamp_seconds is not None else None,
            "track_count": int(self.track_count),
            "tracks_rendered": int(self.tracks_rendered),
            "status": self.status,
            "error_message": self.error_message,
            "visualization_time_ms": round(float(self.visualization_time_ms), 2),
        }


@dataclass
class TrackingVisSummary:
    """Aggregated operational telemetry and summary for a tracking visualization run."""

    schema_version: str = "1.0"
    run_id: str = ""
    dataset_id: str = ""
    tracking_run_id: str = ""
    tracking_file: str = ""
    total_frames_in_tracking_results: int = 0
    total_frames_attempted: int = 0
    successful_visualizations: int = 0
    failed_visualizations: int = 0
    empty_frames: int = 0
    total_tracks_rendered: int = 0
    unique_track_ids_visualized: int = 0
    video_generated: bool = False
    video_path: Optional[str] = None
    video_fps: Optional[float] = None
    total_visualization_time_seconds: float = 0.0
    average_visualization_time_ms: float = 0.0
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "PASS"  # "PASS", "WARNING", "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to JSON dictionary."""
        return asdict(self)


# ==============================================================================
# Tracking Visualizer Engine Architecture
# ==============================================================================


class TrackingVisualizer:
    """Production Tracking Visualization Engine for PerceptGuard AI.

    Decoupled post-processing engine that converts multi-object and single-object tracking
    JSON into annotated frames and optional MP4 video sequences.
    """

    def __init__(
        self,
        show_trajectory: bool = True,
        trajectory_length: int = 30,
        show_confidence: bool = True,
        show_frame_info: bool = True,
        line_thickness: Optional[int] = None,
        font_scale: Optional[float] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize TrackingVisualizer.

        Args:
            show_trajectory: Whether to draw historical motion trajectory lines.
            trajectory_length: Maximum past frame points to retain per trajectory.
            show_confidence: Whether to display detection confidence score in label.
            show_frame_info: Whether to render corner information overlay.
            line_thickness: Custom line thickness (auto-calculated if None).
            font_scale: Custom font scale (auto-calculated if None).
            project_root: Repository root path.
        """
        self.show_trajectory = show_trajectory
        self.trajectory_length = max(1, int(trajectory_length))
        self.show_confidence = show_confidence
        self.show_frame_info = show_frame_info
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.project_root = Path(project_root).resolve() if project_root else _project_root

        # Internal state
        # Mapping: track_id -> List[Tuple[x, y, sequence_index]]
        self.trajectories: Dict[int, List[Tuple[int, int, int]]] = {}
        self.frame_records: List[FrameTrackingVisRecord] = []
        self.error_records: List[Dict[str, Any]] = []
        self.unique_track_ids: Set[int] = set()

    # --------------------------------------------------------------------------
    # Single-Frame Annotation Rendering
    # --------------------------------------------------------------------------

    def annotate_frame(
        self,
        image: np.ndarray,
        tracks: Sequence[Any],
        frame_id: str,
        sequence_index: int = 0,
        source_frame_index: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
        run_id: str = "",
    ) -> Tuple[np.ndarray, int]:
        """Annotate a single image copy with tracking bounding boxes, centers, trajectories, and overlay.

        Args:
            image: Source BGR numpy image array (never modified in place).
            tracks: Sequence of track dictionaries or TrackRecord/TrackedObject instances.
            frame_id: Frame identifier string.
            sequence_index: Monotonic sequential frame index in visualization run.
            source_frame_index: Optional physical video frame index.
            timestamp_seconds: Optional elapsed video timestamp.
            run_id: Visualization run identifier for overlay.

        Returns:
            Tuple of (annotated_image_matrix, count_of_tracks_rendered).
        """
        canvas = image.copy()
        img_h, img_w = canvas.shape[:2]

        # Resolution-adaptive styling
        box_thickness = self.line_thickness or max(1, int(round(min(img_w, img_h) / 320.0)))
        font_scale = self.font_scale or max(0.35, min(img_w, img_h) / 800.0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = max(1, int(round(box_thickness * 0.75)))

        rendered_count = 0
        current_frame_track_ids: Set[int] = set()

        # 1. Parse and validate track annotations
        annotations: List[TrackAnnotation] = []
        for t in tracks:
            ann = self._parse_track_annotation(t, img_w, img_h)
            if ann is not None:
                annotations.append(ann)
                self.unique_track_ids.add(ann.track_id)
                current_frame_track_ids.add(ann.track_id)

                # Record centroid observation in historical trajectory
                cx_int = int(round(ann.center_x))
                cy_int = int(round(ann.center_y))
                if ann.track_id not in self.trajectories:
                    self.trajectories[ann.track_id] = []
                self.trajectories[ann.track_id].append((cx_int, cy_int, sequence_index))

                # Cap trajectory length
                if len(self.trajectories[ann.track_id]) > self.trajectory_length:
                    self.trajectories[ann.track_id] = self.trajectories[ann.track_id][-self.trajectory_length:]

        # 2. Draw Historical Trajectories
        if self.show_trajectory:
            self._render_trajectories(canvas, box_thickness)

        # 3. Draw Bounding Boxes, Centroids & Labels
        for ann in annotations:
            x1, y1, x2, y2 = [int(round(c)) for c in ann.bbox_xyxy]
            color = ann.color_bgr

            # Bounding box rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

            # Center point circle (inner white, outer track color)
            cx = int(round(ann.center_x))
            cy = int(round(ann.center_y))
            cv2.circle(canvas, (cx, cy), max(3, box_thickness + 1), color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), max(1, box_thickness - 1), (255, 255, 255), -1, cv2.LINE_AA)

            # Label text formatting: e.g. "ID: 12 | car | 0.91"
            parts = [f"ID: {ann.track_id}", str(ann.class_name)]
            if self.show_confidence:
                parts.append(f"{ann.confidence:.2f}")
            if ann.lifecycle_state and ann.lifecycle_state != "TRACKING":
                parts.append(f"[{ann.lifecycle_state}]")
            label_text = " | ".join(parts)

            # Label banner geometry
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            pad = 3
            banner_w = text_w + 2 * pad
            banner_h = text_h + baseline + 2 * pad

            # Placement: above box or inside box top if overflowing top edge
            if y1 - banner_h >= 0:
                bg_y1 = y1 - banner_h
                bg_y2 = y1
                text_y = y1 - baseline - pad
            else:
                bg_y1 = y1
                bg_y2 = y1 + banner_h
                text_y = y1 + text_h + pad

            # Clamping right edge
            bg_x1 = max(0, min(x1, img_w - banner_w))
            bg_x2 = min(img_w, bg_x1 + banner_w)
            text_x = bg_x1 + pad

            # Solid banner background
            cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

            # Contrasting text color based on banner luminance
            b, g, r = color
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            text_color = (0, 0, 0) if luminance > 0.55 else (255, 255, 255)

            cv2.putText(
                canvas,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
            rendered_count += 1

        # 4. Optional Information Overlay
        if self.show_frame_info:
            self._render_frame_overlay(
                canvas=canvas,
                frame_id=frame_id,
                source_frame_index=source_frame_index,
                timestamp_seconds=timestamp_seconds,
                active_tracks_count=len(annotations),
                run_id=run_id,
                img_w=img_w,
                img_h=img_h,
            )

        return canvas, rendered_count

    def _render_trajectories(self, canvas: np.ndarray, base_thickness: int) -> None:
        """Draw historical connecting lines between consecutive observations for each track."""
        for t_id, points in self.trajectories.items():
            if len(points) < 2:
                continue

            color = get_deterministic_track_color(t_id)
            for i in range(len(points) - 1):
                p1_x, p1_y, idx1 = points[i]
                p2_x, p2_y, idx2 = points[i + 1]

                # Only connect points if the gap between observations is small (<= 2 frames)
                # to prevent drawing spurious lines across prolonged occlusion periods
                if abs(idx2 - idx1) <= 2:
                    cv2.line(
                        canvas,
                        (p1_x, p1_y),
                        (p2_x, p2_y),
                        color,
                        max(1, int(round(base_thickness * 0.75))),
                        cv2.LINE_AA,
                    )

    def _render_frame_overlay(
        self,
        canvas: np.ndarray,
        frame_id: str,
        source_frame_index: Optional[int],
        timestamp_seconds: Optional[float],
        active_tracks_count: int,
        run_id: str,
        img_w: int,
        img_h: int,
    ) -> None:
        """Render compact, semi-transparent top-left information banner."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, min(img_w, img_h) / 1000.0)
        thickness = 1

        lines = [f"Frame: {frame_id}"]
        if source_frame_index is not None:
            lines.append(f"Index: {source_frame_index}")
        if timestamp_seconds is not None:
            lines.append(f"Time: {timestamp_seconds:.2f}s")
        lines.append(f"Active Tracks: {active_tracks_count}")
        if run_id:
            lines.append(f"Run: {run_id[:16]}")

        # Compute overlay box size
        max_w = 0
        total_h = 6
        line_heights = []
        for line in lines:
            (w, h), bl = cv2.getTextSize(line, font, font_scale, thickness)
            max_w = max(max_w, w)
            line_h = h + bl + 4
            line_heights.append((h, bl, line_h))
            total_h += line_h

        overlay_w = max_w + 14
        overlay_h = total_h

        # Draw semi-transparent dark banner
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (8 + overlay_w, 8 + overlay_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)
        cv2.rectangle(canvas, (8, 8), (8 + overlay_w, 8 + overlay_h), (80, 80, 80), 1, cv2.LINE_AA)

        # Draw text lines
        cur_y = 12
        for idx, line in enumerate(lines):
            h, bl, lh = line_heights[idx]
            cur_y += h + 2
            cv2.putText(
                canvas,
                line,
                (14, cur_y),
                font,
                font_scale,
                (240, 240, 240),
                thickness,
                cv2.LINE_AA,
            )
            cur_y += bl + 2

    def _parse_track_annotation(
        self,
        track_obj: Any,
        img_w: int,
        img_h: int,
    ) -> Optional[TrackAnnotation]:
        """Validate and extract TrackAnnotation contract with coordinate clamping."""
        if isinstance(track_obj, dict):
            t_id = track_obj.get("track_id")
            cls_id = track_obj.get("class_id", 0)
            cls_name = track_obj.get("class_name", str(cls_id))
            conf = track_obj.get("confidence", 1.0)
            bbox = track_obj.get("bbox_xyxy")
            cx = track_obj.get("center_x")
            cy = track_obj.get("center_y")
            state = track_obj.get("lifecycle_state")
        else:
            t_id = getattr(track_obj, "track_id", None)
            cls_id = getattr(track_obj, "class_id", 0)
            cls_name = getattr(track_obj, "class_name", str(cls_id))
            conf = getattr(track_obj, "confidence", 1.0)
            bbox = getattr(track_obj, "bbox_xyxy", None)
            cx = getattr(track_obj, "center_x", None)
            cy = getattr(track_obj, "center_y", None)
            state = getattr(track_obj, "lifecycle_state", None)

        if t_id is None or bbox is None or len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = [float(c) for c in bbox]
            t_id_int = int(t_id)
            cls_id_int = int(cls_id) if cls_id is not None else 0
            conf_val = float(conf) if conf is not None else 1.0
        except (ValueError, TypeError):
            return None

        # Clamp to image dimensions
        clamped_x1 = max(0.0, min(float(img_w), x1))
        clamped_y1 = max(0.0, min(float(img_h), y1))
        clamped_x2 = max(0.0, min(float(img_w), x2))
        clamped_y2 = max(0.0, min(float(img_h), y2))

        # Reject degenerate bounding boxes
        if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
            return None

        # Calculate center if missing
        calc_cx = cx if cx is not None else (clamped_x1 + clamped_x2) / 2.0
        calc_cy = cy if cy is not None else (clamped_y1 + clamped_y2) / 2.0

        color = get_deterministic_track_color(t_id_int)

        return TrackAnnotation(
            track_id=t_id_int,
            class_id=cls_id_int,
            class_name=str(cls_name),
            confidence=conf_val,
            bbox_xyxy=[clamped_x1, clamped_y1, clamped_x2, clamped_y2],
            center_x=float(calc_cx),
            center_y=float(calc_cy),
            color_bgr=color,
            lifecycle_state=str(state) if state else None,
        )

    # --------------------------------------------------------------------------
    # Sequence Processing & Output Persistence
    # --------------------------------------------------------------------------

    def run(
        self,
        tracking_file: Union[str, Path],
        frames_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        save_video: bool = False,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        overwrite: bool = False,
    ) -> Tuple[TrackingVisSummary, List[FrameTrackingVisRecord]]:
        """Execute tracking visualization across a sequential tracking run.

        Args:
            tracking_file: Path to tracking results JSON (MOT, SOT, or ByteTrack pipeline).
            frames_dir: Optional directory containing source video frames.
            output_dir: Destination folder for visualization artifacts.
            save_video: Whether to generate an annotated MP4 video.
            fps: Video frames per second (auto-detected if None).
            max_frames: Frame limit cap.
            start_frame: Start index.
            end_frame: End index.
            overwrite: Whether to overwrite existing destination folder.

        Returns:
            Tuple of (TrackingVisSummary, List[FrameTrackingVisRecord]).
        """
        start_t = time.perf_counter()
        t_path = Path(tracking_file)
        if not t_path.is_absolute():
            t_path = (self.project_root / t_path).resolve()

        if not t_path.exists() or not t_path.is_file():
            raise TrackingVisualizationError(f"Tracking results file not found: {t_path}")

        with open(t_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_id = data.get("dataset_id", "driving_sample")
        tracking_run_id = data.get("run_id", "unknown_tracking_run")
        raw_frame_results = data.get("frame_results", [])
        if not raw_frame_results and "observations" in data:
            # Single object tracking format adapter
            raw_frame_results = [
                {
                    "frame_id": obs.get("frame_id"),
                    "source_frame_index": obs.get("source_frame_index"),
                    "timestamp_seconds": obs.get("timestamp_seconds"),
                    "tracks": [obs] if obs.get("is_visible") else [],
                }
                for obs in data.get("observations", [])
            ]

        if not raw_frame_results:
            raise TrackingJSONValidationError(f"No frame records found in tracking JSON: {t_path}")

        # Resolve frames source directory
        resolved_frames_dir = self._resolve_frames_directory(frames_dir, data)

        # Natural alphanumeric sorting
        sorted_frames = sorted(
            raw_frame_results,
            key=lambda x: _natural_sort_key(x.get("image_filename", x.get("frame_id", ""))),
        )
        total_discovered = len(sorted_frames)

        s_idx = start_frame if start_frame is not None else 0
        e_idx = end_frame if end_frame is not None else total_discovered
        sliced_frames = sorted_frames[s_idx:e_idx]
        if max_frames is not None and max_frames > 0:
            sliced_frames = sliced_frames[:max_frames]

        is_smoke = len(sliced_frames) < total_discovered

        # Reset visualizer state
        self.reset()

        # Destination resolution
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"vis_{dataset_id}_{timestamp_slug}_{os.urandom(3).hex()}"
        if output_dir is not None:
            resolved_out = Path(output_dir).resolve()
        else:
            resolved_out = (
                self.project_root
                / "datasets"
                / "processed"
                / dataset_id
                / "visualizations"
                / "tracking"
                / run_id
            ).resolve()

        if resolved_out.exists() and not overwrite:
            raise FileExistsError(
                f"Output directory exists: {resolved_out}. Pass overwrite=True to replace."
            )

        annotated_frames_dir = resolved_out / "annotated_frames"
        annotated_frames_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optional VideoWriter
        video_writer = None
        video_path = resolved_out / "annotated_video.mp4"
        effective_fps = fps or self._detect_fps(sliced_frames)

        successful_count = 0
        failed_count = 0
        empty_count = 0
        total_rendered = 0
        frame_latencies: List[float] = []

        for seq_idx, f_item in enumerate(sliced_frames):
            t_frame_start = time.perf_counter()
            f_name = f_item.get("image_filename", f_item.get("frame_id", f"frame_{seq_idx + 1:06d}.jpg"))
            src_idx = f_item.get("source_frame_index", f_item.get("extracted_frame_id"))
            ts = f_item.get("timestamp_seconds")
            tracks = f_item.get("tracks", f_item.get("active_tracks", []))

            # Resolve source frame image file
            src_image_path = self._find_source_image(f_name, resolved_frames_dir)
            if src_image_path is None or not src_image_path.exists():
                err_msg = f"Source frame image not found: {f_name}"
                logger.error(err_msg)
                self.error_records.append({
                    "frame_id": f_name,
                    "error": err_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                self.frame_records.append(
                    FrameTrackingVisRecord(
                        frame_id=f_name,
                        source_image_path=str(src_image_path or f_name),
                        output_image_path=None,
                        status="failed",
                        error_message=err_msg,
                    )
                )
                failed_count += 1
                continue

            # Read source image
            img_bgr = cv2.imread(str(src_image_path))
            if img_bgr is None:
                err_msg = f"OpenCV failed to decode source image: {src_image_path}"
                logger.error(err_msg)
                self.error_records.append({
                    "frame_id": f_name,
                    "error": err_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                self.frame_records.append(
                    FrameTrackingVisRecord(
                        frame_id=f_name,
                        source_image_path=str(src_image_path),
                        output_image_path=None,
                        status="failed",
                        error_message=err_msg,
                    )
                )
                failed_count += 1
                continue

            # Annotate frame
            annotated_img, rendered_boxes = self.annotate_frame(
                image=img_bgr,
                tracks=tracks,
                frame_id=f_name,
                sequence_index=seq_idx,
                source_frame_index=src_idx,
                timestamp_seconds=ts,
                run_id=run_id,
            )

            # Save annotated image
            out_img_path = annotated_frames_dir / f_name
            cv2.imwrite(str(out_img_path), annotated_img)

            # Write to video if enabled
            if save_video:
                if video_writer is None:
                    h, w = annotated_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, effective_fps, (w, h))
                    if not video_writer.isOpened():
                        logger.warning("VideoWriter with mp4v failed to open, trying XVID...")
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        video_writer = cv2.VideoWriter(str(video_path), fourcc, effective_fps, (w, h))

                if video_writer.isOpened():
                    video_writer.write(annotated_img)

            proc_ms = (time.perf_counter() - t_frame_start) * 1000.0
            frame_latencies.append(proc_ms)
            total_rendered += rendered_boxes
            successful_count += 1

            status = "empty_frame" if rendered_boxes == 0 and not tracks else "success"
            if status == "empty_frame":
                empty_count += 1

            self.frame_records.append(
                FrameTrackingVisRecord(
                    frame_id=f_name,
                    source_image_path=str(src_image_path),
                    output_image_path=str(out_img_path),
                    extracted_frame_id=src_idx,
                    source_frame_index=src_idx,
                    timestamp_seconds=ts,
                    track_count=len(tracks),
                    tracks_rendered=rendered_boxes,
                    status=status,
                    visualization_time_ms=proc_ms,
                )
            )

        # Finalize video writer
        video_success = False
        if video_writer is not None:
            video_writer.release()
            # Validate output video
            if video_path.exists() and video_path.stat().st_size > 0:
                chk = cv2.VideoCapture(str(video_path))
                if chk.isOpened() and int(chk.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                    video_success = True
                chk.release()

        total_proc_sec = time.perf_counter() - start_t
        avg_latency = float(np.mean(frame_latencies)) if frame_latencies else 0.0

        status_flag = "PASS"
        if is_smoke or failed_count > 0:
            status_flag = "WARNING"
        if successful_count == 0 and total_discovered > 0:
            status_flag = "FAIL"

        summary = TrackingVisSummary(
            schema_version="1.0",
            run_id=run_id,
            dataset_id=dataset_id,
            tracking_run_id=tracking_run_id,
            tracking_file=str(t_path),
            total_frames_in_tracking_results=total_discovered,
            total_frames_attempted=len(sliced_frames),
            successful_visualizations=successful_count,
            failed_visualizations=failed_count,
            empty_frames=empty_count,
            total_tracks_rendered=total_rendered,
            unique_track_ids_visualized=len(self.unique_track_ids),
            video_generated=video_success,
            video_path=str(video_path) if video_success else None,
            video_fps=effective_fps if video_success else None,
            total_visualization_time_seconds=round(total_proc_sec, 2),
            average_visualization_time_ms=round(avg_latency, 2),
            status=status_flag,
        )

        # Persist summary, metadata, errors
        self._persist_metadata_artifacts(resolved_out, summary, str(t_path), resolved_frames_dir)
        return summary, self.frame_records

    # --------------------------------------------------------------------------
    # Helper Drivers
    # --------------------------------------------------------------------------

    def _resolve_frames_directory(
        self,
        frames_dir: Optional[Union[str, Path]],
        tracking_data: Dict[str, Any],
    ) -> Path:
        """Resolve source frames directory from parameter or tracking metadata."""
        if frames_dir is not None:
            p = Path(frames_dir)
            return (self.project_root / p).resolve() if not p.is_absolute() else p

        dataset_id = tracking_data.get("dataset_id", "driving_sample")
        candidates = [
            self.project_root / "datasets" / "processed" / dataset_id / "preprocessed_frames",
            self.project_root / "datasets" / "processed" / dataset_id / "frames",
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c

        return self.project_root / "datasets" / "processed" / dataset_id / "preprocessed_frames"

    def _find_source_image(self, filename: str, frames_dir: Path) -> Optional[Path]:
        """Look for image file in frames_dir or search dataset folders."""
        exact = frames_dir / filename
        if exact.exists():
            return exact

        # Fallback to general search
        dataset_root = frames_dir.parent
        alt_frames = dataset_root / "frames" / filename
        if alt_frames.exists():
            return alt_frames

        return None

    def _detect_fps(self, sliced_frames: List[Dict[str, Any]]) -> float:
        """Detect video fps from frame timestamps or default to 30.0."""
        timestamps = [f.get("timestamp_seconds") for f in sliced_frames if f.get("timestamp_seconds") is not None]
        if len(timestamps) >= 2:
            dt = timestamps[1] - timestamps[0]
            if dt > 0.001:
                return round(1.0 / dt, 1)
        return 30.0

    def _persist_metadata_artifacts(
        self,
        output_dir: Path,
        summary: TrackingVisSummary,
        source_tracking_file: str,
        frames_dir: Path,
    ) -> None:
        """Write summary, metadata, and errors JSON."""
        # 1. visualization_summary.json
        _atomic_write_json(output_dir / "visualization_summary.json", summary.to_dict())

        # 2. visualization_metadata.json
        meta = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "created_at": summary.generated_at,
            "source_tracking_file": source_tracking_file,
            "source_frames_directory": str(frames_dir),
            "styling_parameters": {
                "show_trajectory": self.show_trajectory,
                "trajectory_length": self.trajectory_length,
                "show_confidence": self.show_confidence,
                "show_frame_info": self.show_frame_info,
                "line_thickness": self.line_thickness,
                "font_scale": self.font_scale,
            },
            "environment": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
                "opencv_version": cv2.__version__,
            },
        }
        _atomic_write_json(output_dir / "visualization_metadata.json", meta)

        # 3. errors.json
        errors_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "error_count": len(self.error_records),
            "errors": self.error_records,
        }
        _atomic_write_json(output_dir / "errors.json", errors_payload)

    def reset(self) -> None:
        """Reset internal trajectories, frame records, and session statistics."""
        self.trajectories.clear()
        self.frame_records.clear()
        self.error_records.clear()
        self.unique_track_ids.clear()
        logger.info("TrackingVisualizer state reset successfully.")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line interface for TrackingVisualizer."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Production Tracking Visualization Engine"
    )
    parser.add_argument(
        "--tracking-file",
        "-t",
        type=str,
        default=None,
        help="Path to tracking JSON file (MOT, SOT, or ByteTrack pipeline results)",
    )
    parser.add_argument(
        "--frames",
        "-f",
        type=str,
        default=None,
        help="Directory containing source video frames (auto-detected if None)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination directory for visualization outputs",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Generate annotated MP4 video sequence",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Custom video playback FPS",
    )
    parser.add_argument(
        "--show-trajectory",
        action="store_true",
        default=True,
        help="Draw object motion trajectory lines (default: True)",
    )
    parser.add_argument(
        "--no-trajectory",
        dest="show_trajectory",
        action="store_false",
        help="Disable trajectory line rendering",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=30,
        help="Maximum past observation points per trajectory line (default: 30)",
    )
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        default=True,
        help="Display confidence score in track banner",
    )
    parser.add_argument(
        "--show-frame-info",
        action="store_true",
        default=True,
        help="Render top-left frame overlay banner",
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
        help="Overwrite existing destination directory",
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
        # Default tracking file resolution if omitted
        if args.tracking_file is None:
            default_mot = _project_root / "datasets" / "processed" / "driving_sample" / "tracking" / "multi_object"
            default_pipe = _project_root / "datasets" / "processed" / "driving_sample" / "tracking"
            if default_mot.exists():
                runs = sorted([p for p in default_mot.iterdir() if p.is_dir() and p.name.startswith("mot_")])
                if runs and (runs[-1] / "multi_object_tracking.json").exists():
                    target_file = runs[-1] / "multi_object_tracking.json"
                else:
                    raise TrackingVisualizationError(f"No MOT runs found in {default_mot}")
            elif default_pipe.exists():
                runs = sorted([p for p in default_pipe.iterdir() if p.is_dir() and p.name.startswith("track_")])
                if runs and (runs[-1] / "tracking_results.json").exists():
                    target_file = runs[-1] / "tracking_results.json"
                else:
                    raise TrackingVisualizationError(f"No tracking runs found in {default_pipe}")
            else:
                raise TrackingVisualizationError("No tracking results file found. Specify via --tracking-file.")
        else:
            target_file = Path(args.tracking_file)

        visualizer = TrackingVisualizer(
            show_trajectory=args.show_trajectory,
            trajectory_length=args.trajectory_length,
            show_confidence=args.show_confidence,
            show_frame_info=args.show_frame_info,
        )

        print(f"[INFO] Initializing Tracking Visualization from: '{target_file}'...")
        summary, _ = visualizer.run(
            tracking_file=target_file,
            frames_dir=args.frames,
            output_dir=args.output,
            save_video=args.save_video,
            fps=args.fps,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overwrite=args.overwrite,
        )

        print("-" * 65)
        print("PERCEPTGUARD AI — TRACKING VISUALIZATION COMPLETE")
        print("-" * 65)
        print(f"  Run Identifier           : {summary.run_id}")
        print(f"  Dataset Identifier       : {summary.dataset_id}")
        print(f"  Source Tracking File     : {summary.tracking_file}")
        print(f"  Processed Frames         : {summary.total_frames_attempted} (Success: {summary.successful_visualizations})")
        print(f"  Empty Frames             : {summary.empty_frames}")
        print(f"  Failed Frames            : {summary.failed_visualizations}")
        print(f"  Total Tracks Rendered    : {summary.total_tracks_rendered}")
        print(f"  Unique Track IDs Rendered: {summary.unique_track_ids_visualized}")
        print(f"  Video Generated          : {summary.video_generated} ({summary.video_path or 'none'})")
        if summary.video_fps:
            print(f"  Video FPS                : {summary.video_fps:.1f}")
        print(f"  Mean Frame Latency       : {summary.average_visualization_time_ms:.2f} ms / frame")
        print(f"  Total Visualization Time : {summary.total_visualization_time_seconds:.2f} s")
        print(f"  Final Status             : [{summary.status}]")
        print("-" * 65)
        return 0

    except (TrackingSourceFrameNotFoundError, FileNotFoundError) as e:
        print(f"[ERROR] Source input not found: {e}", file=sys.stderr)
        return 2
    except FileExistsError as e:
        print(f"[ERROR] Destination conflict: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[FATAL] Tracking visualization failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
