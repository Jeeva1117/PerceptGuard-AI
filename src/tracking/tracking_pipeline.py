"""PerceptGuard AI: Production Multi-Object Tracking Pipeline.

Senior Staff-level, sequential tracking pipeline integrating YOLO detections with
ByteTrack to establish temporal tracklet identities across video frames.

Key Architectural Guarantees:
- Zero duplicate YOLO inference: ingests pre-computed batch detections or streams in a single pass.
- Sequential chronological ordering: natural sort key prevents temporal scrambling.
- State continuity: tracker maintains state across consecutive frames, resetting only across sessions.
- Bounded memory footprint: sequential processing keeps RSS usage well within 8 GB host limits.
- Multi-file persistence (Schema v1.0): atomic writing of results, summary, metadata, and errors.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.tracking.bytetrack_engine import (
    ByteTrackEngine,
    InvalidDetectionInputError,
    TrackRecord,
    TrackingEngineError,
)
from src.tracking.tracker_config import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    TrackerConfig,
    TrackerConfigError,
)

logger = logging.getLogger("PerceptGuard.TrackingPipeline")

# ==============================================================================
# Domain Exceptions
# ==============================================================================


class TrackingPipelineError(Exception):
    """Base exception for tracking pipeline failures."""

    pass


class TrackingInputNotFoundError(TrackingPipelineError, FileNotFoundError):
    """Raised when detection JSON or frames directory does not exist."""

    pass


class TrackingOutputExistsError(TrackingPipelineError, FileExistsError):
    """Raised when output directory already exists and overwrite=False."""

    pass


# ==============================================================================
# Data Contracts: FrameTrackingRecord & TrackingSummary
# ==============================================================================


@dataclass
class FrameTrackingRecord:
    """Frame-level tracking record capturing persistent tracks and timing telemetry."""

    schema_version: str = "1.0"
    dataset_id: str = "driving_sample"
    frame_id: str = ""
    image_filename: str = ""
    extracted_frame_id: Optional[int] = None
    source_frame_index: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    image_width: int = 640
    image_height: int = 360
    processing_status: str = "success"
    detection_count: int = 0
    active_track_count: int = 0
    tracks: List[TrackRecord] = field(default_factory=list)
    inference_duration_ms: Optional[float] = None
    tracking_duration_ms: float = 0.0
    total_processing_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame tracking record to JSON dictionary."""
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "frame_id": self.frame_id,
            "image_filename": self.image_filename,
            "extracted_frame_id": self.extracted_frame_id,
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "processing_status": self.processing_status,
            "detection_count": self.detection_count,
            "active_track_count": self.active_track_count,
            "tracks": [t.to_dict() for t in self.tracks],
            "inference_duration_ms": round(self.inference_duration_ms, 2) if self.inference_duration_ms is not None else None,
            "tracking_duration_ms": round(self.tracking_duration_ms, 2),
            "total_processing_duration_ms": round(self.total_processing_duration_ms, 2),
        }


@dataclass
class TrackingSummary:
    """Run-level aggregate summary of multi-object tracking execution."""

    schema_version: str = "1.0"
    run_id: str = ""
    dataset_id: str = "driving_sample"
    model_name: str = "yolo11n"
    tracker_name: str = "bytetrack"
    total_frames: int = 0
    successful_frames: int = 0
    failed_frames: int = 0
    frames_with_detections: int = 0
    frames_with_tracks: int = 0
    total_detections: int = 0
    total_track_instances: int = 0
    unique_track_count: int = 0
    average_tracks_per_frame: float = 0.0
    mean_tracking_latency_ms: float = 0.0
    total_pipeline_time_seconds: float = 0.0
    status: str = "PASS"  # "PASS", "WARNING", "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to JSON dictionary."""
        return asdict(self)


# ==============================================================================
# Helper Utilities: Natural Sorting & Atomic Serialization
# ==============================================================================


def _natural_sort_key(s: str) -> List[Union[int, str]]:
    """Tokenize string into list of strings and integers for natural alphanumeric sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def _atomic_write_json(file_path: Path, payload: Dict[str, Any]) -> None:
    """Safely serialize JSON dictionary using temporary file replacement."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    temp_path.replace(file_path)


# ==============================================================================
# Tracking Pipeline Architecture
# ==============================================================================


class TrackingPipeline:
    """Manages sequential frame ingestion, ByteTrack state lifecycle, and output persistence."""

    def __init__(
        self,
        config: Optional[Union[TrackerConfig, Dict[str, Any], Path, str]] = None,
        dataset_id: str = "driving_sample",
        project_root: Optional[Union[str, Path]] = None,
        frame_rate: int = 30,
    ) -> None:
        """Initialize tracking pipeline.

        Args:
            config: Tracker configuration instance, dict, or path.
            dataset_id: Dataset identifier (default: 'driving_sample').
            project_root: Repository root path.
            frame_rate: Frame rate of the video sequence.
        """
        self.project_root = Path(project_root).resolve() if project_root else _project_root
        self.dataset_id = dataset_id
        self.frame_rate = frame_rate
        self.config = TrackerConfig.from_yaml() if config is None else ByteTrackEngine._resolve_config(config)
        self.engine = ByteTrackEngine(config=self.config, frame_rate=self.frame_rate)

    # --------------------------------------------------------------------------
    # Mode 1: Ingest Pre-computed Detection Results
    # --------------------------------------------------------------------------

    def run_from_detection_results(
        self,
        detection_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        overwrite: bool = False,
    ) -> Tuple[TrackingSummary, List[FrameTrackingRecord]]:
        """Run ByteTrack tracking by ingesting pre-computed YOLO batch detection results.

        Guarantees zero duplicate inference passes.

        Args:
            detection_file: Path to `detection_results.json` from a previous batch run.
            output_dir: Destination directory for tracking artifacts.
            max_frames: Optional ceiling on processed frames (for smoke testing).
            start_frame: 0-based or 1-based start index.
            end_frame: Optional ending frame index.
            overwrite: If True, allows overwriting existing destination folder.

        Returns:
            Tuple of (TrackingSummary, List[FrameTrackingRecord]).
        """
        start_pipeline_t = time.perf_counter()
        det_path = Path(detection_file)
        if not det_path.is_absolute():
            det_path = (self.project_root / det_path).resolve()

        if not det_path.exists() or not det_path.is_file():
            raise TrackingInputNotFoundError(f"Detection results file not found at: {det_path}")

        try:
            with open(det_path, "r", encoding="utf-8") as f:
                batch_data = json.load(f)
        except Exception as e:
            raise TrackingPipelineError(f"Failed to read detection JSON at {det_path}: {e}") from e

        # Extract source metadata
        dataset_id = batch_data.get("dataset_id", self.dataset_id)
        model_name = batch_data.get("model_name", "yolo11n")
        source_run_id = batch_data.get("run_id", "unknown_detection_run")
        raw_frame_results = batch_data.get("frame_results", [])

        if not raw_frame_results:
            raise TrackingPipelineError(f"No 'frame_results' found in detection results: {det_path}")

        # Deterministic natural sorting by filename
        sorted_frames = sorted(
            raw_frame_results,
            key=lambda x: _natural_sort_key(x.get("image_filename", x.get("frame_id", ""))),
        )
        total_discovered_frames = len(sorted_frames)

        # Apply slicing
        s_idx = start_frame if start_frame is not None else 0
        e_idx = end_frame if end_frame is not None else total_discovered_frames
        sliced_frames = sorted_frames[s_idx:e_idx]

        if max_frames is not None and max_frames > 0:
            sliced_frames = sliced_frames[:max_frames]

        is_smoke_test = len(sliced_frames) < total_discovered_frames

        # Generate run ID and resolve output path
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"track_{timestamp_slug}_{os.urandom(3).hex()}"

        if output_dir is not None:
            resolved_out_dir = Path(output_dir).resolve()
        else:
            resolved_out_dir = (
                self.project_root / "datasets" / "processed" / dataset_id / "tracking" / run_id
            ).resolve()

        if resolved_out_dir.exists() and not overwrite:
            raise TrackingOutputExistsError(
                f"Tracking output directory already exists: {resolved_out_dir}. Use overwrite=True to replace."
            )

        # Reset engine state to guarantee session isolation
        self.engine.reset()

        frame_records: List[FrameTrackingRecord] = []
        tracking_errors: List[Dict[str, Any]] = []
        tracking_latencies: List[float] = []
        total_detections_count = 0
        total_track_instances_count = 0
        frames_with_detections = 0
        frames_with_tracks = 0
        successful_frames_count = 0
        failed_frames_count = 0

        # Sequential processing loop
        for frame_item in sliced_frames:
            f_start_t = time.perf_counter()
            filename = frame_item.get("image_filename", frame_item.get("frame_id", f"frame_{len(frame_records)+1:06d}.jpg"))
            extracted_id = frame_item.get("extracted_frame_id")
            source_idx = frame_item.get("source_frame_index")
            timestamp_sec = frame_item.get("timestamp_seconds")
            w = int(frame_item.get("image_width", 640))
            h = int(frame_item.get("image_height", 360))
            inf_dur = frame_item.get("inference_time_ms")
            dets = frame_item.get("detections", [])
            num_dets = len(dets)
            total_detections_count += num_dets
            if num_dets > 0:
                frames_with_detections += 1

            try:
                t0 = time.perf_counter()
                tracks = self.engine.update(detections=dets, frame_id=filename)
                track_duration_ms = (time.perf_counter() - t0) * 1000.0
                tracking_latencies.append(track_duration_ms)

                num_tracks = len(tracks)
                total_track_instances_count += num_tracks
                if num_tracks > 0:
                    frames_with_tracks += 1

                total_proc_ms = (time.perf_counter() - f_start_t) * 1000.0
                successful_frames_count += 1

                rec = FrameTrackingRecord(
                    schema_version="1.0",
                    dataset_id=dataset_id,
                    frame_id=filename,
                    image_filename=filename,
                    extracted_frame_id=extracted_id,
                    source_frame_index=source_idx,
                    timestamp_seconds=timestamp_sec,
                    image_width=w,
                    image_height=h,
                    processing_status="success",
                    detection_count=num_dets,
                    active_track_count=num_tracks,
                    tracks=tracks,
                    inference_duration_ms=inf_dur,
                    tracking_duration_ms=track_duration_ms,
                    total_processing_duration_ms=total_proc_ms,
                )
                frame_records.append(rec)

            except Exception as e:
                failed_frames_count += 1
                logger.error("Tracking failure on %s: %s", filename, e)
                tracking_errors.append({
                    "frame_id": filename,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                })
                total_proc_ms = (time.perf_counter() - f_start_t) * 1000.0
                rec = FrameTrackingRecord(
                    schema_version="1.0",
                    dataset_id=dataset_id,
                    frame_id=filename,
                    image_filename=filename,
                    extracted_frame_id=extracted_id,
                    source_frame_index=source_idx,
                    timestamp_seconds=timestamp_sec,
                    image_width=w,
                    image_height=h,
                    processing_status="error",
                    detection_count=num_dets,
                    active_track_count=0,
                    tracks=[],
                    inference_duration_ms=inf_dur,
                    tracking_duration_ms=0.0,
                    total_processing_duration_ms=total_proc_ms,
                )
                frame_records.append(rec)

        # Compute summary statistics
        total_time_sec = time.perf_counter() - start_pipeline_t
        mean_tracking_ms = float(np.mean(tracking_latencies)) if tracking_latencies else 0.0
        avg_tracks = (
            total_track_instances_count / successful_frames_count
            if successful_frames_count > 0
            else 0.0
        )

        # Operational status protocol
        if failed_frames_count > 0:
            run_status = "WARNING" if successful_frames_count > 0 else "FAIL"
        elif is_smoke_test:
            run_status = "WARNING"
        else:
            run_status = "PASS"

        summary = TrackingSummary(
            schema_version="1.0",
            run_id=run_id,
            dataset_id=dataset_id,
            model_name=model_name,
            tracker_name="bytetrack",
            total_frames=len(sliced_frames),
            successful_frames=successful_frames_count,
            failed_frames=failed_frames_count,
            frames_with_detections=frames_with_detections,
            frames_with_tracks=frames_with_tracks,
            total_detections=total_detections_count,
            total_track_instances=total_track_instances_count,
            unique_track_count=self.engine.unique_track_count,
            average_tracks_per_frame=round(avg_tracks, 2),
            mean_tracking_latency_ms=round(mean_tracking_ms, 2),
            total_pipeline_time_seconds=round(total_time_sec, 2),
            status=run_status,
        )

        # Persist multi-file versioned output structure (Schema v1.0)
        self._persist_artifacts(
            output_dir=resolved_out_dir,
            summary=summary,
            frame_records=frame_records,
            source_file=str(det_path),
            source_run_id=source_run_id,
            tracking_errors=tracking_errors,
        )

        logger.info(
            "Tracking run %s completed: %d frames, %d tracks, %d unique IDs (status: %s)",
            run_id,
            len(frame_records),
            total_track_instances_count,
            self.engine.unique_track_count,
            run_status,
        )
        return summary, frame_records

    # --------------------------------------------------------------------------
    # Multi-File Output Persistence
    # --------------------------------------------------------------------------

    def _persist_artifacts(
        self,
        output_dir: Path,
        summary: TrackingSummary,
        frame_records: List[FrameTrackingRecord],
        source_file: str,
        source_run_id: str,
        tracking_errors: List[Dict[str, Any]],
    ) -> None:
        """Write tracking_results, tracking_summary, tracking_metadata, and tracking_errors."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. tracking_results.json
        results_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": summary.model_name,
            "tracker_name": summary.tracker_name,
            "source_run_id": source_run_id,
            "source_detection_file": source_file,
            "total_frames": summary.total_frames,
            "unique_track_count": summary.unique_track_count,
            "frame_results": [r.to_dict() for r in frame_records],
        }
        _atomic_write_json(output_dir / "tracking_results.json", results_payload)

        # 2. tracking_summary.json
        _atomic_write_json(output_dir / "tracking_summary.json", summary.to_dict())

        # 3. tracking_metadata.json
        metadata_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tracker": {
                "algorithm": "ByteTrack",
                "parameters": self.config.to_dict(),
            },
            "source_detection_run": source_run_id,
            "source_file": source_file,
            "environment": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
                "cpu_count": os.cpu_count() or 1,
            },
        }
        _atomic_write_json(output_dir / "tracking_metadata.json", metadata_payload)

        # 4. tracking_errors.json (if errors occurred)
        if tracking_errors:
            errors_payload = {
                "schema_version": "1.0",
                "run_id": summary.run_id,
                "error_count": len(tracking_errors),
                "errors": tracking_errors,
            }
            _atomic_write_json(output_dir / "tracking_errors.json", errors_payload)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line interface for multi-object tracking pipeline."""
    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: ByteTrack Multi-Object Tracking Pipeline"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to detection_results.json from previous batch detection run",
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
        help="Optional path to custom tracker YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Optional destination directory for tracking artifacts",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=None,
        help="Maximum frames to process (useful for rapid smoke testing)",
    )
    parser.add_argument(
        "--start-frame",
        "-s",
        type=int,
        default=None,
        help="Start frame index for sliced processing",
    )
    parser.add_argument(
        "--end-frame",
        "-e",
        type=int,
        default=None,
        help="End frame index for sliced processing",
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
        pipeline = TrackingPipeline(
            config=args.config,
            dataset_id=args.dataset,
        )

        # Resolve detection input path if not supplied
        if args.input is None:
            # Look for standard Task 4 batch run
            default_det_dir = (
                _project_root
                / "datasets"
                / "processed"
                / args.dataset
                / "detections"
                / "batch_runs"
            )
            if default_det_dir.exists():
                runs = sorted([p for p in default_det_dir.iterdir() if p.is_dir()])
                if runs:
                    latest_run = runs[-1]
                    target_input = latest_run / "detection_results.json"
                else:
                    raise TrackingInputNotFoundError(
                        f"No previous batch runs found in: {default_det_dir}"
                    )
            else:
                raise TrackingInputNotFoundError(
                    f"Batch detections directory not found at: {default_det_dir}"
                )
        else:
            target_input = Path(args.input)

        print(f"[INFO] Initializing ByteTrack tracking from: '{target_input}'...")
        summary, _ = pipeline.run_from_detection_results(
            detection_file=target_input,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overwrite=args.overwrite,
        )

        print("-" * 65)
        print("PERCEPTGUARD AI — BYTETRACK TRACKING EXECUTION COMPLETE")
        print("-" * 65)
        print(f"  Run Identifier           : {summary.run_id}")
        print(f"  Dataset Identifier       : {summary.dataset_id}")
        print(f"  Processed Frames         : {summary.total_frames} (Success: {summary.successful_frames}, Fail: {summary.failed_frames})")
        print(f"  Frames with Detections   : {summary.frames_with_detections}")
        print(f"  Frames with Tracks       : {summary.frames_with_tracks}")
        print(f"  Total Detections Ingested: {summary.total_detections}")
        print(f"  Total Track Instances    : {summary.total_track_instances}")
        print(f"  Unique Persistent Tracks : {summary.unique_track_count}")
        print(f"  Average Tracks / Frame   : {summary.average_tracks_per_frame:.2f}")
        print(f"  Mean Tracking Latency    : {summary.mean_tracking_latency_ms:.2f} ms")
        print(f"  Total Pipeline Duration  : {summary.total_pipeline_time_seconds:.2f} s")
        print(f"  Execution Status         : [{summary.status}]")
        print("-" * 65)
        return 0

    except TrackingInputNotFoundError as e:
        print(f"[ERROR] Input not found: {e}", file=sys.stderr)
        return 2
    except TrackingOutputExistsError as e:
        print(f"[ERROR] Output directory conflict: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[FATAL] Tracking execution failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
