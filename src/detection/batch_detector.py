"""PerceptGuard-AI: Advanced Batch Object Detection Engine.

Senior Staff-level, production-grade batch inference and video analytics engine
for autonomous-driving camera frame sequences.

Key Guarantees:
- Deterministic dataset discovery and natural chronological ordering.
- Full metadata traceability linking detections to Day 2 extracted_frame_id,
  source_frame_index, and physical timestamps.
- Single-pass model initialization; zero redundant deserialization across frames.
- Sequential, memory-bounded CPU inference with immediate array deallocation.
- Isolated frame error containment (fault in frame N does not crash frame N+1).
- Versioned (v1.0), atomic multi-file output generation:
  1. detection_results.json (Per-frame bounding boxes & confidences)
  2. detection_summary.json (Aggregated operational statistics & class frequencies)
  3. detection_errors.json (Exhaustive diagnostic error catalog)
  4. run_metadata.json (System, hardware, and environment telemetry)
- Distinguishes full dataset completion (PASS) from smoke/partial runs (WARNING).
"""

from copy import deepcopy
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import torch
import ultralytics

from src.detection.single_image_detector import (
    DetectedObject,
    DetectionInferenceError,
    DetectionSerializationError,
    ImageDecodeError,
    ImageNotFoundError,
    ImageValidationError,
    SingleImageDetectionError,
    SingleImageDetector,
    UnsupportedImageFormatError,
)
from src.detection.yolo_config import YOLOConfig, load_yolo_config
from src.detection.yolo_model_loader import (
    YOLOInferenceError,
    YOLOModelInfo,
    YOLOModelLoadError,
    YOLOModelLoader,
    YOLOModelNotFoundError,
)

logger = logging.getLogger("PerceptGuard.BatchDetector")


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class BatchDetectionError(Exception):
    """Base exception for all batch detection pipeline failures."""
    pass


class DatasetNotFoundError(BatchDetectionError):
    """Raised when the requested dataset directory does not exist on disk."""
    pass


class DatasetEmptyError(BatchDetectionError):
    """Raised when the dataset directory contains no discoverable image frames."""
    pass


class MetadataParsingError(BatchDetectionError):
    """Raised when Day 2 dataset metadata cannot be read or parsed."""
    pass


class BatchExecutionError(BatchDetectionError):
    """Raised when a fatal, non-recoverable batch error halts execution."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass
class FrameDetectionRecord:
    """Detection results and associated video metadata for a single frame."""
    image_filename: str
    image_path: str
    extracted_frame_id: Optional[int] = None
    source_frame_index: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    image_width: int = 0
    image_height: int = 0
    processing_status: str = "success"
    inference_time_ms: float = 0.0
    detection_count: int = 0
    detections: List[DetectedObject] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary matching versioned schema."""
        return {
            "image_filename": self.image_filename,
            "image_path": self.image_path,
            "extracted_frame_id": self.extracted_frame_id,
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": round(float(self.timestamp_seconds), 4) if self.timestamp_seconds is not None else None,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "processing_status": self.processing_status,
            "inference_time_ms": round(float(self.inference_time_ms), 2),
            "detection_count": int(self.detection_count),
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class BatchErrorRecord:
    """Diagnostic details for an individual frame processing failure."""
    frame_path: str
    frame_id: Optional[int] = None
    source_frame_index: Optional[int] = None
    error_category: str = "UnknownError"
    error_message: str = ""
    failure_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_path": self.frame_path,
            "frame_id": self.frame_id,
            "source_frame_index": self.source_frame_index,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "failure_timestamp": self.failure_timestamp,
        }


@dataclass
class BatchSummary:
    """Exhaustive statistical and operational summary of a batch detection run."""
    schema_version: str = "1.0"
    run_id: str = ""
    dataset_id: str = ""
    total_frames_discovered: int = 0
    total_frames_attempted: int = 0
    successful_frames: int = 0
    failed_frames: int = 0
    frames_with_detections: int = 0
    frames_without_detections: int = 0
    total_detections: int = 0
    average_detections_per_successful_frame: float = 0.0
    average_inference_time_ms: float = 0.0
    median_inference_time_ms: float = 0.0
    total_processing_time_seconds: float = 0.0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    metadata_association_count: int = 0
    missing_metadata_count: int = 0
    model_name: str = "yolo11n"
    device: str = "cpu"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_partial_run: bool = False
    final_status: str = "PASS"  # PASS, WARNING, FAIL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "total_frames_discovered": int(self.total_frames_discovered),
            "total_frames_attempted": int(self.total_frames_attempted),
            "successful_frames": int(self.successful_frames),
            "failed_frames": int(self.failed_frames),
            "frames_with_detections": int(self.frames_with_detections),
            "frames_without_detections": int(self.frames_without_detections),
            "total_detections": int(self.total_detections),
            "average_detections_per_successful_frame": round(float(self.average_detections_per_successful_frame), 2),
            "average_inference_time_ms": round(float(self.average_inference_time_ms), 2),
            "median_inference_time_ms": round(float(self.median_inference_time_ms), 2),
            "total_processing_time_seconds": round(float(self.total_processing_time_seconds), 2),
            "class_distribution": dict(self.class_distribution),
            "metadata_association_count": int(self.metadata_association_count),
            "missing_metadata_count": int(self.missing_metadata_count),
            "model_name": self.model_name,
            "device": self.device,
            "generated_at": self.generated_at,
            "is_partial_run": bool(self.is_partial_run),
            "final_status": self.final_status,
        }


# ==============================================================================
# Batch Object Detection Engine
# ==============================================================================

class BatchDetector:
    """Production-grade batch object detection engine for PerceptGuard AI.

    Sequentially processes driving sequences, maps Day 2 temporal metadata,
    re-uses model instances on CPU, and writes versioned analytics outputs.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        detector: Optional[SingleImageDetector] = None,
        loader: Optional[YOLOModelLoader] = None,
        config: Optional[Union[YOLOConfig, str, Path]] = None,
        preset: Optional[str] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize BatchDetector with injected or newly configured components.

        Args:
            detector: Pre-initialized SingleImageDetector instance.
            loader: Pre-initialized YOLOModelLoader instance.
            config: Configuration instance or path to yolo_config.yaml.
            preset: Operational perception preset name.
            project_root: Base repository root directory.
        """
        self.project_root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).resolve().parents[2]
        )

        if detector is not None:
            self.detector = detector
        else:
            self.detector = SingleImageDetector(
                loader=loader,
                config=config,
                preset=preset,
                project_root=self.project_root,
            )

        self.loader = self.detector.loader
        self.config = self.loader.config

    @staticmethod
    def _natural_sort_key(path: Path) -> List[Union[int, str]]:
        """Natural alphanumeric sort key for chronological frame ordering."""
        parts = re.split(r"(\d+)", path.name)
        return [int(text) if text.isdigit() else text.lower() for text in parts]

    def discover_frames(
        self,
        dataset_name_or_dir: Union[str, Path],
    ) -> Tuple[str, Path, List[Path]]:
        """Discover valid preprocessed driving frames within target dataset.

        Args:
            dataset_name_or_dir: Dataset identifier (e.g. 'driving_sample') or exact directory.

        Returns:
            Tuple[str, Path, List[Path]]: (dataset_id, directory_path, sorted_frame_paths)

        Raises:
            DatasetNotFoundError: If dataset directory does not exist.
            DatasetEmptyError: If directory contains zero supported image files.
        """
        target = Path(dataset_name_or_dir)

        # 1. Resolve path
        dataset_dir: Path
        dataset_id: str
        if target.is_absolute() and target.exists():
            dataset_dir = target
            dataset_id = target.parent.name if target.name in ("preprocessed_frames", "frames") else target.name
        else:
            # Check standard preprocessed path: datasets/processed/<name>/preprocessed_frames/
            candidate = self.project_root / "datasets" / "processed" / target / "preprocessed_frames"
            if candidate.is_dir():
                dataset_dir = candidate
                dataset_id = str(target)
            elif (self.project_root / target).is_dir():
                dataset_dir = (self.project_root / target).resolve()
                dataset_id = dataset_dir.parent.name if dataset_dir.name in ("preprocessed_frames", "frames") else dataset_dir.name
            elif target.is_dir():
                dataset_dir = target.resolve()
                dataset_id = dataset_dir.parent.name if dataset_dir.name in ("preprocessed_frames", "frames") else dataset_dir.name
            else:
                raise DatasetNotFoundError(
                    f"Dataset directory not found for '{dataset_name_or_dir}'. "
                    f"Checked paths: '{candidate}' and '{target}'."
                )

        if not dataset_dir.is_dir():
            raise DatasetNotFoundError(f"Resolved dataset path is not a directory: '{dataset_dir}'")

        # 2. Discover supported image files
        frame_paths = [
            p for p in dataset_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not frame_paths:
            raise DatasetEmptyError(
                f"No supported image files found in '{dataset_dir}'. "
                f"Expected extensions: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        # 3. Deterministic natural sort
        frame_paths.sort(key=self._natural_sort_key)
        return dataset_id, dataset_dir, frame_paths

    def load_metadata_catalog(
        self,
        dataset_dir: Path,
    ) -> Dict[str, Dict[str, Any]]:
        """Load Day 2 metadata.csv mapping frame filenames to temporal metadata.

        Args:
            dataset_dir: Preprocessed frames directory or dataset root.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of filename -> metadata record.
        """
        # Search candidate metadata paths
        candidates = [
            dataset_dir.parent / "metadata.csv",
            dataset_dir / "metadata.csv",
            self.project_root / "datasets" / "processed" / dataset_dir.parent.name / "metadata.csv",
        ]

        metadata_file: Optional[Path] = None
        for c in candidates:
            if c.is_file():
                metadata_file = c
                break

        if metadata_file is None:
            logger.warning("No metadata.csv found for dataset in '%s'. Proceeding without external metadata.", dataset_dir)
            return {}

        catalog: Dict[str, Dict[str, Any]] = {}
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Match by both preprocessed and original image filename
                    pre_fn = row.get("preprocessed_image_filename")
                    orig_fn = row.get("original_image_filename")

                    meta_dict = {
                        "dataset_id": row.get("dataset_id"),
                        "extracted_frame_id": int(row["extracted_frame_id"]) if row.get("extracted_frame_id") and row["extracted_frame_id"].isdigit() else None,
                        "source_frame_index": int(row["source_frame_index"]) if row.get("source_frame_index") and row["source_frame_index"].isdigit() else None,
                        "timestamp_seconds": float(row["timestamp_seconds"]) if row.get("timestamp_seconds") else None,
                        "original_image_filename": orig_fn,
                        "original_image_path": row.get("original_image_path"),
                        "preprocessed_image_filename": pre_fn,
                    }

                    if pre_fn:
                        catalog[pre_fn] = meta_dict
                    if orig_fn and orig_fn not in catalog:
                        catalog[orig_fn] = meta_dict

        except Exception as e:
            logger.warning("Error reading metadata.csv at '%s': %s. Continuing with partial metadata.", metadata_file, e)

        return catalog

    def process_batch(
        self,
        dataset: Union[str, Path] = "driving_sample",
        max_frames: Optional[int] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[BatchSummary, List[FrameDetectionRecord], List[BatchErrorRecord], Path]:
        """Execute sequential batch object detection across discovered frames.

        Args:
            dataset: Dataset identifier or path.
            max_frames: Optional upper limit on processed frames.
            start_index: Optional starting frame index (0-based).
            end_index: Optional ending frame index (exclusive).
            output_dir: Base directory to store batch output runs.
            overwrite: Whether to overwrite existing run directory.
            conf: Confidence threshold override.
            iou: IoU threshold override.
            imgsz: Image dimension override.
            classes: Class filter override.
            run_id: Explicit run identifier.

        Returns:
            Tuple[BatchSummary, List[FrameDetectionRecord], List[BatchErrorRecord], Path]:
                (summary, frame_records, error_records, run_output_directory)

        Raises:
            BatchExecutionError: If model initialization or fatal system errors occur.
        """
        t_batch_start = time.perf_counter()
        active_run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # 1. Discover Frames & Dataset Identity
        dataset_id, dataset_dir, all_frames = self.discover_frames(dataset)
        total_discovered = len(all_frames)

        # 2. Slice Frames according to constraints
        start_idx = start_index if start_index is not None else 0
        end_idx = end_index if end_index is not None else total_discovered

        if start_idx < 0:
            raise BatchExecutionError(f"start_index cannot be negative, got: {start_idx}")
        if end_idx < start_idx:
            raise BatchExecutionError(f"end_index ({end_idx}) cannot be less than start_index ({start_idx})")

        selected_frames = all_frames[start_idx:end_idx]
        if max_frames is not None and max_frames > 0:
            selected_frames = selected_frames[:max_frames]

        total_attempted = len(selected_frames)
        is_partial_run = total_attempted < total_discovered

        # 3. Load Metadata Traceability Catalog
        metadata_catalog = self.load_metadata_catalog(dataset_dir)

        # 4. Resolve Output Directory
        if output_dir is not None:
            base_out = Path(output_dir)
        else:
            base_out = (
                self.project_root
                / "datasets"
                / "processed"
                / dataset_id
                / "detections"
                / "batch_runs"
            )

        run_dir = (base_out / active_run_id).resolve()
        if run_dir.exists() and not overwrite:
            raise BatchExecutionError(
                f"Batch run output directory already exists at '{run_dir}'. Set overwrite=True to allow replacement."
            )
        run_dir.mkdir(parents=True, exist_ok=True)

        # 5. Initialize Model ONCE
        t_model_load_0 = time.perf_counter()
        try:
            self.loader.load_model()
        except Exception as e:
            raise BatchExecutionError(f"Failed to initialize YOLO model on {self.loader.device}: {e}") from e
        model_load_duration_ms = (time.perf_counter() - t_model_load_0) * 1000.0
        logger.info("YOLO Model initialized in %.2f ms on device %s", model_load_duration_ms, self.loader.device)

        # 6. Sequential Processing Loop
        frame_records: List[FrameDetectionRecord] = []
        error_records: List[BatchErrorRecord] = []
        inference_times_ms: List[float] = []
        class_distribution: Dict[str, int] = {}
        successful_frames_count = 0
        failed_frames_count = 0
        frames_with_detections_count = 0
        frames_without_detections_count = 0
        total_detections_count = 0
        metadata_assoc_count = 0
        missing_metadata_count = 0

        for frame_path in selected_frames:
            meta = metadata_catalog.get(frame_path.name, {})
            ext_id = meta.get("extracted_frame_id")
            src_idx = meta.get("source_frame_index")
            ts = meta.get("timestamp_seconds")

            if meta:
                metadata_assoc_count += 1
            else:
                missing_metadata_count += 1

            try:
                # Single-frame inference
                single_res = self.detector.detect(
                    image_path=frame_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=classes,
                    save_json=False,  # Batch handles central persistence
                )

                det_count = single_res.detection_count
                total_detections_count += det_count
                successful_frames_count += 1
                inference_times_ms.append(single_res.inference_time_ms)

                if det_count > 0:
                    frames_with_detections_count += 1
                    for d in single_res.detections:
                        class_distribution[d.class_name] = class_distribution.get(d.class_name, 0) + 1
                else:
                    frames_without_detections_count += 1

                record = FrameDetectionRecord(
                    image_filename=frame_path.name,
                    image_path=str(frame_path),
                    extracted_frame_id=ext_id,
                    source_frame_index=src_idx,
                    timestamp_seconds=ts,
                    image_width=single_res.width,
                    image_height=single_res.height,
                    processing_status="success",
                    inference_time_ms=single_res.inference_time_ms,
                    detection_count=det_count,
                    detections=single_res.detections,
                )
                frame_records.append(record)

            except Exception as e:
                failed_frames_count += 1
                err_category = type(e).__name__
                err_record = BatchErrorRecord(
                    frame_path=str(frame_path),
                    frame_id=ext_id,
                    source_frame_index=src_idx,
                    error_category=err_category,
                    error_message=str(e),
                )
                error_records.append(err_record)
                logger.error("Error processing frame '%s': %s", frame_path.name, e)

                # Record failed frame entry
                failed_frame_record = FrameDetectionRecord(
                    image_filename=frame_path.name,
                    image_path=str(frame_path),
                    extracted_frame_id=ext_id,
                    source_frame_index=src_idx,
                    timestamp_seconds=ts,
                    image_width=0,
                    image_height=0,
                    processing_status="failed",
                    inference_time_ms=0.0,
                    detection_count=0,
                    detections=[],
                )
                frame_records.append(failed_frame_record)

        total_duration_sec = time.perf_counter() - t_batch_start

        # 7. Aggregate Statistics Safely (avoid division by zero)
        avg_det_per_frame = (
            total_detections_count / successful_frames_count
            if successful_frames_count > 0 else 0.0
        )
        avg_infer_time = (
            statistics.mean(inference_times_ms)
            if inference_times_ms else 0.0
        )
        med_infer_time = (
            statistics.median(inference_times_ms)
            if inference_times_ms else 0.0
        )

        # Status determination hierarchy
        final_status: str
        if failed_frames_count > 0:
            final_status = "WARNING" if successful_frames_count > 0 else "FAIL"
        elif is_partial_run:
            final_status = "WARNING"
        else:
            final_status = "PASS"

        summary = BatchSummary(
            schema_version="1.0",
            run_id=active_run_id,
            dataset_id=dataset_id,
            total_frames_discovered=total_discovered,
            total_frames_attempted=total_attempted,
            successful_frames=successful_frames_count,
            failed_frames=failed_frames_count,
            frames_with_detections=frames_with_detections_count,
            frames_without_detections=frames_without_detections_count,
            total_detections=total_detections_count,
            average_detections_per_successful_frame=avg_det_per_frame,
            average_inference_time_ms=avg_infer_time,
            median_inference_time_ms=med_infer_time,
            total_processing_time_seconds=total_duration_sec,
            class_distribution=class_distribution,
            metadata_association_count=metadata_assoc_count,
            missing_metadata_count=missing_metadata_count,
            model_name=self.loader.model_name,
            device=self.loader.device,
            generated_at=datetime.now(timezone.utc).isoformat(),
            is_partial_run=is_partial_run,
            final_status=final_status,
        )

        # 8. Multi-File Atomic Output Serialization
        self._save_outputs(
            run_dir=run_dir,
            summary=summary,
            frame_records=frame_records,
            error_records=error_records,
            dataset_id=dataset_id,
            model_load_duration_ms=model_load_duration_ms,
        )

        return summary, frame_records, error_records, run_dir

    def _save_outputs(
        self,
        run_dir: Path,
        summary: BatchSummary,
        frame_records: List[FrameDetectionRecord],
        error_records: List[BatchErrorRecord],
        dataset_id: str,
        model_load_duration_ms: float,
    ) -> None:
        """Atomically persist 4 versioned detection files to run directory."""
        # File 1: detection_results.json
        results_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": dataset_id,
            "generated_at": summary.generated_at,
            "model_name": self.loader.model_name,
            "model_weights_reference": str(self.loader.weights_path),
            "effective_configuration": self.config.to_dict(),
            "total_frames": len(frame_records),
            "frame_results": [r.to_dict() for r in frame_records],
        }
        self._atomic_write_json(run_dir / "detection_results.json", results_payload)

        # File 2: detection_summary.json
        self._atomic_write_json(run_dir / "detection_summary.json", summary.to_dict())

        # File 3: detection_errors.json
        errors_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "error_count": len(error_records),
            "errors": [e.to_dict() for e in error_records],
        }
        self._atomic_write_json(run_dir / "detection_errors.json", errors_payload)

        # File 4: run_metadata.json
        metadata_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": dataset_id,
            "model_name": self.loader.model_name,
            "execution_device": self.loader.device,
            "model_load_duration_ms": round(model_load_duration_ms, 2),
            "software_versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "opencv": cv2.__version__,
            },
            "environment": {
                "os": platform.platform(),
                "processor": platform.processor(),
                "cuda_available": torch.cuda.is_available(),
            },
            "effective_configuration": self.config.to_dict(),
            "generated_at": summary.generated_at,
        }
        self._atomic_write_json(run_dir / "run_metadata.json", metadata_payload)

    @staticmethod
    def _atomic_write_json(path: Path, data: Any) -> None:
        """Safely write JSON payload to disk using an atomic rename operation."""
        temp_file = path.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(path)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise DetectionSerializationError(f"Failed to atomically write '{path}': {e}") from e


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    """CLI runner for Batch Object Detection Engine."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI – Batch Object Detection Engine"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="driving_sample",
        help="Dataset name under datasets/processed/ or explicit directory.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only first N frames (useful for smoke tests).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start frame index (0-based).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End frame index (exclusive).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store batch runs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to yolo_config.yaml.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["balanced", "high_precision", "high_recall", "cpu_lightweight"],
        help="Optional perception preset to apply.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination run directory if it exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging output.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        engine = BatchDetector(
            config=args.config,
            preset=args.preset,
        )
        summary, _, _, run_dir = engine.process_batch(
            dataset=args.dataset,
            max_frames=args.max_frames,
            start_index=args.start_index,
            end_index=args.end_index,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print("=" * 60)
        print("PERCEPTGUARD AI – BATCH OBJECT DETECTION")
        print("=" * 60)
        print(f"ERROR: {e}")
        print("Final status: FAIL")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print("PERCEPTGUARD AI – BATCH OBJECT DETECTION")
    print("=" * 60)
    print(f"Dataset: {summary.dataset_id}")
    print(f"Run ID: {summary.run_id}")
    print(f"Model: {summary.model_name}")
    print(f"Device: {summary.device}")
    print(f"Output directory: {run_dir}")
    print("")
    print(f"Frames discovered: {summary.total_frames_discovered}")
    print(f"Frames attempted: {summary.total_frames_attempted}")
    print(f"Successful frames: {summary.successful_frames}")
    print(f"Failed frames: {summary.failed_frames}")
    print(f"Frames with detections: {summary.frames_with_detections}")
    print(f"Frames without detections: {summary.frames_without_detections}")
    print(f"Total detections: {summary.total_detections}")
    print("")
    print(f"Average detections/frame: {summary.average_detections_per_successful_frame:.2f}")
    print(f"Average inference time: {summary.average_inference_time_ms:.2f} ms")
    print(f"Median inference time: {summary.median_inference_time_ms:.2f} ms")
    print(f"Total processing time: {summary.total_processing_time_seconds:.2f} s")
    print("")
    print("Output files:")
    print(f"  - {run_dir / 'detection_results.json'}")
    print(f"  - {run_dir / 'detection_summary.json'}")
    print(f"  - {run_dir / 'detection_errors.json'}")
    print(f"  - {run_dir / 'run_metadata.json'}")
    print(f"Final status: {summary.final_status}")
    print("=" * 60)

    sys.exit(0 if summary.final_status in ("PASS", "WARNING") else 1)


if __name__ == "__main__":
    main()

