"""PerceptGuard-AI: Ground-Truth Annotation Processing Pipeline.

Production-grade, modular, deterministic pipeline for ingesting, parsing,
normalizing, validating, and frame-aligning ground-truth object detection
annotations from BDD100K Multiple Object Tracking (MOT) sequences.

Key Guarantees:
- Normalized, versioned canonical annotation schema (Schema v1.0.0).
- Pure mathematical coordinate normalization onto 640x360 canvas.
- Complete preservation of ground-truth object IDs (tracking IDs) when provided.
- Preserves original taxonomy and reports unmapped classes.
- Atomic output generation (JSONL and JSON summary) with strictly relative paths.
- Robust handling of empty frames, missing images, and corrupted records.
- Memory-bounded processing suitable for 8 GB CPU environments.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

ANNOTATION_SCHEMA_VERSION = "1.0.0"

# ==============================================================================
# ENUMS & STATUS CODES
# ==============================================================================

class ValidationStatus(str, Enum):
    """Validation outcome for an individual annotation record."""
    VALID = "VALID"
    INVALID_COORDINATES = "INVALID_COORDINATES"
    INVALID_DIMENSIONS = "INVALID_DIMENSIONS"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    UNKNOWN_CLASS = "UNKNOWN_CLASS"
    NON_FINITE = "NON_FINITE"
    COLLAPSED_BOX = "COLLAPSED_BOX"
    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"


class FrameStatus(str, Enum):
    """Semantic status of a video frame in the ground-truth pipeline."""
    VALID_WITH_OBJECTS = "VALID_WITH_OBJECTS"
    VALID_EMPTY = "VALID_EMPTY"
    MISSING_IMAGE = "MISSING_IMAGE"
    CORRUPTED_ANNOTATION = "CORRUPTED_ANNOTATION"
    PARTIAL_ANNOTATION = "PARTIAL_ANNOTATION"
    UNVERIFIED = "UNVERIFIED"


# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class AnnotationProcessingError(Exception):
    """Base exception for annotation processing errors."""
    pass

class AnnotationParseError(AnnotationProcessingError):
    """Raised when annotation source data fails syntax or schema validation."""
    pass

class CoordinateValidationError(AnnotationProcessingError):
    """Raised when coordinates violate spatial boundaries or invariants."""
    pass

class FrameAlignmentException(AnnotationProcessingError):
    """Raised when video frames cannot be aligned with annotation records."""
    pass


# ==============================================================================
# CANONICAL DATA MODELS
# ==============================================================================

@dataclass
class CanonicalAnnotationRecord:
    """Canonical ground-truth annotation record (17 core fields + provenance)."""
    dataset_id: str
    sequence_id: str
    frame_id: str
    timestamp: Optional[float]
    image_path: str
    image_width: int
    image_height: int
    annotation_id: str
    ground_truth_object_id: Optional[str]
    class_id: int
    class_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    coordinate_format: str = "x_min, y_min, x_max, y_max"
    annotation_source: str = "bdd100k_mot"
    validation_status: str = "VALID"
    # Provenance fields
    original_class_name: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = ANNOTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.image_width = int(self.image_width)
        self.image_height = int(self.image_height)
        self.class_id = int(self.class_id)
        self.x_min = float(self.x_min)
        self.y_min = float(self.y_min)
        self.x_max = float(self.x_max)
        self.y_max = float(self.y_max)
        if self.timestamp is not None:
            self.timestamp = float(self.timestamp)

    @property
    def box_width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def box_height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def box_area(self) -> float:
        return self.box_width * self.box_height

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to portable dictionary with rounded coordinates."""
        d = asdict(self)
        d["x_min"] = round(self.x_min, 2)
        d["y_min"] = round(self.y_min, 2)
        d["x_max"] = round(self.x_max, 2)
        d["y_max"] = round(self.y_max, 2)
        if self.timestamp is not None:
            d["timestamp"] = round(self.timestamp, 4)
        return d

    def to_jsonl(self) -> str:
        """Serialize single record to JSONL line."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


@dataclass
class FrameAnnotationSummary:
    """Frame-level annotation summary."""
    frame_id: str
    frame_index: int
    source_frame_index: int
    timestamp: float
    image_path: str
    image_width: int
    image_height: int
    image_exists: bool
    status: FrameStatus
    object_count: int
    annotations: List[CanonicalAnnotationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "source_frame_index": self.source_frame_index,
            "timestamp": round(self.timestamp, 4),
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_exists": self.image_exists,
            "status": self.status.value if isinstance(self.status, FrameStatus) else str(self.status),
            "object_count": self.object_count,
            "annotations": [a.to_dict() for a in self.annotations]
        }


@dataclass
class ProcessingSummary:
    """Summary metrics of annotation processing execution."""
    pipeline_version: str = ANNOTATION_SCHEMA_VERSION
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_id: str = "driving_sample"
    sequence_id: str = "026c7465-309f6d33"
    total_source_files: int = 1
    total_records_discovered: int = 0
    valid_records_count: int = 0
    invalid_records_count: int = 0
    duplicate_records_count: int = 0
    unmapped_classes_count: int = 0
    frames_aligned_count: int = 0
    frames_not_aligned_count: int = 0
    empty_frames_count: int = 0
    missing_images_count: int = 0
    unique_tracks_count: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    processing_duration_seconds: float = 0.0
    status: str = "PASS"
    error_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["processing_duration_seconds"] = round(self.processing_duration_seconds, 4)
        return d


# ==============================================================================
# CLASS MAPPER
# ==============================================================================

DEFAULT_CLASS_MAPPING: Dict[str, Tuple[str, int]] = {
    "car": ("car", 2),
    "truck": ("truck", 7),
    "bus": ("bus", 5),
    "pedestrian": ("person", 0),
    "rider": ("person", 0),
    "other person": ("person", 0),
    "motorcycle": ("motorcycle", 3),
    "bicycle": ("bicycle", 1),
    "traffic light": ("traffic light", 9),
    "traffic sign": ("traffic light", 9),
    "train": ("train", 6),
    "trailer": ("truck", 7),
    "other vehicle": ("car", 2),
}


class ClassMapper:
    """Maps dataset-specific classes to canonical COCO/YOLO ontology."""

    def __init__(self, mapping: Optional[Dict[str, Tuple[str, int]]] = None) -> None:
        self.mapping = mapping or DEFAULT_CLASS_MAPPING

    def map_class(self, raw_category: str) -> Tuple[str, int, bool]:
        """Map raw category to (canonical_name, canonical_id, is_mapped)."""
        norm_cat = str(raw_category).lower().strip()
        if norm_cat in self.mapping:
            name, cid = self.mapping[norm_cat]
            return name, cid, True
        return norm_cat, -1, False


# ==============================================================================
# COORDINATE NORMALIZER
# ==============================================================================

class CoordinateNormalizer:
    """Transforms raw BDD100K coordinates into canvas pixel space with validation."""

    def __init__(
        self,
        native_width: int = 1280,
        native_height: int = 720,
        raw_video_width: int = 720,
        raw_video_height: int = 1280,
        canvas_width: int = 640,
        canvas_height: int = 360,
        scale_factor: float = 0.28125,
        offset_x: float = 218.75,
        offset_y: float = 0.0,
        min_box_area: float = 4.0,
    ) -> None:
        self.native_width = native_width
        self.native_height = native_height
        self.raw_video_width = raw_video_width
        self.raw_video_height = raw_video_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.scale_factor = scale_factor
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.min_box_area = min_box_area

    def transform_box(
        self,
        x1_native: float,
        y1_native: float,
        x2_native: float,
        y2_native: float,
    ) -> Tuple[Optional[Tuple[float, float, float, float]], ValidationStatus]:
        """Convert native coordinates to canvas coordinates and validate."""
        # 1. Finite check
        coords = (x1_native, y1_native, x2_native, y2_native)
        if not all(math.isfinite(v) for v in coords):
            return None, ValidationStatus.NON_FINITE

        # 2. Inversion check
        if x2_native <= x1_native or y2_native <= y1_native:
            return None, ValidationStatus.INVALID_COORDINATES

        # 3. 90-degree clockwise rotation (native 1280x720 -> raw 720x1280)
        rx1 = float(self.raw_video_width) - y2_native
        rx2 = float(self.raw_video_width) - y1_native
        ry1 = x1_native
        ry2 = x2_native

        raw_x1 = min(rx1, rx2)
        raw_x2 = max(rx1, rx2)
        raw_y1 = min(ry1, ry2)
        raw_y2 = max(ry1, ry2)

        # 4. Scale and offset to letterbox canvas (640x360)
        cx1 = raw_x1 * self.scale_factor + self.offset_x
        cy1 = raw_y1 * self.scale_factor + self.offset_y
        cx2 = raw_x2 * self.scale_factor + self.offset_x
        cy2 = raw_y2 * self.scale_factor + self.offset_y

        # Clamp tiny boundary float discretization (e.g. -0.01 or 640.02)
        cx1 = max(0.0, min(float(self.canvas_width), cx1))
        cy1 = max(0.0, min(float(self.canvas_height), cy1))
        cx2 = max(0.0, min(float(self.canvas_width), cx2))
        cy2 = max(0.0, min(float(self.canvas_height), cy2))

        # Check positive dimensions and minimum area
        w = cx2 - cx1
        h = cy2 - cy1
        if w <= 0.0 or h <= 0.0:
            return None, ValidationStatus.COLLAPSED_BOX
        if (w * h) < self.min_box_area:
            return None, ValidationStatus.INVALID_DIMENSIONS

        return (cx1, cy1, cx2, cy2), ValidationStatus.VALID


# ==============================================================================
# BDD100K PARSER
# ==============================================================================

class BDD100KMOTParser:
    """Parser for BDD100K Multiple Object Tracking JSON sequence format."""

    def __init__(self, normalizer: Optional[CoordinateNormalizer] = None) -> None:
        self.normalizer = normalizer or CoordinateNormalizer()

    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Safely open and parse JSON annotation file."""
        p = Path(file_path)
        if not p.is_file():
            raise AnnotationParseError(f"Annotation file not found: '{p}'")

        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    raise AnnotationParseError(f"Annotation file is empty: '{p}'")
                data = json.loads(content)
        except json.JSONDecodeError as e:
            raise AnnotationParseError(f"Malformed JSON in '{p}': {e}") from e
        except Exception as e:
            raise AnnotationParseError(f"Error reading '{p}': {e}") from e

        if isinstance(data, list):
            if len(data) == 0:
                raise AnnotationParseError(f"JSON list in '{p}' contains zero items.")
            return data[0]
        elif isinstance(data, dict):
            return data
        else:
            raise AnnotationParseError(f"Unexpected JSON root type '{type(data).__name__}' in '{p}'")


# ==============================================================================
# ANNOTATION PROCESSOR PIPELINE
# ==============================================================================

class AnnotationProcessor:
    """Production-grade pipeline orchestrator for ground-truth annotation processing."""

    def __init__(
        self,
        dataset_id: str = "driving_sample",
        sequence_id: str = "026c7465-309f6d33",
        normalizer: Optional[CoordinateNormalizer] = None,
        class_mapper: Optional[ClassMapper] = None,
        parser: Optional[BDD100KMOTParser] = None,
        frame_name_pattern: str = "frame_{:06d}.jpg",
        fps: float = 5.0,
    ) -> None:
        self.dataset_id = dataset_id
        self.sequence_id = sequence_id
        self.normalizer = normalizer or CoordinateNormalizer()
        self.class_mapper = class_mapper or ClassMapper()
        self.parser = parser or BDD100KMOTParser(normalizer=self.normalizer)
        self.frame_name_pattern = frame_name_pattern
        self.fps = fps

    def process_sequence(
        self,
        annotation_file: Union[str, Path],
        frames_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> Tuple[List[FrameAnnotationSummary], ProcessingSummary]:
        """Execute end-to-end processing over annotation sequence."""
        start_time = time.time()
        p_root = Path(project_root) if project_root else Path.cwd()
        raw_data = self.parser.parse_file(annotation_file)

        raw_frames = raw_data.get("frames", [])
        if not isinstance(raw_frames, list) or len(raw_frames) == 0:
            raise AnnotationParseError(f"Sequence contains no 'frames' list in '{annotation_file}'")

        frames_path = Path(frames_dir) if frames_dir else None
        sorted_raw_frames = sorted(raw_frames, key=lambda f: int(f.get("frame_index", 0)))
        if max_frames is not None and max_frames > 0:
            sorted_raw_frames = sorted_raw_frames[:max_frames]

        frame_summaries: List[FrameAnnotationSummary] = []
        unique_tracks: Set[str] = set()
        class_distribution: Dict[str, int] = {}
        error_log: List[Dict[str, Any]] = []

        total_discovered = 0
        valid_records = 0
        invalid_records = 0
        unmapped_classes = 0
        duplicate_records = 0
        empty_frames = 0
        missing_images = 0

        # Discover images in frames_dir if available
        check_images = frames_path is not None and frames_path.is_dir()
        available_images: Set[str] = set()
        if check_images:
            available_images = {p.name for p in frames_path.glob("*.jpg")}

        for idx, rf in enumerate(sorted_raw_frames):
            extracted_frame_id_num = idx + 1
            frame_filename = self.frame_name_pattern.format(extracted_frame_id_num)
            f_idx = int(rf.get("frame_index", idx))
            source_frame_idx = int(rf.get("video_frame", idx * 6))
            timestamp = idx * (1.0 / self.fps)

            # Check image existence
            img_exists = (frame_filename in available_images) if check_images else True
            if not img_exists:
                missing_images += 1

            # Relative path representation
            if frames_path:
                try:
                    rel_img_path = str((frames_path / frame_filename).relative_to(p_root)).replace("\\", "/")
                except ValueError:
                    rel_img_path = str(frames_path / frame_filename).replace("\\", "/")
            else:
                rel_img_path = f"datasets/processed/driving_sample/preprocessed_frames/{frame_filename}"

            raw_labels = rf.get("labels", [])
            canonical_records: List[CanonicalAnnotationRecord] = []
            seen_in_frame: Set[Tuple[str, float, float, float, float]] = set()

            for l_idx, rl in enumerate(raw_labels):
                total_discovered += 1
                lbl_id = str(rl.get("id", f"{f_idx}_{l_idx}"))
                raw_cat = str(rl.get("category", "unknown"))

                # 1. Class mapping
                canon_name, canon_id, is_mapped = self.class_mapper.map_class(raw_cat)
                if not is_mapped:
                    unmapped_classes += 1
                    error_log.append({
                        "frame_id": frame_filename,
                        "label_id": lbl_id,
                        "reason": f"Unmapped raw category '{raw_cat}'",
                        "status": ValidationStatus.UNKNOWN_CLASS.value
                    })

                # 2. Extract bounding box
                box_dict = rl.get("box2d", {})
                if not box_dict or not all(k in box_dict for k in ("x1", "y1", "x2", "y2")):
                    invalid_records += 1
                    error_log.append({
                        "frame_id": frame_filename,
                        "label_id": lbl_id,
                        "reason": "Missing coordinate keys in box2d",
                        "status": ValidationStatus.INVALID_COORDINATES.value
                    })
                    continue

                try:
                    x1 = float(box_dict["x1"])
                    y1 = float(box_dict["y1"])
                    x2 = float(box_dict["x2"])
                    y2 = float(box_dict["y2"])
                except (ValueError, TypeError):
                    invalid_records += 1
                    error_log.append({
                        "frame_id": frame_filename,
                        "label_id": lbl_id,
                        "reason": "Non-numeric coordinates in box2d",
                        "status": ValidationStatus.NON_FINITE.value
                    })
                    continue

                # 3. Coordinate normalization and validation
                transformed, status = self.normalizer.transform_box(x1, y1, x2, y2)
                if status != ValidationStatus.VALID or transformed is None:
                    invalid_records += 1
                    error_log.append({
                        "frame_id": frame_filename,
                        "label_id": lbl_id,
                        "reason": f"Box rejected with status {status.value}",
                        "status": status.value
                    })
                    continue

                cx1, cy1, cx2, cy2 = transformed

                # Check duplicate detection in frame
                box_key = (canon_name, round(cx1, 1), round(cy1, 1), round(cx2, 1), round(cy2, 1))
                if box_key in seen_in_frame:
                    duplicate_records += 1
                    error_log.append({
                        "frame_id": frame_filename,
                        "label_id": lbl_id,
                        "reason": "Duplicate bounding box in same frame",
                        "status": "DUPLICATE"
                    })
                    continue
                seen_in_frame.add(box_key)

                # Track ID preservation
                track_id = str(rl.get("id")) if rl.get("id") is not None else None
                if track_id:
                    unique_tracks.add(track_id)

                attrs = {
                    "crowd": bool(rl.get("attributes", {}).get("crowd", False)),
                    "occluded": bool(rl.get("attributes", {}).get("occluded", False)),
                    "truncated": bool(rl.get("attributes", {}).get("truncated", False)),
                }

                rec = CanonicalAnnotationRecord(
                    dataset_id=self.dataset_id,
                    sequence_id=self.sequence_id,
                    frame_id=frame_filename,
                    timestamp=timestamp,
                    image_path=rel_img_path,
                    image_width=self.normalizer.canvas_width,
                    image_height=self.normalizer.canvas_height,
                    annotation_id=lbl_id,
                    ground_truth_object_id=track_id,
                    class_id=canon_id,
                    class_name=canon_name,
                    x_min=cx1,
                    y_min=cy1,
                    x_max=cx2,
                    y_max=cy2,
                    coordinate_format="x_min, y_min, x_max, y_max",
                    annotation_source="bdd100k_mot",
                    validation_status=ValidationStatus.VALID.value,
                    original_class_name=raw_cat,
                    attributes=attrs,
                )
                canonical_records.append(rec)
                valid_records += 1
                class_distribution[canon_name] = class_distribution.get(canon_name, 0) + 1

            # Determine frame semantic status
            if not img_exists:
                f_status = FrameStatus.MISSING_IMAGE
            elif len(canonical_records) == 0:
                f_status = FrameStatus.VALID_EMPTY
                empty_frames += 1
            else:
                f_status = FrameStatus.VALID_WITH_OBJECTS

            summary = FrameAnnotationSummary(
                frame_id=frame_filename,
                frame_index=f_idx,
                source_frame_index=source_frame_idx,
                timestamp=timestamp,
                image_path=rel_img_path,
                image_width=self.normalizer.canvas_width,
                image_height=self.normalizer.canvas_height,
                image_exists=img_exists,
                status=f_status,
                object_count=len(canonical_records),
                annotations=canonical_records,
            )
            frame_summaries.append(summary)

        duration = time.time() - start_time
        summary_metric = ProcessingSummary(
            pipeline_version=ANNOTATION_SCHEMA_VERSION,
            dataset_id=self.dataset_id,
            sequence_id=self.sequence_id,
            total_source_files=1,
            total_records_discovered=total_discovered,
            valid_records_count=valid_records,
            invalid_records_count=invalid_records,
            duplicate_records_count=duplicate_records,
            unmapped_classes_count=unmapped_classes,
            frames_aligned_count=len(frame_summaries) - missing_images,
            frames_not_aligned_count=missing_images,
            empty_frames_count=empty_frames,
            missing_images_count=missing_images,
            unique_tracks_count=len(unique_tracks),
            class_distribution=class_distribution,
            processing_duration_seconds=duration,
            status="PASS" if invalid_records == 0 or (valid_records > 0 and invalid_records < total_discovered * 0.05) else "FAIL",
            error_log=error_log[:50]  # Cap top 50 errors
        )

        return frame_summaries, summary_metric

    def export_canonical_jsonl(
        self,
        frame_summaries: List[FrameAnnotationSummary],
        output_jsonl_path: Union[str, Path],
    ) -> Path:
        """Write canonical records atomically to JSONL format."""
        out_path = Path(output_jsonl_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first for atomic commit
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="canon_ann_", suffix=".tmp", dir=str(out_path.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                for fs in frame_summaries:
                    for ann in fs.annotations:
                        f.write(ann.to_jsonl() + "\n")
            # Atomic rename / replace
            if os.path.exists(out_path):
                os.replace(tmp_path, out_path)
            else:
                os.rename(tmp_path, out_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        return out_path

    def export_summary_json(
        self,
        summary: ProcessingSummary,
        output_summary_path: Union[str, Path],
    ) -> Path:
        """Write processing summary report atomically to JSON."""
        out_path = Path(output_summary_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(prefix="ann_sum_", suffix=".tmp", dir=str(out_path.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(summary.to_dict(), f, indent=2)
            if os.path.exists(out_path):
                os.replace(tmp_path, out_path)
            else:
                os.rename(tmp_path, out_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        return out_path


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================

def main() -> None:
    """CLI execution entrypoint for Ground-Truth Annotation Processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard-AI: Ground-Truth Annotation Processing Pipeline (Day 8 Task 4)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/annotations/bdd100k/bdd100k_mot_026c7465_309f6d33.json",
        help="Path to raw BDD100K MOT sequence JSON annotation file",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="datasets/processed/driving_sample/preprocessed_frames",
        help="Directory containing preprocessed video frame images (640x360)",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="datasets/processed/driving_sample/annotations/annotations_canonical.jsonl",
        help="Output path for streaming canonical JSONL annotations",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="datasets/processed/driving_sample/annotations/annotation_processing_summary.json",
        help="Output path for JSON processing summary report",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Maximum number of frames to process (default: 200)",
    )
    parser.add_argument(
        "--bounded",
        type=int,
        default=None,
        help="Process a bounded sample of N frames for fast validation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate annotations without writing output files",
    )

    args = parser.parse_args()

    limit_frames = args.bounded if args.bounded is not None else args.max_frames

    print("=" * 80)
    print("  PERCEPTGUARD-AI: GROUND-TRUTH ANNOTATION PROCESSING PIPELINE")
    print("=" * 80)
    print(f"  Annotation Input:   {args.input}")
    print(f"  Frames Directory:   {args.frames_dir}")
    print(f"  Target Frames:      {limit_frames}")
    print(f"  Validate Only:      {args.validate_only}")
    print("-" * 80)

    processor = AnnotationProcessor()
    frame_summaries, summary = processor.process_sequence(
        annotation_file=args.input,
        frames_dir=args.frames_dir,
        max_frames=limit_frames,
    )

    print(f"  Status:             {summary.status}")
    print(f"  Frames Processed:   {len(frame_summaries)}")
    print(f"  Records Discovered: {summary.total_records_discovered}")
    print(f"  Valid Records:      {summary.valid_records_count}")
    print(f"  Invalid Records:    {summary.invalid_records_count}")
    print(f"  Unique Tracks:      {summary.unique_tracks_count}")
    print(f"  Frames Aligned:     {summary.frames_aligned_count}")
    print(f"  Empty Frames:       {summary.empty_frames_count}")
    print(f"  Missing Images:     {summary.missing_images_count}")
    print(f"  Duration:           {summary.processing_duration_seconds:.4f} seconds")
    print(f"  Class Distribution: {summary.class_distribution}")

    if not args.validate_only:
        jsonl_path = processor.export_canonical_jsonl(frame_summaries, args.output_jsonl)
        summary_path = processor.export_summary_json(summary, args.output_summary)
        print("-" * 80)
        print(f"  [OK] Exported Canonical JSONL: {jsonl_path}")
        print(f"  [OK] Exported Summary JSON:   {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()


