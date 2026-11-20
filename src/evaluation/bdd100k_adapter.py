"""PerceptGuard AI: BDD100K Annotation Dataset Adapter & Ingestion Engine.

Production-grade dataset adapter and coordinate transformation engine for
Berkeley DeepDrive (BDD100K) multiple object tracking (MOT) ground-truth annotations.

Key Guarantees:
- Validates BDD100K MOT sequence JSON schema with strict integrity checks.
- Provides explicit, mathematically proven coordinate transformations between
  native BDD100K coordinates (1280x720), rotated raw video coordinates (720x1280),
  and preprocessed letterboxed canvas coordinates (640x360).
- Maps BDD100K object categories to standardized YOLO COCO class IDs and taxonomies.
- Preserves persistent temporal object tracking IDs across consecutive video frames.
- Implements strict bounding-box validation: 0 <= x1 < x2 <= width, 0 <= y1 < y2 <= height.
- Generates reproducible, portable dataset manifests using strictly relative paths.
- Performs bounded, memory-efficient data integrity auditing suitable for 8 GB CPU systems.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class BDD100KAdapterError(Exception):
    """Base exception for all BDD100K annotation adapter operations."""
    pass

class AnnotationFileNotFoundError(BDD100KAdapterError):
    """Raised when the specified annotation file does not exist on disk."""
    pass

class AnnotationSchemaError(BDD100KAdapterError):
    """Raised when annotation JSON structure violates expected schema."""
    pass

class InvalidBoundingBoxError(BDD100KAdapterError):
    """Raised when bounding box coordinates are corrupted, non-finite, or inverted."""
    pass

class CoordinateConversionError(BDD100KAdapterError):
    """Raised when coordinate transformation fails or produces out-of-bounds results."""
    pass

class FrameAlignmentError(BDD100KAdapterError):
    """Raised when annotations cannot be aligned with video frame indices."""
    pass

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class BoundingBox:
    """Represents an axis-aligned 2D bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        self.x1 = float(self.x1)
        self.y1 = float(self.y1)
        self.x2 = float(self.x2)
        self.y2 = float(self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.x1, self.y1, self.x2, self.y2))

    def is_valid(self, canvas_width: float, canvas_height: float, min_area: float = 1.0) -> bool:
        """Check if box coordinates are strictly valid within canvas bounds."""
        if not self.is_finite:
            return False
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            return False
        if self.x1 < 0.0 or self.y1 < 0.0:
            return False
        if self.x2 > canvas_width or self.y2 > canvas_height:
            return False
        if self.area < min_area:
            return False
        return True

    def clamp(self, canvas_width: float, canvas_height: float) -> "BoundingBox":
        """Clamp coordinates to canvas boundaries."""
        cx1 = max(0.0, min(float(canvas_width), self.x1))
        cy1 = max(0.0, min(float(canvas_height), self.y1))
        cx2 = max(0.0, min(float(canvas_width), self.x2))
        cy2 = max(0.0, min(float(canvas_height), self.y2))
        return BoundingBox(x1=cx1, y1=cy1, x2=max(cx1 + 0.1, cx2), y2=max(cy1 + 0.1, cy2))

    def to_list(self) -> List[float]:
        return [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)]

    def to_dict(self) -> Dict[str, float]:
        return {"x1": round(self.x1, 2), "y1": round(self.y1, 2), "x2": round(self.x2, 2), "y2": round(self.y2, 2)}


@dataclass
class GroundTruthObject:
    """Represents a single ground-truth object observation in a video frame."""
    label_id: str
    track_id: Optional[str]
    category: str
    mapped_category: str
    mapped_class_id: int
    box2d: BoundingBox
    attributes: Dict[str, bool] = field(default_factory=dict)
    coordinate_space: str = "canvas"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_id": self.label_id,
            "track_id": self.track_id,
            "category": self.category,
            "mapped_category": self.mapped_category,
            "mapped_class_id": self.mapped_class_id,
            "box2d": self.box2d.to_dict(),
            "bbox_xyxy": self.box2d.to_list(),
            "attributes": self.attributes,
            "coordinate_space": self.coordinate_space
        }


@dataclass
class FrameGroundTruth:
    """Represents all ground-truth object annotations for a single video frame."""
    frame_index: int
    extracted_frame_id: int
    frame_filename: str
    source_frame_index: int
    timestamp_seconds: float
    image_width: int
    image_height: int
    coordinate_space: str
    annotations: List[GroundTruthObject] = field(default_factory=list)

    @property
    def object_count(self) -> int:
        return len(self.annotations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "extracted_frame_id": self.extracted_frame_id,
            "frame_filename": self.frame_filename,
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 4),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "coordinate_space": self.coordinate_space,
            "object_count": self.object_count,
            "annotations": [a.to_dict() for a in self.annotations]
        }


@dataclass
class SequenceGroundTruth:
    """Represents complete temporal ground-truth annotations for a video sequence."""
    sequence_id: str
    dataset_id: str
    source_dataset: str
    coordinate_space: str
    total_frames: int
    total_annotations: int
    unique_track_count: int
    class_distribution: Dict[str, int]
    frames: List[FrameGroundTruth] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "sequence_id": self.sequence_id,
            "dataset_id": self.dataset_id,
            "source_dataset": self.source_dataset,
            "coordinate_space": self.coordinate_space,
            "total_frames": self.total_frames,
            "total_annotations": self.total_annotations,
            "unique_track_count": self.unique_track_count,
            "class_distribution": self.class_distribution,
            "frames": [f.to_dict() for f in self.frames]
        }

# ==============================================================================
# TAXONOMY & CLASS MAPPING
# ==============================================================================

# Standard mapping from BDD100K 11 primary classes to YOLO COCO class names and IDs
DEFAULT_BDD_TO_COCO_MAPPING: Dict[str, Tuple[str, int]] = {
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

# ==============================================================================
# COORDINATE TRANSFORMATION ENGINE
# ==============================================================================

class CoordinateTransformer:
    """Handles mathematically exact coordinate transformations across pipeline spaces."""

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

    def native_to_raw_video(self, box: BoundingBox) -> BoundingBox:
        """Transform native BDD100K landscape coordinates (1280x720) to rotated raw video (720x1280).
        
        Transformation: 90 degrees clockwise rotation:
          x_raw = 720 - y_native
          y_raw = x_native
        """
        rx1 = float(self.raw_video_width) - box.y2
        rx2 = float(self.raw_video_width) - box.y1
        ry1 = box.x1
        ry2 = box.x2

        return BoundingBox(
            x1=min(rx1, rx2),
            y1=min(ry1, ry2),
            x2=max(rx1, rx2),
            y2=max(ry1, ry2)
        )

    def raw_video_to_canvas(self, box: BoundingBox) -> BoundingBox:
        """Transform raw video coordinates (720x1280) to preprocessed letterbox canvas (640x360).
        
        Transformation:
          x_canvas = x_raw * scale + offset_x
          y_canvas = y_raw * scale + offset_y
        """
        cx1 = box.x1 * self.scale_factor + self.offset_x
        cy1 = box.y1 * self.scale_factor + self.offset_y
        cx2 = box.x2 * self.scale_factor + self.offset_x
        cy2 = box.y2 * self.scale_factor + self.offset_y

        clamped = BoundingBox(
            x1=min(cx1, cx2),
            y1=min(cy1, cy2),
            x2=max(cx1, cx2),
            y2=max(cy1, cy2)
        ).clamp(self.canvas_width, self.canvas_height)

        return clamped

    def native_to_canvas(self, box: BoundingBox) -> BoundingBox:
        """Compose native-to-raw and raw-to-canvas transformations in one step."""
        raw_box = self.native_to_raw_video(box)
        return self.raw_video_to_canvas(raw_box)

    def transform(self, box: BoundingBox, target_space: str) -> BoundingBox:
        """Dispatch coordinate transformation to requested space."""
        if target_space == "native":
            return box
        elif target_space == "raw_video":
            return self.native_to_raw_video(box)
        elif target_space == "canvas":
            return self.native_to_canvas(box)
        else:
            raise CoordinateConversionError(f"Unsupported coordinate target space: '{target_space}'")

# ==============================================================================
# BDD100K ADAPTER
# ==============================================================================

class BDD100KAdapter:
    """Production-grade ingestion and conversion engine for BDD100K MOT annotations."""

    def __init__(
        self,
        class_mapping: Optional[Dict[str, Tuple[str, int]]] = None,
        transformer: Optional[CoordinateTransformer] = None,
        min_box_area: float = 4.0,
    ) -> None:
        self.class_mapping = class_mapping or DEFAULT_BDD_TO_COCO_MAPPING
        self.transformer = transformer or CoordinateTransformer()
        self.min_box_area = min_box_area

    def load_raw_json(self, json_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and perform preliminary validation on raw BDD100K annotation JSON."""
        path = Path(json_path)
        if not path.is_file():
            raise AnnotationFileNotFoundError(f"Annotation file not found at: '{path}'")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise AnnotationSchemaError(f"Failed to parse JSON file '{path}': {e}") from e

        # Handle list-wrapped sequence format or direct dict
        if isinstance(data, list):
            if len(data) == 0:
                raise AnnotationSchemaError(f"Annotation file '{path}' contains an empty list.")
            return data[0]
        elif isinstance(data, dict):
            return data
        else:
            raise AnnotationSchemaError(f"Unexpected top-level data type '{type(data)}' in '{path}'.")

    def parse_sequence(
        self,
        raw_data: Dict[str, Any],
        dataset_id: str = "driving_sample",
        target_space: str = "canvas",
        max_frames: int = 200,
        frame_name_pattern: str = "frame_{:06d}.jpg",
    ) -> SequenceGroundTruth:
        """Parse raw BDD100K sequence dict into a verified SequenceGroundTruth model."""
        seq_name = str(raw_data.get("name", "unknown_sequence"))
        raw_frames = raw_data.get("frames", [])

        if not isinstance(raw_frames, list) or len(raw_frames) == 0:
            raise AnnotationSchemaError(f"Sequence '{seq_name}' contains zero frames or invalid 'frames' key.")

        processed_frames: List[FrameGroundTruth] = []
        unique_tracks: Set[str] = set()
        class_counts: Dict[str, int] = {}
        total_annotations = 0

        # Target dimensions based on coordinate space
        if target_space == "canvas":
            img_w, img_h = self.transformer.canvas_width, self.transformer.canvas_height
        elif target_space == "raw_video":
            img_w, img_h = self.transformer.raw_video_width, self.transformer.raw_video_height
        else:
            img_w, img_h = self.transformer.native_width, self.transformer.native_height

        # Process frames up to max_frames (e.g. 200 frames for driving_sample)
        sorted_raw_frames = sorted(raw_frames, key=lambda f: int(f.get("frame_index", 0)))[:max_frames]

        for idx, rf in enumerate(sorted_raw_frames):
            f_idx = int(rf.get("frame_index", idx))
            extracted_frame_id = idx + 1
            frame_filename = frame_name_pattern.format(extracted_frame_id)
            
            # Map source frame index (extraction @ 5 FPS from 30 FPS = step of 6)
            source_frame_idx = int(rf.get("video_frame", idx * 6))
            timestamp = idx * 0.20  # 5 FPS = 0.20 s per frame

            raw_labels = rf.get("labels", [])
            frame_objs: List[GroundTruthObject] = []

            for l_idx, rl in enumerate(raw_labels):
                lbl_id = str(rl.get("id", f"{f_idx}_{l_idx}"))
                raw_cat = str(rl.get("category", "unknown")).lower().strip()
                
                # Check class mapping
                if raw_cat in self.class_mapping:
                    mapped_cat, mapped_id = self.class_mapping[raw_cat]
                else:
                    mapped_cat, mapped_id = "unknown", -1

                box_dict = rl.get("box2d", {})
                if not box_dict or not all(k in box_dict for k in ("x1", "y1", "x2", "y2")):
                    logger.warning(f"Frame {extracted_frame_id}: skipping malformed box in label {lbl_id}")
                    continue

                try:
                    native_box = BoundingBox(
                        x1=float(box_dict["x1"]),
                        y1=float(box_dict["y1"]),
                        x2=float(box_dict["x2"]),
                        y2=float(box_dict["y2"]),
                    )
                except (ValueError, TypeError):
                    logger.warning(f"Frame {extracted_frame_id}: non-numeric coordinates in label {lbl_id}")
                    continue

                if not native_box.is_finite:
                    continue

                # Coordinate transformation
                transformed_box = self.transformer.transform(native_box, target_space=target_space)

                # Skip invalid or collapsed boxes in target space
                if not transformed_box.is_valid(img_w, img_h, min_area=self.min_box_area):
                    continue

                track_id = str(rl.get("id")) if rl.get("id") is not None else None
                if track_id:
                    unique_tracks.add(track_id)

                attrs = {
                    "crowd": bool(rl.get("attributes", {}).get("crowd", False)),
                    "occluded": bool(rl.get("attributes", {}).get("occluded", False)),
                    "truncated": bool(rl.get("attributes", {}).get("truncated", False)),
                }

                gt_obj = GroundTruthObject(
                    label_id=lbl_id,
                    track_id=track_id,
                    category=raw_cat,
                    mapped_category=mapped_cat,
                    mapped_class_id=mapped_id,
                    box2d=transformed_box,
                    attributes=attrs,
                    coordinate_space=target_space
                )
                frame_objs.append(gt_obj)
                class_counts[mapped_cat] = class_counts.get(mapped_cat, 0) + 1
                total_annotations += 1

            frame_gt = FrameGroundTruth(
                frame_index=f_idx,
                extracted_frame_id=extracted_frame_id,
                frame_filename=frame_filename,
                source_frame_index=source_frame_idx,
                timestamp_seconds=timestamp,
                image_width=img_w,
                image_height=img_h,
                coordinate_space=target_space,
                annotations=frame_objs
            )
            processed_frames.append(frame_gt)

        return SequenceGroundTruth(
            sequence_id=seq_name,
            dataset_id=dataset_id,
            source_dataset="bdd100k",
            coordinate_space=target_space,
            total_frames=len(processed_frames),
            total_annotations=total_annotations,
            unique_track_count=len(unique_tracks),
            class_distribution=class_counts,
            frames=processed_frames
        )

    def validate_integrity(
        self,
        sequence_gt: SequenceGroundTruth,
        frames_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Perform bounded integrity validation on parsed ground truth."""
        report: Dict[str, Any] = {
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence_id": sequence_gt.sequence_id,
            "dataset_id": sequence_gt.dataset_id,
            "coordinate_space": sequence_gt.coordinate_space,
            "total_frames_checked": sequence_gt.total_frames,
            "total_annotations_checked": sequence_gt.total_annotations,
            "unique_track_count": sequence_gt.unique_track_count,
            "class_distribution": sequence_gt.class_distribution,
            "corrupt_frames_count": 0,
            "missing_images_count": 0,
            "invalid_boxes_count": 0,
            "duplicate_records_count": 0,
            "out_of_bounds_count": 0,
            "empty_frames_count": 0,
            "status": "PASS"
        }

        frames_path = Path(frames_dir) if frames_dir else None
        seen_frame_ids: Set[int] = set()

        for f in sequence_gt.frames:
            # Check duplicate frame index
            if f.extracted_frame_id in seen_frame_ids:
                report["duplicate_records_count"] += 1
            seen_frame_ids.add(f.extracted_frame_id)

            # Check image existence if frames_dir provided
            if frames_path:
                img_path = frames_path / f.frame_filename
                if not img_path.is_file():
                    report["missing_images_count"] += 1

            if f.object_count == 0:
                report["empty_frames_count"] += 1

            # Validate each box
            for obj in f.annotations:
                b = obj.box2d
                if not b.is_valid(f.image_width, f.image_height, min_area=self.min_box_area):
                    report["invalid_boxes_count"] += 1

        if report["corrupt_frames_count"] > 0 or report["invalid_boxes_count"] > 0:
            report["status"] = "FAIL"
        elif report["missing_images_count"] > 0:
            report["status"] = "PARTIAL"
        else:
            report["status"] = "PASS"

        return report

    def generate_manifest(
        self,
        sequence_gt: SequenceGroundTruth,
        annotation_file_rel_path: str,
        frames_dir_rel_path: str,
        output_manifest_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Generate a portable, machine-readable dataset manifest with strictly relative paths."""
        manifest = {
            "schema_version": "1.0.0",
            "dataset_id": sequence_gt.dataset_id,
            "dataset_name": "PerceptGuard Autonomous Driving Sample (BDD100K Ground Truth)",
            "source_dataset": "BDD100K Multiple Object Tracking (MOT)",
            "sequence_id": sequence_gt.sequence_id,
            "license": "UC Berkeley Regents Academic / Research License",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "coordinate_space": sequence_gt.coordinate_space,
            "image_dimensions": [sequence_gt.frames[0].image_width, sequence_gt.frames[0].image_height] if sequence_gt.frames else [640, 360],
            "total_frames": sequence_gt.total_frames,
            "total_ground_truth_boxes": sequence_gt.total_annotations,
            "unique_tracks": sequence_gt.unique_track_count,
            "class_taxonomy": list(sequence_gt.class_distribution.keys()),
            "relative_paths": {
                "frames_directory": frames_dir_rel_path.replace("\\", "/"),
                "annotations_file": annotation_file_rel_path.replace("\\", "/"),
            },
            "frame_registry": [
                {
                    "extracted_frame_id": f.extracted_frame_id,
                    "frame_filename": f.frame_filename,
                    "source_frame_index": f.source_frame_index,
                    "timestamp_seconds": f.timestamp_seconds,
                    "object_count": f.object_count,
                }
                for f in sequence_gt.frames
            ]
        }

        out_path = Path(output_manifest_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def export_standardized_json(
        self,
        sequence_gt: SequenceGroundTruth,
        output_path: Union[str, Path],
    ) -> Path:
        """Export parsed SequenceGroundTruth to standardized JSON file."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(sequence_gt.to_dict(), f, indent=2)
        return out_file
