"""PerceptGuard AI: Advanced Object Detection Visualization Engine.

Senior Staff-level, production-grade computer vision visualization engine for
autonomous-driving perception verification.

Key Architectural Guarantees:
- Reads and validates versioned Task 4 detection JSON (Schema v1.0).
- Pure visualization decoupling: strictly zero model initialization, zero weight loading,
  and zero re-inference.
- Deterministic, mathematically reproducible class color mapping across all frames and runs.
- High-contrast, dimension-preserving bounding box and label rendering.
- Robust boundary clamping preventing text and box coordinate overflow.
- Clean handling of zero-detection frames without classification as errors.
- Isolated per-frame fault containment; corrupt frames are recorded in an error catalog.
- Memory-bounded sequential CPU execution (< 300 MB RSS).
- Versioned multi-file atomic persistence (Schema v1.0).
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np

logger = logging.getLogger("PerceptGuard.DetectionVisualizer")


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class VisualizationError(Exception):
    """Base exception for all visualization failures."""
    pass


class DetectionJSONValidationError(VisualizationError):
    """Raised when detection JSON fails structural or semantic validation."""
    pass


class UnsupportedSchemaVersionError(DetectionJSONValidationError):
    """Raised when detection JSON schema version is incompatible."""
    pass


class SourceImageNotFoundError(VisualizationError):
    """Raised when the source image corresponding to a detection frame cannot be found."""
    pass


class SourceImageDecodeError(VisualizationError):
    """Raised when the source image file is corrupted or cannot be decoded."""
    pass


class InvalidBoundingBoxError(VisualizationError):
    """Raised when bounding box coordinates violate geometric or finite bounds."""
    pass


class OutputPersistenceError(VisualizationError):
    """Raised when persisting annotated frames or metadata to disk fails."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass(frozen=True)
class BoundingBoxAnnotation:
    """Immutable data contract representing a validated visual annotation."""
    bbox_xyxy: List[float]
    class_id: int
    class_name: str
    confidence: float
    color_bgr: Tuple[int, int, int]


@dataclass
class FrameVisualizationRecord:
    """Visualization outcome and metrics for an individual frame."""
    image_filename: str
    source_image_path: str
    output_image_path: Optional[str]
    extracted_frame_id: Optional[int] = None
    detection_count: int = 0
    boxes_rendered: int = 0
    status: str = "success"  # "success", "failed", "skipped"
    error_message: Optional[str] = None
    visualization_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_filename": self.image_filename,
            "source_image_path": self.source_image_path,
            "output_image_path": self.output_image_path,
            "extracted_frame_id": self.extracted_frame_id,
            "detection_count": int(self.detection_count),
            "boxes_rendered": int(self.boxes_rendered),
            "status": self.status,
            "error_message": self.error_message,
            "visualization_time_ms": round(float(self.visualization_time_ms), 2),
        }


@dataclass
class VisualizationErrorRecord:
    """Diagnostic details for an individual frame visualization failure."""
    frame_identifier: str
    input_image_path: str
    error_category: str
    error_message: str
    failure_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_identifier": self.frame_identifier,
            "input_image_path": self.input_image_path,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "failure_timestamp": self.failure_timestamp,
        }


@dataclass
class VisualizationSummary:
    """Aggregated operational telemetry and metrics for a visualization batch."""
    schema_version: str = "1.0"
    run_id: str = ""
    dataset_id: str = ""
    detection_run_id: str = ""
    total_frames_in_detection_results: int = 0
    total_frames_attempted: int = 0
    successful_visualizations: int = 0
    failed_visualizations: int = 0
    frames_with_detections: int = 0
    frames_without_detections: int = 0
    total_boxes_rendered: int = 0
    invalid_detection_records: int = 0
    total_visualization_time_seconds: float = 0.0
    average_visualization_time_ms: float = 0.0
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    final_status: str = "PASS"  # "PASS", "WARNING", "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "detection_run_id": self.detection_run_id,
            "total_frames_in_detection_results": int(self.total_frames_in_detection_results),
            "total_frames_attempted": int(self.total_frames_attempted),
            "successful_visualizations": int(self.successful_visualizations),
            "failed_visualizations": int(self.failed_visualizations),
            "frames_with_detections": int(self.frames_with_detections),
            "frames_without_detections": int(self.frames_without_detections),
            "total_boxes_rendered": int(self.total_boxes_rendered),
            "invalid_detection_records": int(self.invalid_detection_records),
            "total_visualization_time_seconds": round(float(self.total_visualization_time_seconds), 4),
            "average_visualization_time_ms": round(float(self.average_visualization_time_ms), 2),
            "generated_at": self.generated_at,
            "final_status": self.final_status,
        }


# ==============================================================================
# Deterministic Color Management
# ==============================================================================

def generate_deterministic_palette(num_classes: int = 80) -> Dict[int, Tuple[int, int, int]]:
    """Generate a mathematically deterministic BGR color palette for object classes.
    
    Uses golden-angle hue distribution in HSV space, with explicit high-contrast
    overrides for primary autonomous driving classes.
    """
    palette: Dict[int, Tuple[int, int, int]] = {}

    # Explicit curated BGR colors for primary road users and traffic infrastructure
    # (High visibility against dark asphalt, sky, and foliage)
    curated_colors: Dict[int, Tuple[int, int, int]] = {
        0: (0, 255, 0),       # person: Vibrant Green
        1: (255, 255, 0),     # bicycle: Cyan
        2: (255, 128, 0),     # car: Vivid Blue
        3: (180, 105, 255),   # motorcycle: Pink
        5: (255, 0, 255),     # bus: Magenta
        7: (0, 140, 255),     # truck: Orange
        9: (0, 215, 255),     # traffic light: Gold
        11: (0, 0, 255),      # stop sign: Pure Red
    }

    golden_ratio_conjugate = 0.618033988749895
    h = 0.1  # starting hue offset

    for i in range(max(num_classes, 100)):
        if i in curated_colors:
            palette[i] = curated_colors[i]
            continue

        h = (h + golden_ratio_conjugate) % 1.0
        # Convert HSV (h, 0.85, 0.95) to BGR
        h_deg = h * 360.0
        c = 0.85 * 0.95
        x = c * (1.0 - abs((h_deg / 60.0) % 2 - 1.0))
        m = 0.95 - c

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

        b = int((b_p + m) * 255.0)
        g = int((g_p + m) * 255.0)
        r = int((r_p + m) * 255.0)
        palette[i] = (b, g, r)

    return palette


# Global deterministic palette lookup
DEFAULT_COLOR_PALETTE = generate_deterministic_palette(100)


# ==============================================================================
# DetectionVisualizer Engine
# ==============================================================================

class DetectionVisualizer:
    """Production-grade visualization engine for object detection results."""

    SUPPORTED_SCHEMA_VERSIONS: Set[str] = {"1.0"}

    def __init__(
        self,
        color_palette: Optional[Dict[int, Tuple[int, int, int]]] = None,
        confidence_decimals: int = 2,
        line_thickness: Optional[int] = None,
        font_scale: Optional[float] = None,
        jpeg_quality: int = 95,
        project_root: Optional[Path] = None,
    ) -> None:
        """Initialize DetectionVisualizer with style parameters.
        
        Args:
            color_palette: Optional custom mapping of class_id to BGR tuple.
            confidence_decimals: Number of decimal places in confidence display.
            line_thickness: Bounding box line thickness (auto-scaled if None).
            font_scale: OpenCV font scale (auto-scaled if None).
            jpeg_quality: Output JPEG encoding quality (1-100).
            project_root: Absolute path to project repository root.
        """
        self.color_palette = color_palette or DEFAULT_COLOR_PALETTE
        self.confidence_decimals = max(1, min(confidence_decimals, 4))
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.jpeg_quality = max(1, min(jpeg_quality, 100))
        self.project_root = project_root or _project_root

    # --------------------------------------------------------------------------
    # Detection JSON Loading & Validation
    # --------------------------------------------------------------------------

    def load_and_validate_detection_json(
        self,
        detection_file: Union[str, Path],
    ) -> Dict[str, Any]:
        """Load and structurally validate detection JSON file.
        
        Args:
            detection_file: Path to detection_results.json.
            
        Returns:
            Validated detection data dictionary.
            
        Raises:
            DetectionJSONValidationError: On missing file, invalid JSON, or missing schema fields.
            UnsupportedSchemaVersionError: If schema_version is unsupported.
        """
        path = Path(detection_file)
        if not path.is_file():
            raise DetectionJSONValidationError(f"Detection results file not found: '{path}'")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DetectionJSONValidationError(f"Malformed detection JSON file '{path}': {e}") from e
        except Exception as e:
            raise DetectionJSONValidationError(f"Failed to read detection JSON file '{path}': {e}") from e

        if not isinstance(data, dict):
            raise DetectionJSONValidationError(f"Expected root JSON object in '{path}', got {type(data).__name__}")

        # Schema version check
        schema_version = data.get("schema_version")
        if not schema_version:
            raise DetectionJSONValidationError(f"Missing 'schema_version' in detection JSON: '{path}'")
        if str(schema_version) not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedSchemaVersionError(
                f"Unsupported schema version '{schema_version}' in '{path}'. "
                f"Supported versions: {sorted(self.SUPPORTED_SCHEMA_VERSIONS)}"
            )

        # Core top-level fields
        for field_name in ["run_id", "dataset_id", "frame_results"]:
            if field_name not in data:
                raise DetectionJSONValidationError(f"Missing required field '{field_name}' in detection JSON '{path}'")

        if not isinstance(data["frame_results"], list):
            raise DetectionJSONValidationError(
                f"'frame_results' must be a list in '{path}', got {type(data['frame_results']).__name__}"
            )

        return data

    # --------------------------------------------------------------------------
    # Source Image Resolution & Reading
    # --------------------------------------------------------------------------

    def resolve_source_image_path(
        self,
        frame_dict: Dict[str, Any],
        fallback_image_dir: Optional[Union[str, Path]] = None,
        dataset_id: Optional[str] = None,
    ) -> Path:
        """Resolve physical on-disk path for a frame's source image.
        
        Checks paths in order:
        1. Exact `image_path` from JSON (if exists on host).
        2. `fallback_image_dir / image_filename`.
        3. Standard `datasets/processed/<dataset_id>/preprocessed_frames/<image_filename>`.
        4. Standard `datasets/processed/<dataset_id>/frames/<image_filename>`.
        
        Args:
            frame_dict: Individual frame result dictionary.
            fallback_image_dir: Explicit user-specified images directory.
            dataset_id: Dataset identifier for directory inference.
            
        Returns:
            Resolved Path object to source image.
            
        Raises:
            SourceImageNotFoundError: If the image cannot be resolved on disk.
        """
        filename = frame_dict.get("image_filename", "")
        raw_image_path = frame_dict.get("image_path", "")

        candidates: List[Path] = []

        if raw_image_path:
            candidates.append(Path(raw_image_path))

        if fallback_image_dir and filename:
            candidates.append(Path(fallback_image_dir) / filename)

        if dataset_id and filename:
            candidates.append(
                self.project_root / "datasets" / "processed" / dataset_id / "preprocessed_frames" / filename
            )
            candidates.append(
                self.project_root / "datasets" / "processed" / dataset_id / "frames" / filename
            )

        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate.resolve()
            except Exception:
                continue

        raise SourceImageNotFoundError(
            f"Source image '{filename}' could not be resolved on disk. Checked: {[str(c) for c in candidates]}"
        )

    def load_source_image(self, image_path: Path) -> np.ndarray:
        """Load image into OpenCV BGR numpy array using safe in-memory decoding.
        
        Avoids Windows C-runtime file handle locks and supports unicode paths.
        
        Args:
            image_path: Absolute Path to image file.
            
        Returns:
            Decoded BGR numpy array.
            
        Raises:
            SourceImageDecodeError: If decoding fails or returns an empty matrix.
        """
        try:
            with open(image_path, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            raise SourceImageDecodeError(f"Failed to read image bytes from '{image_path}': {e}") from e

        if not raw_bytes:
            raise SourceImageDecodeError(f"Empty image file (0 bytes): '{image_path}'")

        img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.size == 0 or img.ndim != 3:
            raise SourceImageDecodeError(f"OpenCV failed to decode image: '{image_path}'")

        return img

    # --------------------------------------------------------------------------
    # Bounding Box Validation & Color Assignment
    # --------------------------------------------------------------------------

    def validate_and_create_annotation(
        self,
        det_dict: Dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> Optional[BoundingBoxAnnotation]:
        """Validate detection dictionary and construct BoundingBoxAnnotation.
        
        Args:
            det_dict: Raw detection dictionary.
            image_width: Width of image canvas in pixels.
            image_height: Height of image canvas in pixels.
            
        Returns:
            Validated BoundingBoxAnnotation, or None if detection is malformed.
        """
        # Validate coordinates
        bbox = det_dict.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            logger.warning("Invalid bbox_xyxy format: %s", bbox)
            return None

        try:
            x1, y1, x2, y2 = [float(c) for c in bbox]
        except (ValueError, TypeError):
            logger.warning("Non-numeric coordinates in bbox: %s", bbox)
            return None

        # Assert finiteness (reject NaN, Inf)
        if not all(math.isfinite(c) for c in (x1, y1, x2, y2)):
            logger.warning("Non-finite coordinates in bbox: %s", bbox)
            return None

        # Validate geometry before clamping
        if x2 <= x1 or y2 <= y1:
            logger.warning("Degenerate box geometry (x2 <= x1 or y2 <= y1): %s", bbox)
            return None

        # Boundary clamping to physical image dimensions
        clamped_x1 = max(0.0, min(x1, float(image_width)))
        clamped_y1 = max(0.0, min(y1, float(image_height)))
        clamped_x2 = max(0.0, min(x2, float(image_width)))
        clamped_y2 = max(0.0, min(y2, float(image_height)))

        # Re-check geometry after clamping
        if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
            logger.warning("Box collapsed after boundary clamping: %s -> [%.1f, %.1f, %.1f, %.1f]",
                           bbox, clamped_x1, clamped_y1, clamped_x2, clamped_y2)
            return None

        # Validate confidence
        conf = det_dict.get("confidence", 0.0)
        try:
            conf_val = float(conf)
            if not math.isfinite(conf_val) or conf_val < 0.0 or conf_val > 1.0:
                logger.warning("Invalid confidence value '%s' in detection", conf)
                return None
        except (ValueError, TypeError):
            logger.warning("Non-numeric confidence '%s'", conf)
            return None

        # Validate class identity
        class_id = det_dict.get("class_id", -1)
        try:
            class_id_int = int(class_id)
        except (ValueError, TypeError):
            class_id_int = -1

        class_name = str(det_dict.get("class_name", f"class_{class_id_int}")).strip()
        if not class_name:
            class_name = f"class_{class_id_int}"

        # Assign deterministic color
        color_bgr = self.color_palette.get(
            class_id_int,
            self.color_palette.get(abs(class_id_int) % len(self.color_palette), (255, 255, 255))
        )

        return BoundingBoxAnnotation(
            bbox_xyxy=[clamped_x1, clamped_y1, clamped_x2, clamped_y2],
            class_id=class_id_int,
            class_name=class_name,
            confidence=conf_val,
            color_bgr=color_bgr,
        )

    # --------------------------------------------------------------------------
    # Rendering Engine
    # --------------------------------------------------------------------------

    def render_annotations(
        self,
        image: np.ndarray,
        annotations: Sequence[BoundingBoxAnnotation],
    ) -> Tuple[np.ndarray, int]:
        """Draw bounding boxes and class labels onto an image copy.
        
        Args:
            image: Source BGR numpy image matrix (never modified in place).
            annotations: Sequence of validated BoundingBoxAnnotation records.
            
        Returns:
            Tuple of (annotated_image_matrix, count_of_boxes_rendered).
        """
        canvas = image.copy()
        img_h, img_w = canvas.shape[:2]

        if not annotations:
            return canvas, 0

        # Compute dynamic visual styling parameters based on resolution
        box_thickness = self.line_thickness or max(1, int(round(min(img_w, img_h) / 320.0)))
        font_scale = self.font_scale or max(0.35, min(img_w, img_h) / 800.0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = max(1, int(round(box_thickness * 0.75)))

        rendered_count = 0

        for ann in annotations:
            x1, y1, x2, y2 = [int(round(c)) for c in ann.bbox_xyxy]
            color = ann.color_bgr

            # 1. Draw outer bounding rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

            # 2. Format class label text
            label_text = f"{ann.class_name} {ann.confidence:.{self.confidence_decimals}f}"

            # 3. Calculate label text bounding dimensions
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            pad = 3
            label_bg_w = text_w + 2 * pad
            label_bg_h = text_h + baseline + 2 * pad

            # 4. Determine label banner position (prevent clipping off top or sides)
            # Default: place immediately above box
            if y1 - label_bg_h >= 0:
                bg_y1 = y1 - label_bg_h
                bg_y2 = y1
                text_y = y1 - baseline - pad
            else:
                # Top boundary overflow: draw inside the top of the box
                bg_y1 = y1
                bg_y2 = y1 + label_bg_h
                text_y = y1 + text_h + pad

            # Prevent right horizontal overflow
            bg_x1 = max(0, min(x1, img_w - label_bg_w))
            bg_x2 = min(img_w, bg_x1 + label_bg_w)
            text_x = bg_x1 + pad

            # 5. Draw solid background banner for high-contrast legibility
            cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

            # 6. Calculate contrasting text color (black vs white based on luminance)
            b, g, r = color
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            text_color = (0, 0, 0) if luminance > 0.55 else (255, 255, 255)

            # 7. Render label text
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

        return canvas, rendered_count

    # --------------------------------------------------------------------------
    # Output Persistence (Atomic writing)
    # --------------------------------------------------------------------------

    def save_annotated_image(
        self,
        annotated_image: np.ndarray,
        output_path: Union[str, Path],
        overwrite: bool = True,
    ) -> Path:
        """Save annotated image atomically to disk preserving image dimensions.
        
        Args:
            annotated_image: BGR numpy image matrix.
            output_path: Desired destination file path.
            overwrite: Whether to overwrite existing destination.
            
        Returns:
            Resolved Path of the saved image file.
            
        Raises:
            OutputPersistenceError: If saving fails or destination exists without overwrite.
        """
        dest_path = Path(output_path).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and not overwrite:
            raise OutputPersistenceError(f"Destination image file '{dest_path}' exists and overwrite is False")

        # Encode image to JPEG buffer
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, buffer = cv2.imencode(".jpg", annotated_image, encode_params)
        if not success or buffer is None:
            raise OutputPersistenceError(f"OpenCV failed to encode JPEG image for '{dest_path}'")

        # Atomic write via temporary file
        temp_file = dest_path.with_name(f".tmp_{uuid.uuid4().hex}_{dest_path.name}")
        try:
            with open(temp_file, "wb") as f:
                f.write(buffer.tobytes())
            os.replace(temp_file, dest_path)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise OutputPersistenceError(f"Failed to persist annotated image to '{dest_path}': {e}") from e

        return dest_path

    # --------------------------------------------------------------------------
    # Single-Frame Visualization
    # --------------------------------------------------------------------------

    def visualize_single_frame(
        self,
        frame_dict: Dict[str, Any],
        output_image_path: Optional[Union[str, Path]] = None,
        fallback_image_dir: Optional[Union[str, Path]] = None,
        dataset_id: Optional[str] = None,
        overwrite: bool = True,
    ) -> FrameVisualizationRecord:
        """Visualize detections for a single frame result dictionary.
        
        Args:
            frame_dict: Frame detection result dictionary from Task 4 JSON.
            output_image_path: Optional destination path for annotated image.
            fallback_image_dir: Optional directory to resolve source frame.
            dataset_id: Dataset identifier for directory inference.
            overwrite: Whether to allow overwriting output image.
            
        Returns:
            FrameVisualizationRecord summarizing outcome and metrics.
        """
        t0 = time.perf_counter()
        filename = frame_dict.get("image_filename", "unknown_frame")
        ext_id = frame_dict.get("extracted_frame_id")
        det_count = int(frame_dict.get("detection_count", 0))
        raw_detections = frame_dict.get("detections", [])

        # 1. Resolve source image path
        try:
            src_path = self.resolve_source_image_path(
                frame_dict,
                fallback_image_dir=fallback_image_dir,
                dataset_id=dataset_id,
            )
        except SourceImageNotFoundError as e:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            return FrameVisualizationRecord(
                image_filename=filename,
                source_image_path=frame_dict.get("image_path", ""),
                output_image_path=None,
                extracted_frame_id=ext_id,
                detection_count=det_count,
                boxes_rendered=0,
                status="failed",
                error_message=str(e),
                visualization_time_ms=dur_ms,
            )

        # 2. Load source image safely (read-only)
        try:
            src_image = self.load_source_image(src_path)
        except SourceImageDecodeError as e:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            return FrameVisualizationRecord(
                image_filename=filename,
                source_image_path=str(src_path),
                output_image_path=None,
                extracted_frame_id=ext_id,
                detection_count=det_count,
                boxes_rendered=0,
                status="failed",
                error_message=str(e),
                visualization_time_ms=dur_ms,
            )

        img_h, img_w = src_image.shape[:2]

        # 3. Validate and construct annotations
        annotations: List[BoundingBoxAnnotation] = []
        for det in raw_detections:
            ann = self.validate_and_create_annotation(det, img_w, img_h)
            if ann is not None:
                annotations.append(ann)

        # 4. Render annotations onto canvas
        annotated_image, boxes_rendered = self.render_annotations(src_image, annotations)

        # 5. Persist annotated image if output path specified
        saved_path_str: Optional[str] = None
        if output_image_path:
            try:
                saved_path = self.save_annotated_image(
                    annotated_image,
                    output_image_path,
                    overwrite=overwrite,
                )
                saved_path_str = str(saved_path)
            except OutputPersistenceError as e:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                return FrameVisualizationRecord(
                    image_filename=filename,
                    source_image_path=str(src_path),
                    output_image_path=None,
                    extracted_frame_id=ext_id,
                    detection_count=det_count,
                    boxes_rendered=boxes_rendered,
                    status="failed",
                    error_message=str(e),
                    visualization_time_ms=dur_ms,
                )

        dur_ms = (time.perf_counter() - t0) * 1000.0
        return FrameVisualizationRecord(
            image_filename=filename,
            source_image_path=str(src_path),
            output_image_path=saved_path_str,
            extracted_frame_id=ext_id,
            detection_count=det_count,
            boxes_rendered=boxes_rendered,
            status="success",
            error_message=None,
            visualization_time_ms=dur_ms,
        )

    # --------------------------------------------------------------------------
    # Batch Visualization Execution
    # --------------------------------------------------------------------------

    def visualize_batch(
        self,
        detection_file: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        overwrite: bool = True,
    ) -> Tuple[VisualizationSummary, List[FrameVisualizationRecord], List[VisualizationErrorRecord], Path]:
        """Execute sequential batch visualization across detection results.
        
        Args:
            detection_file: Path to detection_results.json.
            image_dir: Optional directory containing source images.
            output_dir: Target visualization output root directory.
            max_frames: Optional maximum frames cap (for smoke tests).
            start_index: 0-based frame start index for range slicing.
            end_index: 0-based frame end index (exclusive).
            overwrite: Whether to overwrite existing output files.
            
        Returns:
            Tuple of (VisualizationSummary, list of FrameVisualizationRecord,
                      list of VisualizationErrorRecord, output_dir_path).
                      
        Raises:
            OutputPersistenceError: If output directory exists and overwrite is False.
        """
        t_batch_start = time.perf_counter()

        # 1. Load and validate detection JSON
        detection_data = self.load_and_validate_detection_json(detection_file)
        detection_run_id = str(detection_data.get("run_id", "unknown_detection_run"))
        dataset_id = str(detection_data.get("dataset_id", "unknown_dataset"))
        all_frames: List[Dict[str, Any]] = detection_data["frame_results"]
        total_in_results = len(all_frames)

        # 2. Resolve visualization output directory
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_uuid = uuid.uuid4().hex[:6]
        run_id = f"vis_{timestamp_slug}_{run_uuid}"

        if output_dir:
            out_root = Path(output_dir).resolve()
        else:
            out_root = (
                self.project_root
                / "datasets"
                / "processed"
                / dataset_id
                / "visualizations"
                / run_id
            ).resolve()

        if out_root.exists() and not overwrite:
            raise OutputPersistenceError(
                f"Visualization directory '{out_root}' already exists and overwrite=False"
            )

        annotated_frames_dir = out_root / "annotated_frames"
        annotated_frames_dir.mkdir(parents=True, exist_ok=True)

        # 3. Apply frame range slicing
        sliced_frames = all_frames[start_index:end_index] if end_index is not None else all_frames[start_index:]
        if max_frames is not None and max_frames > 0:
            sliced_frames = sliced_frames[:max_frames]

        total_attempted = len(sliced_frames)
        is_partial_run = (total_attempted < total_in_results) or (start_index > 0) or (end_index is not None)

        frame_records: List[FrameVisualizationRecord] = []
        error_records: List[VisualizationErrorRecord] = []

        successful_count = 0
        failed_count = 0
        frames_with_det = 0
        frames_without_det = 0
        total_boxes_rendered = 0
        invalid_det_count = 0

        # 4. Sequential visualization processing loop (memory bounded)
        for frame_dict in sliced_frames:
            filename = frame_dict.get("image_filename", "frame.jpg")
            stem = Path(filename).stem
            dest_image_path = annotated_frames_dir / f"{stem}_annotated.jpg"

            det_count = int(frame_dict.get("detection_count", 0))
            raw_detections = frame_dict.get("detections", [])

            # Check for invalid detection records in input JSON
            if len(raw_detections) != det_count:
                invalid_det_count += abs(len(raw_detections) - det_count)

            record = self.visualize_single_frame(
                frame_dict,
                output_image_path=dest_image_path,
                fallback_image_dir=image_dir,
                dataset_id=dataset_id,
                overwrite=overwrite,
            )
            frame_records.append(record)

            if record.status == "success":
                successful_count += 1
                total_boxes_rendered += record.boxes_rendered
                if record.boxes_rendered > 0 or det_count > 0:
                    frames_with_det += 1
                else:
                    frames_without_det += 1
            else:
                failed_count += 1
                err_record = VisualizationErrorRecord(
                    frame_identifier=record.image_filename,
                    input_image_path=record.source_image_path,
                    error_category="FrameVisualizationError",
                    error_message=record.error_message or "Unknown visualization error",
                )
                error_records.append(err_record)

        total_duration_sec = time.perf_counter() - t_batch_start
        avg_vis_ms = (
            (total_duration_sec * 1000.0 / total_attempted) if total_attempted > 0 else 0.0
        )

        # 5. Determine operational final status
        if failed_count == 0 and not is_partial_run:
            final_status = "PASS"
        elif successful_count > 0:
            final_status = "WARNING"
        else:
            final_status = "FAIL"

        summary = VisualizationSummary(
            schema_version="1.0",
            run_id=run_id,
            dataset_id=dataset_id,
            detection_run_id=detection_run_id,
            total_frames_in_detection_results=total_in_results,
            total_frames_attempted=total_attempted,
            successful_visualizations=successful_count,
            failed_visualizations=failed_count,
            frames_with_detections=frames_with_det,
            frames_without_detections=frames_without_det,
            total_boxes_rendered=total_boxes_rendered,
            invalid_detection_records=invalid_det_count,
            total_visualization_time_seconds=total_duration_sec,
            average_visualization_time_ms=avg_vis_ms,
            final_status=final_status,
        )

        # 6. Generate multi-file JSON outputs
        self._write_multi_file_outputs(
            out_root=out_root,
            summary=summary,
            frame_records=frame_records,
            error_records=error_records,
            detection_file_path=Path(detection_file).resolve(),
        )

        return summary, frame_records, error_records, out_root

    # --------------------------------------------------------------------------
    # Output File Serialization (Atomic)
    # --------------------------------------------------------------------------

    def _write_multi_file_outputs(
        self,
        out_root: Path,
        summary: VisualizationSummary,
        frame_records: List[FrameVisualizationRecord],
        error_records: List[VisualizationErrorRecord],
        detection_file_path: Path,
    ) -> None:
        """Atomically persist summary, errors, and metadata JSON files."""
        # 1. visualization_summary.json
        self._atomic_json_dump(
            out_root / "visualization_summary.json",
            summary.to_dict(),
        )

        # 2. visualization_errors.json
        errors_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "error_count": len(error_records),
            "errors": [e.to_dict() for e in error_records],
        }
        self._atomic_json_dump(
            out_root / "visualization_errors.json",
            errors_payload,
        )

        # 3. visualization_metadata.json
        metadata_payload = {
            "schema_version": "1.0",
            "run_id": summary.run_id,
            "dataset_id": summary.dataset_id,
            "source_detection_run_id": summary.detection_run_id,
            "source_detection_file": str(detection_file_path),
            "annotation_style": {
                "confidence_decimals": self.confidence_decimals,
                "line_thickness": self.line_thickness,
                "font_scale": self.font_scale,
                "jpeg_quality": self.jpeg_quality,
                "coordinate_convention": "[x1, y1, x2, y2]",
                "label_format": "<class_name> <confidence>",
            },
            "color_palette_sample": {
                str(k): list(v) for k, v in list(self.color_palette.items())[:15]
            },
            "software_versions": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "environment": {
                "os": platform.platform(),
                "processor": platform.processor(),
            },
            "generated_at": summary.generated_at,
        }
        self._atomic_json_dump(
            out_root / "visualization_metadata.json",
            metadata_payload,
        )

    @staticmethod
    def _atomic_json_dump(target_path: Path, payload: Dict[str, Any]) -> None:
        """Serialize dictionary to JSON atomically using a temporary file."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_path.with_name(f".tmp_{uuid.uuid4().hex}_{target_path.name}")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(temp_file, target_path)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise OutputPersistenceError(f"Failed to atomically persist JSON to '{target_path}': {e}") from e


# ==============================================================================
# CLI Entrypoint & Formatted Reporting
# ==============================================================================

def main(cli_args: Optional[List[str]] = None) -> int:
    """CLI runner for DetectionVisualizer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI: Advanced Object Detection Visualization Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detection-file",
        type=str,
        required=True,
        help="Path to detection_results.json artifact generated by Task 4.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing source images (auto-resolved if omitted).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for annotated frames and visualization metadata.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to visualize (useful for smoke tests).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based start index of frame sequence.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="0-based end index of frame sequence (exclusive).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination directory if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG diagnostic logging.",
    )

    args = parser.parse_args(cli_args)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    visualizer = DetectionVisualizer(project_root=_project_root)

    try:
        summary, _, _, out_root = visualizer.visualize_batch(
            detection_file=args.detection_file,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            start_index=args.start_index,
            end_index=args.end_index,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"\nFATAL: Visualization execution aborted: {e}", file=sys.stderr)
        return 1

    # Formatted terminal output complying with prompt specification
    print("=" * 60)
    print("PERCEPTGUARD AI – DETECTION VISUALIZATION")
    print("=" * 60)
    print(f"Dataset:                  {summary.dataset_id}")
    print(f"Source detection run:     {summary.detection_run_id}")
    print(f"Input detection file:     {Path(args.detection_file).resolve()}")
    print(f"Input image directory:    {args.image_dir or 'Auto-resolved from dataset layout'}")
    print(f"Output directory:         {out_root}")
    print()
    print(f"Frames discovered:        {summary.total_frames_in_detection_results}")
    print(f"Frames attempted:         {summary.total_frames_attempted}")
    print(f"Successful visualizations:{summary.successful_visualizations}")
    print(f"Failed visualizations:    {summary.failed_visualizations}")
    print(f"Frames with detections:   {summary.frames_with_detections}")
    print(f"Frames without detections:{summary.frames_without_detections}")
    print(f"Total boxes rendered:     {summary.total_boxes_rendered}")
    print()
    print(f"Total visualization time: {summary.total_visualization_time_seconds:.2f} s")
    print(f"Average visualization time:{summary.average_visualization_time_ms:.2f} ms")
    print()
    print(f"Output status:            {summary.final_status}")
    print("=" * 60)

    return 0 if summary.final_status in ("PASS", "WARNING") else 1


if __name__ == "__main__":
    sys.exit(main())
