"""PerceptGuard-AI: Advanced Single Image Object Detection Engine.

Senior Staff-level, production-grade object detection engine for single video frames
and autonomous-driving camera images.

Key Guarantees:
- Strict input image validation (existence, readability, formats, dimensions, channels).
- Seamless integration with validated YOLOConfig and existing YOLOModelLoader.
- Zero redundant model initialization; lazy loading and cached runtime.
- Comprehensive bounding box ($x_1, y_1, x_2, y_2$) and confidence validation.
- Standardized, versioned JSON detection response schema (v1.0).
- Monotonic, high-resolution performance profiling (decode, infer, serialize, total).
- Safe, atomic machine-readable JSON persistence with overwrite protection.
- Honest handling of zero-detection scenarios (success status, zero count, empty list).
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np

from src.detection.yolo_config import YOLOConfig, YOLOConfigError, load_yolo_config
from src.detection.yolo_model_loader import (
    YOLOInferenceError,
    YOLOModelInfo,
    YOLOModelLoadError,
    YOLOModelLoader,
    YOLOModelNotFoundError,
)

logger = logging.getLogger("PerceptGuard.SingleImageDetector")


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class SingleImageDetectionError(Exception):
    """Base exception for all single image detection failures."""
    pass


class ImageValidationError(SingleImageDetectionError):
    """Base exception for image validation failures."""
    pass


class ImageNotFoundError(ImageValidationError):
    """Raised when the specified image file does not exist on disk."""
    pass


class UnsupportedImageFormatError(ImageValidationError):
    """Raised when image extension or container is not supported."""
    pass


class ImageDecodeError(ImageValidationError):
    """Raised when OpenCV fails to decode the image file into a valid matrix."""
    pass


class DetectionInferenceError(SingleImageDetectionError):
    """Raised when model inference execution fails or produces invalid tensors."""
    pass


class DetectionSerializationError(SingleImageDetectionError):
    """Raised when saving detection results to JSON fails."""
    pass


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass(frozen=True)
class DetectedObject:
    """Immutable data contract representing a single validated detection."""
    detection_index: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]  # [x1, y1, x2, y2]

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection record to dictionary."""
        return {
            "detection_index": int(self.detection_index),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [round(float(coord), 1) for coord in self.bbox_xyxy],
        }


@dataclass
class SingleImageDetectionResult:
    """Full structured detection response for a single image complying with Schema v1.0."""
    schema_version: str = "1.0"
    image_filename: str = ""
    image_path: str = ""
    width: int = 0
    height: int = 0
    channels: int = 0
    model_name: str = "yolo11n"
    model_task: str = "detect"
    model_device: str = "cpu"
    image_size_configured: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    image_decode_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    serialization_time_ms: float = 0.0
    total_time_ms: float = 0.0
    processing_status: str = "success"
    detection_count: int = 0
    detections: List[DetectedObject] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert full response to versioned dictionary structure."""
        return {
            "schema_version": self.schema_version,
            "image": {
                "filename": self.image_filename,
                "image_path": self.image_path,
                "width": int(self.width),
                "height": int(self.height),
                "channels": int(self.channels),
            },
            "model": {
                "name": self.model_name,
                "task": self.model_task,
                "device": self.model_device,
            },
            "inference": {
                "image_size": int(self.image_size_configured),
                "confidence_threshold": round(float(self.confidence_threshold), 4),
                "iou_threshold": round(float(self.iou_threshold), 4),
                "max_detections": int(self.max_detections),
                "inference_time_ms": round(float(self.inference_time_ms), 2),
            },
            "performance": {
                "image_decode_time_ms": round(float(self.image_decode_time_ms), 2),
                "inference_time_ms": round(float(self.inference_time_ms), 2),
                "serialization_time_ms": round(float(self.serialization_time_ms), 2),
                "total_time_ms": round(float(self.total_time_ms), 2),
            },
            "result": {
                "processing_status": self.processing_status,
                "detection_count": int(self.detection_count),
                "detections": [d.to_dict() for d in self.detections],
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize response to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, output_path: Union[str, Path], overwrite: bool = False) -> Path:
        """Atomically persist structured detection results to a JSON file.

        Args:
            output_path: Destination filepath.
            overwrite: If False, raises DetectionSerializationError if file exists.

        Returns:
            Path: Absolute resolved path of the saved JSON file.

        Raises:
            DetectionSerializationError: If file exists (and overwrite=False) or write fails.
        """
        p = Path(output_path).resolve()
        if p.exists() and not overwrite:
            raise DetectionSerializationError(
                f"Output file already exists at '{p}'. Set overwrite=True to allow overwriting."
            )

        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via temporary file
        temp_file = p.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}")
        try:
            temp_file.write_text(self.to_json(), encoding="utf-8")
            temp_file.replace(p)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise DetectionSerializationError(f"Failed to write detection JSON to '{p}': {e}") from e

        return p


# ==============================================================================
# Single Image Object Detection Engine
# ==============================================================================

class SingleImageDetector:
    """Production-grade single-image object detection engine for PerceptGuard AI.

    Encapsulates image validation, OpenCV matrix decoding, YOLO inference execution,
    bounding box calibration, timing instrumentation, and atomic result serialization.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        loader: Optional[YOLOModelLoader] = None,
        config: Optional[Union[YOLOConfig, str, Path]] = None,
        preset: Optional[str] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize detector with existing or newly instantiated YOLOModelLoader.

        Args:
            loader: Optional pre-initialized YOLOModelLoader instance.
            config: Optional YOLOConfig instance or path to yolo_config.yaml.
            preset: Optional perception preset name ('balanced', 'high_precision', etc.).
            project_root: Base repository root directory.
        """
        self.project_root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).resolve().parents[2]
        )

        if loader is not None:
            self.loader = loader
            if preset is not None:
                self.loader.config = self.loader.config.with_preset(preset)
        else:
            self.loader = YOLOModelLoader(
                config_path=config if not isinstance(config, YOLOConfig) else None,
                config=config if isinstance(config, YOLOConfig) else None,
                preset=preset,
                project_root=self.project_root,
            )

    @property
    def config(self) -> YOLOConfig:
        """Access effective YOLO configuration."""
        return self.loader.config

    def validate_image(
        self,
        image_path: Union[str, Path],
    ) -> Tuple[Path, np.ndarray, int, int, int]:
        """Perform comprehensive validation and decoding on an input image.

        Args:
            image_path: Path to image file.

        Returns:
            Tuple[Path, np.ndarray, int, int, int]:
                (resolved_path, decoded_bgr_array, width, height, channels)

        Raises:
            ImageNotFoundError: If file does not exist on disk.
            ImageValidationError: If path is a directory, empty, or has invalid dimensions.
            UnsupportedImageFormatError: If file extension is unsupported.
            ImageDecodeError: If OpenCV fails to decode image bytes.
        """
        if not image_path:
            raise ImageValidationError("Image path cannot be empty or None.")

        raw_path = Path(image_path)
        # Resolve against project_root if relative and not found in cwd
        if raw_path.is_absolute():
            resolved = raw_path
        elif raw_path.exists():
            resolved = raw_path.resolve()
        else:
            resolved = (self.project_root / raw_path).resolve()

        # 1. Existence and file check
        if not resolved.exists():
            raise ImageNotFoundError(f"Input image file does not exist: '{resolved}'")

        if not resolved.is_file():
            raise ImageValidationError(f"Input image path is a directory, not a file: '{resolved}'")

        # 2. Extension validation
        ext = resolved.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedImageFormatError(
                f"Unsupported image extension '{resolved.suffix}'. Supported extensions: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        # 3. File size check
        try:
            file_size = resolved.stat().st_size
        except Exception as e:
            raise ImageValidationError(f"Failed to access image file metadata '{resolved}': {e}") from e

        if file_size == 0:
            raise ImageValidationError(f"Input image file is empty (0 bytes): '{resolved}'")

        # 4. Image Decoding (Safe in-memory decode avoiding OS file handle collisions)
        try:
            with open(resolved, "rb") as f:
                raw_bytes = f.read()
            if not raw_bytes:
                raise ImageValidationError(f"Input image file is empty (0 bytes): '{resolved}'")
            decoded = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        except ImageValidationError:
            raise
        except Exception as e:
            raise ImageDecodeError(f"Failed to read image file bytes from '{resolved}': {e}") from e

        if decoded is None or decoded.size == 0:
            raise ImageDecodeError(
                f"OpenCV failed to decode image '{resolved}'. File may be corrupted, truncated, or unreadable."
            )

        # 5. Dimension & Channel Validation
        shape = decoded.shape
        if len(shape) < 2:
            raise ImageValidationError(f"Invalid image array shape: {shape}")

        height, width = shape[0], shape[1]
        if height <= 0 or width <= 0:
            raise ImageValidationError(f"Image dimensions must be positive integers, got: {width}x{height}")

        if len(shape) == 2:
            # Grayscale -> Convert to 3-channel BGR for standardized YOLO processing
            decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
            channels = 3
        elif len(shape) == 3:
            channels = shape[2]
            if channels == 4:
                # BGRA -> Convert to BGR
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
                channels = 3
            elif channels != 3:
                raise ImageValidationError(f"Unsupported number of image channels: {channels}")
        else:
            raise ImageValidationError(f"Unsupported multidimensional image tensor shape: {shape}")

        return resolved, decoded, width, height, channels

    def detect(
        self,
        image_path: Union[str, Path],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        save_json: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> SingleImageDetectionResult:
        """Execute single-image object detection with complete validation and timing.

        Args:
            image_path: Path to input image file.
            conf: Optional confidence threshold override.
            iou: Optional IoU threshold override.
            imgsz: Optional image size override.
            classes: Optional class IDs filter override.
            save_json: Whether to persist machine-readable detection output.
            output_dir: Optional directory destination for saved JSON.
            output_path: Optional explicit filepath destination for saved JSON.
            overwrite: Whether to overwrite existing destination JSON.

        Returns:
            SingleImageDetectionResult: Structured detection response.

        Raises:
            ImageValidationError: If image input is invalid, missing, or corrupted.
            DetectionInferenceError: If model prediction fails.
            DetectionSerializationError: If output JSON saving fails.
        """
        t_start = time.perf_counter()

        # 1. Image Validation & Decoding
        t_decode_0 = time.perf_counter()
        resolved_path, img_bgr, width, height, channels = self.validate_image(image_path)
        t_decode_1 = time.perf_counter()
        image_decode_time_ms = (t_decode_1 - t_decode_0) * 1000.0

        # 2. Model Prediction Execution
        t_infer_0 = time.perf_counter()
        try:
            raw_result = self.loader.predict_single_frame(
                frame=img_bgr,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
            )
        except YOLOInferenceError as e:
            raise DetectionInferenceError(f"YOLO inference failed: {e}") from e
        except Exception as e:
            raise DetectionInferenceError(f"Unexpected error during YOLO inference: {e}") from e
        t_infer_1 = time.perf_counter()
        inference_time_ms = (t_infer_1 - t_infer_0) * 1000.0

        # 3. Detection Extraction & Calibration
        detections: List[DetectedObject] = []
        boxes = getattr(raw_result, "boxes", None)
        model_names: Dict[int, str] = getattr(raw_result, "names", {})

        if boxes is not None and len(boxes) > 0:
            xyxy_tensor = getattr(boxes, "xyxy", None)
            conf_tensor = getattr(boxes, "conf", None)
            cls_tensor = getattr(boxes, "cls", None)

            num_boxes = len(boxes)
            for idx in range(num_boxes):
                # Extract class ID
                c_val = cls_tensor[idx] if cls_tensor is not None else 0
                class_id = int(c_val.item() if hasattr(c_val, "item") else c_val)

                # Extract confidence score
                cf_val = conf_tensor[idx] if conf_tensor is not None else 0.0
                confidence = float(cf_val.item() if hasattr(cf_val, "item") else cf_val)

                # Extract bounding box coordinates [x1, y1, x2, y2]
                box_row = xyxy_tensor[idx] if xyxy_tensor is not None else [0.0, 0.0, 0.0, 0.0]
                if hasattr(box_row, "tolist"):
                    coords = [float(x) for x in box_row.tolist()]
                else:
                    coords = [float(x) for x in box_row]

                # Boundary and integrity validation
                x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
                    logger.warning("Skipping box %d with non-finite coordinates: %s", idx, coords)
                    continue

                # Clamp strictly within physical image bounds
                x1_c = max(0.0, min(float(width), x1))
                y1_c = max(0.0, min(float(height), y1))
                x2_c = max(0.0, min(float(width), x2))
                y2_c = max(0.0, min(float(height), y2))

                # Ensure non-degenerate box
                if x2_c <= x1_c or y2_c <= y1_c:
                    logger.warning("Skipping degenerate zero-area box %d: [%.1f, %.1f, %.1f, %.1f]", idx, x1_c, y1_c, x2_c, y2_c)
                    continue

                class_name = model_names.get(class_id, f"class_{class_id}")
                det_obj = DetectedObject(
                    detection_index=len(detections),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=round(confidence, 4),
                    bbox_xyxy=[round(x1_c, 1), round(y1_c, 1), round(x2_c, 1), round(y2_c, 1)],
                )
                detections.append(det_obj)

        # 4. Construct Detection Result
        effective_conf = conf if conf is not None else self.loader.confidence_threshold
        effective_iou = iou if iou is not None else self.loader.iou_threshold
        effective_imgsz = imgsz if imgsz is not None else self.loader.image_size

        result = SingleImageDetectionResult(
            schema_version="1.0",
            image_filename=resolved_path.name,
            image_path=str(resolved_path),
            width=width,
            height=height,
            channels=channels,
            model_name=self.loader.model_name,
            model_task=self.loader.task,
            model_device=self.loader.device,
            image_size_configured=effective_imgsz,
            confidence_threshold=effective_conf,
            iou_threshold=effective_iou,
            max_detections=self.loader.max_detections,
            image_decode_time_ms=image_decode_time_ms,
            inference_time_ms=inference_time_ms,
            serialization_time_ms=0.0,
            total_time_ms=(time.perf_counter() - t_start) * 1000.0,
            processing_status="success",
            detection_count=len(detections),
            detections=detections,
        )

        # 5. Machine-Readable Serialization (Optional)
        if save_json:
            t_serial_0 = time.perf_counter()
            dest_file: Path
            if output_path is not None:
                dest_file = Path(output_path)
            elif output_dir is not None:
                dest_file = Path(output_dir) / f"{resolved_path.stem}_detections.json"
            else:
                dest_file = (
                    self.project_root
                    / "datasets"
                    / "processed"
                    / "driving_sample"
                    / "detections"
                    / "single_image"
                    / f"{resolved_path.stem}_detections.json"
                )
            result.total_time_ms = (time.perf_counter() - t_start) * 1000.0
            result.serialization_time_ms = max(0.5, (time.perf_counter() - t_serial_0) * 1000.0)
            result.save_json(dest_file, overwrite=overwrite)
            # Re-record exact serialization duration
            result.serialization_time_ms = (time.perf_counter() - t_serial_0) * 1000.0
            result.total_time_ms = (time.perf_counter() - t_start) * 1000.0

        return result


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    """CLI runner for single image object detection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PerceptGuard AI – Single Image Object Detection Engine"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input driving frame or camera image (e.g., .jpg, .png).",
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
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold override in (0.0, 1.0].",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU threshold override in (0.0, 1.0].",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save machine-readable JSON detection results to disk.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store detection JSON.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Exact destination filepath for detection JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination JSON file if it already exists.",
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
        detector = SingleImageDetector(
            config=args.config,
            preset=args.preset,
        )
        res = detector.detect(
            image_path=args.image,
            conf=args.conf,
            iou=args.iou,
            save_json=args.save_json,
            output_dir=args.output_dir,
            output_path=args.output_file,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print("=" * 60)
        print("PERCEPTGUARD AI – SINGLE IMAGE DETECTION")
        print("=" * 60)
        print(f"ERROR: {e}")
        print("Final status: FAILED")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print("PERCEPTGUARD AI – SINGLE IMAGE DETECTION")
    print("=" * 60)
    print(f"Input image: {res.image_filename}")
    print(f"Image dimensions: {res.width}x{res.height} px ({res.channels} channels)")
    print(f"Model: {res.model_name}")
    print(f"Device: {res.model_device}")
    print(f"Confidence threshold: {res.confidence_threshold:.4f}")
    print(f"IoU threshold: {res.iou_threshold:.4f}")
    print("")
    print("Detected objects:")
    if res.detection_count == 0:
        print("  (None detected)")
    else:
        for d in res.detections:
            print(
                f"  - [{d.detection_index}] {d.class_name} (ID {d.class_id}): "
                f"conf={d.confidence:.4f}, box={d.bbox_xyxy}"
            )
    print("")
    print(f"Detection count: {res.detection_count}")
    print(f"Inference time: {res.inference_time_ms:.2f} ms")
    print(f"Total processing time: {res.total_time_ms:.2f} ms")
    if args.save_json:
        if args.output_file:
            out_p = Path(args.output_file).resolve()
        elif args.output_dir:
            out_p = (Path(args.output_dir) / f"{Path(args.image).stem}_detections.json").resolve()
        else:
            out_p = (
                Path(_project_root)
                / "datasets"
                / "processed"
                / "driving_sample"
                / "detections"
                / "single_image"
                / f"{Path(args.image).stem}_detections.json"
            ).resolve()
        print(f"Output JSON: {out_p}")
    print(f"Final status: {res.processing_status.upper()}")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
