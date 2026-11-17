"""PerceptGuard-AI: Object Detection Package.

Integrates real-time object detection models (e.g., YOLO11) to identify
road obstacles, vehicles, pedestrians, and cyclists across video frames.
Extracts bounding boxes, detection confidences, and object classes.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

from src.detection.batch_detector import (
    BatchDetectionError,
    BatchDetector,
    BatchErrorRecord,
    BatchExecutionError,
    BatchSummary,
    DatasetEmptyError,
    DatasetNotFoundError,
    FrameDetectionRecord,
    MetadataParsingError,
)
from src.detection.single_image_detector import (
    DetectedObject,
    DetectionInferenceError,
    DetectionSerializationError,
    ImageDecodeError,
    ImageNotFoundError,
    ImageValidationError,
    SingleImageDetectionError,
    SingleImageDetectionResult,
    SingleImageDetector,
    UnsupportedImageFormatError,
)
from src.detection.yolo_config import (
    PRESETS,
    InferenceConfig,
    ModelConfig,
    PerformanceConfig,
    RuntimeConfig,
    ValidationConfig,
    YOLOConfig,
    YOLOConfigError,
    YOLOEnvironmentError,
    YOLOModelNotFoundError,
    load_yolo_config,
)
from src.detection.yolo_model_loader import (
    YOLOInferenceError,
    YOLOModelInfo,
    YOLOModelLoadError,
    YOLOModelLoader,
    verify_yolo_environment,
)

__all__ = [
    # Batch Object Detection Engine & Contracts
    "BatchDetector",
    "BatchSummary",
    "FrameDetectionRecord",
    "BatchErrorRecord",
    # Single Image Detection Engine & Contracts
    "SingleImageDetector",
    "SingleImageDetectionResult",
    "DetectedObject",
    # Model Loading Engine & Contracts
    "YOLOModelLoader",
    "YOLOModelInfo",
    "verify_yolo_environment",
    # Configuration Architecture
    "YOLOConfig",
    "ModelConfig",
    "RuntimeConfig",
    "InferenceConfig",
    "PerformanceConfig",
    "ValidationConfig",
    "PRESETS",
    "load_yolo_config",
    # Domain Exceptions
    "YOLOEnvironmentError",
    "YOLOConfigError",
    "YOLOModelNotFoundError",
    "YOLOModelLoadError",
    "YOLOInferenceError",
    "SingleImageDetectionError",
    "ImageValidationError",
    "ImageNotFoundError",
    "UnsupportedImageFormatError",
    "ImageDecodeError",
    "DetectionInferenceError",
    "DetectionSerializationError",
    "BatchDetectionError",
    "DatasetNotFoundError",
    "DatasetEmptyError",
    "MetadataParsingError",
    "BatchExecutionError",
]
