"""PerceptGuard-AI: Object Detection Package.

Integrates real-time object detection models (e.g., YOLO11) to identify
road obstacles, vehicles, pedestrians, and cyclists across video frames.
Extracts bounding boxes, detection confidences, and object classes.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

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
]
