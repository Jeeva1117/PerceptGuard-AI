"""PerceptGuard-AI: Object Detection Package.

Integrates real-time object detection models (e.g., YOLO11) to identify
road obstacles, vehicles, pedestrians, and cyclists across video frames.
Extracts bounding boxes, detection confidences, and object classes.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

from src.detection.yolo_model_loader import (
    YOLOConfigError,
    YOLOEnvironmentError,
    YOLOInferenceError,
    YOLOModelInfo,
    YOLOModelLoadError,
    YOLOModelLoader,
    YOLOModelNotFoundError,
    verify_yolo_environment,
)

__all__ = [
    "YOLOModelLoader",
    "YOLOModelInfo",
    "YOLOEnvironmentError",
    "YOLOConfigError",
    "YOLOModelNotFoundError",
    "YOLOModelLoadError",
    "YOLOInferenceError",
    "verify_yolo_environment",
]
