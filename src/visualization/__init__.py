"""PerceptGuard AI: Computer Vision Visualization Package.

Provides production-grade visualization engines for object detection,
multi-object tracking, and reliability forecasting telemetry.
"""

from src.visualization.detection_visualizer import (
    DetectionJSONValidationError,
    DetectionVisualizer,
    FrameVisualizationRecord,
    InvalidBoundingBoxError,
    OutputPersistenceError,
    SourceImageDecodeError,
    SourceImageNotFoundError,
    UnsupportedSchemaVersionError,
    VisualizationError,
    VisualizationErrorRecord,
    VisualizationSummary,
)

__all__ = [
    "DetectionVisualizer",
    "VisualizationSummary",
    "FrameVisualizationRecord",
    "VisualizationErrorRecord",
    "VisualizationError",
    "DetectionJSONValidationError",
    "UnsupportedSchemaVersionError",
    "SourceImageNotFoundError",
    "SourceImageDecodeError",
    "InvalidBoundingBoxError",
    "OutputPersistenceError",
]
