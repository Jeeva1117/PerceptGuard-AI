"""PerceptGuard AI: Computer Vision Visualization Package.

Provides production-grade visualization engines for object detection,
multi-object tracking, and reliability forecasting telemetry.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

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
from src.visualization.tracking_visualizer import (
    FrameTrackingVisRecord,
    TrackAnnotation,
    TrackingJSONValidationError,
    TrackingSourceFrameDecodeError,
    TrackingSourceFrameNotFoundError,
    TrackingVideoGenerationError,
    TrackingVisualizationError,
    TrackingVisualizer,
    TrackingVisSummary,
    get_deterministic_track_color,
)

__all__ = [
    # Detection Visualization
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
    # Tracking Visualization
    "TrackingVisualizer",
    "TrackAnnotation",
    "FrameTrackingVisRecord",
    "TrackingVisSummary",
    "TrackingVisualizationError",
    "TrackingJSONValidationError",
    "TrackingSourceFrameNotFoundError",
    "TrackingSourceFrameDecodeError",
    "TrackingVideoGenerationError",
    "get_deterministic_track_color",
]
