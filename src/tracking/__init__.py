"""PerceptGuard AI: Multi-Object Tracking Architecture.

This package provides multi-object tracking configurations, environment verification,
and production ByteTrack integration to maintain persistent object identities,
compute kinematic motion features, and establish reliability metrics.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

from src.tracking.bytetrack_engine import (
    ByteTrackEngine,
    InvalidDetectionInputError,
    TrackRecord,
    TrackingEngineError,
)
from src.tracking.single_object_tracker import (
    AmbiguousTargetSelectionError,
    InvalidTargetIdError,
    SingleObjectTracker,
    SingleObjectTrackingError,
    SingleObjectTrackingState,
    SingleObjectTrackingSummary,
    TargetNotFoundError,
    TargetObservation,
    TargetSelectionRule,
)
from src.tracking.tracker_config import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    REQUIRED_TRACKER_FIELDS,
    SUPPORTED_TRACKER_TYPES,
    TRACKER_PRESETS,
    TrackerConfig,
    TrackerConfigError,
    TrackerConfigFileNotFoundError,
    TrackingEnvironmentError,
)
from src.tracking.tracking_environment import (
    EnvironmentReport,
    TrackingEnvironmentVerifier,
    format_report,
    verify_tracking_environment,
)
from src.tracking.tracking_pipeline import (
    FrameTrackingRecord,
    TrackingInputNotFoundError,
    TrackingOutputExistsError,
    TrackingPipeline,
    TrackingPipelineError,
    TrackingSummary,
)

__all__ = [
    # Configuration Architecture
    "TrackerConfig",
    "TrackingEnvironmentError",
    "TrackerConfigError",
    "TrackerConfigFileNotFoundError",
    "SUPPORTED_TRACKER_TYPES",
    "REQUIRED_TRACKER_FIELDS",
    "TRACKER_PRESETS",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    # Environment Verification
    "EnvironmentReport",
    "TrackingEnvironmentVerifier",
    "verify_tracking_environment",
    "format_report",
    # ByteTrack Engine & Data Contracts
    "ByteTrackEngine",
    "TrackRecord",
    "TrackingEngineError",
    "InvalidDetectionInputError",
    # Tracking Pipeline & Multi-File Persistence
    "TrackingPipeline",
    "FrameTrackingRecord",
    "TrackingSummary",
    "TrackingPipelineError",
    "TrackingInputNotFoundError",
    "TrackingOutputExistsError",
    # Single-Object Tracking Architecture
    "SingleObjectTracker",
    "SingleObjectTrackingState",
    "TargetSelectionRule",
    "TargetObservation",
    "SingleObjectTrackingSummary",
    "SingleObjectTrackingError",
    "TargetNotFoundError",
    "AmbiguousTargetSelectionError",
    "InvalidTargetIdError",
]
