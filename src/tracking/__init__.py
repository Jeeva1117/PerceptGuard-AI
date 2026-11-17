"""PerceptGuard AI: Multi-Object Tracking Architecture.

This package provides multi-object tracking configurations, environment verification,
and upcoming tracking engines (ByteTrack) to maintain persistent object identities,
compute kinematic motion features, and establish reliability metrics.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

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

__all__ = [
    "TrackerConfig",
    "TrackingEnvironmentError",
    "TrackerConfigError",
    "TrackerConfigFileNotFoundError",
    "SUPPORTED_TRACKER_TYPES",
    "REQUIRED_TRACKER_FIELDS",
    "TRACKER_PRESETS",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "EnvironmentReport",
    "TrackingEnvironmentVerifier",
    "verify_tracking_environment",
    "format_report",
]
