"""PerceptGuard AI: Feature Engineering Package.

Provides strongly typed configuration managers, schema definitions, input validation engines,
and environment verification utilities for multidimensional feature vector extraction
and temporal sequence construction.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

from src.features.feature_config import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    FEATURE_PRESETS,
    FeatureConfig,
    FeatureConfigError,
    FeatureConfigFileNotFoundError,
    InvalidFeatureConfigError,
)
from src.features.feature_schema import (
    FEATURE_FIELD_REGISTRY,
    FEATURE_SCHEMA_VERSION,
    FeatureFieldDescriptor,
    FeatureRecord,
    FeatureSchemaDefinition,
    FeatureSchemaError,
    IncompatibleArtifactError,
    InvalidFeatureRecordError,
    MissingRequiredFieldError,
    ValidationResult,
    validate_feature_record,
    validate_input_artifacts,
)
from src.features.feature_environment import (
    FeatureEnvironmentReport,
    FeatureEnvironmentVerifier,
)
from src.features.detection_feature_extractor import (
    DetectionExtractionErrorRecord,
    DetectionFeatureExtractionError,
    DetectionFeatureExtractor,
    DetectionFeatureRecord,
    DetectionFeatureSummary,
    FrameDetectionFeatureRecord,
    categorize_confidence,
)
from src.features.motion_feature_extractor import (
    FrameMotionFeatureRecord,
    MotionExtractionErrorRecord,
    MotionFeatureExtractionError,
    MotionFeatureExtractor,
    MotionFeatureRecord,
    MotionFeatureSummary,
    TrackMotionSummaryRecord,
    calculate_direction,
)
from src.features.visual_quality_extractor import (
    DEFAULT_BLUR_THRESHOLD_BLURRY,
    DEFAULT_BLUR_THRESHOLD_SHARP,
    DEFAULT_BLUR_THRESHOLDS,
    DEFAULT_BRIGHTNESS_THRESHOLD_BRIGHT,
    DEFAULT_BRIGHTNESS_THRESHOLD_DARK,
    DEFAULT_BRIGHTNESS_THRESHOLDS,
    DEFAULT_CONTRAST_THRESHOLD_HIGH,
    DEFAULT_CONTRAST_THRESHOLD_LOW,
    DEFAULT_CONTRAST_THRESHOLDS,
    CorruptedImageError,
    CorruptedImageFileError,
    EmptyImageFileError,
    FileNotFoundExtractionError,
    FrameVisualQualityRecord,
    InvalidImageDimensionError,
    OutputDirectoryExistsError,
    UnsupportedImageFormatError,
    VisualQualityErrorRecord,
    VisualQualityExtractionError,
    VisualQualityExtractor,
    VisualQualitySummary,
    classify_blur,
    classify_brightness,
    classify_contrast,
)

__all__ = [
    # Configuration
    "FeatureConfig",
    "FeatureConfigError",
    "FeatureConfigFileNotFoundError",
    "InvalidFeatureConfigError",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "FEATURE_PRESETS",
    # Schema & Contracts
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_FIELD_REGISTRY",
    "FeatureFieldDescriptor",
    "FeatureSchemaDefinition",
    "FeatureRecord",
    "ValidationResult",
    "FeatureSchemaError",
    "InvalidFeatureRecordError",
    "MissingRequiredFieldError",
    "IncompatibleArtifactError",
    "validate_feature_record",
    "validate_input_artifacts",
    # Environment Verification
    "FeatureEnvironmentVerifier",
    "FeatureEnvironmentReport",
    # Detection Feature Extraction
    "DetectionFeatureExtractor",
    "DetectionFeatureRecord",
    "FrameDetectionFeatureRecord",
    "DetectionFeatureSummary",
    "DetectionExtractionErrorRecord",
    "DetectionFeatureExtractionError",
    "categorize_confidence",
    # Motion Feature Extraction
    "MotionFeatureExtractor",
    "MotionFeatureRecord",
    "TrackMotionSummaryRecord",
    "FrameMotionFeatureRecord",
    "MotionFeatureSummary",
    "MotionExtractionErrorRecord",
    "MotionFeatureExtractionError",
    "calculate_direction",
    # Visual Quality Feature Extraction
    "VisualQualityExtractor",
    "FrameVisualQualityRecord",
    "VisualQualitySummary",
    "VisualQualityErrorRecord",
    "VisualQualityExtractionError",
    "CorruptedImageError",
    "CorruptedImageFileError",
    "EmptyImageFileError",
    "FileNotFoundExtractionError",
    "InvalidImageDimensionError",
    "OutputDirectoryExistsError",
    "UnsupportedImageFormatError",
    "classify_blur",
    "classify_brightness",
    "classify_contrast",
    "DEFAULT_BLUR_THRESHOLDS",
    "DEFAULT_BLUR_THRESHOLD_SHARP",
    "DEFAULT_BLUR_THRESHOLD_BLURRY",
    "DEFAULT_BRIGHTNESS_THRESHOLDS",
    "DEFAULT_BRIGHTNESS_THRESHOLD_DARK",
    "DEFAULT_BRIGHTNESS_THRESHOLD_BRIGHT",
    "DEFAULT_CONTRAST_THRESHOLDS",
    "DEFAULT_CONTRAST_THRESHOLD_LOW",
    "DEFAULT_CONTRAST_THRESHOLD_HIGH",
]
