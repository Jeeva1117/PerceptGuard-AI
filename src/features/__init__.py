"""PerceptGuard AI: Feature Engineering Package.

Provides strongly typed configuration managers, schema definitions, input validation engines,
and environment verification utilities for multidimensional feature vector extraction
and temporal sequence construction.
"""

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
]
