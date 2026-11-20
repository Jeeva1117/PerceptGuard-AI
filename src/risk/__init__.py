"""PerceptGuard AI — Perception Reliability Risk Scoring Engine.

Transforms multi-horizon reliability forecasts into mathematically bounded,
traceable perception failure risk scores and temporal degradation features.
"""

from src.risk.risk_classifier import ClassificationResult, RiskClassifier
from src.risk.risk_config import (
    AggregationConfig,
    ClassificationConfig,
    RiskConfig,
    RiskConfigError,
    TemporalFeatureConfig,
    ThresholdProfile,
    TransformationConfig,
)
from src.risk.risk_features import TemporalRiskFeatureExtractor
from src.risk.risk_models import (
    AggregatedRiskRecord,
    HorizonRiskRecord,
    TemporalRiskFeatures,
)
from src.risk.risk_scorer import ReliabilityRiskScorer, RiskScoringError
from src.risk.risk_validator import (
    RiskValidationError,
    validate_aggregation_weights,
    validate_horizon,
    validate_reliability_value,
    validate_risk_value,
)
from src.risk.threshold_calibrator import (
    CalibrationError,
    CalibrationReport,
    CalibrationSample,
    ThresholdCalibrator,
    ThresholdCandidateEvaluation,
)

__all__ = [
    # Configuration
    "RiskConfig",
    "RiskConfigError",
    "TransformationConfig",
    "AggregationConfig",
    "TemporalFeatureConfig",
    "ThresholdProfile",
    "ClassificationConfig",
    # Data Models
    "HorizonRiskRecord",
    "AggregatedRiskRecord",
    "TemporalRiskFeatures",
    "ClassificationResult",
    "CalibrationSample",
    "ThresholdCandidateEvaluation",
    "CalibrationReport",
    # Validation
    "RiskValidationError",
    "validate_reliability_value",
    "validate_risk_value",
    "validate_horizon",
    "validate_aggregation_weights",
    # Feature Extraction
    "TemporalRiskFeatureExtractor",
    # Core Engine & Classification
    "ReliabilityRiskScorer",
    "RiskScoringError",
    "RiskClassifier",
    # Calibration
    "ThresholdCalibrator",
    "CalibrationError",
]
