"""PerceptGuard AI — Perception Reliability Risk Scoring Engine.

Transforms multi-horizon reliability forecasts into mathematically bounded,
traceable perception failure risk scores and temporal degradation features.
"""

from src.risk.risk_config import (
    AggregationConfig,
    RiskConfig,
    RiskConfigError,
    TemporalFeatureConfig,
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

__all__ = [
    # Configuration
    "RiskConfig",
    "RiskConfigError",
    "TransformationConfig",
    "AggregationConfig",
    "TemporalFeatureConfig",
    # Data Models
    "HorizonRiskRecord",
    "AggregatedRiskRecord",
    "TemporalRiskFeatures",
    # Validation
    "RiskValidationError",
    "validate_reliability_value",
    "validate_risk_value",
    "validate_horizon",
    "validate_aggregation_weights",
    # Feature Extraction
    "TemporalRiskFeatureExtractor",
    # Core Engine
    "ReliabilityRiskScorer",
    "RiskScoringError",
]
