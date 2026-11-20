"""Domain models and data containers for PerceptGuard AI Risk Scoring Engine.

Defines standardized data structures for single-horizon risk assessments,
multi-horizon aggregations, and retrospective temporal risk features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class HorizonRiskRecord:
    """Individual perception failure risk assessment for a specific forward horizon."""
    prediction_id: str
    sequence_id: str
    source_frame_id: str
    target_frame_id: str
    horizon: int
    predicted_reliability: float
    risk_score: float  # In [0.0, 1.0], where 1.0 indicates highest risk
    source_frame_index: Optional[int] = None
    target_frame_index: Optional[int] = None
    target_timestamp_seconds: Optional[float] = None
    track_id: Optional[int] = None
    model_name: str = "LSTMForecaster"
    checkpoint_id: str = "unknown"
    experiment_id: Optional[str] = None
    transformation_method: str = "linear_inverse"
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


@dataclass
class TemporalRiskFeatures:
    """Retrospective temporal risk and degradation features extracted from history."""
    sequence_id: str
    current_predicted_reliability: float
    horizon_reliabilities: Dict[int, float]
    cross_horizon_spread: Optional[float] = None  # R(H1) - R(H5)
    is_monotonic_degradation: bool = False  # R(H1) >= R(H3) >= R(H5)
    historical_trend_slope: Optional[float] = None  # dR/dt from past observations
    degradation_rate: Optional[float] = None  # Max drop rate across horizons
    recent_volatility: Optional[float] = None  # Std dev of recent reliability scores
    available_history_points: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary."""
        return asdict(self)


@dataclass
class AggregatedRiskRecord:
    """Synthesized multi-horizon risk evaluation for a temporal sequence."""
    sequence_id: str
    source_frame_id: str
    aggregated_risk_score: float  # In [0.0, 1.0]
    aggregation_strategy: str
    primary_horizon: int
    horizon_risk_scores: Dict[int, float]
    horizon_reliabilities: Dict[int, float]
    weights_applied: Dict[int, float]
    target_timestamp_seconds: Optional[float] = None
    source_frame_index: Optional[int] = None
    track_id: Optional[int] = None
    temporal_features: Optional[TemporalRiskFeatures] = None
    is_calibrated_probability: bool = False  # Explicitly states this is not a calibrated probability
    model_name: str = "LSTMForecaster"
    checkpoint_id: str = "unknown"
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregated record to dictionary."""
        data = asdict(self)
        if self.temporal_features:
            data["temporal_features"] = self.temporal_features.to_dict()
        return data
