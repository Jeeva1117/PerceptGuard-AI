"""Early Warning Interfaces and Risk Specifications for PerceptGuard AI.

Defines the core domain types, risk level enumerations, and abstract interfaces
for the Predictive Early Warning and Perception Risk Assessment system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from src.early_warning.early_warning_data_contract import EarlyWarningInputRecord


class RiskLevel(str, Enum):
    """Categorical perception failure risk classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, value: str) -> RiskLevel:
        """Parse string to RiskLevel enum with case-insensitive matching."""
        val = str(value).strip().upper()
        try:
            return cls(val)
        except ValueError:
            return cls.UNKNOWN

    @property
    def severity_rank(self) -> int:
        """Integer rank for ordinal severity comparisons (0 is lowest)."""
        ranks = {
            RiskLevel.UNKNOWN: -1,
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3,
        }
        return ranks.get(self, -1)


class AlertSeverity(str, Enum):
    """Severity classification for early warning alerts."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    TREND_WARNING = "TREND_WARNING"
    DATA_QUALITY_WARNING = "DATA_QUALITY_WARNING"

    @classmethod
    def from_string(cls, value: str) -> AlertSeverity:
        val = str(value).strip().upper()
        try:
            return cls(val)
        except ValueError:
            return cls.WARNING

    @property
    def priority_rank(self) -> int:
        ranks = {
            AlertSeverity.INFO: 0,
            AlertSeverity.DATA_QUALITY_WARNING: 1,
            AlertSeverity.TREND_WARNING: 2,
            AlertSeverity.WARNING: 3,
            AlertSeverity.CRITICAL: 4,
        }
        return ranks.get(self, 0)


class AlertType(str, Enum):
    """Operational trigger categorization of perception early warnings."""
    THRESHOLD_BREACH = "THRESHOLD_BREACH"
    RAPID_DEGRADATION = "RAPID_DEGRADATION"
    STATE_TRANSITION = "STATE_TRANSITION"
    DATA_QUALITY = "DATA_QUALITY"


class AlertLifecycleState(str, Enum):
    """State machine lifecycle status for actionable alerts."""
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"


@dataclass
class RiskScore:
    """Quantitative risk evaluation produced by the risk assessment engine."""
    sequence_id: str
    overall_risk_score: float  # In [0.0, 1.0] where 1.0 is highest risk
    primary_horizon: int
    risk_level: RiskLevel
    horizon_risk_scores: Dict[str, float] = field(default_factory=dict)
    predicted_reliabilities: Dict[str, float] = field(default_factory=dict)
    degradation_slope: Optional[float] = None
    target_timestamp_seconds: Optional[float] = None
    is_missing_data: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert RiskScore to dictionary."""
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        return d


@dataclass
class EarlyWarningAlert:
    """Actionable early warning alert generated when perception risk exceeds thresholds."""
    alert_id: str
    sequence_id: str
    risk_level: RiskLevel
    risk_score: float
    trigger_horizon: int
    source_frame_id: str
    target_frame_id: str
    target_timestamp_seconds: Optional[float]
    warning_message: str
    recommended_action: str
    is_suppressed: bool = False
    severity: AlertSeverity = AlertSeverity.WARNING
    alert_type: AlertType = AlertType.THRESHOLD_BREACH
    lifecycle_state: AlertLifecycleState = AlertLifecycleState.NEW
    video_id: str = "driving_sample"
    frame_id: str = ""
    timestamp: float = 0.0
    generated_at: Optional[str] = None
    forecast_horizon: Optional[int] = None
    risk_category: Optional[RiskLevel] = None
    trigger_reason: str = ""
    risk_trend: str = "UNKNOWN"
    previous_category: Optional[RiskLevel] = None
    current_category: Optional[RiskLevel] = None
    calibration_status: str = "PROVISIONAL"
    model_version: str = "LSTMForecaster"
    rule_id: str = "RULE_DEFAULT"
    acknowledgement_status: str = "UNACKNOWLEDGED"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def __post_init__(self):
        if not self.frame_id:
            self.frame_id = self.source_frame_id
        if self.timestamp == 0.0 and self.target_timestamp_seconds is not None:
            self.timestamp = self.target_timestamp_seconds
        if self.forecast_horizon is None:
            self.forecast_horizon = self.trigger_horizon
        if self.risk_category is None:
            self.risk_category = self.risk_level
        if self.current_category is None:
            self.current_category = self.risk_level
        if not self.trigger_reason:
            self.trigger_reason = self.warning_message
        if not self.generated_at:
            self.generated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert EarlyWarningAlert to dictionary."""
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        d["severity"] = self.severity.value
        d["alert_type"] = self.alert_type.value
        d["lifecycle_state"] = self.lifecycle_state.value
        if self.risk_category:
            d["risk_category"] = self.risk_category.value
        if self.current_category:
            d["current_category"] = self.current_category.value
        if self.previous_category:
            d["previous_category"] = self.previous_category.value
        return d


class IRiskAssessmentEngine(ABC):
    """Abstract interface for multi-horizon perception risk assessment engines."""

    @abstractmethod
    def evaluate_risk(
        self,
        predictions: Sequence[EarlyWarningInputRecord],
        history: Optional[Sequence[EarlyWarningInputRecord]] = None,
    ) -> RiskScore:
        """Compute quantitative risk score from predicted reliability and temporal features."""
        pass


class IRiskClassifier(ABC):
    """Abstract interface for mapping continuous risk scores to discrete RiskLevels."""

    @abstractmethod
    def classify(self, risk_score: float) -> RiskLevel:
        """Classify a raw risk score into a discrete RiskLevel."""
        pass


class ITemporalRiskMonitor(ABC):
    """Abstract interface for temporal persistence tracking, debouncing, and trend analysis."""

    @abstractmethod
    def record_and_check(self, risk_score: RiskScore) -> bool:
        """Record risk score in sliding temporal window and determine if alert condition holds."""
        pass

    @abstractmethod
    def get_recent_scores(self, sequence_id: str, window_size: int) -> List[RiskScore]:
        """Retrieve recent risk scores for a given sequence within the temporal window."""
        pass


class IEarlyWarningGenerator(ABC):
    """Abstract interface for creating, debouncing, and formatting actionable early warning alerts."""

    @abstractmethod
    def generate_alert(
        self,
        risk_score: RiskScore,
        input_record: EarlyWarningInputRecord,
    ) -> Optional[EarlyWarningAlert]:
        """Generate structured alert if risk score qualifies and passes debounce/cooldown rules."""
        pass


class IEarlyWarningDashboard(ABC):
    """Abstract interface for rendering early warning indicators and alert feeds."""

    @abstractmethod
    def display_status(
        self,
        current_risk: RiskScore,
        active_alerts: Sequence[EarlyWarningAlert],
    ) -> Dict[str, Any]:
        """Render current perception health status and active alert notifications."""
        pass
