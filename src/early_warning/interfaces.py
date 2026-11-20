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
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert EarlyWarningAlert to dictionary."""
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
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
