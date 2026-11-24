"""PerceptGuard AI - Safety Recommendation & Early Warning Schemas (Feature 5).

Defines strongly-typed contracts for decision-support early warnings, vehicle stop
recommendations, supporting evidence, priority classifications, and stream summaries.

IMPORTANT SAFETY NOTICE:
PerceptGuard AI is an experimental ADAS decision-support prototype.
All recommendations are advisory only ("when safe to do so") and must never be connected
to direct vehicle actuation or CAN bus commands. Operational safety remains the driver's responsibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class RecommendationState(str, Enum):
    """The 6 fundamental safety advisory states specified for Feature 5."""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    REDUCE_SPEED = "REDUCE SPEED"
    PREPARE_TO_STOP = "PREPARE TO STOP"
    STOP_RECOMMENDED = "STOP RECOMMENDED"
    UNKNOWN = "UNKNOWN"


class WarningPriority(str, Enum):
    """Warning priority hierarchy."""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    ADVISORY = "ADVISORY"
    INFO = "INFO"


class WarningStatus(str, Enum):
    """Lifecycle status of an early warning recommendation."""
    ACTIVE = "ACTIVE"
    SUPPRESSED = "SUPPRESSED"
    EXPIRED = "EXPIRED"
    DEGRADED = "DEGRADED"


@dataclass
class InvolvedObject:
    """Road agent involved in a potential conflict or risk interaction."""
    track_id: int
    object_class: str
    distance_m: Optional[float] = None
    speed_kmh: Optional[float] = None
    relative_speed_kmh: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "object_class": self.object_class,
            "distance_m": round(self.distance_m, 2) if self.distance_m is not None else None,
            "speed_kmh": round(self.speed_kmh, 1) if self.speed_kmh is not None else None,
            "relative_speed_kmh": round(self.relative_speed_kmh, 1) if self.relative_speed_kmh is not None else None,
        }


@dataclass
class SupportingEvidence:
    """Multi-source sensor and perception evidence substantiating a recommendation."""
    trajectory_conflict: bool = False
    conflict_type: Optional[str] = None
    min_ttc_seconds: Optional[float] = None
    closest_distance_m: Optional[float] = None
    prediction_horizon_seconds: Optional[float] = None
    camera_visibility_score: float = 100.0
    camera_condition: str = "CLEAR"
    detection_reliability: float = 1.0
    uncertainty_level: str = "LOW"
    uncertainty_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_conflict": self.trajectory_conflict,
            "conflict_type": self.conflict_type,
            "min_ttc_seconds": round(self.min_ttc_seconds, 2) if self.min_ttc_seconds is not None else None,
            "closest_distance_m": round(self.closest_distance_m, 2) if self.closest_distance_m is not None else None,
            "prediction_horizon_seconds": (
                round(self.prediction_horizon_seconds, 2)
                if self.prediction_horizon_seconds is not None
                else None
            ),
            "camera_visibility_score": round(self.camera_visibility_score, 1),
            "camera_condition": self.camera_condition,
            "detection_reliability": round(self.detection_reliability, 3),
            "uncertainty_level": self.uncertainty_level,
            "uncertainty_score": round(self.uncertainty_score, 3),
        }


@dataclass
class SafetyRecommendation:
    """Actionable driver / ADAS safety advisory recommendation (Feature 5)."""
    warning_id: str
    timestamp_seconds: float
    recommendation_state: RecommendationState
    priority: WarningPriority
    risk_level: str  # LOW, MODERATE, HIGH, CRITICAL, UNKNOWN
    objects_involved: List[InvolvedObject] = field(default_factory=list)
    recommended_action: str = "Maintain nominal speed and road vigilance."
    primary_reason: str = "No immediate conflict detected in trajectory corridor."
    supporting_evidence: SupportingEvidence = field(default_factory=SupportingEvidence)
    prediction_horizon: str = "N/A"
    min_ttc_seconds: Optional[float] = None
    relative_speed_kmh: Optional[float] = None
    camera_visibility_status: str = "Nominal"
    detection_reliability: float = 1.0
    uncertainty_level: str = "LOW"
    status: WarningStatus = WarningStatus.ACTIVE
    disclaimer: str = (
        "PerceptGuard AI is an experimental ADAS prototype. Operational safety remains the "
        "driver's responsibility. Stop recommendations are advisory only (when safe to do so) "
        "and do not command vehicle actuation."
    )

    # Backward-compatible fields matching earlier contracts
    @property
    def primary_action(self) -> str:
        """Map RecommendationState to legacy action strings for backward compatibility."""
        mapping = {
            RecommendationState.STOP_RECOMMENDED: "STOP_RECOMMENDATION",
            RecommendationState.PREPARE_TO_STOP: "PREPARE_TO_STOP",
            RecommendationState.REDUCE_SPEED: "REDUCE_SPEED",
            RecommendationState.CAUTION: "REDUCE_SPEED",
            RecommendationState.NORMAL: "MAINTAIN_SPEED",
            RecommendationState.UNKNOWN: "MAINTAIN_SPEED",
        }
        return mapping.get(self.recommendation_state, "MAINTAIN_SPEED")

    @property
    def urgency_level(self) -> str:
        """Map WarningPriority to urgency string."""
        return self.priority.value

    @property
    def evidence_chain(self) -> List[str]:
        """Convert structured evidence into human-readable list of explanatory bullet points."""
        chain: List[str] = [self.primary_reason]
        if self.supporting_evidence.min_ttc_seconds is not None:
            chain.append(f"Estimated Time-to-Collision: {self.supporting_evidence.min_ttc_seconds:.1f}s")
        if self.supporting_evidence.closest_distance_m is not None:
            chain.append(f"Closest projected obstacle separation: {self.supporting_evidence.closest_distance_m:.1f}m")
        if self.supporting_evidence.camera_visibility_score < 70.0:
            chain.append(
                f"Camera visibility degraded ({self.supporting_evidence.camera_visibility_score:.0f}%, "
                f"{self.supporting_evidence.camera_condition})"
            )
        if self.supporting_evidence.uncertainty_score > 0.40:
            chain.append(
                f"Elevated perception uncertainty: {self.supporting_evidence.uncertainty_level} "
                f"({self.supporting_evidence.uncertainty_score*100:.0f}%)"
            )
        if self.objects_involved:
            objs_str = ", ".join(f"Track #{o.track_id} ({o.object_class})" for o in self.objects_involved)
            chain.append(f"Active road agents involved: {objs_str}")
        return chain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warning_id": self.warning_id,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "recommendation_state": self.recommendation_state.value,
            "priority": self.priority.value,
            "risk_level": self.risk_level,
            "objects_involved": [o.to_dict() for o in self.objects_involved],
            "recommended_action": self.recommended_action,
            "primary_reason": self.primary_reason,
            "supporting_evidence": self.supporting_evidence.to_dict(),
            "prediction_horizon": self.prediction_horizon,
            "min_ttc_seconds": round(self.min_ttc_seconds, 2) if self.min_ttc_seconds is not None else None,
            "relative_speed_kmh": round(self.relative_speed_kmh, 1) if self.relative_speed_kmh is not None else None,
            "camera_visibility_status": self.camera_visibility_status,
            "detection_reliability": round(self.detection_reliability, 3),
            "uncertainty_level": self.uncertainty_level,
            "status": self.status.value,
            "disclaimer": self.disclaimer,
            # Legacy fields for complete backward compatibility
            "primary_action": self.primary_action,
            "urgency_level": self.urgency_level,
            "evidence_chain": self.evidence_chain,
        }


@dataclass
class SafetyStreamSummary:
    """Aggregated stream-level summary across a complete video processing run."""
    total_recommendations: int = 0
    recommendation_counts: Dict[str, int] = field(default_factory=dict)
    priority_counts: Dict[str, int] = field(default_factory=dict)
    most_frequent_recommendation: str = "NORMAL"
    critical_stop_count: int = 0
    state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    action_counts: Dict[str, int] = field(default_factory=dict)
    most_frequent_action: str = "MAINTAIN_SPEED"
    prototype_disclaimer: str = (
        "PerceptGuard AI is an experimental ADAS prototype. Operational safety remains the "
        "driver's responsibility. Stop recommendations are advisory only (when safe to do so) "
        "and do not command vehicle actuation."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_recommendations": self.total_recommendations,
            "recommendation_counts": self.recommendation_counts,
            "priority_counts": self.priority_counts,
            "most_frequent_recommendation": self.most_frequent_recommendation,
            "critical_stop_count": self.critical_stop_count,
            "state_transitions": self.state_transitions,
            "action_counts": self.action_counts,
            "most_frequent_action": self.most_frequent_action,
            "prototype_disclaimer": self.prototype_disclaimer,
        }
