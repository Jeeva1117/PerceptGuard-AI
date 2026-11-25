"""Typed schemas and data contracts for Collision & Road Risk Assessment (Feature 4).

Adheres to strict autonomous safety engineering principles:
- Discrete 5-state risk classification: LOW, MODERATE, HIGH, CRITICAL, UNKNOWN.
- Clear separation between image-space pixels and flat-earth ground metric coordinates.
- Spatio-temporal trajectory conflict validation.
- Explainable contributing factors and uncertainty indicators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple


class CollisionRiskState(str, Enum):
    """Discrete 5-state automotive collision risk scale."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class CollisionRiskDecision(str, Enum):
    """The five definitive product collision risk states for PerceptGuard-AI."""
    NO_COLLISION_RISK = "NO COLLISION RISK"
    NEAR_PASS = "NEAR PASS"
    POTENTIAL_COLLISION_RISK = "POTENTIAL COLLISION RISK"
    COLLISION_OBSERVED = "COLLISION OBSERVED"
    UNKNOWN = "UNKNOWN"


class ConflictType(str, Enum):
    """Categorization of geometric trajectory convergence."""
    REAR_END = "REAR_END"
    CROSSING = "CROSSING"
    HEAD_ON = "HEAD_ON"
    MERGING = "MERGING"
    PEDESTRIAN_CROSSING = "PEDESTRIAN_CROSSING"
    CYCLIST_CONFLICT = "CYCLIST_CONFLICT"
    UNCATEGORIZED = "UNCATEGORIZED"


class UncertaintyLevel(str, Enum):
    """Overall uncertainty rating influencing risk defensibility."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AssessmentStatus(str, Enum):
    """Operational integrity of the assessment."""
    VALID = "VALID"
    UNKNOWN_INSUFFICIENT_DATA = "UNKNOWN_INSUFFICIENT_DATA"
    UNKNOWN_HIGH_UNCERTAINTY = "UNKNOWN_HIGH_UNCERTAINTY"
    DEGRADED = "DEGRADED"


@dataclass
class TrajectoryConflict:
    """Pairwise spatio-temporal conflict prediction between two road agents (or ego vehicle)."""
    track_a: int
    track_b: int
    class_a: str
    class_b: str
    conflict_type: ConflictType
    time_to_conflict_seconds: float
    horizon_label: str  # e.g., 'H1', 'H3', 'H5'
    separation_at_conflict_px: float
    separation_at_conflict_m: Optional[float] = None
    conflict_point_px: Tuple[float, float] = (0.0, 0.0)
    conflict_point_m: Optional[Tuple[float, float]] = None
    is_spatiotemporal_overlap: bool = True
    combined_uncertainty_px: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_a": self.track_a,
            "track_b": self.track_b,
            "class_a": self.class_a,
            "class_b": self.class_b,
            "conflict_type": self.conflict_type.value,
            "time_to_conflict_seconds": round(self.time_to_conflict_seconds, 2),
            "horizon_label": self.horizon_label,
            "separation_pixels": round(self.separation_at_conflict_px, 1),
            "separation_meters": round(self.separation_at_conflict_m, 2) if self.separation_at_conflict_m is not None else None,
            "conflict_point": [round(self.conflict_point_px[0], 1), round(self.conflict_point_px[1], 1)],
            "conflict_point_m": [round(self.conflict_point_m[0], 2), round(self.conflict_point_m[1], 2)] if self.conflict_point_m is not None else None,
            "is_spatiotemporal_overlap": self.is_spatiotemporal_overlap,
            "combined_uncertainty_px": round(self.combined_uncertainty_px, 1),
        }


@dataclass
class CollisionRiskAssessment:
    """Complete explainable risk assessment payload for a single frame."""
    assessment_id: str
    timestamp_seconds: float
    risk_state: CollisionRiskState
    risk_score: Optional[float]  # 0.0 to 1.0, or None if UNKNOWN
    primary_contributing_factors: List[str]
    objects_involved: List[Dict[str, Any]]
    min_ttc_seconds: Optional[float] = None
    closest_track_id: Optional[int] = None
    closest_distance_px: Optional[float] = None
    closest_distance_m: Optional[float] = None
    relative_speed_kmh: Optional[float] = None
    conflicting_pairs: List[TrajectoryConflict] = field(default_factory=list)
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.LOW
    uncertainty_score: float = 0.0
    camera_visibility_score: float = 100.0
    detection_reliability: float = 1.0
    explanation: str = ""
    status: AssessmentStatus = AssessmentStatus.VALID
    collision_decision: str = "NO COLLISION RISK"
    decision_reason: str = "No credible predicted path conflict detected."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "risk_state": self.risk_state.value,
            "collision_decision": self.collision_decision,
            "decision_reason": self.decision_reason,
            "risk_score": round(self.risk_score, 3) if self.risk_score is not None else None,
            "primary_contributing_factors": self.primary_contributing_factors,
            "objects_involved": self.objects_involved,
            "min_ttc_seconds": round(self.min_ttc_seconds, 2) if self.min_ttc_seconds is not None else None,
            "closest_track_id": self.closest_track_id,
            "closest_distance_px": round(self.closest_distance_px, 1) if self.closest_distance_px is not None else None,
            "closest_distance_m": round(self.closest_distance_m, 2) if self.closest_distance_m is not None else None,
            "relative_speed_kmh": round(self.relative_speed_kmh, 1) if self.relative_speed_kmh is not None else None,
            "conflicting_pairs": [p.to_dict() for p in self.conflicting_pairs],
            "uncertainty_level": self.uncertainty_level.value,
            "uncertainty_score": round(self.uncertainty_score, 3),
            "camera_visibility_score": round(self.camera_visibility_score, 1),
            "detection_reliability": round(self.detection_reliability, 3),
            "explanation": self.explanation,
            "status": self.status.value,
            # Legacy backward compatibility aliases
            "perception_uncertainty_factor": round(self.uncertainty_score, 3),
            "fused_risk_score": round(self.risk_score, 3) if self.risk_score is not None else 0.0,
            "rationale": self.explanation,
        }


@dataclass
class CollisionStreamSummary:
    """Stream-level aggregation of collision and trajectory conflict metrics."""
    total_frames_assessed: int = 0
    risk_state_distribution: Dict[str, int] = field(default_factory=lambda: {
        CollisionRiskState.LOW.value: 0,
        CollisionRiskState.MODERATE.value: 0,
        CollisionRiskState.HIGH.value: 0,
        CollisionRiskState.CRITICAL.value: 0,
        CollisionRiskState.UNKNOWN.value: 0,
    })
    min_ttc_seconds: Optional[float] = None
    mean_risk_score: Optional[float] = None
    critical_risk_frames: int = 0
    high_risk_frames: int = 0
    total_conflicts_detected: int = 0
    conflict_type_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_frames_assessed": self.total_frames_assessed,
            "risk_state_distribution": self.risk_state_distribution,
            "min_ttc_seconds": round(self.min_ttc_seconds, 2) if self.min_ttc_seconds is not None else None,
            "mean_risk_score": round(self.mean_risk_score, 3) if self.mean_risk_score is not None else None,
            "critical_risk_frames": self.critical_risk_frames,
            "high_risk_frames": self.high_risk_frames,
            "total_conflicts_detected": self.total_conflicts_detected,
            "conflict_type_distribution": self.conflict_type_distribution,
        }
