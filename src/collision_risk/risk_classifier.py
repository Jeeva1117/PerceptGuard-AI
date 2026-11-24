"""Multi-evidence collision risk classifier.

Maps spatio-temporal conflicts, Time-To-Collision (TTC), relative closing dynamics,
vulnerable road users (pedestrians/cyclists), and multi-source uncertainty into:
- 5 discrete states: LOW, MODERATE, HIGH, CRITICAL, UNKNOWN.
- Continuous explainable risk score (0.00 to 1.00) or None for UNKNOWN.
- Defensible, non-arbitrary rationales.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.collision_risk.schemas import (
    AssessmentStatus,
    CollisionRiskState,
    ConflictType,
    TrajectoryConflict,
    UncertaintyLevel,
)


class CollisionRiskClassifier:
    """Classifies road collision danger using multi-dimensional kinematic and contextual evidence."""

    def __init__(
        self,
        critical_ttc_seconds: float = 1.8,
        high_ttc_seconds: float = 3.5,
        moderate_ttc_seconds: float = 5.0,
    ):
        self.critical_ttc = critical_ttc_seconds
        self.high_ttc = high_ttc_seconds
        self.moderate_ttc = moderate_ttc_seconds

    def classify_risk(
        self,
        min_ttc: Optional[float],
        conflicts: List[TrajectoryConflict],
        closest_dist_px: Optional[float],
        closest_dist_m: Optional[float],
        closest_tid: Optional[int],
        uncertainty_score: float,
        uncertainty_level: UncertaintyLevel,
        is_unreliable: bool,
        uncertainty_factors: List[str],
        num_tracks: int,
    ) -> Tuple[CollisionRiskState, Optional[float], List[str], str, AssessmentStatus]:
        """Synthesize all available evidence into defensible risk state and score.

        Returns:
            Tuple of:
            - risk_state: CollisionRiskState
            - risk_score: Optional[float] (0.0 to 1.0, or None if UNKNOWN)
            - primary_contributing_factors: List[str]
            - explanation: str
            - status: AssessmentStatus
        """
        contributing_factors: List[str] = list(uncertainty_factors)

        # 1. Check for UNKNOWN state triggers
        if is_unreliable:
            return (
                CollisionRiskState.UNKNOWN,
                None,
                contributing_factors,
                "Risk assessment withheld: Environmental or sensor observations are insufficient or degraded.",
                AssessmentStatus.UNKNOWN_HIGH_UNCERTAINTY,
            )

        if num_tracks == 0:
            return (
                CollisionRiskState.LOW,
                0.05,
                ["Empty forward driving corridor; zero moving road users tracked."],
                "Traffic corridor clear: No moving vehicles, pedestrians, or obstacles detected.",
                AssessmentStatus.VALID,
            )

        # Check for vulnerable road users (pedestrians, cyclists) in conflicts
        has_vulnerable_conflict = any(
            c.conflict_type in (ConflictType.PEDESTRIAN_CROSSING, ConflictType.CYCLIST_CONFLICT)
            for c in conflicts
        )

        min_conflict_time: Optional[float] = None
        if conflicts:
            min_conflict_time = min(c.time_to_conflict_seconds for c in conflicts)

        # Evaluate risk conditions
        state = CollisionRiskState.LOW
        base_risk_score = 0.10

        # Condition 1: Imminent Collision (CRITICAL)
        if (min_ttc is not None and min_ttc <= self.critical_ttc) or (
            min_conflict_time is not None and min_conflict_time <= 1.2
        ) or (has_vulnerable_conflict and min_conflict_time is not None and min_conflict_time <= 2.0):
            state = CollisionRiskState.CRITICAL
            base_risk_score = 0.90
            if min_ttc is not None:
                contributing_factors.append(f"Imminent Time-To-Collision ({min_ttc:.1f}s <= {self.critical_ttc}s)")
            if min_conflict_time is not None:
                contributing_factors.append(f"Predicted trajectory collision within {min_conflict_time:.1f}s")
            if has_vulnerable_conflict:
                contributing_factors.append("Trajectory conflict involves vulnerable road user (pedestrian/cyclist)")

        # Condition 2: High Danger (HIGH)
        elif (min_ttc is not None and min_ttc <= self.high_ttc) or (
            min_conflict_time is not None and min_conflict_time <= 3.0
        ) or (has_vulnerable_conflict and min_conflict_time is not None and min_conflict_time <= 3.5):
            state = CollisionRiskState.HIGH
            base_risk_score = 0.70
            if min_ttc is not None:
                contributing_factors.append(f"Reduced Time-To-Collision ({min_ttc:.1f}s <= {self.high_ttc}s)")
            if min_conflict_time is not None:
                contributing_factors.append(f"Projected trajectory conflict within {min_conflict_time:.1f}s")
            if has_vulnerable_conflict:
                contributing_factors.append("Active trajectory convergence with pedestrian or cyclist")

        # Condition 3: Moderate Caution (MODERATE)
        # Requires verified closing motion (valid TTC) OR an active multi-horizon trajectory conflict.
        # Proximity alone without closing dynamics (e.g. parked cars or parallel lane flow) does NOT trigger collision risk.
        elif (min_ttc is not None and min_ttc <= self.moderate_ttc) or len(conflicts) > 0:
            state = CollisionRiskState.MODERATE
            base_risk_score = 0.40
            if min_ttc is not None:
                contributing_factors.append(f"Approaching vehicle with closing TTC {min_ttc:.1f}s <= {self.moderate_ttc}s")
            if conflicts:
                contributing_factors.append(f"{len(conflicts)} trajectory intersection(s) within forward horizons")

        # Condition 4: Nominal Operation (LOW) - NO COLLISION RISK DETECTED
        else:
            state = CollisionRiskState.LOW
            base_risk_score = 0.05
            contributing_factors.append("Safe headway and lateral separation maintained; vehicles following lanes normally")
            contributing_factors.append("No trajectory conflict points predicted within 5.0s horizon")

        # Calibrate continuous fused risk score with uncertainty
        # Under elevated uncertainty, risk score scales up moderately to account for perception margin
        fused_score = min(1.0, base_risk_score + 0.15 * uncertainty_score)

        # Formulate explanation
        if state == CollisionRiskState.CRITICAL:
            explanation = (
                f"CRITICAL HAZARD: Immediate collision risk identified. "
                f"{'; '.join(contributing_factors[:2])}. Immediate braking / avoidance required."
            )
        elif state == CollisionRiskState.HIGH:
            explanation = (
                f"HIGH RISK: Elevated trajectory convergence detected. "
                f"{'; '.join(contributing_factors[:2])}. Prepare for decelerating maneuvers."
            )
        elif state == CollisionRiskState.MODERATE:
            explanation = (
                f"MODERATE ADVISORY: Potential conflict pattern identified. "
                f"{'; '.join(contributing_factors[:2])}."
            )
        else:
            explanation = "NO COLLISION RISK DETECTED: Nominal vehicle clearances maintained; zero trajectory conflicts detected."

        return state, round(fused_score, 3), contributing_factors, explanation, AssessmentStatus.VALID
