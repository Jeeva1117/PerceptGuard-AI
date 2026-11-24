"""PerceptGuard AI - Warning History & Hysteresis State Manager (Feature 5).

Maintains warning transitions, applies de-escalation hysteresis to eliminate state chatter,
and tracks recovery from degraded or UNKNOWN states back to reliable operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from src.safety_recommendation.schemas import RecommendationState, SafetyRecommendation

logger = logging.getLogger(__name__)

# Severity mapping for hysteresis ordering
STATE_SEVERITY: Dict[RecommendationState, int] = {
    RecommendationState.NORMAL: 1,
    RecommendationState.CAUTION: 2,
    RecommendationState.REDUCE_SPEED: 3,
    RecommendationState.PREPARE_TO_STOP: 4,
    RecommendationState.STOP_RECOMMENDED: 5,
    RecommendationState.UNKNOWN: 2,  # Treat UNKNOWN at advisory level for hysteresis
}


@dataclass
class RecommendationTransition:
    """Audit record capturing a change in safety advisory state."""
    from_state: str
    to_state: str
    timestamp_seconds: float
    frame_index: int
    trigger_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "frame_index": self.frame_index,
            "trigger_reason": self.trigger_reason,
        }


class WarningHistoryManager:
    """Tracks state history and applies de-escalation hysteresis."""

    def __init__(self, deescalation_confirmation_frames: int = 2) -> None:
        self.deescalation_confirmation_frames = deescalation_confirmation_frames

        self.current_state: RecommendationState = RecommendationState.NORMAL
        self.pending_lower_state: Optional[RecommendationState] = None
        self.pending_lower_count: int = 0
        self.transitions: List[RecommendationTransition] = []
        self.recommendations_log: List[SafetyRecommendation] = []

    def filter_with_hysteresis(
        self,
        candidate: SafetyRecommendation,
        frame_index: int,
    ) -> SafetyRecommendation:
        """Apply hysteresis smoothing to de-escalations while allowing immediate escalation."""
        cand_state = candidate.recommendation_state
        cand_sev = STATE_SEVERITY.get(cand_state, 1)
        curr_sev = STATE_SEVERITY.get(self.current_state, 1)

        # 1. Immediate Escalation: candidate is MORE severe than current state
        if cand_sev > curr_sev:
            self._record_transition(
                from_st=self.current_state,
                to_st=cand_state,
                timestamp=candidate.timestamp_seconds,
                frame_idx=frame_index,
                reason=candidate.primary_reason,
            )
            self.current_state = cand_state
            self.pending_lower_state = None
            self.pending_lower_count = 0
            self.recommendations_log.append(candidate)
            return candidate

        # 2. Same severity level
        if cand_sev == curr_sev:
            if cand_state != self.current_state:
                self._record_transition(
                    from_st=self.current_state,
                    to_st=cand_state,
                    timestamp=candidate.timestamp_seconds,
                    frame_idx=frame_index,
                    reason=candidate.primary_reason,
                )
                self.current_state = cand_state
            self.pending_lower_state = None
            self.pending_lower_count = 0
            self.recommendations_log.append(candidate)
            return candidate

        # 3. De-escalation: candidate is LESS severe than current state
        # Require confirmation over multiple consecutive frames
        if self.pending_lower_state == cand_state:
            self.pending_lower_count += 1
        else:
            self.pending_lower_state = cand_state
            self.pending_lower_count = 1

        if self.pending_lower_count >= self.deescalation_confirmation_frames:
            # Confirmed de-escalation
            self._record_transition(
                from_st=self.current_state,
                to_st=cand_state,
                timestamp=candidate.timestamp_seconds,
                frame_idx=frame_index,
                reason=f"De-escalation confirmed after {self.pending_lower_count} frames: {candidate.primary_reason}",
            )
            self.current_state = cand_state
            self.pending_lower_state = None
            self.pending_lower_count = 0
            self.recommendations_log.append(candidate)
            return candidate
        else:
            # Hysteresis holds current higher state for 1 more frame to avoid flicker
            logger.debug(
                f"Holding state {self.current_state.value} (de-escalation candidate {cand_state.value} pending {self.pending_lower_count}/{self.deescalation_confirmation_frames})"
            )
            # Clone candidate with held state
            held_rec = SafetyRecommendation(
                warning_id=candidate.warning_id,
                timestamp_seconds=candidate.timestamp_seconds,
                recommendation_state=self.current_state,
                priority=candidate.priority,
                risk_level=candidate.risk_level,
                objects_involved=candidate.objects_involved,
                recommended_action=candidate.recommended_action,
                primary_reason=f"[Hysteresis Hold] {candidate.primary_reason}",
                supporting_evidence=candidate.supporting_evidence,
                prediction_horizon=candidate.prediction_horizon,
                min_ttc_seconds=candidate.min_ttc_seconds,
                relative_speed_kmh=candidate.relative_speed_kmh,
                camera_visibility_status=candidate.camera_visibility_status,
                detection_reliability=candidate.detection_reliability,
                uncertainty_level=candidate.uncertainty_level,
                status=candidate.status,
            )
            self.recommendations_log.append(held_rec)
            return held_rec

    def _record_transition(
        self,
        from_st: RecommendationState,
        to_st: RecommendationState,
        timestamp: float,
        frame_idx: int,
        reason: str,
    ) -> None:
        """Append an audit transition."""
        trans = RecommendationTransition(
            from_state=from_st.value,
            to_state=to_st.value,
            timestamp_seconds=timestamp,
            frame_index=frame_idx,
            trigger_reason=reason,
        )
        self.transitions.append(trans)
        if len(self.transitions) > 100:
            self.transitions.pop(0)

    def reset(self) -> None:
        """Reset hysteresis tracking."""
        self.current_state = RecommendationState.NORMAL
        self.pending_lower_state = None
        self.pending_lower_count = 0
        self.transitions.clear()
        self.recommendations_log.clear()
