"""PerceptGuard AI - Warning Priority & Suppression Manager (Feature 5).

Manages warning priority arbitration, duplicate alert suppression, and stale warning
expiration to ensure the driver receives timely, non-spammy, high-priority safety guidance.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from src.safety_recommendation.schemas import (
    RecommendationState,
    SafetyRecommendation,
    WarningPriority,
    WarningStatus,
)

logger = logging.getLogger(__name__)

PRIORITY_LEVELS: Dict[WarningPriority, int] = {
    WarningPriority.INFO: 1,
    WarningPriority.ADVISORY: 2,
    WarningPriority.WARNING: 3,
    WarningPriority.CRITICAL: 4,
}


class WarningPriorityManager:
    """Arbitrates warning priorities, suppresses frame-by-frame spam, and expires stale warnings."""

    def __init__(
        self,
        suppression_window_sec: float = 0.35,
        stale_expiration_sec: float = 0.60,
    ) -> None:
        self.suppression_window_sec = suppression_window_sec
        self.stale_expiration_sec = stale_expiration_sec

        self.active_recommendation: Optional[SafetyRecommendation] = None
        self.last_emitted_recommendation: Optional[SafetyRecommendation] = None
        self.suppression_count: int = 0

    def process(self, candidate: SafetyRecommendation) -> SafetyRecommendation:
        """Arbitrate candidate recommendation against active state."""
        # 1. Check if active recommendation is stale
        if self.active_recommendation is not None:
            time_since_active = candidate.timestamp_seconds - self.active_recommendation.timestamp_seconds
            if time_since_active > self.stale_expiration_sec or time_since_active < 0.0:
                self.active_recommendation.status = WarningStatus.EXPIRED

        # 2. Priority check: CRITICAL or higher priority always immediately replaces lower priority
        cand_prio = PRIORITY_LEVELS.get(candidate.priority, 1)
        active_prio = (
            PRIORITY_LEVELS.get(self.active_recommendation.priority, 1)
            if (self.active_recommendation and self.active_recommendation.status == WarningStatus.ACTIVE)
            else 0
        )

        if cand_prio > active_prio:
            # Escalation to higher priority is immediate and never suppressed
            if candidate.status != WarningStatus.DEGRADED:
                candidate.status = WarningStatus.ACTIVE
            self.active_recommendation = candidate
            self.last_emitted_recommendation = candidate
            return candidate

        # 3. Duplicate Suppression Check
        if self._is_duplicate(candidate, self.last_emitted_recommendation):
            time_delta = candidate.timestamp_seconds - self.last_emitted_recommendation.timestamp_seconds
            if 0.0 <= time_delta <= self.suppression_window_sec:
                # Mark candidate as SUPPRESSED to prevent notification churn
                candidate.status = WarningStatus.SUPPRESSED
                self.suppression_count += 1
                # Retain original warning ID for continuity
                candidate.warning_id = self.last_emitted_recommendation.warning_id
                return candidate

        # 4. State transition or interval passed: Emit as ACTIVE (or keep DEGRADED)
        if candidate.status != WarningStatus.DEGRADED:
            candidate.status = WarningStatus.ACTIVE
        self.active_recommendation = candidate
        self.last_emitted_recommendation = candidate
        return candidate

    def _is_duplicate(
        self,
        current: SafetyRecommendation,
        previous: Optional[SafetyRecommendation],
    ) -> bool:
        """Determine whether the current recommendation duplicates the previous emission."""
        if previous is None:
            return False

        if current.recommendation_state != previous.recommendation_state:
            return False

        if current.priority != previous.priority:
            return False

        # Compare involved tracks
        curr_tracks = {o.track_id for o in current.objects_involved}
        prev_tracks = {o.track_id for o in previous.objects_involved}
        if curr_tracks != prev_tracks:
            return False

        return True

    def reset(self) -> None:
        """Reset internal arbitration state."""
        self.active_recommendation = None
        self.last_emitted_recommendation = None
        self.suppression_count = 0
