"""Temporal risk tracker with hysteresis and track persistence monitoring.

Prevents chattering / flickering between LOW and HIGH states while ensuring:
1. Immediate escalation upon real detected danger.
2. Controlled, stabilized de-escalation requiring consecutive nominal confirmations.
3. Stale observation detection and disappearing track alerts.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.collision_risk.schemas import CollisionRiskState


class TemporalRiskTracker:
    """Maintains temporal risk smoothing and hysteresis across consecutive frames."""

    def __init__(
        self,
        de_escalation_patience_frames: int = 2,
        stale_threshold_seconds: float = 0.5,
        max_history_length: int = 30,
    ):
        self.patience = de_escalation_patience_frames
        self.stale_threshold = stale_threshold_seconds
        self.max_history = max_history_length

        self._current_state: CollisionRiskState = CollisionRiskState.LOW
        self._last_timestamp: Optional[float] = None
        self._state_history: Deque[CollisionRiskState] = deque(maxlen=max_history_length)
        self._candidate_de_escalation_frames: int = 0
        self._previous_active_track_ids: set[int] = set()

    def update(
        self,
        raw_state: CollisionRiskState,
        timestamp_seconds: float,
        current_track_ids: List[int],
    ) -> Tuple[CollisionRiskState, bool, Optional[str]]:
        """Apply hysteresis smoothing and observation continuity checks.

        Returns:
            Tuple of:
            - smoothed_state: CollisionRiskState
            - is_stale: bool
            - temporal_note: Optional[str]
        """
        temporal_note: Optional[str] = None
        is_stale = False

        # 1. Check observation latency / frame gaps
        if self._last_timestamp is not None:
            dt = timestamp_seconds - self._last_timestamp
            if dt > self.stale_threshold:
                is_stale = True
                temporal_note = f"Stale frame gap detected: {dt:.2f}s elapsed between assessments."
        self._last_timestamp = timestamp_seconds

        # 2. Check disappearing track IDs
        curr_set = set(current_track_ids)
        disappeared = self._previous_active_track_ids - curr_set
        self._previous_active_track_ids = curr_set
        if disappeared and self._current_state in (CollisionRiskState.HIGH, CollisionRiskState.CRITICAL):
            temporal_note = f"Track(s) #{sorted(disappeared)} disappeared while in elevated risk zone."

        # 3. UNKNOWN states bypass hysteresis
        if raw_state == CollisionRiskState.UNKNOWN:
            self._current_state = CollisionRiskState.UNKNOWN
            self._candidate_de_escalation_frames = 0
            self._state_history.append(self._current_state)
            return self._current_state, is_stale, temporal_note

        # Rank state severity: LOW(0) < MODERATE(1) < HIGH(2) < CRITICAL(3)
        severity_rank = {
            CollisionRiskState.LOW: 0,
            CollisionRiskState.MODERATE: 1,
            CollisionRiskState.HIGH: 2,
            CollisionRiskState.CRITICAL: 3,
        }

        current_rank = severity_rank.get(self._current_state, 0)
        raw_rank = severity_rank.get(raw_state, 0)

        # Immediate escalation: if raw_rank > current_rank, elevate at once
        if raw_rank > current_rank:
            self._current_state = raw_state
            self._candidate_de_escalation_frames = 0
        # De-escalation: requires multiple consecutive lower-risk frames
        elif raw_rank < current_rank:
            self._candidate_de_escalation_frames += 1
            if self._candidate_de_escalation_frames >= self.patience:
                self._current_state = raw_state
                self._candidate_de_escalation_frames = 0
        else:
            self._candidate_de_escalation_frames = 0

        self._state_history.append(self._current_state)
        return self._current_state, is_stale, temporal_note

    def get_recent_history(self) -> List[str]:
        """Return list of recent risk state strings."""
        return [s.value for s in self._state_history]
