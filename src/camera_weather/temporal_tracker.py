"""Temporal Frame Analysis and State Transition Tracking for Camera & Weather Intelligence.

Maintains bounded temporal history, applies adaptive exponential smoothing,
computes rate of change (velocity of degradation), and classifies:
- Sudden degradation (e.g. sudden water splash, entering dark underpass, direct high-beam glare)
- Gradual degradation (e.g. progressive dusk, thickening fog, worsening rain)
- Temporary visibility interruptions (e.g. wiper sweep, passing shadow)
- Abrupt vs smooth state transitions across conditions
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import enum
from typing import Any, Deque, Dict, List, Optional, Tuple


class TemporalDegradationType(str, enum.Enum):
    STABLE = "STABLE"
    IMPROVING = "IMPROVING"
    GRADUAL_DEGRADATION = "GRADUAL_DEGRADATION"
    SUDDEN_DEGRADATION = "SUDDEN_DEGRADATION"
    TEMPORARY_INTERRUPTION = "TEMPORARY_INTERRUPTION"


@dataclass
class ConditionTransition:
    from_condition: str
    to_condition: str
    frame_index: int
    timestamp_seconds: float
    visibility_delta: float
    transition_type: str  # "ABRUPT" or "GRADUAL"


@dataclass
class TemporalFrameRecord:
    frame_index: int
    timestamp_seconds: float
    raw_visibility_score: float
    smoothed_visibility_score: float
    primary_condition: str
    rate_of_change: float  # points per second (dS/dt)
    degradation_type: TemporalDegradationType


class TemporalWeatherTracker:
    """Production temporal tracker smoothing measurements and monitoring degradation dynamics."""

    def __init__(
        self,
        history_window: int = 30,
        base_alpha: float = 0.25,
        adaptive_alpha: float = 0.65,
    ):
        self.history_window = history_window
        self.base_alpha = base_alpha
        self.adaptive_alpha = adaptive_alpha

        self.history: Deque[TemporalFrameRecord] = deque(maxlen=history_window)
        self.transitions: List[ConditionTransition] = []
        self._current_smoothed: Optional[float] = None
        self._last_condition: Optional[str] = None

    def update(
        self,
        frame_index: int,
        timestamp_seconds: float,
        raw_visibility: float,
        primary_condition: str,
    ) -> Tuple[float, float, TemporalDegradationType]:
        """Update tracker with the latest frame's raw visibility score.

        Returns:
            smoothed_visibility: Adaptive EMA smoothed score
            rate_of_change: Velocity in visibility points / second
            degradation_type: Categorical classification of temporal change
        """
        if self._current_smoothed is None:
            # First frame initialization
            smoothed = raw_visibility
            rate_of_change = 0.0
            deg_type = TemporalDegradationType.STABLE
        else:
            delta = raw_visibility - self._current_smoothed
            dt = 0.033
            if len(self.history) > 0:
                dt = max(0.001, timestamp_seconds - self.history[-1].timestamp_seconds)

            # Adaptive smoothing: if delta is large (> 25 pts), switch to fast alpha to catch sudden events
            if abs(delta) > 25.0:
                alpha = self.adaptive_alpha
            else:
                alpha = self.base_alpha

            smoothed = alpha * raw_visibility + (1.0 - alpha) * self._current_smoothed
            rate_of_change = float((smoothed - self._current_smoothed) / dt)

            # Check for temporary interruption (e.g. wiper sweep or brief shadow)
            # Characteristic: dropped sharply 1-3 frames ago but now rebounding back
            is_temporary = False
            if len(self.history) >= 3:
                recent_scores = [rec.smoothed_visibility_score for rec in list(self.history)[-3:]]
                min_recent = min(recent_scores)
                # If there was a sudden dip and recovery:
                if (recent_scores[0] - min_recent > 20.0) and (smoothed - min_recent > 15.0):
                    is_temporary = True

            if is_temporary:
                deg_type = TemporalDegradationType.TEMPORARY_INTERRUPTION
            elif rate_of_change < -20.0:
                deg_type = TemporalDegradationType.SUDDEN_DEGRADATION
            elif rate_of_change < -3.0:
                deg_type = TemporalDegradationType.GRADUAL_DEGRADATION
            elif rate_of_change > 3.0:
                deg_type = TemporalDegradationType.IMPROVING
            else:
                deg_type = TemporalDegradationType.STABLE

        self._current_smoothed = smoothed

        # Track condition transitions (never forcing a rigid sequence)
        if self._last_condition is not None and primary_condition != self._last_condition:
            vis_delta = (
                smoothed - self.history[-1].smoothed_visibility_score if self.history else 0.0
            )
            trans_type = "ABRUPT" if abs(rate_of_change) > 15.0 else "GRADUAL"
            self.transitions.append(
                ConditionTransition(
                    from_condition=self._last_condition,
                    to_condition=primary_condition,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    visibility_delta=round(vis_delta, 2),
                    transition_type=trans_type,
                )
            )

        self._last_condition = primary_condition

        rec = TemporalFrameRecord(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            raw_visibility_score=raw_visibility,
            smoothed_visibility_score=smoothed,
            primary_condition=primary_condition,
            rate_of_change=rate_of_change,
            degradation_type=deg_type,
        )
        self.history.append(rec)

        return smoothed, rate_of_change, deg_type

    def get_history_scores(self) -> List[Dict[str, Any]]:
        """Return bounded history for charting and temporal inspection."""
        return [
            {
                "frame_index": r.frame_index,
                "timestamp_seconds": round(r.timestamp_seconds, 3),
                "raw_score": round(r.raw_visibility_score, 1),
                "smoothed_score": round(r.smoothed_visibility_score, 1),
                "condition": r.primary_condition,
                "rate_of_change": round(r.rate_of_change, 2),
                "degradation_type": r.degradation_type.value,
            }
            for r in self.history
        ]
