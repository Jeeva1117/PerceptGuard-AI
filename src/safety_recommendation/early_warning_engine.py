"""PerceptGuard AI - Early Warning Engine (Feature 5).

Provides timestamp-aware temporal risk trend monitoring to detect developing road
dangers across consecutive frames before a critical incident materializes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TemporalObservation:
    """Historical snapshot of upstream perception risk at a discrete timestamp."""
    timestamp_seconds: float
    frame_index: int
    risk_state: str  # LOW, MODERATE, HIGH, CRITICAL, UNKNOWN
    fused_risk_score: float
    min_ttc_seconds: Optional[float]
    closest_distance_m: Optional[float]
    visibility_score: float
    uncertainty_score: float
    active_track_count: int
    conflict_count: int


@dataclass
class EarlyWarningSignal:
    """Distinct early warning signal identified by temporal analysis."""
    signal_id: str
    signal_type: str  # e.g. "ESCALATING_RISK", "DECREASING_TTC", "PEDESTRIAN_APPROACHING", "DEGRADING_VISIBILITY"
    severity: str     # "INFO", "ADVISORY", "WARNING", "CRITICAL"
    description: str
    timestamp_seconds: float


class EarlyWarningEngine:
    """Monitors temporal sequences of perception and risk data to detect developing hazards."""

    def __init__(self, history_capacity: int = 15) -> None:
        self.history_capacity = history_capacity
        self.history: List[TemporalObservation] = []
        self.active_signals: List[EarlyWarningSignal] = []

    def update(
        self,
        frame_index: int,
        timestamp_seconds: float,
        collision_data: Dict[str, Any],
        weather_intel: Dict[str, Any],
        trajectories: List[Dict[str, Any]],
        active_tracks: List[Dict[str, Any]],
    ) -> List[EarlyWarningSignal]:
        """Record current frame observation and evaluate developing temporal danger patterns."""
        risk_state = str(collision_data.get("risk_state", "UNKNOWN")).upper()
        fused_risk = float(collision_data.get("fused_risk_score", 0.0))
        min_ttc = collision_data.get("min_ttc_seconds")
        if min_ttc is not None:
            min_ttc = float(min_ttc)
        closest_dist = collision_data.get("closest_distance_m")
        if closest_dist is not None:
            closest_dist = float(closest_dist)
        conflicts = collision_data.get("conflicting_pairs", [])

        vis_score = float(weather_intel.get("visibility_score", 100.0))
        uncert = float(
            collision_data.get("uncertainty_score")
            or collision_data.get("perception_uncertainty_factor")
            or 0.15
        )

        obs = TemporalObservation(
            timestamp_seconds=timestamp_seconds,
            frame_index=frame_index,
            risk_state=risk_state,
            fused_risk_score=fused_risk,
            min_ttc_seconds=min_ttc,
            closest_distance_m=closest_dist,
            visibility_score=vis_score,
            uncertainty_score=uncert,
            active_track_count=len(active_tracks),
            conflict_count=len(conflicts),
        )

        self.history.append(obs)
        if len(self.history) > self.history_capacity:
            self.history.pop(0)

        signals = self._detect_temporal_patterns(obs, trajectories)
        self.active_signals = signals
        return signals

    def _detect_temporal_patterns(
        self,
        current: TemporalObservation,
        trajectories: List[Dict[str, Any]],
    ) -> List[EarlyWarningSignal]:
        """Detect risk escalation, rapidly decreasing TTC, and approaching agents."""
        signals: List[EarlyWarningSignal] = []

        if len(self.history) < 2:
            return signals

        prev = self.history[-2]

        # 1. State Escalation: LOW -> MODERATE or MODERATE -> HIGH/CRITICAL
        risk_hierarchy = {"LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4, "UNKNOWN": 0}
        curr_val = risk_hierarchy.get(current.risk_state, 0)
        prev_val = risk_hierarchy.get(prev.risk_state, 0)

        if curr_val > prev_val and prev_val > 0:
            sev = "CRITICAL" if current.risk_state == "CRITICAL" else ("WARNING" if current.risk_state == "HIGH" else "ADVISORY")
            signals.append(
                EarlyWarningSignal(
                    signal_id=f"esc_{current.frame_index}",
                    signal_type="RISK_ESCALATION",
                    severity=sev,
                    description=f"Road risk escalated from {prev.risk_state} to {current.risk_state}.",
                    timestamp_seconds=current.timestamp_seconds,
                )
            )

        # 2. Decreasing TTC Trend
        if (
            current.min_ttc_seconds is not None
            and prev.min_ttc_seconds is not None
            and current.min_ttc_seconds < prev.min_ttc_seconds
        ):
            dt = max(current.timestamp_seconds - prev.timestamp_seconds, 0.001)
            ttc_rate = (prev.min_ttc_seconds - current.min_ttc_seconds) / dt
            if current.min_ttc_seconds <= 3.5:
                signals.append(
                    EarlyWarningSignal(
                        signal_id=f"ttc_drop_{current.frame_index}",
                        signal_type="DECREASING_TTC",
                        severity="WARNING" if current.min_ttc_seconds <= 2.0 else "ADVISORY",
                        description=f"Time-to-Collision contracting rapidly: {current.min_ttc_seconds:.1f}s (rate: -{ttc_rate:.1f}s/s).",
                        timestamp_seconds=current.timestamp_seconds,
                    )
                )

        # 3. Pedestrian Entering Path
        for trk in trajectories:
            c_name = str(trk.get("class_name", "")).lower()
            if c_name in ("person", "pedestrian"):
                spd = trk.get("speed_kmh", 0.0)
                if current.closest_distance_m is not None and current.closest_distance_m < 15.0:
                    signals.append(
                        EarlyWarningSignal(
                            signal_id=f"ped_entry_{current.frame_index}_{trk.get('track_id')}",
                            signal_type="PEDESTRIAN_CROSSING_CORRIDOR",
                            severity="WARNING",
                            description=f"Pedestrian (Track #{trk.get('track_id')}) detected in forward safety corridor ({current.closest_distance_m:.1f}m).",
                            timestamp_seconds=current.timestamp_seconds,
                        )
                    )
                    break

        # 4. Deteriorating Camera Visibility Trend
        if len(self.history) >= 4:
            oldest_vis = self.history[-4].visibility_score
            vis_delta = oldest_vis - current.visibility_score
            if vis_delta > 15.0:
                signals.append(
                    EarlyWarningSignal(
                        signal_id=f"vis_degrade_{current.frame_index}",
                        signal_type="VISIBILITY_DETERIORATION",
                        severity="ADVISORY",
                        description=f"Visibility deteriorating (-{vis_delta:.0f}% over 4 frames). Perception buffers reducing.",
                        timestamp_seconds=current.timestamp_seconds,
                    )
                )

        return signals

    def reset(self) -> None:
        """Clear historical tracking buffers."""
        self.history.clear()
        self.active_signals.clear()
