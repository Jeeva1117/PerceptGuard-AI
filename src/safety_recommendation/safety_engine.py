"""PerceptGuard AI - Central Safety Recommendation Engine (Feature 5).

Orchestrates deterministic rule evaluation, temporal early warning monitoring, priority
arbitration, duplicate alert suppression, and de-escalation hysteresis to deliver
explainable, evidence-backed safety advisories.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.safety_recommendation.early_warning_engine import EarlyWarningEngine
from src.safety_recommendation.recommendation_rules import RecommendationRulesEngine
from src.safety_recommendation.schemas import (
    RecommendationState,
    SafetyRecommendation,
    SafetyStreamSummary,
    WarningPriority,
    WarningStatus,
)
from src.safety_recommendation.warning_history_manager import WarningHistoryManager
from src.safety_recommendation.warning_priority_manager import WarningPriorityManager

logger = logging.getLogger(__name__)


class SafetyRecommendationEngine:
    """Master orchestrator for Feature 5: Early Warning & Vehicle Stop Recommendation."""

    def __init__(
        self,
        critical_ttc_threshold: float = 1.8,
        warning_ttc_threshold: float = 3.5,
        max_uncertainty_threshold: float = 0.70,
        min_visibility_threshold: float = 20.0,
        deescalation_confirmation_frames: int = 2,
        suppression_window_sec: float = 0.35,
        stale_expiration_sec: float = 0.60,
    ) -> None:
        self.rules_engine = RecommendationRulesEngine(
            critical_ttc_threshold=critical_ttc_threshold,
            warning_ttc_threshold=warning_ttc_threshold,
            max_uncertainty_threshold=max_uncertainty_threshold,
            min_visibility_threshold=min_visibility_threshold,
            stale_data_timeout_sec=stale_expiration_sec,
        )
        self.early_warning = EarlyWarningEngine(history_capacity=15)
        self.priority_manager = WarningPriorityManager(
            suppression_window_sec=suppression_window_sec,
            stale_expiration_sec=stale_expiration_sec,
        )
        self.history_manager = WarningHistoryManager(
            deescalation_confirmation_frames=deescalation_confirmation_frames
        )

        self.evaluated_recommendations: List[SafetyRecommendation] = []

    def evaluate_frame(
        self,
        collision_data: Optional[Dict[str, Any]],
        weather_intel: Optional[Dict[str, Any]],
        trajectories: Optional[List[Dict[str, Any]]] = None,
        active_tracks: Optional[List[Dict[str, Any]]] = None,
        timestamp_seconds: float = 0.0,
        frame_index: int = 0,
        pedestrian_telemetry: Optional[Dict[str, Any]] = None,
    ) -> SafetyRecommendation:
        """Run full Feature 5 evaluation pipeline on the current video frame."""
        # Step 1: Temporal early warning pattern evaluation
        trajs = trajectories if trajectories is not None else []
        trks = active_tracks if active_tracks is not None else []
        c_data = collision_data or {}
        w_intel = weather_intel or {}

        signals = self.early_warning.update(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            collision_data=c_data,
            weather_intel=w_intel,
            trajectories=trajs,
            active_tracks=trks,
        )

        # Step 2: Deterministic decision rules
        candidate_rec = self.rules_engine.evaluate(
            collision_data=collision_data,
            weather_intel=weather_intel,
            trajectories=trajectories,
            active_tracks=active_tracks,
            timestamp_seconds=timestamp_seconds,
            previous_recommendation=(
                self.evaluated_recommendations[-1] if self.evaluated_recommendations else None
            ),
        )

        # Step 2b: Pedestrian awareness escalation if hazard confirmed
        if pedestrian_telemetry:
            ped_state = pedestrian_telemetry.get("primary_state", "")
            if ped_state == "POTENTIAL PEDESTRIAN COLLISION RISK":
                candidate_rec.recommendation_state = RecommendationState.STOP_RECOMMENDED
                candidate_rec.priority = WarningPriority.CRITICAL
                candidate_rec.risk_level = "CRITICAL"
                candidate_rec.recommended_action = "Stop vehicle when safe to do so. Pedestrian collision hazard in projected travel corridor."
                candidate_rec.primary_reason = "Imminent pedestrian-vehicle path intersection detected with low TTC."
                candidate_rec.status = WarningStatus.ACTIVE
            elif ped_state == "PEDESTRIAN CROSSING" and candidate_rec.recommendation_state == RecommendationState.NORMAL:
                candidate_rec.recommendation_state = RecommendationState.REDUCE_SPEED
                candidate_rec.priority = WarningPriority.ADVISORY
                candidate_rec.risk_level = "MODERATE"
                candidate_rec.recommended_action = "Reduce vehicle speed and yield right-of-way to crossing pedestrian."
                candidate_rec.primary_reason = "Pedestrian crossing ahead within forward roadway zone."
                candidate_rec.status = WarningStatus.ACTIVE
            elif ped_state == "PEDESTRIAN APPROACHING ROAD" and candidate_rec.recommendation_state == RecommendationState.NORMAL:
                candidate_rec.recommendation_state = RecommendationState.CAUTION
                candidate_rec.priority = WarningPriority.ADVISORY
                candidate_rec.risk_level = "CAUTION"
                candidate_rec.recommended_action = "Monitor pedestrian trajectory and reduce speed if necessary."
                candidate_rec.primary_reason = "Pedestrian approaching road edge from sidewalk."
                candidate_rec.status = WarningStatus.ACTIVE

        # Enrich primary reason with any active early warning signals
        if signals and candidate_rec.recommendation_state not in (
            RecommendationState.NORMAL,
            RecommendationState.UNKNOWN,
        ):
            sig_text = "; ".join(s.description for s in signals[:2])
            if sig_text not in candidate_rec.primary_reason:
                candidate_rec.primary_reason = f"{candidate_rec.primary_reason} [{sig_text}]"

        # Step 3: History & Hysteresis Filtering (immediate escalation, smoothed de-escalation)
        hysteresis_rec = self.history_manager.filter_with_hysteresis(
            candidate=candidate_rec,
            frame_index=frame_index,
        )

        # Step 4: Priority Arbitration & Duplicate Alert Suppression
        final_rec = self.priority_manager.process(hysteresis_rec)

        self.evaluated_recommendations.append(final_rec)
        return final_rec

    def generate_stream_summary(self) -> SafetyStreamSummary:
        """Generate high-level stream statistics across the full evaluated sequence."""
        rec_counts: Dict[str, int] = {}
        prio_counts: Dict[str, int] = {}
        action_counts: Dict[str, int] = {}
        stop_count = 0

        for r in self.evaluated_recommendations:
            st = r.recommendation_state.value
            rec_counts[st] = rec_counts.get(st, 0) + 1

            pr = r.priority.value
            prio_counts[pr] = prio_counts.get(pr, 0) + 1

            act = r.primary_action
            action_counts[act] = action_counts.get(act, 0) + 1

            if r.recommendation_state == RecommendationState.STOP_RECOMMENDED:
                stop_count += 1

        most_freq_rec = max(rec_counts.items(), key=lambda x: x[1])[0] if rec_counts else "NORMAL"
        most_freq_act = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else "MAINTAIN_SPEED"

        return SafetyStreamSummary(
            total_recommendations=len(self.evaluated_recommendations),
            recommendation_counts=rec_counts,
            priority_counts=prio_counts,
            most_frequent_recommendation=most_freq_rec,
            critical_stop_count=stop_count,
            state_transitions=[t.to_dict() for t in self.history_manager.transitions],
            action_counts=action_counts,
            most_frequent_action=most_freq_act,
        )

    def reset(self) -> None:
        """Reset internal pipeline components."""
        self.early_warning.reset()
        self.priority_manager.reset()
        self.history_manager.reset()
        self.evaluated_recommendations.clear()
