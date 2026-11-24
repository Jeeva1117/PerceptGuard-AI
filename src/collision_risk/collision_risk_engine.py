"""Integrated Collision & Road Risk Assessment Engine for PerceptGuard AI.

Fuses:
1. Distance and spatial separation analysis (pixels & ground-plane meters).
2. Relative velocity, acceleration, and closing dynamics along line-of-sight.
3. Defensible Time-To-Collision (TTC) calculations under valid kinematic assumptions.
4. Spatio-temporal trajectory conflict detection across multi-horizon forecasts.
5. Multi-source uncertainty fusion (weather, detection, tracking maturity).
6. Temporal smoothing and hysteresis to eliminate alarm flickering.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.collision_risk.distance_analyzer import DistanceAnalyzer
from src.collision_risk.relative_motion_analyzer import RelativeMotionAnalyzer
from src.collision_risk.risk_classifier import CollisionRiskClassifier
from src.collision_risk.schemas import (
    AssessmentStatus,
    CollisionRiskAssessment,
    CollisionRiskState,
    CollisionStreamSummary,
    TrajectoryConflict,
    UncertaintyLevel,
)
from src.collision_risk.temporal_risk_tracker import TemporalRiskTracker
from src.collision_risk.trajectory_conflict_analyzer import TrajectoryConflictAnalyzer
from src.collision_risk.ttc_calculator import TTCCalculator
from src.collision_risk.uncertainty_analyzer import UncertaintyAnalyzer


class CollisionRiskEngine:
    """Production safety prototype engine analyzing road user trajectories and collision hazards."""

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        distance_analyzer: Optional[DistanceAnalyzer] = None,
        relative_analyzer: Optional[RelativeMotionAnalyzer] = None,
        ttc_calculator: Optional[TTCCalculator] = None,
        conflict_analyzer: Optional[TrajectoryConflictAnalyzer] = None,
        uncertainty_analyzer: Optional[UncertaintyAnalyzer] = None,
        classifier: Optional[CollisionRiskClassifier] = None,
        temporal_tracker: Optional[TemporalRiskTracker] = None,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.distance_analyzer = distance_analyzer or DistanceAnalyzer(frame_width, frame_height)
        self.relative_analyzer = relative_analyzer or RelativeMotionAnalyzer()
        self.ttc_calculator = ttc_calculator or TTCCalculator()
        self.conflict_analyzer = conflict_analyzer or TrajectoryConflictAnalyzer()
        self.uncertainty_analyzer = uncertainty_analyzer or UncertaintyAnalyzer()
        self.classifier = classifier or CollisionRiskClassifier()
        self.temporal_tracker = temporal_tracker or TemporalRiskTracker()

        # Cumulative stream telemetry
        self._all_assessments: List[CollisionRiskAssessment] = []

    def assess_frame(
        self,
        trajectories: List[Dict[str, Any]],
        weather_intel: Dict[str, Any],
        timestamp_seconds: float,
        fps: float = 10.0,
    ) -> CollisionRiskAssessment:
        """Execute complete multi-factor collision hazard assessment for the current video frame."""
        assessment_id = f"col_risk_{uuid.uuid4().hex[:8]}"

        # 1. Uncertainty Analysis
        (
            uncertainty_score,
            uncertainty_level,
            uncertainty_factors,
            is_unreliable,
        ) = self.uncertainty_analyzer.evaluate_uncertainty(weather_intel, trajectories, fps)

        # 2. Distance and Separation Analysis
        dist_data = self.distance_analyzer.analyze_scene_distances(trajectories)
        closest_tid = dist_data["closest_track_id"]
        closest_px = dist_data["closest_distance_px"]
        closest_m = dist_data["closest_distance_m"]

        # 3. Relative Motion & TTC with Ego Vehicle
        ego_center_px = (self.frame_width / 2.0, float(self.frame_height))
        min_ttc: Optional[float] = None
        closest_rel_speed_kmh: Optional[float] = None
        objects_involved: List[Dict[str, Any]] = []

        for trk in trajectories:
            tid = trk["track_id"]
            cls_name = trk.get("class_name", "vehicle")
            cx, cy = trk["center"]
            vx = trk.get("velocity", {}).get("vx", 0.0)
            vy = trk.get("velocity", {}).get("vy", 0.0)
            ax = trk.get("acceleration", {}).get("ax", 0.0)
            ay = trk.get("acceleration", {}).get("ay", 0.0)

            objects_involved.append({"track_id": tid, "object_class": cls_name})

            # Check ego closing motion
            ego_rel = self.relative_analyzer.compute_ego_relative_motion(
                target_pos_px=(cx, cy),
                target_vel_px=(vx, vy),
                target_acc_px=(ax, ay),
                ego_pos_px=ego_center_px,
            )

            if tid == closest_tid and trk.get("speed_kmh") is not None:
                closest_rel_speed_kmh = float(trk["speed_kmh"])

            # Evaluate TTC if approaching ego
            if ego_rel["is_approaching"]:
                ttc_eval = self.ttc_calculator.evaluate_ego_ttc(
                    distance_px=ego_rel["distance_px"],
                    closing_speed_px_s=ego_rel["closing_speed_px_s"],
                    closing_acc_px_s2=ego_rel["closing_acceleration_px_s2"],
                    target_center_px=(cx, cy),
                    target_velocity_px=(vx, vy),
                    ego_center_px=ego_center_px,
                )
                if ttc_eval["is_valid"] and ttc_eval["ttc_seconds"] is not None:
                    if min_ttc is None or ttc_eval["ttc_seconds"] < min_ttc:
                        min_ttc = ttc_eval["ttc_seconds"]

        # 4. Spatio-temporal Trajectory Conflict Analysis
        conflicts = self.conflict_analyzer.detect_conflicts(trajectories, fps=fps)

        # 5. Risk Classification
        (
            raw_state,
            raw_score,
            contributing_factors,
            explanation,
            status,
        ) = self.classifier.classify_risk(
            min_ttc=min_ttc,
            conflicts=conflicts,
            closest_dist_px=closest_px,
            closest_dist_m=closest_m,
            closest_tid=closest_tid,
            uncertainty_score=uncertainty_score,
            uncertainty_level=uncertainty_level,
            is_unreliable=is_unreliable,
            uncertainty_factors=uncertainty_factors,
            num_tracks=len(trajectories),
        )

        # 6. Temporal Hysteresis & Continuity Tracking
        active_ids = [t["track_id"] for t in trajectories]
        smoothed_state, is_stale, temporal_note = self.temporal_tracker.update(
            raw_state=raw_state,
            timestamp_seconds=timestamp_seconds,
            current_track_ids=active_ids,
        )

        if temporal_note:
            contributing_factors.append(temporal_note)
        if is_stale:
            status = AssessmentStatus.DEGRADED

        assessment = CollisionRiskAssessment(
            assessment_id=assessment_id,
            timestamp_seconds=timestamp_seconds,
            risk_state=smoothed_state,
            risk_score=raw_score if smoothed_state != CollisionRiskState.UNKNOWN else None,
            primary_contributing_factors=contributing_factors,
            objects_involved=objects_involved,
            min_ttc_seconds=min_ttc,
            closest_track_id=closest_tid,
            closest_distance_px=closest_px,
            closest_distance_m=closest_m,
            relative_speed_kmh=closest_rel_speed_kmh,
            conflicting_pairs=conflicts,
            uncertainty_level=uncertainty_level,
            uncertainty_score=uncertainty_score,
            camera_visibility_score=float(weather_intel.get("visibility_score", 100.0)),
            detection_reliability=round(1.0 - uncertainty_score * 0.5, 3),
            explanation=explanation,
            status=status,
        )

        self._all_assessments.append(assessment)
        return assessment

    def generate_stream_summary(self) -> CollisionStreamSummary:
        """Generate cumulative stream summary across all processed frames."""
        total = len(self._all_assessments)
        if total == 0:
            return CollisionStreamSummary()

        dist: Dict[str, int] = {
            CollisionRiskState.LOW.value: 0,
            CollisionRiskState.MODERATE.value: 0,
            CollisionRiskState.HIGH.value: 0,
            CollisionRiskState.CRITICAL.value: 0,
            CollisionRiskState.UNKNOWN.value: 0,
        }
        conflict_types: Dict[str, int] = {}
        all_ttcs: List[float] = []
        valid_scores: List[float] = []
        total_conflicts = 0

        for a in self._all_assessments:
            s_val = a.risk_state.value
            dist[s_val] = dist.get(s_val, 0) + 1

            if a.min_ttc_seconds is not None:
                all_ttcs.append(a.min_ttc_seconds)
            if a.risk_score is not None:
                valid_scores.append(a.risk_score)

            total_conflicts += len(a.conflicting_pairs)
            for c in a.conflicting_pairs:
                ct = c.conflict_type.value
                conflict_types[ct] = conflict_types.get(ct, 0) + 1

        return CollisionStreamSummary(
            total_frames_assessed=total,
            risk_state_distribution=dist,
            min_ttc_seconds=min(all_ttcs) if all_ttcs else None,
            mean_risk_score=round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None,
            critical_risk_frames=dist.get(CollisionRiskState.CRITICAL.value, 0),
            high_risk_frames=dist.get(CollisionRiskState.HIGH.value, 0),
            total_conflicts_detected=total_conflicts,
            conflict_type_distribution=conflict_types,
        )
