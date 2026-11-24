"""PerceptGuard AI - Deterministic Safety Recommendation Rules (Feature 5).

Evaluates multi-source evidence from Features 1, 2, 3, and 4 and produces
explainable, evidence-backed safety recommendations adhering strictly to
ADAS advisory principles.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

from src.safety_recommendation.schemas import (
    InvolvedObject,
    RecommendationState,
    SafetyRecommendation,
    SupportingEvidence,
    WarningPriority,
    WarningStatus,
)

logger = logging.getLogger(__name__)


class RecommendationRulesEngine:
    """Evaluates multi-feature inputs to produce an explainable safety recommendation."""

    def __init__(
        self,
        critical_ttc_threshold: float = 1.8,
        warning_ttc_threshold: float = 3.5,
        max_uncertainty_threshold: float = 0.70,
        min_visibility_threshold: float = 20.0,
        stale_data_timeout_sec: float = 0.50,
    ) -> None:
        self.critical_ttc_threshold = critical_ttc_threshold
        self.warning_ttc_threshold = warning_ttc_threshold
        self.max_uncertainty_threshold = max_uncertainty_threshold
        self.min_visibility_threshold = min_visibility_threshold
        self.stale_data_timeout_sec = stale_data_timeout_sec

    def evaluate(
        self,
        collision_data: Optional[Dict[str, Any]],
        weather_intel: Optional[Dict[str, Any]],
        trajectories: Optional[List[Dict[str, Any]]],
        active_tracks: Optional[List[Dict[str, Any]]],
        timestamp_seconds: float,
        previous_recommendation: Optional[SafetyRecommendation] = None,
    ) -> SafetyRecommendation:
        """Evaluate upstream sensor & risk telemetry and generate an explainable recommendation."""
        # 1. Check for Missing or Invalid Upstream Data -> UNKNOWN
        if collision_data is None or weather_intel is None:
            return self._build_unknown_recommendation(
                timestamp_seconds=timestamp_seconds,
                reason="Required upstream telemetry (collision or weather data) is missing.",
            )

        # 2. Check for Timestamp Freshness / Stale Assessment -> UNKNOWN
        col_timestamp = collision_data.get("timestamp_seconds")
        if col_timestamp is not None and abs(timestamp_seconds - col_timestamp) > self.stale_data_timeout_sec:
            return self._build_unknown_recommendation(
                timestamp_seconds=timestamp_seconds,
                reason=f"Stale collision risk assessment detected (age delta {abs(timestamp_seconds - col_timestamp):.2f}s > {self.stale_data_timeout_sec}s).",
            )

        # 3. Extract core metrics
        risk_state = str(collision_data.get("risk_state", "UNKNOWN")).upper()
        fused_risk_score = float(collision_data.get("fused_risk_score", 0.0))
        min_ttc = collision_data.get("min_ttc_seconds")
        if min_ttc is not None:
            min_ttc = float(min_ttc)
        closest_distance_m = collision_data.get("closest_distance_m")
        if closest_distance_m is not None:
            closest_distance_m = float(closest_distance_m)
        closest_track_id = collision_data.get("closest_track_id")
        conflicting_pairs = collision_data.get("conflicting_pairs", [])

        vis_score = float(weather_intel.get("visibility_score", 100.0))
        vis_cond = str(weather_intel.get("condition") or weather_intel.get("primary_condition") or "CLEAR").upper()
        uncertainty_score = float(
            collision_data.get("uncertainty_score")
            or collision_data.get("perception_uncertainty_factor")
            or weather_intel.get("uncertainty")
            or 0.15
        )
        uncertainty_level = str(collision_data.get("uncertainty_level", "LOW")).upper()
        if uncertainty_score >= 0.70:
            uncertainty_level = "CRITICAL"
        elif uncertainty_score >= 0.45:
            uncertainty_level = "HIGH"
        elif uncertainty_score >= 0.25:
            uncertainty_level = "MEDIUM"
        else:
            uncertainty_level = "LOW"

        detection_reliability = float(collision_data.get("detection_reliability", 1.0 - uncertainty_score))

        # Check for invalid trajectory predictions or contradictory signals
        trajectories_list = trajectories if trajectories is not None else []
        tracks_list = active_tracks if active_tracks is not None else []

        # If tracks exist but trajectories are completely empty/missing in multi-frame context
        if len(tracks_list) > 0 and trajectories is None:
            return self._build_unknown_recommendation(
                timestamp_seconds=timestamp_seconds,
                reason="Object tracks exist but trajectory prediction subsystem output is missing.",
            )

        # 4. Check for UNKNOWN state triggers from upstream or high uncertainty
        if (
            risk_state == "UNKNOWN"
            or uncertainty_score >= self.max_uncertainty_threshold
            or vis_score < self.min_visibility_threshold
        ):
            reason_parts: List[str] = []
            if risk_state == "UNKNOWN":
                reason_parts.append("Collision risk classifier returned UNKNOWN state")
            if uncertainty_score >= self.max_uncertainty_threshold:
                reason_parts.append(f"Perception uncertainty excessive ({uncertainty_score*100:.0f}% >= {self.max_uncertainty_threshold*100:.0f}%)")
            if vis_score < self.min_visibility_threshold:
                reason_parts.append(f"Severe camera visibility blackout ({vis_score:.1f}% < {self.min_visibility_threshold:.1f}%)")

            return self._build_unknown_recommendation(
                timestamp_seconds=timestamp_seconds,
                reason="; ".join(reason_parts) + ". Reliable safety advisory cannot be defensibly established.",
                supporting_evidence=SupportingEvidence(
                    trajectory_conflict=len(conflicting_pairs) > 0,
                    min_ttc_seconds=min_ttc,
                    closest_distance_m=closest_distance_m,
                    camera_visibility_score=vis_score,
                    camera_condition=vis_cond,
                    detection_reliability=detection_reliability,
                    uncertainty_level=uncertainty_level,
                    uncertainty_score=uncertainty_score,
                ),
            )

        # 5. Build involved objects list
        involved_objects = self._extract_involved_objects(
            closest_track_id=closest_track_id,
            closest_distance_m=closest_distance_m,
            conflicting_pairs=conflicting_pairs,
            trajectories=trajectories_list,
            tracks=tracks_list,
        )

        # Determine dominant prediction horizon
        pred_horizon = "N/A"
        pred_horizon_sec: Optional[float] = None
        if conflicting_pairs:
            h_str = conflicting_pairs[0].get("horizon", "h3")
            pred_horizon = f"{h_str.upper()}"
            pred_horizon_sec = 1.0 if "1" in h_str else (3.0 if "3" in h_str else 5.0)

        # Relative speed approximation
        rel_speed_kmh: Optional[float] = None
        for obj in involved_objects:
            if obj.speed_kmh is not None and (rel_speed_kmh is None or obj.speed_kmh > rel_speed_kmh):
                rel_speed_kmh = obj.speed_kmh

        has_pedestrian_conflict = any(
            obj.object_class.lower() in ("person", "pedestrian") for obj in involved_objects
        ) or any(
            p.get("class_a") in ("person", "pedestrian") or p.get("class_b") in ("person", "pedestrian")
            for p in conflicting_pairs
        )

        has_vehicle_conflict = any(
            obj.object_class.lower() in ("car", "truck", "bus", "motorcycle") for obj in involved_objects
        )

        supporting_evidence = SupportingEvidence(
            trajectory_conflict=len(conflicting_pairs) > 0,
            conflict_type="PEDESTRIAN_CROSSING" if has_pedestrian_conflict else ("MULTI_VEHICLE" if len(conflicting_pairs) > 0 else None),
            min_ttc_seconds=min_ttc,
            closest_distance_m=closest_distance_m,
            prediction_horizon_seconds=pred_horizon_sec,
            camera_visibility_score=vis_score,
            camera_condition=vis_cond,
            detection_reliability=detection_reliability,
            uncertainty_level=uncertainty_level,
            uncertainty_score=uncertainty_score,
        )

        # ----------------------------------------------------------------------
        # Rule Branch E: STOP RECOMMENDED
        # Criteria:
        # - Strong reliable evidence of imminent danger:
        #   (risk_state == "CRITICAL" and uncertainty < 0.65)
        #   AND (min_ttc <= critical_ttc_threshold OR (closest_distance_m <= 6.0 and rel_speed_kmh and rel_speed_kmh > 5.0))
        # ----------------------------------------------------------------------
        is_imminent_ttc = (min_ttc is not None and min_ttc <= self.critical_ttc_threshold and min_ttc > 0.0)
        is_imminent_close = (closest_distance_m is not None and closest_distance_m < 6.0)

        if (risk_state == "CRITICAL" and uncertainty_score < 0.65) and (is_imminent_ttc or is_imminent_close):
            ttc_str = f"TTC {min_ttc:.1f}s" if min_ttc is not None else "imminent spatial convergence"
            reason = (
                f"Critical collision hazard detected ({ttc_str} <= {self.critical_ttc_threshold}s). "
                f"Active trajectory intersection supported by low uncertainty ({uncertainty_score*100:.0f}%)."
            )
            return SafetyRecommendation(
                warning_id=f"warn_{uuid.uuid4().hex[:8]}",
                timestamp_seconds=timestamp_seconds,
                recommendation_state=RecommendationState.STOP_RECOMMENDED,
                priority=WarningPriority.CRITICAL,
                risk_level=risk_state,
                objects_involved=involved_objects,
                recommended_action="Stop vehicle when safe to do so. Immediate collision hazard in projected travel corridor.",
                primary_reason=reason,
                supporting_evidence=supporting_evidence,
                prediction_horizon=pred_horizon,
                min_ttc_seconds=min_ttc,
                relative_speed_kmh=rel_speed_kmh,
                camera_visibility_status="Degraded" if vis_score < 70.0 else "Nominal",
                detection_reliability=detection_reliability,
                uncertainty_level=uncertainty_level,
                status=WarningStatus.ACTIVE,
            )

        # ----------------------------------------------------------------------
        # Rule Branch D: PREPARE TO STOP
        # Criteria:
        # - High collision risk developing:
        #   (risk_state == "HIGH" or (min_ttc is not None and min_ttc <= warning_ttc_threshold))
        #   or approaching high-risk trajectory conflict in H3/H5 horizon with decreasing separation
        # ----------------------------------------------------------------------
        is_warning_ttc = (min_ttc is not None and min_ttc <= self.warning_ttc_threshold and min_ttc > 0.0)
        if (
            risk_state in ("HIGH", "CRITICAL")
            or is_warning_ttc
            or (fused_risk_score >= 0.60 and len(conflicting_pairs) > 0)
        ):
            ttc_str = f"TTC {min_ttc:.1f}s" if min_ttc is not None else "projected convergence"
            reason = (
                f"Elevated collision potential approaching ({ttc_str}). "
                f"Active road agent in projected travel corridor requiring brake preparation."
            )
            return SafetyRecommendation(
                warning_id=f"warn_{uuid.uuid4().hex[:8]}",
                timestamp_seconds=timestamp_seconds,
                recommendation_state=RecommendationState.PREPARE_TO_STOP,
                priority=WarningPriority.WARNING,
                risk_level=risk_state,
                objects_involved=involved_objects,
                recommended_action="Prepare for controlled stop. Road agent converging in projected path.",
                primary_reason=reason,
                supporting_evidence=supporting_evidence,
                prediction_horizon=pred_horizon,
                min_ttc_seconds=min_ttc,
                relative_speed_kmh=rel_speed_kmh,
                camera_visibility_status="Degraded" if vis_score < 70.0 else "Nominal",
                detection_reliability=detection_reliability,
                uncertainty_level=uncertainty_level,
                status=WarningStatus.ACTIVE,
            )

        # ----------------------------------------------------------------------
        # Rule Branch C: REDUCE SPEED
        # Criteria:
        # - Moderate risk escalation from Feature 4
        # - Vehicle-pedestrian conflict developing within horizon
        # - Decreasing separation / dangerous trajectory interaction
        # - Reduced visibility (< 65%) COMBINED WITH relevant road-agent risk (close obstacles or active conflict)
        # Note: Do NOT recommend reducing speed solely for bad weather if ZERO objects/risk present!
        # ----------------------------------------------------------------------
        road_agent_at_risk = (
            (closest_distance_m is not None and closest_distance_m < 20.0)
            or (min_ttc is not None and min_ttc < 6.0)
            or risk_state in ("MODERATE", "HIGH", "CRITICAL")
        )
        weather_combined_risk = (vis_score < 65.0 or vis_cond in ("FOG", "RAIN", "LOW_LIGHT")) and road_agent_at_risk

        if (
            risk_state == "MODERATE"
            or has_pedestrian_conflict
            or len(conflicting_pairs) > 0
            or weather_combined_risk
        ):
            if has_pedestrian_conflict:
                reason = "Potential vehicle-pedestrian trajectory interaction developing within prediction horizon."
                action = "Reduce vehicle speed and yield right-of-way to approaching pedestrian."
            elif weather_combined_risk and vis_score < 65.0:
                reason = f"Deteriorated visibility ({vis_score:.0f}%, {vis_cond}) with active road agents in roadway."
                action = "Reduce vehicle speed to expand safety buffer under compromised optical conditions."
            elif len(conflicting_pairs) > 0:
                reason = f"Trajectory conflict detected between road agents in upcoming {pred_horizon} horizon."
                action = "Reduce vehicle speed and increase following distance."
            else:
                reason = "Moderate risk escalation: narrowing trajectory clearance to active road agents."
                action = "Reduce speed and maintain heightened situational awareness."

            return SafetyRecommendation(
                warning_id=f"warn_{uuid.uuid4().hex[:8]}",
                timestamp_seconds=timestamp_seconds,
                recommendation_state=RecommendationState.REDUCE_SPEED,
                priority=WarningPriority.ADVISORY,
                risk_level=risk_state,
                objects_involved=involved_objects,
                recommended_action=action,
                primary_reason=reason,
                supporting_evidence=supporting_evidence,
                prediction_horizon=pred_horizon,
                min_ttc_seconds=min_ttc,
                relative_speed_kmh=rel_speed_kmh,
                camera_visibility_status="Degraded" if vis_score < 70.0 else "Nominal",
                detection_reliability=detection_reliability,
                uncertainty_level=uncertainty_level,
                status=WarningStatus.ACTIVE,
            )

        # ----------------------------------------------------------------------
        # Rule Branch B: CAUTION
        # Criteria:
        # - Visibility deteriorating or adverse weather without road agent collision threat
        # - Tracking quality instability or elevated uncertainty (0.35 <= U < 0.70)
        # - Note: Clean normal road with detected vehicles or pedestrians maintaining safe headway
        #   is evaluated as NORMAL, NOT Caution!
        # ----------------------------------------------------------------------
        has_caution_condition = (
            (vis_score < 65.0 and vis_cond != "CLEAR")
            or uncertainty_score >= 0.40
            or detection_reliability < 0.60
            or (risk_state == "MODERATE")
        )

        if has_caution_condition:
            if vis_score < 65.0 and vis_cond != "CLEAR":
                reason = f"Environmental optical degradation detected ({vis_cond}, {vis_score:.0f}% visibility); safe separation maintained."
                action = "Pay increased attention; adverse atmospheric/camera conditions present."
            elif uncertainty_score >= 0.40:
                reason = f"Elevated perception uncertainty ({uncertainty_score*100:.0f}%); monitor road environment."
                action = "Pay increased attention; perception confidence fluctuating."
            else:
                reason = "Moderate corridor activity detected; safe headway currently maintained."
                action = "Pay increased attention and maintain active scan of adjacent corridors."

            return SafetyRecommendation(
                warning_id=f"warn_{uuid.uuid4().hex[:8]}",
                timestamp_seconds=timestamp_seconds,
                recommendation_state=RecommendationState.CAUTION,
                priority=WarningPriority.ADVISORY,
                risk_level=risk_state,
                objects_involved=involved_objects,
                recommended_action=action,
                primary_reason=reason,
                supporting_evidence=supporting_evidence,
                prediction_horizon="N/A",
                min_ttc_seconds=min_ttc,
                relative_speed_kmh=rel_speed_kmh,
                camera_visibility_status="Degraded" if vis_score < 70.0 else "Nominal",
                detection_reliability=detection_reliability,
                uncertainty_level=uncertainty_level,
                status=WarningStatus.ACTIVE,
            )

        # ----------------------------------------------------------------------
        # Rule Branch A: NORMAL
        # Criteria:
        # - Normal road conditions, vehicles following lanes, safe separation, zero conflicts.
        # ----------------------------------------------------------------------
        return SafetyRecommendation(
            warning_id=f"warn_{uuid.uuid4().hex[:8]}",
            timestamp_seconds=timestamp_seconds,
            recommendation_state=RecommendationState.NORMAL,
            priority=WarningPriority.INFO,
            risk_level=risk_state,
            objects_involved=[],
            recommended_action="Maintain nominal speed and road vigilance.",
            primary_reason="Driving conditions normal: Road users following nominal trajectories; zero trajectory conflicts detected.",
            supporting_evidence=supporting_evidence,
            prediction_horizon="N/A",
            min_ttc_seconds=None,
            relative_speed_kmh=None,
            camera_visibility_status="Nominal",
            detection_reliability=detection_reliability,
            uncertainty_level=uncertainty_level,
            status=WarningStatus.ACTIVE,
        )

    def _build_unknown_recommendation(
        self,
        timestamp_seconds: float,
        reason: str,
        supporting_evidence: Optional[SupportingEvidence] = None,
    ) -> SafetyRecommendation:
        """Construct an explicit UNKNOWN advisory recommendation."""
        ev = supporting_evidence or SupportingEvidence(
            uncertainty_level="CRITICAL",
            uncertainty_score=1.0,
            camera_visibility_score=0.0,
            camera_condition="UNKNOWN",
            detection_reliability=0.0,
        )
        return SafetyRecommendation(
            warning_id=f"warn_{uuid.uuid4().hex[:8]}",
            timestamp_seconds=timestamp_seconds,
            recommendation_state=RecommendationState.UNKNOWN,
            priority=WarningPriority.ADVISORY,
            risk_level="UNKNOWN",
            objects_involved=[],
            recommended_action="Sensor/Perception unreliability detected. Exercise manual visual caution; automated advisory unavailable.",
            primary_reason=reason,
            supporting_evidence=ev,
            prediction_horizon="N/A",
            min_ttc_seconds=None,
            relative_speed_kmh=None,
            camera_visibility_status="Unreliable",
            detection_reliability=ev.detection_reliability,
            uncertainty_level=ev.uncertainty_level,
            status=WarningStatus.DEGRADED,
        )

    def _extract_involved_objects(
        self,
        closest_track_id: Optional[int],
        closest_distance_m: Optional[float],
        conflicting_pairs: List[Dict[str, Any]],
        trajectories: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
    ) -> List[InvolvedObject]:
        """Resolve metadata for road agents participating in the risk assessment."""
        objs_map: Dict[int, InvolvedObject] = {}

        # Build quick track_id -> track dict lookup
        tracks_by_id: Dict[int, Dict[str, Any]] = {}
        for trk in tracks:
            tid = trk.get("track_id")
            if tid is not None:
                tracks_by_id[int(tid)] = trk

        for traj in trajectories:
            tid = traj.get("track_id")
            if tid is not None:
                tid = int(tid)
                c_name = traj.get("class_name", "object")
                spd = traj.get("speed_kmh")
                if tid not in objs_map:
                    objs_map[tid] = InvolvedObject(
                        track_id=tid,
                        object_class=c_name,
                        distance_m=None,
                        speed_kmh=spd,
                    )

        # Highlight closest track only if actual tracks or trajectories exist
        if closest_track_id is not None and (len(trajectories) > 0 or len(tracks) > 0):
            cid = int(closest_track_id)
            if cid in objs_map:
                objs_map[cid].distance_m = closest_distance_m
            else:
                trk_info = tracks_by_id.get(cid, {})
                c_name = trk_info.get("class_name", "object")
                objs_map[cid] = InvolvedObject(
                    track_id=cid,
                    object_class=c_name,
                    distance_m=closest_distance_m,
                    speed_kmh=None,
                )

        # Highlight conflicting pair agents
        for pair in conflicting_pairs:
            for k_id, k_cls in [("track_a", "class_a"), ("track_b", "class_b")]:
                if k_id in pair:
                    pid = int(pair[k_id])
                    p_cls = pair.get(k_cls, "object")
                    if pid in objs_map:
                        objs_map[pid].object_class = p_cls
                    else:
                        objs_map[pid] = InvolvedObject(
                            track_id=pid,
                            object_class=p_cls,
                            distance_m=None,
                            speed_kmh=None,
                        )

        return list(objs_map.values())
