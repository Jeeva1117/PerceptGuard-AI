"""PerceptGuard AI: Advanced Pedestrian-Aware Continuous Driving Intelligence Engine.

Implements the five specified pedestrian awareness decision states:
1. PERSON DETECTED (Sidewalk / safe separation -> LOW risk, NO ACTIVE WARNING)
2. PEDESTRIAN APPROACHING ROAD (Approaching roadway -> CAUTION, MONITOR AND REDUCE SPEED)
3. PEDESTRIAN CROSSING (Crossing roadway -> UNDER EVALUATION)
4. POTENTIAL PEDESTRIAN COLLISION RISK (Predicted travel path conflict -> HIGH/CRITICAL, PREPARE TO STOP)
5. PEDESTRIAN TRACKING LOST (Occluded or lost tracking -> UNKNOWN, MAINTAIN CAUTION)

Features:
- Directional spatial tracking (left vs right road side).
- Persistent ByteTrack track lifecycle and occlusion tracking.
- Context-appropriate voice alert generation with cooldown.
- Visual state telemetry for canvas overlays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.collision_risk.road_context_analyzer import (
    LaneState,
    PedestrianMovementIntent,
    PedestrianRoadPosition,
    RoadContextAnalyzer,
    RoadContextData,
)


class PedestrianAwarenessState(str, Enum):
    PERSON_DETECTED = "PERSON DETECTED"
    PEDESTRIAN_APPROACHING_ROAD = "PEDESTRIAN APPROACHING ROAD"
    PEDESTRIAN_CROSSING = "PEDESTRIAN CROSSING DETECTED"
    POTENTIAL_PEDESTRIAN_COLLISION_RISK = "POTENTIAL PEDESTRIAN COLLISION RISK"
    PEDESTRIAN_TRACKING_LOST = "PEDESTRIAN TRACKING LOST"
    NONE_DETECTED = "NO PEDESTRIANS IN SCENE"


@dataclass
class PedestrianObjectDetail:
    track_id: int
    bbox: List[float]
    center: Tuple[float, float]
    confidence: float
    road_position: PedestrianRoadPosition
    movement_intent: PedestrianMovementIntent
    state: PedestrianAwarenessState
    risk_level: str  # LOW, CAUTION, MODERATE, HIGH, CRITICAL, UNKNOWN
    speed_kmh: float
    distance_to_corridor_px: float
    is_in_ego_path: bool
    ttc_seconds: Optional[float]
    side: str  # "LEFT", "RIGHT", "CENTER"
    recommended_action: str
    voice_prompt: Optional[str]
    tracking_age_frames: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": [round(c, 1) for c in self.bbox],
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "confidence": round(self.confidence, 3),
            "road_position": self.road_position.value,
            "movement_intent": self.movement_intent.value,
            "state": self.state.value,
            "risk_level": self.risk_level,
            "speed_kmh": round(self.speed_kmh, 1),
            "distance_to_corridor_px": round(self.distance_to_corridor_px, 1),
            "is_in_ego_path": self.is_in_ego_path,
            "ttc_seconds": round(self.ttc_seconds, 2) if self.ttc_seconds is not None else None,
            "side": self.side,
            "recommended_action": self.recommended_action,
            "voice_prompt": self.voice_prompt,
            "tracking_age_frames": self.tracking_age_frames,
        }


@dataclass
class FramePedestrianTelemetry:
    frame_index: int
    timestamp_seconds: float
    pedestrians_count: int
    primary_state: PedestrianAwarenessState
    active_pedestrians: List[PedestrianObjectDetail]
    lost_pedestrian_ids: List[int]
    highest_risk_level: str
    recommended_action: str
    primary_voice_alert: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "pedestrians_count": self.pedestrians_count,
            "primary_state": self.primary_state.value,
            "active_pedestrians": [p.to_dict() for p in self.active_pedestrians],
            "lost_pedestrian_ids": self.lost_pedestrian_ids,
            "highest_risk_level": self.highest_risk_level,
            "recommended_action": self.recommended_action,
            "primary_voice_alert": self.primary_voice_alert,
        }


class PedestrianAwarenessEngine:
    """Engine continuously evaluating pedestrian presence, movement dynamics, and spatial hazard interaction."""

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._road_analyzer = RoadContextAnalyzer(frame_width=frame_width, frame_height=frame_height)

        # Track history of previously active pedestrian track IDs for lost tracking detection
        self._prev_active_pedestrian_ids: Set[int] = set()
        self._track_consecutive_missing: Dict[int, int] = {}
        self._all_frame_telemetry: List[FramePedestrianTelemetry] = []

    def evaluate_frame(
        self,
        active_tracks: List[Dict[str, Any]],
        trajectories: List[Dict[str, Any]],
        road_ctx: Union[RoadContextData, Dict[str, Any]],
        timestamp_seconds: float,
        frame_index: int,
        fps: float = 10.0,
    ) -> FramePedestrianTelemetry:
        """Evaluate all pedestrian tracks in the current frame and assign awareness states."""
        w, h = self.frame_width, self.frame_height

        if isinstance(road_ctx, dict):
            road_ctx_obj = RoadContextData(
                lane_state=LaneState(road_ctx.get("lane_state", "LANE DETECTED")),
                left_boundary_x=float(road_ctx.get("left_boundary_x", w * 0.25)),
                right_boundary_x=float(road_ctx.get("right_boundary_x", w * 0.75)),
                lane_center_px=float(road_ctx.get("lane_center_px", w / 2.0)),
                lane_width_px=float(road_ctx.get("lane_width_px", w * 0.5)),
                ego_lane_offset_px=float(road_ctx.get("ego_lane_offset_px", 0.0)),
                road_surface_quality=str(road_ctx.get("road_surface_quality", "DRY")),
                crosswalk_detected=bool(road_ctx.get("crosswalk_detected", False)),
                confidence=float(road_ctx.get("confidence", 0.85)),
            )
        else:
            road_ctx_obj = road_ctx

        ego_corridor_left = road_ctx_obj.left_boundary_x
        ego_corridor_right = road_ctx_obj.right_boundary_x
        ego_center_x = w / 2.0

        current_active_ped_ids: Set[int] = set()
        pedestrian_details: List[PedestrianObjectDetail] = []

        # Map trajectories by track_id for fast lookup
        traj_by_id = {t["track_id"]: t for t in trajectories}

        for trk in active_tracks:
            cls_name = trk.get("class_name", "").lower()
            if cls_name not in ("person", "pedestrian"):
                continue

            tid = trk["track_id"]
            current_active_ped_ids.add(tid)
            bbox = trk.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            conf = float(trk.get("confidence", 0.85))

            # Determine side of road
            side = "RIGHT" if cx >= ego_center_x else "LEFT"

            # 1. Road position classification
            pos = self._road_analyzer.classify_pedestrian_position(cx, cy, bbox, road_ctx_obj)

            # 2. Movement intent classification
            traj_data = traj_by_id.get(tid, {})
            history = traj_data.get("history", [])
            speed_kmh = float(traj_data.get("speed_kmh", 0.0))
            intent = self._road_analyzer.classify_movement_intent(history, pos, fps, road_ctx_obj)

            # 3. Trajectory corridor interaction check
            # Distance from ego travel corridor
            if cx < ego_corridor_left:
                dist_to_corridor = ego_corridor_left - cx
            elif cx > ego_corridor_right:
                dist_to_corridor = cx - ego_corridor_right
            else:
                dist_to_corridor = 0.0

            is_in_ego_path = (dist_to_corridor == 0.0)

            # Check future predicted waypoints
            pred_wps = traj_data.get("primary_trajectory", [])
            will_enter_corridor = False
            time_to_corridor_sec: Optional[float] = None

            for wp in pred_wps:
                wp_x = float(wp.get("x_px", wp.get("x", cx)))
                wp_t = float(wp.get("horizon_seconds", 3.0))
                if ego_corridor_left <= wp_x <= ego_corridor_right:
                    will_enter_corridor = True
                    if time_to_corridor_sec is None or wp_t < time_to_corridor_sec:
                        time_to_corridor_sec = wp_t

            # TTC estimate if in path or entering path
            ttc: Optional[float] = None
            if is_in_ego_path or will_enter_corridor:
                ttc = time_to_corridor_sec if time_to_corridor_sec is not None else 2.5

            # 4. State decision logic (5 strict states)
            if (is_in_ego_path and ttc is not None and ttc <= 3.5) or (will_enter_corridor and ttc is not None and ttc <= 2.5):
                # State D: Potential Pedestrian Collision Risk
                state = PedestrianAwarenessState.POTENTIAL_PEDESTRIAN_COLLISION_RISK
                risk_level = "CRITICAL" if (ttc is not None and ttc <= 1.8) else "HIGH"
                action = "Reduce speed and prepare to stop when safe. Pedestrian conflict in travel path."
                voice = "High pedestrian collision risk. Reduce speed and prepare to stop."
            elif intent == PedestrianMovementIntent.CROSSING or pos == PedestrianRoadPosition.CROSSWALK:
                # State C: Pedestrian Crossing
                state = PedestrianAwarenessState.PEDESTRIAN_CROSSING
                risk_level = "MODERATE"
                action = "Pedestrian crossing detected. Monitor corridor and prepare to yield."
                voice = "Warning. Pedestrian crossing ahead."
            elif intent == PedestrianMovementIntent.APPROACHING_ROAD or pos == PedestrianRoadPosition.ROAD_EDGE:
                # State B: Pedestrian Approaching Road
                state = PedestrianAwarenessState.PEDESTRIAN_APPROACHING_ROAD
                risk_level = "CAUTION"
                action = "Monitor pedestrian trajectory and reduce speed if necessary."
                voice = "Pedestrian approaching the roadway."
            else:
                # State A: Person Detected (Safe on Sidewalk)
                state = PedestrianAwarenessState.PERSON_DETECTED
                risk_level = "LOW"
                action = "Nominal driving speed maintained. Pedestrian operating safely on sidewalk/periphery."
                voice = None  # No voice alert required for safe sidewalk presence!

            ped_detail = PedestrianObjectDetail(
                track_id=tid,
                bbox=bbox,
                center=(cx, cy),
                confidence=conf,
                road_position=pos,
                movement_intent=intent,
                state=state,
                risk_level=risk_level,
                speed_kmh=speed_kmh,
                distance_to_corridor_px=dist_to_corridor,
                is_in_ego_path=is_in_ego_path,
                ttc_seconds=ttc,
                side=side,
                recommended_action=action,
                voice_prompt=voice,
                tracking_age_frames=len(history),
            )
            pedestrian_details.append(ped_detail)

        # 5. Handle tracking loss (State E: Pedestrian Tracking Lost)
        lost_ids: List[int] = []
        for prev_id in self._prev_active_pedestrian_ids:
            if prev_id not in current_active_ped_ids:
                count = self._track_consecutive_missing.get(prev_id, 0) + 1
                self._track_consecutive_missing[prev_id] = count
                if count <= 10:  # Report lost for up to 10 frames after disappearance
                    lost_ids.append(prev_id)
            else:
                self._track_consecutive_missing[prev_id] = 0

        self._prev_active_pedestrian_ids = current_active_ped_ids

        # 6. Synthesize frame-level pedestrian telemetry
        if not pedestrian_details:
            if lost_ids:
                primary_state = PedestrianAwarenessState.PEDESTRIAN_TRACKING_LOST
                highest_risk = "UNKNOWN"
                rec_action = "Pedestrian tracking lost due to occlusion. Maintain caution."
                voice = "Pedestrian tracking lost. Maintain caution."
            else:
                primary_state = PedestrianAwarenessState.NONE_DETECTED
                highest_risk = "LOW"
                rec_action = "No pedestrians visible in forward driving scene."
                voice = None
        else:
            # Pick highest risk state among active pedestrians
            risk_priority = {
                PedestrianAwarenessState.POTENTIAL_PEDESTRIAN_COLLISION_RISK: 4,
                PedestrianAwarenessState.PEDESTRIAN_CROSSING: 3,
                PedestrianAwarenessState.PEDESTRIAN_APPROACHING_ROAD: 2,
                PedestrianAwarenessState.PERSON_DETECTED: 1,
                PedestrianAwarenessState.PEDESTRIAN_TRACKING_LOST: 0,
                PedestrianAwarenessState.NONE_DETECTED: 0,
            }
            pedestrian_details.sort(key=lambda p: risk_priority.get(p.state, 0), reverse=True)
            top_ped = pedestrian_details[0]
            primary_state = top_ped.state
            highest_risk = top_ped.risk_level
            rec_action = top_ped.recommended_action
            voice = top_ped.voice_prompt

        telemetry = FramePedestrianTelemetry(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            pedestrians_count=len(pedestrian_details),
            primary_state=primary_state,
            active_pedestrians=pedestrian_details,
            lost_pedestrian_ids=lost_ids,
            highest_risk_level=highest_risk,
            recommended_action=rec_action,
            primary_voice_alert=voice,
        )
        self._all_frame_telemetry.append(telemetry)
        return telemetry

    def generate_summary(self) -> Dict[str, Any]:
        """Generate high-level pedestrian awareness summary across analyzed sequence."""
        if not self._all_frame_telemetry:
            return {
                "total_unique_pedestrians": 0,
                "frames_with_pedestrians": 0,
                "primary_state_distribution": {},
                "potential_collision_events": 0,
                "crossings_observed": 0,
                "approaches_observed": 0,
                "lost_tracking_events": 0,
            }

        all_ids: Set[int] = set()
        frames_with_peds = 0
        state_counts: Dict[str, int] = {}
        collision_events = 0
        crossings = 0
        approaches = 0
        lost_events = 0

        for f in self._all_frame_telemetry:
            st = f.primary_state.value
            state_counts[st] = state_counts.get(st, 0) + 1
            if f.pedestrians_count > 0:
                frames_with_peds += 1
            for p in f.active_pedestrians:
                all_ids.add(p.track_id)
            if f.primary_state == PedestrianAwarenessState.POTENTIAL_PEDESTRIAN_COLLISION_RISK:
                collision_events += 1
            elif f.primary_state == PedestrianAwarenessState.PEDESTRIAN_CROSSING:
                crossings += 1
            elif f.primary_state == PedestrianAwarenessState.PEDESTRIAN_APPROACHING_ROAD:
                approaches += 1
            if f.lost_pedestrian_ids:
                lost_events += len(f.lost_pedestrian_ids)

        return {
            "total_unique_pedestrians": len(all_ids),
            "frames_with_pedestrians": frames_with_peds,
            "primary_state_distribution": state_counts,
            "potential_collision_events": collision_events,
            "crossings_observed": crossings,
            "approaches_observed": approaches,
            "lost_tracking_events": lost_events,
        }
