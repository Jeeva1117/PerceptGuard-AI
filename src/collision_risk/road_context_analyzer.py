"""PerceptGuard AI: Road Surface, Lane Boundary & Context Analyzer.

Analyzes:
1. Road surface and perspective lane boundaries using computer vision Hough line fitting.
2. Lane detection states: LANE DETECTED, LANE CHANGE OBSERVED, LANE DEPARTURE SUSPECTED,
   LANE NOT VISIBLE, UNKNOWN.
3. Scene spatial segmentation into:
   - Left Sidewalk
   - Left Road Edge
   - Ego Travel Lane Corridor
   - Right Road Edge
   - Right Sidewalk
   - Crosswalk zones
4. Categorization of road agents and pedestrians into spatial zones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class LaneState(str, Enum):
    LANE_DETECTED = "LANE DETECTED"
    LANE_CHANGE_OBSERVED = "LANE CHANGE OBSERVED"
    LANE_DEPARTURE_SUSPECTED = "LANE DEPARTURE SUSPECTED"
    LANE_NOT_VISIBLE = "LANE NOT VISIBLE"
    UNKNOWN = "UNKNOWN"


class PedestrianRoadPosition(str, Enum):
    SIDEWALK = "SIDEWALK"
    CROSSWALK = "CROSSWALK"
    ROAD_EDGE = "ROAD EDGE"
    ROADWAY = "ROADWAY"
    UNKNOWN = "UNKNOWN"


class PedestrianMovementIntent(str, Enum):
    STATIONARY = "STATIONARY"
    ALONG_SIDEWALK = "ALONG_SIDEWALK"
    APPROACHING_ROAD = "APPROACHING_ROAD"
    CROSSING = "CROSSING"
    MOVING_AWAY = "MOVING_AWAY"
    UNKNOWN = "UNKNOWN"


@dataclass
class RoadContextData:
    lane_state: LaneState
    left_boundary_x: float
    right_boundary_x: float
    lane_center_px: float
    lane_width_px: float
    ego_lane_offset_px: float
    road_surface_quality: str
    crosswalk_detected: bool
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_state": self.lane_state.value,
            "left_boundary_x": round(self.left_boundary_x, 1),
            "right_boundary_x": round(self.right_boundary_x, 1),
            "lane_center_px": round(self.lane_center_px, 1),
            "lane_width_px": round(self.lane_width_px, 1),
            "ego_lane_offset_px": round(self.ego_lane_offset_px, 1),
            "road_surface_quality": self.road_surface_quality,
            "crosswalk_detected": self.crosswalk_detected,
            "confidence": round(self.confidence, 3),
        }


class RoadContextAnalyzer:
    """Production computer vision engine for road context, lane boundaries, and spatial zone mapping."""

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        horizon_ratio: float = 0.45,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.horizon_ratio = horizon_ratio

        # Default geometric road boundaries when line markings are faint/absent
        self.default_left_ratio = 0.22
        self.default_right_ratio = 0.78
        self.edge_margin_ratio = 0.08  # Width of transition road-edge zone
        self._history: List[RoadContextData] = []

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        visibility_score: float = 100.0,
    ) -> RoadContextData:
        """Extract lane boundaries and road surface context from video frame."""
        h, w = frame_bgr.shape[:2]
        horizon_y = int(h * self.horizon_ratio)

        if visibility_score < 30.0:
            # Severely degraded optical perception
            default_left = w * self.default_left_ratio
            default_right = w * self.default_right_ratio
            data = RoadContextData(
                lane_state=LaneState.UNKNOWN,
                left_boundary_x=default_left,
                right_boundary_x=default_right,
                lane_center_px=(default_left + default_right) / 2.0,
                lane_width_px=default_right - default_left,
                ego_lane_offset_px=0.0,
                road_surface_quality="UNKNOWN",
                crosswalk_detected=False,
                confidence=0.20,
            )
            self._history.append(data)
            return data

        # 1. Define lower road Region of Interest (ROI)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr
        road_roi = gray[horizon_y:, :]

        # 2. Edge and gradient analysis for converging linear lane markings
        blurred = cv2.GaussianBlur(road_roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Hough transform for lane lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=int(h * 0.12),
            maxLineGap=25,
        )

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                l = line[0] if (hasattr(line, "__len__") and len(line) == 1) else line
                if not hasattr(l, "__len__") or len(l) < 4:
                    continue
                x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
                # Adjust y back to full image coordinates
                y1_full = y1 + horizon_y
                y2_full = y2 + horizon_y
                dx = x2 - x1
                dy = y2 - y1

                if dx == 0:
                    continue
                slope = dy / dx

                # Left lane lines typically have negative slope in image coordinates (-2.0 to -0.3)
                # Right lane lines typically have positive slope (0.3 to 2.0)
                if -2.2 < slope < -0.35 and x1 < w * 0.55:
                    left_lines.append((x1, y1_full, x2, y2_full, slope))
                elif 0.35 < slope < 2.2 and x2 > w * 0.45:
                    right_lines.append((x1, y1_full, x2, y2_full, slope))

        # Check for crosswalk pattern (periodic horizontal stripes in lower-middle ROI)
        crosswalk_roi = edges[int(road_roi.shape[0] * 0.4) : int(road_roi.shape[0] * 0.8), int(w * 0.25) : int(w * 0.75)]
        horiz_proj = np.sum(crosswalk_roi > 0, axis=1)
        peaks = 0
        if len(horiz_proj) > 10:
            mean_p = np.mean(horiz_proj)
            peaks = sum(1 for i in range(1, len(horiz_proj) - 1) if horiz_proj[i] > mean_p * 1.5 and horiz_proj[i] > horiz_proj[i-1] and horiz_proj[i] > horiz_proj[i+1])
        crosswalk_detected = peaks >= 3

        # 3. Determine lane boundaries at bottom of frame (y = h)
        if len(left_lines) >= 1 and len(right_lines) >= 1:
            # Extrapolate left boundary to bottom of image
            left_bottom_xs = []
            for x1, y1, x2, y2, slope in left_lines:
                # y - y1 = slope * (x - x1) => x = x1 + (y - y1) / slope
                x_bot = x1 + (h - y1) / slope
                if 0 <= x_bot <= w * 0.60:
                    left_bottom_xs.append(x_bot)

            right_bottom_xs = []
            for x1, y1, x2, y2, slope in right_lines:
                x_bot = x1 + (h - y1) / slope
                if w * 0.40 <= x_bot <= w:
                    right_bottom_xs.append(x_bot)

            left_x = float(np.median(left_bottom_xs)) if left_bottom_xs else (w * self.default_left_ratio)
            right_x = float(np.median(right_bottom_xs)) if right_bottom_xs else (w * self.default_right_ratio)
            lane_detected = True
            conf = 0.88
        else:
            left_x = w * self.default_left_ratio
            right_x = w * self.default_right_ratio
            lane_detected = False
            conf = 0.50

        # Sanity bounds
        left_x = max(10.0, min(w * 0.45, left_x))
        right_x = max(w * 0.55, min(w - 10.0, right_x))
        lane_width = right_x - left_x
        lane_center = (left_x + right_x) / 2.0
        ego_center = w / 2.0
        ego_offset = ego_center - lane_center

        # Determine lane state
        if lane_detected:
            offset_ratio = abs(ego_offset) / max(1.0, lane_width)
            if offset_ratio > 0.30:
                lane_state = LaneState.LANE_DEPARTURE_SUSPECTED
            elif offset_ratio > 0.18:
                lane_state = LaneState.LANE_CHANGE_OBSERVED
            else:
                lane_state = LaneState.LANE_DETECTED
        else:
            lane_state = LaneState.LANE_NOT_VISIBLE

        # Road surface quality check
        mean_bright = float(np.mean(road_roi))
        std_bright = float(np.std(road_roi))
        if visibility_score > 75.0 and std_bright > 18.0:
            surface_quality = "DRY"
        elif visibility_score <= 50.0:
            surface_quality = "DEGRADED"
        else:
            surface_quality = "NORMAL"

        data = RoadContextData(
            lane_state=lane_state,
            left_boundary_x=left_x,
            right_boundary_x=right_x,
            lane_center_px=lane_center,
            lane_width_px=lane_width,
            ego_lane_offset_px=ego_offset,
            road_surface_quality=surface_quality,
            crosswalk_detected=crosswalk_detected,
            confidence=conf,
        )
        self._history.append(data)
        return data

    def generate_summary(self) -> Dict[str, Any]:
        """Generate high-level summary of road and lane context across analyzed sequence."""
        if not self._history:
            return {
                "dominant_lane_state": "UNKNOWN",
                "lane_detected_percentage": 0.0,
                "crosswalk_observed": False,
                "average_lane_width_px": 0.0,
                "road_surface_quality": "UNKNOWN",
                "state_distribution": {},
            }

        state_counts: Dict[str, int] = {}
        widths: List[float] = []
        surface_counts: Dict[str, int] = {}
        crosswalk_seen = False
        detected_count = 0

        for r in self._history:
            st = r.lane_state.value
            state_counts[st] = state_counts.get(st, 0) + 1
            if r.lane_state in (LaneState.LANE_DETECTED, LaneState.LANE_CHANGE_OBSERVED):
                detected_count += 1
            widths.append(r.lane_width_px)
            surface_counts[r.road_surface_quality] = surface_counts.get(r.road_surface_quality, 0) + 1
            if r.crosswalk_detected:
                crosswalk_seen = True

        dominant_st = max(state_counts.items(), key=lambda x: x[1])[0]
        dominant_surf = max(surface_counts.items(), key=lambda x: x[1])[0]

        return {
            "dominant_lane_state": dominant_st,
            "lane_detected_percentage": round((detected_count / len(self._history)) * 100.0, 1),
            "crosswalk_observed": crosswalk_seen,
            "average_lane_width_px": round(float(np.mean(widths)), 1),
            "road_surface_quality": dominant_surf,
            "state_distribution": state_counts,
        }

    def classify_pedestrian_position(
        self,
        center_x: float,
        center_y: float,
        bbox: List[float],
        road_ctx: RoadContextData,
    ) -> PedestrianRoadPosition:
        """Classify a pedestrian's spatial road zone based on lane boundaries and perspective."""
        h, w = self.frame_height, self.frame_width
        horizon_y = h * self.horizon_ratio

        # Points at or above the horizon cannot be positioned on the roadway
        if center_y < horizon_y:
            return PedestrianRoadPosition.UNKNOWN

        # Perspective interpolation of lane boundaries at this object's y position
        # At horizon: left = right = road_ctx.lane_center_px
        # At bottom: left = road_ctx.left_boundary_x, right = road_ctx.right_boundary_x
        v_fraction = max(0.0, min(1.0, (center_y - horizon_y) / max(1.0, h - horizon_y)))
        left_at_y = road_ctx.lane_center_px + (road_ctx.left_boundary_x - road_ctx.lane_center_px) * v_fraction
        right_at_y = road_ctx.lane_center_px + (road_ctx.right_boundary_x - road_ctx.lane_center_px) * v_fraction

        lane_w_at_y = max(20.0, right_at_y - left_at_y)
        edge_margin = lane_w_at_y * 0.12

        # Check crosswalk
        if road_ctx.crosswalk_detected and (left_at_y - edge_margin) <= center_x <= (right_at_y + edge_margin):
            return PedestrianRoadPosition.CROSSWALK

        # Check roadway corridor
        if (left_at_y + edge_margin) <= center_x <= (right_at_y - edge_margin):
            return PedestrianRoadPosition.ROADWAY

        # Check road edge
        if (left_at_y - edge_margin) <= center_x <= (left_at_y + edge_margin) or \
           (right_at_y - edge_margin) <= center_x <= (right_at_y + edge_margin):
            return PedestrianRoadPosition.ROAD_EDGE

        # Outside lane boundaries => Sidewalk
        return PedestrianRoadPosition.SIDEWALK

    def classify_movement_intent(
        self,
        history: List[Dict[str, Any]],
        current_pos: PedestrianRoadPosition,
        fps: float,
        road_ctx: RoadContextData,
    ) -> PedestrianMovementIntent:
        """Determine a pedestrian's directional movement intent relative to the roadway."""
        if len(history) < 2:
            return PedestrianMovementIntent.STATIONARY

        curr = history[-1]
        prev = history[0] if len(history) <= 3 else history[-3]
        dt = max(0.01, curr.get("timestamp", 0.0) - prev.get("timestamp", 0.0))

        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        speed_px_s = math.hypot(dx, dy) / dt

        # Stationary threshold
        if speed_px_s < 8.0:
            return PedestrianMovementIntent.STATIONARY

        cx = curr["x"]
        lane_center = road_ctx.lane_center_px

        # Lateral movement towards the center of the road
        moving_towards_center = (cx < lane_center and dx > 0) or (cx > lane_center and dx < 0)
        lateral_speed = abs(dx) / dt

        if current_pos == PedestrianRoadPosition.ROADWAY or current_pos == PedestrianRoadPosition.CROSSWALK:
            if lateral_speed > 10.0:
                return PedestrianMovementIntent.CROSSING
            return PedestrianMovementIntent.APPROACHING_ROAD

        if current_pos in (PedestrianRoadPosition.SIDEWALK, PedestrianRoadPosition.ROAD_EDGE):
            if moving_towards_center and lateral_speed > 10.0:
                return PedestrianMovementIntent.APPROACHING_ROAD
            elif not moving_towards_center and lateral_speed > 10.0:
                return PedestrianMovementIntent.MOVING_AWAY
            else:
                return PedestrianMovementIntent.ALONG_SIDEWALK

        return PedestrianMovementIntent.UNKNOWN
