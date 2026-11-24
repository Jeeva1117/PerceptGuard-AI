"""Kinematics feature engineering and coordinate transformation engine for Trajectory Prediction.

Computes:
- Temporal finite-difference velocity and acceleration vectors using true timestamps (dt)
- Heading orientation and angular turn rates
- Motion state categorization (Stationary, Cruising, Accelerating, Decelerating, Turning)
- Pinhole camera flat-ground metric projection (approximate bird's-eye-view meters)
"""

from __future__ import annotations

import math
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np

from src.trajectory_prediction.schemas import (
    CoordinateFrame,
    MotionState,
)


class KinematicsEngine:
    """Production kinematics calculator for automotive object tracks."""

    def __init__(
        self,
        camera_height_meters: float = 1.35,
        horizon_fraction: float = 0.45,
    ):
        self.camera_height_meters = camera_height_meters
        self.horizon_fraction = horizon_fraction

    def project_to_ground_plane(
        self,
        cx: float,
        cy: float,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Project an image pixel point (cx, cy) to ground-plane coordinates in meters.

        Uses monocular flat-earth pinhole geometry:
        - Camera mounted at height H_cam looking forward with horizon at v0.
        - Distance Y_m = (H_cam * f_y) / (cy - v0)
        - Lateral X_m = (cx - u0) * Y_m / f_x
        """
        v0 = frame_height * self.horizon_fraction
        u0 = frame_width / 2.0
        fy = float(frame_height)  # Approximate focal length ~ height
        fx = float(frame_width)

        # Points at or above the horizon line cannot be projected onto the ground plane ahead
        if cy <= v0 + 10.0:
            return None, None

        y_m = (self.camera_height_meters * fy) / (cy - v0)
        x_m = ((cx - u0) * y_m) / fx

        return round(float(x_m), 2), round(float(y_m), 2)

    def compute_motion_features(
        self,
        history: List[Dict[str, Any]],
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, Any]:
        """Compute kinematic telemetry from bounded observation sequence using true timestamps."""
        if not history:
            return {
                "velocity": {"vx": 0.0, "vy": 0.0},
                "acceleration": {"ax": 0.0, "ay": 0.0},
                "speed_kmh": 0.0,
                "heading_degrees": 0.0,
                "angular_velocity_deg_s": 0.0,
                "motion_state": MotionState.STATIONARY,
                "metric_position": (None, None),
            }

        curr = history[-1]
        cx, cy = curr["x"], curr["y"]
        x_m, y_m = self.project_to_ground_plane(cx, cy, frame_width, frame_height)

        if len(history) < 2:
            return {
                "velocity": {"vx": 0.0, "vy": 0.0},
                "acceleration": {"ax": 0.0, "ay": 0.0},
                "speed_kmh": 0.0,
                "heading_degrees": 0.0,
                "angular_velocity_deg_s": 0.0,
                "motion_state": MotionState.STATIONARY,
                "metric_position": (x_m, y_m),
            }

        prev = history[-2]
        dt = max(0.001, curr.get("timestamp", 0.0) - prev.get("timestamp", 0.0))
        if dt <= 0.001 or dt > 5.0:
            dt = 1.0 / max(1.0, fps)

        # 1. Finite difference velocity (pixels / sec)
        vx = (curr["x"] - prev["x"]) / dt
        vy = (curr["y"] - prev["y"]) / dt

        # Smooth velocity with 3-frame window if available
        if len(history) >= 3:
            p2 = history[-3]
            dt2 = max(0.002, curr.get("timestamp", 0.0) - p2.get("timestamp", 0.0))
            vx = 0.6 * vx + 0.4 * ((curr["x"] - p2["x"]) / dt2)
            vy = 0.6 * vy + 0.4 * ((curr["y"] - p2["y"]) / dt2)

        speed_px_s = float(math.hypot(vx, vy))

        # 2. Acceleration (pixels / sec^2)
        ax = 0.0
        ay = 0.0
        if len(history) >= 3:
            p_prev = history[-2]
            p_prior = history[-3]
            dt_prev = max(0.001, p_prev.get("timestamp", 0.0) - p_prior.get("timestamp", 0.0))
            v_prev_x = (p_prev["x"] - p_prior["x"]) / dt_prev
            v_prev_y = (p_prev["y"] - p_prior["y"]) / dt_prev
            ax = (vx - v_prev_x) / dt
            ay = (vy - v_prev_y) / dt

        # 3. Ground speed estimation (km/h)
        # Use metric ground displacement if available; otherwise fallback to calibrated pixel scale
        speed_kmh = 0.0
        if len(history) >= 2 and x_m is not None and y_m is not None:
            prev_x_m, prev_y_m = self.project_to_ground_plane(prev["x"], prev["y"], frame_width, frame_height)
            if prev_x_m is not None and prev_y_m is not None:
                dist_m = math.hypot(x_m - prev_x_m, y_m - prev_y_m)
                speed_m_s = dist_m / dt
                speed_kmh = float(speed_m_s * 3.6)

        # Fallback to perspective pixel scaling
        if speed_kmh <= 0.0 and speed_px_s > 0.0:
            # Perspective factor: pixels near bottom represent fewer meters per pixel than pixels near horizon
            pixels_per_meter = max(5.0, 15.0 + (cy / frame_height) * 25.0)
            speed_kmh = float((speed_px_s / pixels_per_meter) * 3.6)

        # 4. Heading orientation (degrees [-180, 180])
        heading_deg = float(math.degrees(math.atan2(vy, vx))) if speed_px_s > 2.0 else 0.0

        # 5. Angular turn velocity
        angular_vel = 0.0
        if len(history) >= 3:
            prev_heading = float(math.degrees(math.atan2(v_prev_y, v_prev_x))) if math.hypot(v_prev_x, v_prev_y) > 2.0 else heading_deg
            d_heading = (heading_deg - prev_heading + 180.0) % 360.0 - 180.0
            angular_vel = float(d_heading / dt)

        # 6. Categorize Motion State using longitudinal acceleration projection
        # a_parallel = (v . a) / |v| represents true speeding up (> 0) or braking (< 0)
        a_parallel = (vx * ax + vy * ay) / max(0.001, speed_px_s) if speed_px_s > 2.0 else 0.0

        if speed_kmh < 2.5 or speed_px_s < 3.0:
            state = MotionState.STATIONARY
        elif angular_vel < -15.0:
            state = MotionState.TURNING_LEFT
        elif angular_vel > 15.0:
            state = MotionState.TURNING_RIGHT
        elif a_parallel > 25.0:
            state = MotionState.ACCELERATING
        elif a_parallel < -25.0:
            state = MotionState.DECELERATING
        else:
            state = MotionState.CRUISING

        return {
            "velocity": {"vx": float(vx), "vy": float(vy)},
            "acceleration": {"ax": float(ax), "ay": float(ay)},
            "speed_kmh": float(min(180.0, max(0.0, speed_kmh))),
            "heading_degrees": heading_deg,
            "angular_velocity_deg_s": angular_vel,
            "motion_state": state,
            "metric_position": (x_m, y_m),
        }
