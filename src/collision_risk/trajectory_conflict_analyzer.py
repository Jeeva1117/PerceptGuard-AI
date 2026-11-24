"""Spatio-temporal trajectory conflict and convergence analyzer.

Core Principle:
Two tracks only constitute a collision conflict if they occupy overlapping
spatial regions AT APPROXIMATELY THE SAME TIME. Spatially intersecting paths
with temporal separation (e.g. crossing an intersection seconds apart) do NOT
trigger collision alerts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.collision_risk.schemas import ConflictType, TrajectoryConflict


class TrajectoryConflictAnalyzer:
    """Detects and categorizes synchronized spatio-temporal trajectory overlap across horizons."""

    def __init__(
        self,
        base_collision_radius_px: float = 35.0,
        temporal_tolerance_seconds: float = 0.5,
    ):
        self.base_collision_radius = base_collision_radius_px
        self.temporal_tolerance = temporal_tolerance_seconds

        # Physical dimensional bounding radii per object class in pixel space
        self.class_radii_px = {
            "car": 30.0,
            "truck": 50.0,
            "bus": 55.0,
            "motorcycle": 20.0,
            "bicycle": 18.0,
            "pedestrian": 14.0,
        }

    def get_object_radius(self, class_name: str) -> float:
        """Return conservative spatial radius of object class."""
        return self.class_radii_px.get(class_name.lower(), self.base_collision_radius)

    def determine_conflict_type(
        self,
        class_a: str,
        class_b: str,
        vel_a: Tuple[float, float],
        vel_b: Tuple[float, float],
    ) -> ConflictType:
        """Classify conflict geometry based on agent semantics and velocity vectors."""
        ca = class_a.lower()
        cb = class_b.lower()

        if ca == "pedestrian" or cb == "pedestrian":
            return ConflictType.PEDESTRIAN_CROSSING
        if ca in ("bicycle", "motorcycle") or cb in ("bicycle", "motorcycle"):
            return ConflictType.CYCLIST_CONFLICT

        vax, vay = vel_a
        vbx, vby = vel_b
        speed_a = math.hypot(vax, vay)
        speed_b = math.hypot(vbx, vby)

        if speed_a < 1.0 or speed_b < 1.0:
            return ConflictType.REAR_END

        # Velocity alignment (cosine of angle between headings)
        cos_align = (vax * vbx + vay * vby) / (speed_a * speed_b)
        cos_align = max(-1.0, min(1.0, cos_align))

        if cos_align > 0.70:
            return ConflictType.REAR_END
        elif cos_align < -0.60:
            return ConflictType.HEAD_ON
        elif abs(cos_align) <= 0.60:
            return ConflictType.CROSSING
        else:
            return ConflictType.MERGING

    def _extract_waypoint(self, wp: Any, k: int, fps: float) -> Tuple[float, float, float, float]:
        """Extract (x, y, horizon_seconds, uncertainty) from waypoint object or dict."""
        if isinstance(wp, dict):
            x = float(wp.get("x_px", wp.get("x", 0.0)))
            y = float(wp.get("y_px", wp.get("y", 0.0)))
            t = float(wp.get("horizon_seconds", (k + 1) / fps))
            u = float(wp.get("uncertainty_radius_px", wp.get("uncertainty_radius", 5.0)))
        else:
            x = float(getattr(wp, "x_px", getattr(wp, "x", 0.0)))
            y = float(getattr(wp, "y_px", getattr(wp, "y", 0.0)))
            t = float(getattr(wp, "horizon_seconds", (k + 1) / fps))
            u = float(getattr(wp, "uncertainty_radius_px", getattr(wp, "uncertainty_radius", 5.0)))
        return x, y, t, u

    def detect_conflicts(
        self,
        trajectories: List[Dict[str, Any]],
        fps: float = 10.0,
    ) -> List[TrajectoryConflict]:
        """Examine predicted future waypoints across all active tracks for synchronous overlap."""
        conflicts: List[TrajectoryConflict] = []
        n = len(trajectories)
        if n < 2:
            return conflicts

        for i in range(n):
            for j in range(i + 1, n):
                t1 = trajectories[i]
                t2 = trajectories[j]

                c_type = self.determine_conflict_type(
                    t1.get("class_name", "car"),
                    t2.get("class_name", "car"),
                    (t1.get("velocity", {}).get("vx", 0.0), t1.get("velocity", {}).get("vy", 0.0)),
                    (t2.get("velocity", {}).get("vx", 0.0), t2.get("velocity", {}).get("vy", 0.0)),
                )

                r1 = self.get_object_radius(t1.get("class_name", "car"))
                r2 = self.get_object_radius(t2.get("class_name", "car"))
                nominal_threshold = r1 + r2

                # Extract waypoints from primary trajectory
                wps1 = t1.get("primary_trajectory", [])
                wps2 = t2.get("primary_trajectory", [])

                # If primary waypoints not present, check predicted_horizons
                if not wps1 or not wps2:
                    h1_map = t1.get("predicted_horizons", {})
                    h2_map = t2.get("predicted_horizons", {})
                    # Check H1, H3, H5
                    for h_name, h_time in [("h1", 1.0), ("h3", 3.0), ("h5", 5.0)]:
                        if h_name in h1_map and h_name in h2_map:
                            p1 = h1_map[h_name]
                            p2 = h2_map[h_name]
                            dist = math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])
                            uncert = float(p1.get("uncertainty_radius", 5.0) + p2.get("uncertainty_radius", 5.0))
                            dynamic_thresh = nominal_threshold + 0.35 * uncert

                            if dist <= dynamic_thresh:
                                c_point_px = ((p1["x"] + p2["x"]) / 2.0, (p1["y"] + p2["y"]) / 2.0)
                                p1_xm, p1_ym = p1.get("x_m"), p1.get("y_m")
                                p2_xm, p2_ym = p2.get("x_m"), p2.get("y_m")
                                dist_m: Optional[float] = None
                                c_point_m: Optional[Tuple[float, float]] = None
                                if p1_xm is not None and p1_ym is not None and p2_xm is not None and p2_ym is not None:
                                    dist_m = float(math.hypot(p1_xm - p2_xm, p1_ym - p2_ym))
                                    c_point_m = ((p1_xm + p2_xm) / 2.0, (p1_ym + p2_ym) / 2.0)

                                conflicts.append(
                                    TrajectoryConflict(
                                        track_a=t1["track_id"],
                                        track_b=t2["track_id"],
                                        class_a=t1.get("class_name", "car"),
                                        class_b=t2.get("class_name", "car"),
                                        conflict_type=c_type,
                                        time_to_conflict_seconds=h_time,
                                        horizon_label=h_name.upper(),
                                        separation_at_conflict_px=dist,
                                        separation_at_conflict_m=dist_m,
                                        conflict_point_px=c_point_px,
                                        conflict_point_m=c_point_m,
                                        is_spatiotemporal_overlap=True,
                                        combined_uncertainty_px=uncert,
                                    )
                                )
                                break
                    continue

                # Synchronous step-by-step waypoint collision sweep
                min_steps = min(len(wps1), len(wps2))
                conflict_found = False

                for k in range(min_steps):
                    wp1 = wps1[k]
                    wp2 = wps2[k]

                    # Synchronous time check: must be at approximately same time step
                    x1, y1, t_k1, u1 = self._extract_waypoint(wp1, k, fps)
                    x2, y2, t_k2, u2 = self._extract_waypoint(wp2, k, fps)
                    t_k = (t_k1 + t_k2) / 2.0

                    dist = math.hypot(x1 - x2, y1 - y2)
                    uncert = u1 + u2

                    dynamic_thresh = nominal_threshold + 0.35 * uncert

                    if dist <= dynamic_thresh:
                        # Conflict confirmed at time t_k
                        h_label = "H1" if t_k <= 1.5 else ("H3" if t_k <= 3.5 else "H5")
                        c_point_px = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                        conflicts.append(
                            TrajectoryConflict(
                                track_a=t1["track_id"],
                                track_b=t2["track_id"],
                                class_a=t1.get("class_name", "car"),
                                class_b=t2.get("class_name", "car"),
                                conflict_type=c_type,
                                time_to_conflict_seconds=t_k,
                                horizon_label=h_label,
                                separation_at_conflict_px=dist,
                                separation_at_conflict_m=None,
                                conflict_point_px=c_point_px,
                                conflict_point_m=None,
                                is_spatiotemporal_overlap=True,
                                combined_uncertainty_px=uncert,
                            )
                        )
                        conflict_found = True
                        break

        return conflicts
