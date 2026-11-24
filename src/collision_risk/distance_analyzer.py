"""Distance and separation analyzer for automotive scene agents.

Maintains strict separation between:
1. Exact image-plane pixel distances (Euclidean pixels).
2. Pinhole ground-plane metric distances (meters), populated ONLY when valid
   flat-earth projective metric coordinates exist.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


class DistanceAnalyzer:
    """Computes pairwise and ego-relative physical separation across detected road agents."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        # Virtual ego vehicle reference point: bottom center of camera frame
        self.ego_center_px = (frame_width / 2.0, float(frame_height))
        # Ego vehicle ground metric position: 0m lateral, 0m forward (at camera origin)
        self.ego_center_m = (0.0, 0.0)

    def compute_ego_distance(
        self,
        target_center_px: Tuple[float, float],
        target_center_m: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> Tuple[float, Optional[float]]:
        """Compute distance from ego vehicle to target agent.

        Returns:
            Tuple of (distance_px, distance_m). distance_m is None if metric coordinates are unavailable.
        """
        cx, cy = target_center_px
        ex, ey = self.ego_center_px
        dist_px = float(math.hypot(cx - ex, cy - ey))

        dist_m: Optional[float] = None
        if target_center_m is not None:
            tx_m, ty_m = target_center_m
            if tx_m is not None and ty_m is not None:
                # Forward distance y_m, lateral distance x_m from ego camera origin
                dist_m = float(math.hypot(tx_m, ty_m))

        return dist_px, dist_m

    def compute_pairwise_distance(
        self,
        center_a_px: Tuple[float, float],
        center_b_px: Tuple[float, float],
        center_a_m: Optional[Tuple[Optional[float], Optional[float]]] = None,
        center_b_m: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> Tuple[float, Optional[float]]:
        """Compute Euclidean separation between two road agents.

        Returns:
            Tuple of (distance_px, distance_m).
        """
        ax, ay = center_a_px
        bx, by = center_b_px
        dist_px = float(math.hypot(ax - bx, ay - by))

        dist_m: Optional[float] = None
        if center_a_m is not None and center_b_m is not None:
            ax_m, ay_m = center_a_m
            bx_m, by_m = center_b_m
            if ax_m is not None and ay_m is not None and bx_m is not None and by_m is not None:
                dist_m = float(math.hypot(ax_m - bx_m, ay_m - by_m))

        return dist_px, dist_m

    def analyze_scene_distances(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze distance relationships for all tracks in current frame.

        Returns:
            Dictionary containing:
            - closest_track_id: ID of the track closest to ego vehicle.
            - closest_distance_px: Pixel separation to closest track.
            - closest_distance_m: Metric separation to closest track (or None).
            - pairwise_distances: Dict of (track_a, track_b) -> (dist_px, dist_m).
        """
        if not trajectories:
            return {
                "closest_track_id": None,
                "closest_distance_px": None,
                "closest_distance_m": None,
                "pairwise_distances": {},
            }

        closest_tid: Optional[int] = None
        min_dist_px = float("inf")
        closest_dist_m: Optional[float] = None

        pairwise: Dict[Tuple[int, int], Tuple[float, Optional[float]]] = {}

        # 1. Ego-relative distances
        for trk in trajectories:
            tid = trk["track_id"]
            cx, cy = trk["center"]
            metric_pos = trk.get("metric_position")
            dist_px, dist_m = self.compute_ego_distance((cx, cy), metric_pos)

            if dist_px < min_dist_px:
                min_dist_px = dist_px
                closest_tid = tid
                closest_dist_m = dist_m

        # 2. Pairwise inter-agent distances
        n = len(trajectories)
        for i in range(n):
            for j in range(i + 1, n):
                t1 = trajectories[i]
                t2 = trajectories[j]
                c1 = t1["center"]
                c2 = t2["center"]
                m1 = t1.get("metric_position")
                m2 = t2.get("metric_position")
                pair_px, pair_m = self.compute_pairwise_distance(c1, c2, m1, m2)
                pairwise[(t1["track_id"], t2["track_id"])] = (pair_px, pair_m)

        return {
            "closest_track_id": closest_tid,
            "closest_distance_px": min_dist_px if min_dist_px != float("inf") else None,
            "closest_distance_m": closest_dist_m,
            "pairwise_distances": pairwise,
        }
