"""Multi-Agent Interaction Modeling and Spatial Context Extraction.

Analyzes pairwise spatial arrangements, relative velocities, and trajectory convergence
between all active road users to identify:
- Car following behavior (lead vehicle following distance, closing speed)
- Crossing paths (converging pedestrian or cross-traffic intersection)
- Approaching opposite direction traffic
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.trajectory_prediction.schemas import (
    InteractionType,
    MultiAgentInteraction,
)


class MultiAgentInteractionModel:
    """Extracts spatial interaction context and following telemetry across tracked agents."""

    def analyze_interactions(
        self,
        target_track: Dict[str, Any],
        all_tracks: List[Dict[str, Any]],
        fps: float,
    ) -> MultiAgentInteraction:
        """Evaluate spatial relationships between target track and surrounding road users."""
        target_id = target_track["track_id"]
        tcx, tcy = target_track["center"]
        tvx = target_track.get("velocity", {}).get("vx", 0.0)
        tvy = target_track.get("velocity", {}).get("vy", 0.0)
        tspeed_kmh = target_track.get("speed_kmh", 0.0)

        nearest_id: Optional[int] = None
        min_dist = float("inf")
        nearest_agent: Optional[Dict[str, Any]] = None

        for other in all_tracks:
            if other["track_id"] == target_id:
                continue
            ocx, ocy = other["center"]
            dist = math.hypot(ocx - tcx, ocy - tcy)
            if dist < min_dist:
                min_dist = dist
                nearest_id = other["track_id"]
                nearest_agent = other

        if nearest_agent is None or min_dist == float("inf"):
            return MultiAgentInteraction(
                interaction_type=InteractionType.FREE_FLOW,
                interaction_notes="Free flow unobstructed motion; zero neighboring agents within proximity.",
            )

        ocx, ocy = nearest_agent["center"]
        ovx = nearest_agent.get("velocity", {}).get("vx", 0.0)
        ovy = nearest_agent.get("velocity", {}).get("vy", 0.0)
        ospeed_kmh = nearest_agent.get("speed_kmh", 0.0)

        # Vector from target to other agent: (dx, dy)
        dx = ocx - tcx
        dy = ocy - tcy

        # Relative speed (closing velocity component along line of sight)
        rel_vx = tvx - ovx
        rel_vy = tvy - ovy
        closing_speed_px_s = (rel_vx * dx + rel_vy * dy) / max(0.001, min_dist)
        rel_speed_kmh = tspeed_kmh - ospeed_kmh

        # Project metric distance (approx 20 px/m)
        dist_m = min_dist / 20.0

        # Time-to-headway if closing
        tth = (min_dist / closing_speed_px_s) if closing_speed_px_s > 1.0 else None

        # Determine interaction type:
        # Check alignment: dot product with travel direction
        heading_align = (tvx * dx + tvy * dy) / (max(0.001, math.hypot(tvx, tvy)) * max(0.001, min_dist))

        # Check velocity opposition: dot product between tv and ov
        velocity_align = (tvx * ovx + tvy * ovy) / (
            max(0.001, math.hypot(tvx, tvy)) * max(0.001, math.hypot(ovx, ovy))
        )

        if heading_align > 0.65 and abs(velocity_align) > 0.5:
            # Other agent is directly in front along our travel corridor
            int_type = InteractionType.CAR_FOLLOWING
            notes = (
                f"Following Track #{nearest_id} ({nearest_agent.get('class_name', 'agent')}). "
                f"Separation: {dist_m:.1f}m. Closing rate: {rel_speed_kmh:+.1f} km/h."
            )
        elif velocity_align < -0.6:
            int_type = InteractionType.APPROACHING_OPPOSITE
            notes = f"Approaching opposing traffic Track #{nearest_id}. Separation: {dist_m:.1f}m."
        elif abs(heading_align) < 0.65 and min_dist < 120.0:
            int_type = InteractionType.CROSSING_PATH
            notes = f"Cross-path interaction with Track #{nearest_id}. Potential lateral intersection."
        else:
            int_type = InteractionType.FREE_FLOW
            notes = f"Nearest road user is Track #{nearest_id} at {dist_m:.1f}m separation."

        return MultiAgentInteraction(
            interaction_type=int_type,
            nearest_agent_id=nearest_id,
            nearest_agent_distance_px=min_dist,
            nearest_agent_distance_m=dist_m,
            relative_speed_kmh=rel_speed_kmh,
            time_to_headway_seconds=tth,
            interaction_notes=notes,
        )
