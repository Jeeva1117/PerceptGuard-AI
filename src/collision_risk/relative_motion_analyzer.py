"""Relative motion and closing velocity analyzer for automotive road agents.

Computes:
1. Relative velocity vectors (v_rel = v_a - v_b).
2. Projected closing speed along line-of-sight unit vector.
3. Separation change classification: APPROACHING, DIVERGING, or PARALLEL.
4. Relative acceleration along line-of-sight corridor.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


class RelativeMotionAnalyzer:
    """Analyzes dynamic relative kinematics between road users and ego camera."""

    def __init__(self, closing_speed_threshold_px_s: float = 2.0):
        self.closing_speed_threshold = closing_speed_threshold_px_s

    def compute_relative_kinematics(
        self,
        pos_a: Tuple[float, float],
        vel_a: Tuple[float, float],
        acc_a: Tuple[float, float],
        pos_b: Tuple[float, float],
        vel_b: Tuple[float, float],
        acc_b: Tuple[float, float],
    ) -> Dict[str, Any]:
        """Compute relative motion parameters between agent A and agent B.

        Line of sight vector u_los points from A to B:
        u_los = (pos_b - pos_a) / ||pos_b - pos_a||

        Relative velocity v_rel = vel_a - vel_b
        Closing speed v_closing = - (vel_b - vel_a) . u_los = (vel_a - vel_b) . u_los
        (positive means distance between A and B is decreasing)
        """
        ax, ay = pos_a
        bx, by = pos_b
        dx = bx - ax
        dy = by - ay
        dist = math.hypot(dx, dy)

        if dist < 0.001:
            return {
                "distance_px": 0.0,
                "closing_speed_px_s": 0.0,
                "closing_acceleration_px_s2": 0.0,
                "motion_pattern": "COINCIDENT",
                "is_approaching": False,
                "relative_speed_magnitude_px_s": 0.0,
            }

        # Line of sight unit vector from A to B
        ux = dx / dist
        uy = dy / dist

        # Relative velocity vector (motion of A relative to B)
        vax, vay = vel_a
        vbx, vby = vel_b
        # Closing speed: component of relative velocity that shrinks distance
        # Rate of decrease of distance: -d(dist)/dt
        # dist^2 = (bx - ax)^2 + (by - ay)^2
        # 2*dist*d(dist)/dt = 2*(bx - ax)*(vbx - vax) + 2*(by - ay)*(vby - vay)
        # d(dist)/dt = (vbx - vax)*ux + (vby - vay)*uy
        # closing_speed = - d(dist)/dt = (vax - vbx)*ux + (vay - vby)*uy
        closing_speed = (vax - vbx) * ux + (vay - vby) * uy

        # Relative acceleration along line of sight
        aax, aay = acc_a
        abx, aby = acc_b
        closing_acc = (aax - abx) * ux + (aay - aby) * uy

        rel_speed_mag = math.hypot(vax - vbx, vay - vby)

        if closing_speed > self.closing_speed_threshold:
            pattern = "APPROACHING"
            is_approaching = True
        elif closing_speed < -self.closing_speed_threshold:
            pattern = "DIVERGING"
            is_approaching = False
        else:
            pattern = "PARALLEL"
            is_approaching = False

        return {
            "distance_px": dist,
            "closing_speed_px_s": float(closing_speed),
            "closing_acceleration_px_s2": float(closing_acc),
            "motion_pattern": pattern,
            "is_approaching": is_approaching,
            "relative_speed_magnitude_px_s": float(rel_speed_mag),
        }

    def compute_ego_relative_motion(
        self,
        target_pos_px: Tuple[float, float],
        target_vel_px: Tuple[float, float],
        target_acc_px: Tuple[float, float],
        ego_pos_px: Tuple[float, float],
    ) -> Dict[str, Any]:
        """Compute relative motion between stationary/host vehicle camera and target agent."""
        # Virtual ego vehicle is assumed at bottom center with zero sensor-frame velocity
        return self.compute_relative_kinematics(
            pos_a=ego_pos_px,
            vel_a=(0.0, 0.0),
            acc_a=(0.0, 0.0),
            pos_b=target_pos_px,
            vel_b=target_vel_px,
            acc_b=target_acc_px,
        )
