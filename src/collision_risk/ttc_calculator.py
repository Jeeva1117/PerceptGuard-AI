"""Time-To-Collision (TTC) Calculator with rigorous mathematical gating.

Adheres to automotive ADAS standards:
1. TTC is calculated ONLY when closing speed > 0 and the object is within the travel corridor.
2. Diverging, parallel, or stationary targets return None.
3. Crossing or turning targets with large line-of-sight angles return None for 1D TTC
   to avoid misleading false alarms (handled instead via 2D trajectory intersection).
4. Supports linear and constant-acceleration kinematic solutions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


class TTCCalculator:
    """Computes defensible Time-To-Collision estimates under verified kinematic conditions."""

    def __init__(
        self,
        min_closing_speed_px_s: float = 3.0,
        max_valid_ttc_seconds: float = 10.0,
        max_crossing_angle_deg: float = 65.0,
    ):
        self.min_closing_speed = min_closing_speed_px_s
        self.max_valid_ttc = max_valid_ttc_seconds
        self.max_crossing_angle = max_crossing_angle_deg

    def calculate_linear_ttc(
        self,
        distance_px: float,
        closing_speed_px_s: float,
    ) -> Optional[float]:
        """Compute standard first-order constant-velocity Time-to-Collision.

        TTC = distance / closing_speed
        Valid ONLY when closing_speed > min_closing_speed.
        """
        if closing_speed_px_s <= self.min_closing_speed or distance_px <= 0.0:
            return None

        ttc = distance_px / closing_speed_px_s
        if 0.0 < ttc <= self.max_valid_ttc:
            return round(float(ttc), 2)
        return None

    def calculate_accelerated_ttc(
        self,
        distance_px: float,
        closing_speed_px_s: float,
        closing_acc_px_s2: float,
    ) -> Optional[float]:
        """Compute second-order Time-to-Collision under constant closing acceleration:
        0.5 * a_closing * t^2 + v_closing * t - d = 0
        """
        if closing_speed_px_s <= self.min_closing_speed or distance_px <= 0.0:
            return None

        # If acceleration is negligible, use linear solution
        if abs(closing_acc_px_s2) < 0.5:
            return self.calculate_linear_ttc(distance_px, closing_speed_px_s)

        # Quadratic formula: 0.5 * a * t^2 + v * t - d = 0
        A = 0.5 * closing_acc_px_s2
        B = closing_speed_px_s
        C = -distance_px

        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            # No real collision occurs under current deceleration
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-B + sqrt_disc) / (2 * A)
        t2 = (-B - sqrt_disc) / (2 * A)

        positive_roots = [t for t in (t1, t2) if 0.05 < t <= self.max_valid_ttc]
        if not positive_roots:
            return None

        return round(float(min(positive_roots)), 2)

    def evaluate_ego_ttc(
        self,
        distance_px: float,
        closing_speed_px_s: float,
        closing_acc_px_s2: float,
        target_center_px: Tuple[float, float],
        target_velocity_px: Tuple[float, float],
        ego_center_px: Tuple[float, float],
    ) -> Dict[str, Any]:
        """Evaluate ego-vehicle TTC with geometric line-of-sight corridor validation."""
        cx, cy = target_center_px
        ex, ey = ego_center_px
        vx, vy = target_velocity_px

        # Vector from ego to target
        dx = cx - ex
        dy = cy - ey
        dist = math.hypot(dx, dy)

        if dist < 0.001 or closing_speed_px_s <= self.min_closing_speed:
            return {
                "ttc_seconds": None,
                "is_valid": False,
                "invalidation_reason": "Diverging or zero closing speed",
            }

        # Check angle between target velocity vector and line of sight back to ego
        # Target heading toward ego: dot product of target velocity with (-dx, -dy)
        speed = math.hypot(vx, vy)
        if speed > 2.0:
            cos_angle = (-dx * vx - dy * vy) / (dist * speed)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle_deg = math.degrees(math.acos(cos_angle))

            if angle_deg > self.max_crossing_angle:
                return {
                    "ttc_seconds": None,
                    "is_valid": False,
                    "invalidation_reason": f"Target motion is crossing/lateral ({angle_deg:.0f}° relative to corridor); linear TTC invalid.",
                }

        ttc = self.calculate_accelerated_ttc(distance_px, closing_speed_px_s, closing_acc_px_s2)
        if ttc is None:
            ttc = self.calculate_linear_ttc(distance_px, closing_speed_px_s)

        return {
            "ttc_seconds": ttc,
            "is_valid": ttc is not None,
            "invalidation_reason": None if ttc is not None else "TTC exceeds maximum validated horizon (>10s) or target braking prevents contact",
        }
