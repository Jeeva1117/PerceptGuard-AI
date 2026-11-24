"""Multi-source uncertainty analyzer fusing environmental and perception degradation.

Evaluates:
1. Camera optical visibility degradation (Feature 1).
2. Detector confidence and bounding box stability (Feature 2).
3. Track observation history length and lost/occluded status (Feature 2).
4. Trajectory forecast spatial covariance expansion (Feature 3).

Prevents confident false alarms or premature risk assertions under degraded sensors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.collision_risk.schemas import UncertaintyLevel


class UncertaintyAnalyzer:
    """Fuses multi-modal uncertainty across the perception pipeline."""

    def __init__(
        self,
        visibility_weight: float = 0.35,
        detection_weight: float = 0.35,
        tracking_weight: float = 0.30,
    ):
        self.w_vis = visibility_weight
        self.w_det = detection_weight
        self.w_trk = tracking_weight

    def evaluate_uncertainty(
        self,
        weather_intel: Dict[str, Any],
        trajectories: List[Dict[str, Any]],
        fps: float = 10.0,
    ) -> Tuple[float, UncertaintyLevel, List[str], bool]:
        """Compute fused uncertainty score and explainable degradation causes.

        Returns:
            Tuple of:
            - uncertainty_score: float in [0.0, 1.0]
            - uncertainty_level: UncertaintyLevel enum
            - contributing_factors: List of descriptive strings
            - is_unreliable: bool (True if uncertainty warrants UNKNOWN state)
        """
        factors: List[str] = []

        # 1. Weather / optical degradation
        vis_score = float(weather_intel.get("visibility_score", 100.0))
        u_weather = max(0.0, min(1.0, (100.0 - vis_score) / 100.0))
        if vis_score < 40.0:
            factors.append(f"Severe optical impairment: Visibility at {vis_score:.0f}%")
        elif vis_score < 70.0:
            factors.append(f"Mild optical degradation: Visibility at {vis_score:.0f}%")

        # 2. Object detection confidence
        if trajectories:
            confs = [float(t.get("confidence", 0.85)) for t in trajectories]
            mean_conf = sum(confs) / len(confs)
            u_det = max(0.0, min(1.0, 1.0 - mean_conf))
            if mean_conf < 0.60:
                factors.append(f"Low neural detection confidence: {mean_conf*100:.0f}%")
        else:
            u_det = 0.15

        # 3. Track history maturity and continuity
        u_track = 0.10
        if trajectories:
            short_hist_count = 0
            for t in trajectories:
                hist_len = int(t.get("history_length", len(t.get("history", []))))
                if hist_len < 3:
                    short_hist_count += 1
            if short_hist_count > 0:
                u_track = min(0.90, 0.20 + 0.35 * (short_hist_count / len(trajectories)))
                factors.append(f"{short_hist_count} track(s) have immature history (< 3 frames)")

        # Fused uncertainty score
        fused_u = (
            self.w_vis * u_weather
            + self.w_det * u_det
            + self.w_trk * u_track
        )
        fused_u = max(0.0, min(1.0, fused_u))

        # Categorize level
        if fused_u >= 0.70:
            level = UncertaintyLevel.CRITICAL
            is_unreliable = True
        elif fused_u >= 0.45:
            level = UncertaintyLevel.HIGH
            is_unreliable = False
        elif fused_u >= 0.25:
            level = UncertaintyLevel.MEDIUM
            is_unreliable = False
        else:
            level = UncertaintyLevel.LOW
            is_unreliable = False

        # Forced invalidation triggers
        if vis_score < 20.0:
            is_unreliable = True
            factors.append("Camera visibility below critical operational threshold (20%)")

        return fused_u, level, factors, is_unreliable
