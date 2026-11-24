"""Lightweight Multi-Horizon Visibility Degradation Forecaster for CPU Execution.

Implements:
1. Holt's Linear Trend Exponential Smoothing (Level + Trend projection with damping)
2. Moving Average Baseline comparison
3. Quantitative uncertainty expansion over lookahead horizons
4. Categorical trend classification:
   - Improving
   - Stable
   - Gradually degrading
   - Rapidly degrading
   - Uncertain / INSUFFICIENT_DATA
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from src.camera_weather.schemas import DegradationTrend, VisibilityForecast


class VisibilityForecaster:
    """CPU-optimized predictive engine projecting near-future visibility scores."""

    def __init__(
        self,
        alpha: float = 0.35,  # Level smoothing factor
        beta: float = 0.20,   # Trend smoothing factor
        damping_factor: float = 0.92, # Trend damping to prevent runaway divergence
        min_history_required: int = 5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.damping_factor = damping_factor
        self.min_history_required = min_history_required

    def forecast(
        self,
        history_scores: List[float],
        current_score: float,
        fps: float = 30.0,
    ) -> VisibilityForecast:
        """Forecast visibility scores across Short (+5 fr), Med (+15 fr), and Long (+30 fr) horizons.

        Args:
            history_scores: Historical smoothed scores (oldest to newest)
            current_score: Current smoothed visibility score in [0.0, 100.0]
            fps: Video frames per second

        Returns:
            VisibilityForecast with trend, per-horizon projections, and uncertainty bounds.
        """
        # If insufficient history, strictly return INSUFFICIENT_DATA
        if len(history_scores) < self.min_history_required:
            return VisibilityForecast(
                degradation_trend=DegradationTrend.INSUFFICIENT_DATA,
                trend_velocity=0.0,
                horizons={
                    "h_short": {
                        "horizon_frames": 5,
                        "horizon_seconds": round(5 / max(1.0, fps), 3),
                        "predicted_score": round(current_score, 1),
                        "uncertainty_radius": 15.0,
                        "confidence": 0.40,
                    },
                    "h_med": {
                        "horizon_frames": 15,
                        "horizon_seconds": round(15 / max(1.0, fps), 3),
                        "predicted_score": round(current_score, 1),
                        "uncertainty_radius": 25.0,
                        "confidence": 0.30,
                    },
                    "h_long": {
                        "horizon_frames": 30,
                        "horizon_seconds": round(30 / max(1.0, fps), 3),
                        "predicted_score": round(current_score, 1),
                        "uncertainty_radius": 35.0,
                        "confidence": 0.20,
                    },
                },
                confidence=0.40,
                uncertainty=0.60,
                forecast_method="Awaiting minimum temporal sequence history (>= 5 frames)",
            )

        # 1. Holt's Linear Exponential Smoothing Calibration
        series = list(history_scores)
        if series[-1] != current_score:
            series.append(current_score)

        level = series[0]
        trend = series[1] - series[0] if len(series) > 1 else 0.0

        for val in series[1:]:
            prev_level = level
            level = self.alpha * val + (1.0 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend

        # Trend velocity in points per second (dT/dt)
        trend_velocity_pts_s = float(trend * fps)

        # 2. Moving Average Baseline
        window_size = min(len(series), 10)
        ma_baseline = float(np.mean(series[-window_size:]))

        # 3. Categorize degradation trend
        if trend_velocity_pts_s > 3.0:
            trend_category = DegradationTrend.IMPROVING
        elif -2.5 <= trend_velocity_pts_s <= 2.5:
            trend_category = DegradationTrend.STABLE
        elif -8.0 <= trend_velocity_pts_s < -2.5:
            trend_category = DegradationTrend.GRADUALLY_DEGRADING
        elif trend_velocity_pts_s < -8.0:
            trend_category = DegradationTrend.RAPIDLY_DEGRADING
        else:
            trend_category = DegradationTrend.UNCERTAIN

        # 4. Multi-Horizon Projections (Short = 5 frames, Med = 15 frames, Long = 30 frames)
        horizons_dict: Dict[str, Dict[str, Any]] = {}
        horizon_specs = [
            ("h_short", 5),
            ("h_med", 15),
            ("h_long", 30),
        ]

        for key, h_frames in horizon_specs:
            h_seconds = float(h_frames / max(1.0, fps))
            # Damped trend projection: sum(phi^i for i=1..h)
            damped_steps = sum(self.damping_factor**i for i in range(1, h_frames + 1))
            projected = level + trend * damped_steps

            # Ensemble blend with moving average (70% trend projection, 30% rolling baseline)
            blend_score = 0.70 * projected + 0.30 * ma_baseline
            final_pred = float(np.clip(blend_score, 0.0, 100.0))

            # Uncertainty expands with horizon length and volatility
            uncertainty_radius = float(min(40.0, 4.0 + 0.5 * h_frames + abs(trend_velocity_pts_s) * 0.4))
            conf = float(np.clip(1.0 - (uncertainty_radius / 50.0), 0.30, 0.95))

            horizons_dict[key] = {
                "horizon_frames": h_frames,
                "horizon_seconds": round(h_seconds, 2),
                "predicted_score": round(final_pred, 1),
                "uncertainty_radius": round(uncertainty_radius, 1),
                "confidence": round(conf, 3),
            }

        overall_uncertainty = float(np.clip(horizons_dict["h_med"]["uncertainty_radius"] / 50.0, 0.05, 0.50))
        overall_confidence = float(1.0 - overall_uncertainty)

        return VisibilityForecast(
            degradation_trend=trend_category,
            trend_velocity=trend_velocity_pts_s,
            horizons=horizons_dict,
            confidence=overall_confidence,
            uncertainty=overall_uncertainty,
            forecast_method="Holt Linear Exponential Smoothing + Damped Trend Ensemble",
        )
