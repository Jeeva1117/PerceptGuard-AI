"""Temporal risk feature extraction module for PerceptGuard AI.

Calculates retrospective trend slopes, degradation rates, cross-horizon spreads,
and prediction volatility using strictly retrospective information.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.risk.risk_config import TemporalFeatureConfig
from src.risk.risk_models import TemporalRiskFeatures


class TemporalRiskFeatureExtractor:
    """Extracts temporal risk and degradation features from retrospective predictions."""

    def __init__(self, config: Optional[TemporalFeatureConfig] = None):
        self.config = config or TemporalFeatureConfig()

    def extract_features(
        self,
        sequence_id: str,
        horizon_reliabilities: Dict[int, float],
        history_records: Optional[Sequence[Any]] = None,
    ) -> TemporalRiskFeatures:
        """Extract temporal risk features using current horizons and retrospective history.
        
        Parameters
        ----------
        sequence_id : str
            Identifier of current temporal sequence.
        horizon_reliabilities : Dict[int, float]
            Map of horizon (e.g. 1, 3, 5) to predicted reliability score.
        history_records : Optional[Sequence[Any]]
            Historical prediction records strictly prior to or up to the current sequence.
            MUST be chronological and retrospective (no future frames).
        """
        # Current primary reliability
        primary_rel = horizon_reliabilities.get(1) or horizon_reliabilities.get(3) or 0.0

        # Cross-horizon spread: R(H1) - R(H5) if both present
        cross_horizon_spread: Optional[float] = None
        if 1 in horizon_reliabilities and 5 in horizon_reliabilities:
            cross_horizon_spread = float(horizon_reliabilities[1] - horizon_reliabilities[5])

        # Monotonic degradation check: R(H1) >= R(H3) >= R(H5)
        is_monotonic = False
        if (
            1 in horizon_reliabilities
            and 3 in horizon_reliabilities
            and 5 in horizon_reliabilities
        ):
            r1 = horizon_reliabilities[1]
            r3 = horizon_reliabilities[3]
            r5 = horizon_reliabilities[5]
            is_monotonic = (r1 >= r3 >= r5)

        # Degradation rate across horizons (drop per frame ahead)
        degradation_rate: Optional[float] = None
        if 1 in horizon_reliabilities and 5 in horizon_reliabilities:
            # Positive value indicates worsening perception over time
            drop = horizon_reliabilities[1] - horizon_reliabilities[5]
            degradation_rate = float(drop / 4.0)  # over 4 frames (from t+1 to t+5)

        # Historical trend slope and volatility from past observations
        trend_slope: Optional[float] = None
        volatility: Optional[float] = None
        valid_history_count = 0

        if history_records:
            # Extract historical reliability values chronologically
            history_values: List[float] = []
            for item in history_records[-self.config.history_window_size :]:
                val = getattr(item, "predicted_reliability", None)
                if val is None and isinstance(item, dict):
                    val = item.get("predicted_reliability")
                if val is not None and not (math.isnan(val) or math.isinf(val)):
                    history_values.append(float(val))

            valid_history_count = len(history_values)
            if valid_history_count >= self.config.min_history_points:
                # Compute OLS linear trend slope: y = a*x + b
                x = np.arange(valid_history_count, dtype=np.float64)
                y = np.array(history_values, dtype=np.float64)
                x_mean = float(np.mean(x))
                y_mean = float(np.mean(y))
                denom = float(np.sum((x - x_mean) ** 2))
                if denom > 1e-9:
                    trend_slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
                else:
                    trend_slope = 0.0

                # Compute recent volatility (standard deviation)
                if self.config.detect_volatility and valid_history_count >= 2:
                    volatility = float(np.std(y, ddof=1))

        return TemporalRiskFeatures(
            sequence_id=sequence_id,
            current_predicted_reliability=float(primary_rel),
            horizon_reliabilities=dict(horizon_reliabilities),
            cross_horizon_spread=cross_horizon_spread,
            is_monotonic_degradation=is_monotonic,
            historical_trend_slope=trend_slope,
            degradation_rate=degradation_rate,
            recent_volatility=volatility,
            available_history_points=valid_history_count,
        )
