"""Tests for retrospective temporal risk features and degradation rates."""

import pytest

from src.early_warning.early_warning_data_contract import EarlyWarningInputRecord
from src.risk.risk_config import TemporalFeatureConfig
from src.risk.risk_features import TemporalRiskFeatureExtractor


def test_cross_horizon_spread_and_monotonicity():
    """Verify cross-horizon spread (H1 - H5) and monotonic degradation flag."""
    extractor = TemporalRiskFeatureExtractor()

    # Monotonic degradation: H1=0.8, H3=0.6, H5=0.3
    features = extractor.extract_features(
        sequence_id="seq_001",
        horizon_reliabilities={1: 0.80, 3: 0.60, 5: 0.30},
        history_records=None,
    )
    assert features.is_monotonic_degradation is True
    assert features.cross_horizon_spread == pytest.approx(0.50)
    # Degradation rate over 4 frames: (0.80 - 0.30) / 4 = 0.125
    assert features.degradation_rate == pytest.approx(0.125)


def test_non_monotonic_spread():
    """Verify non-monotonic pattern (e.g. H1=0.3, H3=0.7, H5=0.5)."""
    extractor = TemporalRiskFeatureExtractor()
    features = extractor.extract_features(
        sequence_id="seq_002",
        horizon_reliabilities={1: 0.30, 3: 0.70, 5: 0.50},
        history_records=None,
    )
    assert features.is_monotonic_degradation is False
    assert features.cross_horizon_spread == pytest.approx(-0.20)


def test_historical_trend_slope_and_volatility():
    """Verify linear trend slope and volatility calculation over retrospective history."""
    extractor = TemporalRiskFeatureExtractor(
        TemporalFeatureConfig(history_window_size=5, min_history_points=3)
    )

    # Linearly decreasing historical reliability: 0.9, 0.8, 0.7, 0.6, 0.5 (slope = -0.10)
    history = [
        EarlyWarningInputRecord(
            prediction_id=f"seq_prev_{i}_H1",
            sequence_id=f"seq_prev_{i}",
            source_frame_id=f"frame_{i}.jpg",
            target_frame_id=f"frame_{i+1}.jpg",
            horizon=1,
            predicted_reliability=rel,
        )
        for i, rel in enumerate([0.9, 0.8, 0.7, 0.6, 0.5])
    ]

    features = extractor.extract_features(
        sequence_id="seq_curr",
        horizon_reliabilities={1: 0.50, 3: 0.40, 5: 0.30},
        history_records=history,
    )

    assert features.available_history_points == 5
    assert features.historical_trend_slope == pytest.approx(-0.10, abs=1e-3)
    assert features.recent_volatility is not None
    assert features.recent_volatility > 0.0


def test_insufficient_history_handling():
    """Verify graceful handling when history points are below min_history_points."""
    extractor = TemporalRiskFeatureExtractor(
        TemporalFeatureConfig(history_window_size=5, min_history_points=3)
    )

    history = [
        {"predicted_reliability": 0.85}  # only 1 point
    ]

    features = extractor.extract_features(
        sequence_id="seq_curr",
        horizon_reliabilities={1: 0.80},
        history_records=history,
    )

    assert features.available_history_points == 1
    assert features.historical_trend_slope is None
    assert features.recent_volatility is None
