"""Comprehensive unit and integration test suite for Camera & Weather Intelligence (Feature 1)."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.camera_weather import (
    BlurClassification,
    CameraBlurEvidence,
    DegradationTrend,
    FogClassification,
    FogHazeEvidence,
    GlareClassification,
    LightingClassification,
    LowLightEvidence,
    RainClassification,
    RainEvidence,
    RiskImpactLevel,
    SunGlareEvidence,
    TemporalDegradationType,
    TemporalWeatherTracker,
    VisibilityForecaster,
    VisibilityLevel,
    VisibilityQualityEngine,
    analyze_blur,
    analyze_fog,
    analyze_glare,
    analyze_lighting,
    analyze_rain,
    estimate_perception_impact,
)
from src.video_processing.video_pipeline_runner import VideoPipelineRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ------------------------------------------------------------------------------
# 1. Optical Analyzer Unit Tests
# ------------------------------------------------------------------------------

def test_blur_analyzer_sharp_vs_blurred():
    """Verify blur analyzer correctly distinguishes sharp checkerboard edges from Gaussian blurred images."""
    # Synthetic sharp image: high frequency checkerboard pattern
    sharp = np.zeros((240, 320), dtype=np.uint8)
    sharp[::20, :] = 255
    sharp[:, ::20] = 255

    sharp_ev = analyze_blur(sharp)
    assert sharp_ev.classification in (BlurClassification.SHARP, BlurClassification.SLIGHT_BLUR)
    assert sharp_ev.severity < 0.35
    assert sharp_ev.laplacian_variance > 100.0

    # Gaussian heavily blurred version
    blurred = cv2.GaussianBlur(sharp, (31, 31), 10.0)
    blur_ev = analyze_blur(blurred)
    assert blur_ev.classification in (BlurClassification.MODERATE_BLUR, BlurClassification.SEVERE_BLUR)
    assert blur_ev.severity > sharp_ev.severity
    assert blur_ev.laplacian_variance < 50.0


def test_lighting_analyzer_normal_vs_dark():
    """Verify lighting analyzer distinguishes daylight illumination from critical darkness."""
    # Normal daylight image: mean ~ 130
    normal = np.full((240, 320), 130, dtype=np.uint8)
    norm_ev = analyze_lighting(normal)
    assert norm_ev.classification == LightingClassification.NORMAL
    assert norm_ev.severity < 0.20
    assert norm_ev.mean_luminance > 100.0

    # Critical darkness image: mean ~ 15
    dark = np.full((240, 320), 15, dtype=np.uint8)
    dark_ev = analyze_lighting(dark)
    assert dark_ev.classification in (LightingClassification.CRITICAL_DARKNESS, LightingClassification.VERY_LOW_LIGHT)
    assert dark_ev.severity > 0.65
    assert dark_ev.underexposed_pixel_ratio > 0.80


def test_glare_analyzer_overexposed():
    """Verify glare analyzer flags saturated sensor blooms."""
    normal = np.full((240, 320, 3), 120, dtype=np.uint8)
    gray_norm = np.full((240, 320), 120, dtype=np.uint8)
    norm_glare = analyze_glare(normal, gray_norm)
    assert norm_glare.classification == GlareClassification.NORMAL
    assert norm_glare.severity < 0.15

    # Blinding glare: top center saturated pixels
    glare_bgr = normal.copy()
    glare_gray = gray_norm.copy()
    glare_bgr[:120, 60:260] = 255
    glare_gray[:120, 60:260] = 255

    glare_ev = analyze_glare(glare_bgr, glare_gray)
    assert glare_ev.classification in (GlareClassification.SEVERE_GLARE, GlareClassification.OVEREXPOSURE, GlareClassification.ELEVATED_GLARE)
    assert glare_ev.saturated_pixel_ratio > 0.05
    assert glare_ev.severity > 0.25


def test_fog_analyzer_dense_vs_clear():
    """Verify fog analyzer detects contrast attenuation and atmospheric veil."""
    # Clear high-contrast image
    clear_bgr = np.random.randint(20, 220, (240, 320, 3), dtype=np.uint8)
    clear_gray = cv2.cvtColor(clear_bgr, cv2.COLOR_BGR2GRAY)
    clear_ev = analyze_fog(clear_bgr, clear_gray)
    assert clear_ev.classification in (FogClassification.CLEAR, FogClassification.LIGHT_HAZE)

    # Dense fog simulation: high atmospheric baseline + washed-out contrast
    fog_bgr = np.full((240, 320, 3), 165, dtype=np.uint8)
    fog_bgr += np.random.randint(-10, 10, (240, 320, 3), dtype=np.int8).astype(np.uint8)
    fog_gray = cv2.cvtColor(fog_bgr, cv2.COLOR_BGR2GRAY)
    fog_ev = analyze_fog(fog_bgr, fog_gray)
    assert fog_ev.classification in (FogClassification.DENSE_FOG, FogClassification.MODERATE_FOG)
    assert fog_ev.severity > 0.40


def test_rain_analyzer_streak_detection():
    """Verify rain analyzer detects directional vertical streak responses."""
    # Synthetic vertical streak image
    rain_bgr = np.full((240, 320, 3), 100, dtype=np.uint8)
    rain_bgr[::5, :] = 190
    rain_gray = cv2.cvtColor(rain_bgr, cv2.COLOR_BGR2GRAY)
    rain_ev = analyze_rain(rain_bgr, rain_gray)
    assert rain_ev.streak_variance > 50.0
    assert rain_ev.classification in (RainClassification.LIGHT_RAIN, RainClassification.MODERATE_RAIN, RainClassification.HEAVY_RAIN, RainClassification.UNCERTAIN)


# ------------------------------------------------------------------------------
# 2. Unified Visibility Engine Tests
# ------------------------------------------------------------------------------

def test_visibility_engine_scoring_and_normalization():
    """Verify VisibilityQualityEngine produces mathematically bounded and explainable scores."""
    engine = VisibilityQualityEngine()

    # Clear daylight frame with high-contrast edges and balanced luminance
    clear_bgr = np.full((240, 320, 3), 130, dtype=np.uint8)
    clear_bgr[::20, :] = 230
    clear_bgr[:, ::20] = 30
    (
        vis_score,
        vis_level,
        cam_quality,
        primary_cond,
        cond_labels,
        cond_sev,
        rain_ev,
        fog_ev,
        blur_ev,
        light_ev,
        glare_ev,
        quality_metrics,
        conf,
        uncert,
        reasons,
    ) = engine.analyze_frame(clear_bgr)

    assert 0.0 <= vis_score <= 100.0
    assert 0.0 <= cam_quality <= 100.0
    assert vis_level in (VisibilityLevel.EXCELLENT, VisibilityLevel.GOOD)
    assert 0.0 <= conf <= 1.0
    assert 0.0 <= uncert <= 1.0
    assert len(reasons) >= 1
    assert "sharpness" in quality_metrics.to_dict()


# ------------------------------------------------------------------------------
# 3. Temporal Tracker & Smoothing Tests
# ------------------------------------------------------------------------------

def test_temporal_tracker_sudden_degradation():
    """Verify TemporalWeatherTracker detects sudden degradation vs gradual dynamics."""
    tracker = TemporalWeatherTracker(history_window=20)

    # Prime with normal clear scores
    for i in range(5):
        smoothed, roc, deg = tracker.update(i, i * 0.033, 95.0, "CLEAR")

    # Sudden severe drop (e.g. tunnel or mud splash)
    smoothed, roc, deg = tracker.update(6, 6 * 0.033, 20.0, "LOW_LIGHT")
    assert deg == TemporalDegradationType.SUDDEN_DEGRADATION
    assert roc < -15.0
    assert len(tracker.transitions) >= 1
    assert tracker.transitions[-1].from_condition == "CLEAR"
    assert tracker.transitions[-1].to_condition == "LOW_LIGHT"


def test_temporal_tracker_temporary_interruption():
    """Verify TemporalWeatherTracker correctly classifies temporary dips (wiper sweep)."""
    tracker = TemporalWeatherTracker(history_window=20)

    tracker.update(0, 0.00, 90.0, "CLEAR")
    tracker.update(1, 0.03, 90.0, "CLEAR")
    tracker.update(2, 0.06, 35.0, "BLUR")  # Temporary dip
    smoothed, roc, deg = tracker.update(3, 0.10, 88.0, "CLEAR")  # Immediate rebound

    assert deg == TemporalDegradationType.TEMPORARY_INTERRUPTION


# ------------------------------------------------------------------------------
# 4. Multi-Horizon Forecasting Tests
# ------------------------------------------------------------------------------

def test_visibility_forecaster_multi_horizon():
    """Verify VisibilityForecaster projects Short, Med, and Long lookahead horizons."""
    forecaster = VisibilityForecaster()

    # Decreasing history scores (gradual degradation)
    history = [92.0, 90.0, 87.0, 84.0, 80.0, 77.0, 74.0]
    forecast = forecaster.forecast(history, 72.0, fps=30.0)

    assert forecast.degradation_trend in (DegradationTrend.GRADUALLY_DEGRADING, DegradationTrend.RAPIDLY_DEGRADING)
    assert forecast.trend_velocity < 0.0

    horizons = forecast.horizons
    assert "h_short" in horizons
    assert "h_med" in horizons
    assert "h_long" in horizons

    # Near future should continue projected downward slope
    assert horizons["h_short"]["predicted_score"] <= 72.0
    assert horizons["h_med"]["predicted_score"] <= horizons["h_short"]["predicted_score"]

    # Uncertainty should expand over time
    assert horizons["h_long"]["uncertainty_radius"] > horizons["h_short"]["uncertainty_radius"]


def test_visibility_forecaster_insufficient_data():
    """Verify forecaster safely reports INSUFFICIENT_DATA when history < 5 frames."""
    forecaster = VisibilityForecaster()
    forecast = forecaster.forecast([90.0, 89.0], 88.0, fps=30.0)
    assert forecast.degradation_trend == DegradationTrend.INSUFFICIENT_DATA


# ------------------------------------------------------------------------------
# 5. Perception Impact Interface Tests
# ------------------------------------------------------------------------------

def test_perception_impact_indicator():
    """Verify downstream interface formats estimated detection impact indicators."""
    blur_ev = CameraBlurEvidence(BlurClassification.SEVERE_BLUR, 0.85, 30.0, 50.0, 0.01, "Severe blur")
    fog_ev = FogHazeEvidence(FogClassification.CLEAR, 0.05, 30.0, 0.1, 0.8, "Clear")
    rain_ev = RainEvidence(RainClassification.NO_RAIN, 0.05, 0.0, 10.0, 0.1, "No rain")
    light_ev = LowLightEvidence(LightingClassification.NORMAL, 0.05, 120.0, 0.01, 80.0, "Normal")
    glare_ev = SunGlareEvidence(GlareClassification.NORMAL, 0.05, 0.01, 0.01, 0.0, "Normal")

    impact = estimate_perception_impact(
        visibility_score=45.0,
        blur_ev=blur_ev,
        fog_ev=fog_ev,
        rain_ev=rain_ev,
        light_ev=light_ev,
        glare_ev=glare_ev,
    )

    assert impact.is_estimated is True
    assert impact.expected_confidence_penalty > 0.15
    assert impact.boundary_degradation_risk in (RiskImpactLevel.HIGH, RiskImpactLevel.CRITICAL)
    assert impact.suggested_detection_sensitivity in ("HEIGHTENED", "CONSERVATIVE")


# ------------------------------------------------------------------------------
# 6. Real Video Clip End-to-End Integration Tests
# ------------------------------------------------------------------------------

def test_real_video_clip_weather_processing(tmp_path):
    """Execute end-to-end pipeline on real driving video clip and verify structured weather intelligence."""
    clip_path = PROJECT_ROOT / "outputs" / "day14_validation" / "clips" / "scenario_a_normal_daytime.mp4"
    if not clip_path.exists():
        pytest.skip("Scenario A video clip not found in test environment.")

    runner = VideoPipelineRunner(project_root=PROJECT_ROOT)
    result = runner.run(
        video_path=clip_path,
        job_id="test_weather_run_01",
        output_dir=tmp_path / "job_out",
        max_frames=8,
    )

    assert result.frames_processed == 8
    assert "weather_summary" in result.to_dict()
    summary = result.weather_summary
    assert "mean_visibility_score" in summary
    assert "predominant_condition" in summary
    assert summary["total_frames_analyzed"] == 8

    # Inspect first frame weather intelligence
    frame0 = result.frames[0]
    w_intel = frame0.weather_intelligence
    assert "visibility_score" in w_intel
    assert "visibility_level" in w_intel
    assert "camera_quality_score" in w_intel
    assert "rain" in w_intel
    assert "fog" in w_intel
    assert "blur" in w_intel
    assert "lighting" in w_intel
    assert "glare" in w_intel
    assert "forecast" in w_intel
    assert "perception_impact" in w_intel
    assert isinstance(w_intel["degradation_reasons"], list)


def test_fastapi_weather_endpoint_schema():
    """Verify FastAPI router returns valid structured weather intelligence or raises ResourceNotFoundError."""
    from src.api.routers.video import get_video_job_weather
    from src.api.exceptions import ResourceNotFoundError

    # Non-existent job raises ResourceNotFoundError (HTTP 404)
    with pytest.raises(ResourceNotFoundError):
        get_video_job_weather("non_existent_12345")
