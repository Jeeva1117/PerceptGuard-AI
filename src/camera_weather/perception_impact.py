"""Downstream Object Detection Reliability Impact Interface.

Formulates estimated perception impact indicators for Feature 2 (Object Detection & Tracking)
based on current optical degradation (blur, fog, rain, low-light, glare).

IMPORTANT DISTINCTION:
Camera image quality is not equivalent to actual object detector accuracy.
This module produces an *estimated perception-impact indicator*, clearly labeled as an estimate,
enabling downstream perception modules to adjust detection sensitivity and uncertainty margins.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from src.camera_weather.schemas import (
    BlurClassification,
    CameraBlurEvidence,
    FogClassification,
    FogHazeEvidence,
    GlareClassification,
    LightingClassification,
    LowLightEvidence,
    PerceptionImpactIndicator,
    RainClassification,
    RainEvidence,
    RiskImpactLevel,
    SunGlareEvidence,
)


def estimate_perception_impact(
    visibility_score: float,
    blur_ev: CameraBlurEvidence,
    fog_ev: FogHazeEvidence,
    rain_ev: RainEvidence,
    light_ev: LowLightEvidence,
    glare_ev: SunGlareEvidence,
) -> PerceptionImpactIndicator:
    """Compute estimated perception impact indicators for downstream detection models.

    Args:
        visibility_score: Unified visibility score [0.0, 100.0]
        blur_ev: Camera blur measurements
        fog_ev: Fog and haze measurements
        rain_ev: Precipitation measurements
        light_ev: Low-light measurements
        glare_ev: Sun glare measurements

    Returns:
        PerceptionImpactIndicator dataclass for downstream consumption.
    """
    # 1. Expected confidence penalty (0.0 to 0.40)
    # Scaled monotonically with visibility degradation
    degradation_factor = (100.0 - visibility_score) / 100.0
    expected_conf_penalty = float(np.clip(degradation_factor * 0.35, 0.0, 0.35))

    # 2. Boundary degradation risk (affects bounding box tightness and IoU overlap)
    # Strongly driven by blur and water droplet distortion
    boundary_score = blur_ev.severity * 0.55 + rain_ev.severity * 0.30 + glare_ev.severity * 0.15
    if boundary_score > 0.65:
        boundary_risk = RiskImpactLevel.CRITICAL
    elif boundary_score > 0.45:
        boundary_risk = RiskImpactLevel.HIGH
    elif boundary_score > 0.25:
        boundary_risk = RiskImpactLevel.MODERATE
    else:
        boundary_risk = RiskImpactLevel.LOW

    # 3. Small-object visibility risk (distant pedestrians, bicycles, distant cars)
    # Strongly driven by fog veil, low contrast, and darkness
    small_obj_score = fog_ev.severity * 0.50 + light_ev.severity * 0.35 + blur_ev.severity * 0.15
    if small_obj_score > 0.65:
        small_obj_risk = RiskImpactLevel.CRITICAL
    elif small_obj_score > 0.45:
        small_obj_risk = RiskImpactLevel.HIGH
    elif small_obj_score > 0.25:
        small_obj_risk = RiskImpactLevel.MODERATE
    else:
        small_obj_risk = RiskImpactLevel.LOW

    # 4. Suggested detection sensitivity mode
    if visibility_score >= 75.0 and boundary_risk == RiskImpactLevel.LOW:
        suggested_mode = "NORMAL"
        rationale = (
            "Nominal optical conditions. Standard YOLO confidence thresholds (>=0.25) "
            "and ByteTrack association parameters recommended."
        )
    elif visibility_score >= 50.0:
        suggested_mode = "HEIGHTENED"
        rationale = (
            f"Mild-to-moderate degradation detected (Visibility {visibility_score:.0f}%). "
            f"Estimated detector confidence penalty is ~{expected_conf_penalty*100:.1f}%. "
            "Lowering confidence threshold to 0.18 with multi-frame track confirmation advised."
        )
    else:
        suggested_mode = "CONSERVATIVE"
        rationale = (
            f"Severe optical degradation (Visibility {visibility_score:.0f}%). "
            f"High risk of missed small objects and bounding box jitter. "
            "Flagging object detector tracks with elevated perception uncertainty."
        )

    return PerceptionImpactIndicator(
        expected_confidence_penalty=expected_conf_penalty,
        boundary_degradation_risk=boundary_risk,
        small_object_risk=small_obj_risk,
        suggested_detection_sensitivity=suggested_mode,
        rationale=rationale,
        is_estimated=True,
    )
