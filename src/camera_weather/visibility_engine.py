"""Unified Visibility Analysis Engine for PerceptGuard AI.

Combines optical measurements across:
- Sharpness and focal blur
- Dynamic contrast and range
- Ambient luminance and underexposure
- Sensor saturation and glare blooming
- Atmospheric fog/haze attenuation
- Precipitation streak and droplet distortion
- Field obstruction and vignetting

DOCUMENTATION OF SCORING FORMULATION:
-------------------------------------
1. Camera Quality Score (0 - 100):
   Measures hardware sensor performance and optical fidelity:
   - Sharpness (30%): Min(100, LaplacianVar / 2.0)
   - Dynamic Contrast (25%): Min(100, (ContrastStd / 60.0) * 100)
   - Exposure Balance (25%): 100 - 100 * (|MeanLum - 120| / 120) * 0.7 - SaturatedRatio * 200
   - Field Uniformity & Dynamic Spread (20%): Spread / 1.8

2. Environmental Visibility Score (0 - 100):
   Measures scene-level road and obstacle visibility available to human drivers and perception AI:
   Base = 100.0. Penalties are weighted deductions:
   - P_blur    = 35.0 * blur_evidence.severity
   - P_fog     = 40.0 * fog_evidence.severity
   - P_rain    = 35.0 * rain_evidence.severity
   - P_dark    = 45.0 * lighting_evidence.severity
   - P_glare   = 35.0 * glare_evidence.severity
   - P_obstr   = 30.0 * obstruction_severity
   VisibilityScore = Clip(100.0 - TotalPenalties, 0.0, 100.0).

3. Categorical Thresholds:
   - EXCELLENT: VisibilityScore >= 80.0
   - GOOD:      65.0 <= VisibilityScore < 80.0
   - DEGRADED:  45.0 <= VisibilityScore < 65.0
   - CRITICAL:  VisibilityScore < 45.0

4. Scientific Caveats & Limitations:
   The visibility score is an empirical optical-atmospheric degradation index.
   It cannot guarantee physical meteorological visibility distance (e.g., in meters)
   without calibrated optical depth sensors and external ground-truth transmissometers.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from src.camera_weather.optical_analyzers import (
    analyze_blur,
    analyze_fog,
    analyze_glare,
    analyze_lighting,
    analyze_rain,
)
from src.camera_weather.schemas import (
    BlurClassification,
    CameraBlurEvidence,
    CameraQualityMetrics,
    FogClassification,
    FogHazeEvidence,
    GlareClassification,
    LightingClassification,
    LowLightEvidence,
    RainClassification,
    RainEvidence,
    SunGlareEvidence,
    VisibilityLevel,
)


class VisibilityQualityEngine:
    """Production engine combining multi-spectral optical measurements into unified visibility telemetry."""

    def __init__(
        self,
        blur_weight: float = 35.0,
        fog_weight: float = 40.0,
        rain_weight: float = 35.0,
        lighting_weight: float = 45.0,
        glare_weight: float = 35.0,
        obstruction_weight: float = 30.0,
    ):
        self.blur_weight = blur_weight
        self.fog_weight = fog_weight
        self.rain_weight = rain_weight
        self.lighting_weight = lighting_weight
        self.glare_weight = glare_weight
        self.obstruction_weight = obstruction_weight

    def _check_obstruction(self, gray_img: np.ndarray) -> Tuple[float, Optional[str]]:
        """Detect field-of-view irregularity, dirt blobs, or camera lens vignetting."""
        h, w = gray_img.shape
        center_roi = gray_img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_brightness = float(np.mean(center_roi))

        corner_brightness = float((
            np.mean(gray_img[: h // 8, : w // 8]) +
            np.mean(gray_img[: h // 8, -w // 8 :]) +
            np.mean(gray_img[-h // 8 :, : w // 8]) +
            np.mean(gray_img[-h // 8 :, -w // 8 :])
        ) / 4.0)

        delta = abs(center_brightness - corner_brightness)
        if delta > 75.0:
            severity = float(min(1.0, (delta - 75.0) / 75.0 + 0.5))
            reason = f"Lens obstruction or heavy vignetting (center-corner delta {delta:.1f} > 75.0)."
            return severity, reason
        elif delta > 50.0:
            severity = float((delta - 50.0) / 50.0 * 0.4)
            reason = f"Mild optical vignetting detected (center-corner delta {delta:.1f})."
            return severity, reason
        return 0.0, None

    def _compute_camera_quality(
        self,
        blur: CameraBlurEvidence,
        lighting: LowLightEvidence,
        glare: SunGlareEvidence,
        fog: FogHazeEvidence,
        gray_img: np.ndarray,
    ) -> CameraQualityMetrics:
        """Measure intrinsic camera image quality across 5 dimensions."""
        # 1. Sharpness index (0 to 100)
        sharpness = float(np.clip((blur.laplacian_variance / 180.0) * 100.0, 0.0, 100.0))

        # 2. Contrast index (0 to 100)
        contrast_std = float(np.std(gray_img))
        contrast = float(np.clip((contrast_std / 58.0) * 100.0, 0.0, 100.0))

        # 3. Luminance index (0 to 100)
        lum_dev = abs(lighting.mean_luminance - 120.0)
        luminance = float(np.clip(100.0 - (lum_dev / 120.0) * 85.0, 0.0, 100.0))

        # 4. Exposure fidelity (0 to 100)
        exposure = float(np.clip(
            100.0 - (lighting.underexposed_pixel_ratio * 70.0 + glare.saturated_pixel_ratio * 120.0),
            0.0,
            100.0,
        ))

        # 5. Field uniformity (0 to 100)
        h, w = gray_img.shape
        quads = [
            np.mean(gray_img[: h // 2, : w // 2]),
            np.mean(gray_img[: h // 2, w // 2 :]),
            np.mean(gray_img[h // 2 :, : w // 2]),
            np.mean(gray_img[h // 2 :, w // 2 :]),
        ]
        quad_std = float(np.std(quads))
        uniformity = float(np.clip(100.0 - (quad_std / 30.0) * 80.0, 0.0, 100.0))

        return CameraQualityMetrics(
            sharpness=sharpness,
            contrast=contrast,
            luminance=luminance,
            exposure=exposure,
            field_uniformity=uniformity,
        )

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        prev_gray: Optional[np.ndarray] = None,
    ) -> Tuple[
        float,  # visibility_score
        VisibilityLevel,
        float,  # camera_quality_score
        str,    # primary_condition
        Dict[str, str],   # condition_labels
        Dict[str, float], # condition_severity
        RainEvidence,
        FogHazeEvidence,
        CameraBlurEvidence,
        LowLightEvidence,
        SunGlareEvidence,
        CameraQualityMetrics,
        float,  # confidence
        float,  # uncertainty
        List[str], # degradation_reasons
    ]:
        """Execute end-to-end multi-spectral analysis on a single video frame."""
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            raise ValueError("Invalid frame: Frame buffer is None, non-array, or empty.")

        if len(frame_bgr.shape) == 2:
            gray_img = frame_bgr
            frame_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        else:
            gray_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Individual condition analyzers
        blur_ev = analyze_blur(gray_img)
        light_ev = analyze_lighting(gray_img)
        glare_ev = analyze_glare(frame_bgr, gray_img)
        fog_ev = analyze_fog(frame_bgr, gray_img)
        rain_ev = analyze_rain(frame_bgr, gray_img, prev_gray=prev_gray)
        obstr_sev, obstr_reason = self._check_obstruction(gray_img)

        # 2. Camera quality metrics
        quality_metrics = self._compute_camera_quality(blur_ev, light_ev, glare_ev, fog_ev, gray_img)
        camera_quality_score = float((
            quality_metrics.sharpness * 0.30 +
            quality_metrics.contrast * 0.25 +
            quality_metrics.luminance * 0.20 +
            quality_metrics.exposure * 0.15 +
            quality_metrics.field_uniformity * 0.10
        ))

        # 3. Calculate visibility deductions
        degradation_reasons: List[str] = []
        penalties = 0.0

        if blur_ev.severity > 0.15:
            p = self.blur_weight * blur_ev.severity
            penalties += p
            degradation_reasons.append(blur_ev.details)

        if fog_ev.severity > 0.15:
            p = self.fog_weight * fog_ev.severity
            penalties += p
            degradation_reasons.append(fog_ev.details)

        if rain_ev.severity > 0.15:
            p = self.rain_weight * rain_ev.severity
            penalties += p
            degradation_reasons.append(rain_ev.details)

        if light_ev.severity > 0.15:
            p = self.lighting_weight * light_ev.severity
            penalties += p
            degradation_reasons.append(light_ev.details)

        if glare_ev.severity > 0.15:
            p = self.glare_weight * glare_ev.severity
            penalties += p
            degradation_reasons.append(glare_ev.details)

        if obstr_sev > 0.15 and obstr_reason:
            penalties += self.obstruction_weight * obstr_sev
            degradation_reasons.append(obstr_reason)

        raw_vis_score = max(0.0, min(100.0, 100.0 - penalties))

        # 4. Determine categorical visibility level
        if raw_vis_score >= 80.0:
            level = VisibilityLevel.EXCELLENT
        elif raw_vis_score >= 65.0:
            level = VisibilityLevel.GOOD
        elif raw_vis_score >= 45.0:
            level = VisibilityLevel.DEGRADED
        else:
            level = VisibilityLevel.CRITICAL

        # 5. Determine primary dominant condition
        severities = {
            "RAIN": rain_ev.severity,
            "FOG": fog_ev.severity,
            "BLUR": blur_ev.severity,
            "LOW_LIGHT": light_ev.severity,
            "SUN_GLARE": glare_ev.severity,
        }
        max_cond, max_sev = max(severities.items(), key=lambda item: item[1])
        if max_sev < 0.20 and obstr_sev < 0.25:
            primary_condition = "CLEAR"
        elif obstr_sev > max_sev:
            primary_condition = "OBSTRUCTION"
        else:
            primary_condition = max_cond

        if not degradation_reasons:
            degradation_reasons.append("Atmospheric and sensor clarity optimal. No visual degradation detected.")

        # 6. Condition labels & severities mapping
        condition_labels = {
            "rain": rain_ev.classification.value,
            "fog": fog_ev.classification.value,
            "blur": blur_ev.classification.value,
            "lighting": light_ev.classification.value,
            "glare": glare_ev.classification.value,
        }

        # 7. Confidence & uncertainty estimation
        # Confidence is high when measurements are definitive (strong contrast / clear signals)
        uncertainty = float(np.clip(
            0.08 + (0.15 if rain_ev.classification == RainClassification.UNCERTAIN else 0.0) +
            (0.12 if fog_ev.classification == FogClassification.UNKNOWN else 0.0) +
            (0.10 if blur_ev.classification == BlurClassification.UNKNOWN else 0.0),
            0.05,
            0.45,
        ))
        confidence = float(1.0 - uncertainty)

        return (
            raw_vis_score,
            level,
            camera_quality_score,
            primary_condition,
            condition_labels,
            severities,
            rain_ev,
            fog_ev,
            blur_ev,
            light_ev,
            glare_ev,
            quality_metrics,
            confidence,
            uncertainty,
            degradation_reasons,
        )
