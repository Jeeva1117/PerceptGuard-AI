"""Optical analyzers and computer vision algorithms for Camera & Weather Intelligence.

Implements rigorous numerical measurements for:
- Camera blur (Laplacian variance, Tenengrad gradient energy, edge density)
- Ambient illumination and low-light degradation
- Sun glare, sensor blooming, and optical overexposure
- Atmospheric haze and fog (Dark Channel Prior approximation and contrast loss)
- Precipitation and rain artifacts (vertical streak variance, droplet patterns)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from src.camera_weather.schemas import (
    BlurClassification,
    CameraBlurEvidence,
    FogClassification,
    FogHazeEvidence,
    GlareClassification,
    LightingClassification,
    LowLightEvidence,
    RainClassification,
    RainEvidence,
    SunGlareEvidence,
)


def analyze_blur(gray_img: np.ndarray) -> CameraBlurEvidence:
    """Analyze camera optical sharpness and categorize blur severity.

    Measures:
    1. Focus measure: Variance of the Laplacian kernel response
    2. Tenengrad gradient energy: Mean of squared Sobel gradients
    3. High-frequency structural edge density using Canny detector

    Calibrated severity scale [0.0 = perfectly sharp, 1.0 = severely blurred].
    """
    h, w = gray_img.shape
    total_pixels = float(h * w)

    # 1. Laplacian variance
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    lap_var = float(laplacian.var())

    # 2. Tenengrad gradient energy
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobel_x**2 + sobel_y**2))

    # 3. Edge density
    edges = cv2.Canny(gray_img, 50, 150)
    edge_density = float(np.count_nonzero(edges) / total_pixels)

    # Calibrated classification
    if lap_var < 45.0 or (lap_var < 60.0 and tenengrad < 150.0):
        classification = BlurClassification.SEVERE_BLUR
        severity = float(np.clip(1.0 - (lap_var / 50.0) * 0.25, 0.75, 1.0))
        details = f"Severe optical defocus or heavy motion blur (Laplacian {lap_var:.1f} < 45.0, Tenengrad {tenengrad:.1f})."
    elif lap_var < 95.0 or (lap_var < 110.0 and edge_density < 0.025):
        classification = BlurClassification.MODERATE_BLUR
        severity = float(np.clip(0.45 + (1.0 - lap_var / 95.0) * 0.30, 0.45, 0.75))
        details = f"Moderate blur affecting fine vehicle and pedestrian contours (Laplacian {lap_var:.1f}, Edge density {edge_density:.3f})."
    elif lap_var < 150.0 or edge_density < 0.04:
        classification = BlurClassification.SLIGHT_BLUR
        severity = float(np.clip(0.15 + (1.0 - lap_var / 150.0) * 0.30, 0.15, 0.45))
        details = f"Slight optical softening detected (Laplacian {lap_var:.1f}, Edge density {edge_density:.3f})."
    else:
        classification = BlurClassification.SHARP
        severity = float(np.clip((150.0 - min(lap_var, 300.0)) / 1000.0, 0.0, 0.15))
        details = f"Optical sharpness optimal with crisp object boundaries (Laplacian {lap_var:.1f})."

    return CameraBlurEvidence(
        classification=classification,
        severity=severity,
        laplacian_variance=lap_var,
        tenengrad_gradient_mean=tenengrad,
        edge_density=edge_density,
        details=details,
    )


def analyze_lighting(gray_img: np.ndarray) -> LowLightEvidence:
    """Analyze ambient scene illumination and dark exposure degradation.

    Measures:
    1. Mean luminance (overall intensity level in [0, 255])
    2. Underexposed pixel fraction (pixels with intensity < 30)
    3. Dynamic range spread (difference between 95th and 5th percentiles)

    Calibrated severity scale [0.0 = normal daylight/well-lit, 1.0 = pitch darkness].
    """
    h, w = gray_img.shape
    total_pixels = float(h * w)

    mean_lum = float(np.mean(gray_img))
    underexposed_count = np.count_nonzero(gray_img < 30)
    underexposed_ratio = float(underexposed_count / total_pixels)

    p95 = float(np.percentile(gray_img, 95))
    p5 = float(np.percentile(gray_img, 5))
    spread = float(max(0.0, p95 - p5))

    # Calibrated classification
    if mean_lum < 28.0 or underexposed_ratio > 0.65:
        classification = LightingClassification.CRITICAL_DARKNESS
        severity = float(np.clip(0.80 + (1.0 - mean_lum / 28.0) * 0.20, 0.80, 1.0))
        details = f"Critical darkness: severe loss of scene visibility (Luminance {mean_lum:.1f}, {underexposed_ratio*100:.1f}% underexposed)."
    elif mean_lum < 52.0 or underexposed_ratio > 0.40:
        classification = LightingClassification.VERY_LOW_LIGHT
        severity = float(np.clip(0.55 + (1.0 - mean_lum / 52.0) * 0.25, 0.55, 0.80))
        details = f"Very low ambient light: high noise and object silhouette masking (Luminance {mean_lum:.1f})."
    elif mean_lum < 80.0 or (mean_lum < 95.0 and spread < 60.0):
        classification = LightingClassification.LOW_LIGHT
        severity = float(np.clip(0.20 + (1.0 - mean_lum / 80.0) * 0.35, 0.20, 0.55))
        details = f"Low ambient lighting conditions (Luminance {mean_lum:.1f}, Dynamic spread {spread:.1f})."
    else:
        classification = LightingClassification.NORMAL
        severity = float(np.clip((80.0 - min(mean_lum, 180.0)) / 500.0, 0.0, 0.20))
        details = f"Normal scene illumination within optimal dynamic range (Luminance {mean_lum:.1f})."

    return LowLightEvidence(
        classification=classification,
        severity=severity,
        mean_luminance=mean_lum,
        underexposed_pixel_ratio=underexposed_ratio,
        dynamic_range_spread=spread,
        details=details,
    )


def analyze_glare(frame_bgr: np.ndarray, gray_img: np.ndarray) -> SunGlareEvidence:
    """Analyze optical sun glare, headlight blinding, and sensor overexposure.

    Measures:
    1. Saturated pixel fraction (pixels with intensity > 245)
    2. Cluster ratio of saturated pixels in upper half of the camera view
    3. Dynamic contrast washout caused by forward scatter

    Calibrated severity scale [0.0 = normal exposure, 1.0 = blinding overexposure].
    """
    h, w = gray_img.shape
    total_pixels = float(h * w)

    # Upper 60% of frame where sun and streetlights predominantly cause direct lens flare
    upper_h = int(h * 0.60)
    upper_gray = gray_img[:upper_h, :]

    saturated_mask = (gray_img > 245).astype(np.uint8)
    saturated_count = int(np.count_nonzero(saturated_mask))
    saturated_ratio = float(saturated_count / total_pixels)

    # Measure spatial clustering of glare in upper half
    upper_saturated = saturated_mask[:upper_h, :]
    cluster_ratio = float(np.count_nonzero(upper_saturated) / float(upper_h * w))

    # Washout check: low contrast standard deviation in the bright region
    if saturated_count > 100:
        bright_pixels = gray_img[saturated_mask > 0]
        washout = float(1.0 - min(1.0, np.std(bright_pixels) / 10.0))
    else:
        washout = 0.0

    # Calibrated classification
    if saturated_ratio > 0.15 or (cluster_ratio > 0.12 and saturated_ratio > 0.08):
        classification = GlareClassification.OVEREXPOSURE
        severity = float(np.clip(0.75 + (saturated_ratio - 0.15) * 1.5, 0.75, 1.0))
        details = f"Direct sensor blinding or total overexposure ({saturated_ratio*100:.1f}% saturated pixels)."
    elif saturated_ratio > 0.07 or cluster_ratio > 0.08:
        classification = GlareClassification.SEVERE_GLARE
        severity = float(np.clip(0.45 + (saturated_ratio - 0.07) * 3.5, 0.45, 0.75))
        details = f"Severe sun glare or headlight reflection causing localized field blinding ({saturated_ratio*100:.1f}% saturated)."
    elif saturated_ratio > 0.025 or cluster_ratio > 0.035:
        classification = GlareClassification.ELEVATED_GLARE
        severity = float(np.clip(0.15 + (saturated_ratio - 0.025) * 5.0, 0.15, 0.45))
        details = f"Elevated glare reflections detected in camera field ({saturated_ratio*100:.1f}% saturated)."
    else:
        classification = GlareClassification.NORMAL
        severity = float(np.clip(saturated_ratio * 3.0, 0.0, 0.15))
        details = f"Exposure within balanced dynamic limits; zero disabling glare detected."

    return SunGlareEvidence(
        classification=classification,
        severity=severity,
        saturated_pixel_ratio=saturated_ratio,
        high_intensity_cluster_ratio=cluster_ratio,
        contrast_washout=washout,
        details=details,
    )


def analyze_fog(frame_bgr: np.ndarray, gray_img: np.ndarray) -> FogHazeEvidence:
    """Analyze atmospheric fog and haze using Dark Channel Prior principles.

    In foggy scenes, atmospheric veil (airlight) scatters light, elevating the
    minimum intensity across RGB channels while attenuating edge gradients and
    reducing Michelson contrast.
    """
    h, w = gray_img.shape

    # 1. Dark Channel Prior approximation (minimum across color channels in small patches)
    # Downsample slightly for real-time CPU efficiency
    small_bgr = cv2.resize(frame_bgr, (min(w, 320), min(h, 180)), interpolation=cv2.INTER_AREA)
    min_channel = np.min(small_bgr, axis=2)
    # Morphological min filter (erosion) with 7x7 patch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dark_channel = cv2.erode(min_channel, kernel)

    # Airlight intensity: mean of top 10% brightest pixels in dark channel (predominantly sky / horizon)
    sh, sw = dark_channel.shape
    horizon_dark = dark_channel[: int(sh * 0.65), :]
    airlight_val = float(np.percentile(horizon_dark, 90))

    # 2. Contrast loss
    contrast_std = float(np.std(gray_img))
    contrast_loss = float(np.clip(1.0 - (contrast_std / 58.0), 0.0, 1.0))

    # 3. Depth visibility ratio: edge attenuation in horizon vs foreground
    edges = cv2.Canny(gray_img, 40, 120)
    upper_edges = np.count_nonzero(edges[: h // 2, :])
    lower_edges = np.count_nonzero(edges[h // 2 :, :])
    edge_ratio = float(upper_edges / max(1, lower_edges))

    # Calibrated classification
    if airlight_val > 125.0 and contrast_loss > 0.55:
        classification = FogClassification.DENSE_FOG
        severity = float(np.clip(0.75 + (contrast_loss - 0.55) * 0.5, 0.75, 1.0))
        details = f"Dense fog: severe atmospheric attenuation masking horizon and road obstacles (Airlight {airlight_val:.1f}, Contrast loss {contrast_loss:.2f})."
    elif airlight_val > 85.0 and contrast_loss > 0.38:
        classification = FogClassification.MODERATE_FOG
        severity = float(np.clip(0.45 + (contrast_loss - 0.38) * 1.5, 0.45, 0.75))
        details = f"Moderate fog / atmospheric haze reducing visibility of distant targets (Airlight {airlight_val:.1f}, Contrast loss {contrast_loss:.2f})."
    elif airlight_val > 60.0 and contrast_loss > 0.25:
        classification = FogClassification.LIGHT_HAZE
        severity = float(np.clip(0.15 + (contrast_loss - 0.25) * 2.0, 0.15, 0.45))
        details = f"Light atmospheric haze with mild contrast reduction (Airlight {airlight_val:.1f})."
    else:
        classification = FogClassification.CLEAR
        severity = float(np.clip(contrast_loss * 0.2, 0.0, 0.15))
        details = f"Atmosphere clear; high dynamic range and visible distant landmarks."

    return FogHazeEvidence(
        classification=classification,
        severity=severity,
        airlight_intensity=airlight_val,
        contrast_loss=contrast_loss,
        depth_visibility_ratio=edge_ratio,
        details=details,
    )


def analyze_rain(
    frame_bgr: np.ndarray,
    gray_img: np.ndarray,
    prev_gray: Optional[np.ndarray] = None,
) -> RainEvidence:
    """Analyze rain streaks, water droplets, and precipitation degradation.

    Measures:
    1. High-frequency vertical gradient response (Sobel Y vs Sobel X streak ratio)
    2. Localized circular droplet blur / distortion patterns
    3. Temporal inter-frame variance when consecutive frames are available
    """
    h, w = gray_img.shape
    total_pixels = float(h * w)

    # 1. Vertical streak energy: directional Sobel filtering
    # Rain creates strong vertical edge energy relative to horizontal
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    y_var = float(np.var(sobel_y))
    x_var = float(np.var(sobel_x))
    streak_variance = float(max(0.0, y_var - x_var))

    # 2. Droplet patch detection: high variance in local circular kernels
    # Water droplets on lens distort light into sharp micro-refractions
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    droplet_mask = (np.abs(lap) > 85.0).astype(np.uint8)
    droplet_density = float(np.count_nonzero(droplet_mask) / total_pixels)

    # 3. Contrast attenuation check
    contrast = float(np.std(gray_img))
    contrast_attenuation = float(np.clip(1.0 - (contrast / 55.0), 0.0, 1.0))

    # Inter-frame temporal flicker if previous frame provided
    temp_diff = 0.0
    if prev_gray is not None and prev_gray.shape == gray_img.shape:
        diff = cv2.absdiff(gray_img, prev_gray)
        temp_diff = float(np.mean(diff))

    # Calibrated classification
    if (streak_variance > 500.0 and droplet_density > 0.04) or (streak_variance > 750.0):
        classification = RainClassification.HEAVY_RAIN
        severity = float(np.clip(0.75 + (streak_variance / 2000.0) * 0.25, 0.75, 1.0))
        details = f"Heavy precipitation: dense rain streaks and water droplets obscuring sensor (Streak variance {streak_variance:.1f}, Droplet density {droplet_density*100:.2f}%)."
    elif (streak_variance > 320.0 and droplet_density > 0.02) or (streak_variance > 450.0):
        classification = RainClassification.MODERATE_RAIN
        severity = float(np.clip(0.45 + (streak_variance / 1000.0) * 0.30, 0.45, 0.75))
        details = f"Moderate rain detected with noticeable road splash and lens droplet refraction (Streak variance {streak_variance:.1f})."
    elif streak_variance > 180.0 or droplet_density > 0.015:
        classification = RainClassification.LIGHT_RAIN
        severity = float(np.clip(0.15 + (streak_variance / 500.0) * 0.30, 0.15, 0.45))
        details = f"Possible light rain or intermittent windshield mist (Streak variance {streak_variance:.1f})."
    elif contrast_attenuation > 0.4 and y_var > 350.0:
        classification = RainClassification.UNCERTAIN
        severity = 0.25
        details = f"Uncertain precipitation indicators: wet surface reflections without defined airborne streaks."
    else:
        classification = RainClassification.NO_RAIN
        severity = 0.05
        details = f"No visible rain evidence or windshield water droplet interference."

    return RainEvidence(
        classification=classification,
        severity=severity,
        droplet_density=droplet_density,
        streak_variance=streak_variance,
        contrast_attenuation=contrast_attenuation,
        details=details,
    )
