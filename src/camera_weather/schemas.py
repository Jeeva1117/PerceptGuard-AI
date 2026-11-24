"""Data schemas and interfaces for Camera & Weather Intelligence (Feature 1).

Defines structured representations for:
- Rain detection evidence & severity
- Fog/haze atmospheric scattering analysis
- Defocus and motion blur measurements
- Low-light and darkness classifications
- Sun glare and sensor overexposure
- Unified visibility score and camera quality
- Temporal degradation tracking and transitions
- Multi-horizon visibility forecasting
- Object detection reliability impact interface
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RainClassification(str, Enum):
    NO_RAIN = "No visible rain evidence"
    LIGHT_RAIN = "Possible light rain"
    MODERATE_RAIN = "Moderate rain"
    HEAVY_RAIN = "Heavy rain"
    UNCERTAIN = "Uncertain rain condition"


class FogClassification(str, Enum):
    CLEAR = "Clear"
    LIGHT_HAZE = "Light haze"
    MODERATE_FOG = "Moderate fog/haze"
    DENSE_FOG = "Dense fog"
    UNKNOWN = "Unknown"


class BlurClassification(str, Enum):
    SHARP = "Sharp"
    SLIGHT_BLUR = "Slight blur"
    MODERATE_BLUR = "Moderate blur"
    SEVERE_BLUR = "Severe blur"
    UNKNOWN = "Unknown"


class LightingClassification(str, Enum):
    NORMAL = "Normal lighting"
    LOW_LIGHT = "Low light"
    VERY_LOW_LIGHT = "Very low light"
    CRITICAL_DARKNESS = "Critical darkness"


class GlareClassification(str, Enum):
    NORMAL = "Normal exposure"
    ELEVATED_GLARE = "Elevated glare"
    SEVERE_GLARE = "Severe glare"
    OVEREXPOSURE = "Overexposure"
    UNKNOWN = "Unknown"


class VisibilityLevel(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


class DegradationTrend(str, Enum):
    IMPROVING = "Improving"
    STABLE = "Stable"
    GRADUALLY_DEGRADING = "Gradually degrading"
    RAPIDLY_DEGRADING = "Rapidly degrading"
    UNCERTAIN = "Uncertain"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RiskImpactLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RainEvidence:
    """Quantitative evidence and classification for precipitation."""
    classification: RainClassification
    severity: float  # 0.0 (none) to 1.0 (extreme)
    droplet_density: float  # circular blob / droplet indicator ratio
    streak_variance: float  # high-frequency vertical gradient variance
    contrast_attenuation: float  # local contrast reduction
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification.value,
            "severity": round(self.severity, 3),
            "droplet_density": round(self.droplet_density, 4),
            "streak_variance": round(self.streak_variance, 2),
            "contrast_attenuation": round(self.contrast_attenuation, 3),
            "details": self.details,
        }


@dataclass
class FogHazeEvidence:
    """Quantitative evidence and classification for atmospheric scattering/fog."""
    classification: FogClassification
    severity: float  # 0.0 (clear) to 1.0 (dense fog)
    airlight_intensity: float  # dark channel prior / minimum intensity baseline
    contrast_loss: float  # scene dynamic contrast loss
    depth_visibility_ratio: float  # high-frequency edge attenuation in upper regions
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification.value,
            "severity": round(self.severity, 3),
            "airlight_intensity": round(self.airlight_intensity, 2),
            "contrast_loss": round(self.contrast_loss, 3),
            "depth_visibility_ratio": round(self.depth_visibility_ratio, 3),
            "details": self.details,
        }


@dataclass
class CameraBlurEvidence:
    """Quantitative optical sharpness and blur measurements."""
    classification: BlurClassification
    severity: float  # 0.0 (sharp) to 1.0 (severe blur)
    laplacian_variance: float  # focus measure
    tenengrad_gradient_mean: float  # Sobel gradient energy
    edge_density: float  # Canny edge fraction
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification.value,
            "severity": round(self.severity, 3),
            "laplacian_variance": round(self.laplacian_variance, 2),
            "tenengrad_gradient_mean": round(self.tenengrad_gradient_mean, 2),
            "edge_density": round(self.edge_density, 4),
            "details": self.details,
        }


@dataclass
class LowLightEvidence:
    """Quantitative luminance and darkness analysis."""
    classification: LightingClassification
    severity: float  # 0.0 (normal) to 1.0 (pitch black)
    mean_luminance: float  # 0 to 255
    underexposed_pixel_ratio: float  # fraction of pixels < 30
    dynamic_range_spread: float  # 95th - 5th percentile spread
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification.value,
            "severity": round(self.severity, 3),
            "mean_luminance": round(self.mean_luminance, 2),
            "underexposed_pixel_ratio": round(self.underexposed_pixel_ratio, 4),
            "dynamic_range_spread": round(self.dynamic_range_spread, 2),
            "details": self.details,
        }


@dataclass
class SunGlareEvidence:
    """Quantitative optical glare and sensor overexposure analysis."""
    classification: GlareClassification
    severity: float  # 0.0 (normal) to 1.0 (blinding)
    saturated_pixel_ratio: float  # fraction of pixels > 245
    high_intensity_cluster_ratio: float  # large connected overexposed areas
    contrast_washout: float  # glare-induced luminance spread loss
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification.value,
            "severity": round(self.severity, 3),
            "saturated_pixel_ratio": round(self.saturated_pixel_ratio, 4),
            "high_intensity_cluster_ratio": round(self.high_intensity_cluster_ratio, 4),
            "contrast_washout": round(self.contrast_washout, 3),
            "details": self.details,
        }


@dataclass
class CameraQualityMetrics:
    """Composite optical camera performance indices (0 to 100)."""
    sharpness: float
    contrast: float
    luminance: float
    exposure: float
    field_uniformity: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "sharpness": round(self.sharpness, 1),
            "contrast": round(self.contrast, 1),
            "luminance": round(self.luminance, 1),
            "exposure": round(self.exposure, 1),
            "field_uniformity": round(self.field_uniformity, 1),
        }


@dataclass
class PerceptionImpactIndicator:
    """Downstream interface estimating optical condition impact on object detection."""
    expected_confidence_penalty: float  # Estimated drop in detection conf (e.g., 0.05 - 0.35)
    boundary_degradation_risk: RiskImpactLevel  # Bounding box jitter / border loss risk
    small_object_risk: RiskImpactLevel  # Risk of missing distant / small objects
    suggested_detection_sensitivity: str  # NORMAL, HEIGHTENED, or CONSERVATIVE
    rationale: str
    is_estimated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_confidence_penalty": round(self.expected_confidence_penalty, 3),
            "boundary_degradation_risk": self.boundary_degradation_risk.value,
            "small_object_risk": self.small_object_risk.value,
            "suggested_detection_sensitivity": self.suggested_detection_sensitivity,
            "rationale": self.rationale,
            "is_estimated": self.is_estimated,
        }


@dataclass
class VisibilityForecast:
    """Multi-horizon near-future visibility degradation forecast."""
    degradation_trend: DegradationTrend
    trend_velocity: float  # change in visibility points per second
    horizons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confidence: float = 0.85
    uncertainty: float = 0.15
    forecast_method: str = "Holt Linear Trend & Rolling Baseline"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "degradation_trend": self.degradation_trend.value,
            "trend_velocity": round(self.trend_velocity, 2),
            "horizons": self.horizons,
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "forecast_method": self.forecast_method,
        }


@dataclass
class FrameWeatherIntelligence:
    """Complete structured intelligence record for a single video frame."""
    frame_index: int
    timestamp_seconds: float
    visibility_score: float  # 0.0 to 100.0 (higher = clearer)
    visibility_level: VisibilityLevel
    camera_quality_score: float  # 0.0 to 100.0
    primary_condition: str
    condition_labels: Dict[str, str]
    condition_severity: Dict[str, float]
    rain: RainEvidence
    fog: FogHazeEvidence
    blur: CameraBlurEvidence
    lighting: LowLightEvidence
    glare: SunGlareEvidence
    quality_metrics: CameraQualityMetrics
    perception_impact: PerceptionImpactIndicator
    forecast: VisibilityForecast
    confidence: float
    uncertainty: float
    degradation_reasons: List[str]
    is_degraded: bool

    def to_dict(self) -> Dict[str, Any]:
        vis_stat = "CLEAR" if self.visibility_score >= 75.0 else ("MODERATE" if self.visibility_score >= 45.0 else "LOW")
        img_qual = "GOOD" if self.camera_quality_score >= 70.0 else ("DEGRADED" if self.camera_quality_score >= 40.0 else "POOR")
        weath_stat = "CLEAR" if self.primary_condition == "CLEAR" else (
            "RAIN" if "RAIN" in self.primary_condition else (
                "FOG" if "FOG" in self.primary_condition else (
                    "HAZE" if "HAZE" in self.primary_condition else "UNKNOWN"
                )
            )
        )
        degr_stat = "NONE DETECTED" if not self.is_degraded and self.primary_condition == "CLEAR" else self.primary_condition

        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "visibility_score": round(self.visibility_score, 1),
            "visibility_level": self.visibility_level.value,
            "visibility_status": vis_stat,
            "weather_status": weath_stat,
            "image_quality_status": img_qual,
            "degradation_status": degr_stat,
            "camera_quality_score": round(self.camera_quality_score, 1),
            "primary_condition": self.primary_condition,
            "condition": self.primary_condition,
            "is_degraded": self.is_degraded,
            "condition_labels": self.condition_labels,
            "condition_severity": {k: round(v, 3) for k, v in self.condition_severity.items()},
            "rain": self.rain.to_dict(),
            "fog": self.fog.to_dict(),
            "blur": self.blur.to_dict(),
            "lighting": self.lighting.to_dict(),
            "glare": self.glare.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict(),
            "perception_impact": self.perception_impact.to_dict(),
            "forecast": self.forecast.to_dict(),
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "degradation_reasons": self.degradation_reasons,
        }


@dataclass
class WeatherStreamSummary:
    """High-level summary of camera & weather conditions across an entire video stream."""
    total_frames_analyzed: int
    mean_visibility_score: float
    min_visibility_score: float
    predominant_condition: str
    condition_frequency: Dict[str, int]
    degraded_frames_count: int
    degraded_frames_percentage: float
    trend_summary: str
    average_camera_quality: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_frames_analyzed": self.total_frames_analyzed,
            "mean_visibility_score": round(self.mean_visibility_score, 1),
            "min_visibility_score": round(self.min_visibility_score, 1),
            "predominant_condition": self.predominant_condition,
            "condition_frequency": self.condition_frequency,
            "degraded_frames_count": self.degraded_frames_count,
            "degraded_frames_percentage": round(self.degraded_frames_percentage, 1),
            "trend_summary": self.trend_summary,
            "average_camera_quality": round(self.average_camera_quality, 1),
        }
