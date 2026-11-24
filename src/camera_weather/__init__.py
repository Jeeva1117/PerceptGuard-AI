"""PerceptGuard-AI: Camera & Weather Intelligence Package (Feature 1).

Public interfaces, schemas, and engines:
- VisibilityQualityEngine
- TemporalWeatherTracker
- VisibilityForecaster
- estimate_perception_impact
- Schemas & Classifications
"""

from src.camera_weather.schemas import (
    BlurClassification,
    CameraBlurEvidence,
    CameraQualityMetrics,
    DegradationTrend,
    FogClassification,
    FogHazeEvidence,
    FrameWeatherIntelligence,
    GlareClassification,
    LightingClassification,
    LowLightEvidence,
    PerceptionImpactIndicator,
    RainClassification,
    RainEvidence,
    RiskImpactLevel,
    SunGlareEvidence,
    VisibilityForecast,
    VisibilityLevel,
    WeatherStreamSummary,
)
from src.camera_weather.optical_analyzers import (
    analyze_blur,
    analyze_fog,
    analyze_glare,
    analyze_lighting,
    analyze_rain,
)
from src.camera_weather.visibility_engine import VisibilityQualityEngine
from src.camera_weather.temporal_tracker import TemporalDegradationType, TemporalWeatherTracker
from src.camera_weather.visibility_forecaster import VisibilityForecaster
from src.camera_weather.perception_impact import estimate_perception_impact

__all__ = [
    "BlurClassification",
    "CameraBlurEvidence",
    "CameraQualityMetrics",
    "DegradationTrend",
    "FogClassification",
    "FogHazeEvidence",
    "FrameWeatherIntelligence",
    "GlareClassification",
    "LightingClassification",
    "LowLightEvidence",
    "PerceptionImpactIndicator",
    "RainClassification",
    "RainEvidence",
    "RiskImpactLevel",
    "SunGlareEvidence",
    "VisibilityForecast",
    "VisibilityLevel",
    "WeatherStreamSummary",
    "analyze_blur",
    "analyze_fog",
    "analyze_glare",
    "analyze_lighting",
    "analyze_rain",
    "VisibilityQualityEngine",
    "TemporalDegradationType",
    "TemporalWeatherTracker",
    "VisibilityForecaster",
    "estimate_perception_impact",
]
