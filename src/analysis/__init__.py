"""PerceptGuard AI: Detection Metrics & Perception Analysis Package.

Provides production-grade metric evaluation engines, statistical aggregators,
confidence profiling, temporal stability analysis, and ground-truth validation
for autonomous-driving computer vision pipelines.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

from src.analysis.detection_metrics import (
    ClassMetricRecord,
    DetectionMetrics,
    DetectionMetricsError,
    FrameMetricRecord,
    GroundTruthEvaluationError,
    GroundTruthEvaluationResult,
    InvalidDetectionDataError,
    MetricsAnalysisSummary,
    MetricsPersistenceError,
    TemporalConsistencyMetrics,
    UnsupportedSchemaError,
)

__all__ = [
    "DetectionMetrics",
    "MetricsAnalysisSummary",
    "ClassMetricRecord",
    "FrameMetricRecord",
    "TemporalConsistencyMetrics",
    "GroundTruthEvaluationResult",
    "DetectionMetricsError",
    "InvalidDetectionDataError",
    "UnsupportedSchemaError",
    "GroundTruthEvaluationError",
    "MetricsPersistenceError",
]

