"""PerceptGuard AI: Perception Metrics & Evaluation Package.

Provides production-grade metric evaluation engines, statistical aggregators,
confidence profiling, temporal stability analysis, tracking continuity evaluation,
and ground-truth validation for autonomous-driving computer vision pipelines.
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
from src.analysis.tracking_metrics import (
    ClassTrackingMetrics,
    FrameTrackingMetrics,
    GroundTruthEvaluation,
    InvalidTrackingDataError,
    TrackObservationMetrics,
    TrackingAnalysisSummary,
    TrackingMetricsAnalyzer,
    TrackingMetricsError,
    TrackingMetricsPersistenceError,
    UnsupportedTrackingSchemaError,
)

__all__ = [
    # Detection Metrics
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
    # Tracking Metrics
    "TrackingMetricsAnalyzer",
    "TrackingAnalysisSummary",
    "TrackObservationMetrics",
    "FrameTrackingMetrics",
    "ClassTrackingMetrics",
    "GroundTruthEvaluation",
    "TrackingMetricsError",
    "InvalidTrackingDataError",
    "UnsupportedTrackingSchemaError",
    "TrackingMetricsPersistenceError",
]
