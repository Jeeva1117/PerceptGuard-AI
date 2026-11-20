"""PerceptGuard-AI: Evaluation and Ground-Truth Ingestion Package.

Generates ground-truth reliability targets using bounding box Intersection over
Union (IoU) with annotated ground-truth labels, provides BDD100K ground-truth
annotation adapters, defines canonical reliability labeling schemas, and evaluates
model predictions using regression (MSE, MAE) and classification reliability
degradation metrics.
"""

from src.forecasting.model_evaluator import (
    AlignedEvaluationPair,
    ForecastingModelEvaluator,
    GroundTruthObservation,
    HorizonEvaluationMetrics,
    ModelEvaluationError,
)
from src.forecasting.evaluation_visualizer import (
    ForecastingEvaluationVisualizer,
    VisualizationError,
)
from src.evaluation.bdd100k_adapter import (
    BDD100KAdapter,
    BDD100KAdapterError,
    AnnotationFileNotFoundError,
    AnnotationSchemaError,
    InvalidBoundingBoxError,
    CoordinateConversionError,
    FrameAlignmentError,
    BoundingBox,
    GroundTruthObject,
    FrameGroundTruth,
    SequenceGroundTruth,
    CoordinateTransformer,
    DEFAULT_BDD_TO_COCO_MAPPING,
)
from src.evaluation.reliability_schema import (
    CanonicalBoundingBox,
    MatchingOutcome,
    TargetStatus,
    ObjectReliabilityRecord,
    FrameReliabilityRecord,
    HorizonTargetSpec,
    TemporalTargetRecord,
    ReliabilityDataContract,
    calculate_iou,
    compute_frame_f1,
    compute_composite_reliability,
    SCHEMA_VERSION as RELIABILITY_SCHEMA_VERSION,
)
from src.evaluation.annotation_processor import (
    ANNOTATION_SCHEMA_VERSION,
    ValidationStatus,
    FrameStatus,
    AnnotationProcessingError,
    AnnotationParseError,
    CoordinateValidationError,
    FrameAlignmentException,
    CanonicalAnnotationRecord,
    FrameAnnotationSummary,
    ProcessingSummary,
    DEFAULT_CLASS_MAPPING,
    ClassMapper,
    CoordinateNormalizer,
    BDD100KMOTParser,
    AnnotationProcessor,
)

__all__ = [
    # Model evaluation & visualization
    "ForecastingModelEvaluator",
    "ModelEvaluationError",
    "GroundTruthObservation",
    "AlignedEvaluationPair",
    "HorizonEvaluationMetrics",
    "ForecastingEvaluationVisualizer",
    "VisualizationError",
    # BDD100K annotation ingestion & transformation
    "BDD100KAdapter",
    "BDD100KAdapterError",
    "AnnotationFileNotFoundError",
    "AnnotationSchemaError",
    "InvalidBoundingBoxError",
    "CoordinateConversionError",
    "FrameAlignmentError",
    "BoundingBox",
    "GroundTruthObject",
    "FrameGroundTruth",
    "SequenceGroundTruth",
    "CoordinateTransformer",
    "DEFAULT_BDD_TO_COCO_MAPPING",
    # Canonical reliability schemas & contracts
    "CanonicalBoundingBox",
    "MatchingOutcome",
    "TargetStatus",
    "ObjectReliabilityRecord",
    "FrameReliabilityRecord",
    "HorizonTargetSpec",
    "TemporalTargetRecord",
    "ReliabilityDataContract",
    "calculate_iou",
    "compute_frame_f1",
    "compute_composite_reliability",
    "RELIABILITY_SCHEMA_VERSION",
    # Ground-truth annotation processor
    "ANNOTATION_SCHEMA_VERSION",
    "ValidationStatus",
    "FrameStatus",
    "AnnotationProcessingError",
    "AnnotationParseError",
    "CoordinateValidationError",
    "FrameAlignmentException",
    "CanonicalAnnotationRecord",
    "FrameAnnotationSummary",
    "ProcessingSummary",
    "DEFAULT_CLASS_MAPPING",
    "ClassMapper",
    "CoordinateNormalizer",
    "BDD100KMOTParser",
    "AnnotationProcessor",
]
