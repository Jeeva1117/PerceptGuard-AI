"""PerceptGuard-AI: Evaluation and Ground-Truth Ingestion Package.

Generates ground-truth reliability targets using bounding box Intersection over
Union (IoU) with annotated ground-truth labels, provides BDD100K ground-truth
annotation adapters, and evaluates model predictions using regression (MSE, MAE)
and classification reliability degradation metrics.
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
]
