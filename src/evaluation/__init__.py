"""Evaluation & Reliability Ground-Truth Labeling Module.

Generates ground-truth reliability targets using bounding box Intersection over
Union (IoU) with annotated ground-truth labels and evaluates model predictions
using regression (MSE, MAE) and classification reliability degradation metrics
(Precision, Recall, ROC-AUC, Reliability Drop Early Warning F1).
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

__all__ = [
    "ForecastingModelEvaluator",
    "ModelEvaluationError",
    "GroundTruthObservation",
    "AlignedEvaluationPair",
    "HorizonEvaluationMetrics",
    "ForecastingEvaluationVisualizer",
    "VisualizationError",
]


