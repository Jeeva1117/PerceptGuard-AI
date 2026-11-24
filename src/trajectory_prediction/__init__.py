"""PerceptGuard-AI: Object & Trajectory Prediction Package (Feature 3).

Public interfaces, schemas, and engines:
- MultimodalTrajectoryForecaster
- KalmanTrajectoryPredictor
- KinematicsEngine
- MultiAgentInteractionModel
- Schemas & Enums
"""

from src.trajectory_prediction.schemas import (
    CoordinateFrame,
    InteractionType,
    MotionState,
    MultiAgentInteraction,
    ObjectTrajectoryPrediction,
    PredictionMode,
    PredictionStatus,
    TrajectoryHypothesis,
    TrajectoryStreamSummary,
    TrajectoryWaypoint,
)
from src.trajectory_prediction.kinematics_engine import KinematicsEngine
from src.trajectory_prediction.kalman_trajectory_predictor import KalmanTrajectoryPredictor
from src.trajectory_prediction.interaction_model import MultiAgentInteractionModel
from src.trajectory_prediction.multimodal_forecaster import MultimodalTrajectoryForecaster

__all__ = [
    "CoordinateFrame",
    "InteractionType",
    "MotionState",
    "MultiAgentInteraction",
    "ObjectTrajectoryPrediction",
    "PredictionMode",
    "PredictionStatus",
    "TrajectoryHypothesis",
    "TrajectoryStreamSummary",
    "TrajectoryWaypoint",
    "KinematicsEngine",
    "KalmanTrajectoryPredictor",
    "MultiAgentInteractionModel",
    "MultimodalTrajectoryForecaster",
]
