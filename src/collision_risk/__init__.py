"""PerceptGuard AI: Collision & Road Risk Assessment Package (Feature 4).

Public interfaces, schemas, and engines:
- CollisionRiskEngine
- DistanceAnalyzer
- RelativeMotionAnalyzer
- TTCCalculator
- TrajectoryConflictAnalyzer
- UncertaintyAnalyzer
- CollisionRiskClassifier
- TemporalRiskTracker
- Schemas & Enums
"""

from src.collision_risk.schemas import (
    AssessmentStatus,
    CollisionRiskAssessment,
    CollisionRiskState,
    CollisionStreamSummary,
    ConflictType,
    TrajectoryConflict,
    UncertaintyLevel,
)
from src.collision_risk.distance_analyzer import DistanceAnalyzer
from src.collision_risk.relative_motion_analyzer import RelativeMotionAnalyzer
from src.collision_risk.ttc_calculator import TTCCalculator
from src.collision_risk.trajectory_conflict_analyzer import TrajectoryConflictAnalyzer
from src.collision_risk.uncertainty_analyzer import UncertaintyAnalyzer
from src.collision_risk.risk_classifier import CollisionRiskClassifier
from src.collision_risk.temporal_risk_tracker import TemporalRiskTracker
from src.collision_risk.collision_risk_engine import CollisionRiskEngine
from src.collision_risk.road_context_analyzer import (
    RoadContextAnalyzer,
    RoadContextData,
    LaneState,
    PedestrianRoadPosition,
    PedestrianMovementIntent,
)

__all__ = [
    "AssessmentStatus",
    "CollisionRiskAssessment",
    "CollisionRiskState",
    "CollisionStreamSummary",
    "ConflictType",
    "TrajectoryConflict",
    "UncertaintyLevel",
    "DistanceAnalyzer",
    "RelativeMotionAnalyzer",
    "TTCCalculator",
    "TrajectoryConflictAnalyzer",
    "UncertaintyAnalyzer",
    "CollisionRiskClassifier",
    "TemporalRiskTracker",
    "CollisionRiskEngine",
    "RoadContextAnalyzer",
    "RoadContextData",
    "LaneState",
    "PedestrianRoadPosition",
    "PedestrianMovementIntent",
]
