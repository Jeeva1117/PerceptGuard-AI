"""PerceptGuard AI - Early Warning & Vehicle Stop Recommendation (Feature 5).

Provides decision-support early warnings, vehicle stop recommendations, supporting evidence
chains, and priority arbitration for automotive safety intelligence.
"""

from src.safety_recommendation.early_warning_engine import EarlyWarningEngine, EarlyWarningSignal
from src.safety_recommendation.pedestrian_awareness_engine import (
    PedestrianAwarenessEngine,
    PedestrianAwarenessState,
    PedestrianObjectDetail,
    FramePedestrianTelemetry,
)
from src.safety_recommendation.recommendation_rules import RecommendationRulesEngine
from src.safety_recommendation.safety_engine import SafetyRecommendationEngine
from src.safety_recommendation.schemas import (
    InvolvedObject,
    RecommendationState,
    SafetyRecommendation,
    SafetyStreamSummary,
    SupportingEvidence,
    WarningPriority,
    WarningStatus,
)
from src.safety_recommendation.warning_history_manager import WarningHistoryManager
from src.safety_recommendation.warning_priority_manager import WarningPriorityManager

__all__ = [
    "SafetyRecommendationEngine",
    "RecommendationRulesEngine",
    "EarlyWarningEngine",
    "EarlyWarningSignal",
    "PedestrianAwarenessEngine",
    "PedestrianAwarenessState",
    "PedestrianObjectDetail",
    "FramePedestrianTelemetry",
    "WarningPriorityManager",
    "WarningHistoryManager",
    "RecommendationState",
    "WarningPriority",
    "WarningStatus",
    "InvolvedObject",
    "SupportingEvidence",
    "SafetyRecommendation",
    "SafetyStreamSummary",
]
