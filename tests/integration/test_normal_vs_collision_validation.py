"""Integration test suite validating Evidence-Driven Safety Decisions.

Proves:
1. Normal driving conditions (clean road, moving cars maintaining lane/headway,
   parked roadside cars) produce NORMAL status, NO COLLISION RISK DETECTED,
   RISK LEVEL: LOW, and NO ACTIVE ALERTS.
2. Genuine collision hazards (cross-cutting path conflicts, low TTC <= 2.5s)
   produce COLLISION RISK DETECTED, HIGH/CRITICAL risk, and STOP RECOMMENDED.
3. Degraded / excessive uncertainty conditions produce UNKNOWN (never false LOW or HIGH).
"""

import pytest
import numpy as np

from src.collision_risk.risk_classifier import CollisionRiskClassifier
from src.collision_risk.schemas import (
    CollisionRiskState,
    ConflictType,
    TrajectoryConflict,
    UncertaintyLevel,
)
from src.safety_recommendation.recommendation_rules import RecommendationRulesEngine
from src.safety_recommendation.schemas import RecommendationState, WarningPriority


class TestNormalVsCollisionValidation:
    """Test suite proving normal driving does NOT trigger false alarms, and hazards trigger true alarms."""

    @pytest.fixture
    def risk_classifier(self) -> CollisionRiskClassifier:
        return CollisionRiskClassifier()

    @pytest.fixture
    def rules_engine(self) -> RecommendationRulesEngine:
        return RecommendationRulesEngine()

    def test_normal_driving_with_lead_vehicle_produces_normal_and_low_risk(
        self, risk_classifier: CollisionRiskClassifier, rules_engine: RecommendationRulesEngine
    ):
        """A lead car in front driving at same speed maintaining safe headway (TTC=None or > 8s).

        Must output:
        - Collision Risk: LOW
        - Safety Recommendation: NORMAL
        - Priority: INFO
        - NO Active Warning
        """
        # Feature 4 Classification
        risk_state, score, factors, explanation, status = risk_classifier.classify_risk(
            min_ttc=None,
            conflicts=[],
            closest_dist_px=140.0,
            closest_dist_m=25.0,
            closest_tid=1,
            uncertainty_score=0.12,
            uncertainty_level=UncertaintyLevel.LOW,
            is_unreliable=False,
            uncertainty_factors=[],
            num_tracks=1,
        )
        assert risk_state == CollisionRiskState.LOW

        # Feature 5 Safety Evaluation
        collision_data = {
            "risk_state": "LOW",
            "fused_risk_score": 0.08,
            "min_ttc_seconds": None,
            "closest_distance_m": 25.0,
            "closest_track_id": 1,
            "conflicting_pairs": [],
            "timestamp_seconds": 1.0,
            "uncertainty_score": 0.12,
        }
        weather_intel = {
            "visibility_score": 92.0,
            "condition": "CLEAR",
            "is_degraded": False,
        }
        tracks = [{
            "track_id": 1,
            "class_name": "car",
            "confidence": 0.92,
            "bbox": [550.0, 380.0, 680.0, 480.0],
        }]
        trajectories = [{
            "track_id": 1,
            "speed_kmh": 45.0,
            "motion_state": "CRUISING",
        }]

        rec = rules_engine.evaluate(
            collision_data=collision_data,
            weather_intel=weather_intel,
            trajectories=trajectories,
            active_tracks=tracks,
            timestamp_seconds=1.0,
        )

        assert rec.recommendation_state == RecommendationState.NORMAL
        assert rec.priority == WarningPriority.INFO
        assert rec.risk_level == "LOW"
        assert "normal" in rec.primary_reason.lower()

    def test_parked_roadside_car_does_not_trigger_false_caution_or_stop(
        self, risk_classifier: CollisionRiskClassifier, rules_engine: RecommendationRulesEngine
    ):
        """A parked car on the roadside (stationary, zero closing velocity, lateral offset).

        Must output:
        - Collision Risk: LOW
        - Safety Recommendation: NORMAL (NOT Caution!)
        """
        # Feature 4: Parked car has relative speed ~ 0 or opening distance, zero trajectory conflict
        risk_state, score, factors, explanation, status = risk_classifier.classify_risk(
            min_ttc=None,
            conflicts=[],
            closest_dist_px=110.0,
            closest_dist_m=12.0,
            closest_tid=5,
            uncertainty_score=0.15,
            uncertainty_level=UncertaintyLevel.LOW,
            is_unreliable=False,
            uncertainty_factors=[],
            num_tracks=1,
        )
        assert risk_state == CollisionRiskState.LOW

        collision_data = {
            "risk_state": "LOW",
            "fused_risk_score": 0.10,
            "min_ttc_seconds": None,
            "closest_distance_m": 12.0,
            "closest_track_id": 5,
            "conflicting_pairs": [],
            "timestamp_seconds": 2.0,
            "uncertainty_score": 0.15,
        }
        weather_intel = {
            "visibility_score": 88.0,
            "condition": "CLEAR",
            "is_degraded": False,
        }
        tracks = [{
            "track_id": 5,
            "class_name": "car",
            "confidence": 0.89,
            "bbox": [180.0, 450.0, 320.0, 560.0],
        }]

        rec = rules_engine.evaluate(
            collision_data=collision_data,
            weather_intel=weather_intel,
            trajectories=[],
            active_tracks=tracks,
            timestamp_seconds=2.0,
        )

        assert rec.recommendation_state == RecommendationState.NORMAL
        assert rec.priority == WarningPriority.INFO

    def test_cross_cutting_motorcycle_triggers_genuine_collision_and_stop(
        self, risk_classifier: CollisionRiskClassifier, rules_engine: RecommendationRulesEngine
    ):
        """A motorcycle cutting laterally across vehicle path with closing speed and low TTC (1.4s).

        Must output:
        - Collision Risk: HIGH or CRITICAL
        - Safety Recommendation: STOP_RECOMMENDED
        - Priority: CRITICAL
        """
        conflict_obj = TrajectoryConflict(
            track_a=12,
            track_b=0,
            class_a="motorcycle",
            class_b="ego",
            conflict_type=ConflictType.CROSSING,
            time_to_conflict_seconds=1.4,
            horizon_label="H1",
            separation_at_conflict_px=18.0,
            separation_at_conflict_m=4.5,
            conflict_point_px=(600.0, 500.0),
            combined_uncertainty_px=5.0,
            is_spatiotemporal_overlap=True,
        )
        risk_state, score, factors, explanation, status = risk_classifier.classify_risk(
            min_ttc=1.4,
            conflicts=[conflict_obj],
            closest_dist_px=22.0,
            closest_dist_m=6.5,
            closest_tid=12,
            uncertainty_score=0.18,
            uncertainty_level=UncertaintyLevel.LOW,
            is_unreliable=False,
            uncertainty_factors=[],
            num_tracks=1,
        )
        assert risk_state in (CollisionRiskState.HIGH, CollisionRiskState.CRITICAL)

        conflicts_dict = [{
            "track_a": 12,
            "track_b": 0,
            "class_a": "motorcycle",
            "class_b": "ego",
            "horizon": "h1",
            "separation_pixels": 18.0,
        }]
        collision_data = {
            "risk_state": risk_state.value,
            "fused_risk_score": 0.88,
            "min_ttc_seconds": 1.4,
            "closest_distance_m": 6.5,
            "closest_track_id": 12,
            "conflicting_pairs": conflicts_dict,
            "timestamp_seconds": 3.0,
            "uncertainty_score": 0.18,
        }
        weather_intel = {
            "visibility_score": 85.0,
            "condition": "CLEAR",
            "is_degraded": False,
        }
        tracks = [{
            "track_id": 12,
            "class_name": "motorcycle",
            "confidence": 0.94,
            "bbox": [580.0, 420.0, 660.0, 520.0],
        }]

        rec = rules_engine.evaluate(
            collision_data=collision_data,
            weather_intel=weather_intel,
            trajectories=[],
            active_tracks=tracks,
            timestamp_seconds=3.0,
        )

        assert rec.recommendation_state == RecommendationState.STOP_RECOMMENDED
        assert rec.priority == WarningPriority.CRITICAL
        assert "stop" in rec.recommended_action.lower()

    def test_sensor_blackout_produces_unknown(
        self, rules_engine: RecommendationRulesEngine
    ):
        """Severe visibility blackout (< 20%) or excessive uncertainty (> 0.70).

        Must output:
        - RecommendationState: UNKNOWN
        - Reason: explicitly explaining sensor unreliability
        """
        collision_data = {
            "risk_state": "UNKNOWN",
            "fused_risk_score": 0.50,
            "min_ttc_seconds": None,
            "conflicting_pairs": [],
            "timestamp_seconds": 4.0,
            "uncertainty_score": 0.85,
        }
        weather_intel = {
            "visibility_score": 12.0,  # Below 20%
            "condition": "FOG",
            "is_degraded": True,
        }

        rec = rules_engine.evaluate(
            collision_data=collision_data,
            weather_intel=weather_intel,
            trajectories=[],
            active_tracks=[],
            timestamp_seconds=4.0,
        )

        assert rec.recommendation_state == RecommendationState.UNKNOWN
        assert rec.priority == WarningPriority.ADVISORY
        assert "unknown" in rec.primary_reason.lower() or "uncertainty" in rec.primary_reason.lower()
