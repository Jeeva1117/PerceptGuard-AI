"""Integration tests for Pedestrian-Aware Driving Intelligence (Feature 5 & Feature 4).

Validates the five specified pedestrian awareness decision states:
1. PERSON DETECTED (Sidewalk / safe separation -> LOW risk, NO ACTIVE WARNING)
2. PEDESTRIAN APPROACHING ROAD (Approaching roadway -> CAUTION, MONITOR AND REDUCE SPEED)
3. PEDESTRIAN CROSSING (Crossing roadway -> UNDER EVALUATION)
4. POTENTIAL PEDESTRIAN COLLISION RISK (Predicted travel path conflict -> HIGH/CRITICAL, PREPARE TO STOP)
5. PEDESTRIAN TRACKING LOST (Occluded or lost tracking -> UNKNOWN, MAINTAIN CAUTION)
"""

import pytest
import numpy as np

from src.collision_risk.road_context_analyzer import (
    RoadContextAnalyzer,
    RoadContextData,
    LaneState,
    PedestrianRoadPosition,
    PedestrianMovementIntent,
)
from src.safety_recommendation.pedestrian_awareness_engine import (
    PedestrianAwarenessEngine,
    PedestrianAwarenessState,
)
from src.safety_recommendation.safety_engine import SafetyRecommendationEngine
from src.safety_recommendation.schemas import RecommendationState, WarningPriority


class TestPedestrianAwarenessIntegration:
    """Test suite verifying pedestrian awareness states and safety engine integration."""

    @pytest.fixture
    def road_analyzer(self) -> RoadContextAnalyzer:
        return RoadContextAnalyzer(frame_width=1280, frame_height=720)

    @pytest.fixture
    def ped_engine(self) -> PedestrianAwarenessEngine:
        return PedestrianAwarenessEngine(frame_width=1280, frame_height=720)

    @pytest.fixture
    def dummy_road_ctx(self, road_analyzer: RoadContextAnalyzer) -> RoadContextData:
        # Synthetic clean roadway: lane from x=350 to x=930, center=640
        return RoadContextData(
            lane_state=LaneState.LANE_DETECTED,
            left_boundary_x=350.0,
            right_boundary_x=930.0,
            lane_center_px=640.0,
            lane_width_px=580.0,
            ego_lane_offset_px=0.0,
            road_surface_quality="DRY",
            crosswalk_detected=False,
            confidence=0.92,
        )

    def test_pedestrian_on_sidewalk_produces_person_detected(
        self, ped_engine: PedestrianAwarenessEngine, dummy_road_ctx: RoadContextData
    ):
        """Test A: Pedestrian on sidewalk produces PERSON DETECTED, LOW risk, NO audible alert."""
        # Pedestrian at x=150 (far left sidewalk, well outside lane 350-930)
        tracks = [{
            "track_id": 101,
            "class_name": "person",
            "confidence": 0.88,
            "bbox": [130.0, 400.0, 170.0, 520.0],
        }]
        trajectories = [{
            "track_id": 101,
            "history": [
                {"x": 150.0, "y": 420.0, "timestamp": 0.0},
                {"x": 150.0, "y": 440.0, "timestamp": 0.5},
                {"x": 150.0, "y": 460.0, "timestamp": 1.0},
            ],
            "speed_kmh": 4.5,
            "primary_trajectory": [
                {"x_px": 150.0, "y_px": 480.0, "horizon_seconds": 1.0},
                {"x_px": 150.0, "y_px": 500.0, "horizon_seconds": 3.0},
            ],
        }]

        telemetry = ped_engine.evaluate_frame(
            active_tracks=tracks,
            trajectories=trajectories,
            road_ctx=dummy_road_ctx,
            timestamp_seconds=1.0,
            frame_index=10,
        )

        assert telemetry.pedestrians_count == 1
        assert telemetry.primary_state == PedestrianAwarenessState.PERSON_DETECTED
        assert telemetry.highest_risk_level == "LOW"
        assert telemetry.primary_voice_alert is None  # Must remain silent on sidewalk!

        ped = telemetry.active_pedestrians[0]
        assert ped.road_position == PedestrianRoadPosition.SIDEWALK
        assert ped.is_in_ego_path is False

    def test_pedestrian_approaching_road_produces_caution(
        self, ped_engine: PedestrianAwarenessEngine, dummy_road_ctx: RoadContextData
    ):
        """Test B: Pedestrian at road edge moving toward roadway produces CAUTION."""
        # Pedestrian at x=280 moving toward road edge (left boundary is at 350)
        tracks = [{
            "track_id": 102,
            "class_name": "person",
            "confidence": 0.90,
            "bbox": [260.0, 480.0, 300.0, 600.0],
        }]
        trajectories = [{
            "track_id": 102,
            "history": [
                {"x": 240.0, "y": 500.0, "timestamp": 0.0},
                {"x": 260.0, "y": 520.0, "timestamp": 0.5},
                {"x": 280.0, "y": 540.0, "timestamp": 1.0},
            ],
            "speed_kmh": 5.2,
            "primary_trajectory": [
                {"x_px": 310.0, "y_px": 550.0, "horizon_seconds": 1.0},
                {"x_px": 335.0, "y_px": 560.0, "horizon_seconds": 2.0},
            ],
        }]

        telemetry = ped_engine.evaluate_frame(
            active_tracks=tracks,
            trajectories=trajectories,
            road_ctx=dummy_road_ctx,
            timestamp_seconds=1.0,
            frame_index=15,
        )

        assert telemetry.primary_state == PedestrianAwarenessState.PEDESTRIAN_APPROACHING_ROAD
        assert telemetry.highest_risk_level == "CAUTION"
        assert "approaching" in (telemetry.primary_voice_alert or "").lower()

    def test_pedestrian_crossing_roadway_produces_crossing_state(
        self, ped_engine: PedestrianAwarenessEngine, dummy_road_ctx: RoadContextData
    ):
        """Test C: Pedestrian crossing within roadway produces PEDESTRIAN CROSSING."""
        # Pedestrian at x=450 moving across corridor
        tracks = [{
            "track_id": 103,
            "class_name": "person",
            "confidence": 0.91,
            "bbox": [430.0, 460.0, 470.0, 580.0],
        }]
        trajectories = [{
            "track_id": 103,
            "history": [
                {"x": 380.0, "y": 500.0, "timestamp": 0.0},
                {"x": 415.0, "y": 510.0, "timestamp": 0.5},
                {"x": 450.0, "y": 520.0, "timestamp": 1.0},
            ],
            "speed_kmh": 5.8,
            "primary_trajectory": [
                {"x_px": 500.0, "y_px": 530.0, "horizon_seconds": 1.0},
            ],
        }]

        telemetry = ped_engine.evaluate_frame(
            active_tracks=tracks,
            trajectories=trajectories,
            road_ctx=dummy_road_ctx,
            timestamp_seconds=1.0,
            frame_index=20,
        )

        assert telemetry.primary_state in (
            PedestrianAwarenessState.PEDESTRIAN_CROSSING,
            PedestrianAwarenessState.POTENTIAL_PEDESTRIAN_COLLISION_RISK,
        )

    def test_pedestrian_collision_risk_escalates_to_stop_recommended(
        self, ped_engine: PedestrianAwarenessEngine, dummy_road_ctx: RoadContextData
    ):
        """Test D: Imminent trajectory conflict with pedestrian triggers POTENTIAL PEDESTRIAN COLLISION RISK and STOP."""
        # Pedestrian dead center in ego travel path at x=640, with low TTC (1.2s)
        tracks = [{
            "track_id": 104,
            "class_name": "person",
            "confidence": 0.95,
            "bbox": [620.0, 500.0, 660.0, 640.0],
        }]
        trajectories = [{
            "track_id": 104,
            "history": [
                {"x": 640.0, "y": 500.0, "timestamp": 0.0},
                {"x": 640.0, "y": 550.0, "timestamp": 0.5},
                {"x": 640.0, "y": 570.0, "timestamp": 1.0},
            ],
            "speed_kmh": 3.0,
            "primary_trajectory": [
                {"x_px": 640.0, "y_px": 590.0, "horizon_seconds": 1.2},
            ],
        }]

        telemetry = ped_engine.evaluate_frame(
            active_tracks=tracks,
            trajectories=trajectories,
            road_ctx=dummy_road_ctx,
            timestamp_seconds=1.0,
            frame_index=25,
        )

        assert telemetry.primary_state == PedestrianAwarenessState.POTENTIAL_PEDESTRIAN_COLLISION_RISK
        assert telemetry.highest_risk_level in ("HIGH", "CRITICAL")
        assert "stop" in (telemetry.primary_voice_alert or "").lower()

        # Check safety engine integration
        safety_engine = SafetyRecommendationEngine()
        collision_data = {
            "risk_state": "HIGH",
            "min_ttc_seconds": 1.2,
            "conflicting_pairs": [{"track_a": 104, "track_b": 0, "class_a": "person", "class_b": "ego"}],
            "timestamp_seconds": 1.0,
        }
        weather_intel = {"visibility_score": 90.0, "condition": "CLEAR", "is_degraded": False}

        rec = safety_engine.evaluate_frame(
            collision_data=collision_data,
            weather_intel=weather_intel,
            trajectories=trajectories,
            active_tracks=tracks,
            timestamp_seconds=1.0,
            frame_index=25,
            pedestrian_telemetry=telemetry.to_dict(),
        )

        assert rec.recommendation_state in (
            RecommendationState.STOP_RECOMMENDED,
            RecommendationState.PREPARE_TO_STOP,
        )
        assert rec.priority in (WarningPriority.CRITICAL, WarningPriority.WARNING)

    def test_pedestrian_tracking_loss_triggers_state_e(
        self, ped_engine: PedestrianAwarenessEngine, dummy_road_ctx: RoadContextData
    ):
        """Test E: Occlusion/disappearance triggers PEDESTRIAN TRACKING LOST and UNKNOWN risk."""
        # Frame 1: Pedestrian active
        tracks = [{
            "track_id": 105,
            "class_name": "person",
            "confidence": 0.85,
            "bbox": [150.0, 400.0, 190.0, 520.0],
        }]
        ped_engine.evaluate_frame(
            active_tracks=tracks,
            trajectories=[],
            road_ctx=dummy_road_ctx,
            timestamp_seconds=1.0,
            frame_index=1,
        )

        # Frame 2: Pedestrian disappears (e.g. occluded behind pole or parked car)
        telemetry2 = ped_engine.evaluate_frame(
            active_tracks=[],
            trajectories=[],
            road_ctx=dummy_road_ctx,
            timestamp_seconds=1.5,
            frame_index=2,
        )

        assert telemetry2.pedestrians_count == 0
        assert telemetry2.primary_state == PedestrianAwarenessState.PEDESTRIAN_TRACKING_LOST
        assert telemetry2.highest_risk_level == "UNKNOWN"
        assert 105 in telemetry2.lost_pedestrian_ids

    def test_summary_generation(
        self, ped_engine: PedestrianAwarenessEngine, dummy_road_ctx: RoadContextData
    ):
        """Test stream summary generation."""
        summary = ped_engine.generate_summary()
        assert "total_unique_pedestrians" in summary
        assert "frames_with_pedestrians" in summary
        assert "potential_collision_events" in summary
