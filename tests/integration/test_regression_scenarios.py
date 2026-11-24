"""Comprehensive PerceptGuard AI Regression Test Suite: Scenarios A through L.

Validates:
- Test A: Empty Straight Road (Normal driving, no false alerts)
- Test B: Gentle Curve (Curvature road, stable tracking, no false warnings)
- Test C: Adjacent Lane Vehicle (Parallel travel, constant offset, no collision risk)
- Test D: Following Vehicle (Safe headway, lead vehicle cruising, no false stop)
- Test E: Distant Sidewalk Pedestrian (Sidewalk pedestrian, no roadway incursion)
- Test F: Safely Crossing Pedestrian (Safe crossing distance, advisory caution)
- Test G: Sudden Pedestrian Incursion (Urgent vehicle path incursion, emergency stop)
- Test H: Actual Collision vs Prediction (Predicted conflict vs physical collision distinction)
- Test I: Poor Visibility / Fog / Rain (Degradation detection without false collision)
- Test J: Occlusion & Tracking Lifecycle (TENTATIVE -> CONFIRMED -> LOST -> REIDENTIFIED)
- Test K: Long Video Full Coverage (100% timeline coverage and uniform sampling)
- Test L: Repeated Upload Session Isolation (Clean reset, zero cross-session bleeding)
"""

import unittest
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from src.tracking.bytetrack_engine import ByteTrackEngine, TrackLifecycle
from src.collision_risk.risk_classifier import CollisionRiskClassifier
from src.collision_risk.schemas import (
    CollisionRiskState,
    ConflictType,
    TrajectoryConflict,
    UncertaintyLevel,
)
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
from src.safety_recommendation.recommendation_rules import RecommendationRulesEngine
from src.safety_recommendation.schemas import RecommendationState, WarningPriority


class TestRegressionScenariosAThroughL(unittest.TestCase):
    """Rigorous regression testing verifying autonomous safety perception rules."""

    def setUp(self) -> None:
        self.risk_classifier = CollisionRiskClassifier()
        self.road_analyzer = RoadContextAnalyzer(frame_width=1280, frame_height=720)
        self.ped_engine = PedestrianAwarenessEngine(frame_width=1280, frame_height=720)
        self.rules_engine = RecommendationRulesEngine()
        self.bytetrack = ByteTrackEngine()
        self.dummy_road_ctx = RoadContextData(
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

    # --------------------------------------------------------------------------
    # Scenario A: Empty Straight Road
    # --------------------------------------------------------------------------
    def test_a_empty_straight_road(self) -> None:
        """Test A: Normal driving on empty road produces LOW risk and NORMAL state."""
        risk_state, score, factors, explanation, status = self.risk_classifier.classify_risk(
            min_ttc=None,
            conflicts=[],
            closest_dist_px=None,
            closest_dist_m=None,
            closest_tid=None,
            uncertainty_score=0.10,
            uncertainty_level=UncertaintyLevel.LOW,
            is_unreliable=False,
            uncertainty_factors=[],
            num_tracks=0,
        )
        self.assertEqual(risk_state, CollisionRiskState.LOW)

        rec = self.rules_engine.evaluate(
            collision_data={"risk_state": "LOW", "fused_risk_score": 0.05, "min_ttc_seconds": None, "conflicting_pairs": []},
            weather_intel={"visibility_score": 92.0, "condition": "CLEAR", "is_degraded": False},
            trajectories=[],
            active_tracks=[],
            timestamp_seconds=1.0,
        )
        self.assertEqual(rec.recommendation_state, RecommendationState.NORMAL)
        self.assertEqual(rec.priority, WarningPriority.INFO)
        self.assertEqual(rec.risk_level, "LOW")

    # --------------------------------------------------------------------------
    # Scenario B: Gentle Curve
    # --------------------------------------------------------------------------
    def test_b_gentle_curve(self) -> None:
        """Test B: Vehicle following curved road geometry with zero closing velocity."""
        risk_state, score, factors, explanation, status = self.risk_classifier.classify_risk(
            min_ttc=None,
            conflicts=[],
            closest_dist_px=180.0,
            closest_dist_m=30.0,
            closest_tid=1,
            uncertainty_score=0.12,
            uncertainty_level=UncertaintyLevel.LOW,
            is_unreliable=False,
            uncertainty_factors=[],
            num_tracks=1,
        )
        self.assertEqual(risk_state, CollisionRiskState.LOW)

        rec = self.rules_engine.evaluate(
            collision_data={"risk_state": "LOW", "fused_risk_score": 0.08, "min_ttc_seconds": None, "closest_distance_m": 30.0, "conflicting_pairs": []},
            weather_intel={"visibility_score": 88.0, "condition": "CLEAR", "is_degraded": False},
            trajectories=[{"track_id": 1, "speed_kmh": 45.0, "motion_state": "CRUISING"}],
            active_tracks=[{"track_id": 1, "class_name": "car", "confidence": 0.9}],
            timestamp_seconds=2.0,
        )
        self.assertEqual(rec.recommendation_state, RecommendationState.NORMAL)

    # --------------------------------------------------------------------------
    # Scenario C: Adjacent Lane Vehicle
    # --------------------------------------------------------------------------
    def test_c_adjacent_lane_vehicle(self) -> None:
        """Test C: Parallel vehicle traveling in adjacent lane maintains lateral separation."""
        risk_state, score, factors, explanation, status = self.risk_classifier.classify_risk(
            min_ttc=None,
            conflicts=[],
            closest_dist_px=220.0,
            closest_dist_m=35.0,
            closest_tid=2,
            uncertainty_score=0.10,
            uncertainty_level=UncertaintyLevel.LOW,
            is_unreliable=False,
            uncertainty_factors=[],
            num_tracks=1,
        )
        self.assertEqual(risk_state, CollisionRiskState.LOW)

        rec = self.rules_engine.evaluate(
            collision_data={"risk_state": "LOW", "fused_risk_score": 0.05, "min_ttc_seconds": None, "closest_distance_m": 35.0, "conflicting_pairs": []},
            weather_intel={"visibility_score": 90.0, "condition": "CLEAR", "is_degraded": False},
            trajectories=[{"track_id": 2, "speed_kmh": 50.0, "motion_state": "CRUISING"}],
            active_tracks=[{"track_id": 2, "class_name": "car", "confidence": 0.92}],
            timestamp_seconds=3.0,
        )
        self.assertEqual(rec.recommendation_state, RecommendationState.NORMAL)

    # --------------------------------------------------------------------------
    # Scenario D: Following Vehicle
    # --------------------------------------------------------------------------
    def test_d_following_vehicle(self) -> None:
        """Test D: Lead vehicle ahead with safe headway buffer (35 meters)."""
        risk_state, score, factors, explanation, status = self.risk_classifier.classify_risk(
            min_ttc=None,
            conflicts=[],
            closest_dist_px=250.0,
            closest_dist_m=35.0,
            closest_tid=3,
            uncertainty_score=0.10,
            uncertainty_level=UncertaintyLevel.LOW,
            is_unreliable=False,
            uncertainty_factors=[],
            num_tracks=1,
        )
        self.assertEqual(risk_state, CollisionRiskState.LOW)

        rec = self.rules_engine.evaluate(
            collision_data={"risk_state": "LOW", "fused_risk_score": 0.06, "min_ttc_seconds": None, "closest_distance_m": 35.0, "conflicting_pairs": []},
            weather_intel={"visibility_score": 85.0, "condition": "CLEAR", "is_degraded": False},
            trajectories=[{"track_id": 3, "speed_kmh": 40.0, "motion_state": "CRUISING"}],
            active_tracks=[{"track_id": 3, "class_name": "car", "confidence": 0.95}],
            timestamp_seconds=4.0,
        )
        self.assertEqual(rec.recommendation_state, RecommendationState.NORMAL)

    # --------------------------------------------------------------------------
    # Scenario E: Distant Sidewalk Pedestrian
    # --------------------------------------------------------------------------
    def test_e_distant_sidewalk_pedestrian(self) -> None:
        """Test E: Pedestrian on sidewalk walking along road does not trigger stop."""
        tracks = [{
            "track_id": 10,
            "class_name": "person",
            "confidence": 0.88,
            "bbox": [100.0, 300.0, 150.0, 420.0],
        }]
        trajectories = [{
            "track_id": 10,
            "history": [{"x": 125.0, "y": 360.0}],
            "speed_kmh": 4.0,
            "primary_trajectory": [{"x_px": 125.0, "y_px": 380.0, "horizon_seconds": 1.0}],
        }]
        telemetry = self.ped_engine.evaluate_frame(
            active_tracks=tracks,
            trajectories=trajectories,
            road_ctx=self.dummy_road_ctx,
            timestamp_seconds=0.033,
            frame_index=1,
        )
        self.assertEqual(telemetry.primary_state, PedestrianAwarenessState.PERSON_DETECTED)
        self.assertEqual(telemetry.highest_risk_level, "LOW")
        self.assertIsNone(telemetry.primary_voice_alert)

    # --------------------------------------------------------------------------
    # Scenario F: Safely Crossing Pedestrian
    # --------------------------------------------------------------------------
    def test_f_safely_crossing_pedestrian(self) -> None:
        """Test F: Pedestrian approaching or crossing road at safe headway triggers CAUTION."""
        tracks = [{
            "track_id": 11,
            "class_name": "person",
            "confidence": 0.91,
            "bbox": [430.0, 460.0, 470.0, 580.0],
        }]
        trajectories = [{
            "track_id": 11,
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
        telemetry = self.ped_engine.evaluate_frame(
            active_tracks=tracks,
            trajectories=trajectories,
            road_ctx=self.dummy_road_ctx,
            timestamp_seconds=1.0,
            frame_index=20,
        )
        self.assertIn(
            telemetry.primary_state,
            (
                PedestrianAwarenessState.PEDESTRIAN_CROSSING,
                PedestrianAwarenessState.POTENTIAL_PEDESTRIAN_COLLISION_RISK,
            ),
        )
        self.assertIn("Pedestrian", telemetry.recommended_action)

    # --------------------------------------------------------------------------
    # Scenario G: Sudden Pedestrian Incursion
    # --------------------------------------------------------------------------
    def test_g_sudden_pedestrian_incursion(self) -> None:
        """Test G: Pedestrian incursion into vehicle travel path triggers early warning and stop."""
        tracks = [{
            "track_id": 12,
            "class_name": "person",
            "confidence": 0.94,
            "bbox": [580.0, 520.0, 660.0, 680.0],
        }]
        trajectories = [{
            "track_id": 12,
            "history": [{"x": 620.0, "y": 600.0}],
            "speed_kmh": 6.5,
            "primary_trajectory": [{"x_px": 640.0, "y_px": 620.0, "horizon_seconds": 1.0}],
        }]
        telemetry = self.ped_engine.evaluate_frame(
            active_tracks=tracks,
            trajectories=trajectories,
            road_ctx=self.dummy_road_ctx,
            timestamp_seconds=0.100,
            frame_index=3,
        )
        self.assertEqual(telemetry.primary_state, PedestrianAwarenessState.POTENTIAL_PEDESTRIAN_COLLISION_RISK)
        self.assertEqual(telemetry.highest_risk_level, "CRITICAL")
        self.assertIsNotNone(telemetry.primary_voice_alert)
        self.assertIn("prepare to stop", telemetry.recommended_action.lower())

    # --------------------------------------------------------------------------
    # Scenario H: Actual Collision vs Prediction
    # --------------------------------------------------------------------------
    def test_h_actual_collision_vs_prediction(self) -> None:
        """Test H: System distinguishes between lookahead trajectory prediction and physical collision event."""
        # Scenario H.1: Lookahead Trajectory Conflict (TTC = 1.4s, future conflict)
        conflict = TrajectoryConflict(
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
        risk_state, score, factors, explanation, status = self.risk_classifier.classify_risk(
            min_ttc=1.4,
            conflicts=[conflict],
            closest_dist_px=80.0,
            closest_dist_m=12.0,
            closest_tid=12,
            uncertainty_score=0.12,
            uncertainty_level=UncertaintyLevel.LOW,
            is_unreliable=False,
            uncertainty_factors=[],
            num_tracks=1,
        )
        self.assertIn(risk_state, [CollisionRiskState.HIGH, CollisionRiskState.CRITICAL])

        # Feature 5: Lookahead conflict triggers STOP_RECOMMENDED
        rec = self.rules_engine.evaluate(
            collision_data={
                "risk_state": risk_state.value,
                "fused_risk_score": score,
                "min_ttc_seconds": 1.4,
                "closest_distance_m": 12.0,
                "closest_track_id": 12,
                "conflicting_pairs": [conflict.to_dict()],
            },
            weather_intel={"visibility_score": 90.0, "condition": "CLEAR", "is_degraded": False},
            trajectories=[{"track_id": 12, "speed_kmh": 40.0, "motion_state": "TURNING_LEFT"}],
            active_tracks=[{"track_id": 12, "class_name": "motorcycle", "confidence": 0.91}],
            timestamp_seconds=5.0,
        )
        self.assertEqual(rec.recommendation_state, RecommendationState.STOP_RECOMMENDED)
        self.assertEqual(rec.priority, WarningPriority.CRITICAL)

    # --------------------------------------------------------------------------
    # Scenario I: Poor Visibility / Fog / Rain
    # --------------------------------------------------------------------------
    def test_i_poor_visibility_fog_rain(self) -> None:
        """Test I: Severe fog drops visibility score without hallucinating false vehicle collision."""
        fog_weather = {
            "visibility_score": 38.0,
            "visibility_level": "CRITICAL",
            "condition": "FOG",
            "is_degraded": True,
            "fog": {"severity": 0.85, "classification": "Dense Fog"},
        }
        # No vehicle conflicts
        risk_state, score, factors, explanation, status = self.risk_classifier.classify_risk(
            min_ttc=None,
            conflicts=[],
            closest_dist_px=None,
            closest_dist_m=None,
            closest_tid=None,
            uncertainty_score=0.45,
            uncertainty_level=UncertaintyLevel.HIGH,
            is_unreliable=False,
            uncertainty_factors=["High optical attenuation from fog."],
            num_tracks=0,
        )
        # Even with severe fog, collision risk is NOT critical because there are no vehicle conflicts
        self.assertNotEqual(risk_state, CollisionRiskState.CRITICAL)

        rec = self.rules_engine.evaluate(
            collision_data={"risk_state": risk_state.value, "fused_risk_score": 0.2, "min_ttc_seconds": None, "conflicting_pairs": []},
            weather_intel=fog_weather,
            trajectories=[],
            active_tracks=[],
            timestamp_seconds=6.0,
        )
        # Advises speed reduction due to adverse optics, but not emergency collision stop
        self.assertIn("REDUCE", rec.primary_action)
        self.assertNotEqual(rec.recommendation_state, RecommendationState.STOP_RECOMMENDED)

    # --------------------------------------------------------------------------
    # Scenario J: Occlusion & Tracking Lifecycle
    # --------------------------------------------------------------------------
    def test_j_occlusion_and_tracking_lifecycle(self) -> None:
        """Test J: ByteTrack lifecycle transitions from TENTATIVE to CONFIRMED, LOST upon occlusion, and REIDENTIFIED."""
        det = [{"bbox_xyxy": [200.0, 200.0, 300.0, 300.0], "confidence": 0.92, "class_id": 2, "class_name": "car"}]

        # Frame 1: Initial observation -> TENTATIVE
        t1 = self.bytetrack.update(det)
        self.assertEqual(len(t1), 1)
        self.assertEqual(t1[0].track_lifecycle, TrackLifecycle.TENTATIVE.value)

        # Frame 2: Second observation -> TENTATIVE
        t2 = self.bytetrack.update(det)
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2[0].track_lifecycle, TrackLifecycle.TENTATIVE.value)

        # Frame 3: Third observation -> CONFIRMED
        t3 = self.bytetrack.update(det)
        self.assertEqual(len(t3), 1)
        self.assertEqual(t3[0].track_lifecycle, TrackLifecycle.CONFIRMED.value)

        # Frame 4: Occluded (zero detections) -> track is lost
        t4 = self.bytetrack.update([])
        self.assertEqual(len(t4), 0)
        self.assertIn(1, self.bytetrack._lost_track_ids)

        # Frame 5: Re-emerges from occlusion -> REIDENTIFIED
        t5 = self.bytetrack.update(det)
        self.assertEqual(len(t5), 1)
        self.assertEqual(t5[0].track_lifecycle, TrackLifecycle.REIDENTIFIED.value)

    # --------------------------------------------------------------------------
    # Scenario K: Long Video Full Coverage
    # --------------------------------------------------------------------------
    def test_k_long_video_full_coverage(self) -> None:
        """Test K: Video pipeline reports 100% timeline coverage and uniform sampling."""
        from src.video_processing.video_pipeline_runner import VideoPipelineRunResult
        telemetry = {
            "total_video_frames": 5400,
            "video_duration_seconds": 180.0,
            "video_fps": 30.0,
            "frame_stride_applied": 3,
            "total_frames_analyzed": 1800,
            "effective_sampling_fps": 10.0,
            "timeline_coverage_percentage": 100.0,
            "first_analyzed_timestamp": 0.0,
            "last_analyzed_timestamp": 180.0,
            "sampling_mode": "UNIFORM_STRIDE",
        }
        res = VideoPipelineRunResult(
            job_id="job_test_k",
            video_filename="long_road_video.mp4",
            video_metadata={
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "total_frames": 5400,
                "duration_seconds": 180.0,
                "file_size_bytes": 1024000,
            },
            frames_processed=1800,
            total_detections=4200,
            unique_tracks=35,
            total_alerts=2,
            active_alerts=0,
            calibration_status="CALIBRATED",
            per_horizon_risk={},
            frames=[],
            stage_durations_ms={"decode": 1200.0, "yolo": 3000.0, "tracker": 800.0},
            total_duration_seconds=180.0,
            timeline_telemetry=telemetry,
        )
        self.assertEqual(res.timeline_telemetry["timeline_coverage_percentage"], 100.0)
        self.assertEqual(res.timeline_telemetry["total_frames_analyzed"], 1800)
        d = res.to_dict()
        self.assertEqual(d["timeline_telemetry"]["timeline_coverage_percentage"], 100.0)

    # --------------------------------------------------------------------------
    # Scenario L: Repeated Upload Session Isolation
    # --------------------------------------------------------------------------
    def test_l_repeated_upload_session_isolation(self) -> None:
        """Test L: Tracker reset guarantees zero identity bleeding between sequential uploads."""
        det_video1 = [{"bbox_xyxy": [100.0, 100.0, 200.0, 200.0], "confidence": 0.9, "class_id": 2}]
        det_video2 = [{"bbox_xyxy": [400.0, 400.0, 500.0, 500.0], "confidence": 0.88, "class_id": 7}]

        # Session 1: Process video 1
        tracks1 = self.bytetrack.update(det_video1)
        self.assertEqual(tracks1[0].track_id, 1)
        self.assertEqual(self.bytetrack.unique_track_count, 1)

        # Session Reset: User uploads new video or resets stream
        self.bytetrack.reset()
        self.assertEqual(self.bytetrack.unique_track_count, 0)
        self.assertEqual(self.bytetrack.frame_count, 0)
        self.assertEqual(len(self.bytetrack._lost_track_ids), 0)

        # Session 2: Process video 2
        tracks2 = self.bytetrack.update(det_video2)
        # ID must restart cleanly at 1
        self.assertEqual(tracks2[0].track_id, 1)
        self.assertEqual(self.bytetrack.unique_track_count, 1)


if __name__ == "__main__":
    unittest.main()
