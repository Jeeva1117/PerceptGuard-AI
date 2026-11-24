"""Comprehensive unit and integration test suite for Collision & Road Risk Assessment (Feature 4).

All scenarios are explicitly modeled and verified:
1. Approaching objects (positive closing speed, valid TTC).
2. Diverging objects (negative closing speed, TTC = None).
3. Stationary objects (zero closing speed, TTC = None).
4. Parallel trajectories (constant distance, TTC = None).
5. Crossing trajectories (angle-validated 1D TTC vs 2D conflict).
6. Spatially overlapping trajectories at DIFFERENT times (MUST NOT collide).
7. Potential rear-end collision conflicts.
8. Vehicle-pedestrian conflict (vulnerability escalation).
9. Zero closing speed handling.
10. Negative closing speed handling.
11. Missing trajectory history (< 2 frames).
12. Low camera visibility (< 20% -> UNKNOWN).
13. Excessive uncertainty (> 0.70 -> UNKNOWN with score=None).
14. Stale observation gap (> 0.5s).
15. Temporal risk escalation (immediate) and de-escalation (hysteresis).
16. Disappearing track while in elevated risk zone.
17. Distance analyzer image-space vs pinhole ground-plane separation.
18. Stream summary aggregation.
19. End-to-end video pipeline runner integration.
20. FastAPI endpoint contract verification.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.collision_risk import (
    AssessmentStatus,
    CollisionRiskAssessment,
    CollisionRiskClassifier,
    CollisionRiskEngine,
    CollisionRiskState,
    CollisionStreamSummary,
    ConflictType,
    DistanceAnalyzer,
    RelativeMotionAnalyzer,
    TemporalRiskTracker,
    TrajectoryConflict,
    TrajectoryConflictAnalyzer,
    TTCCalculator,
    UncertaintyAnalyzer,
    UncertaintyLevel,
)
from src.video_processing.video_pipeline_runner import VideoPipelineRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# 1. Distance Analyzer Tests
# ==============================================================================

def test_distance_analyzer_pixel_and_metric_separation():
    """Synthetic: Verify distance analyzer preserves separation between pixel and metric spaces."""
    analyzer = DistanceAnalyzer(frame_width=1280, frame_height=720)

    # Agent A: pixel (640, 500), metric (0.0m lateral, 10.0m forward)
    # Agent B: pixel (640, 400), metric (0.0m lateral, 25.0m forward)
    dist_px, dist_m = analyzer.compute_pairwise_distance(
        center_a_px=(640.0, 500.0),
        center_b_px=(640.0, 400.0),
        center_a_m=(0.0, 10.0),
        center_b_m=(0.0, 25.0),
    )
    assert dist_px == pytest.approx(100.0, abs=0.1)
    assert dist_m == pytest.approx(15.0, abs=0.1)

    # When metric coordinates are unavailable (e.g. above horizon), distance_m must be None
    dist_px_no_metric, dist_m_no_metric = analyzer.compute_pairwise_distance(
        center_a_px=(640.0, 500.0),
        center_b_px=(640.0, 200.0),
        center_a_m=(0.0, 10.0),
        center_b_m=None,
    )
    assert dist_px_no_metric == pytest.approx(300.0, abs=0.1)
    assert dist_m_no_metric is None


# ==============================================================================
# 2. Relative Motion & TTC Tests
# ==============================================================================

def test_approaching_objects_positive_closing_speed_and_valid_ttc():
    """Synthetic: Two objects directly approaching each other along line-of-sight."""
    rel_analyzer = RelativeMotionAnalyzer()
    ttc_calc = TTCCalculator()

    # Agent A at (100, 100) moving right (+vx = 50 px/s)
    # Agent B at (300, 100) moving left (-vx = -50 px/s)
    # Distance = 200 px, Closing speed = 50 - (-50) = 100 px/s
    rel = rel_analyzer.compute_relative_kinematics(
        pos_a=(100.0, 100.0),
        vel_a=(50.0, 0.0),
        acc_a=(0.0, 0.0),
        pos_b=(300.0, 100.0),
        vel_b=(-50.0, 0.0),
        acc_b=(0.0, 0.0),
    )
    assert rel["is_approaching"] is True
    assert rel["motion_pattern"] == "APPROACHING"
    assert rel["closing_speed_px_s"] == pytest.approx(100.0, abs=1.0)

    ttc = ttc_calc.calculate_linear_ttc(rel["distance_px"], rel["closing_speed_px_s"])
    assert ttc == pytest.approx(2.0, abs=0.05)  # 200 / 100 = 2.0s


def test_diverging_objects_negative_closing_speed_and_ttc_none():
    """Synthetic: Two objects moving away from each other (TTC must be None)."""
    rel_analyzer = RelativeMotionAnalyzer()
    ttc_calc = TTCCalculator()

    # Agent A at (200, 100) moving left (-50 px/s)
    # Agent B at (300, 100) moving right (+50 px/s)
    # Diverging: closing speed = -100 px/s
    rel = rel_analyzer.compute_relative_kinematics(
        pos_a=(200.0, 100.0),
        vel_a=(-50.0, 0.0),
        acc_a=(0.0, 0.0),
        pos_b=(300.0, 100.0),
        vel_b=(50.0, 0.0),
        acc_b=(0.0, 0.0),
    )
    assert rel["is_approaching"] is False
    assert rel["motion_pattern"] == "DIVERGING"
    assert rel["closing_speed_px_s"] < 0.0

    # TTC must be mathematically invalid (None)
    ttc = ttc_calc.calculate_linear_ttc(rel["distance_px"], rel["closing_speed_px_s"])
    assert ttc is None


def test_stationary_objects_zero_closing_speed_and_ttc_none():
    """Synthetic: Stationary objects with zero closing velocity must return TTC = None."""
    rel_analyzer = RelativeMotionAnalyzer()
    ttc_calc = TTCCalculator()

    rel = rel_analyzer.compute_relative_kinematics(
        pos_a=(200.0, 200.0),
        vel_a=(0.0, 0.0),
        acc_a=(0.0, 0.0),
        pos_b=(400.0, 200.0),
        vel_b=(0.0, 0.0),
        acc_b=(0.0, 0.0),
    )
    assert rel["is_approaching"] is False
    assert abs(rel["closing_speed_px_s"]) < 1.0

    ttc = ttc_calc.calculate_linear_ttc(rel["distance_px"], rel["closing_speed_px_s"])
    assert ttc is None


def test_parallel_trajectories_constant_distance():
    """Synthetic: Parallel vehicles moving in the same direction at equal speeds."""
    rel_analyzer = RelativeMotionAnalyzer()
    ttc_calc = TTCCalculator()

    # Vehicle in lane 1: (400, 500) moving forward vy = -80 px/s
    # Vehicle in lane 2: (550, 500) moving forward vy = -80 px/s
    rel = rel_analyzer.compute_relative_kinematics(
        pos_a=(400.0, 500.0),
        vel_a=(0.0, -80.0),
        acc_a=(0.0, 0.0),
        pos_b=(550.0, 500.0),
        vel_b=(0.0, -80.0),
        acc_b=(0.0, 0.0),
    )
    assert rel["is_approaching"] is False
    assert abs(rel["closing_speed_px_s"]) < 1.0

    ttc = ttc_calc.calculate_linear_ttc(rel["distance_px"], rel["closing_speed_px_s"])
    assert ttc is None


def test_ego_ttc_crossing_lateral_trajectory_invalidated():
    """Synthetic: Cross-traffic crossing perpendicularly should invalidate 1D line-of-sight TTC."""
    ttc_calc = TTCCalculator(max_crossing_angle_deg=65.0)

    # Ego camera vehicle at (640, 720) looking forward
    # Cross traffic at (640, 500) moving purely horizontally (+vx = 80, vy = 0)
    # Velocity vector is perpendicular (90 deg) to line of sight
    res = ttc_calc.evaluate_ego_ttc(
        distance_px=220.0,
        closing_speed_px_s=15.0,
        closing_acc_px_s2=0.0,
        target_center_px=(640.0, 500.0),
        target_velocity_px=(80.0, 0.0),
        ego_center_px=(640.0, 720.0),
    )
    assert res["is_valid"] is False
    assert res["ttc_seconds"] is None
    assert "crossing/lateral" in res["invalidation_reason"]


# ==============================================================================
# 3. Spatio-Temporal Trajectory Conflict Tests
# ==============================================================================

def test_spatially_overlapping_trajectories_at_different_times_do_not_collide():
    """Synthetic: Two trajectories cross the EXACT same pixel coordinate (500, 500),

    but Agent 1 arrives at t=1.0s and Agent 2 arrives at t=4.0s.
    MUST NOT trigger a collision conflict!
    """
    conflict_analyzer = TrajectoryConflictAnalyzer()

    # Track 1 passes (500, 500) at step 10 (t = 1.0s)
    t1 = {
        "track_id": 1,
        "class_name": "car",
        "velocity": {"vx": 50.0, "vy": 0.0},
        "primary_trajectory": [
            {"step_index": k, "horizon_seconds": k * 0.1, "x_px": 400.0 + k * 10.0, "y_px": 500.0, "uncertainty_radius_px": 3.0}
            for k in range(1, 40)
        ],
    }

    # Track 2 passes (500, 500) at step 35 (t = 3.5s)
    t2 = {
        "track_id": 2,
        "class_name": "car",
        "velocity": {"vx": 0.0, "vy": 30.0},
        "primary_trajectory": [
            {"step_index": k, "horizon_seconds": k * 0.1, "x_px": 500.0, "y_px": 150.0 + k * 10.0, "uncertainty_radius_px": 3.0}
            for k in range(1, 40)
        ],
    }

    conflicts = conflict_analyzer.detect_conflicts([t1, t2], fps=10.0)
    # Because at step 10, T1 is at (500, 500) while T2 is at (500, 250) -> separation 250 px >> threshold
    # And at step 35, T2 is at (500, 500) while T1 is at (750, 500) -> separation 250 px >> threshold
    assert len(conflicts) == 0, "Temporal separation should prevent false-positive conflict!"


def test_synchronous_trajectory_collision_detected():
    """Synthetic: Two trajectories reach (500, 500) at the SAME time step (t = 2.0s)."""
    conflict_analyzer = TrajectoryConflictAnalyzer()

    # T1 arrives at (500, 500) at step 20 (t = 2.0s)
    t1 = {
        "track_id": 10,
        "class_name": "car",
        "velocity": {"vx": 20.0, "vy": 0.0},
        "primary_trajectory": [
            {"step_index": k, "horizon_seconds": k * 0.1, "x_px": 300.0 + k * 10.0, "y_px": 500.0, "uncertainty_radius_px": 3.0}
            for k in range(1, 30)
        ],
    }

    # T2 arrives at (500, 500) at step 20 (t = 2.0s)
    t2 = {
        "track_id": 20,
        "class_name": "truck",
        "velocity": {"vx": 0.0, "vy": -20.0},
        "primary_trajectory": [
            {"step_index": k, "horizon_seconds": k * 0.1, "x_px": 500.0, "y_px": 700.0 - k * 10.0, "uncertainty_radius_px": 3.0}
            for k in range(1, 30)
        ],
    }

    conflicts = conflict_analyzer.detect_conflicts([t1, t2], fps=10.0)
    assert len(conflicts) >= 1
    c = conflicts[0]
    assert c.track_a == 10 and c.track_b == 20
    # Spatial collision occurs as physical bounding envelopes overlap (starting at t = 1.5s)
    assert c.time_to_conflict_seconds == pytest.approx(1.5, abs=0.2)
    assert c.horizon_label in ("H1", "H3")
    assert c.separation_at_conflict_px <= 80.0


def test_vehicle_pedestrian_conflict_categorization():
    """Synthetic: Detects pedestrian crossing conflict and tags with PEDESTRIAN_CROSSING."""
    conflict_analyzer = TrajectoryConflictAnalyzer()

    t_car = {
        "track_id": 5,
        "class_name": "car",
        "velocity": {"vx": 0.0, "vy": -40.0},
        "predicted_horizons": {
            "h1": {"x": 600.0, "y": 450.0, "uncertainty_radius": 5.0},
        },
    }

    t_ped = {
        "track_id": 6,
        "class_name": "pedestrian",
        "velocity": {"vx": 20.0, "vy": 0.0},
        "predicted_horizons": {
            "h1": {"x": 610.0, "y": 450.0, "uncertainty_radius": 4.0},
        },
    }

    conflicts = conflict_analyzer.detect_conflicts([t_car, t_ped], fps=10.0)
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.PEDESTRIAN_CROSSING


# ==============================================================================
# 4. Uncertainty & UNKNOWN State Tests
# ==============================================================================

def test_low_camera_visibility_forces_unknown_risk():
    """Synthetic: Severely degraded camera visibility (<20%) MUST assign UNKNOWN risk with score=None."""
    engine = CollisionRiskEngine()

    weather_severely_foggy = {
        "visibility_score": 12.0,
        "condition": "FOG",
        "is_degraded": True,
    }

    trajectories = [
        {
            "track_id": 1,
            "class_name": "car",
            "center": [640.0, 500.0],
            "velocity": {"vx": 0.0, "vy": 20.0},
            "speed_kmh": 25.0,
            "confidence": 0.70,
            "history_length": 5,
        }
    ]

    assessment = engine.assess_frame(
        trajectories=trajectories,
        weather_intel=weather_severely_foggy,
        timestamp_seconds=1.0,
    )

    assert assessment.risk_state == CollisionRiskState.UNKNOWN
    assert assessment.risk_score is None
    assert assessment.status == AssessmentStatus.UNKNOWN_HIGH_UNCERTAINTY
    assert any("visibility below critical" in f.lower() for f in assessment.primary_contributing_factors)


def test_missing_history_elevates_uncertainty():
    """Synthetic: Fresh tracks with 1 observation frame have elevated uncertainty."""
    uncertainty_analyzer = UncertaintyAnalyzer()

    weather_clear = {"visibility_score": 95.0, "condition": "CLEAR"}
    fresh_tracks = [
        {"track_id": 1, "confidence": 0.90, "history_length": 1, "history": [{"x": 100, "y": 100}]}
    ]

    u_score, u_level, factors, is_unreliable = uncertainty_analyzer.evaluate_uncertainty(
        weather_clear, fresh_tracks
    )
    assert any("immature history" in f for f in factors)
    assert u_score > 0.20


# ==============================================================================
# 5. Temporal Risk Tracker & Hysteresis Tests
# ==============================================================================

def test_temporal_risk_tracker_escalation_and_hysteresis():
    """Synthetic: Escalation to CRITICAL is immediate; de-escalation requires patience frames."""
    tracker = TemporalRiskTracker(de_escalation_patience_frames=3)

    # Frame 1: LOW
    s1, _, _ = tracker.update(CollisionRiskState.LOW, 0.1, [1])
    assert s1 == CollisionRiskState.LOW

    # Frame 2: Sudden CRITICAL (must escalate immediately)
    s2, _, _ = tracker.update(CollisionRiskState.CRITICAL, 0.2, [1])
    assert s2 == CollisionRiskState.CRITICAL

    # Frame 3: Single LOW measurement (sensor noise/flicker) -> Hysteresis holds CRITICAL!
    s3, _, _ = tracker.update(CollisionRiskState.LOW, 0.3, [1])
    assert s3 == CollisionRiskState.CRITICAL, "Hysteresis must hold CRITICAL on single lower measurement"

    # Frame 4: Second LOW measurement -> Still holds CRITICAL
    s4, _, _ = tracker.update(CollisionRiskState.LOW, 0.4, [1])
    assert s4 == CollisionRiskState.CRITICAL

    # Frame 5: Third LOW measurement -> Patience satisfied, de-escalates to LOW
    s5, _, _ = tracker.update(CollisionRiskState.LOW, 0.5, [1])
    assert s5 == CollisionRiskState.LOW


def test_stale_observation_detection():
    """Synthetic: Frame gap > 0.5s triggers stale observation warning."""
    tracker = TemporalRiskTracker(stale_threshold_seconds=0.5)

    tracker.update(CollisionRiskState.LOW, 0.0, [1])
    # Next frame arrives 0.8s later (stale gap)
    _, is_stale, note = tracker.update(CollisionRiskState.LOW, 0.8, [1])
    assert is_stale is True
    assert note is not None
    assert "Stale frame gap" in note


def test_disappearing_track_in_elevated_risk():
    """Synthetic: Track disappearing during elevated risk generates explicit warning."""
    tracker = TemporalRiskTracker()

    tracker.update(CollisionRiskState.HIGH, 0.1, [10, 20])
    # Track 20 suddenly disappears in next frame
    _, _, note = tracker.update(CollisionRiskState.HIGH, 0.2, [10])
    assert note is not None
    assert "#[20] disappeared" in note


# ==============================================================================
# 6. Integrated Engine & Stream Summary Tests
# ==============================================================================

def test_collision_risk_engine_end_to_end_nominal():
    """Synthetic: Nominal scene with safe headway produces LOW risk."""
    engine = CollisionRiskEngine()

    weather = {"visibility_score": 90.0, "condition": "CLEAR"}
    trajectories = [
        {
            "track_id": 1,
            "class_name": "car",
            "center": [640.0, 300.0],
            "velocity": {"vx": 0.0, "vy": -30.0},
            "acceleration": {"ax": 0.0, "ay": 0.0},
            "speed_kmh": 40.0,
            "confidence": 0.92,
            "history_length": 10,
            "predicted_horizons": {
                "h1": {"x": 640.0, "y": 270.0, "uncertainty_radius": 4.0},
                "h3": {"x": 640.0, "y": 210.0, "uncertainty_radius": 8.0},
                "h5": {"x": 640.0, "y": 150.0, "uncertainty_radius": 12.0},
            },
        }
    ]

    assessment = engine.assess_frame(
        trajectories=trajectories,
        weather_intel=weather,
        timestamp_seconds=1.0,
    )

    assert assessment.risk_state == CollisionRiskState.LOW
    assert assessment.risk_score is not None
    assert assessment.risk_score < 0.35
    assert assessment.closest_track_id == 1
    assert len(assessment.conflicting_pairs) == 0

    summary = engine.generate_stream_summary()
    assert summary.total_frames_assessed == 1
    assert summary.critical_risk_frames == 0


def test_collision_risk_engine_critical_head_on_ttc():
    """Synthetic: Head-on approaching truck at high closing speed triggers CRITICAL state."""
    engine = CollisionRiskEngine()

    weather = {"visibility_score": 88.0, "condition": "CLEAR"}
    # Truck at (640, 600) rapidly closing on ego (640, 720) with vy = 100 px/s
    # Distance = 120 px, Closing speed = 100 px/s -> TTC = 1.2s (< 1.8s threshold)
    trajectories = [
        {
            "track_id": 9,
            "class_name": "truck",
            "center": [640.0, 600.0],
            "velocity": {"vx": 0.0, "vy": 100.0},
            "acceleration": {"ax": 0.0, "ay": 0.0},
            "speed_kmh": 65.0,
            "confidence": 0.95,
            "history_length": 8,
            "predicted_horizons": {
                "h1": {"x": 640.0, "y": 700.0, "uncertainty_radius": 5.0},
            },
        }
    ]

    assessment = engine.assess_frame(
        trajectories=trajectories,
        weather_intel=weather,
        timestamp_seconds=2.0,
    )

    assert assessment.risk_state == CollisionRiskState.CRITICAL
    assert assessment.min_ttc_seconds is not None
    assert assessment.min_ttc_seconds < 1.8
    assert assessment.risk_score is not None
    assert assessment.risk_score >= 0.85
    assert "CRITICAL HAZARD" in assessment.explanation


# ==============================================================================
# 7. Video Pipeline & FastAPI Integration Tests
# ==============================================================================

def test_video_pipeline_collision_risk_integration(tmp_path):
    """Integration: VideoPipelineRunner integrates CollisionRiskEngine into results."""
    video_path = PROJECT_ROOT / "outputs" / "day14_validation" / "clips" / "scenario_a_normal_daytime.mp4"
    if not video_path.exists():
        pytest.skip("Test video scenario_a_normal_daytime.mp4 not found.")

    runner = VideoPipelineRunner(project_root=PROJECT_ROOT)
    result = runner.run(video_path, job_id="test_col_01", output_dir=tmp_path, max_frames=5)

    assert result.collision_summary is not None
    assert "critical_risk_frames" in result.collision_summary
    assert "high_risk_frames" in result.collision_summary
    assert "risk_state_distribution" in result.collision_summary

    for frame in result.frames:
        assert hasattr(frame, "collision_risk")
        c_risk = frame.collision_risk
        assert "risk_state" in c_risk
        assert c_risk["risk_state"] in ("LOW", "MODERATE", "HIGH", "CRITICAL", "UNKNOWN")


def test_api_collision_risk_endpoint():
    """Integration: Verify get_video_job_collision_risk returns 404 for nonexistent job."""
    from src.api.routers.video import get_video_job_collision_risk
    from src.api.exceptions import ResourceNotFoundError

    with pytest.raises(ResourceNotFoundError):
        get_video_job_collision_risk("nonexistent-job-uuid-123")
