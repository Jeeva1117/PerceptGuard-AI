"""Comprehensive unit and integration test suite for Object & Trajectory Prediction (Feature 3)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List
import pytest
from src.trajectory_prediction import (
    CoordinateFrame,
    InteractionType,
    KinematicsEngine,
    KalmanTrajectoryPredictor,
    MotionState,
    MultiAgentInteraction,
    MultiAgentInteractionModel,
    MultimodalTrajectoryForecaster,
    ObjectTrajectoryPrediction,
    PredictionMode,
    PredictionStatus,
    TrajectoryHypothesis,
    TrajectoryWaypoint,
)
from src.video_processing.video_pipeline_runner import VideoPipelineRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# 1. Kinematics Engine Unit Tests
# ==============================================================================

def test_kinematics_velocity_and_acceleration_calculation():
    """Verify velocity and acceleration calculation using true delta time between frames."""
    engine = KinematicsEngine()
    fps = 10.0
    width, height = 1280, 720

    # Synthetic observations with constant speed of 100 px/s forward (+y)
    history = [
        {"frame": 1, "timestamp": 0.0, "x": 640.0, "y": 400.0},
        {"frame": 2, "timestamp": 0.1, "x": 640.0, "y": 410.0},
        {"frame": 3, "timestamp": 0.2, "x": 640.0, "y": 420.0},
    ]

    features = engine.compute_motion_features(history, fps, width, height)
    vel = features["velocity"]
    acc = features["acceleration"]

    assert abs(vel["vx"]) < 1.0  # Zero lateral motion
    assert 95.0 <= vel["vy"] <= 105.0  # Approx 100 px/s forward
    assert abs(acc["ax"]) < 5.0
    assert abs(acc["ay"]) < 15.0  # Minimal acceleration
    assert features["speed_kmh"] > 0.0
    assert features["motion_state"] == MotionState.CRUISING


def test_pinhole_ground_plane_projection():
    """Verify flat-earth pinhole camera metric projection."""
    engine = KinematicsEngine(camera_height_meters=1.35, horizon_fraction=0.45)
    width, height = 1280, 720
    v0 = height * 0.45  # Horizon line at y = 324

    # Point 1: At or above the horizon line cannot project to ground ahead
    x_sky, y_sky = engine.project_to_ground_plane(640, 200, width, height)
    assert x_sky is None
    assert y_sky is None

    # Point 2: Near bottom of image (close to vehicle, cy = 680)
    x_near, y_near = engine.project_to_ground_plane(640, 680, width, height)
    assert x_near is not None and y_near is not None
    assert abs(x_near) < 0.2  # Centered column -> near 0 lateral offset
    assert 1.0 < y_near < 5.0  # Close forward distance in meters

    # Point 3: Mid-distance (cy = 400)
    x_far, y_far = engine.project_to_ground_plane(640, 400, width, height)
    assert y_far is not None
    assert y_far > y_near  # Farther distance than near point


def test_motion_state_classification():
    """Verify classification of CRUISING, ACCELERATING, DECELERATING, STATIONARY, and TURNING."""
    engine = KinematicsEngine()
    fps = 10.0
    w, h = 1280, 720

    # 1. Stationary target
    stat_hist = [
        {"frame": 1, "timestamp": 0.0, "x": 500.0, "y": 500.0},
        {"frame": 2, "timestamp": 0.1, "x": 500.2, "y": 500.1},
        {"frame": 3, "timestamp": 0.2, "x": 500.1, "y": 500.2},
    ]
    feat_stat = engine.compute_motion_features(stat_hist, fps, w, h)
    assert feat_stat["motion_state"] == MotionState.STATIONARY

    # 2. Accelerating target
    accel_hist = [
        {"frame": 1, "timestamp": 0.0, "x": 500.0, "y": 400.0},
        {"frame": 2, "timestamp": 0.1, "x": 500.0, "y": 405.0},  # vy = 50
        {"frame": 3, "timestamp": 0.2, "x": 500.0, "y": 420.0},  # vy = 150
    ]
    feat_accel = engine.compute_motion_features(accel_hist, fps, w, h)
    assert feat_accel["motion_state"] == MotionState.ACCELERATING

    # 3. Decelerating target
    decel_hist = [
        {"frame": 1, "timestamp": 0.0, "x": 500.0, "y": 400.0},
        {"frame": 2, "timestamp": 0.1, "x": 500.0, "y": 415.0},  # vy = 150
        {"frame": 3, "timestamp": 0.2, "x": 500.0, "y": 420.0},  # vy = 50
    ]
    feat_decel = engine.compute_motion_features(decel_hist, fps, w, h)
    assert feat_decel["motion_state"] == MotionState.DECELERATING

    # 4. Turning target
    turn_hist = [
        {"frame": 1, "timestamp": 0.0, "x": 500.0, "y": 400.0},
        {"frame": 2, "timestamp": 0.1, "x": 505.0, "y": 410.0},
        {"frame": 3, "timestamp": 0.2, "x": 520.0, "y": 415.0},
    ]
    feat_turn = engine.compute_motion_features(turn_hist, fps, w, h)
    assert feat_turn["motion_state"] in (MotionState.TURNING_RIGHT, MotionState.TURNING_LEFT, MotionState.CRUISING)


# ==============================================================================
# 2. Kalman Trajectory Predictor Unit Tests
# ==============================================================================

def test_kalman_trajectory_predictor_forward_recursion():
    """Verify Kalman trajectory forward extrapolation with acceleration damping."""
    predictor = KalmanTrajectoryPredictor(damping_factor=0.90)
    current_center = (600.0, 500.0)
    velocity = {"vx": 20.0, "vy": 80.0}
    acceleration = {"ax": 0.0, "ay": -5.0}
    fps = 10.0
    lookahead = 2.0  # 20 steps

    waypoints = predictor.predict_trajectory(
        current_center=current_center,
        velocity=velocity,
        acceleration=acceleration,
        fps=fps,
        lookahead_seconds=lookahead,
        speed_kmh=35.0,
    )

    assert len(waypoints) == 20
    assert waypoints[0].step_index == 1
    assert waypoints[0].horizon_seconds == pytest.approx(0.1, abs=0.01)
    assert waypoints[-1].step_index == 20
    assert waypoints[-1].horizon_seconds == pytest.approx(2.0, abs=0.01)

    # Monotonic progression forward
    for i in range(1, len(waypoints)):
        assert waypoints[i].y_px > waypoints[i - 1].y_px
        assert waypoints[i].x_px > waypoints[i - 1].x_px


def test_kalman_uncertainty_expansion():
    """Verify that spatial covariance and uncertainty radius expand monotonically into future horizons."""
    predictor = KalmanTrajectoryPredictor()
    current_center = (500.0, 500.0)
    velocity = {"vx": 0.0, "vy": 50.0}
    acceleration = {"ax": 0.0, "ay": 0.0}
    fps = 10.0

    waypoints = predictor.predict_trajectory(
        current_center=current_center,
        velocity=velocity,
        acceleration=acceleration,
        fps=fps,
        lookahead_seconds=3.0,
        speed_kmh=25.0,
    )

    # Check uncertainty radius monotonically grows over time
    for i in range(1, len(waypoints)):
        assert waypoints[i].uncertainty_radius_px >= waypoints[i - 1].uncertainty_radius_px


# ==============================================================================
# 3. Multi-Agent Interaction Model Unit Tests
# ==============================================================================

def test_multi_agent_interaction_car_following():
    """Verify car-following detection when lead vehicle is in the forward path corridor."""
    model = MultiAgentInteractionModel()
    fps = 10.0

    # Target vehicle following behind
    target_track = {
        "track_id": 1,
        "class_name": "car",
        "center": [600.0, 500.0],
        "velocity": {"vx": 0.0, "vy": -60.0},
        "speed_kmh": 40.0,
    }

    # Lead vehicle directly ahead
    lead_track = {
        "track_id": 2,
        "class_name": "truck",
        "center": [600.0, 420.0],
        "velocity": {"vx": 0.0, "vy": -55.0},
        "speed_kmh": 36.0,
    }

    interaction = model.analyze_interactions(target_track, [target_track, lead_track], fps)
    assert interaction.interaction_type == InteractionType.CAR_FOLLOWING
    assert interaction.nearest_agent_id == 2
    assert interaction.nearest_agent_distance_px == pytest.approx(80.0, abs=1.0)
    assert interaction.time_to_headway_seconds is not None


def test_multi_agent_interaction_crossing_path():
    """Verify crossing-path interaction when adjacent vehicle has perpendicular heading."""
    model = MultiAgentInteractionModel()
    fps = 10.0

    target = {
        "track_id": 10,
        "class_name": "car",
        "center": [500.0, 500.0],
        "velocity": {"vx": 0.0, "vy": -50.0},
        "speed_kmh": 30.0,
    }

    crossing = {
        "track_id": 11,
        "class_name": "pedestrian",
        "center": [550.0, 500.0],
        "velocity": {"vx": -30.0, "vy": 0.0},
        "speed_kmh": 6.0,
    }

    interaction = model.analyze_interactions(target, [target, crossing], fps)
    assert interaction.interaction_type == InteractionType.CROSSING_PATH
    assert interaction.nearest_agent_id == 11


def test_multi_agent_interaction_free_flow():
    """Verify free-flow classification when no neighboring obstacles exist."""
    model = MultiAgentInteractionModel()
    fps = 10.0

    solo = {
        "track_id": 99,
        "class_name": "car",
        "center": [640.0, 500.0],
        "velocity": {"vx": 0.0, "vy": -40.0},
        "speed_kmh": 30.0,
    }

    interaction = model.analyze_interactions(solo, [solo], fps)
    assert interaction.interaction_type == InteractionType.FREE_FLOW
    assert interaction.nearest_agent_id is None


# ==============================================================================
# 4. Multimodal Forecaster & Time Horizons Unit Tests
# ==============================================================================

def test_time_horizon_step_calculation_at_different_fps():
    """Verify that physical time horizons (1.0s, 3.0s, 5.0s) scale correctly with FPS."""
    forecaster = MultimodalTrajectoryForecaster()

    # 10 FPS test
    fps_10 = 10.0
    tracks_10 = [
        {"track_id": 1, "center": [640.0, 500.0], "class_name": "car", "confidence": 0.9}
    ]
    hist_10 = {
        1: [
            {"frame": 1, "timestamp": 0.0, "x": 640.0, "y": 520.0},
            {"frame": 2, "timestamp": 0.1, "x": 640.0, "y": 510.0},
            {"frame": 3, "timestamp": 0.2, "x": 640.0, "y": 500.0},
        ]
    }
    preds_10 = forecaster.generate_predictions(tracks_10, hist_10, fps_10, 1280, 720)
    assert len(preds_10) == 1
    h_10 = preds_10[0].horizons
    assert "h1" in h_10 and "h3" in h_10 and "h5" in h_10
    assert h_10["h1"].step_index == 10  # 1.0s * 10fps
    assert h_10["h3"].step_index == 30  # 3.0s * 10fps
    assert h_10["h5"].step_index == 50  # 5.0s * 10fps

    # 30 FPS test
    fps_30 = 30.0
    tracks_30 = [
        {"track_id": 2, "center": [640.0, 500.0], "class_name": "car", "confidence": 0.9}
    ]
    hist_30 = {
        2: [
            {"frame": 1, "timestamp": 0.0, "x": 640.0, "y": 506.0},
            {"frame": 2, "timestamp": 0.033, "x": 640.0, "y": 503.0},
            {"frame": 3, "timestamp": 0.066, "x": 640.0, "y": 500.0},
        ]
    }
    preds_30 = forecaster.generate_predictions(tracks_30, hist_30, fps_30, 1280, 720)
    assert len(preds_30) == 1
    h_30 = preds_30[0].horizons
    assert h_30["h1"].step_index == 30  # 1.0s * 30fps
    assert h_30["h3"].step_index == 90  # 3.0s * 30fps
    assert h_30["h5"].step_index == 150  # 5.0s * 30fps


def test_multimodal_hypotheses_generation():
    """Verify generation of multimodal trajectory hypotheses summing to 1.0 probability."""
    forecaster = MultimodalTrajectoryForecaster()
    tracks = [
        {"track_id": 5, "center": [640.0, 500.0], "class_name": "car", "confidence": 0.92}
    ]
    history = {
        5: [
            {"frame": 1, "timestamp": 0.0, "x": 640.0, "y": 520.0},
            {"frame": 2, "timestamp": 0.1, "x": 640.0, "y": 510.0},
            {"frame": 3, "timestamp": 0.2, "x": 640.0, "y": 500.0},
        ]
    }

    preds = forecaster.generate_predictions(tracks, history, 10.0, 1280, 720)
    assert len(preds) == 1
    hyps = preds[0].alternative_hypotheses
    assert len(hyps) >= 3

    modes = {h.mode for h in hyps}
    assert PredictionMode.PRIMARY_CONTINUATION in modes
    assert PredictionMode.LEFT_MANEUVER in modes
    assert PredictionMode.RIGHT_MANEUVER in modes

    total_prob = sum(h.mode_probability for h in hyps)
    assert total_prob == pytest.approx(1.0, abs=0.01)


def test_ade_and_fde_evaluation():
    """Verify Average Displacement Error (ADE) and Final Displacement Error (FDE) computation."""
    forecaster = MultimodalTrajectoryForecaster()

    # Predicted waypoints: (10, 10), (20, 20), (30, 30)
    preds = [
        TrajectoryWaypoint(step_index=1, horizon_seconds=0.1, x_px=10.0, y_px=10.0, uncertainty_radius_px=2.0),
        TrajectoryWaypoint(step_index=2, horizon_seconds=0.2, x_px=20.0, y_px=20.0, uncertainty_radius_px=3.0),
        TrajectoryWaypoint(step_index=3, horizon_seconds=0.3, x_px=30.0, y_px=30.0, uncertainty_radius_px=4.0),
    ]

    # Ground truth: (13, 14) -> error 5.0; (20, 20) -> error 0.0; (33, 34) -> error 5.0
    gt_future = [(13.0, 14.0), (20.0, 20.0), (33.0, 34.0)]

    ade, fde = forecaster.evaluate_displacement_errors(preds, gt_future)
    expected_ade = (5.0 + 0.0 + 5.0) / 3.0
    expected_fde = 5.0

    assert ade == pytest.approx(expected_ade, abs=0.05)
    assert fde == pytest.approx(expected_fde, abs=0.05)


def test_insufficient_history_handling():
    """Verify graceful handling when track history has fewer than 2 frames."""
    forecaster = MultimodalTrajectoryForecaster()
    tracks = [{"track_id": 42, "center": [300.0, 400.0], "class_name": "car"}]
    history = {42: [{"frame": 1, "timestamp": 0.0, "x": 300.0, "y": 400.0}]}

    preds = forecaster.generate_predictions(tracks, history, 10.0, 1280, 720)
    assert len(preds) == 1
    assert preds[0].prediction_status == PredictionStatus.INSUFFICIENT_HISTORY
    assert preds[0].horizons["h1"].x_px == 300.0
    assert preds[0].horizons["h1"].y_px == 400.0


# ==============================================================================
# 5. Video Pipeline & API Integration Tests
# ==============================================================================

def test_video_pipeline_trajectory_integration(tmp_path):
    """Verify VideoPipelineRunner integrates multimodal trajectory predictions into results."""
    # Find a test video from outputs
    video_path = PROJECT_ROOT / "outputs" / "day14_validation" / "clips" / "scenario_a_normal_daytime.mp4"
    if not video_path.exists():
        pytest.skip("Test video scenario_a_normal_daytime.mp4 not found.")

    runner = VideoPipelineRunner(project_root=PROJECT_ROOT)
    result = runner.run(video_path, job_id="test_traj_01", output_dir=tmp_path, max_frames=5)

    assert result.trajectory_summary is not None
    assert "mean_speed_kmh" in result.trajectory_summary
    assert "max_speed_kmh" in result.trajectory_summary
    assert "total_observed_trajectories" in result.trajectory_summary

    # Verify frame-level trajectory output
    for frame in result.frames:
        assert hasattr(frame, "trajectories")
        if frame.trajectories:
            first_traj = frame.trajectories[0]
            assert "predicted_horizons" in first_traj
            assert "velocity" in first_traj
            assert "speed_kmh" in first_traj


def test_api_trajectory_endpoint():
    """Verify trajectory endpoint logic returns structured telemetry or raises ResourceNotFoundError for unknown jobs."""
    from src.api.routers.video import get_video_job_trajectories
    from src.api.exceptions import ResourceNotFoundError

    # Nonexistent job raises ResourceNotFoundError (which maps to HTTP 404 in FastAPI)
    with pytest.raises(ResourceNotFoundError):
        get_video_job_trajectories("nonexistent-job-uuid")
