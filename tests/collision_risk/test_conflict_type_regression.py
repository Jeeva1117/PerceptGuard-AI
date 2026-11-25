"""Regression test suite for ConflictType import and collision conflict handling.

Specifically reproduces and verifies the resolution of:
    Pipeline execution failed: name 'ConflictType' is not defined
Ensuring that CollisionRiskEngine, TrajectoryConflict, and the video pipeline
runner correctly assess, serialize, and report multi-horizon trajectory conflicts.
"""

from __future__ import annotations

import json
from src.collision_risk import (
    CollisionRiskEngine,
    CollisionRiskState,
    ConflictType,
    TrajectoryConflict,
    AssessmentStatus,
)
from src.collision_risk.collision_risk_engine import ConflictType as EngineConflictType


def test_conflict_type_imported_and_accessible_in_engine_module():
    """Verify ConflictType is exported and directly imported in collision_risk_engine."""
    assert EngineConflictType is not None
    assert EngineConflictType is ConflictType
    assert ConflictType.PEDESTRIAN_CROSSING == "PEDESTRIAN_CROSSING"
    assert ConflictType.CYCLIST_CONFLICT == "CYCLIST_CONFLICT"
    assert ConflictType.REAR_END == "REAR_END"
    assert ConflictType.CROSSING == "CROSSING"
    assert ConflictType.HEAD_ON == "HEAD_ON"


def test_assess_frame_with_pedestrian_conflict_does_not_raise_name_error():
    """Verify assess_frame with pedestrian crossing conflict executes without NameError.

    Original crash occurred when `conflicts` had items and line 168 checked:
    c.conflict_type in (ConflictType.PEDESTRIAN_CROSSING, ConflictType.CYCLIST_CONFLICT)
    """
    engine = CollisionRiskEngine(frame_width=1280, frame_height=720)

    # Ego/car moving forward along road corridor
    t_car = {
        "track_id": 1,
        "class_name": "car",
        "bbox": [580, 400, 700, 560],
        "center": [640.0, 480.0],
        "velocity": {"vx": 0.0, "vy": 30.0},
        "acceleration": {"ax": 0.0, "ay": 0.0},
        "speed_kmh": 45.0,
        "confidence": 0.88,
        "history_length": 6,
        "predicted_horizons": {
            "h1": {"x": 640.0, "y": 510.0, "uncertainty_radius": 10.0},
            "h3": {"x": 640.0, "y": 570.0, "uncertainty_radius": 20.0},
            "h5": {"x": 640.0, "y": 630.0, "uncertainty_radius": 30.0},
        },
    }

    # Pedestrian crossing directly into ego path at H3
    t_ped = {
        "track_id": 2,
        "class_name": "pedestrian",
        "bbox": [500, 540, 540, 600],
        "center": [520.0, 570.0],
        "velocity": {"vx": 40.0, "vy": 0.0},
        "acceleration": {"ax": 0.0, "ay": 0.0},
        "speed_kmh": 5.0,
        "confidence": 0.85,
        "history_length": 6,
        "predicted_horizons": {
            "h1": {"x": 560.0, "y": 570.0, "uncertainty_radius": 8.0},
            "h3": {"x": 640.0, "y": 570.0, "uncertainty_radius": 15.0},
            "h5": {"x": 720.0, "y": 570.0, "uncertainty_radius": 25.0},
        },
    }

    # This MUST NOT raise NameError: name 'ConflictType' is not defined
    assessment = engine.assess_frame(
        trajectories=[t_car, t_ped],
        weather_intel={"visibility_score": 85.0, "primary_condition": "CLEAR", "is_degraded": False},
        timestamp_seconds=2.5,
        fps=10.0,
    )

    assert assessment is not None
    assert assessment.collision_decision in ("POTENTIAL COLLISION RISK", "COLLISION OBSERVED")
    assert assessment.decision_reason is not None
    assert len(assessment.decision_reason) > 0

    # Verify serialization of conflicts
    d = assessment.to_dict()
    assert "conflicting_pairs" in d
    assert len(d["conflicting_pairs"]) > 0

    # Ensure JSON serializable without Enum error
    json_str = json.dumps(d)
    assert "PEDESTRIAN_CROSSING" in json_str or "CROSSING" in json_str


def test_assess_frame_with_cyclist_conflict():
    """Verify assess_frame with cyclist conflict executes smoothly."""
    engine = CollisionRiskEngine(frame_width=1280, frame_height=720)

    t_car = {
        "track_id": 10,
        "class_name": "car",
        "bbox": [580, 400, 700, 560],
        "center": [640.0, 480.0],
        "velocity": {"vx": 0.0, "vy": 25.0},
        "speed_kmh": 35.0,
        "confidence": 0.90,
        "history_length": 5,
        "predicted_horizons": {
            "h1": {"x": 640.0, "y": 505.0, "uncertainty_radius": 8.0},
            "h3": {"x": 640.0, "y": 555.0, "uncertainty_radius": 18.0},
        },
    }

    t_cyclist = {
        "track_id": 11,
        "class_name": "bicycle",
        "bbox": [620, 520, 660, 580],
        "center": [640.0, 550.0],
        "velocity": {"vx": 0.0, "vy": 5.0},
        "speed_kmh": 12.0,
        "confidence": 0.85,
        "history_length": 5,
        "predicted_horizons": {
            "h1": {"x": 640.0, "y": 555.0, "uncertainty_radius": 6.0},
            "h3": {"x": 640.0, "y": 565.0, "uncertainty_radius": 14.0},
        },
    }

    assessment = engine.assess_frame(
        trajectories=[t_car, t_cyclist],
        weather_intel={"visibility_score": 80.0, "primary_condition": "CLEAR", "is_degraded": False},
        timestamp_seconds=3.0,
        fps=10.0,
    )

    assert assessment is not None
    assert assessment.collision_decision in ("POTENTIAL COLLISION RISK", "COLLISION OBSERVED")

    # Verify stream summary aggregation
    summary = engine.generate_stream_summary()
    assert summary.total_conflicts_detected >= 1
    summary_dict = summary.to_dict()
    assert "conflict_type_distribution" in summary_dict
    json_summary = json.dumps(summary_dict)
    assert len(json_summary) > 0


def test_trajectory_conflict_to_dict_handles_string_and_enum():
    """Verify TrajectoryConflict handles both ConflictType enum and string values."""
    conflict_enum = TrajectoryConflict(
        track_a=1,
        track_b=2,
        class_a="car",
        class_b="motorcycle",
        conflict_type=ConflictType.HEAD_ON,
        time_to_conflict_seconds=1.2,
        horizon_label="H1",
        separation_at_conflict_px=15.0,
        separation_at_conflict_m=1.0,
    )
    d_enum = conflict_enum.to_dict()
    assert d_enum["conflict_type"] == "HEAD_ON"

    # Also string fallback
    conflict_str = TrajectoryConflict(
        track_a=3,
        track_b=4,
        class_a="car",
        class_b="bus",
        conflict_type="REAR_END",  # type: ignore
        time_to_conflict_seconds=2.0,
        horizon_label="H3",
        separation_at_conflict_px=30.0,
    )
    d_str = conflict_str.to_dict()
    assert d_str["conflict_type"] == "REAR_END"
