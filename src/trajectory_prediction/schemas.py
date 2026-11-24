"""Data schemas and interfaces for Object & Trajectory Prediction (Feature 3).

Defines structured representations for:
- Coordinate frames (Image Pixel vs Metric Ground Plane)
- Motion states (Stationary, Cruising, Accelerating, Decelerating, Turning)
- Multi-horizon trajectory waypoints (H1: 1.0s, H3: 3.0s, H5: 5.0s)
- Multimodal trajectory hypotheses with mode probabilities
- Multi-agent spatial interaction and following telemetry
- ADE (Average Displacement Error) and FDE (Final Displacement Error) validation metrics
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CoordinateFrame(str, Enum):
    IMAGE_PIXEL = "IMAGE_PIXEL"  # [x_px, y_px] in camera image coordinate space
    APPROX_GROUND_PLANE_METRIC = "APPROX_GROUND_PLANE_METRIC"  # [X_m, Y_m] flat ground projection


class MotionState(str, Enum):
    STATIONARY = "STATIONARY"
    CRUISING = "CRUISING"
    ACCELERATING = "ACCELERATING"
    DECELERATING = "DECELERATING"
    TURNING_LEFT = "TURNING_LEFT"
    TURNING_RIGHT = "TURNING_RIGHT"
    UNSTABLE = "UNSTABLE"


class PredictionMode(str, Enum):
    PRIMARY_CONTINUATION = "PRIMARY_CONTINUATION"
    LEFT_MANEUVER = "LEFT_MANEUVER"
    RIGHT_MANEUVER = "RIGHT_MANEUVER"
    DECELERATING_STOP = "DECELERATING_STOP"


class InteractionType(str, Enum):
    FREE_FLOW = "FREE_FLOW"
    CAR_FOLLOWING = "CAR_FOLLOWING"
    CROSSING_PATH = "CROSSING_PATH"
    APPROACHING_OPPOSITE = "APPROACHING_OPPOSITE"


class PredictionStatus(str, Enum):
    VALID = "VALID"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    DEGRADED_PERCEPTION = "DEGRADED_PERCEPTION"
    OCCLUDED_TRACK = "OCCLUDED_TRACK"


@dataclass
class TrajectoryWaypoint:
    """A single spatial waypoint along a predicted trajectory."""
    step_index: int
    horizon_seconds: float
    x_px: float
    y_px: float
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    uncertainty_radius_px: float = 8.0
    velocity_px_s: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "horizon_seconds": round(self.horizon_seconds, 3),
            "x": round(self.x_px, 1),
            "y": round(self.y_px, 1),
            "x_px": round(self.x_px, 1),
            "y_px": round(self.y_px, 1),
            "x_m": round(self.x_m, 2) if self.x_m is not None else None,
            "y_m": round(self.y_m, 2) if self.y_m is not None else None,
            "uncertainty_radius": round(self.uncertainty_radius_px, 1),
            "uncertainty_radius_px": round(self.uncertainty_radius_px, 1),
            "velocity_px_s": [round(self.velocity_px_s[0], 1), round(self.velocity_px_s[1], 1)],
        }


@dataclass
class TrajectoryHypothesis:
    """A multimodal trajectory hypothesis with associated mode probability."""
    mode: PredictionMode
    mode_probability: float
    waypoints: List[TrajectoryWaypoint]
    intent_description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "mode_probability": round(self.mode_probability, 3),
            "intent_description": self.intent_description,
            "waypoints": [w.to_dict() for w in self.waypoints],
        }


@dataclass
class MultiAgentInteraction:
    """Spatial and kinematic relationship with neighboring road agents."""
    interaction_type: InteractionType
    nearest_agent_id: Optional[int] = None
    nearest_agent_distance_px: Optional[float] = None
    nearest_agent_distance_m: Optional[float] = None
    relative_speed_kmh: Optional[float] = None
    time_to_headway_seconds: Optional[float] = None
    interaction_notes: str = "Free flow unobstructed motion."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_type": self.interaction_type.value,
            "nearest_agent_id": self.nearest_agent_id,
            "nearest_agent_distance_px": round(self.nearest_agent_distance_px, 1) if self.nearest_agent_distance_px is not None else None,
            "nearest_agent_distance_m": round(self.nearest_agent_distance_m, 1) if self.nearest_agent_distance_m is not None else None,
            "relative_speed_kmh": round(self.relative_speed_kmh, 1) if self.relative_speed_kmh is not None else None,
            "time_to_headway_seconds": round(self.time_to_headway_seconds, 2) if self.time_to_headway_seconds is not None else None,
            "interaction_notes": self.interaction_notes,
        }


@dataclass
class ObjectTrajectoryPrediction:
    """Comprehensive trajectory and motion forecasting record for a single tracked object."""
    track_id: int
    object_class: str
    confidence: float
    current_bbox: List[float]
    current_center: List[float]
    motion_state: MotionState
    speed_kmh: float
    velocity: Dict[str, float]  # vx, vy in px/s
    acceleration: Dict[str, float]  # ax, ay in px/s^2
    heading_degrees: float
    track_age_frames: int
    history_length: int
    observed_history: List[Dict[str, Any]]
    primary_trajectory: List[TrajectoryWaypoint]
    alternative_hypotheses: List[TrajectoryHypothesis]
    horizons: Dict[str, TrajectoryWaypoint]  # "h1", "h3", "h5"
    interaction: MultiAgentInteraction
    uncertainty_factor: float
    prediction_status: PredictionStatus
    coordinate_frame: CoordinateFrame = CoordinateFrame.IMAGE_PIXEL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.object_class,
            "object_class": self.object_class,
            "confidence": round(self.confidence, 3),
            "bbox": [round(c, 1) for c in self.current_bbox],
            "center": [round(c, 1) for c in self.current_center],
            "motion_state": self.motion_state.value,
            "speed_kmh": round(self.speed_kmh, 1),
            "velocity": {k: round(v, 2) for k, v in self.velocity.items()},
            "acceleration": {k: round(v, 2) for k, v in self.acceleration.items()},
            "heading_degrees": round(self.heading_degrees, 1),
            "track_age_frames": self.track_age_frames,
            "history_length": self.history_length,
            "history": self.observed_history,
            "observed_history": self.observed_history,
            "primary_trajectory": [w.to_dict() for w in self.primary_trajectory],
            "alternative_hypotheses": [h.to_dict() for h in self.alternative_hypotheses],
            "predicted_horizons": {k: v.to_dict() for k, v in self.horizons.items()},
            "horizons": {k: v.to_dict() for k, v in self.horizons.items()},
            "interaction": self.interaction.to_dict(),
            "uncertainty_factor": round(self.uncertainty_factor, 3),
            "prediction_status": self.prediction_status.value,
            "coordinate_frame": self.coordinate_frame.value,
        }


@dataclass
class TrajectoryStreamSummary:
    """Summary of motion kinematics and trajectory forecasting performance across an entire video stream."""
    total_observed_trajectories: int
    active_trajectories_count: int
    max_speed_kmh: float
    mean_speed_kmh: float
    motion_state_distribution: Dict[str, int]
    mean_ade_px: float
    mean_fde_px: float
    coordinate_system: str = "IMAGE_PIXEL (with Approx Ground Plane Metric Projection)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_observed_trajectories": self.total_observed_trajectories,
            "active_trajectories_count": self.active_trajectories_count,
            "max_speed_kmh": round(self.max_speed_kmh, 1),
            "mean_speed_kmh": round(self.mean_speed_kmh, 1),
            "motion_state_distribution": self.motion_state_distribution,
            "mean_ade_px": round(self.mean_ade_px, 1),
            "mean_fde_px": round(self.mean_fde_px, 1),
            "coordinate_system": self.coordinate_system,
        }
