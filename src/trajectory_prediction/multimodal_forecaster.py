"""Multimodal Trajectory Forecaster with Dynamic Lookahead Horizons (H1, H3, H5).

Implements:
1. Multi-horizon trajectory generation for 1.0s, 3.0s, and 5.0s actual time horizons
2. Multimodal hypothesis generation (Primary continuation, Left maneuver, Right maneuver, Decelerating stop)
3. Uncertainty scaling influenced by Camera Visibility (Feature 1) and Track Stability (Feature 2)
4. Evaluation engine for Average Displacement Error (ADE) and Final Displacement Error (FDE)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.trajectory_prediction.kalman_trajectory_predictor import KalmanTrajectoryPredictor
from src.trajectory_prediction.kinematics_engine import KinematicsEngine
from src.trajectory_prediction.interaction_model import MultiAgentInteractionModel
from src.trajectory_prediction.schemas import (
    CoordinateFrame,
    InteractionType,
    MotionState,
    ObjectTrajectoryPrediction,
    PredictionMode,
    PredictionStatus,
    TrajectoryHypothesis,
    TrajectoryStreamSummary,
    TrajectoryWaypoint,
)


class MultimodalTrajectoryForecaster:
    """Production multi-agent trajectory prediction system."""

    def __init__(
        self,
        kalman_predictor: Optional[KalmanTrajectoryPredictor] = None,
        kinematics_engine: Optional[KinematicsEngine] = None,
        interaction_model: Optional[MultiAgentInteractionModel] = None,
    ):
        self.kalman = kalman_predictor or KalmanTrajectoryPredictor()
        self.kinematics = kinematics_engine or KinematicsEngine()
        self.interactions = interaction_model or MultiAgentInteractionModel()

    def generate_predictions(
        self,
        active_tracks: List[Dict[str, Any]],
        track_histories: Dict[int, List[Dict[str, Any]]],
        fps: float,
        frame_width: int,
        frame_height: int,
        camera_visibility_score: float = 100.0,
    ) -> List[ObjectTrajectoryPrediction]:
        """Generate multimodal trajectory predictions across 1.0s, 3.0s, and 5.0s horizons for all active tracks."""
        fps = max(1.0, fps)
        predictions: List[ObjectTrajectoryPrediction] = []

        # 1. First pass: compute kinematics for all active tracks to enable interaction modeling
        track_kinematics: Dict[int, Dict[str, Any]] = {}
        for trk in active_tracks:
            tid = trk["track_id"]
            hist = track_histories.get(tid, [])
            kin = self.kinematics.compute_motion_features(hist, fps, frame_width, frame_height)
            track_kinematics[tid] = kin
            trk["velocity"] = kin["velocity"]
            trk["speed_kmh"] = kin["speed_kmh"]

        # 2. Second pass: multimodal prediction and horizon sampling
        for trk in active_tracks:
            tid = trk["track_id"]
            hist = track_histories.get(tid, [])
            kin = track_kinematics[tid]
            cx, cy = trk["center"]
            cls_name = trk.get("class_name", "vehicle")
            conf = float(trk.get("confidence", 0.85))

            # Check history sufficiency
            if len(hist) < 2:
                pred_status = PredictionStatus.INSUFFICIENT_HISTORY
                uncert_factor = 0.50
            elif camera_visibility_score < 55.0:
                pred_status = PredictionStatus.DEGRADED_PERCEPTION
                uncert_factor = 0.35
            else:
                pred_status = PredictionStatus.VALID
                uncert_factor = 0.15

            # Multi-agent interaction
            interaction = self.interactions.analyze_interactions(trk, active_tracks, fps)

            # Generate primary Kalman filtered trajectory up to 5.0 seconds
            # Total steps for 5.0 seconds lookahead
            total_steps = max(5, int(round(5.0 * fps)))
            primary_waypoints = self.kalman.predict_trajectory(
                current_center=(cx, cy),
                velocity=kin["velocity"],
                acceleration=kin["acceleration"],
                fps=fps,
                lookahead_seconds=5.0,
                steps=total_steps,
                speed_kmh=kin["speed_kmh"],
            )

            # Add metric ground coordinates to waypoints
            for wp in primary_waypoints:
                xm, ym = self.kinematics.project_to_ground_plane(wp.x_px, wp.y_px, frame_width, frame_height)
                wp.x_m = xm
                wp.y_m = ym
                # Inflate uncertainty if camera visibility is degraded
                vis_penalty = max(0.0, (100.0 - camera_visibility_score) / 100.0)
                wp.uncertainty_radius_px = round(wp.uncertainty_radius_px * (1.0 + vis_penalty * 0.8), 1)

            # Extract explicit time horizons: H1 (1.0s), H3 (3.0s), H5 (5.0s)
            h1_idx = min(len(primary_waypoints) - 1, max(0, int(round(1.0 * fps)) - 1))
            h3_idx = min(len(primary_waypoints) - 1, max(0, int(round(3.0 * fps)) - 1))
            h5_idx = min(len(primary_waypoints) - 1, max(0, int(round(5.0 * fps)) - 1))

            horizons = {
                "h1": primary_waypoints[h1_idx],
                "h3": primary_waypoints[h3_idx],
                "h5": primary_waypoints[h5_idx],
            }

            # 3. Generate Multimodal Hypotheses:
            # Mode A: Primary Continuation
            # Mode B: Left Maneuver (lateral curving divergence)
            # Mode C: Right Maneuver (lateral curving divergence)
            # Mode D: Decelerating Stop (if following lead car or stopping)
            hypotheses: List[TrajectoryHypothesis] = []

            # Mode A: Primary
            hypotheses.append(
                TrajectoryHypothesis(
                    mode=PredictionMode.PRIMARY_CONTINUATION,
                    mode_probability=0.70,
                    waypoints=primary_waypoints,
                    intent_description="Continuous path along primary kinematic velocity vector.",
                )
            )

            # Compute lateral normal vector for turn maneuvers
            vx, vy = kin["velocity"]["vx"], kin["velocity"]["vy"]
            speed = math.hypot(vx, vy)
            if speed > 3.0:
                nx = -vy / speed
                ny = vx / speed
            else:
                nx, ny = 1.0, 0.0

            # Mode B: Left Maneuver
            left_waypoints: List[TrajectoryWaypoint] = []
            for wp in primary_waypoints:
                t = wp.horizon_seconds
                offset = 4.0 * (t**1.4)  # Quadratic lateral drift
                lx = wp.x_px + nx * offset
                ly = wp.y_px + ny * offset
                left_waypoints.append(
                    TrajectoryWaypoint(
                        step_index=wp.step_index,
                        horizon_seconds=wp.horizon_seconds,
                        x_px=float(lx),
                        y_px=float(ly),
                        uncertainty_radius_px=wp.uncertainty_radius_px * 1.3,
                        velocity_px_s=wp.velocity_px_s,
                    )
                )

            # Mode C: Right Maneuver
            right_waypoints: List[TrajectoryWaypoint] = []
            for wp in primary_waypoints:
                t = wp.horizon_seconds
                offset = 4.0 * (t**1.4)
                rx = wp.x_px - nx * offset
                ry = wp.y_px - ny * offset
                right_waypoints.append(
                    TrajectoryWaypoint(
                        step_index=wp.step_index,
                        horizon_seconds=wp.horizon_seconds,
                        x_px=float(rx),
                        y_px=float(ry),
                        uncertainty_radius_px=wp.uncertainty_radius_px * 1.3,
                        velocity_px_s=wp.velocity_px_s,
                    )
                )

            prob_left = 0.15
            prob_right = 0.15
            if kin["motion_state"] == MotionState.TURNING_LEFT:
                prob_left = 0.35
            elif kin["motion_state"] == MotionState.TURNING_RIGHT:
                prob_right = 0.35

            hypotheses.append(
                TrajectoryHypothesis(
                    mode=PredictionMode.LEFT_MANEUVER,
                    mode_probability=prob_left,
                    waypoints=left_waypoints,
                    intent_description="Leftward path divergence / lane change.",
                )
            )
            hypotheses.append(
                TrajectoryHypothesis(
                    mode=PredictionMode.RIGHT_MANEUVER,
                    mode_probability=prob_right,
                    waypoints=right_waypoints,
                    intent_description="Rightward path divergence / lane change.",
                )
            )

            # If following another car or decelerating, add Decelerating Stop mode
            if interaction.interaction_type == InteractionType.CAR_FOLLOWING or kin["motion_state"] == MotionState.DECELERATING:
                stop_waypoints: List[TrajectoryWaypoint] = []
                for wp in primary_waypoints:
                    t = wp.horizon_seconds
                    decay = math.exp(-0.8 * t)
                    sx = cx + (wp.x_px - cx) * decay
                    sy = cy + (wp.y_px - cy) * decay
                    stop_waypoints.append(
                        TrajectoryWaypoint(
                            step_index=wp.step_index,
                            horizon_seconds=wp.horizon_seconds,
                            x_px=float(sx),
                            y_px=float(sy),
                            uncertainty_radius_px=wp.uncertainty_radius_px * 1.1,
                            velocity_px_s=(wp.velocity_px_s[0] * decay, wp.velocity_px_s[1] * decay),
                        )
                    )
                hypotheses.append(
                    TrajectoryHypothesis(
                        mode=PredictionMode.DECELERATING_STOP,
                        mode_probability=0.30,
                        waypoints=stop_waypoints,
                        intent_description="Decelerating to a stop behind lead vehicle.",
                    )
                )

            pred_record = ObjectTrajectoryPrediction(
                track_id=tid,
                object_class=cls_name,
                confidence=conf,
                current_bbox=trk.get("bbox", [cx - 20, cy - 20, cx + 20, cy + 20]),
                current_center=[cx, cy],
                motion_state=kin["motion_state"],
                speed_kmh=kin["speed_kmh"],
                velocity=kin["velocity"],
                acceleration=kin["acceleration"],
                heading_degrees=kin["heading_degrees"],
                track_age_frames=len(hist),
                history_length=len(hist),
                observed_history=hist[-15:],
                primary_trajectory=primary_waypoints,
                alternative_hypotheses=hypotheses,
                horizons=horizons,
                interaction=interaction,
                uncertainty_factor=uncert_factor,
                prediction_status=pred_status,
                coordinate_frame=CoordinateFrame.IMAGE_PIXEL,
            )
            predictions.append(pred_record)

        return predictions

    def evaluate_displacement_errors(
        self,
        predicted_waypoints: List[TrajectoryWaypoint],
        ground_truth_future_points: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Compute Average Displacement Error (ADE) and Final Displacement Error (FDE).

        ADE: Mean Euclidean distance across all matched future steps.
        FDE: Euclidean distance at the final horizon step.
        """
        if not predicted_waypoints or not ground_truth_future_points:
            return 0.0, 0.0

        n = min(len(predicted_waypoints), len(ground_truth_future_points))
        if n == 0:
            return 0.0, 0.0

        errors = []
        for i in range(n):
            wp = predicted_waypoints[i]
            gt_x, gt_y = ground_truth_future_points[i]
            dist = math.hypot(wp.x_px - gt_x, wp.y_px - gt_y)
            errors.append(dist)

        ade = float(np.mean(errors))
        fde = float(errors[-1])
        return round(ade, 2), round(fde, 2)
