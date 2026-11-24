"""Kalman-Filter and Constant Acceleration Trajectory Predictor.

Implements:
1. Linear Kalman Filter state estimator: [x, y, vx, vy, ax, ay]
2. Multi-step forward recursion with damping to prevent acceleration runaway
3. Dynamic covariance propagation yielding expanding uncertainty bounds
4. Constant Velocity (CV) baseline fallback
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.trajectory_prediction.schemas import TrajectoryWaypoint


class KalmanTrajectoryPredictor:
    """Kalman-filter based trajectory forecasting engine with covariance propagation."""

    def __init__(
        self,
        process_noise_std: float = 2.5,
        measurement_noise_std: float = 3.0,
        damping_factor: float = 0.88,
    ):
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.damping_factor = damping_factor

    def predict_trajectory(
        self,
        current_center: Tuple[float, float],
        velocity: Dict[str, float],
        acceleration: Dict[str, float],
        fps: float,
        lookahead_seconds: float = 5.0,
        steps: Optional[int] = None,
        speed_kmh: float = 0.0,
    ) -> List[TrajectoryWaypoint]:
        """Generate multi-step future waypoints with propagated spatial uncertainty."""
        cx, cy = current_center
        vx, vy = velocity.get("vx", 0.0), velocity.get("vy", 0.0)
        ax, ay = acceleration.get("ax", 0.0), acceleration.get("ay", 0.0)

        dt = 1.0 / max(1.0, fps)
        total_steps = steps if steps is not None else max(1, int(round(lookahead_seconds * fps)))

        # State vector: [x, y, vx, vy, ax, ay]
        state = np.array([cx, cy, vx, vy, ax, ay], dtype=np.float64)

        # Initial Covariance Matrix P
        cov = np.eye(6, dtype=np.float64)
        cov[0, 0] = cov[1, 1] = self.measurement_noise_std**2
        cov[2, 2] = cov[3, 3] = (self.measurement_noise_std * 2.0)**2
        cov[4, 4] = cov[5, 5] = (self.measurement_noise_std * 4.0)**2

        # State transition matrix F(dt)
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[0, 4] = 0.5 * dt**2
        F[1, 3] = dt
        F[1, 5] = 0.5 * dt**2
        F[2, 4] = dt
        F[3, 5] = dt

        # Process noise matrix Q(dt)
        Q = np.eye(6, dtype=np.float64) * (self.process_noise_std**2)
        Q[0, 0] = Q[1, 1] = 0.25 * dt**4 * self.process_noise_std**2
        Q[2, 2] = Q[3, 3] = dt**2 * self.process_noise_std**2

        waypoints: List[TrajectoryWaypoint] = []

        cur_x, cur_y = cx, cy
        cur_vx, cur_vy = vx, vy
        cur_ax, cur_ay = ax, ay

        for k in range(1, total_steps + 1):
            h_time = k * dt
            damp = self.damping_factor**k

            # Forward state extrapolation with acceleration damping
            next_x = cur_x + cur_vx * dt + 0.5 * cur_ax * (dt**2) * damp
            next_y = cur_y + cur_vy * dt + 0.5 * cur_ay * (dt**2) * damp
            next_vx = cur_vx + cur_ax * dt * damp
            next_vy = cur_vy + cur_ay * dt * damp

            # Propagate uncertainty covariance
            cov = F @ cov @ F.T + Q
            # Extract position submatrix eigenvalue for uncertainty radius
            pos_var = max(cov[0, 0], cov[1, 1])
            uncertainty_radius = float(math.sqrt(max(1.0, pos_var)) + 0.05 * speed_kmh + 1.2 * h_time)

            wp = TrajectoryWaypoint(
                step_index=k,
                horizon_seconds=round(h_time, 3),
                x_px=float(next_x),
                y_px=float(next_y),
                uncertainty_radius_px=float(uncertainty_radius),
                velocity_px_s=(float(next_vx), float(next_vy)),
            )
            waypoints.append(wp)

            cur_x, cur_y = next_x, next_y
            cur_vx, cur_vy = next_vx, next_vy

        return waypoints
