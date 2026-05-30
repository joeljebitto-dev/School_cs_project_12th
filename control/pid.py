"""PID controller and history buffers for the robot demo."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import INTEGRAL_LIMIT, PID_PLOT_MAX_POINTS, TORQUE_LIMIT
from kinematics.common import wrap_angles


@dataclass
class PIDController:
    """Simple joint-space PID controller."""

    kp: float
    ki: float
    kd: float
    torque_limit: float = TORQUE_LIMIT
    integral_limit: float = INTEGRAL_LIMIT
    integral_error: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def reset(self) -> None:
        """Clear the stored integral error."""

        self.integral_error = np.zeros(3)

    def compute(
        self,
        target_angles: np.ndarray,
        current_angles: np.ndarray,
        current_velocities: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Return torque commands for the three joints."""

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        target_angles = np.asarray(target_angles, dtype=float)
        current_angles = np.asarray(current_angles, dtype=float)
        current_velocities = np.asarray(current_velocities, dtype=float)

        error = wrap_angles(target_angles - current_angles)
        self.integral_error += error * dt
        self.integral_error = np.clip(
            self.integral_error,
            -self.integral_limit,
            self.integral_limit,
        )

        derivative = -current_velocities
        torque = self.kp * error + self.ki * self.integral_error + self.kd * derivative
        return np.clip(torque, -self.torque_limit, self.torque_limit)


@dataclass
class PIDHistory:
    """Small rolling buffer used by the PID response plot."""

    max_points: int = PID_PLOT_MAX_POINTS
    time_history: list[float] = field(default_factory=list)
    error_history: list[float] = field(default_factory=list)
    torque_history: list[float] = field(default_factory=list)
    plot_time: float = 0.0

    def record(self, elapsed_time: float, error_norm: float, torque_norm: float) -> None:
        """Append one plotted PID sample and keep the buffer short."""

        self.plot_time += elapsed_time
        self.time_history.append(self.plot_time)
        self.error_history.append(error_norm)
        self.torque_history.append(torque_norm)

        self.time_history = self.time_history[-self.max_points :]
        self.error_history = self.error_history[-self.max_points :]
        self.torque_history = self.torque_history[-self.max_points :]

    def reset(self) -> None:
        """Clear all plotted PID samples."""

        self.time_history.clear()
        self.error_history.clear()
        self.torque_history.clear()
        self.plot_time = 0.0
