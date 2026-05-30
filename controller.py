"""PID controller for the 3-DOF robot joints."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from constants import INTEGRAL_LIMIT, TORQUE_LIMIT
from robot_math import wrap_angles


@dataclass
class PIDController:
    """Simple joint-space PID controller.

    The controller computes one motor torque for each joint:

        torque = Kp*error + Ki*integral + Kd*derivative

    For a fixed target, derivative(error) is approximately -joint_velocity.
    """

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
