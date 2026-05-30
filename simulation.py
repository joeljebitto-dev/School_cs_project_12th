"""Small MuJoCo simulation wrapper for the robot demo.

This file keeps the physics details out of the Tkinter code. The UI can ask
for simple actions like "open the viewer", "set joint angles", or "step PID",
while this class handles ``mjData``, ``mj_step()``, and viewer locking.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from constants import (
    SIMULATION_STEPS_PER_FRAME,
    SIMULATION_TIMESTEP,
)
from controller import PIDController
from mujoco_model import load_mujoco_model
from robot_math import wrap_angles


@dataclass
class PidStepSample:
    """Small bundle of values recorded after one UI update."""

    elapsed_time: float
    error_norm: float
    torque_norm: float


class RobotSimulation:
    """Own the MuJoCo model, data, viewer, and simulation stepping."""

    def __init__(self) -> None:
        self.model = load_mujoco_model()
        self.data = mujoco.MjData(self.model)
        self.viewer = None

    def open_viewer(self) -> bool:
        """Open the passive MuJoCo viewer.

        Returns ``False`` when the viewer is already open, so the UI can show a
        friendly message instead of launching a duplicate window.
        """

        if self.viewer_is_open():
            return False

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.sync_viewer()
        return True

    def close_viewer(self) -> None:
        """Close the viewer window if it exists."""

        if self.viewer is not None:
            self.viewer.close()

    def viewer_is_open(self) -> bool:
        """Return True while the passive viewer window is still running."""

        return self.viewer is not None and self.viewer.is_running()

    def set_joint_angles(self, joint_angles: np.ndarray) -> None:
        """Place the robot at a joint pose without running PID.

        ``mj_forward()`` updates all MuJoCo-derived positions, including the
        end-effector site, after qpos changes directly.
        """

        with self._viewer_lock():
            self.data.qpos[:3] = wrap_angles(joint_angles)
            self.data.qvel[:3] = 0.0
            self.data.ctrl[:3] = 0.0
            mujoco.mj_forward(self.model, self.data)

        self.sync_viewer()

    def set_target_marker(self, target_position: np.ndarray) -> None:
        """Move the red marker that shows the desired Cartesian target."""

        if self.model.nmocap == 0:
            return

        with self._viewer_lock():
            self.data.mocap_pos[0] = np.asarray(target_position, dtype=float)
            mujoco.mj_forward(self.model, self.data)

        self.sync_viewer()

    def current_joint_angles(self) -> np.ndarray:
        """Return a copy of the current three joint angles."""

        return self.data.qpos[:3].copy()

    def current_motor_torques(self) -> np.ndarray:
        """Return a copy of the current three motor commands."""

        return self.data.ctrl[:3].copy()

    def stop_motors(self) -> None:
        """Set all motor commands to zero."""

        with self._viewer_lock():
            self.data.ctrl[:3] = 0.0

    def step_pid_control(
        self,
        controller: PIDController,
        target_angles: np.ndarray,
    ) -> PidStepSample:
        """Advance MuJoCo a few small steps using PID torque commands.

        Tkinter calls this method through ``after()``, so the UI remains
        responsive. The inner loop keeps the physics timestep small without
        requiring a background thread.
        """

        with self._viewer_lock():
            for _ in range(SIMULATION_STEPS_PER_FRAME):
                torque = controller.compute(
                    target_angles,
                    self.data.qpos[:3],
                    self.data.qvel[:3],
                    SIMULATION_TIMESTEP,
                )
                self.data.ctrl[:3] = torque
                mujoco.mj_step(self.model, self.data)

        error = wrap_angles(target_angles - self.data.qpos[:3])
        return PidStepSample(
            elapsed_time=SIMULATION_TIMESTEP * SIMULATION_STEPS_PER_FRAME,
            error_norm=float(np.linalg.norm(error)),
            torque_norm=float(np.linalg.norm(self.data.ctrl[:3])),
        )

    def sync_viewer(self) -> None:
        """Refresh the MuJoCo viewer if it is open."""

        if self.viewer_is_open():
            self.viewer.sync()

    def _viewer_lock(self):
        """Use the viewer lock only while the viewer exists."""

        if self.viewer_is_open():
            return self.viewer.lock()
        return nullcontext()
