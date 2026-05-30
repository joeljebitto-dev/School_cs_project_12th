"""MuJoCo model and simulation wrapper for the 3-DOF robot demo."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from config import (
    BASE_HEIGHT,
    DEFAULT_TARGET_POSITION,
    JOINT_LIMITS_RADIANS,
    LINK_LENGTHS,
    SIMULATION_STEPS_PER_FRAME,
    SIMULATION_TIMESTEP,
    TORQUE_LIMIT,
)
from control.pid import PIDController
from kinematics.common import wrap_angles


@dataclass
class PidStepSample:
    """Small bundle of values recorded after one UI update."""

    elapsed_time: float
    error_norm: float
    torque_norm: float


def create_mjcf() -> str:
    """Return the MJCF XML model as a string."""

    l1, l2, l3 = LINK_LENGTHS
    target_x, target_y, target_z = DEFAULT_TARGET_POSITION
    q1_min, q1_max = JOINT_LIMITS_RADIANS[0]
    q2_min, q2_max = JOINT_LIMITS_RADIANS[1]
    q3_min, q3_max = JOINT_LIMITS_RADIANS[2]

    return f"""
<mujoco model="three_dof_xyz_arm">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="{SIMULATION_TIMESTEP}" gravity="0 0 0"/>

  <default>
    <joint type="hinge" limited="true" damping="0.10" armature="0.015"/>
    <geom friction="0.7 0.1 0.1"/>
  </default>

  <worldbody>
    <light name="top_light" pos="0 -0.4 2.2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1.4 1.4 0.02"
          rgba="0.92 0.92 0.88 1"/>

    <body name="target" mocap="true" pos="{target_x} {target_y} {target_z}">
      <geom name="target_marker" type="sphere" size="0.035"
            contype="0" conaffinity="0" rgba="0.95 0.20 0.20 0.70"/>
    </body>

    <body name="base_yaw" pos="0 0 0">
      <joint name="joint1" axis="0 0 1" range="{q1_min} {q1_max}"/>
      <geom name="base_column" type="capsule" fromto="0 0 0 0 0 {BASE_HEIGHT}"
            size="0.045" rgba="0.16 0.17 0.19 1"/>
      <geom name="base_plate" type="cylinder" pos="0 0 0.015"
            size="0.09 0.015" rgba="0.12 0.13 0.15 1"/>

      <body name="shoulder_pitch" pos="0 0 {BASE_HEIGHT}">
        <joint name="joint2" axis="0 -1 0" range="{q2_min} {q2_max}"/>
        <geom name="shoulder_joint" type="sphere" size="0.055"
              rgba="0.12 0.13 0.15 1"/>
        <geom name="upper_arm" type="capsule" fromto="0 0 0 {l1} 0 0"
              size="0.030" rgba="0.20 0.42 0.85 1"/>

        <body name="elbow_pitch" pos="{l1} 0 0">
          <joint name="joint3" axis="0 -1 0" range="{q3_min} {q3_max}"/>
          <geom name="elbow_joint" type="sphere" size="0.046"
                rgba="0.12 0.13 0.15 1"/>
          <geom name="forearm" type="capsule" fromto="0 0 0 {l2} 0 0"
                size="0.026" rgba="0.20 0.62 0.42 1"/>

          <body name="tool" pos="{l2} 0 0">
            <geom name="tool_link" type="capsule" fromto="0 0 0 {l3} 0 0"
                  size="0.020" rgba="0.92 0.58 0.18 1"/>
            <site name="end_effector" pos="{l3} 0 0" type="sphere"
                  size="0.032" rgba="0.08 0.08 0.08 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor1" joint="joint1" gear="1"
           ctrllimited="true" ctrlrange="-{TORQUE_LIMIT} {TORQUE_LIMIT}"/>
    <motor name="motor2" joint="joint2" gear="1"
           ctrllimited="true" ctrlrange="-{TORQUE_LIMIT} {TORQUE_LIMIT}"/>
    <motor name="motor3" joint="joint3" gear="1"
           ctrllimited="true" ctrlrange="-{TORQUE_LIMIT} {TORQUE_LIMIT}"/>
  </actuator>
</mujoco>
""".strip()


def load_mujoco_model():
    """Create and return a MuJoCo model."""

    return mujoco.MjModel.from_xml_string(create_mjcf())


class RobotSimulation:
    """Own the MuJoCo model, data, viewer, and simulation stepping."""

    def __init__(self) -> None:
        self.model = load_mujoco_model()
        self.data = mujoco.MjData(self.model)
        self.viewer = None

    def open_viewer(self) -> bool:
        """Open the passive MuJoCo viewer, returning False if already open."""

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
        """Place the robot at a joint pose without running PID."""

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
        """Advance MuJoCo a few small steps using PID torque commands."""

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
