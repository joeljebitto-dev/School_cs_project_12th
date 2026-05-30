"""MuJoCo model for the 3D 3-DOF robot arm."""

from __future__ import annotations

from constants import (
    BASE_HEIGHT,
    DEFAULT_TARGET_POSITION,
    JOINT_LIMITS_RADIANS,
    LINK_LENGTHS,
    SIMULATION_TIMESTEP,
    TORQUE_LIMIT,
)


def create_mjcf() -> str:
    """Return the MJCF XML model as a string.

    The joints match the math in ``robot_math.py``:

        joint1: yaw about z
        joint2: shoulder pitch about local -y
        joint3: elbow pitch about local -y

    The -y pitch axis makes positive pitch raise the arm in +z.
    """

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
    """Create and return a MuJoCo model.

    Importing MuJoCo inside the function lets tests import this file even when
    MuJoCo is not installed yet.
    """

    import mujoco

    return mujoco.MjModel.from_xml_string(create_mjcf())


def load_model():
    """Backward-compatible name for older tests or notes."""

    return load_mujoco_model()
