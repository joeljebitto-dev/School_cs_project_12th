"""Shared constants for the 3-DOF robot demo.

Keeping these values in one file makes it easy to change the robot size,
solver behavior, or controller limits without hunting through the project.
"""

from __future__ import annotations

import numpy as np


# Link lengths are in meters.
#
# L1 is the upper arm.
# L2 is the forearm.
# L3 is a small tool/end-effector extension.
LINK_LENGTHS = np.array([0.45, 0.35, 0.25], dtype=float)

# The shoulder joint is raised above the floor by this amount.
BASE_HEIGHT = 0.18

# Natural classroom-demo joint limits, in degrees.
#
# q1: yaw
# q2: shoulder pitch, restricted to avoid shoulder-down poses
# q3: elbow pitch, restricted to the elevated-elbow branch
JOINT_LIMITS_DEGREES = np.array(
    [
        [-180.0, 180.0],
        [0.0, 110.0],
        [-135.0, 0.0],
    ],
    dtype=float,
)
JOINT_LIMITS_RADIANS = np.radians(JOINT_LIMITS_DEGREES)

# Moving robot points must stay above the floor by this margin.
MIN_SAFE_Z = 0.02

# Inverse kinematics settings.
IK_MAX_ITERATIONS = 120
IK_TOLERANCE = 1e-4
IK_DAMPING = 0.05
IK_STEP_SIZE = 0.7
IK_MAX_STEP = 0.20

# MuJoCo and PID settings.
SIMULATION_TIMESTEP = 0.005
SIMULATION_STEPS_PER_FRAME = 3
UI_UPDATE_INTERVAL_MS = 20
TORQUE_LIMIT = 8.0
INTEGRAL_LIMIT = 3.0
PID_PLOT_MAX_POINTS = 350

# Default Cartesian target shown by the red MuJoCo marker.
DEFAULT_TARGET_POSITION = np.array([0.65, 0.20, 0.35], dtype=float)

# Default joint target used on the PID tab, in degrees for easy reading.
DEFAULT_PID_TARGET_DEGREES = np.array([25.0, 35.0, -45.0], dtype=float)

# Slider ranges for the Cartesian target controls.
MAX_REACH = float(np.sum(LINK_LENGTHS))
IK_TARGET_XY_LIMIT = MAX_REACH
IK_TARGET_Z_MAX = BASE_HEIGHT + MAX_REACH
