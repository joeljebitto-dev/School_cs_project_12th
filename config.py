"""Configuration values for the 3-DOF robot demo."""

from __future__ import annotations

import numpy as np


# Robot dimensions are in meters.
LINK_LENGTHS = np.array([0.45, 0.35, 0.25], dtype=float)
BASE_HEIGHT = 0.18

# Natural classroom-demo joint limits, in degrees.
JOINT_LIMITS_DEGREES = np.array(
    [
        [-180.0, 180.0],
        [0.0, 110.0],
        [-135.0, 0.0],
    ],
    dtype=float,
)
JOINT_LIMITS_RADIANS = np.radians(JOINT_LIMITS_DEGREES)
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
NO_GRAVITY = np.array([0.0, 0.0, 0.0], dtype=float)
PID_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)
INTEGRAL_LIMIT = 3.0
PID_PLOT_MAX_POINTS = 2000
TORQUE_LIMIT = 16.0
DEFAULT_PID_GAINS = {
    "joint1": (300.0, 0.0, 4.0),     # yaw
    "joint2": (320.0, 11.5, 4.7),    # shoulder (Ki fights gravity)
    "joint3": (283.3, 0.0, 4.0),     # elbow
}

# Default values shown in the UI.
DEFAULT_TARGET_POSITION = np.array([0.65, 0.20, 0.35], dtype=float)
DEFAULT_PID_TARGET_DEGREES = np.array([25.0, 35.0, -45.0], dtype=float)
MAX_REACH = float(np.sum(LINK_LENGTHS))
IK_TARGET_XY_LIMIT = MAX_REACH
IK_TARGET_Z_MAX = BASE_HEIGHT + MAX_REACH

# UI constants.
WINDOW_WIDTH_FRACTION = 0.90
WINDOW_HEIGHT_FRACTION = 0.90
MIN_WINDOW_WIDTH = 1024
MIN_WINDOW_HEIGHT = 700

UI_COLORS = {
    "bg": "#09090b",
    "panel": "#18181b",
    "panel_alt": "#27272a",
    "border": "#3f3f46",
    "text": "#f4f4f5",
    "muted": "#a1a1aa",
    "accent": "#3b82f6",
    "accent_dark": "#2563eb",
    "entry_bg": "#000000",
    "equation_bg": "#09090b",
    "result_bg": "#000000",
}
