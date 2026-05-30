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
TORQUE_LIMIT = 8.0
INTEGRAL_LIMIT = 3.0
PID_PLOT_MAX_POINTS = 350
DEFAULT_PID_GAINS = (35.0, 0.0, 3.0)

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
    "bg": "#111827",
    "panel": "#182235",
    "panel_alt": "#202c42",
    "border": "#334155",
    "text": "#e5e7eb",
    "muted": "#94a3b8",
    "accent": "#38bdf8",
    "accent_dark": "#0f766e",
    "entry_bg": "#0f172a",
    "equation_bg": "#111827",
    "result_bg": "#0b1220",
}
