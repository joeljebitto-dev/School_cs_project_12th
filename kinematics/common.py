"""Small shared math helpers for the robot kinematics modules."""

from __future__ import annotations

import numpy as np


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    """Wrap an array of angles to the range [-pi, pi)."""

    return (np.asarray(angles, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def as_three_values(values: np.ndarray, name: str) -> np.ndarray:
    """Convert input to a length-3 float array with a helpful error message."""

    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")
    return array


def position_error(target_position: np.ndarray, current_position: np.ndarray) -> np.ndarray:
    """Return target - current for Cartesian position [x, y, z]."""

    target_position = as_three_values(target_position, "target_position")
    current_position = as_three_values(current_position, "current_position")
    return target_position - current_position
