"""Forward kinematics and Jacobian math for the 3-DOF robot arm."""

from __future__ import annotations

import numpy as np

from config import BASE_HEIGHT, LINK_LENGTHS
from kinematics.common import as_three_values


def forward_kinematics(
    joint_angles: np.ndarray,
    link_lengths: np.ndarray = LINK_LENGTHS,
    base_height: float = BASE_HEIGHT,
) -> np.ndarray:
    """Return the end-effector position [x, y, z]."""

    q1, q2, q3 = as_three_values(joint_angles, "joint_angles")
    l1, l2, l3 = as_three_values(link_lengths, "link_lengths")

    forearm_and_tool = l2 + l3
    elbow_angle = q2 + q3

    radial_reach = l1 * np.cos(q2) + forearm_and_tool * np.cos(elbow_angle)
    x = radial_reach * np.cos(q1)
    y = radial_reach * np.sin(q1)
    z = base_height + l1 * np.sin(q2) + forearm_and_tool * np.sin(elbow_angle)

    return np.array([x, y, z], dtype=float)


def joint_positions(
    joint_angles: np.ndarray,
    link_lengths: np.ndarray = LINK_LENGTHS,
    base_height: float = BASE_HEIGHT,
) -> np.ndarray:
    """Return floor base, shoulder, elbow, wrist, and end-effector positions."""

    q1, q2, q3 = as_three_values(joint_angles, "joint_angles")
    l1, l2, l3 = as_three_values(link_lengths, "link_lengths")

    radial_direction = np.array([np.cos(q1), np.sin(q1), 0.0])
    vertical_direction = np.array([0.0, 0.0, 1.0])
    elbow_angle = q2 + q3

    floor_base = np.array([0.0, 0.0, 0.0])
    shoulder = np.array([0.0, 0.0, base_height])

    upper_arm = (
        l1 * np.cos(q2) * radial_direction
        + l1 * np.sin(q2) * vertical_direction
    )
    forearm = (
        l2 * np.cos(elbow_angle) * radial_direction
        + l2 * np.sin(elbow_angle) * vertical_direction
    )
    tool = (
        l3 * np.cos(elbow_angle) * radial_direction
        + l3 * np.sin(elbow_angle) * vertical_direction
    )

    elbow = shoulder + upper_arm
    wrist = elbow + forearm
    end_effector = wrist + tool

    return np.vstack([floor_base, shoulder, elbow, wrist, end_effector])


def jacobian(
    joint_angles: np.ndarray,
    link_lengths: np.ndarray = LINK_LENGTHS,
) -> np.ndarray:
    """Return the 3x3 position Jacobian d[x, y, z] / d[q1, q2, q3]."""

    q1, q2, q3 = as_three_values(joint_angles, "joint_angles")
    l1, l2, l3 = as_three_values(link_lengths, "link_lengths")

    forearm_and_tool = l2 + l3
    elbow_angle = q2 + q3

    radial_reach = l1 * np.cos(q2) + forearm_and_tool * np.cos(elbow_angle)
    radial_change_q2 = -l1 * np.sin(q2) - forearm_and_tool * np.sin(elbow_angle)
    radial_change_q3 = -forearm_and_tool * np.sin(elbow_angle)

    dz_dq2 = l1 * np.cos(q2) + forearm_and_tool * np.cos(elbow_angle)
    dz_dq3 = forearm_and_tool * np.cos(elbow_angle)

    cos_yaw = np.cos(q1)
    sin_yaw = np.sin(q1)

    return np.array(
        [
            [
                -radial_reach * sin_yaw,
                radial_change_q2 * cos_yaw,
                radial_change_q3 * cos_yaw,
            ],
            [
                radial_reach * cos_yaw,
                radial_change_q2 * sin_yaw,
                radial_change_q3 * sin_yaw,
            ],
            [0.0, dz_dq2, dz_dq3],
        ],
        dtype=float,
    )
