"""Math functions for a 3D 3-DOF robot arm.

The robot uses three revolute joints:

    q1: base yaw about the world z-axis
    q2: shoulder pitch
    q3: elbow pitch

The three joints control end-effector position [x, y, z]. They do not control
full 3D orientation, which would require more degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants import (
    BASE_HEIGHT,
    IK_DAMPING,
    IK_MAX_ITERATIONS,
    IK_MAX_STEP,
    IK_STEP_SIZE,
    IK_TOLERANCE,
    JOINT_LIMITS_RADIANS,
    LINK_LENGTHS,
    MIN_SAFE_Z,
)


@dataclass
class IKResult:
    """Result returned by the inverse kinematics solver."""

    joint_angles: np.ndarray
    converged: bool
    iterations: int
    final_error: np.ndarray
    error_norm: float
    message: str


def wrap_angle(angle: float) -> float:
    """Wrap one angle to the range [-pi, pi)."""

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    """Wrap an array of angles to the range [-pi, pi)."""

    return (np.asarray(angles, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _as_three_values(values: np.ndarray, name: str) -> np.ndarray:
    """Convert input to a length-3 float array with a helpful error message."""

    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")
    return array


def forward_kinematics(
    joint_angles: np.ndarray,
    link_lengths: np.ndarray = LINK_LENGTHS,
    base_height: float = BASE_HEIGHT,
) -> np.ndarray:
    """Return the end-effector position [x, y, z].

    For yaw q1, shoulder pitch q2, elbow pitch q3:

        r = L1*cos(q2) + (L2 + L3)*cos(q2 + q3)
        x = r*cos(q1)
        y = r*sin(q1)
        z = base_height + L1*sin(q2) + (L2 + L3)*sin(q2 + q3)

    The value r is the horizontal distance from the base axis.
    """

    q1, q2, q3 = _as_three_values(joint_angles, "joint_angles")
    l1, l2, l3 = _as_three_values(link_lengths, "link_lengths")

    forearm_and_tool = l2 + l3
    shoulder_angle = q2
    elbow_angle = q2 + q3

    radial_reach = (
        l1 * np.cos(shoulder_angle)
        + forearm_and_tool * np.cos(elbow_angle)
    )
    x = radial_reach * np.cos(q1)
    y = radial_reach * np.sin(q1)
    z = (
        base_height
        + l1 * np.sin(shoulder_angle)
        + forearm_and_tool * np.sin(elbow_angle)
    )

    return np.array([x, y, z], dtype=float)


def joint_positions(
    joint_angles: np.ndarray,
    link_lengths: np.ndarray = LINK_LENGTHS,
    base_height: float = BASE_HEIGHT,
) -> np.ndarray:
    """Return important 3D points along the robot.

    The returned array has shape (5, 3):

        0. floor base
        1. shoulder joint
        2. elbow joint
        3. wrist/tool-base point
        4. end effector
    """

    q1, q2, q3 = _as_three_values(joint_angles, "joint_angles")
    l1, l2, l3 = _as_three_values(link_lengths, "link_lengths")

    radial_direction = np.array([np.cos(q1), np.sin(q1), 0.0])
    vertical_direction = np.array([0.0, 0.0, 1.0])

    shoulder_angle = q2
    elbow_angle = q2 + q3

    floor_base = np.array([0.0, 0.0, 0.0])
    shoulder = np.array([0.0, 0.0, base_height])

    upper_arm = (
        l1 * np.cos(shoulder_angle) * radial_direction
        + l1 * np.sin(shoulder_angle) * vertical_direction
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
    """Return the 3x3 position Jacobian d[x, y, z] / d[q1, q2, q3].

    Column j tells how the end-effector position changes when joint j moves.
    """

    q1, q2, q3 = _as_three_values(joint_angles, "joint_angles")
    l1, l2, l3 = _as_three_values(link_lengths, "link_lengths")

    forearm_and_tool = l2 + l3
    shoulder_angle = q2
    elbow_angle = q2 + q3

    radial_reach = (
        l1 * np.cos(shoulder_angle)
        + forearm_and_tool * np.cos(elbow_angle)
    )
    radial_change_q2 = (
        -l1 * np.sin(shoulder_angle)
        - forearm_and_tool * np.sin(elbow_angle)
    )
    radial_change_q3 = -forearm_and_tool * np.sin(elbow_angle)

    dz_dq2 = (
        l1 * np.cos(shoulder_angle)
        + forearm_and_tool * np.cos(elbow_angle)
    )
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


def position_error(target_position: np.ndarray, current_position: np.ndarray) -> np.ndarray:
    """Return target - current for Cartesian position [x, y, z]."""

    target_position = _as_three_values(target_position, "target_position")
    current_position = _as_three_values(current_position, "current_position")
    return target_position - current_position


def clamp_to_joint_limits(joint_angles: np.ndarray) -> np.ndarray:
    """Return joint angles clamped to the natural demo limits."""

    joint_angles = wrap_angles(_as_three_values(joint_angles, "joint_angles"))
    return np.clip(
        joint_angles,
        JOINT_LIMITS_RADIANS[:, 0],
        JOINT_LIMITS_RADIANS[:, 1],
    )


def safe_joint_message(
    joint_angles: np.ndarray,
    link_lengths: np.ndarray = LINK_LENGTHS,
    base_height: float = BASE_HEIGHT,
    min_safe_z: float = MIN_SAFE_Z,
) -> str:
    """Return an empty string if the joint pose is safe, otherwise explain why."""

    joint_angles = wrap_angles(_as_three_values(joint_angles, "joint_angles"))
    q1, q2, q3 = joint_angles
    lower_limits = JOINT_LIMITS_RADIANS[:, 0]
    upper_limits = JOINT_LIMITS_RADIANS[:, 1]
    tolerance = 1e-9

    if q1 < lower_limits[0] - tolerance or q1 > upper_limits[0] + tolerance:
        return "Blocked: yaw is outside the allowed range."
    if q2 < lower_limits[1] - tolerance:
        return "Blocked: shoulder-down poses are not allowed."
    if q2 > upper_limits[1] + tolerance:
        return "Blocked: shoulder pitch is above the natural range."
    if q3 < lower_limits[2] - tolerance:
        return "Blocked: elbow is folded beyond the natural range."
    if q3 > upper_limits[2] + tolerance:
        return "Blocked: positive/back-folding elbow poses are not allowed."

    # Skip row 0 because it is the fixed floor base at z = 0.
    moving_points = joint_positions(joint_angles, link_lengths, base_height)[1:]
    minimum_point_z = float(np.min(moving_points[:, 2]))
    if minimum_point_z < min_safe_z - tolerance:
        return "Blocked: the arm would go below the floor."

    return ""


def is_safe_joint_position(
    joint_angles: np.ndarray,
    link_lengths: np.ndarray = LINK_LENGTHS,
    base_height: float = BASE_HEIGHT,
    min_safe_z: float = MIN_SAFE_Z,
) -> bool:
    """Return True when the joint pose obeys the natural demo limits."""

    return (
        safe_joint_message(joint_angles, link_lengths, base_height, min_safe_z)
        == ""
    )


def geometric_ik_candidates(
    target_position: np.ndarray,
    link_lengths: np.ndarray = LINK_LENGTHS,
    base_height: float = BASE_HEIGHT,
) -> list[np.ndarray]:
    """Return closed-form IK candidates for the yaw-shoulder-elbow arm.

    The shoulder and elbow behave like a 2-link arm in a vertical plane. The
    two candidates are the two possible elbow branches for the same target.
    """

    target_position = _as_three_values(target_position, "target_position")
    l1, l2, l3 = _as_three_values(link_lengths, "link_lengths")

    forearm_and_tool = l2 + l3
    x, y, z = target_position
    radial_distance = float(np.hypot(x, y))
    vertical_distance = float(z - base_height)
    distance_squared = radial_distance**2 + vertical_distance**2
    distance = float(np.sqrt(distance_squared))

    maximum_reach = l1 + forearm_and_tool
    minimum_reach = abs(l1 - forearm_and_tool)
    if distance > maximum_reach + 1e-9 or distance < minimum_reach - 1e-9:
        return []

    if radial_distance < 1e-9:
        yaw = 0.0
    else:
        yaw = float(np.arctan2(y, x))

    cos_elbow = (
        (distance_squared - l1**2 - forearm_and_tool**2)
        / (2.0 * l1 * forearm_and_tool)
    )
    elbow_magnitude = float(np.arccos(np.clip(cos_elbow, -1.0, 1.0)))

    candidates = []
    for elbow_angle in (-elbow_magnitude, elbow_magnitude):
        shoulder_angle = float(
            np.arctan2(vertical_distance, radial_distance)
            - np.arctan2(
                forearm_and_tool * np.sin(elbow_angle),
                l1 + forearm_and_tool * np.cos(elbow_angle),
            )
        )
        candidates.append(wrap_angles(np.array([yaw, shoulder_angle, elbow_angle])))

    return candidates


def preferred_ik_seed(
    target_position: np.ndarray,
    initial_guess: np.ndarray | None = None,
    link_lengths: np.ndarray = LINK_LENGTHS,
    base_height: float = BASE_HEIGHT,
) -> np.ndarray:
    """Choose a clean starting point for numerical IK.

    A pure damped-least-squares solve can converge to either elbow branch. For
    classroom demos, the branch with the higher elbow is usually clearer
    because the first link does not dive downward before the arm reaches up.
    """

    candidates = [
        candidate
        for candidate in geometric_ik_candidates(target_position, link_lengths, base_height)
        if is_safe_joint_position(candidate, link_lengths, base_height)
    ]
    if not candidates:
        if initial_guess is None:
            return np.zeros(3, dtype=float)
        return clamp_to_joint_limits(_as_three_values(initial_guess, "initial_guess"))

    if initial_guess is None:
        initial_guess = np.zeros(3, dtype=float)
    else:
        initial_guess = wrap_angles(_as_three_values(initial_guess, "initial_guess"))

    def score(candidate: np.ndarray) -> tuple[float, float]:
        elbow_height = float(joint_positions(candidate, link_lengths, base_height)[2, 2])
        distance_from_initial = float(np.linalg.norm(wrap_angles(candidate - initial_guess)))
        return elbow_height, -distance_from_initial

    return max(candidates, key=score)


def inverse_kinematics(
    target_position: np.ndarray,
    initial_guess: np.ndarray | None = None,
    link_lengths: np.ndarray = LINK_LENGTHS,
    base_height: float = BASE_HEIGHT,
    max_iterations: int = IK_MAX_ITERATIONS,
    tolerance: float = IK_TOLERANCE,
    damping: float = IK_DAMPING,
    step_size: float = IK_STEP_SIZE,
    max_step: float = IK_MAX_STEP,
) -> IKResult:
    """Solve Cartesian IK with an elevated-elbow seed and damped least squares.

    The geometric seed chooses the cleaner of the two elbow branches before the
    numerical loop starts. This avoids the visually confusing solution where
    the shoulder dives downward and the elbow folds back up.

    The solver repeatedly linearizes the robot motion:

        position_error ~= J(q) * dq

    Then it computes a stable joint update:

        dq = step_size * J.T * inv(J*J.T + damping^2*I) * position_error

    Damping prevents large jumps near singular positions.
    """

    target_position = _as_three_values(target_position, "target_position")
    link_lengths = _as_three_values(link_lengths, "link_lengths")

    geometric_candidates = geometric_ik_candidates(
        target_position,
        link_lengths,
        base_height,
    )
    safe_candidates = [
        candidate
        for candidate in geometric_candidates
        if is_safe_joint_position(candidate, link_lengths, base_height)
    ]
    shoulder_position = np.array([0.0, 0.0, base_height])
    distance_from_shoulder = float(np.linalg.norm(target_position - shoulder_position))
    maximum_reach = float(np.sum(link_lengths))

    if not safe_candidates:
        if distance_from_shoulder > maximum_reach:
            message = "Did not converge: target position is outside the arm reach."
        elif geometric_candidates:
            message = (
                "Did not converge: target is reachable only with an unsafe "
                "or unnatural arm posture."
            )
        else:
            message = "Did not converge: target is outside the safe workspace."

        fallback_angles = (
            clamp_to_joint_limits(initial_guess)
            if initial_guess is not None
            else np.zeros(3, dtype=float)
        )
        current_position = forward_kinematics(
            fallback_angles,
            link_lengths,
            base_height,
        )
        final_error = position_error(target_position, current_position)
        return IKResult(
            joint_angles=fallback_angles,
            converged=False,
            iterations=0,
            final_error=final_error,
            error_norm=float(np.linalg.norm(final_error)),
            message=message,
        )

    joint_angles = preferred_ik_seed(
        target_position,
        initial_guess,
        link_lengths,
        base_height,
    )

    identity = np.eye(3)
    final_error = np.zeros(3, dtype=float)

    for iteration in range(1, max_iterations + 1):
        current_position = forward_kinematics(
            joint_angles,
            link_lengths,
            base_height,
        )
        final_error = position_error(target_position, current_position)
        error_norm = float(np.linalg.norm(final_error))

        if error_norm < tolerance:
            safety_message = safe_joint_message(joint_angles, link_lengths, base_height)
            if safety_message:
                return IKResult(
                    joint_angles=joint_angles,
                    converged=False,
                    iterations=iteration - 1,
                    final_error=final_error,
                    error_norm=error_norm,
                    message=safety_message,
                )

            return IKResult(
                joint_angles=joint_angles,
                converged=True,
                iterations=iteration - 1,
                final_error=final_error,
                error_norm=error_norm,
                message="Converged: end-effector position reached the target tolerance.",
            )

        robot_jacobian = jacobian(joint_angles, link_lengths)
        stable_matrix = robot_jacobian @ robot_jacobian.T + (damping**2) * identity
        joint_step = step_size * robot_jacobian.T @ np.linalg.solve(
            stable_matrix,
            final_error,
        )

        step_norm = float(np.linalg.norm(joint_step))
        if step_norm > max_step:
            joint_step *= max_step / step_norm

        joint_angles = clamp_to_joint_limits(joint_angles + joint_step)

    current_position = forward_kinematics(joint_angles, link_lengths, base_height)
    final_error = position_error(target_position, current_position)
    error_norm = float(np.linalg.norm(final_error))

    if distance_from_shoulder > maximum_reach:
        message = "Did not converge: target position is outside the arm reach."
    elif safe_joint_message(joint_angles, link_lengths, base_height):
        message = "Did not converge: no safe natural arm posture reaches this target."
    else:
        message = "Did not converge: try a different initial joint position."

    return IKResult(
        joint_angles=joint_angles,
        converged=False,
        iterations=max_iterations,
        final_error=final_error,
        error_norm=error_norm,
        message=message,
    )
