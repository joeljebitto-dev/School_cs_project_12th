"""Inverse kinematics and safety checks for the 3-DOF robot arm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
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
from kinematics.common import as_three_values, position_error, wrap_angles
from kinematics.forward import forward_kinematics, jacobian, joint_positions


@dataclass
class IKResult:
    """Result returned by the inverse kinematics solver."""

    joint_angles: np.ndarray
    converged: bool
    iterations: int
    final_error: np.ndarray
    error_norm: float
    message: str


def clamp_to_joint_limits(joint_angles: np.ndarray) -> np.ndarray:
    """Return joint angles clamped to the natural demo limits."""

    joint_angles = wrap_angles(as_three_values(joint_angles, "joint_angles"))
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

    joint_angles = wrap_angles(as_three_values(joint_angles, "joint_angles"))
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
    """Return the two geometric elbow branches for the target, if reachable."""

    target_position = as_three_values(target_position, "target_position")
    l1, l2, l3 = as_three_values(link_lengths, "link_lengths")

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

    yaw = 0.0 if radial_distance < 1e-9 else float(np.arctan2(y, x))
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
    """Choose the safe elbow branch with the higher elbow when possible."""

    candidates = [
        candidate
        for candidate in geometric_ik_candidates(target_position, link_lengths, base_height)
        if is_safe_joint_position(candidate, link_lengths, base_height)
    ]
    if not candidates:
        if initial_guess is None:
            return np.zeros(3, dtype=float)
        return clamp_to_joint_limits(as_three_values(initial_guess, "initial_guess"))

    if initial_guess is None:
        initial_guess = np.zeros(3, dtype=float)
    else:
        initial_guess = wrap_angles(as_three_values(initial_guess, "initial_guess"))

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
    """Solve Cartesian IK with an elevated-elbow seed and damped least squares."""

    target_position = as_three_values(target_position, "target_position")
    link_lengths = as_three_values(link_lengths, "link_lengths")

    geometric_candidates = geometric_ik_candidates(target_position, link_lengths, base_height)
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
        current_position = forward_kinematics(fallback_angles, link_lengths, base_height)
        final_error = position_error(target_position, current_position)
        return IKResult(
            joint_angles=fallback_angles,
            converged=False,
            iterations=0,
            final_error=final_error,
            error_norm=float(np.linalg.norm(final_error)),
            message=message,
        )

    joint_angles = preferred_ik_seed(target_position, initial_guess, link_lengths, base_height)
    identity = np.eye(3)
    final_error = np.zeros(3, dtype=float)

    for iteration in range(1, max_iterations + 1):
        current_position = forward_kinematics(joint_angles, link_lengths, base_height)
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
