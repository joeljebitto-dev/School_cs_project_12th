import numpy as np

from config import BASE_HEIGHT, DEFAULT_TARGET_POSITION, LINK_LENGTHS
from kinematics.common import position_error
from kinematics.forward import forward_kinematics, jacobian
from kinematics.inverse import (
    inverse_kinematics,
    is_safe_joint_position,
    safe_joint_message,
)


def test_forward_kinematics_zero_angles():
    position = forward_kinematics(np.array([0.0, 0.0, 0.0]))

    expected = np.array([np.sum(LINK_LENGTHS), 0.0, BASE_HEIGHT])
    assert np.allclose(position, expected)


def test_forward_kinematics_yaw_rotates_reach_into_y_axis():
    position = forward_kinematics(np.array([np.pi / 2.0, 0.0, 0.0]))

    expected = np.array([0.0, np.sum(LINK_LENGTHS), BASE_HEIGHT])
    assert np.allclose(position, expected, atol=1e-12)


def test_forward_kinematics_shoulder_pitch_raises_arm():
    position = forward_kinematics(np.array([0.0, np.pi / 2.0, 0.0]))

    expected = np.array([0.0, 0.0, BASE_HEIGHT + np.sum(LINK_LENGTHS)])
    assert np.allclose(position, expected, atol=1e-12)


def test_jacobian_matches_finite_difference():
    joint_angles = np.array([0.4, 0.5, -0.7])
    robot_jacobian = jacobian(joint_angles)
    finite_difference = np.zeros((3, 3))
    step = 1e-6

    for column in range(3):
        offset = np.zeros(3)
        offset[column] = step
        position_plus = forward_kinematics(joint_angles + offset)
        position_minus = forward_kinematics(joint_angles - offset)
        finite_difference[:, column] = (position_plus - position_minus) / (2.0 * step)

    assert np.allclose(robot_jacobian, finite_difference, atol=1e-6)


def test_inverse_kinematics_reaches_known_position():
    expected_angles = np.array([0.45, 0.35, -0.55])
    target_position = forward_kinematics(expected_angles)

    result = inverse_kinematics(
        target_position,
        initial_guess=np.array([0.2, 0.1, -0.1]),
    )

    assert result.converged
    assert result.error_norm < 1e-4
    assert (
        np.linalg.norm(
            position_error(target_position, forward_kinematics(result.joint_angles))
        )
        < 1e-4
    )


def test_inverse_kinematics_prefers_elevated_elbow_branch():
    result = inverse_kinematics(DEFAULT_TARGET_POSITION, initial_guess=np.zeros(3))

    assert result.converged
    assert result.joint_angles[1] > 0.0
    assert result.joint_angles[2] < 0.0
    assert is_safe_joint_position(result.joint_angles)


def test_inverse_kinematics_reports_unreachable_target():
    unreachable_position = np.array([2.0, 0.0, BASE_HEIGHT])

    result = inverse_kinematics(unreachable_position)

    assert not result.converged
    assert "outside the arm reach" in result.message


def test_safe_joint_position_accepts_natural_pose():
    joint_angles = np.array([0.25, 0.60, -0.80])

    assert is_safe_joint_position(joint_angles)
    assert safe_joint_message(joint_angles) == ""


def test_safe_joint_position_rejects_below_floor_pose():
    joint_angles = np.array([0.0, 0.0, -2.0])

    assert not is_safe_joint_position(joint_angles)
    assert "below the floor" in safe_joint_message(joint_angles)


def test_safe_joint_position_rejects_shoulder_down_pose():
    joint_angles = np.array([0.0, -0.10, -0.20])

    assert not is_safe_joint_position(joint_angles)
    assert "shoulder-down" in safe_joint_message(joint_angles)


def test_safe_joint_position_rejects_positive_elbow_pose():
    joint_angles = np.array([0.0, 0.20, 0.20])

    assert not is_safe_joint_position(joint_angles)
    assert "elbow" in safe_joint_message(joint_angles)


def test_inverse_kinematics_rejects_unsafe_reachable_target():
    unsafe_target = forward_kinematics(np.array([0.0, -0.20, 0.0]))

    result = inverse_kinematics(unsafe_target)

    assert not result.converged
    assert "unsafe" in result.message or "safe workspace" in result.message
