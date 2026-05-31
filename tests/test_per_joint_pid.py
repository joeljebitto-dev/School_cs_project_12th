"""Tests for per-joint PID gains and tuned values."""

import math

import numpy as np
import pytest

from config import DEFAULT_PID_GAINS, DEFAULT_PID_TARGET_DEGREES
from control.pid import PIDController
from kinematics.common import wrap_angles


# ---- Per-joint gain structure tests ----


def test_default_pid_gains_has_all_three_joints():
    """DEFAULT_PID_GAINS must define gains for joint1, joint2, and joint3."""
    assert "joint1" in DEFAULT_PID_GAINS
    assert "joint2" in DEFAULT_PID_GAINS
    assert "joint3" in DEFAULT_PID_GAINS
    assert len(DEFAULT_PID_GAINS) == 3


def test_default_pid_gains_are_positive_tuples():
    """Each joint's gains must be a 3-tuple of non-negative floats."""
    for key in ("joint1", "joint2", "joint3"):
        gains = DEFAULT_PID_GAINS[key]
        assert len(gains) == 3, f"{key} should have (kp, ki, kd)"
        kp, ki, kd = gains
        assert kp >= 0.0, f"{key} Kp must be non-negative"
        assert ki >= 0.0, f"{key} Ki must be non-negative"
        assert kd >= 0.0, f"{key} Kd must be non-negative"


def test_default_pid_gains_differ_across_joints():
    """The tuned gains should not all be identical — each joint has different dynamics."""
    gains = [DEFAULT_PID_GAINS[f"joint{i+1}"] for i in range(3)]
    assert not (gains[0] == gains[1] == gains[2]), (
        "All joints have the same gains; per-joint tuning should produce different values."
    )


def test_pid_controller_accepts_per_joint_arrays():
    """PIDController should accept array gains and produce per-joint torques."""
    kp = np.array([300.0, 320.0, 283.3])
    ki = np.array([0.0, 11.5, 0.0])
    kd = np.array([4.0, 4.7, 4.0])

    controller = PIDController(kp=kp, ki=ki, kd=kd)

    assert controller.kp.shape == (3,)
    assert controller.ki.shape == (3,)
    assert controller.kd.shape == (3,)

    torque = controller.compute(
        target_angles=np.array([0.05, 0.03, 0.05]),
        current_angles=np.zeros(3),
        current_velocities=np.zeros(3),
        dt=0.005,
    )
    assert torque.shape == (3,)
    # Higher Kp on joint 1 should produce higher torque for same-magnitude error
    # joints 1 and 3 have the same target, but joint 1 has Kp=300 vs joint 3 Kp=283.3
    assert abs(torque[0]) > abs(torque[2])


# ---- Tuned value convergence tests ----


def _simulate_joint(kp, ki, kd, target_rad, inertia, damping, gravity_torque=0.0,
                    sim_time=3.0, dt=0.005):
    """Run a single-joint PID simulation and return the final angle."""
    integral_limit = 3.0
    torque_limit = 16.0
    steps = int(math.ceil(sim_time / dt))

    angle = 0.0
    vel = 0.0
    integral_error = 0.0

    for _ in range(steps):
        error = target_rad - angle
        error = (error + math.pi) % (2 * math.pi) - math.pi
        integral_error += error * dt
        integral_error = max(-integral_limit, min(integral_limit, integral_error))
        derivative = -vel
        torque = kp * error + ki * integral_error + kd * derivative
        torque = max(-torque_limit, min(torque_limit, torque))

        acc = (torque - gravity_torque - damping * vel) / inertia
        vel += acc * dt
        angle += vel * dt

    return angle


@pytest.mark.parametrize(
    "joint_key, target_deg, inertia, damping, gravity_torque",
    [
        ("joint1", DEFAULT_PID_TARGET_DEGREES[0], 0.025, 0.12, 0.0),
        ("joint2", DEFAULT_PID_TARGET_DEGREES[1], 0.030, 0.15, 2.5),
        ("joint3", DEFAULT_PID_TARGET_DEGREES[2], 0.015, 0.08, 1.0),
    ],
)
def test_tuned_gains_converge_to_target(joint_key, target_deg, inertia, damping, gravity_torque):
    """Each joint's tuned gains must drive the plant to within 1° of the target."""
    kp, ki, kd = DEFAULT_PID_GAINS[joint_key]
    target_rad = math.radians(target_deg)

    final_angle = _simulate_joint(
        kp, ki, kd, target_rad, inertia, damping, gravity_torque=gravity_torque,
    )

    error_deg = abs(math.degrees(final_angle) - target_deg)
    assert error_deg < 1.0, (
        f"{joint_key}: final error {error_deg:.3f}° exceeds 1° threshold "
        f"(target={target_deg}°, final={math.degrees(final_angle):.3f}°)"
    )


def test_tuned_gains_full_arm_convergence():
    """All three joints running together with tuned gains should converge."""
    kp = np.array([DEFAULT_PID_GAINS[f"joint{i+1}"][0] for i in range(3)])
    ki = np.array([DEFAULT_PID_GAINS[f"joint{i+1}"][1] for i in range(3)])
    kd = np.array([DEFAULT_PID_GAINS[f"joint{i+1}"][2] for i in range(3)])

    controller = PIDController(kp=kp, ki=ki, kd=kd)
    target = np.radians(DEFAULT_PID_TARGET_DEGREES)

    # Simple second-order plant per joint with gravity
    inertias = np.array([0.025, 0.030, 0.015])
    dampings = np.array([0.12, 0.15, 0.08])
    gravity_torques = np.array([0.0, 2.5, 1.0])
    angle = np.zeros(3)
    vel = np.zeros(3)
    dt = 0.005

    for _ in range(int(3.0 / dt)):
        torque = controller.compute(target, angle, vel, dt)
        acc = (torque - gravity_torques - dampings * vel) / inertias
        vel += acc * dt
        angle += vel * dt

    error_deg = np.abs(np.degrees(wrap_angles(target - angle)))
    for i in range(3):
        assert error_deg[i] < 1.0, (
            f"Joint {i+1}: final error {error_deg[i]:.3f}° exceeds 1° threshold"
        )
