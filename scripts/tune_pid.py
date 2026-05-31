"""Simple PID tuner that grid-searches gains for each joint independently.

Runs a closed-loop simulation for each joint separately and scores
performance by integrated error + overshoot + control effort.
Includes a gravity torque term for joints that fight gravity (shoulder, elbow).

Usage: python scripts/tune_pid.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DEFAULT_PID_TARGET_DEGREES


def simulate_single_joint(
    kp: float,
    ki: float,
    kd: float,
    target_rad: float,
    inertia: float = 0.02,
    damping: float = 0.1,
    gravity_torque: float = 0.0,
    sim_time: float = 3.0,
    dt: float = 0.005,
) -> tuple[float, float, float]:
    """Simulate a single joint driven by PID and return (cost, final_error, max_overshoot).

    The joint model is a simple second-order plant: I*theta_dd + b*theta_d = torque - gravity
    with small inertia, viscous damping, and an optional constant gravity load.
    """
    integral_limit = 3.0
    torque_limit = 16.0
    steps = int(math.ceil(sim_time / dt))

    angle = 0.0
    vel = 0.0
    integral_error = 0.0

    control_effort_sum = 0.0
    error_integral = 0.0
    max_overshoot = 0.0

    for _ in range(steps):
        error = target_rad - angle
        # wrap to [-pi, pi]
        error = (error + math.pi) % (2 * math.pi) - math.pi
        integral_error += error * dt
        integral_error = max(-integral_limit, min(integral_limit, integral_error))
        derivative = -vel
        torque = kp * error + ki * integral_error + kd * derivative
        torque = max(-torque_limit, min(torque_limit, torque))

        # dynamics: theta_dd = (torque - gravity_torque - b*vel) / I
        acc = (torque - gravity_torque - damping * vel) / inertia
        vel = vel + acc * dt
        angle = angle + vel * dt

        err = abs(target_rad - angle)
        error_integral += err * dt
        control_effort_sum += abs(torque) * dt
        overshoot = max(0.0, abs(angle) - abs(target_rad))
        max_overshoot = max(max_overshoot, overshoot)

    final_error = abs(target_rad - angle)
    # cost weights: prioritize steady-state error and overshoot
    cost = error_integral + 0.5 * max_overshoot + 0.001 * control_effort_sum + 5.0 * final_error
    return cost, final_error, max_overshoot


def grid_search(kp_range, ki_range, kd_range, target_rad, inertia, damping, gravity_torque):
    best = (float("inf"), None)
    total = len(kp_range) * len(ki_range) * len(kd_range)
    seen = 0
    for kp in kp_range:
        for ki in ki_range:
            for kd in kd_range:
                seen += 1
                cost, final_err, overshoot = simulate_single_joint(
                    kp, ki, kd, target_rad,
                    inertia=inertia, damping=damping, gravity_torque=gravity_torque,
                )
                if cost < best[0]:
                    best = (cost, (kp, ki, kd))
                if seen % 200 == 0:
                    print(
                        f"  Tried {seen}/{total} combos, best cost={best[0]:.4f} gains={best[1]}"
                    )
    return best


# Per-joint physics: different inertia, damping, and gravity load per joint.
# Gravity torque approximates the load each joint must counteract.
# Joint 1 (yaw) rotates around the vertical axis so no gravity torque.
# Joint 2 (shoulder) supports the full arm weight.
# Joint 3 (elbow) supports the forearm + tool weight.
JOINT_PARAMS = {
    "joint1": {"name": "yaw",      "inertia": 0.025, "damping": 0.12, "gravity_torque": 0.0},
    "joint2": {"name": "shoulder",  "inertia": 0.030, "damping": 0.15, "gravity_torque": 2.5},
    "joint3": {"name": "elbow",     "inertia": 0.015, "damping": 0.08, "gravity_torque": 1.0},
}


def main():
    target_rads = np.radians(DEFAULT_PID_TARGET_DEGREES)
    results = {}

    for idx, (joint_key, params) in enumerate(JOINT_PARAMS.items()):
        joint_name = params["name"]
        inertia = params["inertia"]
        damping = params["damping"]
        gravity_torque = params["gravity_torque"]
        target_rad = target_rads[idx]

        print(f"\n{'='*60}")
        print(f"Tuning {joint_key} ({joint_name}) — target={np.degrees(target_rad):.1f}°")
        print(f"  inertia={inertia}, damping={damping}, gravity_torque={gravity_torque}")
        print(f"{'='*60}")

        # coarse grid
        kp_range = np.linspace(20.0, 300.0, 29)
        ki_range = np.linspace(0.0, 10.0, 21)
        kd_range = np.linspace(0.0, 40.0, 21)

        print("  Coarse grid search...")
        cost, gains = grid_search(
            kp_range, ki_range, kd_range, target_rad, inertia, damping, gravity_torque,
        )
        print(f"  Coarse best: cost={cost:.4f} gains={gains}")

        # fine search around best coarse gains
        kp0, ki0, kd0 = gains
        kp_range = np.linspace(max(0.0, kp0 - 20), kp0 + 20, 13)
        ki_range = np.linspace(max(0.0, ki0 - 1.5), ki0 + 1.5, 13)
        kd_range = np.linspace(max(0.0, kd0 - 8), kd0 + 8, 13)

        print("  Fine grid search...")
        cost, gains = grid_search(
            kp_range, ki_range, kd_range, target_rad, inertia, damping, gravity_torque,
        )
        # verify convergence
        kp, ki, kd = gains
        _, final_err, _ = simulate_single_joint(
            kp, ki, kd, target_rad, inertia=inertia, damping=damping,
            gravity_torque=gravity_torque,
        )
        print(f"  Fine best: cost={cost:.6f} gains={gains}")
        print(f"  Final steady-state error: {math.degrees(final_err):.4f}°")

        results[joint_key] = gains

    print(f"\n{'='*60}")
    print("Recommended DEFAULT_PID_GAINS = {")
    for joint_key, gains in results.items():
        name = JOINT_PARAMS[joint_key]["name"]
        kp, ki, kd = gains
        print(f'    "{joint_key}": ({kp:.1f}, {ki:.2f}, {kd:.1f}),   # {name}')
    print("}")


if __name__ == "__main__":
    main()
