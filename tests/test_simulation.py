import numpy as np
import pytest

from control.pid import PIDController


def test_robot_simulation_sets_pose_and_steps_pid():
    pytest.importorskip("mujoco")

    from simulation.mujoco_sim import RobotSimulation

    simulation = RobotSimulation()
    target_angles = np.array([0.15, 0.20, -0.25])
    controller = PIDController(kp=8.0, ki=0.0, kd=0.4)

    simulation.set_joint_angles(np.zeros(3))
    sample = simulation.step_pid_control(controller, target_angles)

    assert sample.elapsed_time > 0.0
    assert sample.error_norm >= 0.0
    assert sample.torque_norm >= 0.0
    assert np.all(np.isfinite(simulation.current_joint_angles()))
