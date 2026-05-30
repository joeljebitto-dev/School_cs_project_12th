import numpy as np

from control.pid import PIDController, PIDHistory


def test_pid_limits_torque_output():
    controller = PIDController(kp=100.0, ki=0.0, kd=0.0, torque_limit=2.0)

    torque = controller.compute(
        target_angles=np.array([1.0, -1.0, 0.5]),
        current_angles=np.zeros(3),
        current_velocities=np.zeros(3),
        dt=0.01,
    )

    assert np.all(torque <= 2.0)
    assert np.all(torque >= -2.0)


def test_pid_clamps_integral_error():
    controller = PIDController(
        kp=0.0,
        ki=1.0,
        kd=0.0,
        torque_limit=100.0,
        integral_limit=0.1,
    )

    for _ in range(100):
        controller.compute(
            target_angles=np.ones(3),
            current_angles=np.zeros(3),
            current_velocities=np.zeros(3),
            dt=0.1,
        )

    assert np.allclose(controller.integral_error, np.array([0.1, 0.1, 0.1]))


def test_pid_moves_in_direction_that_reduces_error():
    controller = PIDController(kp=2.0, ki=0.0, kd=0.0, torque_limit=10.0)
    target_angles = np.array([0.5, -0.25, 0.15])
    current_angles = np.zeros(3)
    initial_error = np.linalg.norm(target_angles - current_angles)

    for _ in range(10):
        torque = controller.compute(
            target_angles=target_angles,
            current_angles=current_angles,
            current_velocities=np.zeros(3),
            dt=0.05,
        )
        current_angles = current_angles + torque * 0.05

    final_error = np.linalg.norm(target_angles - current_angles)
    assert final_error < initial_error


def test_pid_history_records_and_resets_rolling_samples():
    history = PIDHistory(max_points=2)

    history.record(0.1, 1.0, 2.0)
    history.record(0.2, 0.8, 1.5)
    history.record(0.3, 0.5, 1.0)

    assert history.time_history == [0.30000000000000004, 0.6000000000000001]
    assert history.error_history == [0.8, 0.5]
    assert history.torque_history == [1.5, 1.0]

    history.reset()

    assert history.time_history == []
    assert history.error_history == []
    assert history.torque_history == []
    assert history.plot_time == 0.0
