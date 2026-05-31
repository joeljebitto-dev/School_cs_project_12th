import numpy as np

from config import DEFAULT_PID_TARGET_DEGREES
from control.pid import PIDController
from ui.pid_tab import JointGainControls
from ui.app import KinematicsPidApp


class FakeVariable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeControl:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeGridWidget:
    def __init__(self):
        self.visible = None

    def grid(self):
        self.visible = True

    def grid_remove(self):
        self.visible = False


def make_app_without_tk():
    app = KinematicsPidApp.__new__(KinematicsPidApp)
    app.pid_controller = PIDController(kp=35.0, ki=0.0, kd=3.0)
    app.pid_target_angles = np.radians(DEFAULT_PID_TARGET_DEGREES)
    app.last_safe_pid_target_angles = app.pid_target_angles.copy()
    app.pid_running = False
    app._updating_sliders = False

    app.pid_motion_enabled_var = FakeVariable(False)
    app.pid_run_button_text = FakeVariable("Run PID")
    app.pid_status_var = FakeVariable("")
    app.status_var = FakeVariable("")

    app.pid_q_controls = [
        FakeControl(DEFAULT_PID_TARGET_DEGREES[0]),
        FakeControl(DEFAULT_PID_TARGET_DEGREES[1]),
        FakeControl(DEFAULT_PID_TARGET_DEGREES[2]),
    ]
    app.pid_joint_gains = [
        JointGainControls(
            kp=FakeControl(35.0), ki=FakeControl(0.0), kd=FakeControl(3.0),
        ),
        JointGainControls(
            kp=FakeControl(35.0), ki=FakeControl(0.0), kd=FakeControl(3.0),
        ),
        JointGainControls(
            kp=FakeControl(35.0), ki=FakeControl(0.0), kd=FakeControl(3.0),
        ),
    ]
    app.fk_apply_row = FakeGridWidget()
    app._update_pid_live_values = lambda: None
    return app


def test_apply_joint_target_starts_pid_when_pid_motion_is_enabled():
    app = make_app_without_tk()
    app.pid_motion_enabled_var.set(True)
    target_angles = np.array([0.2, 0.4, -0.5])

    assert app._apply_joint_target(target_angles, "FK sliders")

    assert app.pid_running
    assert app.pid_run_button_text.get() == "Pause PID"
    assert np.allclose(app.pid_target_angles, target_angles)
    assert np.allclose(
        [control.get() for control in app.pid_q_controls],
        np.degrees(target_angles),
    )
    assert app.status_var.get() == "FK sliders moving with PID."


def test_pid_target_change_starts_live_pid_motion():
    app = make_app_without_tk()
    app.pid_q_controls[0].set(15.0)
    app.pid_q_controls[1].set(25.0)
    app.pid_q_controls[2].set(-35.0)

    app._on_pid_target_changed()

    assert app.pid_motion_enabled_var.get()
    assert app.pid_running
    assert app.pid_run_button_text.get() == "Pause PID"
    assert np.allclose(app.pid_target_angles, np.radians([15.0, 25.0, -35.0]))


def test_pid_gain_change_updates_gains_without_changing_target():
    app = make_app_without_tk()
    original_target = app.pid_target_angles.copy()
    # Update gains for joint 1 only
    app.pid_joint_gains[0].kp.set(12.0)
    app.pid_joint_gains[0].ki.set(1.5)
    app.pid_joint_gains[0].kd.set(4.0)

    app._on_pid_gain_changed()

    assert app.pid_controller.kp[0] == 12.0
    assert app.pid_controller.ki[0] == 1.5
    assert app.pid_controller.kd[0] == 4.0
    # Other joints should remain unchanged
    assert app.pid_controller.kp[1] == 35.0
    assert app.pid_controller.kp[2] == 35.0
    assert np.allclose(app.pid_target_angles, original_target)


def test_fk_apply_control_hides_when_pid_mode_is_disabled():
    app = make_app_without_tk()
    app.pid_motion_enabled_var.set(False)

    app._update_pid_mode_controls()

    assert app.fk_apply_row.visible is False


def test_fk_apply_control_shows_when_pid_mode_is_enabled():
    app = make_app_without_tk()
    app.pid_motion_enabled_var.set(True)

    app._update_pid_mode_controls()

    assert app.fk_apply_row.visible is True


def test_start_pid_motion_shows_fk_apply_control():
    app = make_app_without_tk()
    app.pid_motion_enabled_var.set(False)

    assert app.start_pid_motion(
        np.array([0.2, 0.3, -0.4]),
        "PID test",
        sync_sliders=False,
        reset_integral=True,
    )

    assert app.fk_apply_row.visible is True
