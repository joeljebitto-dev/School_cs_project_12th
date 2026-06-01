"""PID Control tab actions — handles PID slider, button, and plot events."""

import numpy as np

from kinematics.common import wrap_angles
from kinematics.forward import forward_kinematics
from kinematics.inverse import safe_joint_message


# ------------------------------------------------------------------
# PID mode UI
# ------------------------------------------------------------------

def update_mode_controls(app):
    """Show FK Apply button only when PID motion is on."""
    if not hasattr(app, "fk_apply_row"):
        return
    if app.pid_motion_enabled_var.get():
        app.fk_apply_row.grid()
    else:
        app.fk_apply_row.grid_remove()


def on_motion_toggled(app):
    """Called when the 'Use PID Motion' checkbox changes."""
    if app.pid_motion_enabled_var.get():
        set_target(app, app._current_joint_angles(), sync_sliders=True, reset_integral=True)
        app.status_var.set("PID motion enabled. Press Run PID when ready.")
        update_mode_controls(app)
        update_live_values(app)
        return

    pause(app)
    app.status_var.set("PID motion disabled. FK sliders are direct again.")
    update_mode_controls(app)


# ------------------------------------------------------------------
# PID start / pause / hold
# ------------------------------------------------------------------

def toggle(app):
    """Start PID if paused, or pause it if already running."""
    if app.pid_running:
        pause(app)
        return

    target_angles = read_target_angles(app)
    start_motion(app, target_angles, "PID tab target", sync_sliders=False, reset_integral=True)


def start_motion(app, joint_angles, source, sync_sliders, reset_integral):
    """Set a PID target and immediately start smooth motion."""
    if not set_target(app, joint_angles, sync_sliders, reset_integral):
        return False

    app.pid_motion_enabled_var.set(True)
    app.pid_running = True
    app.pid_run_button_text.set("Pause PID")
    app.pid_status_var.set(f"PID moving toward {source}.")
    app.status_var.set(f"{source} moving with PID.")
    update_mode_controls(app)
    update_live_values(app)
    return True


def pause(app):
    """Pause the PID controller and remove motor torque."""
    app.pid_running = False
    app.simulation.stop_motors()
    app.pid_run_button_text.set("Run PID")
    app.pid_status_var.set("PID is paused.")
    app.status_var.set("PID controller paused.")
    update_live_values(app)


def hold_current(app):
    """Use the current robot pose as the PID target."""
    current_angles = app._current_joint_angles()
    if not set_target(app, current_angles, sync_sliders=True, reset_integral=True):
        return
    app.pid_motion_enabled_var.set(True)
    app.status_var.set("PID target set to the current robot pose.")
    update_mode_controls(app)
    update_live_values(app)


# ------------------------------------------------------------------
# PID target and gain changes
# ------------------------------------------------------------------

def on_target_changed(app):
    """Called when a PID target slider moves."""
    target_angles = read_target_angles(app)
    if not accept_slider_angles(app, target_angles):
        return
    start_motion(app, target_angles, "PID target input", sync_sliders=False, reset_integral=True)


def on_gain_changed(app):
    """Called when a PID gain slider moves."""
    update_gains(app)
    update_live_values(app)
    if app.pid_running:
        app.status_var.set("PID gains updated while the controller is running.")
    else:
        app.status_var.set("PID gains updated.")


def update_gains(app):
    """Read the gain sliders and push values into the PID controller."""
    kp = np.array([jg.kp.get() for jg in app.pid_joint_gains])
    ki = np.array([jg.ki.get() for jg in app.pid_joint_gains])
    kd = np.array([jg.kd.get() for jg in app.pid_joint_gains])
    app.pid_controller.kp = kp
    app.pid_controller.ki = ki
    app.pid_controller.kd = kd


def set_target(app, joint_angles, sync_sliders, reset_integral):
    """Validate and store new PID target angles."""
    safety_message = safe_joint_message(joint_angles)
    if safety_message:
        app.status_var.set(safety_message)
        return False

    app.pid_target_angles = wrap_angles(joint_angles)
    app.last_safe_pid_target_angles = app.pid_target_angles.copy()
    update_gains(app)
    if reset_integral:
        app.pid_controller.reset()
    if sync_sliders:
        set_input_degrees(app, app.pid_target_angles)
    return True


# ------------------------------------------------------------------
# Safety validation
# ------------------------------------------------------------------

def accept_slider_angles(app, joint_angles):
    """Check if PID target angles are safe. If not, revert the sliders."""
    safety_message = safe_joint_message(joint_angles)
    if safety_message:
        set_input_degrees(app, app.last_safe_pid_target_angles)
        app.status_var.set(safety_message)
        return False
    app.last_safe_pid_target_angles = wrap_angles(joint_angles)
    return True


# ------------------------------------------------------------------
# Slider helpers
# ------------------------------------------------------------------

def set_input_degrees(app, joint_angles):
    """Push angle values into the PID sliders without triggering callbacks."""
    app._pause_callbacks()
    for control, value in zip(app.pid_q_controls, np.degrees(joint_angles)):
        control.set(value)
    app._resume_callbacks()


def read_target_angles(app):
    """Read the three PID target sliders as radians."""
    return np.radians(app._read_slider_values(app.pid_q_controls))


# ------------------------------------------------------------------
# PID plot
# ------------------------------------------------------------------

def record_sample(app, sample):
    """Save one PID step to history and refresh the plot."""
    app.pid_history.record(sample.elapsed_time, sample.error_norm, sample.torque_norm)
    app.pid_status_var.set(
        "PID running. "
        f"error norm = {sample.error_norm:.4f}, "
        f"torque norm = {sample.torque_norm:.4f}"
    )
    update_plot(app)


def update_plot(app):
    """Redraw the PID error/torque graph."""
    times = app.pid_history.time_history
    errors = app.pid_history.error_history
    torques = app.pid_history.torque_history

    app.error_line.set_data(times, errors)
    app.torque_line.set_data(times, torques)

    if times:
        t_min, t_max = times[0], times[-1]
        if t_max <= t_min:
            t_max = t_min + 1.0
        app.pid_axis.set_xlim(t_min, t_max)

        all_values = errors + torques
        y_max = max(all_values) if all_values else 1.0
        y_max = max(y_max, 0.1)
        app.pid_axis.set_ylim(0.0, y_max * 1.1)

    app.pid_canvas.draw_idle()


def reset_plot(app):
    """Clear the PID plot lines and reset the axes."""
    app.error_line.set_data([], [])
    app.torque_line.set_data([], [])
    app.pid_axis.set_xlim(0.0, 5.0)
    app.pid_axis.set_ylim(0.0, 1.0)
    app.pid_canvas.draw_idle()


def clear_history(app):
    """Clear all recorded PID data and reset the plot."""
    app.pid_history.reset()
    reset_plot(app)


# ------------------------------------------------------------------
# Live values display
# ------------------------------------------------------------------

def update_live_values(app):
    """Refresh the live PID values panel."""
    current_angles = app._current_joint_angles()
    target_angles = app.pid_target_angles
    error_angles = wrap_angles(target_angles - current_angles)
    torques = app.simulation.current_motor_torques()
    end_effector_position = forward_kinematics(current_angles)
    state = "running" if app.pid_running else "paused"
    gravity_state = "on" if app.simulation.gravity_is_enabled() else "off"

    output = [
        f"State: {state}",
        f"Gravity: {gravity_state}",
        f"Current q (deg): {app._format_vector(np.degrees(current_angles))}",
        f"Target q  (deg): {app._format_vector(np.degrees(target_angles))}",
        f"Error q   (deg): {app._format_vector(np.degrees(error_angles))}",
        f"Torque command:  {app._format_vector(torques)}",
        f"End effector m:  {app._format_vector(end_effector_position)}",
    ]
    app.pid_live_values_var.set("\n".join(output))
