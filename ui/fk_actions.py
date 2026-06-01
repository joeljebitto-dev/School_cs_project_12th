"""Forward Kinematics tab actions — handles FK slider and button events."""

import numpy as np

from config import LINK_LENGTHS
from kinematics.common import wrap_angles
from kinematics.forward import forward_kinematics, jacobian, joint_positions
from kinematics.inverse import safe_joint_message


def on_slider_changed(app):
    """Called when an FK slider moves."""
    joint_angles = np.radians(app._read_slider_values(app.fk_q_controls))
    if not accept_slider_angles(app, joint_angles):
        return

    if app.pid_motion_enabled_var.get():
        app.status_var.set("FK target staged. Click Apply FK Target to start PID motion.")
        return

    app._set_joint_angles(joint_angles)
    update_output(app)
    app.status_var.set("Direct FK mode: sliders/input boxes move the robot live.")


def apply_target(app):
    """Apply the joint angles currently shown in the FK sliders."""
    joint_angles = np.radians(app._read_slider_values(app.fk_q_controls))
    if not accept_slider_angles(app, joint_angles):
        return
    app._apply_joint_target(joint_angles, "FK sliders")


def accept_slider_angles(app, joint_angles):
    """Check if FK angles are safe. If not, revert the sliders."""
    safety_message = safe_joint_message(joint_angles)
    if safety_message:
        set_input_degrees(app, app.last_safe_fk_angles)
        app.status_var.set(safety_message)
        return False
    app.last_safe_fk_angles = wrap_angles(joint_angles)
    return True


def set_input_degrees(app, joint_angles):
    """Push angle values into the FK sliders without triggering callbacks."""
    app._pause_callbacks()
    for control, value in zip(app.fk_q_controls, np.degrees(joint_angles)):
        control.set(value)
    app._resume_callbacks()


def update_output(app):
    """Refresh the FK text panel with the current robot state."""
    joint_angles = app._current_joint_angles()
    position = forward_kinematics(joint_angles)
    positions = joint_positions(joint_angles)
    robot_jacobian = jacobian(joint_angles)

    output = [
        "Current joint angles:",
        f"  radians: {app._format_vector(joint_angles)}",
        f"  degrees: {app._format_vector(np.degrees(joint_angles))}",
        "",
        "Link lengths (m):",
        f"  upper arm, forearm, tool: {app._format_vector(LINK_LENGTHS)}",
        "",
        "End-effector position:",
        f"  x = {position[0]:.6f} m",
        f"  y = {position[1]:.6f} m",
        f"  z = {position[2]:.6f} m",
        "",
        "Robot points [x, y, z]:",
        "  rows: floor base, shoulder, elbow, wrist, end effector",
        app._format_matrix(positions),
        "",
        "Jacobian d[x, y, z] / d[q1, q2, q3]:",
        app._format_matrix(robot_jacobian),
    ]
    app._set_text(app.fk_output, "\n".join(output))

    # Also refresh the PID live values panel since joint angles changed.
    from ui import pid_actions
    pid_actions.update_live_values(app)
