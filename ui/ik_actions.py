"""Inverse Kinematics tab actions — handles IK slider and button events."""

import numpy as np
from tkinter import messagebox

from kinematics.forward import forward_kinematics, joint_positions
from kinematics.inverse import inverse_kinematics


def on_target_slider_changed(app):
    """Called when an IK target slider moves."""
    target_position = read_target_position(app)
    app.last_ik_solution = None
    app._set_target_marker(target_position)
    app._set_text(
        app.ik_output,
        "Target marker moved.\n\nClick Solve IK to compute joint angles for this target.",
    )
    app.status_var.set("IK target marker updated from sliders.")


def solve(app):
    """Run the IK solver and display the result."""
    target_position = read_target_position(app)

    app._set_target_marker(target_position)
    result = inverse_kinematics(
        target_position,
        initial_guess=app._current_joint_angles(),
    )
    app.last_ik_solution = result

    solution_position = forward_kinematics(result.joint_angles)
    solution_points = joint_positions(result.joint_angles)
    output = [
        result.message,
        "",
        f"Converged: {result.converged}",
        f"Iterations: {result.iterations}",
        f"Final error norm: {result.error_norm:.6f}",
        f"Final position error [x, y, z]: {app._format_vector(result.final_error)}",
        f"Elbow height: {solution_points[2, 2]:.6f} m",
        "",
        "Solved joint angles:",
        f"  radians: {app._format_vector(result.joint_angles)}",
        f"  degrees: {app._format_vector(np.degrees(result.joint_angles))}",
        "",
        "Position produced by solution:",
        f"  x, y, z: {app._format_vector(solution_position)}",
    ]
    app._set_text(app.ik_output, "\n".join(output))
    app.status_var.set(result.message)
    if not result.converged:
        app.last_ik_solution = None


def apply_solution(app):
    """Apply the last IK solution to the robot."""
    if app.last_ik_solution is None or not app.last_ik_solution.converged:
        messagebox.showinfo("IK", "Solve IK before applying a solution.")
        return
    app._apply_joint_target(app.last_ik_solution.joint_angles, "IK solution")


def set_input_position(app, target_position):
    """Push position values into the IK sliders without triggering callbacks."""
    app._pause_callbacks()
    app.ik_x_control.set(target_position[0])
    app.ik_y_control.set(target_position[1])
    app.ik_z_control.set(target_position[2])
    app._resume_callbacks()


def read_target_position(app):
    """Read the three IK target sliders as a [x, y, z] array."""
    return np.array(
        [
            app.ik_x_control.get(),
            app.ik_y_control.get(),
            app.ik_z_control.get(),
        ],
        dtype=float,
    )
