"""Event handlers, PID logic, and helper methods for the application.

This module defines ``AppHandlersMixin``, a mixin class that provides every
callback and helper used by ``KinematicsPidApp``.  The mixin expects the
host class to set the attributes created in ``KinematicsPidApp.__init__``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import numpy as np
import tkinter as tk
from tkinter import messagebox

from config import (
    DEFAULT_PID_GAINS,
    DEFAULT_PID_TARGET_DEGREES,
    DEFAULT_TARGET_POSITION,
    LINK_LENGTHS,
    UI_UPDATE_INTERVAL_MS,
)
from kinematics.common import wrap_angles
from kinematics.forward import forward_kinematics, jacobian, joint_positions
from kinematics.inverse import inverse_kinematics, safe_joint_message
from simulation.mujoco_sim import PidStepSample
from ui.widgets import SliderControl


class AppHandlersMixin:
    """Mixin supplying all event-handler and helper methods.

    Designed to be inherited by ``KinematicsPidApp`` alongside any other
    bases.  Every method accesses state through ``self``, relying on the
    attributes initialised in ``KinematicsPidApp.__init__``.
    """

    # ------------------------------------------------------------------
    # Callback guard
    # ------------------------------------------------------------------

    def _callbacks_suspended(self) -> bool:
        return self._updating_sliders

    # ------------------------------------------------------------------
    # PID-mode UI helpers
    # ------------------------------------------------------------------

    def _update_pid_mode_controls(self) -> None:
        """Show FK Apply only when FK targets should go through PID."""

        if not hasattr(self, "fk_apply_row"):
            return

        if self.pid_motion_enabled_var.get():
            self.fk_apply_row.grid()
        else:
            self.fk_apply_row.grid_remove()

    # ------------------------------------------------------------------
    # Top-bar button handlers
    # ------------------------------------------------------------------

    def open_mujoco_viewer(self) -> None:
        """Button callback that opens the MuJoCo viewer window."""

        try:
            opened = self.simulation.open_viewer()
        except Exception as error:
            messagebox.showerror(
                "MuJoCo Viewer Error",
                f"Could not open the MuJoCo viewer:\n{error}",
            )
            return

        if not opened:
            messagebox.showinfo("MuJoCo Viewer", "The MuJoCo viewer is already open.")
            return

        self.status_var.set("MuJoCo viewer opened.")

    def reset_simulation(self) -> None:
        """Reset the robot, target marker, PID state, and displayed values."""

        self.pid_running = False
        self.pid_run_button_text.set("Run PID")
        self.pid_motion_enabled_var.set(False)
        self.pid_controller.reset()
        self.pid_target_angles = np.radians(DEFAULT_PID_TARGET_DEGREES)
        self.last_safe_fk_angles = np.zeros(3)
        self.last_safe_pid_target_angles = self.pid_target_angles.copy()
        self._set_joint_angles(np.zeros(3))
        self._set_fk_input_degrees(np.zeros(3))
        self._set_pid_input_degrees(self.pid_target_angles)
        self._set_ik_input_position(DEFAULT_TARGET_POSITION)
        self._set_target_marker(DEFAULT_TARGET_POSITION)
        self._clear_pid_history()
        self._update_fk_output()
        self.pid_status_var.set("PID is paused.")
        self.status_var.set("Robot reset to zero joint angles.")
        self._update_pid_mode_controls()
        self._update_pid_live_values()

    # ------------------------------------------------------------------
    # Forward-kinematics handlers
    # ------------------------------------------------------------------

    def apply_fk_target(self) -> None:
        """Apply the joint angles currently shown in the FK sliders."""

        joint_angles = np.radians(self._read_slider_values(self.fk_q_controls))
        if not self._accept_fk_slider_angles(joint_angles):
            return

        self._apply_joint_target(joint_angles, "FK sliders")

    def _on_fk_slider_changed(self) -> None:
        joint_angles = np.radians(self._read_slider_values(self.fk_q_controls))
        if not self._accept_fk_slider_angles(joint_angles):
            return

        if self.pid_motion_enabled_var.get():
            self.status_var.set("FK target staged. Click Apply FK Target to start PID motion.")
            return

        self._set_joint_angles(joint_angles)
        self._update_fk_output()
        self.status_var.set("Direct FK mode: sliders/input boxes move the robot live.")

    # ------------------------------------------------------------------
    # Inverse-kinematics handlers
    # ------------------------------------------------------------------

    def _solve_ik(self) -> None:
        target_position = self._read_ik_target_position()

        self._set_target_marker(target_position)
        result = inverse_kinematics(
            target_position,
            initial_guess=self._current_joint_angles(),
        )
        self.last_ik_solution = result

        solution_position = forward_kinematics(result.joint_angles)
        solution_points = joint_positions(result.joint_angles)
        output = [
            result.message,
            "",
            f"Converged: {result.converged}",
            f"Iterations: {result.iterations}",
            f"Final error norm: {result.error_norm:.6f}",
            f"Final position error [x, y, z]: {self._format_vector(result.final_error)}",
            f"Elbow height: {solution_points[2, 2]:.6f} m",
            "",
            "Solved joint angles:",
            f"  radians: {self._format_vector(result.joint_angles)}",
            f"  degrees: {self._format_vector(np.degrees(result.joint_angles))}",
            "",
            "Position produced by solution:",
            f"  x, y, z: {self._format_vector(solution_position)}",
        ]
        self._set_text(self.ik_output, "\n".join(output))
        self.status_var.set(result.message)
        if not result.converged:
            self.last_ik_solution = None

    def _on_ik_target_slider_changed(self) -> None:
        target_position = self._read_ik_target_position()
        self.last_ik_solution = None
        self._set_target_marker(target_position)
        self._set_text(
            self.ik_output,
            "Target marker moved.\n\nClick Solve IK to compute joint angles for this target.",
        )
        self.status_var.set("IK target marker updated from sliders.")

    def _apply_ik_solution(self) -> None:
        if self.last_ik_solution is None or not self.last_ik_solution.converged:
            messagebox.showinfo("IK", "Solve IK before applying a solution.")
            return

        self._apply_joint_target(self.last_ik_solution.joint_angles, "IK solution")

    # ------------------------------------------------------------------
    # PID handlers
    # ------------------------------------------------------------------

    def toggle_pid_control(self) -> None:
        """Start PID if paused, or pause it if already running."""

        if self.pid_running:
            self.pause_pid()
            return

        target_angles = self._read_pid_target_angles()
        self.start_pid_motion(
            target_angles,
            "PID tab target",
            sync_sliders=False,
            reset_integral=True,
        )

    def _on_pid_target_changed(self) -> None:
        target_angles = self._read_pid_target_angles()
        if not self._accept_pid_slider_angles(target_angles):
            return

        self.start_pid_motion(
            target_angles,
            "PID target input",
            sync_sliders=False,
            reset_integral=True,
        )

    def _on_pid_gain_changed(self) -> None:
        self._update_pid_gains()
        self._update_pid_live_values()
        if self.pid_running:
            self.status_var.set("PID gains updated while the controller is running.")
        else:
            self.status_var.set("PID gains updated.")

    def _update_pid_gains(self) -> None:
        kp_arr, ki_arr, kd_arr = self._read_pid_gains()
        self.pid_controller.kp = kp_arr
        self.pid_controller.ki = ki_arr
        self.pid_controller.kd = kd_arr

    def start_pid_motion(
        self,
        joint_angles: np.ndarray,
        source: str,
        sync_sliders: bool,
        reset_integral: bool,
    ) -> bool:
        """Set a PID target and immediately start smooth motion."""

        if not self._set_pid_target(joint_angles, sync_sliders, reset_integral):
            return False

        self.pid_motion_enabled_var.set(True)
        self.pid_running = True
        self.pid_run_button_text.set("Pause PID")
        self.pid_status_var.set(f"PID moving toward {source}.")
        self.status_var.set(f"{source} moving with PID.")
        self._update_pid_mode_controls()
        self._update_pid_live_values()
        return True

    def pause_pid(self) -> None:
        """Pause the PID controller and remove motor torque."""

        self.pid_running = False
        self.simulation.stop_motors()
        self.pid_run_button_text.set("Run PID")
        self.pid_status_var.set("PID is paused.")
        self.status_var.set("PID controller paused.")
        self._update_pid_live_values()

    def hold_current_position(self) -> None:
        """Use the current robot pose as the PID target."""

        current_angles = self._current_joint_angles()
        if not self._set_pid_target(current_angles, sync_sliders=True, reset_integral=True):
            return

        self.pid_motion_enabled_var.set(True)
        self.status_var.set("PID target set to the current robot pose.")
        self._update_pid_mode_controls()
        self._update_pid_live_values()

    def _on_pid_motion_toggled(self) -> None:
        if self.pid_motion_enabled_var.get():
            self._set_pid_target(
                self._current_joint_angles(),
                sync_sliders=True,
                reset_integral=True,
            )
            self.status_var.set("PID motion enabled. Press Run PID when ready.")
            self._update_pid_mode_controls()
            self._update_pid_live_values()
            return

        self.pause_pid()
        self.status_var.set("PID motion disabled. FK sliders are direct again.")
        self._update_pid_mode_controls()

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def update_simulation(self) -> None:
        """Step the simulation once and schedule the next Tkinter update."""

        try:
            if self.pid_running:
                sample = self.simulation.step_pid_control(
                    self.pid_controller,
                    self.pid_target_angles,
                )
                self._record_pid_sample(sample)
                self._update_fk_output()
            else:
                self.simulation.stop_motors()
                self._update_pid_live_values()

            self.simulation.sync_viewer()
        finally:
            self.root.after(UI_UPDATE_INTERVAL_MS, self.update_simulation)

    # ------------------------------------------------------------------
    # PID plot helpers
    # ------------------------------------------------------------------

    def _record_pid_sample(self, sample: PidStepSample) -> None:
        self.pid_history.record(sample.elapsed_time, sample.error_norm, sample.torque_norm)
        self.pid_status_var.set(
            "PID running. "
            f"error norm = {sample.error_norm:.4f}, "
            f"torque norm = {sample.torque_norm:.4f}"
        )
        self._update_pid_plot()

    def _update_pid_plot(self) -> None:
        times = self.pid_history.time_history
        errors = self.pid_history.error_history
        torques = self.pid_history.torque_history

        self.error_line.set_data(times, errors)
        self.torque_line.set_data(times, torques)

        if times:
            t_min, t_max = times[0], times[-1]
            if t_max <= t_min:
                t_max = t_min + 1.0
            self.pid_axis.set_xlim(t_min, t_max)

            all_values = errors + torques
            y_max = max(all_values) if all_values else 1.0
            y_max = max(y_max, 0.1)  # ensure a visible range
            self.pid_axis.set_ylim(0.0, y_max * 1.1)

        self.pid_canvas.draw_idle()

    def _reset_pid_plot(self) -> None:
        self.error_line.set_data([], [])
        self.torque_line.set_data([], [])
        self.pid_axis.set_xlim(0.0, 5.0)
        self.pid_axis.set_ylim(0.0, 1.0)
        self.pid_canvas.draw_idle()

    def _clear_pid_history(self) -> None:
        self.pid_history.reset()
        self._reset_pid_plot()

    # ------------------------------------------------------------------
    # FK / PID output updates
    # ------------------------------------------------------------------

    def _update_fk_output(self) -> None:
        joint_angles = self._current_joint_angles()
        position = forward_kinematics(joint_angles)
        positions = joint_positions(joint_angles)
        robot_jacobian = jacobian(joint_angles)

        output = [
            "Current joint angles:",
            f"  radians: {self._format_vector(joint_angles)}",
            f"  degrees: {self._format_vector(np.degrees(joint_angles))}",
            "",
            "Link lengths (m):",
            f"  upper arm, forearm, tool: {self._format_vector(LINK_LENGTHS)}",
            "",
            "End-effector position:",
            f"  x = {position[0]:.6f} m",
            f"  y = {position[1]:.6f} m",
            f"  z = {position[2]:.6f} m",
            "",
            "Robot points [x, y, z]:",
            "  rows: floor base, shoulder, elbow, wrist, end effector",
            self._format_matrix(positions),
            "",
            "Jacobian d[x, y, z] / d[q1, q2, q3]:",
            self._format_matrix(robot_jacobian),
        ]
        self._set_text(self.fk_output, "\n".join(output))
        self._update_pid_live_values()

    def _update_pid_live_values(self) -> None:
        current_angles = self._current_joint_angles()
        target_angles = self.pid_target_angles
        error_angles = wrap_angles(target_angles - current_angles)
        torques = self.simulation.current_motor_torques()
        end_effector_position = forward_kinematics(current_angles)
        state = "running" if self.pid_running else "paused"
        gravity_state = "on" if self.simulation.gravity_is_enabled() else "off"

        output = [
            f"State: {state}",
            f"Gravity: {gravity_state}",
            f"Current q (deg): {self._format_vector(np.degrees(current_angles))}",
            f"Target q  (deg): {self._format_vector(np.degrees(target_angles))}",
            f"Error q   (deg): {self._format_vector(np.degrees(error_angles))}",
            f"Torque command:  {self._format_vector(torques)}",
            f"End effector m:  {self._format_vector(end_effector_position)}",
        ]
        self.pid_live_values_var.set("\n".join(output))

    # ------------------------------------------------------------------
    # Simulation wrappers
    # ------------------------------------------------------------------

    def _set_joint_angles(self, joint_angles: np.ndarray) -> None:
        self.simulation.set_joint_angles(joint_angles)

    def _set_target_marker(self, target_position: np.ndarray) -> None:
        self.simulation.set_target_marker(target_position)

    def _current_joint_angles(self) -> np.ndarray:
        return self.simulation.current_joint_angles()

    # ------------------------------------------------------------------
    # Slider / input helpers
    # ------------------------------------------------------------------

    def _set_fk_input_degrees(self, joint_angles: np.ndarray) -> None:
        with self._suspend_slider_callbacks():
            for control, value in zip(
                self.fk_q_controls,
                np.degrees(joint_angles),
                strict=True,
            ):
                control.set(value)

    def _set_ik_input_position(self, target_position: np.ndarray) -> None:
        with self._suspend_slider_callbacks():
            self.ik_x_control.set(target_position[0])
            self.ik_y_control.set(target_position[1])
            self.ik_z_control.set(target_position[2])

    def _set_pid_input_degrees(self, joint_angles: np.ndarray) -> None:
        with self._suspend_slider_callbacks():
            for control, value in zip(
                self.pid_q_controls,
                np.degrees(joint_angles),
                strict=True,
            ):
                control.set(value)

    def _read_ik_target_position(self) -> np.ndarray:
        return np.array(
            [
                self.ik_x_control.get(),
                self.ik_y_control.get(),
                self.ik_z_control.get(),
            ],
            dtype=float,
        )

    def _read_pid_target_angles(self) -> np.ndarray:
        return np.radians(self._read_slider_values(self.pid_q_controls))

    def _read_pid_gains(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kp = np.array([jg.kp.get() for jg in self.pid_joint_gains])
        ki = np.array([jg.ki.get() for jg in self.pid_joint_gains])
        kd = np.array([jg.kd.get() for jg in self.pid_joint_gains])
        return kp, ki, kd

    # ------------------------------------------------------------------
    # Safety validation
    # ------------------------------------------------------------------

    def _accept_fk_slider_angles(self, joint_angles: np.ndarray) -> bool:
        safety_message = safe_joint_message(joint_angles)
        if safety_message:
            self._set_fk_input_degrees(self.last_safe_fk_angles)
            self.status_var.set(safety_message)
            return False

        self.last_safe_fk_angles = wrap_angles(joint_angles)
        return True

    def _accept_pid_slider_angles(self, joint_angles: np.ndarray) -> bool:
        safety_message = safe_joint_message(joint_angles)
        if safety_message:
            self._set_pid_input_degrees(self.last_safe_pid_target_angles)
            self.status_var.set(safety_message)
            return False

        self.last_safe_pid_target_angles = wrap_angles(joint_angles)
        return True

    def _apply_joint_target(self, joint_angles: np.ndarray, source: str) -> bool:
        safety_message = safe_joint_message(joint_angles)
        if safety_message:
            self.status_var.set(safety_message)
            return False

        if self.pid_motion_enabled_var.get():
            return self.start_pid_motion(
                joint_angles,
                source,
                sync_sliders=True,
                reset_integral=True,
            )

        self._set_joint_angles(joint_angles)
        self._set_fk_input_degrees(joint_angles)
        self._update_fk_output()
        self.status_var.set(f"{source} applied directly to the robot.")
        return True

    def _set_pid_target(
        self,
        joint_angles: np.ndarray,
        sync_sliders: bool,
        reset_integral: bool,
    ) -> bool:
        safety_message = safe_joint_message(joint_angles)
        if safety_message:
            self.status_var.set(safety_message)
            return False

        self.pid_target_angles = wrap_angles(joint_angles)
        self.last_safe_pid_target_angles = self.pid_target_angles.copy()
        self._update_pid_gains()
        if reset_integral:
            self.pid_controller.reset()
        if sync_sliders:
            self._set_pid_input_degrees(self.pid_target_angles)
        return True

    # ------------------------------------------------------------------
    # Generic utilities
    # ------------------------------------------------------------------

    def _read_slider_values(
        self,
        controls: Iterable[SliderControl],
    ) -> np.ndarray:
        return np.array([control.get() for control in controls], dtype=float)

    @contextmanager
    def _suspend_slider_callbacks(self):
        previous_state = self._updating_sliders
        self._updating_sliders = True
        try:
            yield
        finally:
            self._updating_sliders = previous_state

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def _format_vector(self, values: np.ndarray) -> str:
        return "[" + ", ".join(f"{value: .6f}" for value in values) + "]"

    def _format_matrix(self, values: np.ndarray) -> str:
        rows = []
        for row in values:
            rows.append("  [" + ", ".join(f"{value: .6f}" for value in row) + "]")
        return "\n".join(rows)
