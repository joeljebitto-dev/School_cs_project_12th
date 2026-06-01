"""Main Tkinter application for the 3D kinematics and PID demo."""

import numpy as np
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import (
    DEFAULT_PID_GAINS,
    DEFAULT_PID_TARGET_DEGREES,
    DEFAULT_TARGET_POSITION,
    PID_PLOT_MAX_POINTS,
    UI_UPDATE_INTERVAL_MS,
)
from control.pid import PIDController, PIDHistory
from kinematics.inverse import safe_joint_message
from simulation.mujoco_sim import RobotSimulation
from ui import fk_actions, ik_actions, pid_actions
from ui.layout import build_app_layout, configure_window
from ui.style import configure_app_style


class KinematicsPidApp:
    """Coordinate Tkinter controls, kinematics math, PID, and MuJoCo."""

    def __init__(self, root):
        self.root = root
        self.root.title("3D 3-DOF MuJoCo Kinematics and PID Demo")
        configure_window(self.root)
        configure_app_style(self.root)

        # Set up the MuJoCo simulation.
        self.simulation = RobotSimulation()

        # Set up the PID controller with per-joint default gains.
        default_kp = np.array([DEFAULT_PID_GAINS[f"joint{i+1}"][0] for i in range(3)])
        default_ki = np.array([DEFAULT_PID_GAINS[f"joint{i+1}"][1] for i in range(3)])
        default_kd = np.array([DEFAULT_PID_GAINS[f"joint{i+1}"][2] for i in range(3)])
        self.pid_controller = PIDController(kp=default_kp, ki=default_ki, kd=default_kd)
        self.pid_history = PIDHistory(max_points=PID_PLOT_MAX_POINTS)
        self.pid_running = False
        self.pid_target_angles = np.radians(DEFAULT_PID_TARGET_DEGREES)
        self.last_safe_fk_angles = np.zeros(3)
        self.last_safe_pid_target_angles = self.pid_target_angles.copy()
        self.last_ik_solution = None

        # Tkinter variables shared with the UI.
        self.equation_canvases: list[FigureCanvasTkAgg] = []
        self._updating_sliders = False
        self.pid_motion_enabled_var = tk.BooleanVar(value=False)
        self.pid_run_button_text = tk.StringVar(value="Run PID")
        self.status_var = tk.StringVar(
            value="Ready. Open the MuJoCo viewer to see the 3D arm."
        )
        self.pid_status_var = tk.StringVar(value="PID is paused.")
        self.pid_live_values_var = tk.StringVar()

        # Build the full UI (top bar, tabs, status bar).
        build_app_layout(self)

        # Set the starting state.
        pid_actions.update_mode_controls(self)
        self._set_joint_angles(np.zeros(3))
        self._set_target_marker(DEFAULT_TARGET_POSITION)
        ik_actions.set_input_position(self, DEFAULT_TARGET_POSITION)
        fk_actions.update_output(self)
        pid_actions.update_live_values(self)
        pid_actions.reset_plot(self)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(UI_UPDATE_INTERVAL_MS, self.update_simulation)

    def run(self):
        """Start the Tkinter event loop."""
        self.root.mainloop()

    def close(self):
        """Close the viewer and the Tkinter window."""
        self.pid_running = False
        self.simulation.close_viewer()
        self.root.destroy()

    # ------------------------------------------------------------------
    # FK tab actions (delegated to fk_actions.py)
    # ------------------------------------------------------------------

    def apply_fk_target(self):
        fk_actions.apply_target(self)

    def _on_fk_slider_changed(self):
        fk_actions.on_slider_changed(self)

    # ------------------------------------------------------------------
    # IK tab actions (delegated to ik_actions.py)
    # ------------------------------------------------------------------

    def _solve_ik(self):
        ik_actions.solve(self)

    def _on_ik_target_slider_changed(self):
        ik_actions.on_target_slider_changed(self)

    def _apply_ik_solution(self):
        ik_actions.apply_solution(self)

    # ------------------------------------------------------------------
    # PID tab actions (delegated to pid_actions.py)
    # ------------------------------------------------------------------

    def toggle_pid_control(self):
        pid_actions.toggle(self)

    def _on_pid_target_changed(self):
        pid_actions.on_target_changed(self)

    def _on_pid_gain_changed(self):
        pid_actions.on_gain_changed(self)

    def start_pid_motion(self, joint_angles, source, sync_sliders, reset_integral):
        return pid_actions.start_motion(self, joint_angles, source, sync_sliders, reset_integral)

    def pause_pid(self):
        pid_actions.pause(self)

    def hold_current_position(self):
        pid_actions.hold_current(self)

    def _on_pid_motion_toggled(self):
        pid_actions.on_motion_toggled(self)

    def _update_pid_mode_controls(self):
        pid_actions.update_mode_controls(self)

    # ------------------------------------------------------------------
    # Top-bar buttons
    # ------------------------------------------------------------------

    def open_mujoco_viewer(self):
        """Open the 3D MuJoCo viewer window."""
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

    def reset_simulation(self):
        """Reset the robot, target marker, PID state, and all displayed values."""
        self.pid_running = False
        self.pid_run_button_text.set("Run PID")
        self.pid_motion_enabled_var.set(False)
        self.pid_controller.reset()
        self.pid_target_angles = np.radians(DEFAULT_PID_TARGET_DEGREES)
        self.last_safe_fk_angles = np.zeros(3)
        self.last_safe_pid_target_angles = self.pid_target_angles.copy()
        self._set_joint_angles(np.zeros(3))
        fk_actions.set_input_degrees(self, np.zeros(3))
        pid_actions.set_input_degrees(self, self.pid_target_angles)
        ik_actions.set_input_position(self, DEFAULT_TARGET_POSITION)
        self._set_target_marker(DEFAULT_TARGET_POSITION)
        pid_actions.clear_history(self)
        fk_actions.update_output(self)
        self.pid_status_var.set("PID is paused.")
        self.status_var.set("Robot reset to zero joint angles.")
        pid_actions.update_mode_controls(self)
        pid_actions.update_live_values(self)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def update_simulation(self):
        """Step the simulation once and schedule the next Tkinter update."""
        try:
            if self.pid_running:
                sample = self.simulation.step_pid_control(
                    self.pid_controller,
                    self.pid_target_angles,
                )
                pid_actions.record_sample(self, sample)
                fk_actions.update_output(self)
            else:
                self.simulation.stop_motors()
                pid_actions.update_live_values(self)

            self.simulation.sync_viewer()
        finally:
            self.root.after(UI_UPDATE_INTERVAL_MS, self.update_simulation)

    # ------------------------------------------------------------------
    # Shared helpers used by the action files
    # ------------------------------------------------------------------

    def _callbacks_suspended(self):
        """Returns True when slider callbacks should be ignored."""
        return self._updating_sliders

    def _pause_callbacks(self):
        """Temporarily stop slider callbacks from firing."""
        self._updating_sliders = True

    def _resume_callbacks(self):
        """Re-enable slider callbacks."""
        self._updating_sliders = False

    def _set_joint_angles(self, joint_angles):
        self.simulation.set_joint_angles(joint_angles)

    def _set_target_marker(self, target_position):
        self.simulation.set_target_marker(target_position)

    def _current_joint_angles(self):
        return self.simulation.current_joint_angles()

    def _read_slider_values(self, controls):
        """Read a list of slider controls into a numpy array."""
        return np.array([control.get() for control in controls], dtype=float)

    def _apply_joint_target(self, joint_angles, source):
        """Apply joint angles either directly or through PID depending on mode."""
        safety_message = safe_joint_message(joint_angles)
        if safety_message:
            self.status_var.set(safety_message)
            return False

        if self.pid_motion_enabled_var.get():
            return self.start_pid_motion(
                joint_angles, source, sync_sliders=True, reset_integral=True,
            )

        self._set_joint_angles(joint_angles)
        fk_actions.set_input_degrees(self, joint_angles)
        fk_actions.update_output(self)
        self.status_var.set(f"{source} applied directly to the robot.")
        return True

    def _set_text(self, widget, text):
        """Replace all text in a read-only Text widget."""
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def _format_vector(self, values):
        """Format a numpy array like [ 0.1,  0.2,  0.3]."""
        return "[" + ", ".join(f"{v: .6f}" for v in values) + "]"

    def _format_matrix(self, values):
        """Format a 2D numpy array as indented rows."""
        rows = []
        for row in values:
            rows.append("  [" + ", ".join(f"{v: .6f}" for v in row) + "]")
        return "\n".join(rows)
