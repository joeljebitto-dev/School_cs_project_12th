"""Main Tkinter application for the 3D kinematics and PID demo."""

from __future__ import annotations

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import (
    DEFAULT_PID_GAINS,
    DEFAULT_PID_TARGET_DEGREES,
    DEFAULT_TARGET_POSITION,
    PID_PLOT_MAX_POINTS,
    UI_UPDATE_INTERVAL_MS,
)
from control.pid import PIDController, PIDHistory
from simulation.mujoco_sim import RobotSimulation
from ui.handlers import AppHandlersMixin
from ui.layout import build_app_layout, configure_window
from ui.style import configure_app_style


class KinematicsPidApp(AppHandlersMixin):
    """Coordinate Tkinter controls, kinematics math, PID, and MuJoCo."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("3D 3-DOF MuJoCo Kinematics and PID Demo")
        configure_window(self.root)
        configure_app_style(self.root)

        self.simulation = RobotSimulation()
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

        self.equation_canvases: list[FigureCanvasTkAgg] = []
        self._updating_sliders = False
        self.pid_motion_enabled_var = tk.BooleanVar(value=False)
        self.pid_run_button_text = tk.StringVar(value="Run PID")
        self.status_var = tk.StringVar(
            value="Ready. Open the MuJoCo viewer to see the 3D arm."
        )
        self.pid_status_var = tk.StringVar(value="PID is paused.")
        self.pid_live_values_var = tk.StringVar()

        layout = build_app_layout(
            self.root,
            status_var=self.status_var,
            pid_motion_enabled_var=self.pid_motion_enabled_var,
            pid_run_button_text=self.pid_run_button_text,
            pid_status_var=self.pid_status_var,
            pid_live_values_var=self.pid_live_values_var,
            equation_canvases=self.equation_canvases,
            on_open_viewer=self.open_mujoco_viewer,
            on_reset=self.reset_simulation,
            on_pid_motion_toggled=self._on_pid_motion_toggled,
            callbacks_suspended=self._callbacks_suspended,
            on_fk_slider_changed=self._on_fk_slider_changed,
            on_apply_fk_target=self.apply_fk_target,
            on_ik_target_changed=self._on_ik_target_slider_changed,
            on_solve_ik=self._solve_ik,
            on_apply_ik_solution=self._apply_ik_solution,
            on_pid_target_changed=self._on_pid_target_changed,
            on_pid_gain_changed=self._on_pid_gain_changed,
            on_toggle_pid=self.toggle_pid_control,
            on_hold_current=self.hold_current_position,
        )

        self.fk_q_controls = layout.fk_q_controls
        self.fk_output = layout.fk_output
        self.fk_apply_row = layout.fk_apply_row
        self.ik_x_control = layout.ik_x_control
        self.ik_y_control = layout.ik_y_control
        self.ik_z_control = layout.ik_z_control
        self.ik_output = layout.ik_output
        self.pid_q_controls = layout.pid_q_controls
        self.pid_joint_gains = layout.pid_joint_gains
        self.pid_axis = layout.pid_axis
        self.error_line = layout.error_line
        self.torque_line = layout.torque_line
        self.pid_canvas = layout.pid_canvas

        self._update_pid_mode_controls()
        self._set_joint_angles(np.zeros(3))
        self._set_target_marker(DEFAULT_TARGET_POSITION)
        self._set_ik_input_position(DEFAULT_TARGET_POSITION)
        self._update_fk_output()
        self._update_pid_live_values()
        self._reset_pid_plot()

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(UI_UPDATE_INTERVAL_MS, self.update_simulation)

    def run(self) -> None:
        """Start the Tkinter event loop."""

        self.root.mainloop()

    def close(self) -> None:
        """Close the viewer and the Tkinter window."""

        self.pid_running = False
        self.simulation.close_viewer()
        self.root.destroy()
