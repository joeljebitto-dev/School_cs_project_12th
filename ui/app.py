"""Main Tkinter application for the 3D kinematics and PID demo."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, ttk

from config import (
    DEFAULT_PID_GAINS,
    DEFAULT_PID_TARGET_DEGREES,
    DEFAULT_TARGET_POSITION,
    LINK_LENGTHS,
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    PID_PLOT_MAX_POINTS,
    UI_COLORS,
    UI_UPDATE_INTERVAL_MS,
    WINDOW_HEIGHT_FRACTION,
    WINDOW_WIDTH_FRACTION,
)
from control.pid import PIDController, PIDHistory
from kinematics.common import wrap_angles
from kinematics.forward import forward_kinematics, jacobian, joint_positions
from kinematics.inverse import inverse_kinematics, safe_joint_message
from simulation.mujoco_sim import PidStepSample, RobotSimulation
from ui.forward_tab import build_forward_tab
from ui.inverse_tab import build_inverse_tab
from ui.pid_tab import build_pid_tab
from ui.widgets import ScrollableFrame, SliderControl


class KinematicsPidApp:
    """Coordinate Tkinter controls, kinematics math, PID, and MuJoCo."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("3D 3-DOF MuJoCo Kinematics and PID Demo")
        self._configure_window()
        self._configure_style()

        self.simulation = RobotSimulation()
        default_kp, default_ki, default_kd = DEFAULT_PID_GAINS
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

        self._build_layout()
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

    def _configure_window(self) -> None:
        screen_width = max(self.root.winfo_screenwidth(), 800)
        screen_height = max(self.root.winfo_screenheight(), 600)
        max_width = max(640, screen_width - 40)
        max_height = max(520, screen_height - 80)
        width = min(max(int(screen_width * WINDOW_WIDTH_FRACTION), MIN_WINDOW_WIDTH), max_width)
        height = min(max(int(screen_height * WINDOW_HEIGHT_FRACTION), MIN_WINDOW_HEIGHT), max_height)
        x = max((screen_width - width) // 2, 0)
        y = max((screen_height - height) // 2, 0)

        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(min(MIN_WINDOW_WIDTH, width), min(MIN_WINDOW_HEIGHT, height))

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.root.configure(bg=UI_COLORS["bg"])
        self.root.option_add("*Font", ("TkDefaultFont", 10))

        style.configure("TFrame", background=UI_COLORS["bg"])
        style.configure("App.TFrame", background=UI_COLORS["bg"])
        style.configure("Card.TFrame", background=UI_COLORS["panel"])
        style.configure("TLabel", background=UI_COLORS["bg"], foreground=UI_COLORS["text"])
        style.configure(
            "Title.TLabel",
            background=UI_COLORS["bg"],
            foreground=UI_COLORS["text"],
            font=("TkDefaultFont", 16, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=UI_COLORS["bg"],
            foreground=UI_COLORS["muted"],
        )
        style.configure(
            "Muted.TLabel",
            background=UI_COLORS["panel"],
            foreground=UI_COLORS["muted"],
        )
        style.configure(
            "Card.TLabel",
            background=UI_COLORS["panel"],
            foreground=UI_COLORS["text"],
        )
        style.configure(
            "Status.TLabel",
            padding=(12, 8),
            background=UI_COLORS["panel"],
            foreground=UI_COLORS["muted"],
        )
        style.configure(
            "TNotebook",
            background=UI_COLORS["bg"],
            borderwidth=0,
            tabmargins=(0, 4, 0, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background=UI_COLORS["panel_alt"],
            foreground=UI_COLORS["muted"],
            padding=(16, 9),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", UI_COLORS["panel"])],
            foreground=[("selected", UI_COLORS["text"])],
        )
        style.configure(
            "Card.TLabelframe",
            background=UI_COLORS["panel"],
            foreground=UI_COLORS["text"],
            bordercolor=UI_COLORS["border"],
            relief=tk.SOLID,
            padding=12,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=UI_COLORS["panel"],
            foreground=UI_COLORS["text"],
            font=("TkDefaultFont", 10, "bold"),
        )
        style.configure(
            "TButton",
            background=UI_COLORS["panel_alt"],
            foreground=UI_COLORS["text"],
            bordercolor=UI_COLORS["border"],
            focusthickness=1,
            focuscolor=UI_COLORS["accent"],
            padding=(12, 6),
        )
        style.map(
            "TButton",
            background=[("active", UI_COLORS["border"])],
            foreground=[("disabled", UI_COLORS["muted"])],
        )
        style.configure(
            "Accent.TButton",
            background=UI_COLORS["accent_dark"],
            foreground=UI_COLORS["text"],
            padding=(12, 6),
        )
        style.map("Accent.TButton", background=[("active", UI_COLORS["accent"])])
        style.configure(
            "TCheckbutton",
            background=UI_COLORS["bg"],
            foreground=UI_COLORS["text"],
        )
        style.map(
            "TCheckbutton",
            background=[("active", UI_COLORS["bg"])],
            foreground=[("active", UI_COLORS["text"])],
        )
        style.configure(
            "Dark.TEntry",
            fieldbackground=UI_COLORS["entry_bg"],
            foreground=UI_COLORS["text"],
            insertcolor=UI_COLORS["text"],
            bordercolor=UI_COLORS["border"],
        )
        style.configure(
            "Horizontal.TScale",
            background=UI_COLORS["panel"],
            troughcolor=UI_COLORS["entry_bg"],
        )

    def _build_layout(self) -> None:
        top_bar = ttk.Frame(self.root, padding=(18, 14, 18, 8), style="App.TFrame")
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_area = ttk.Frame(top_bar, style="App.TFrame")
        title_area.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(title_area, text="3D 3-DOF Robot Arm", style="Title.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(
            title_area,
            text="Forward kinematics, Cartesian inverse kinematics, and joint PID control",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        ttk.Button(
            top_bar,
            text="Open MuJoCo Viewer",
            style="Accent.TButton",
            command=self.open_mujoco_viewer,
        ).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(top_bar, text="Reset Robot", command=self.reset_simulation).pack(
            side=tk.RIGHT,
            padx=(8, 0),
        )
        ttk.Checkbutton(
            top_bar,
            text="Use PID Motion",
            variable=self.pid_motion_enabled_var,
            command=self._on_pid_motion_toggled,
        ).pack(side=tk.RIGHT, padx=(8, 0))

        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=18, pady=(0, 10))

        self.fk_scroll = ScrollableFrame(notebook)
        self.ik_scroll = ScrollableFrame(notebook)
        self.pid_scroll = ScrollableFrame(notebook)
        notebook.add(self.fk_scroll, text="Forward Kinematics")
        notebook.add(self.ik_scroll, text="Inverse Kinematics")
        notebook.add(self.pid_scroll, text="PID Control")

        forward_tab = build_forward_tab(
            self.fk_scroll.content,
            status_callback=self.status_var.set,
            callbacks_suspended=self._callbacks_suspended,
            on_slider_changed=self._on_fk_slider_changed,
            on_apply_target=self.apply_fk_target,
            equation_canvases=self.equation_canvases,
        )
        self.fk_q_controls = forward_tab.q_controls
        self.fk_output = forward_tab.output
        self.fk_apply_row = forward_tab.apply_row
        self.fk_apply_button = forward_tab.apply_button
        self._update_pid_mode_controls()

        inverse_tab = build_inverse_tab(
            self.ik_scroll.content,
            status_callback=self.status_var.set,
            callbacks_suspended=self._callbacks_suspended,
            on_target_changed=self._on_ik_target_slider_changed,
            on_solve=self._solve_ik,
            on_apply_solution=self._apply_ik_solution,
            equation_canvases=self.equation_canvases,
        )
        self.ik_x_control = inverse_tab.x_control
        self.ik_y_control = inverse_tab.y_control
        self.ik_z_control = inverse_tab.z_control
        self.ik_output = inverse_tab.output

        pid_tab = build_pid_tab(
            self.pid_scroll.content,
            status_callback=self.status_var.set,
            callbacks_suspended=self._callbacks_suspended,
            run_button_text=self.pid_run_button_text,
            status_var=self.pid_status_var,
            live_values_var=self.pid_live_values_var,
            on_target_changed=self._on_pid_target_changed,
            on_gain_changed=self._on_pid_gain_changed,
            on_toggle_pid=self.toggle_pid_control,
            on_hold_current=self.hold_current_position,
            on_reset=self.reset_simulation,
        )
        self.pid_q_controls = pid_tab.q_controls
        self.pid_kp_control = pid_tab.kp_control
        self.pid_ki_control = pid_tab.ki_control
        self.pid_kd_control = pid_tab.kd_control
        self.pid_axis = pid_tab.axis
        self.error_line = pid_tab.error_line
        self.torque_line = pid_tab.torque_line
        self.pid_canvas = pid_tab.canvas

        status_bar = ttk.Frame(self.root, padding=(18, 0, 18, 14), style="App.TFrame")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(
            status_bar,
            textvariable=self.status_var,
            style="Status.TLabel",
            relief=tk.FLAT,
            anchor=tk.W,
        ).pack(fill=tk.X)

    def _callbacks_suspended(self) -> bool:
        return self._updating_sliders

    def _update_pid_mode_controls(self) -> None:
        """Show FK Apply only when FK targets should go through PID."""

        if not hasattr(self, "fk_apply_row"):
            return

        if self.pid_motion_enabled_var.get():
            self.fk_apply_row.grid()
        else:
            self.fk_apply_row.grid_remove()

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
        kp, ki, kd = self._read_pid_gains()
        self.pid_controller.kp = kp
        self.pid_controller.ki = ki
        self.pid_controller.kd = kd

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

    def _record_pid_sample(self, sample: PidStepSample) -> None:
        self.pid_history.record(sample.elapsed_time, sample.error_norm, sample.torque_norm)
        self.pid_status_var.set(
            "PID running. "
            f"error norm = {sample.error_norm:.4f}, "
            f"torque norm = {sample.torque_norm:.4f}"
        )
        self._update_pid_plot()

    def _update_pid_plot(self) -> None:
        self.error_line.set_data(
            self.pid_history.time_history,
            self.pid_history.error_history,
        )
        self.torque_line.set_data(
            self.pid_history.time_history,
            self.pid_history.torque_history,
        )
        self.pid_axis.relim()
        self.pid_axis.autoscale_view()
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

        output = [
            f"State: {state}",
            f"Current q (deg): {self._format_vector(np.degrees(current_angles))}",
            f"Target q  (deg): {self._format_vector(np.degrees(target_angles))}",
            f"Error q   (deg): {self._format_vector(np.degrees(error_angles))}",
            f"Torque command:  {self._format_vector(torques)}",
            f"End effector m:  {self._format_vector(end_effector_position)}",
        ]
        self.pid_live_values_var.set("\n".join(output))

    def _set_joint_angles(self, joint_angles: np.ndarray) -> None:
        self.simulation.set_joint_angles(joint_angles)

    def _set_target_marker(self, target_position: np.ndarray) -> None:
        self.simulation.set_target_marker(target_position)

    def _current_joint_angles(self) -> np.ndarray:
        return self.simulation.current_joint_angles()

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

    def _read_pid_gains(self) -> tuple[float, float, float]:
        return (
            self.pid_kp_control.get(),
            self.pid_ki_control.get(),
            self.pid_kd_control.get(),
        )

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
