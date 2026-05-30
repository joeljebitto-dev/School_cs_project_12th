"""Tkinter user interface for the 3D kinematics and PID demo."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox, scrolledtext, ttk

from constants import (
    DEFAULT_PID_TARGET_DEGREES,
    DEFAULT_TARGET_POSITION,
    IK_TARGET_XY_LIMIT,
    IK_TARGET_Z_MAX,
    JOINT_LIMITS_DEGREES,
    LINK_LENGTHS,
    MIN_SAFE_Z,
    PID_PLOT_MAX_POINTS,
    UI_UPDATE_INTERVAL_MS,
)
from controller import PIDController
from robot_math import (
    forward_kinematics,
    inverse_kinematics,
    jacobian,
    joint_positions,
    safe_joint_message,
    wrap_angles,
)
from simulation import PidStepSample, RobotSimulation


EQUATION_BACKGROUND = "#fbfcfe"
EQUATION_FOREGROUND = "#111827"


def create_equation_figure(
    equations: list[str],
    height_inches: float,
    width_inches: float = 3.8,
) -> Figure:
    """Create a Matplotlib figure that renders mathtext equations."""

    figure = Figure(
        figsize=(width_inches, height_inches),
        dpi=100,
        facecolor=EQUATION_BACKGROUND,
    )
    axis = figure.add_subplot(111)
    axis.set_axis_off()
    axis.set_facecolor(EQUATION_BACKGROUND)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)

    if len(equations) == 1:
        y_positions = [0.5]
    else:
        y_positions = np.linspace(0.86, 0.14, len(equations))

    for equation, y_position in zip(equations, y_positions, strict=True):
        axis.text(
            0.02,
            float(y_position),
            equation,
            color=EQUATION_FOREGROUND,
            fontsize=13,
            ha="left",
            va="center",
            transform=axis.transAxes,
        )

    return figure


@dataclass
class SliderControl:
    """A slider plus a typed numeric input box."""

    label: str
    variable: tk.DoubleVar
    entry_variable: tk.StringVar
    display_format: str
    minimum: float
    maximum: float
    status_callback: Callable[[str], None] | None = None

    def get(self) -> float:
        """Return the current slider value."""

        return float(self.variable.get())

    def set(self, value: float) -> None:
        """Set the slider value, clamped to its configured range."""

        clamped_value = min(max(float(value), self.minimum), self.maximum)
        self.variable.set(clamped_value)

    def refresh_entry(self) -> None:
        """Update the typed input box from the slider value."""

        self.entry_variable.set(self.display_format.format(self.get()))

    def apply_entry_value(self) -> bool:
        """Apply the typed value to the slider.

        The value is accepted when the user presses Enter or leaves the box.
        Invalid text is restored to the last valid value.
        """

        typed_value = self.entry_variable.get().strip()
        try:
            value = float(typed_value)
        except ValueError:
            self.refresh_entry()
            if self.status_callback is not None:
                self.status_callback(f"Invalid number for {self.label}.")
            return False

        clamped_value = min(max(value, self.minimum), self.maximum)
        self.set(clamped_value)
        self.refresh_entry()
        if clamped_value != value and self.status_callback is not None:
            self.status_callback(
                f"{self.label} was clamped to the allowed range."
            )
        return True


class KinematicsPidApp:
    """Main Tkinter application.

    Tkinter collects inputs and shows the math. MuJoCo handles the physics and
    draws the robot in a separate viewer window.
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("3D 3-DOF MuJoCo Kinematics and PID Demo")
        self.root.geometry("1080x760")
        self.root.minsize(960, 680)

        self._configure_style()

        self.simulation = RobotSimulation()

        self.pid_controller = PIDController(kp=35.0, ki=0.0, kd=3.0)
        self.pid_running = False
        self.pid_target_angles = np.radians(DEFAULT_PID_TARGET_DEGREES)
        self.last_safe_fk_angles = np.zeros(3)
        self.last_safe_pid_target_angles = self.pid_target_angles.copy()

        self.last_ik_solution = None
        self.time_history: list[float] = []
        self.error_history: list[float] = []
        self.torque_history: list[float] = []
        self.equation_canvases: list[FigureCanvasTkAgg] = []
        self.plot_time = 0.0
        self._updating_sliders = False
        self.pid_motion_enabled_var = tk.BooleanVar(value=False)
        self.pid_run_button_text = tk.StringVar(value="Run PID")
        self.status_var = tk.StringVar(
            value="Ready. Open the MuJoCo viewer to see the 3D arm."
        )

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

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.root.configure(bg="#eef1f4")
        self.root.option_add("*Font", ("TkDefaultFont", 10))

        style.configure("TFrame", background="#eef1f4")
        style.configure("TLabel", background="#eef1f4", foreground="#1f2933")
        style.configure("Title.TLabel", font=("TkDefaultFont", 15, "bold"))
        style.configure("Subtitle.TLabel", foreground="#536273")
        style.configure("Status.TLabel", padding=(12, 6), foreground="#334155")
        style.configure(
            "EquationNote.TLabel",
            background="#eef1f4",
            foreground="#536273",
        )
        style.configure("TNotebook", background="#eef1f4", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8))
        style.configure("Panel.TLabelframe", background="#eef1f4", padding=10)
        style.configure(
            "Panel.TLabelframe.Label",
            background="#eef1f4",
            foreground="#1f2933",
            font=("TkDefaultFont", 10, "bold"),
        )
        style.configure("Accent.TButton", padding=(12, 6))

    def _build_layout(self) -> None:
        top_bar = ttk.Frame(self.root, padding=(16, 12, 16, 8))
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_area = ttk.Frame(top_bar)
        title_area.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            title_area,
            text="3D 3-DOF Robot Arm",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
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
        ttk.Button(
            top_bar,
            text="Reset Robot",
            command=self.reset_simulation,
        ).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Checkbutton(
            top_bar,
            text="Use PID Motion",
            variable=self.pid_motion_enabled_var,
            command=self._on_pid_motion_toggled,
        ).pack(side=tk.RIGHT, padx=(8, 0))

        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(0, 10))

        self.fk_tab = ttk.Frame(notebook, padding=14)
        self.ik_tab = ttk.Frame(notebook, padding=14)
        self.pid_tab = ttk.Frame(notebook, padding=14)

        notebook.add(self.fk_tab, text="Forward Kinematics")
        notebook.add(self.ik_tab, text="Inverse Kinematics")
        notebook.add(self.pid_tab, text="PID Control")

        self._build_fk_tab()
        self._build_ik_tab()
        self._build_pid_tab()

        status_bar = ttk.Frame(self.root, padding=(16, 0, 16, 12))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(
            status_bar,
            textvariable=self.status_var,
            style="Status.TLabel",
            relief=tk.GROOVE,
            anchor=tk.W,
        ).pack(fill=tk.X)

    def _build_fk_tab(self) -> None:
        self.fk_tab.columnconfigure(0, weight=0)
        self.fk_tab.columnconfigure(1, weight=1)
        self.fk_tab.rowconfigure(1, weight=1)

        input_panel = self._make_panel(self.fk_tab, "Joint Angles")
        input_panel.grid(row=0, column=0, sticky="new", padx=(0, 12), pady=(0, 12))
        input_panel.columnconfigure(0, weight=1)

        self.fk_q_controls = [
            self._add_slider(
                input_panel,
                "q1 yaw (deg)",
                -180,
                180,
                0,
                0,
                0,
                "{:.1f}",
                self._on_fk_slider_changed,
            ),
            self._add_slider(
                input_panel,
                "q2 shoulder (deg)",
                JOINT_LIMITS_DEGREES[1, 0],
                JOINT_LIMITS_DEGREES[1, 1],
                0,
                1,
                0,
                "{:.1f}",
                self._on_fk_slider_changed,
            ),
            self._add_slider(
                input_panel,
                "q3 elbow (deg)",
                JOINT_LIMITS_DEGREES[2, 0],
                JOINT_LIMITS_DEGREES[2, 1],
                0,
                2,
                0,
                "{:.1f}",
                self._on_fk_slider_changed,
            ),
        ]

        self.fk_apply_row = ttk.Frame(input_panel)
        self.fk_apply_row.grid(row=3, column=0, sticky=tk.EW, pady=(8, 0))
        self.fk_apply_button = ttk.Button(
            self.fk_apply_row,
            text="Apply FK Target",
            command=self.apply_fk_target,
        )
        self.fk_apply_button.pack(side=tk.LEFT)
        self._update_pid_mode_controls()

        math_panel = self._make_panel(self.fk_tab, "Core Math")
        math_panel.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 12))
        self._add_latex_equation_box(
            math_panel,
            equations=[
                r"$r = L_1\cos(q_2) + (L_2 + L_3)\cos(q_2 + q_3)$",
                r"$x = r\cos(q_1)$",
                r"$y = r\sin(q_1)$",
                r"$z = h + L_1\sin(q_2) + (L_2 + L_3)\sin(q_2 + q_3)$",
            ],
            note="q₁ turns the arm around the base. q₂ and q₃ set reach and height.",
            height_inches=1.45,
        )

        result_panel = self._make_panel(self.fk_tab, "Computed Position")
        result_panel.grid(row=0, column=1, rowspan=2, sticky=tk.NSEW)
        result_panel.rowconfigure(0, weight=1)
        result_panel.columnconfigure(0, weight=1)
        self.fk_output = self._make_output(result_panel, height=24)
        self.fk_output.grid(row=0, column=0, sticky=tk.NSEW)

    def _build_ik_tab(self) -> None:
        self.ik_tab.columnconfigure(0, weight=0)
        self.ik_tab.columnconfigure(1, weight=1)
        self.ik_tab.rowconfigure(1, weight=1)

        target_panel = self._make_panel(self.ik_tab, "Cartesian Target")
        target_panel.grid(row=0, column=0, sticky="new", padx=(0, 12), pady=(0, 12))
        target_panel.columnconfigure(0, weight=1)

        self.ik_x_control = self._add_slider(
            target_panel,
            "target x (m)",
            -IK_TARGET_XY_LIMIT,
            IK_TARGET_XY_LIMIT,
            0.65,
            0,
            0,
            "{:.3f}",
            self._on_ik_target_slider_changed,
        )
        self.ik_y_control = self._add_slider(
            target_panel,
            "target y (m)",
            -IK_TARGET_XY_LIMIT,
            IK_TARGET_XY_LIMIT,
            0.20,
            1,
            0,
            "{:.3f}",
            self._on_ik_target_slider_changed,
        )
        self.ik_z_control = self._add_slider(
            target_panel,
            "target z (m)",
            MIN_SAFE_Z,
            IK_TARGET_Z_MAX,
            0.35,
            2,
            0,
            "{:.3f}",
            self._on_ik_target_slider_changed,
        )

        button_row = ttk.Frame(target_panel)
        button_row.grid(row=3, column=0, sticky=tk.EW, pady=(8, 0))
        ttk.Button(button_row, text="Solve IK", command=self._solve_ik).pack(side=tk.LEFT)
        ttk.Button(
            button_row,
            text="Apply Solution",
            command=self._apply_ik_solution,
        ).pack(side=tk.LEFT, padx=(8, 0))

        math_panel = self._make_panel(self.ik_tab, "Solver Math")
        math_panel.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 12))
        self._add_latex_equation_box(
            math_panel,
            equations=[
                r"$\Delta p = p_{\mathrm{target}} - p_{\mathrm{current}}$",
                r"$\Delta p \approx J(q)\Delta q$",
                r"$\Delta q = \alpha J^T (J J^T + \lambda^2 I)^{-1}\Delta p$",
            ],
            note="Damping λ keeps the solver stable near difficult arm positions.",
            height_inches=1.25,
        )

        result_panel = self._make_panel(self.ik_tab, "IK Result")
        result_panel.grid(row=0, column=1, rowspan=2, sticky=tk.NSEW)
        result_panel.rowconfigure(0, weight=1)
        result_panel.columnconfigure(0, weight=1)
        self.ik_output = self._make_output(result_panel, height=24)
        self.ik_output.grid(row=0, column=0, sticky=tk.NSEW)

    def _build_pid_tab(self) -> None:
        self.pid_tab.columnconfigure(0, weight=0)
        self.pid_tab.columnconfigure(1, weight=1)
        self.pid_tab.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.pid_tab)
        controls.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 12))

        target_panel = self._make_panel(controls, "Target Angles")
        target_panel.pack(fill=tk.X, pady=(0, 12))
        target_panel.columnconfigure(0, weight=1)
        self.pid_q_controls = [
            self._add_slider(
                target_panel,
                "target yaw q1 (deg)",
                -180,
                180,
                DEFAULT_PID_TARGET_DEGREES[0],
                0,
                0,
                "{:.1f}",
                self._on_pid_target_changed,
            ),
            self._add_slider(
                target_panel,
                "target shoulder q2 (deg)",
                JOINT_LIMITS_DEGREES[1, 0],
                JOINT_LIMITS_DEGREES[1, 1],
                DEFAULT_PID_TARGET_DEGREES[1],
                1,
                0,
                "{:.1f}",
                self._on_pid_target_changed,
            ),
            self._add_slider(
                target_panel,
                "target elbow q3 (deg)",
                JOINT_LIMITS_DEGREES[2, 0],
                JOINT_LIMITS_DEGREES[2, 1],
                DEFAULT_PID_TARGET_DEGREES[2],
                2,
                0,
                "{:.1f}",
                self._on_pid_target_changed,
            ),
        ]

        gains_panel = self._make_panel(controls, "PID Gains")
        gains_panel.pack(fill=tk.X, pady=(0, 12))
        gains_panel.columnconfigure(0, weight=1)
        self.pid_kp_control = self._add_slider(
            gains_panel,
            "Kp",
            0,
            80,
            35,
            0,
            0,
            "{:.1f}",
            self._on_pid_gain_changed,
        )
        self.pid_ki_control = self._add_slider(
            gains_panel,
            "Ki",
            0,
            10,
            0,
            1,
            0,
            "{:.2f}",
            self._on_pid_gain_changed,
        )
        self.pid_kd_control = self._add_slider(
            gains_panel,
            "Kd",
            0,
            20,
            3,
            2,
            0,
            "{:.1f}",
            self._on_pid_gain_changed,
        )

        action_panel = self._make_panel(controls, "Controller")
        action_panel.pack(fill=tk.X)
        ttk.Button(
            action_panel,
            textvariable=self.pid_run_button_text,
            command=self.toggle_pid_control,
        ).pack(
            side=tk.LEFT,
        )
        ttk.Button(
            action_panel,
            text="Hold Current",
            command=self.hold_current_position,
        ).pack(
            side=tk.LEFT,
            padx=(8, 0),
        )
        ttk.Button(action_panel, text="Reset Robot", command=self.reset_simulation).pack(
            side=tk.LEFT,
            padx=(8, 0),
        )
        self.pid_status_var = tk.StringVar(value="PID is paused.")
        ttk.Label(
            action_panel,
            textvariable=self.pid_status_var,
            wraplength=320,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(12, 0))

        live_panel = self._make_panel(controls, "Live Values")
        live_panel.pack(fill=tk.X, pady=(12, 0))
        self.pid_live_values_var = tk.StringVar()
        ttk.Label(
            live_panel,
            textvariable=self.pid_live_values_var,
            font=("TkFixedFont", 10),
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(fill=tk.X, anchor=tk.W)

        plot_panel = self._make_panel(self.pid_tab, "Live Response")
        plot_panel.grid(row=0, column=1, sticky=tk.NSEW)
        plot_panel.rowconfigure(0, weight=1)
        plot_panel.columnconfigure(0, weight=1)

        figure = Figure(figsize=(7.2, 4.0), dpi=100)
        self.pid_axis = figure.add_subplot(111)
        self.pid_axis.set_title("PID error and torque over time")
        self.pid_axis.set_xlabel("time (s)")
        self.pid_axis.set_ylabel("norm")
        self.pid_axis.grid(True, alpha=0.3)
        (self.error_line,) = self.pid_axis.plot([], [], label="joint error norm")
        (self.torque_line,) = self.pid_axis.plot([], [], label="torque norm")
        self.pid_axis.legend(loc="upper right")

        self.pid_canvas = FigureCanvasTkAgg(figure, master=plot_panel)
        self.pid_canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)

    def _make_panel(self, parent: tk.Widget, title: str) -> ttk.LabelFrame:
        return ttk.LabelFrame(parent, text=title, style="Panel.TLabelframe", padding=12)

    def _make_output(self, parent: tk.Widget, height: int) -> scrolledtext.ScrolledText:
        output = scrolledtext.ScrolledText(
            parent,
            height=height,
            wrap=tk.NONE,
            font=("TkFixedFont", 10),
            bg="#fbfcfe",
            fg="#111827",
            insertbackground="#111827",
            relief=tk.FLAT,
            borderwidth=1,
        )
        output.configure(state=tk.DISABLED)
        return output

    def _add_latex_equation_box(
        self,
        parent: tk.Widget,
        equations: list[str],
        note: str,
        height_inches: float,
    ) -> None:
        """Show equations with Matplotlib mathtext inside Tkinter."""

        figure = create_equation_figure(equations, height_inches)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.X, anchor=tk.W)
        self.equation_canvases.append(canvas)

        ttk.Label(
            parent,
            text=note,
            style="EquationNote.TLabel",
            justify=tk.LEFT,
            wraplength=340,
        ).pack(fill=tk.X, anchor=tk.W, pady=(8, 0))

    def _add_slider(
        self,
        parent: tk.Widget,
        label: str,
        minimum: float,
        maximum: float,
        default: float,
        row: int,
        column: int,
        display_format: str,
        command: Callable[[], None] | None = None,
    ) -> SliderControl:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, sticky=tk.EW, pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky=tk.EW)
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=label).grid(row=0, column=0, sticky=tk.W)

        variable = tk.DoubleVar(value=default)
        entry_variable = tk.StringVar()

        control = SliderControl(
            label=label,
            variable=variable,
            entry_variable=entry_variable,
            display_format=display_format,
            minimum=minimum,
            maximum=maximum,
            status_callback=self.status_var.set,
        )

        def on_change(*_args: object) -> None:
            control.refresh_entry()
            if command is not None and not self._updating_sliders:
                command()

        variable.trace_add("write", on_change)
        control.refresh_entry()

        entry = ttk.Entry(
            header,
            textvariable=entry_variable,
            width=10,
            justify=tk.RIGHT,
        )
        entry.grid(row=0, column=1, sticky=tk.E)

        def apply_typed_value(_event: tk.Event | None = None) -> str:
            control.apply_entry_value()
            return "break"

        entry.bind("<Return>", apply_typed_value)
        entry.bind("<FocusOut>", apply_typed_value)

        ttk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            orient=tk.HORIZONTAL,
            variable=variable,
        ).grid(row=1, column=0, sticky=tk.EW, pady=(3, 0))

        range_label = ttk.Label(
            frame,
            text=f"{minimum:g} to {maximum:g}",
            style="Subtitle.TLabel",
        )
        range_label.grid(row=2, column=0, sticky=tk.W)
        return control

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

    def _read_current_robot_into_fk(self) -> None:
        self._set_fk_input_degrees(self._current_joint_angles())
        self._update_fk_output()
        self.status_var.set("Current robot angles copied into the FK inputs.")

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
        """Step the simulation once and schedule the next Tkinter update.

        Tkinter's ``after()`` keeps the simulation on the main event loop. The
        UI stays responsive because each update performs only a few MuJoCo
        steps, then returns control to Tkinter.
        """

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
        self.plot_time += sample.elapsed_time

        self.time_history.append(self.plot_time)
        self.error_history.append(sample.error_norm)
        self.torque_history.append(sample.torque_norm)

        self.time_history = self.time_history[-PID_PLOT_MAX_POINTS:]
        self.error_history = self.error_history[-PID_PLOT_MAX_POINTS:]
        self.torque_history = self.torque_history[-PID_PLOT_MAX_POINTS:]

        self.pid_status_var.set(
            "PID running. "
            f"error norm = {sample.error_norm:.4f}, "
            f"torque norm = {sample.torque_norm:.4f}"
        )
        self._update_pid_plot()

    def _update_pid_plot(self) -> None:
        self.error_line.set_data(self.time_history, self.error_history)
        self.torque_line.set_data(self.time_history, self.torque_history)
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
        self.time_history.clear()
        self.error_history.clear()
        self.torque_history.clear()
        self.plot_time = 0.0
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
        kp, ki, kd = self._read_pid_gains()
        self.pid_controller.kp = kp
        self.pid_controller.ki = ki
        self.pid_controller.kd = kd
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
