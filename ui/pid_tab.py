"""PID Control tab construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk

from config import DEFAULT_PID_GAINS, DEFAULT_PID_TARGET_DEGREES, JOINT_LIMITS_DEGREES, UI_COLORS
from ui.widgets import SliderControl, add_slider, make_card


@dataclass
class JointGainControls:
    """PID gain sliders for a single joint."""

    kp: SliderControl
    ki: SliderControl
    kd: SliderControl


@dataclass
class PidTab:
    """Widgets owned by the PID Control tab."""

    q_controls: list[SliderControl]
    joint_gains: list[JointGainControls]  # one per joint
    axis: object
    error_line: object
    torque_line: object
    canvas: FigureCanvasTkAgg


_JOINT_LABELS = [
    ("Joint 1 — Yaw", "joint1"),
    ("Joint 2 — Shoulder", "joint2"),
    ("Joint 3 — Elbow", "joint3"),
]


def build_pid_tab(
    parent: tk.Widget,
    *,
    status_callback: Callable[[str], None],
    callbacks_suspended: Callable[[], bool],
    run_button_text: tk.StringVar,
    status_var: tk.StringVar,
    live_values_var: tk.StringVar,
    on_target_changed: Callable[[], None],
    on_gain_changed: Callable[[], None],
    on_toggle_pid: Callable[[], None],
    on_hold_current: Callable[[], None],
    on_reset: Callable[[], None],
) -> PidTab:
    """Build the PID Control tab UI."""

    parent.columnconfigure(0, weight=0)
    parent.columnconfigure(1, weight=1)
    parent.rowconfigure(0, weight=1)

    controls = ttk.Frame(parent, style="App.TFrame")
    controls.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 12))

    target_panel = make_card(controls, "Target Angles")
    target_panel.pack(fill=tk.X, pady=(0, 8))
    target_panel.columnconfigure(0, weight=1)
    q_controls = [
        add_slider(
            target_panel,
            "target yaw q1 (deg)",
            -180,
            180,
            DEFAULT_PID_TARGET_DEGREES[0],
            0,
            0,
            "{:.1f}",
            status_callback=status_callback,
            command=on_target_changed,
            callbacks_suspended=callbacks_suspended,
        ),
        add_slider(
            target_panel,
            "target shoulder q2 (deg)",
            JOINT_LIMITS_DEGREES[1, 0],
            JOINT_LIMITS_DEGREES[1, 1],
            DEFAULT_PID_TARGET_DEGREES[1],
            1,
            0,
            "{:.1f}",
            status_callback=status_callback,
            command=on_target_changed,
            callbacks_suspended=callbacks_suspended,
        ),
        add_slider(
            target_panel,
            "target elbow q3 (deg)",
            JOINT_LIMITS_DEGREES[2, 0],
            JOINT_LIMITS_DEGREES[2, 1],
            DEFAULT_PID_TARGET_DEGREES[2],
            2,
            0,
            "{:.1f}",
            status_callback=status_callback,
            command=on_target_changed,
            callbacks_suspended=callbacks_suspended,
        ),
    ]

    # --- Per-joint PID gain sections ---
    joint_gains: list[JointGainControls] = []
    for idx, (label, key) in enumerate(_JOINT_LABELS):
        default_kp, default_ki, default_kd = DEFAULT_PID_GAINS[key]
        gains_panel = make_card(controls, f"PID Gains — {label}")
        gains_panel.pack(fill=tk.X, pady=(0, 8))
        gains_panel.columnconfigure(0, weight=1)

        kp_ctrl = add_slider(
            gains_panel,
            "Kp",
            0,
            400,
            default_kp,
            0,
            0,
            "{:.1f}",
            status_callback=status_callback,
            command=on_gain_changed,
            callbacks_suspended=callbacks_suspended,
        )
        ki_ctrl = add_slider(
            gains_panel,
            "Ki",
            0,
            15,
            default_ki,
            1,
            0,
            "{:.2f}",
            status_callback=status_callback,
            command=on_gain_changed,
            callbacks_suspended=callbacks_suspended,
        )
        kd_ctrl = add_slider(
            gains_panel,
            "Kd",
            0,
            20,
            default_kd,
            2,
            0,
            "{:.1f}",
            status_callback=status_callback,
            command=on_gain_changed,
            callbacks_suspended=callbacks_suspended,
        )
        joint_gains.append(JointGainControls(kp=kp_ctrl, ki=ki_ctrl, kd=kd_ctrl))

    action_panel = make_card(controls, "Controller")
    action_panel.pack(fill=tk.X)
    ttk.Button(
        action_panel,
        textvariable=run_button_text,
        command=on_toggle_pid,
        style="Accent.TButton",
    ).pack(side=tk.LEFT)
    ttk.Button(
        action_panel,
        text="Hold Current",
        command=on_hold_current,
    ).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(action_panel, text="Reset Robot", command=on_reset).pack(
        side=tk.LEFT,
        padx=(8, 0),
    )
    ttk.Label(
        action_panel,
        textvariable=status_var,
        style="Card.TLabel",
        wraplength=340,
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(12, 0))

    live_panel = make_card(controls, "Live Values")
    live_panel.pack(fill=tk.X, pady=(12, 0))
    ttk.Label(
        live_panel,
        textvariable=live_values_var,
        font=("Consolas", 10),
        style="Card.TLabel",
        justify=tk.LEFT,
        anchor=tk.W,
    ).pack(fill=tk.X, anchor=tk.W)

    plot_panel = make_card(parent, "Live Response")
    plot_panel.grid(row=0, column=1, sticky=tk.NSEW)
    plot_panel.rowconfigure(0, weight=1)
    plot_panel.columnconfigure(0, weight=1)

    figure = Figure(figsize=(7.2, 4.0), dpi=100, facecolor=UI_COLORS["panel"])
    axis = figure.add_subplot(111)
    axis.set_facecolor(UI_COLORS["result_bg"])
    axis.set_title("PID error and torque over time", color=UI_COLORS["text"])
    axis.set_xlabel("time (s)", color=UI_COLORS["muted"])
    axis.set_ylabel("norm", color=UI_COLORS["muted"])
    axis.grid(True, alpha=0.28, color=UI_COLORS["border"])
    axis.tick_params(colors=UI_COLORS["muted"])
    for spine in axis.spines.values():
        spine.set_color(UI_COLORS["border"])
    (error_line,) = axis.plot([], [], label="joint error norm", color=UI_COLORS["accent"])
    (torque_line,) = axis.plot([], [], label="torque norm", color=UI_COLORS["accent_secondary"])
    legend = axis.legend(loc="upper right")
    legend.get_frame().set_facecolor(UI_COLORS["panel"])
    legend.get_frame().set_edgecolor(UI_COLORS["border"])
    for text in legend.get_texts():
        text.set_color(UI_COLORS["text"])

    canvas = FigureCanvasTkAgg(figure, master=plot_panel)
    canvas.get_tk_widget().configure(background=UI_COLORS["panel"], highlightthickness=0)
    canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)

    return PidTab(
        q_controls=q_controls,
        joint_gains=joint_gains,
        axis=axis,
        error_line=error_line,
        torque_line=torque_line,
        canvas=canvas,
    )
