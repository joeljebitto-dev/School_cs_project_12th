"""Forward Kinematics tab construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

from config import JOINT_LIMITS_DEGREES
from ui.widgets import SliderControl, add_latex_equation_box, add_slider, make_card, make_output


@dataclass
class ForwardTab:
    """Widgets owned by the Forward Kinematics tab."""

    q_controls: list[SliderControl]
    output: tk.Text
    apply_row: ttk.Frame
    apply_button: ttk.Button


def build_forward_tab(
    parent: tk.Widget,
    *,
    status_callback: Callable[[str], None],
    callbacks_suspended: Callable[[], bool],
    on_slider_changed: Callable[[], None],
    on_apply_target: Callable[[], None],
    equation_canvases: list[FigureCanvasTkAgg],
) -> ForwardTab:
    """Build the Forward Kinematics tab UI."""

    parent.columnconfigure(0, weight=0)
    parent.columnconfigure(1, weight=1)
    parent.rowconfigure(1, weight=1)

    input_panel = make_card(parent, "Joint Angles")
    input_panel.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 12), pady=(0, 12))
    input_panel.columnconfigure(0, weight=1)

    q_controls = [
        add_slider(
            input_panel,
            "q1 yaw (deg)",
            -180,
            180,
            0,
            0,
            0,
            "{:.1f}",
            status_callback=status_callback,
            command=on_slider_changed,
            callbacks_suspended=callbacks_suspended,
        ),
        add_slider(
            input_panel,
            "q2 shoulder (deg)",
            JOINT_LIMITS_DEGREES[1, 0],
            JOINT_LIMITS_DEGREES[1, 1],
            0,
            1,
            0,
            "{:.1f}",
            status_callback=status_callback,
            command=on_slider_changed,
            callbacks_suspended=callbacks_suspended,
        ),
        add_slider(
            input_panel,
            "q3 elbow (deg)",
            JOINT_LIMITS_DEGREES[2, 0],
            JOINT_LIMITS_DEGREES[2, 1],
            0,
            2,
            0,
            "{:.1f}",
            status_callback=status_callback,
            command=on_slider_changed,
            callbacks_suspended=callbacks_suspended,
        ),
    ]

    apply_row = ttk.Frame(input_panel, style="Card.TFrame")
    apply_row.grid(row=3, column=0, sticky=tk.EW, pady=(6, 0))
    apply_button = ttk.Button(
        apply_row,
        text="Apply FK Target",
        command=on_apply_target,
        style="Accent.TButton",
    )
    apply_button.pack(side=tk.LEFT)

    math_panel = make_card(parent, "Core Math")
    math_panel.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 12))
    add_latex_equation_box(
        math_panel,
        equations=[
            r"$r = L_1\cos(q_2) + (L_2 + L_3)\cos(q_2 + q_3)$",
            r"$x = r\cos(q_1)$",
            r"$y = r\sin(q_1)$",
            r"$z = h + L_1\sin(q_2) + (L_2 + L_3)\sin(q_2 + q_3)$",
        ],
        note="q1 turns the arm around the base. q2 and q3 set reach and height.",
        equation_canvases=equation_canvases,
        height_inches=1.45,
    )

    result_panel = make_card(parent, "Computed Position")
    result_panel.grid(row=0, column=1, rowspan=2, sticky=tk.NSEW)
    result_panel.rowconfigure(0, weight=1)
    result_panel.columnconfigure(0, weight=1)
    output = make_output(result_panel, height=24)
    output.grid(row=0, column=0, sticky=tk.NSEW)

    return ForwardTab(
        q_controls=q_controls,
        output=output,
        apply_row=apply_row,
        apply_button=apply_button,
    )
