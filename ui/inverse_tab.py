"""Inverse Kinematics tab construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

from config import DEFAULT_TARGET_POSITION, IK_TARGET_XY_LIMIT, IK_TARGET_Z_MAX, MIN_SAFE_Z
from ui.widgets import SliderControl, add_latex_equation_box, add_slider, make_card, make_output


@dataclass
class InverseTab:
    """Widgets owned by the Inverse Kinematics tab."""

    x_control: SliderControl
    y_control: SliderControl
    z_control: SliderControl
    output: tk.Text


def build_inverse_tab(
    parent: tk.Widget,
    *,
    status_callback: Callable[[str], None],
    callbacks_suspended: Callable[[], bool],
    on_target_changed: Callable[[], None],
    on_solve: Callable[[], None],
    on_apply_solution: Callable[[], None],
    equation_canvases: list[FigureCanvasTkAgg],
) -> InverseTab:
    """Build the Inverse Kinematics tab UI."""

    parent.columnconfigure(0, weight=0)
    parent.columnconfigure(1, weight=1)
    parent.rowconfigure(1, weight=1)

    target_panel = make_card(parent, "Cartesian Target")
    target_panel.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 10), pady=(0, 8))
    target_panel.columnconfigure(0, weight=1)

    x_control = add_slider(
        target_panel,
        "target x (m)",
        -IK_TARGET_XY_LIMIT,
        IK_TARGET_XY_LIMIT,
        DEFAULT_TARGET_POSITION[0],
        0,
        0,
        "{:.3f}",
        status_callback=status_callback,
        command=on_target_changed,
        callbacks_suspended=callbacks_suspended,
    )
    y_control = add_slider(
        target_panel,
        "target y (m)",
        -IK_TARGET_XY_LIMIT,
        IK_TARGET_XY_LIMIT,
        DEFAULT_TARGET_POSITION[1],
        1,
        0,
        "{:.3f}",
        status_callback=status_callback,
        command=on_target_changed,
        callbacks_suspended=callbacks_suspended,
    )
    z_control = add_slider(
        target_panel,
        "target z (m)",
        MIN_SAFE_Z,
        IK_TARGET_Z_MAX,
        DEFAULT_TARGET_POSITION[2],
        2,
        0,
        "{:.3f}",
        status_callback=status_callback,
        command=on_target_changed,
        callbacks_suspended=callbacks_suspended,
    )

    button_row = ttk.Frame(target_panel, style="Card.TFrame")
    button_row.grid(row=3, column=0, sticky=tk.EW, pady=(6, 0))
    ttk.Button(button_row, text="Solve IK", command=on_solve, style="Accent.TButton").pack(
        side=tk.LEFT
    )
    ttk.Button(
        button_row,
        text="Apply Solution",
        command=on_apply_solution,
    ).pack(side=tk.LEFT, padx=(8, 0))

    math_panel = make_card(parent, "Solver Math")
    math_panel.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 10))
    add_latex_equation_box(
        math_panel,
        equations=[
            r"$\Delta p = p_{\mathrm{target}} - p_{\mathrm{current}}$",
            r"$\Delta p \approx J(q)\Delta q$",
            r"$\Delta q = \alpha J^T (J J^T + \lambda^2 I)^{-1}\Delta p$",
        ],
        note="Damping lambda keeps the solver stable near difficult arm positions.",
        equation_canvases=equation_canvases,
        height_inches=1.25,
    )

    result_panel = make_card(parent, "IK Result")
    result_panel.grid(row=0, column=1, rowspan=2, sticky=tk.NSEW)
    result_panel.rowconfigure(0, weight=1)
    result_panel.columnconfigure(0, weight=1)
    output = make_output(result_panel, height=24)
    output.grid(row=0, column=0, sticky=tk.NSEW)

    return InverseTab(
        x_control=x_control,
        y_control=y_control,
        z_control=z_control,
        output=output,
    )
