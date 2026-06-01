"""Window geometry and top-level layout assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

from config import (
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    WINDOW_HEIGHT_FRACTION,
    WINDOW_WIDTH_FRACTION,
)
from ui.forward_tab import build_forward_tab
from ui.inverse_tab import build_inverse_tab
from ui.pid_tab import build_pid_tab
from ui.widgets import ScrollableFrame, SliderControl


@dataclass
class AppLayout:
    """References to every widget the application needs after layout is built."""

    # Forward kinematics
    fk_q_controls: list[SliderControl]
    fk_output: tk.Text
    fk_apply_row: ttk.Frame

    # Inverse kinematics
    ik_x_control: SliderControl
    ik_y_control: SliderControl
    ik_z_control: SliderControl
    ik_output: tk.Text

    # PID control
    pid_q_controls: list[SliderControl]
    pid_joint_gains: list[Any]
    pid_axis: Any
    error_line: Any
    torque_line: Any
    pid_canvas: FigureCanvasTkAgg


def configure_window(root: tk.Tk) -> None:
    """Centre the window on screen at a sensible default size."""

    screen_width = max(root.winfo_screenwidth(), 800)
    screen_height = max(root.winfo_screenheight(), 600)
    max_width = max(640, screen_width - 40)
    max_height = max(520, screen_height - 80)
    width = min(max(int(screen_width * WINDOW_WIDTH_FRACTION), MIN_WINDOW_WIDTH), max_width)
    height = min(max(int(screen_height * WINDOW_HEIGHT_FRACTION), MIN_WINDOW_HEIGHT), max_height)
    x = max((screen_width - width) // 2, 0)
    y = max((screen_height - height) // 2, 0)

    root.geometry(f"{width}x{height}+{x}+{y}")
    root.minsize(min(MIN_WINDOW_WIDTH, width), min(MIN_WINDOW_HEIGHT, height))


def build_app_layout(
    root: tk.Tk,
    *,
    status_var: tk.StringVar,
    pid_motion_enabled_var: tk.BooleanVar,
    pid_run_button_text: tk.StringVar,
    pid_status_var: tk.StringVar,
    pid_live_values_var: tk.StringVar,
    equation_canvases: list[FigureCanvasTkAgg],
    on_open_viewer: Callable[[], None],
    on_reset: Callable[[], None],
    on_pid_motion_toggled: Callable[[], None],
    callbacks_suspended: Callable[[], bool],
    on_fk_slider_changed: Callable[[], None],
    on_apply_fk_target: Callable[[], None],
    on_ik_target_changed: Callable[[], None],
    on_solve_ik: Callable[[], None],
    on_apply_ik_solution: Callable[[], None],
    on_pid_target_changed: Callable[[], None],
    on_pid_gain_changed: Callable[[], None],
    on_toggle_pid: Callable[[], None],
    on_hold_current: Callable[[], None],
) -> AppLayout:
    """Build the top bar, tabbed notebook, and status bar.

    Returns an ``AppLayout`` with references to every widget the app needs.
    """

    # ---- Top bar ----
    top_bar = ttk.Frame(root, padding=(16, 10, 16, 6), style="App.TFrame")
    top_bar.pack(side=tk.TOP, fill=tk.X)

    title_area = ttk.Frame(top_bar, style="App.TFrame")
    title_area.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Label(title_area, text="3D 3-DOF Robot Arm", style="Title.TLabel").pack(
        anchor=tk.W
    )
    ttk.Label(
        title_area,
        text="Forward kinematics · Inverse kinematics · PID control",
        style="Subtitle.TLabel",
    ).pack(anchor=tk.W, pady=(1, 0))

    ttk.Button(
        top_bar,
        text="Open MuJoCo Viewer",
        style="Accent.TButton",
        command=on_open_viewer,
    ).pack(side=tk.RIGHT, padx=(6, 0))
    ttk.Button(top_bar, text="Reset Robot", command=on_reset).pack(
        side=tk.RIGHT,
        padx=(6, 0),
    )
    ttk.Checkbutton(
        top_bar,
        text="Use PID Motion",
        variable=pid_motion_enabled_var,
        command=on_pid_motion_toggled,
    ).pack(side=tk.RIGHT, padx=(6, 0))

    # ---- Notebook ----
    notebook = ttk.Notebook(root)
    notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(0, 6))

    fk_scroll = ScrollableFrame(notebook)
    ik_scroll = ScrollableFrame(notebook)
    pid_scroll = ScrollableFrame(notebook)
    notebook.add(fk_scroll, text="  Forward Kinematics  ")
    notebook.add(ik_scroll, text="  Inverse Kinematics  ")
    notebook.add(pid_scroll, text="  PID Control  ")

    # ---- Tab contents ----
    forward_tab = build_forward_tab(
        fk_scroll.content,
        status_callback=status_var.set,
        callbacks_suspended=callbacks_suspended,
        on_slider_changed=on_fk_slider_changed,
        on_apply_target=on_apply_fk_target,
        equation_canvases=equation_canvases,
    )

    inverse_tab = build_inverse_tab(
        ik_scroll.content,
        status_callback=status_var.set,
        callbacks_suspended=callbacks_suspended,
        on_target_changed=on_ik_target_changed,
        on_solve=on_solve_ik,
        on_apply_solution=on_apply_ik_solution,
        equation_canvases=equation_canvases,
    )

    pid_tab = build_pid_tab(
        pid_scroll.content,
        status_callback=status_var.set,
        callbacks_suspended=callbacks_suspended,
        run_button_text=pid_run_button_text,
        status_var=pid_status_var,
        live_values_var=pid_live_values_var,
        on_target_changed=on_pid_target_changed,
        on_gain_changed=on_pid_gain_changed,
        on_toggle_pid=on_toggle_pid,
        on_hold_current=on_hold_current,
        on_reset=on_reset,
    )

    # ---- Status bar ----
    status_bar = ttk.Frame(root, padding=(16, 0, 16, 8), style="App.TFrame")
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    ttk.Label(
        status_bar,
        textvariable=status_var,
        style="Status.TLabel",
        relief=tk.FLAT,
        anchor=tk.W,
    ).pack(fill=tk.X)

    return AppLayout(
        fk_q_controls=forward_tab.q_controls,
        fk_output=forward_tab.output,
        fk_apply_row=forward_tab.apply_row,
        ik_x_control=inverse_tab.x_control,
        ik_y_control=inverse_tab.y_control,
        ik_z_control=inverse_tab.z_control,
        ik_output=inverse_tab.output,
        pid_q_controls=pid_tab.q_controls,
        pid_joint_gains=pid_tab.joint_gains,
        pid_axis=pid_tab.axis,
        error_line=pid_tab.error_line,
        torque_line=pid_tab.torque_line,
        pid_canvas=pid_tab.canvas,
    )
