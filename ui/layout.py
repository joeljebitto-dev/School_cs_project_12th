"""Window geometry and top-level layout assembly."""

import tkinter as tk
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
from ui.widgets import ScrollableFrame


def configure_window(root):
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
    root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

def build_app_layout(app):
    """Build the top bar, tabbed notebook, and status bar.

    Reads callbacks from the app object and stores created widgets
    back on it so the rest of the app can use them.
    """

    root = app.root

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
        command=app.open_mujoco_viewer,
    ).pack(side=tk.RIGHT, padx=(6, 0))
    ttk.Button(top_bar, text="Reset Robot", command=app.reset_simulation).pack(
        side=tk.RIGHT,
        padx=(6, 0),
    )
    ttk.Checkbutton(
        top_bar,
        text="Use PID Motion",
        variable=app.pid_motion_enabled_var,
        command=app._on_pid_motion_toggled,
    ).pack(side=tk.RIGHT, padx=(6, 0))

    # ---- Notebook with three tabs ----
    notebook = ttk.Notebook(root)
    notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(0, 6))

    fk_scroll = ScrollableFrame(notebook)
    ik_scroll = ScrollableFrame(notebook)
    pid_scroll = ScrollableFrame(notebook)
    notebook.add(fk_scroll, text="  Forward Kinematics  ")
    notebook.add(ik_scroll, text="  Inverse Kinematics  ")
    notebook.add(pid_scroll, text="  PID Control  ")

    # ---- Forward Kinematics tab ----
    forward_tab = build_forward_tab(
        fk_scroll.content,
        status_callback=app.status_var.set,
        callbacks_suspended=app._callbacks_suspended,
        on_slider_changed=app._on_fk_slider_changed,
        on_apply_target=app.apply_fk_target,
        equation_canvases=app.equation_canvases,
    )
    app.fk_q_controls = forward_tab.q_controls
    app.fk_output = forward_tab.output
    app.fk_apply_row = forward_tab.apply_row

    # ---- Inverse Kinematics tab ----
    inverse_tab = build_inverse_tab(
        ik_scroll.content,
        status_callback=app.status_var.set,
        callbacks_suspended=app._callbacks_suspended,
        on_target_changed=app._on_ik_target_slider_changed,
        on_solve=app._solve_ik,
        on_apply_solution=app._apply_ik_solution,
        equation_canvases=app.equation_canvases,
    )
    app.ik_x_control = inverse_tab.x_control
    app.ik_y_control = inverse_tab.y_control
    app.ik_z_control = inverse_tab.z_control
    app.ik_output = inverse_tab.output

    # ---- PID Control tab ----
    pid_tab = build_pid_tab(
        pid_scroll.content,
        status_callback=app.status_var.set,
        callbacks_suspended=app._callbacks_suspended,
        run_button_text=app.pid_run_button_text,
        status_var=app.pid_status_var,
        live_values_var=app.pid_live_values_var,
        on_target_changed=app._on_pid_target_changed,
        on_gain_changed=app._on_pid_gain_changed,
        on_toggle_pid=app.toggle_pid_control,
        on_hold_current=app.hold_current_position,
        on_reset=app.reset_simulation,
    )
    app.pid_q_controls = pid_tab.q_controls
    app.pid_joint_gains = pid_tab.joint_gains
    app.pid_axis = pid_tab.axis
    app.error_line = pid_tab.error_line
    app.torque_line = pid_tab.torque_line
    app.pid_canvas = pid_tab.canvas

    # ---- Status bar ----
    status_bar = ttk.Frame(root, padding=(16, 0, 16, 8), style="App.TFrame")
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    ttk.Label(
        status_bar,
        textvariable=app.status_var,
        style="Status.TLabel",
        relief=tk.FLAT,
        anchor=tk.W,
    ).pack(fill=tk.X)
