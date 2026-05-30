"""Reusable Tkinter widgets for the robot demo UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import scrolledtext, ttk

from config import UI_COLORS


EQUATION_BACKGROUND = UI_COLORS["equation_bg"]
EQUATION_FOREGROUND = UI_COLORS["text"]


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

    y_positions = [0.5] if len(equations) == 1 else np.linspace(0.86, 0.14, len(equations))
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
        """Apply typed input on Enter or focus loss."""

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
            self.status_callback(f"{self.label} was clamped to the allowed range.")
        return True


class ScrollableFrame(ttk.Frame):
    """A dark scrollable frame used inside each notebook tab."""

    def __init__(self, parent: tk.Widget, *, padding: int = 14) -> None:
        super().__init__(parent, style="App.TFrame")
        self.canvas = tk.Canvas(
            self,
            borderwidth=0,
            highlightthickness=0,
            background=UI_COLORS["bg"],
        )
        self.content = ttk.Frame(self.canvas, padding=padding, style="App.TFrame")
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor=tk.NW)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_content)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        self.content.bind("<Enter>", self._bind_mousewheel)
        self.content.bind("<Leave>", self._unbind_mousewheel)

    def _update_scroll_region(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_content(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(delta, "units")


def make_card(parent: tk.Widget, title: str) -> ttk.LabelFrame:
    """Create a styled card-like section."""

    return ttk.LabelFrame(parent, text=title, style="Card.TLabelframe", padding=14)


def make_output(parent: tk.Widget, height: int) -> scrolledtext.ScrolledText:
    """Create a dark read-only text panel."""

    output = scrolledtext.ScrolledText(
        parent,
        height=height,
        wrap=tk.NONE,
        font=("TkFixedFont", 10),
        bg=UI_COLORS["result_bg"],
        fg=UI_COLORS["text"],
        insertbackground=UI_COLORS["text"],
        selectbackground=UI_COLORS["accent_dark"],
        selectforeground=UI_COLORS["text"],
        relief=tk.FLAT,
        borderwidth=0,
    )
    output.configure(state=tk.DISABLED)
    return output


def add_latex_equation_box(
    parent: tk.Widget,
    equations: list[str],
    note: str,
    equation_canvases: list[FigureCanvasTkAgg],
    height_inches: float,
) -> None:
    """Show equations with Matplotlib mathtext inside Tkinter."""

    figure = create_equation_figure(equations, height_inches)
    canvas = FigureCanvasTkAgg(figure, master=parent)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.configure(background=EQUATION_BACKGROUND, highlightthickness=0)
    widget.pack(fill=tk.X, anchor=tk.W)
    equation_canvases.append(canvas)

    ttk.Label(
        parent,
        text=note,
        style="Muted.TLabel",
        justify=tk.LEFT,
        wraplength=360,
    ).pack(fill=tk.X, anchor=tk.W, pady=(8, 0))


def add_slider(
    parent: tk.Widget,
    label: str,
    minimum: float,
    maximum: float,
    default: float,
    row: int,
    column: int,
    display_format: str,
    *,
    status_callback: Callable[[str], None] | None,
    command: Callable[[], None] | None = None,
    callbacks_suspended: Callable[[], bool] | None = None,
) -> SliderControl:
    """Create a labeled slider with a synchronized numeric entry box."""

    frame = ttk.Frame(parent, style="Card.TFrame")
    frame.grid(row=row, column=column, sticky=tk.EW, pady=(0, 12))
    frame.columnconfigure(0, weight=1)

    header = ttk.Frame(frame, style="Card.TFrame")
    header.grid(row=0, column=0, sticky=tk.EW)
    header.columnconfigure(0, weight=1)
    ttk.Label(header, text=label, style="Card.TLabel").grid(row=0, column=0, sticky=tk.W)

    variable = tk.DoubleVar(value=default)
    entry_variable = tk.StringVar()
    control = SliderControl(
        label=label,
        variable=variable,
        entry_variable=entry_variable,
        display_format=display_format,
        minimum=minimum,
        maximum=maximum,
        status_callback=status_callback,
    )

    def on_change(*_args: object) -> None:
        control.refresh_entry()
        if command is not None and not (callbacks_suspended and callbacks_suspended()):
            command()

    variable.trace_add("write", on_change)
    control.refresh_entry()

    entry = ttk.Entry(
        header,
        textvariable=entry_variable,
        width=10,
        justify=tk.RIGHT,
        style="Dark.TEntry",
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
    ).grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))

    ttk.Label(
        frame,
        text=f"{minimum:g} to {maximum:g}",
        style="Muted.TLabel",
    ).grid(row=2, column=0, sticky=tk.W, pady=(2, 0))
    return control
