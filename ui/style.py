"""VS Code Dark+ inspired theme configuration for the application."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from config import UI_COLORS


def configure_app_style(root: tk.Tk) -> None:
    """Apply the VS Code Dark+ inspired theme to all ttk widgets."""

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    # Font stack: prefer Segoe UI (VS Code default), fallback to system sans
    font_family = "Segoe UI"
    base_font = (font_family, 10)
    bold_font = (font_family, 10, "bold")
    small_font = (font_family, 9)

    root.configure(bg=UI_COLORS["bg"])
    root.option_add("*Font", base_font)

    # --- Frames ---
    style.configure("TFrame", background=UI_COLORS["bg"])
    style.configure("App.TFrame", background=UI_COLORS["bg"])
    style.configure("Card.TFrame", background=UI_COLORS["panel"])

    # --- Labels ---
    style.configure(
        "TLabel",
        background=UI_COLORS["bg"],
        foreground=UI_COLORS["text"],
        font=base_font,
    )
    style.configure(
        "Title.TLabel",
        background=UI_COLORS["bg"],
        foreground="#ffffff",
        font=(font_family, 18, "bold"),
    )
    style.configure(
        "Subtitle.TLabel",
        background=UI_COLORS["bg"],
        foreground=UI_COLORS["muted"],
        font=small_font,
    )
    style.configure(
        "Muted.TLabel",
        background=UI_COLORS["panel"],
        foreground=UI_COLORS["muted"],
        font=small_font,
    )
    style.configure(
        "Card.TLabel",
        background=UI_COLORS["panel"],
        foreground=UI_COLORS["text"],
        font=base_font,
    )
    style.configure(
        "Status.TLabel",
        padding=(12, 6),
        background=UI_COLORS["panel_alt"],
        foreground=UI_COLORS["muted"],
        font=small_font,
    )

    # --- Notebook (tabs) ---
    style.configure(
        "TNotebook",
        background=UI_COLORS["bg"],
        borderwidth=0,
        tabmargins=(0, 0, 0, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=UI_COLORS["panel_alt"],
        foreground=UI_COLORS["muted"],
        padding=(16, 8),
        borderwidth=0,
        font=bold_font,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", UI_COLORS["panel"])],
        foreground=[("selected", "#ffffff")],
    )

    # --- Cards (LabelFrames) ---
    style.configure(
        "Card.TLabelframe",
        background=UI_COLORS["panel"],
        foreground=UI_COLORS["text"],
        bordercolor=UI_COLORS["border"],
        relief=tk.FLAT,
        borderwidth=1,
        padding=14,
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=UI_COLORS["panel"],
        foreground=UI_COLORS["accent"],
        font=bold_font,
    )

    # --- Buttons (Tailwind-style: flat, padded, bold) ---
    style.configure(
        "TButton",
        background=UI_COLORS["panel_alt"],
        foreground=UI_COLORS["text"],
        bordercolor=UI_COLORS["border"],
        borderwidth=1,
        focusthickness=0,
        focuscolor=UI_COLORS["accent"],
        padding=(14, 6),
        font=bold_font,
        relief=tk.FLAT,
    )
    style.map(
        "TButton",
        background=[("active", UI_COLORS["hover"]), ("pressed", UI_COLORS["border"])],
        foreground=[("disabled", UI_COLORS["muted"])],
    )
    style.configure(
        "Accent.TButton",
        background=UI_COLORS["accent"],
        foreground="#ffffff",
        bordercolor=UI_COLORS["accent_dark"],
        borderwidth=0,
        padding=(14, 6),
        font=bold_font,
    )
    style.map(
        "Accent.TButton",
        background=[("active", UI_COLORS["accent_dark"]), ("pressed", "#1d4ed8")],
    )

    # --- Checkbutton ---
    style.configure(
        "TCheckbutton",
        background=UI_COLORS["bg"],
        foreground=UI_COLORS["text"],
        font=base_font,
        indicatorsize=12,
    )
    style.map(
        "TCheckbutton",
        background=[("active", UI_COLORS["bg"])],
        foreground=[("active", UI_COLORS["text"])],
        indicatorcolor=[("selected", UI_COLORS["accent"])],
    )

    # --- Entry ---
    style.configure(
        "Dark.TEntry",
        fieldbackground=UI_COLORS["entry_bg"],
        foreground=UI_COLORS["text"],
        insertcolor=UI_COLORS["text"],
        bordercolor=UI_COLORS["border"],
        padding=4,
    )

    # --- Scale (slider) ---
    style.configure(
        "Horizontal.TScale",
        background=UI_COLORS["panel"],
        troughcolor=UI_COLORS["entry_bg"],
        troughrelief=tk.FLAT,
        sliderthickness=14,
    )

    # --- Scrollbar ---
    style.configure(
        "Vertical.TScrollbar",
        background=UI_COLORS["panel_alt"],
        troughcolor=UI_COLORS["bg"],
        bordercolor=UI_COLORS["bg"],
        arrowcolor=UI_COLORS["muted"],
        relief=tk.FLAT,
    )
    style.map(
        "Vertical.TScrollbar",
        background=[("active", UI_COLORS["border"])],
    )
