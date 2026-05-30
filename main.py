"""Entry point for the MuJoCo kinematics and PID demo."""

from __future__ import annotations

import sys


def main() -> int:
    """Start the Tkinter application."""

    try:
        import tkinter as tk
        from ui.app import KinematicsPidApp
    except ImportError as error:
        print("Could not start the app because a dependency is missing.")
        print(f"Missing detail: {error}")
        print()
        print("Install dependencies with:")
        print("  sudo apt install python3-tk")
        print("  python -m pip install -r requirements.txt")
        return 1

    try:
        root = tk.Tk()
        app = KinematicsPidApp(root)
    except Exception as error:
        print("Could not create the app window.")
        print(f"Startup detail: {error}")
        print()
        print("If this mentions Tkinter, install it with:")
        print("  sudo apt install python3-tk")
        return 1

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
