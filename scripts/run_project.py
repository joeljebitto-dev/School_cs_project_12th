"""Run the robot demo using the project virtual environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_DIR / ".venv"
MAIN_FILE = PROJECT_DIR / "main.py"


def find_venv_python() -> Path | None:
    """Return the virtual environment Python path, if it exists."""

    candidates = [
        VENV_DIR / "bin" / "python",
        VENV_DIR / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    """Start the Tkinter application from the virtual environment."""

    python_path = find_venv_python()
    if python_path is None:
        print("Could not find a virtual environment Python in .venv.")
        print("Run ./scripts/setup.sh first.")
        return 1

    if not MAIN_FILE.exists():
        print("Could not find main.py.")
        return 1

    return subprocess.call([str(python_path), str(MAIN_FILE)], cwd=PROJECT_DIR)


if __name__ == "__main__":
    sys.exit(main())
