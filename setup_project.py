"""Create the local virtual environment and install project dependencies."""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"


def venv_python_path() -> Path:
    """Return the Python executable path inside the project virtual environment."""

    linux_python = VENV_DIR / "bin" / "python"
    windows_python = VENV_DIR / "Scripts" / "python.exe"
    if linux_python.exists():
        return linux_python
    return windows_python


def run_command(command: list[str]) -> None:
    """Run a setup command and stop if it fails."""

    print("+ " + " ".join(command))
    subprocess.check_call(command, cwd=PROJECT_DIR)


def tkinter_is_available(python_path: Path) -> bool:
    """Return True when Tkinter can be imported from the virtual environment."""

    result = subprocess.run(
        [str(python_path), "-c", "import tkinter"],
        cwd=PROJECT_DIR,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main() -> int:
    """Set up the project for running the demo."""

    if not REQUIREMENTS_FILE.exists():
        print("Could not find requirements.txt.")
        return 1

    if not VENV_DIR.exists():
        print("Creating virtual environment in .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print("Using existing virtual environment in .venv.")

    python_path = venv_python_path()
    if not python_path.exists():
        print("Could not find the Python executable inside .venv.")
        print("Delete .venv and run ./setup.sh again.")
        return 1

    run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    run_command([str(python_path), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])

    if tkinter_is_available(python_path):
        print("Tkinter check passed.")
    else:
        print()
        print("Tkinter could not be imported.")
        print("On Ubuntu, install it with:")
        print("  sudo apt install python3-tk")
        print("Then run ./setup.sh again if needed.")

    print()
    print("Setup complete. Start the app with:")
    print("  ./run.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
