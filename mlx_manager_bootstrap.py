#!/usr/bin/env python3
"""
Bootstrap helper for mlx_manager.py.

Goals:
- create ./mlx-env if missing
- install/repair the minimum runtime packages
- launch mlx_manager.py in that runtime

This script is stdlib-only so it can run before PyQt6 or other dependencies exist.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MANAGER = REPO_ROOT / "mlx_manager.py"
VENV_DIR = REPO_ROOT / "mlx-env"
VENV_PYTHON = VENV_DIR / "bin" / "python"
REQUIRED_PACKAGES = [
    "pip",
    "setuptools",
    "wheel",
    "PyQt6",
    "huggingface_hub",
    "mlx-lm",
]


def log(msg: str) -> None:
    print(f"[mlx-bootstrap] {msg}", flush=True)


def run(cmd: list[str]) -> None:
    log("cmd: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_venv() -> None:
    if VENV_PYTHON.exists():
        log(f"Using existing runtime: {VENV_PYTHON}")
        return
    log(f"Creating runtime: {VENV_DIR}")
    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=True, upgrade_deps=False)
    builder.create(VENV_DIR)


def ensure_packages() -> None:
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", *REQUIRED_PACKAGES])


def diagnostics() -> None:
    log(f"Repo: {REPO_ROOT}")
    log(f"Manager: {DEFAULT_MANAGER}")
    log(f"Runtime python: {VENV_PYTHON} ({'present' if VENV_PYTHON.exists() else 'missing'})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap mlx_manager runtime")
    parser.add_argument("--launch", default=str(DEFAULT_MANAGER), help="Path to mlx_manager.py to launch after setup")
    parser.add_argument("--ensure-only", action="store_true", help="Only create/repair the runtime; do not launch the GUI")
    parser.add_argument("--reason", default="", help="Optional reason shown in logs")
    args = parser.parse_args()

    if args.reason:
        log(f"Reason: {args.reason}")
    diagnostics()

    try:
        ensure_venv()
        ensure_packages()
    except subprocess.CalledProcessError as exc:
        log(f"Setup failed with exit code {exc.returncode}")
        return exc.returncode or 1
    except Exception as exc:
        log(f"Setup failed: {exc}")
        return 1

    if args.ensure_only:
        log("Runtime is ready.")
        return 0

    manager = Path(args.launch).resolve()
    if not manager.exists():
        log(f"Manager not found: {manager}")
        return 1

    env = dict(os.environ)
    env["MLX_MANAGER_SKIP_REEXEC"] = "1"
    log(f"Launching manager: {manager}")
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(manager)], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
