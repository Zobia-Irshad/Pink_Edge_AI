#!/usr/bin/env python3
"""
Pink Edge AI — Streamlit Cloud / repo-root entrypoint.

Streamlit Community Cloud often runs `streamlit_app.py` from the repository root.
All product code lives under `App/` (including modular `App/components/`).
This file only puts `App/` on sys.path and executes the real app.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "App"
if not APP_DIR.is_dir():
    raise SystemExit(f"App directory not found at {APP_DIR}")

# Prefer App/ for GUI, components, inference, auth_manager, etc.
sys.path.insert(0, str(APP_DIR))

# Execute the real Streamlit script as __main__ so Streamlit page config runs once.
runpy.run_path(str(APP_DIR / "streamlit_app.py"), run_name="__main__")
