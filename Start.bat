@echo off
REM Pink Edge AI (Desktop) launcher.
REM First run downloads real model weights from Hugging Face (a few hundred MB) and installs
REM Python deps if missing -- needs internet once. After that, GUI.py runs fully offline.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ from https://python.org and try again.
    pause
    exit /b 1
)

echo Checking Python dependencies (this only installs anything missing)...
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency install failed. Check your internet connection and re-run Start.bat.
    pause
    exit /b 1
)

echo Launching Pink Edge AI...
python GUI.py

if errorlevel 1 (
    echo.
    echo Pink Edge AI exited with an error. See the message above.
    pause
)
