@echo off
REM Pink Edge AI (Streamlit web edition) launcher — opens a local browser UI, same real models
REM and SQLite cache as the desktop app (Start.bat / GUI.py). First run needs internet once to
REM download model weights + Python deps; after that it's a local-only web server (no cloud calls).

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
    echo Dependency install failed. Check your internet connection and re-run Start_Web.bat.
    pause
    exit /b 1
)

echo Launching Pink Edge AI (Streamlit)...
python -m streamlit run streamlit_app.py

if errorlevel 1 (
    echo.
    echo Pink Edge AI exited with an error. See the message above.
    pause
)
