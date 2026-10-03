@echo off
REM Pink Edge AI (HTML/JS web edition) launcher -- a third UI, separate from the Tkinter desktop
REM app (Start.bat) and the Streamlit web app (Start_Web.bat), but backed by the same real models
REM and the same SQLite cache. Starts App\web_api.py (Flask), which also serves the static files
REM in "Web GUI\" at the same address -- open it in your browser once it says "Running on".

cd /d "%~dp0App"

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
    echo Dependency install failed. Check your internet connection and re-run Start_Web_GUI.bat.
    pause
    exit /b 1
)

echo Launching Pink Edge AI (HTML/JS edition) on http://127.0.0.1:5000 ...
start "" http://127.0.0.1:5000
python web_api.py

if errorlevel 1 (
    echo.
    echo Pink Edge AI exited with an error. See the message above.
    pause
)
