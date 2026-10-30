@echo off
REM ===================================================================
REM  Start the web app. If setup_windows.bat was not run yet, this
REM  script tells you instead of failing with a confusing error.
REM ===================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo The virtual environment is missing.
    echo Run setup_windows.bat first, then try again.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
echo Starting the app at http://localhost:8501 ...
echo Close this window (or press Ctrl+C) to stop the app.
streamlit run app.py
pause
