@echo off
REM ===================================================================
REM  Self Updating Documentation Assistant - one time setup (Windows)
REM  Double click this file, or run it from the project folder.
REM ===================================================================
cd /d "%~dp0"

echo.
echo [1/3] Creating the virtual environment (.venv) ...
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m venv .venv
) else (
    python -m venv .venv
)
if errorlevel 1 (
    echo.
    echo ERROR: Python was not found. Install Python 3.10 or newer from
    echo        https://www.python.org/downloads/ and tick
    echo        "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

echo [2/3] Upgrading pip ...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip

echo [3/3] Installing the packages from requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed. Check your internet connection
    echo        and try again.
    pause
    exit /b 1
)

echo.
echo Setup finished. Start the app with:  run_app.bat
pause
