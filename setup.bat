@echo off
REM ── QuantEdge Setup Script (Windows) ─────────────────────────────────────────
REM Run once: setup.bat

echo ========================================
echo   QuantEdge -- Setup Script (Windows)
echo ========================================

echo.
echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)
echo       Done: venv\ created

echo.
echo [2/5] Activating venv and upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

echo.
echo [3/5] Installing dependencies (this may take 2-5 minutes)...
pip install -r requirements.txt --quiet
echo       Done: all packages installed

echo.
echo [4/5] Setting up .env file...
if not exist .env (
    copy .env.example .env
    echo       Created .env from .env.example
    echo       ^>^> Edit .env to add your API keys (optional^)
) else (
    echo       .env already exists -- skipping
)

echo.
echo [5/5] Creating data directories...
if not exist data\cache   mkdir data\cache
if not exist data\exports mkdir data\exports
echo       Done

echo.
echo ========================================
echo   Setup complete!
echo ========================================
echo.
echo   To run the Streamlit UI:
echo     venv\Scripts\activate
echo     start_streamlit.bat
echo.
echo   To run the backend API:
echo     venv\Scripts\activate
echo     start_api.bat
echo.
echo   To run tests:
echo     venv\Scripts\activate
echo     python -m pytest tests\ -v
echo.
echo   DEMO_MODE=true by default -- no API keys needed.
echo   Edit .env to add keys and set DEMO_MODE=false for live data.
echo.
pause
