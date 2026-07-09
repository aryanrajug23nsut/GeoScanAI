@echo off
echo ===================================================
echo   GeoScan.AI - Auto Environment Setup (Windows)
echo ===================================================

cd /d "%~dp0backend"

echo [1/4] Deleting old virtual environment (if exists)...
if exist ".venv" rmdir /s /q ".venv"

echo [2/4] Creating new Python virtual environment...
python -m venv .venv

echo [3/4] Activating environment and upgrading pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [4/4] Installing dependencies (auto-resolving best versions)...
pip install -r requirements.txt

echo.
echo ===================================================
echo   Setup Complete!
echo   To start the backend, run: start.bat
echo ===================================================
pause