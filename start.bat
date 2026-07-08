@echo off
cd /d "%~dp0backend"
if not exist ".venv" python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt -q
set GEOSCAN_DATABASE_URL=sqlite:///./dev.db
start "Backend" cmd /k "uvicorn app.main:app --host 0.0.0.0 --port 8766"
timeout /t 2 /nobreak > nul
cd ..\flask
start "Frontend" cmd /k "python app.py"
echo Frontend: http://localhost:8765/index.html
