#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/backend"
[ ! -d ".venv" ] && python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -q
touch models/best_roof.pt models/best_solar.pt 2>/dev/null || true
export GEOSCAN_DATABASE_URL="sqlite:///./dev.db"
uvicorn app.main:app --host 0.0.0.0 --port 8766 --log-level info &
sleep 2
cd ../flask
python app.py &
echo "Frontend: http://localhost:8765/index.html  |  API: http://localhost:8766/docs"
wait
