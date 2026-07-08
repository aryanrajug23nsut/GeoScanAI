# Quick Start — GeoScan.AI

## Option 1: One-Command Start (Unix: Mac/Linux/WSL)

```bash
unzip geoscan-ai.zip
cd geoscan-ai
./start.sh
```

Open: http://localhost:8765/index.html

## Option 2: One-Command Start (Windows)

```cmd
unzip geoscan-ai.zip
cd geoscan-ai
start.bat
```

Open: http://localhost:8765/index.html

## Option 3: Docker (Production)

```bash
unzip geoscan-ai.zip
cd geoscan-ai
docker compose up --build
```

Open: http://localhost:8080

## Option 4: Manual Step-by-Step

### Terminal 1 — Backend

```bash
cd geoscan-ai/02-backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
touch models/best_roof.pt models/best_solar.pt   # placeholder weights
export GEOSCAN_DATABASE_URL="sqlite:///./dev.db"  # Windows: set GEOSCAN_DATABASE_URL=sqlite:///./dev.db
uvicorn app.main:app --host 0.0.0.0 --port 8766 --reload
```

### Terminal 2 — Frontend Proxy

```bash
cd geoscan-ai
python 03-scripts/dev_server.py
```

### Browser

Open: http://localhost:8765/index.html

## Verifying It Works

1. Visit http://localhost:8765/api/health — should return JSON with `status: "ok"`
2. Open http://localhost:8765/index.html — dashboard loads
3. Upload any `.png`, `.jpg`, or `.tif` image (≤100 MB, ≥96 DPI)
4. Click "Run Detection" — polygons appear on the satellite map centered on Lucknow, UP, India

## Stopping the Servers

- **Unix:** Press `Ctrl+C` in the terminal running `start.sh`
- **Windows:** Close both command windows that opened
- **Manual:** `pkill -f uvicorn && pkill -f dev_server.py`

## Adding Real YOLO Weights

The system runs in "mock mode" by default (uses adaptive thresholding instead
of real YOLO inference). To enable real detection:

1. Drop your `best_roof.pt` and `best_solar.pt` (YOLOv8 weights) into `02-backend/models/`
2. Edit `02-backend/requirements.txt` and uncomment these lines:
   ```
   ultralytics==8.1.47
   torch==2.2.0
   ```
3. Reinstall: `pip install -r requirements.txt`
4. Restart the backend

The system will auto-detect the weights and switch from mock to real inference.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: cv2` | `pip install opencv-python-headless` |
| Port 8765 already in use | Edit `03-scripts/dev_server.py` line `PORT = 8765` |
| Port 8766 already in use | Change uvicorn `--port` and update `dev_server.py`'s `BACKEND` constant |
| Database locked | Delete `02-backend/dev.db` and restart |
| Polygons not appearing | Open browser DevTools (F12) → Console tab → check for errors |
| Upload returns 415 | File extension not in `.tif/.tiff/.jpg/.jpeg/.png/.ecw` |
| Upload returns 413 | File > 100 MB |
| Upload shows "DPI too low" | Image < 96 DPI — re-export at higher resolution |

## Production Deployment (Docker Compose)

```bash
docker compose up --build -d          # start in background
docker compose logs -f api            # tail API logs
docker compose down                   # stop
docker compose down -v                # stop + delete database
```

Services:
- **Frontend:** http://localhost:8080
- **API:** http://localhost:8000 (Swagger docs at /docs)
- **PostgreSQL+PostGIS:** localhost:5432 (user: geoscan, pass: geoscan)

For GPU trainer: `docker compose --profile gpu up`

## Folder Structure

```
geoscan-ai/
├── 01-frontend/             # Dashboard (vanilla HTML/CSS/JS + Leaflet)
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── 02-backend/              # FastAPI + SQLAlchemy + OpenCV
│   ├── requirements.txt
│   ├── Dockerfile.api
│   ├── Dockerfile.trainer
│   ├── .env.example
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py        # All thresholds + settings
│   │   ├── database.py      # SQLAlchemy setup
│   │   ├── models.py        # 6 database tables
│   │   ├── schemas.py       # Pydantic models
│   │   ├── geo_utils.py     # Polygon engine (GeoTIFF, WKT, KML, CSV)
│   │   ├── postprocessing.py # v7.6-ROADBLOCK 5-layer gauntlet
│   │   ├── ensemble.py      # Weighted/Union/Intersection merge
│   │   ├── ml_pipeline.py   # Multi-scale YOLO sweep
│   │   ├── retrainer.py     # Background training + continuous learning
│   │   └── main.py          # FastAPI app with 10 endpoints
│   ├── models/
│   │   ├── best_roof.pt     # Placeholder (replace with real weights)
│   │   ├── best_solar.pt    # Placeholder (replace with real weights)
│   │   └── user_models/     # User-trained models appear here
│   ├── storage/             # Uploaded images stored here
│   ├── datasets/            # User-uploaded training datasets
│   └── feedback_data/       # Continuous-learning corrections
│
├── 03-scripts/              # Dev tooling
│   └── dev_server.py        # Local dev proxy server
│
├── 04-docker/               # Docker orchestration
│   └── docker-compose.yml   # 4-service production stack
│
├── 05-docs/                 # Documentation
│   ├── QUICKSTART.md        # This file
│   ├── README.md            # Full documentation
│   ├── MASTER_BRIEF.md      # 900-line technical brief for handoff
│   └── FILE_MANIFEST.md     # File-by-file index
│
├── start.sh                 # One-command start (Unix)
└── start.bat                # One-command start (Windows)
```

## Need Help?

Read `05-docs/MASTER_BRIEF.md` for the complete technical documentation — it's a
~900-line self-contained brief covering architecture, API, database schema,
the v7.6-ROADBLOCK inference engine, ensemble strategies, retraining pipeline,
continuous learning loop, and what's mocked vs real.
