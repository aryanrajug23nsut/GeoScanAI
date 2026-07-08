# File Manifest — GeoScan.AI

Quick-reference index of every file in the project, with line counts and
one-line descriptions. Use this to navigate the codebase.

Last updated: 2026-07-06

## Frontend (vanilla HTML/CSS/JS + Leaflet)

| File | Lines | Purpose |
|------|------:|---------|
| `index.html` | 403 | Dashboard structure: sidebar (6 step panels) + main view (Leaflet map + topbar + stats panel) |
| `styles.css` | 1142 | Dark-green sidebar aesthetic, flat design, responsive (collapses on mobile <900px) |
| `app.js` | 1356 | 11 IIFE modules: map init, opacity slider, upload validation, model config, detection, exports, retrain, feedback |

## Backend (FastAPI + SQLAlchemy + OpenCV)

| File | Lines | Purpose |
|------|------:|---------|
| `backend/app/__init__.py` | 2 | Package marker (`__version__ = "2.0.0"`) |
| `backend/app/config.py` | 125 | All tunable thresholds (INFERENCE, ENSEMBLE, RETRAIN, ENERGY), filesystem paths, Settings class |
| `backend/app/database.py` | 36 | SQLAlchemy engine + SessionLocal + `init_db()` (creates all tables on startup) |
| `backend/app/models.py` | 118 | 6 ORM tables: Upload, Rooftop, SolarPanel, UserModel, EnsembleJob, Feedback |
| `backend/app/schemas.py` | 101 | Pydantic request/response models for all 10 endpoints |
| `backend/app/geo_utils.py` | 269 | **THE POLYGON ENGINE** — GeoTransform, GeoTIFF parser, mask_to_polygons, ring_area_sqm, WKT/KML/CSV exporters |
| `backend/app/postprocessing.py` | 245 | v7.6-ROADBLOCK 5-layer gauntlet (boundary straightening, internal structure, large-blob integrity, road blocker, shadow, road texture, vegetation) |
| `backend/app/ensemble.py` | 88 | Real numpy merge: weighted (α·base+(1-α)·user), union (OR), intersection (AND) |
| `backend/app/ml_pipeline.py` | 353 | Multi-scale sliding window sweep + mockable YOLO inference + `detect()` and `detect_ensemble()` |
| `backend/app/retrainer.py` | 221 | Background training thread + continuous-learning daemon (every 5min checks /feedback_data) |
| `backend/app/main.py` | 566 | FastAPI app with 10 endpoints + startup hook (init_db + start_continuous_loop) |

## Docker & Deployment

| File | Lines | Purpose |
|------|------:|---------|
| `backend/requirements.txt` | 20 | Python deps: fastapi, uvicorn, sqlalchemy, psycopg2, opencv, numpy. (ultralytics/torch commented out) |
| `backend/Dockerfile.api` | 24 | CPU-only API container (python:3.11-slim + libgl1) |
| `backend/Dockerfile.trainer` | 28 | GPU trainer container (nvidia/cuda:12.1.0 + PyTorch + Ultralytics) |
| `docker-compose.yml` | 73 | Orchestrates 4 services: db (postgis), api, trainer (optional profile), frontend (nginx) |

## Dev Tooling

| File | Lines | Purpose |
|------|------:|---------|
| `scripts/dev_server.py` | ~90 | Static file server + reverse proxy: serves `/download` on :8765, proxies `/api/*` to backend on :8766 |

## Documentation

| File | Lines | Purpose |
|------|------:|---------|
| `README.md` | 377 | Quick start + polygon mapping explanation + API examples |
| `MASTER_BRIEF.md` | ~900 | **Comprehensive handoff document** — everything an agent needs to know |
| `FILE_MANIFEST.md` | ~80 | This file — quick-reference index |

## Total

**~5,400 lines across 22 files** (19 source + 3 docs)

## Directory tree

```
download/
├── README.md
├── MASTER_BRIEF.md
├── FILE_MANIFEST.md
├── index.html
├── styles.css
├── app.js
├── docker-compose.yml
└── backend/
    ├── requirements.txt
    ├── Dockerfile.api
    ├── Dockerfile.trainer
    └── app/
        ├── __init__.py
        ├── config.py
        ├── database.py
        ├── models.py
        ├── schemas.py
        ├── geo_utils.py
        ├── postprocessing.py
        ├── ensemble.py
        ├── ml_pipeline.py
        ├── retrainer.py
        └── main.py

scripts/
└── dev_server.py
```

## Quick lookup

| Question | Look in |
|----------|---------|
| How are polygons generated? | `backend/app/geo_utils.py` → `mask_to_polygons()` |
| How are polygons mapped to lat/lon? | `backend/app/geo_utils.py` → `GeoTransform.pixel_to_world()` |
| What are the v7.6-ROADBLOCK thresholds? | `backend/app/config.py` → `INFERENCE` dict |
| How does ensemble merge work? | `backend/app/ensemble.py` |
| How does the gauntlet reject contours? | `backend/app/postprocessing.py` → `run_gauntlet()` |
| What endpoints exist? | `backend/app/main.py` (10 endpoints) |
| How does the frontend call the backend? | `app.js` → `ENDPOINTS` object + `runDetection()` |
| How does retraining work? | `backend/app/retrainer.py` → `start_user_retrain()` |
| How does continuous learning work? | `backend/app/retrainer.py` → `continuous_learning_loop()` |
| How to run locally? | `README.md` → "Quick Start" |
| How to deploy with Docker? | `docker-compose.yml` + `README.md` |
