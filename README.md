<<<<<<< HEAD
# GeoScan.AI — Minimal Production Setup

## Files (only 20 total)
```
geoscan-min/
├── frontend/           (3 files: index.html, styles.css, app.js)
├── backend/            (15 files: requirements.txt, Dockerfile.api, .env.example, app/*.py, models/*.pt)
├── flask/              (1 file: app.py)
├── docker-compose.yml  (1 file: 3-service stack — db + api + flask)
├── start.sh            (Mac/Linux)
├── start.bat           (Windows)
└── README.md           (this file)
```

## Quick Start (Local)
```bash
./start.sh    # or start.bat on Windows
```
Open: http://localhost:8765/index.html

## Quick Start (Docker)
```bash
docker compose up --build
```
Open: http://localhost:8080

## Add Real YOLO Weights
Replace `backend/models/best_roof.pt` and `best_solar.pt` with your real weights.
Uncomment `ultralytics` + `torch` in `backend/requirements.txt`.
Run `pip install -r backend/requirements.txt` and restart.

## Architecture
Browser → Flask (:8765) → FastAPI (:8766) → SQLite/PostgreSQL + OpenCV
=======
# GeoScan.AI — Full Stack Geospatial Detection System

A production-ready, Dockerized pipeline for detecting rooftops (and any other
class) from satellite/aerial imagery. Implements the **v7.6-ROADBLOCK**
inference engine, a **User-Driven Ensemble Retraining Pipeline**, and an
**Unsupervised Continuous Learning Loop**.

> **About the "polygons on basemap" concern:** Every detected object is
> returned as a **full GeoJSON Polygon ring** (5+ `[lon, lat]` points
> forming a closed shape), not just two lat/lon coordinates. The two-number
> `(lat, lon)` you may see in CSV exports is only the **centroid** for quick
> reference — the actual geometry is the full polygon ring, stored in the
> database, returned by the API, and rendered on Leaflet as `L.polygon()`.
> See [§Polygon Mapping](#polygon-mapping--how-it-actually-works) below.

---

## Project Structure

```
download/
├── index.html                       # Frontend dashboard (vanilla HTML/CSS/JS + Leaflet)
├── styles.css                       # Dashboard styling (dark-green sidebar aesthetic)
├── app.js                           # Dashboard logic (file upload, model select, API calls)
├── docker-compose.yml               # Orchestrates db + api + trainer + frontend
└── backend/
    ├── app/
    │   ├── __init__.py
    │   ├── config.py                # All tunable thresholds + paths + settings
    │   ├── database.py              # SQLAlchemy engine + session + init_db()
    │   ├── models.py                # 6 tables: uploads, rooftops, panels, user_models, ensemble_jobs, feedback
    │   ├── schemas.py               # Pydantic request/response models
    │   ├── geo_utils.py             # GeoTIFF transform, polygon ring construction, WKT, KML, CSV
    │   ├── postprocessing.py        # v7.6-ROADBLOCK 5-layer gauntlet (real OpenCV code)
    │   ├── ensemble.py              # Weighted / Union / Intersection merge (real numpy)
    │   ├── ml_pipeline.py           # Multi-scale sweep + mockable inference
    │   ├── retrainer.py             # User retraining + continuous-learning loop
    │   └── main.py                  # FastAPI app with all 8 endpoints
    ├── requirements.txt
    ├── Dockerfile.api               # FastAPI container (CPU-only, lightweight)
    └── Dockerfile.trainer           # GPU trainer (CUDA + PyTorch + Ultralytics)
```

---

## Quick Start

### Option A — Docker Compose (recommended for production)

```bash
cd download
docker compose up --build
```

This starts:
- **db**        → PostgreSQL + PostGIS at `localhost:5432`
- **api**       → FastAPI at `localhost:8000`  (interactive docs at `/docs`)
- **frontend**  → nginx serving the dashboard at `localhost:8080`
- **trainer**   → GPU worker (optional, start with `docker compose --profile gpu up`)

Open `http://localhost:8080` to use the dashboard.

### Option B — Local development (no Docker)

```bash
# Terminal 1: backend
cd download/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEOSCAN_DATABASE_URL="sqlite:///./dev.db"   # SQLite for dev
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: static file server with /api proxy
python3 /home/z/my-project/scripts/dev_server.py   # serves on :8765, proxies /api → :8766
# (edit the BACKEND constant in dev_server.py to point at :8000 instead)

# Open http://localhost:8765/index.html
```

The dashboard automatically detects whether the backend is reachable. If not,
it falls back to a **mock detection** so you can still see how polygons render
on the map without running the full stack.

---

## Polygon Mapping — How It Actually Works

This is the answer to: *"I don't know if only two lat/lon are being generated
how it will be mapped."*

### The data flow

```
Backend ML pipeline                  API response                     Leaflet frontend
─────────────────                    ─────────────                    ────────────────
YOLO + postprocessing        →       GeoJSON FeatureCollection  →     L.polygon(ring)
produces a binary mask               with FULL polygon rings          renders it on map
contours → list of (x, y)            (NOT just centroids)             with fill + popup
pixel points

Example single feature returned by GET /api/results/{task_id}:

{
  "type": "Feature",
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [     ← This is the RING: 5+ [lon, lat] points, closed (first == last)
        [80.9461968, 26.8466902],
        [80.9461965, 26.8466899],
        [80.9461957, 26.8466899],
        ... 22 more points ...
        [80.9461968, 26.8466902]    ← closes the ring
      ]
    ]
  },
  "properties": {
    "type": "rooftop",
    "area_m2": 14.91,
    "confidence": 1.0,
    "model": "base-v7.6",
    "usable_area_sqm": 8.95,
    "panel_count": 4,
    "energy_kwh_yr": 403.0,
    "centroid": [26.8466727, 80.9462143]   ← Just for quick reference
  }
}
```

### Step-by-step pipeline (backend)

1. **Image upload** (`POST /api/upload`):
   The image is saved to `/storage/{task_id}`. If it's a GeoTIFF, its
   `ModelTiepointTag` + `ModelPixelScaleTag` are parsed (pure Python, no GDAL
   needed) to build a `GeoTransform` that maps pixel `(col, row) → world (lon, lat)`
   in EPSG:4326. If it's a plain `.jpg`/`.png`, a synthetic transform is
   built anchored at the `center_lat`/`center_lon` from the upload form
   (defaults to Lucknow, UP, India).

2. **Multi-scale sweep** (`ml_pipeline.multi_scale_sweep`):
   Slides 128/256/512/640 px windows across the image with overlapping
   strides. Each tile runs through YOLO (if `best_roof.pt` is available)
   or `mock_segmentation()` (adaptive threshold + morphology, used when
   no weights are installed). The vote counts accumulate into a 2D
   `ratio_map` (float32, 0.0–1.0).

3. **Binarize + contour** (`cv2.findContours`):
   `ratio_map >= 0.15` → binary mask → list of contours. Each contour is
   a list of pixel `(x, y)` points.

4. **Post-processing gauntlet** (`postprocessing.run_gauntlet`):
   Each contour passes through the 5-layer v7.6-ROADBLOCK gauntlet:
   - Boundary straightening (`approxPolyDP` at 1.2% of perimeter)
   - Internal structure exclusion (Canny edges inside contour > 15%)
   - Large-blob integrity (solidity ≥ 0.60, extent ≥ 0.60, compactness ≤ 35)
   - Road blocker (aspect ratio > 3.2 OR min width < 2.5 m)
   - Shadow rejection (mean grayscale < 65)
   - Road texture rejection (std < 12 for areas > 80 m²)
   - Vegetation color rejection (HSV green > 50%)

5. **Pixel → world conversion** (`geo_utils.pixel_to_world`):
   Every contour point `(col, row)` is converted to `[lon, lat]` using
   the GeoTransform. The ring is closed (last point == first point).

6. **Area calculation** (`geo_utils.ring_area_sqm`):
   Shoelace formula with equirectangular projection at the ring's
   centroid latitude. Returns m².

7. **Database persistence**:
   Each surviving contour is stored as a `Rooftop` row with:
   - `geometry`: full GeoJSON polygon (TEXT)
   - `lat`, `lon`: centroid (for quick SQL queries)
   - `area_sqm`, `confidence`, `model`, `usable_area_sqm`, `panel_count`,
     `energy_kwh_yr`

8. **API response** (`GET /api/results/{task_id}`):
   Returns a `FeatureCollection` with all features including their
   full polygon geometries. The frontend's `plotDetections()` function
   passes each ring to `L.polygon(ring, style)` which renders it on the
   map with the class color and a popup showing area + energy yield.

### Why the CSV shows only 2 numbers

The CSV export has a `centroid_lat, centroid_lon` column for quick sort/filter
in Excel — but it ALSO has a `polygon_wkt` column with the full WKT polygon:
```
POLYGON((80.9461968 26.8466902, 80.9461965 26.8466899, 80.9461957 26.8466899, ...))
```
The WKT has all 25+ coordinate pairs. Open it in QGIS as a delimited-text
layer with WKT geometry to see the actual polygons.

---

## API Endpoints

| Method | Path                                | Purpose                                                        |
|--------|-------------------------------------|----------------------------------------------------------------|
| GET    | `/api/health`                       | Liveness probe + base model inventory                          |
| GET    | `/api/models`                       | List base + user-trained models                                |
| POST   | `/api/upload`                       | Upload image + run base inference (returns task_id)            |
| GET    | `/api/results/{task_id}`            | Fetch detection results as GeoJSON FeatureCollection           |
| POST   | `/api/retrain`                      | Upload dataset .zip + trigger background training              |
| GET    | `/api/retrain/status/{job_id}`      | Poll training progress                                         |
| POST   | `/api/ensemble/{task_id}`           | Re-run inference with merged base + user models                |
| DELETE | `/api/models/{user_model_id}`       | Delete a user-trained model                                    |
| POST   | `/api/feedback`                     | Submit a correction (feeds continuous-learning loop)           |
| GET    | `/api/export/{task_id}?format=...`  | Download results: `geojson` / `kml` / `csv` / `json` / `shapefile` |

### Example: upload + get results

```bash
# Upload (defaults to Lucknow UP center for non-GeoTIFF images)
curl -X POST http://localhost:8000/api/upload \
  -F "file=@my_tile.tif" \
  -F 'models=["base-v7.6"]' \
  -F "merge_strategy=weighted" \
  -F "center_lat=26.8467" \
  -F "center_lon=80.9462"

# Response: { "task_id": "20260706100801_my_tile.tif", "status": "done", ... }

# Fetch results
curl http://localhost:8000/api/results/20260706100801_my_tile.tif | jq .

# Download as KML
curl -OJ http://localhost:8000/api/export/20260706100801_my_tile.tif?format=kml
```

---

## v7.6-ROADBLOCK Inference Engine

All thresholds live in `backend/app/config.py` under `INFERENCE`. The
5-layer gauntlet is in `backend/app/postprocessing.py`. Key parameters:

| Layer                              | Threshold                            |
|------------------------------------|--------------------------------------|
| Multi-scale tiles                  | 128, 256, 512, 640 px                |
| YOLO confidence                    | 0.10 (low → catch everything)        |
| Vote ratio threshold               | 0.15                                 |
| Vegetation skip                    | > 90% green pixels                   |
| Large-blob integrity (>50 m²)      | solidity ≥ 0.60, extent ≥ 0.60, compactness ≤ 35 |
| Road blocker                       | aspect > 3.2 OR min width < 2.5 m    |
| Shadow rejection                   | mean grayscale < 65                  |
| Road texture rejection (>80 m²)    | std grayscale < 12                   |
| Boundary straightening             | approxPolyDP at 1.2% of perimeter    |
| Internal structure exclusion       | Canny edges inside contour ≥ 15%     |

---

## Ensemble Merge Strategies

Implemented in `backend/app/ensemble.py` (real numpy, no mocks):

| Strategy       | Formula                                              | Use when                                |
|----------------|------------------------------------------------------|-----------------------------------------|
| `weighted`     | `final = α·base + (1-α)·user`, default α=0.6         | Default. Trusts base, lets user fill gaps. |
| `union`        | `final = binary_base OR binary_user`                 | User model finds roofs base misses. Max recall. |
| `intersection` | `final = binary_base AND binary_user`                | Both models noisy. Max precision.       |

The frontend's "Merging Strategy" dropdown (revealed when 2+ models selected)
sends `weighted` / `union` / `intersection` to the API.

---

## User-Driven Ensemble Retraining

1. User uploads `.tif`, runs base model → results not satisfactory
2. User uploads their labeled dataset as `.zip` via `POST /api/retrain`
3. Backend unpacks the zip to `/datasets/{job_id}/`, writes `dataset.yaml`
4. `retrainer.py` spawns a background thread that:
   - Loads `best_roof.pt` as starting weights (transfer learning)
   - Calls `YOLO.train()` with `epochs=30, lr0=0.001, freeze=10, amp=True`
   - Saves the resulting `.pt` to `/models/user_models/{job_id}/weights/best.pt`
   - Persists a `UserModel` row to the database
5. User polls `GET /api/retrain/status/{job_id}` until `status: done`
6. User calls `POST /api/ensemble/{task_id}` with `user_model_id` + strategy
7. Backend runs both models, merges their vote maps, post-processes once,
   and returns the merged GeoJSON results

If `ultralytics` is not installed (CPU-only dev box), the trainer writes a
mock `.pt` file so the API contract is testable end-to-end.

---

## Continuous Learning Loop

`retrainer.continuous_learning_loop()` runs as a daemon thread on FastAPI
startup. Every 5 minutes it checks `/feedback_data/*.json`:

- If count ≥ 50 (configurable via `RETRAIN['feedback_trigger']`):
  - In production: convert feedback to YOLO labels, run fine-tuning
  - In dev: archive feedback files (real training hook is a TODO)
- The resulting `best.pt` would hot-swap into production (zero-downtime)

---

## Database Schema

6 tables in `backend/app/models.py`:

| Table           | Key columns                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `uploads`       | id, filename, status, created_at, scale_sqm, crs, bounds_geojson           |
| `rooftops`      | id, upload_id, category, area_sqm, lat, lon, geometry (GeoJSON), confidence, model, usable_area_sqm, panel_count, energy_kwh_yr |
| `solar_panels`  | id, upload_id, area_sqm, lat, lon, geometry, confidence, model             |
| `user_models`   | id, user_id, name, base_model, pt_path, dataset_path, epochs, metrics_json |
| `ensemble_jobs` | id, upload_id, base_model_id, user_model_id, strategy, alpha, status       |
| `feedback`      | id, upload_id, correction_type, image_path, label_path, note               |

Production uses PostgreSQL + PostGIS (the `geometry` column could be promoted
to a real PostGIS `GEOMETRY(POLYGON, 4326)` for spatial indexing). Dev mode
uses SQLite with `geometry` as plain TEXT — the API parses it on read.

---

## Frontend (Web Dashboard)

The dashboard is at `download/index.html` (vanilla HTML/CSS/JS + Leaflet 1.9.4
via CDN — no build step, no framework). Key features:

- **Map centered on Uttar Pradesh, India** (Lucknow, 26.8467°N, 80.9462°E)
  at zoom 7 by default
- **Single basemap**: Esri World Imagery (satellite)
- **Drag-drop upload** with strict validation:
  - File extensions: `.tif .tiff .jpg .jpeg .png .ecw`
  - Max size: 100 MB
  - Min DPI: 96 (parsed from PNG `pHYs` chunk or JPEG `JFIF APP0` density)
- **Multi-model selection**: 4 checkboxes (Base v7.6, Custom Local, SAM-Large, U-Former)
- **Ensemble merge dropdown**: appears when 2+ models selected
  (Weighted Vote / Union / Intersection)
- **Polygon opacity slider** (10–100%) — live updates all polygons on map
- **Stats panel**: total objects, classes, total area, avg confidence
- **Dynamic legend**: per-class color swatches with counts
- **Export panel**: Shapefile / GeoJSON / KML / CSV / JSON
- **Polygon popups**: clicking a polygon shows area, confidence, model, centroid
- **Responsive**: sidebar collapses to off-canvas drawer on mobile

When you click "Run Detection":
1. `app.js` POSTs the file to `/api/upload` (via the dev proxy or nginx in prod)
2. Receives `task_id`, immediately fetches `/api/results/{task_id}`
3. Normalizes features and stats to a consistent internal shape
4. Calls `L.polygon(ring, style)` for each feature → renders on map
5. Updates legend, stats panel, export panel

If the backend is unreachable, it falls back to `mockDetectionResult()`
which generates random polygons around the current map center — useful
for previewing the UI without running the full stack.

---

## Environment Strategy: Local → Production

Per the master plan §5:

1. **Local testing** — Run with SQLite + CPU-only `mock_segmentation`.
   No GPU, no YOLO weights, no Postgres needed.
2. **SFTP transfer** — Once validated locally, `scp` the `backend/` dir
   and the model `.pt` files to the production server.
3. **Production** — `docker compose up` on the prod server. The compose
   file mounts `/storage`, `/models`, `/datasets`, `/feedback_data` as
   named volumes so they persist across container restarts.

---

## Adding Real YOLO Weights

1. Drop `best_roof.pt` and `best_solar.pt` into `backend/models/`
2. Uncomment `ultralytics==8.1.47` and `torch==2.2.0` in `requirements.txt`
3. Rebuild the API container: `docker compose build api`
4. The `get_model()` function in `ml_pipeline.py` will auto-detect the
   weights and switch from mock to real inference

The mock segmentation (`mock_segmentation()`) uses adaptive thresholding +
morphology on the grayscale image. It produces plausible rectangular blobs
that exercise the full pipeline (contours → gauntlet → GeoJSON → map), but
obviously real YOLO weights are needed for production accuracy.
>>>>>>> 7b117ba2923814babbc017ba693bb2e219e99904
