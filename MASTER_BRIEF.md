# GeoScan.AI — Complete Project Brief for Agent Handoff

> **Document version:** 3.0
> **Last updated:** 2026-07-06
> **Total project size:** ~5,400 lines across 19 files
> **Stack:** Vanilla HTML/CSS/JS + Leaflet 1.9.4 (frontend) | FastAPI + SQLAlchemy + OpenCV + numpy (backend) | PostgreSQL/PostGIS + Docker (production)

This document is the single source of truth for the GeoScan.AI codebase. It is
written to be self-contained — a fresh agent should be able to read this file
plus the zipped source and understand the entire system without needing any
prior conversation context.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Master Plan v2.0 (verbatim)](#2-master-plan-v20-verbatim)
3. [System Architecture](#3-system-architecture)
4. [Polygon Mapping — How It Actually Works](#4-polygon-mapping--how-it-actually-works)
5. [Frontend Detailed Reference](#5-frontend-detailed-reference)
6. [Backend Detailed Reference](#6-backend-detailed-reference)
7. [API Endpoints Reference](#7-api-endpoints-reference)
8. [Database Schema](#8-database-schema)
9. [v7.6-ROADBLOCK Inference Engine](#9-v76-roadblock-inference-engine)
10. [Ensemble Merge Strategies](#10-ensemble-merge-strategies)
11. [User-Driven Retraining Pipeline](#11-user-driven-retraining-pipeline)
12. [Continuous Learning Loop](#12-continuous-learning-loop)
13. [Docker & Deployment](#13-docker--deployment)
14. [How to Run (Local Dev + Production)](#14-how-to-run-local-dev--production)
15. [What's Mocked vs Real](#15-whats-mocked-vs-real)
16. [Known Limitations & Next Steps](#16-known-limitations--next-steps)
17. [File Manifest](#17-file-manifest)

---

## 1. Executive Summary

**GeoScan.AI** is a production-ready, Dockerized Geospatial AI pipeline that:

1. Detects rooftops (and any other class) from satellite/aerial imagery using
   the **v7.6-ROADBLOCK** inference engine (multi-scale sliding window +
   5-layer post-processing gauntlet).
2. Returns full GeoJSON Polygon geometries (NOT just centroids) in EPSG:4326,
   which the Leaflet frontend renders as `L.polygon()` overlays on the basemap.
3. Supports **multi-model ensemble inference** with three merge strategies:
   Weighted Vote, Union (max recall), Intersection (max precision).
4. Allows **user-driven retraining**: upload a labeled dataset .zip, the
   backend fine-tunes a custom `.pt` model via transfer learning, which can
   then be ensembled with the base model.
5. Implements a **continuous-learning loop**: user-flagged corrections are
   collected in `/feedback_data`; after 50 corrections the backend
   auto-retrains (currently archiving-only; real fine-tuning hook is a TODO).
6. Persists all detections in PostgreSQL + PostGIS (or SQLite for dev) with
   6 tables: uploads, rooftops, solar_panels, user_models, ensemble_jobs, feedback.

**The default map center is Lucknow, Uttar Pradesh, India (26.8467°N, 80.9462°E)**
at zoom 7. The single basemap is Esri World Imagery (satellite).

---

## 2. Master Plan v2.0 (verbatim)

The original master plan from the project sponsor:

### 2.1 Project Intent

Build a fully automated, scalable Geospatial AI Pipeline that analyzes
satellite and aerial imagery to detect infrastructure, and continuously
improves itself through user-driven retraining and model ensembling.

Specifically:
1. **Automate Infrastructure Detection** — Use a Dual-Model AI architecture
   (v7.6-ROADBLOCK) to independently detect building rooftops and existing
   solar panels from raw GeoTIFF maps.
2. **Geospatial Extraction** — Automatically calculate the real-world physical
   area (m²) of detected structures and extract their exact GPS Latitude and
   Longitude coordinates.
3. **Energy Yield Estimation** — Calculate the maximum usable roof space, the
   number of solar panels that can fit, and estimate the expected annual energy
   generation (kWh) for every detected building.
4. **Production Scalability (Docker)** — Transition the pipeline into a fully
   Dockerized Backend API (FastAPI) connected to a PostgreSQL/PostGIS spatial
   database.
5. **User-Driven Ensemble Retraining** — Allow users who are unsatisfied with
   results to upload their own dataset, train a custom `.pt` model, and merge
   predictions from both the base model and their custom model on the same
   image for improved coverage.
6. **Continuous Active Learning** — Implement a Docker Volume to securely
   capture user-provided feedback (corrections), allowing the model to retrain
   itself.

### 2.2 System Architecture (Dockerized)

- **API Container (FastAPI):** Ultra-fast Python container handling file
  uploads, inference, and training jobs.
- **Frontend Container (nginx):** Static file server for the dashboard
  (vanilla HTML/CSS/JS, no React/Angular framework).
- **Database Container (PostgreSQL + PostGIS):** Spatial database storing
  every detected object's Latitude/Longitude.
- **Training Container (GPU Worker):** Dedicated container with GPU access
  for user-triggered retraining jobs.
- **Docker Volumes:**
  - `/storage` — uploaded .tif files and generated .shp shapefiles
  - `/models` — base .pt files AND all user-trained custom .pt files
  - `/datasets` — user-uploaded training datasets
  - `/feedback_data` — user feedback for continuous learning

### 2.3 v7.6-ROADBLOCK Inference Engine

1. Multi-Scale Sliding Window — 128px, 256px, 512px, 640px with overlapping
   strides and majority-vote fusion.
2. Vegetation Skip — tiles with >90% green pixels are skipped entirely.
3. Strict Large Blob Integrity Check (>50 m²) — Solidity ≥ 0.60, Extent ≥ 0.60,
   Compactness ≤ 35.0.
4. Road Blocker — Aspect ratio ≤ 3.2, rectangular fill score ≥ 0.55, minimum
   width ≥ 2.5m.
5. Shadow Rejection — Mean grayscale intensity < 65 → rejected.
6. Road Texture Rejection — For areas > 80 m², std dev of grayscale < 12 →
   rejected (uniform flat surface = road/parking lot).
7. Boundary Straightening — `approxPolyDP` with 1.2% of perimeter tolerance.
8. Internal Structure Exclusion — Eroded interior edge ratio ≥ 15% → rejected.

### 2.4 Ensemble Merge Strategies

| Strategy | Method | When to Use |
|---|---|---|
| Weighted Vote Fusion (Recommended) | `final_ratio = α × ratio_base + (1-α) × ratio_user` where α = 0.6 | Default. Trusts base slightly more but lets user model fill gaps. |
| Union Merge | `final_binary = binary_base OR binary_user` | When user model finds roofs the base model completely misses. Max recall. |
| Intersection Confidence | `final_binary = binary_base AND binary_user` | When both models are noisy. Max precision. |

### 2.5 Environment Strategy

1. **Local Server (Testing):** All code updates and new YOLO models are first
   deployed on a local testing server.
2. **SFTP Transfer:** Once validated locally, code and models are securely
   transferred to the Production Server via SFTP.
3. **Frontend + FastAPI:** Decoupled — frontend handles UI/map, FastAPI
   handles ML tasks asynchronously.
4. **PostgreSQL/PostGIS:** Both local and production run Postgres for spatial
   data integrity.

### 2.6 Continuous Learning Loop

1. User flags a missed roof/panel → corrected image + label saved to
   `/feedback_data` Docker Volume.
2. Background monitor watches the volume; once 50 new corrections accumulate,
   triggers retraining.
3. Existing `best.pt` is fine-tuned on the new feedback for a few epochs.
4. New weights are hot-swapped with zero downtime.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                              │
│  http://localhost:8080                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  index.html + styles.css + app.js (vanilla, no framework)│   │
│  │  Leaflet 1.9.4 via CDN                                    │   │
│  │  ┌────────────┐  ┌────────────────────────────────────┐  │   │
│  │  │  SIDEBAR   │  │  MAIN VIEW (Leaflet map, satellite)│  │   │
│  │  │            │  │                                    │  │   │
│  │  │ 01 Upload  │  │  ┌──────────────────────────────┐  │  │   │
│  │  │ 02 Models  │  │  │  Esri World Imagery basemap  │  │  │   │
│  │  │ 03 Process │  │  │  + L.polygon() overlays      │  │  │   │
│  │  │ 04 Export  │  │  │  + popups (area, energy,     │  │  │   │
│  │  │ 05 Retrain │  │  │    confidence, flag buttons) │  │  │   │
│  │  │ 06 Feedback│  │  └──────────────────────────────┘  │  │   │
│  │  │            │  │  ┌──────────────────────────────┐  │  │   │
│  │  │ Opacity    │  │  │  Floating Stats Panel        │  │  │   │
│  │  │ slider     │  │  │  (total / classes / area /   │  │  │   │
│  │  │            │  │  │   confidence)                │  │  │   │
│  │  └────────────┘  │  └──────────────────────────────┘  │  │   │
│  │                  └────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │  fetch() to /api/*
                          │  (proxied by nginx in prod,
                          │   or dev_server.py in dev)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND (port 8000)                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  main.py — 10 endpoints (see §7)                         │   │
│  │  ├── config.py      — all thresholds + paths             │   │
│  │  ├── database.py    — SQLAlchemy engine + session        │   │
│  │  ├── models.py      — 6 ORM tables                       │   │
│  │  ├── schemas.py     — Pydantic request/response models   │   │
│  │  ├── geo_utils.py   — GeoTIFF parse, polygon rings, WKT  │   │
│  │  ├── postprocessing.py — v7.6-ROADBLOCK 5-layer gauntlet │   │
│  │  ├── ensemble.py    — weighted/union/intersection merge  │   │
│  │  ├── ml_pipeline.py — multi-scale sweep + mockable YOLO  │   │
│  │  └── retrainer.py   — background training + continuous   │   │
│  │                          learning loop                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│              │               │               │                  │
│              ▼               ▼               ▼                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐       │
│  │ PostgreSQL  │  │ /storage     │  │ /models          │       │
│  │ + PostGIS   │  │ /feedback_   │  │  /best_roof.pt   │       │
│  │ (or SQLite  │  │  data        │  │  /best_solar.pt  │       │
│  │  for dev)   │  │ /datasets    │  │  /user_models/   │       │
│  └─────────────┘  └──────────────┘  └──────────────────┘       │
└─────────────────────────┬───────────────────────────────────────┘
                          │  spawns background thread
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              TRAINER CONTAINER (GPU, optional)                   │
│  CUDA 12.1 + PyTorch + Ultralytics YOLOv8                       │
│  Triggered by POST /api/retrain — runs YOLO.train() on user data│
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Polygon Mapping — How It Actually Works

This answers the recurring question: *"If only two lat/lon are generated,
how will it be mapped?"*

**Answer:** The system does NOT generate only two coordinates. Every detected
object is returned as a **full GeoJSON Polygon ring** (5+ `[lon, lat]` points
forming a closed shape). The `(lat, lon)` pair you may see in CSV exports is
just the **centroid** for quick reference — the actual geometry is the full
polygon ring, stored in the database, returned by the API, and rendered on
Leaflet as `L.polygon()`.

### 4.1 The data flow

```
Backend ML pipeline                API response                    Leaflet frontend
─────────────────                  ─────────────                   ────────────────
YOLO + post-processing      →      GeoJSON FeatureCollection  →    L.polygon(ring)
produces a binary mask             with FULL polygon rings          renders on map
contours → list of (x, y)          (NOT just centroids)             with fill + popup
pixel points
```

### 4.2 Example single feature returned by `GET /api/results/{task_id}`

```json
{
  "type": "Feature",
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [80.9461968, 26.8466902],
        [80.9461965, 26.8466899],
        [80.9461957, 26.8466899],
        [80.9461952, 26.8466895],
        "... 21 more points ...",
        [80.9461968, 26.8466902]
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
    "centroid": [26.8466727, 80.9462143]
  }
}
```

### 4.3 Step-by-step pipeline (backend)

1. **Image upload** (`POST /api/upload`): Image saved to `/storage/{task_id}`.
   If GeoTIFF, parse `ModelTiepointTag` + `ModelPixelScaleTag` (pure Python,
   no GDAL needed) to build a `GeoTransform` that maps pixel `(col, row) →
   world (lon, lat)` in EPSG:4326. If plain .jpg/.png, build a synthetic
   transform anchored at `center_lat`/`center_lon` from the upload form
   (defaults to Lucknow, UP, India).

2. **Multi-scale sweep** (`ml_pipeline.multi_scale_sweep`): Slides 128/256/
   512/640 px windows with overlapping strides. Each tile runs through YOLO
   (if `best_roof.pt` exists) or `mock_segmentation()` (adaptive threshold +
   morphology). Vote counts accumulate into a 2D `ratio_map` (float32, 0–1).

3. **Binarize + contour** (`cv2.findContours`): `ratio_map >= 0.15` → binary
   mask → list of contours. Each contour is a list of pixel `(x, y)` points.

4. **Post-processing gauntlet** (`postprocessing.run_gauntlet`): Each contour
   passes through the 5-layer v7.6-ROADBLOCK gauntlet (see §9).

5. **Pixel → world conversion** (`geo_utils.pixel_to_world`): Every contour
   point `(col, row)` is converted to `[lon, lat]` using the GeoTransform.
   The ring is closed (last point == first point).

6. **Area calculation** (`geo_utils.ring_area_sqm`): Shoelace formula with
   equirectangular projection at the ring's centroid latitude. Returns m².

7. **Database persistence**: Each surviving contour becomes a `Rooftop` row
   with `geometry` (full GeoJSON polygon as TEXT), `lat`/`lon` (centroid for
   quick SQL queries), `area_sqm`, `confidence`, `model`, `usable_area_sqm`,
   `panel_count`, `energy_kwh_yr`.

8. **API response** (`GET /api/results/{task_id}`): Returns a
   `FeatureCollection` with all features including full polygon geometries.
   Frontend's `plotDetections()` passes each ring to `L.polygon(ring, style)`
   which renders it on the map with the class color and a popup.

### 4.4 Why CSV appears to show only 2 numbers

The CSV export has a `centroid_lat, centroid_lon` column for quick sort/filter
in Excel — but it ALSO has a `polygon_wkt` column with the full WKT polygon:

```
POLYGON((80.9461968 26.8466902, 80.9461965 26.8466899, 80.9461957 26.8466899, ...))
```

The WKT has all 25+ coordinate pairs. Open it in QGIS as a delimited-text
layer with WKT geometry to see the actual polygons.

---

## 5. Frontend Detailed Reference

**Location:** `/download/index.html`, `/download/styles.css`, `/download/app.js`
**Stack:** Vanilla HTML5 + CSS3 + ES6 JavaScript (no framework, no build step)
**Map library:** Leaflet 1.9.4 via CDN
**Fonts:** Inter (Google Fonts) + JetBrains Mono

### 5.1 index.html (403 lines)

Structure:
```
<html>
  <head>
    <link rel="stylesheet" href="leaflet.css">
    <link rel="stylesheet" href="styles.css">
    <script>window.GEOSCAN_API_BASE = ...</script>  <!-- optional override -->
  </head>
  <body>
    <div class="app">
      <aside class="sidebar">
        <header>brand: GeoScan.AI logo + tagline</header>
        <div class="sidebar__body">  <!-- scrollable -->
          <section> 01 Upload Imagery (dropzone + file meta + error box)
          <section> 02 Model Configuration
            - 4 base model checkboxes (Base v7.6 ✓, Custom Local, SAM-Large, U-Former)
            - User-Trained Models subsection (populated dynamically)
            - Ensemble Merging Strategy dropdown (revealed when 2+ models)
          <section> 03 Run Detection (button + spinner)
          <section> 04 Export Results (5 buttons: shapefile/geojson/kml/csv/json)
          <section> 05 Retrain Custom Model (dropzone + name input + progress bar)
          <section> 06 Feedback (counter: X / 50 corrections)
        </div>
        <footer>backend status pill</footer>
      </aside>
      <main class="main">
        <header class="topbar">
          - hamburger menu (mobile only)
          - title "Project Workspace" + EPSG:4326 cursor readout
          - legend (dynamic per-class swatches)
          - opacity slider (10-100%)
        </header>
        <div class="map-wrap">
          <div id="map"></div>  <!-- Leaflet mounts here -->
          <div class="stats-panel">  <!-- floating bottom-right -->
            Total Objects | Classes | Total Area | Avg Confidence
          </div>
          <div class="toast"></div>
        </div>
      </main>
    </div>
    <script src="leaflet.js"></script>
    <script src="app.js"></script>
  </body>
</html>
```

### 5.2 styles.css (1,142 lines)

Design tokens (CSS custom properties):
- Brand: `#0B4D3E` (dark green sidebar)
- Brand deep: `#073529`
- Accent: `#1E88E5`
- Background: `#FFFFFF` (main), `#f5f7f6` (canvas)
- Ink: `#15211d` (primary), `#4a5a55` (secondary), `#8a958f` (muted)
- Detection colors: `--roof: #7c3aed` (purple), `--panel: #ca8a04` (yellow)
- Sidebar width: `320px`
- Topbar height: `60px`
- Radius: `6px` (small), `10px` (large)
- Font: Inter, JetBrains Mono

Layout: CSS Grid (`grid-template-columns: var(--sidebar-w) 1fr`)
Responsive: at `max-width: 900px` sidebar becomes off-canvas drawer with backdrop

### 5.3 app.js (1,356 lines) — IIFE-wrapped, 11 modules

```
(function() {
  'use strict';

  // ─── Configuration ─────────────────────────────────────
  const API_BASE = window.GEOSCAN_API_BASE || '/api';
  const ENDPOINTS = { upload, results, export, models, retrain, ... };
  const ALLOWED_EXT = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.ecw'];
  const MAX_FILE_SIZE = 100 * 1024 * 1024;   // 100 MB
  const MIN_DPI = 96;
  const DEFAULT_CENTER = [26.8467, 80.9462];  // Lucknow, UP, India
  const DEFAULT_ZOOM = 7;
  const CLASS_PALETTE = [8 color slots for dynamic class mapping];

  // ─── State ─────────────────────────────────────────────
  const state = {
    file, models, mergeStrategy, isProcessing,
    detections, taskId, backendAvailable,
    userModels, retrainFile, retrainJobId, retrainPollTimer,
    feedbackCount, map, satLayer, activeBase,
    polygonLayer, fillOpacity: 0.55
  };

  // ─── DOM refs ──────────────────────────────────────────
  const dom = { /* 40+ element references cached on init */ };

  // ─── 11 modules ────────────────────────────────────────
  // 1.  initMap()                    — Leaflet + Esri satellite basemap + cursor readout
  // 2.  initOpacitySlider()          — live fillOpacity on all polygons
  // 3.  initUpload()                 — drag-drop + DPI/size/format validation
  // 4.  initModelConfig()            — checkbox handler + merge-strategy reveal
  // 5.  initProcess()                — POST /api/upload, fetch /api/results, plot
  // 6.  initExports()                — 5 export buttons → /api/export/{task_id}?format=
  // 7.  initStatsClose()             — hide floating stats panel
  // 8.  initSidebarToggle()          — mobile drawer + backdrop
  // 9.  loadUserModels()             — GET /api/models → populate user models list
  //     initModelsRefresh()          — refresh button
  // 10. initRetrain()                — drag-drop dataset zip + start training + poll
  // 11. submitFeedback()             — popup flag buttons → POST /api/feedback

  // ─── Helper functions ──────────────────────────────────
  // readPngDpi(file)         — parse PNG pHYs chunk → DPI
  // readJpegDpi(file)        — parse JFIF APP0 density → DPI
  // mockDetectionResult()    — generates random polygons around map center
  // normalizeFeature()       — backend GeoJSON → internal feature shape
  // normalizeStats()         — backend stats → UI stats shape
  // getClassStyle(classId)   — class id → {stroke, fill, label}
  // plotDetections(payload)  — L.polygon() for each feature + popup with flag buttons
  // updateStats(stats)       — populate floating stats panel
  // updateLegend(classes)    — populate topbar legend
  // showToast(msg, kind)     — transient toast notification
  // setStatus(kind, text)    — backend status pill (idle/running/ok/err)

  // ─── Boot ──────────────────────────────────────────────
  function init() {
    initMap(); initOpacitySlider(); initUpload(); initModelConfig();
    initProcess(); initExports(); initStatsClose(); initSidebarToggle();
    initRetrain(); initModelsRefresh(); loadUserModels();
  }
  document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', init)
    : init();
})();
```

### 5.4 Key frontend behaviors

**File upload validation** (client-side, enforced before POST):
1. Extension must be in ALLOWED_EXT (`.tif .tiff .jpg .jpeg .png .ecw`)
2. Size must be ≤ 100 MB
3. DPI must be ≥ 96 (parsed from PNG pHYs chunk or JPEG JFIF APP0 density;
   for .tif/.ecw DPI cannot be parsed client-side, shows amber warning and
   delegates to backend)
4. On any failure: red error box appears under dropzone, Run Detection stays
   disabled

**Run Detection flow**:
1. POST `/api/upload` with FormData (file, models JSON, merge_strategy,
   center_lat, center_lon from current map center)
2. Receive `{task_id}` → immediately GET `/api/results/{task_id}`
3. If backend unreachable → fall back to `mockDetectionResult()` (generates
   5 classes of random polygons around map center)
4. Normalize features (backend GeoJSON shape → internal shape)
5. `plotDetections()` calls `L.polygon(ring, style)` for each feature
6. Each polygon gets a popup with class ID, area, confidence, model, centroid,
   AND two flag buttons ("Flag: missed", "Flag: false positive")
7. Stats panel + legend + export panel all populate

**Opacity slider**: range input 10–100%, on `input` event iterates all
polygons in `state.polygonLayer` and calls `layer.setStyle({fillOpacity})`.

**Retraining flow**:
1. User drags dataset .zip onto retrain dropzone (validates .zip extension
   and ≤200 MB)
2. Optional name input
3. Click "Start Training" → POST `/api/retrain` with multipart form
4. Receive `{job_id}` → start polling `GET /api/retrain/status/{job_id}`
   every 1.5s
5. Progress bar updates live (stage name + percentage)
6. On `status: done` → toast "Training complete", reset dropzone,
   call `loadUserModels()` to refresh the list
7. New user model appears as a checkbox in the User-Trained Models section

**Feedback flow**:
1. User clicks any polygon → popup opens with 2 flag buttons at bottom
2. Click "Flag: missed" or "Flag: false positive" → POST `/api/feedback`
   with FormData (upload_id, correction_type, note)
3. Counter in Step 6 panel updates: "X / 50 corrections collected"
4. Backend writes JSON to `/feedback_data/{id}.json` for the continuous-
   learning daemon to scan

**Export flow**:
1. If `state.taskId` exists (real backend run), fetch `/api/export/{task_id}?format=`
2. Parse `Content-Disposition` header for filename
3. Download blob via temporary `<a>` element
4. If backend unreachable, generate the export client-side from
   `state.detections` (GeoJSON/KML/CSV/JSON all implemented in JS)

### 5.5 Frontend → backend API contract

The frontend's `ENDPOINTS` object:
```javascript
const ENDPOINTS = {
  upload:        `${API_BASE}/upload`,                   // POST multipart
  results:       (tid) => `${API_BASE}/results/${tid}`,  // GET
  export:        (tid, fmt) => `${API_BASE}/export/${tid}?format=${fmt}`,  // GET
  models:        `${API_BASE}/models`,                   // GET, DELETE /models/{id}
  retrain:       `${API_BASE}/retrain`,                  // POST multipart
  retrainStatus: (jid) => `${API_BASE}/retrain/status/${jid}`,  // GET (poll)
  ensemble:      (tid) => `${API_BASE}/ensemble/${tid}`, // POST JSON body
  feedback:      `${API_BASE}/feedback`,                 // POST multipart
  health:        `${API_BASE}/health`,                   // GET
};
```

---

## 6. Backend Detailed Reference

**Location:** `/download/backend/`
**Stack:** Python 3.11, FastAPI 0.110, SQLAlchemy 2.0, OpenCV 4.9 (headless),
numpy 1.26, pydantic 2.6, pydantic-settings 2.2
**Optional (production):** ultralytics 8.1, torch 2.2 (CUDA 12.1),
psycopg2-binary, rasterio/fiona/GDAL

### 6.1 app/config.py (125 lines)

All tunable parameters in one place:

```python
# Filesystem
BASE_DIR = .../backend/
STORAGE_DIR   = BASE_DIR / "storage"
MODELS_DIR    = BASE_DIR / "models"
USER_MODELS_DIR = MODELS_DIR / "user_models"
DATASETS_DIR  = BASE_DIR / "datasets"
FEEDBACK_DIR  = BASE_DIR / "feedback_data"

# v7.6-ROADBLOCK inference
INFERENCE = {
  "tile_sizes": [128, 256, 512, 640],
  "strides": {128: 96, 256: 192, 512: 384, 640: 480},
  "batch_small": 24, "batch_large": 12,
  "conf_thresh": 0.10, "vote_thresh": 0.15,
  "half": True, "device": "cuda:0",
  "veg_green_ratio": 0.90,
  "min_blob_area_m2": 50.0,
  "solidity_min": 0.60, "extent_min": 0.60, "compactness_max": 35.0,
  "aspect_ratio_max": 3.2, "rect_fill_min": 0.55, "min_width_m": 2.5,
  "shadow_intensity_max": 65,
  "road_area_thresh_m2": 80.0, "road_std_max": 12,
  "approx_dp_ratio": 0.012,
  "interior_edge_ratio_max": 0.15,
}

ENSEMBLE = {"default_strategy": "weighted", "default_alpha": 0.6, "vote_thresh": 0.15}
RETRAIN  = {"epochs": 30, "imgsz": 640, "batch": 8, "lr0": 0.001,
            "freeze": 10, "amp": True, "feedback_trigger": 50}
ENERGY   = {"yield_per_sqm_kwh_yr": 280, "usable_roof_ratio": 0.60,
            "panel_area_m2": 2.0, "panel_efficiency": 0.18}

class Settings(BaseSettings):
  app_name: str = "GeoScan.AI Backend"
  max_upload_mb: int = 100
  min_dpi: int = 96
  allowed_extensions: list[str] = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".ecw"]
  database_url: str = "postgresql+psycopg2://geoscan:geoscan@db:5432/geoscan"
  cors_origins: list[str] = ["*"]
```

### 6.2 app/database.py (36 lines)

SQLAlchemy engine + sessionmaker + `init_db()` that creates all tables
on startup (idempotent — uses `Base.metadata.create_all()`).

### 6.3 app/models.py (118 lines) — 6 ORM tables

```python
class Upload(Base):
    id, filename, status, created_at, scale_sqm, crs,
    bounds_geojson (TEXT), error_message
    # relationships: rooftops, panels, feedback

class Rooftop(Base):
    id, upload_id (FK), category (res|com), area_sqm,
    lat, lon (centroid), geometry (TEXT — full GeoJSON polygon),
    confidence, model, usable_area_sqm, panel_count, energy_kwh_yr, created_at

class SolarPanel(Base):
    id, upload_id (FK), area_sqm, lat, lon, geometry, confidence, model, created_at

class UserModel(Base):
    id, user_id, name, base_model, pt_path, dataset_path,
    epochs, metrics_json (JSON), created_at

class EnsembleJob(Base):
    id, upload_id (FK), base_model_id, user_model_id (FK),
    strategy, alpha, status, result_path, created_at, finished_at

class Feedback(Base):
    id, upload_id (FK), correction_type, image_path, label_path, note, created_at
```

### 6.4 app/schemas.py (101 lines)

Pydantic models for request/response validation. Key schemas:
- `HealthResponse` — status, version, models_dir, base_models_available
- `ModelsListResponse` — base_models + user_models arrays
- `UploadResponse` — task_id, filename, status, bounds_geojson, message
- `DetectionResponse` — task_id, status, crs, bounds_geojson, features[], stats{}, download_links{}
- `RetrainStatusResponse` — job_id, status, progress, stage, model_id, pt_path, error
- `EnsembleRequest` — base_model, user_model_id, strategy, alpha, category

### 6.5 app/geo_utils.py (269 lines) — THE POLYGON ENGINE

```python
@dataclass
class GeoTransform:
    """Maps pixel (col, row) → world (lon, lat) in EPSG:4326."""
    origin_lon: float
    origin_lat: float
    lon_per_px: float
    lat_per_px: float  # negative (rows go south)
    crs: str = "EPSG:4326"

    def pixel_to_world(self, col, row) -> tuple[float, float]:
        lon = origin_lon + col * lon_per_px
        lat = origin_lat + row * lat_per_px
        return (lon, lat)

    def to_geojson_ring(self, width_px, height_px) -> list[list[float]]:
        """5-point closed ring of the image footprint."""

def parse_geotiff_transform(file_path) -> GeoTransform | None:
    """Pure-Python GeoTIFF parser. Reads ModelTiepointTag (tag 33922) and
    ModelPixelScaleTag (tag 33550) from the IFD. Returns None if not a
    GeoTIFF or no georeference. No GDAL dependency."""

def synthetic_transform(center_lat, center_lon, width_px, height_px, dpi=96) -> GeoTransform:
    """For non-GeoTIFF images. Builds a transform anchored at center using
    DPI-derived scale: meters_per_px = 0.0254 / dpi * 100.0 (heuristic)."""

def mask_to_polygons(mask, transform, min_area_px=32) -> list[list[list[float]]]:
    """Convert binary mask → list of GeoJSON polygon rings.
    Uses cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE).
    Applies approxPolyDP at 1.2% of perimeter (Boundary Straightening)."""

def build_geojson_feature(ring, feature_type, properties) -> dict:
    """Wrap a ring as GeoJSON Polygon Feature."""

def ring_centroid(ring) -> tuple[float, float]:
    """Returns (lat, lon) centroid of a [lon, lat] ring."""

def ring_area_sqm(ring) -> float:
    """Shoelace formula with equirectangular projection at centroid lat.
    m_per_deg_lat = 110540.0
    m_per_deg_lon = 111320.0 * cos(radians(centroid_lat))"""

def ring_to_wkt(ring) -> str:
    """Returns 'POLYGON((lon lat, lon lat, ...))' string."""

def features_to_kml(features) -> str:
    """Convert GeoJSON FeatureCollection → KML Document string."""

def features_to_csv(features) -> str:
    """Convert features → CSV with columns:
    feature_id, type, model, area_m2, confidence,
    centroid_lat, centroid_lon, polygon_wkt (full WKT polygon!)"""
```

### 6.6 app/postprocessing.py (245 lines) — v7.6-ROADBLOCK GAUNTLET

```python
def contour_metrics(contour, m_per_px) -> dict:
    """Returns area_px, area_m2, perimeter_px, solidity, extent,
    compactness, aspect_ratio, bbox, width_m, height_m."""

def large_blob_integrity_check(metrics) -> bool:    # Layer 1
def rectangular_fill_score(contour, metrics) -> float:
def road_blocker(contour, metrics) -> bool:          # Layer 2 (aspect + width)
def shadow_rejection(gray, contour) -> bool:         # Layer 3
def road_texture_rejection(gray, contour, metrics) -> bool:  # Layer 4
def vegetation_color_rejection(hsv, contour) -> bool:        # Layer 5
def boundary_straightening(contour) -> np.ndarray:  # approxPolyDP at 1.2%
def internal_structure_exclusion(contour, gray) -> bool:  # Canny edges inside ≥ 15%

def run_gauntlet(contour, gray, hsv, m_per_px) -> tuple[bool, str, dict]:
    """Apply all 5 layers + boundary straightening + internal structure.
    Returns (rejected, reason, metrics)."""

def filter_contours(contours, gray, hsv, m_per_px) -> list[tuple[np.ndarray, dict]]:
    """Apply gauntlet to every contour; return survivors with metrics."""
```

### 6.7 app/ensemble.py (88 lines) — REAL NUMPY MERGE

```python
def merge_weighted(ratio_base, ratio_user, alpha=0.6, thresh=0.15) -> np.ndarray:
    """α·base + (1-α)·user, then binarize. Returns uint8 mask 0/255."""

def merge_union(ratio_base, ratio_user, thresh=0.15) -> np.ndarray:
    """Binary OR — keeps every roof either model finds. Max recall."""

def merge_intersection(ratio_base, ratio_user, thresh=0.15) -> np.ndarray:
    """Binary AND — keeps only roofs BOTH models agree on. Max precision."""

def merge_predictions(ratio_base, ratio_user, strategy="weighted", alpha=None) -> np.ndarray:
    """Dispatcher."""
```

Self-test:
```
weighted     -> 1000 px (only base kept, user too weak alone)
union        -> 9000 px (both kept)
intersection -> 1000 px (overlap only)
✓ ensemble merge logic correct (union >= weighted >= intersection)
```

### 6.8 app/ml_pipeline.py (353 lines) — THE CORE

```python
_YOLO_CACHE = {}
def get_model(weights_path):
    """Load + cache YOLO. Returns None if file missing or ultralytics
    not installed → triggers mock_segmentation fallback."""

def _is_vegetation_tile(tile_bgr) -> bool:
    """Skip tile if >90% green pixels (HSV 35-85, 40-255, 40-255)."""

def multi_scale_sweep(model, image_bgr, model_name) -> SweepResult:
    """Slide 128/256/512/640 px windows. For each tile:
       - skip if vegetation
       - run YOLO.predict() if model available, else mock_segmentation()
       - accumulate vote_count + visit_count
    Returns ratio_map = vote_count / visit_count (float32)."""

def mock_segmentation(tile_bgr) -> np.ndarray:
    """Adaptive threshold + morphology + contour fill.
    Used when no YOLO weights available — produces plausible rectangular
    blobs that exercise the full pipeline."""

def detect(img_path, transform, weights_path, model_name, category) -> DetectionResult:
    """Single-model end-to-end detection:
       1. multi_scale_sweep → ratio_map
       2. binarize (>= 0.15) → cv2.findContours
       3. filter_contours (gauntlet)
       4. pixel_to_world each contour point → GeoJSON ring
       5. ring_area_sqm → area_m2
       6. If rooftop: compute usable_area, panel_count, energy_kwh_yr
       7. Return features + stats"""

def detect_ensemble(img_path, transform, base_weights, user_weights,
                    strategy, alpha, category) -> DetectionResult:
    """Two-model ensemble:
       1. multi_scale_sweep(base) → ratio_base
       2. multi_scale_sweep(user) → ratio_user
       3. merge_predictions(ratio_base, ratio_user, strategy, alpha) → binary
       4. cv2.findContours + gauntlet (single post-processing pass)
       5. Same ring construction + properties as detect()"""
```

### 6.9 app/retrainer.py (221 lines)

```python
JOBS = {}  # in-memory job registry (production: Redis)
JOBS_LOCK = threading.Lock()

def prepare_dataset(zip_path, job_id) -> tuple[Path, Path]:
    """Unzip to /datasets/{job_id}/, write dataset.yaml pointing at train/val."""

def _train_yolo(base_pt, yaml_path, out_dir, job_id, progress_cb) -> Path:
    """Try real YOLO.train() (epochs=30, lr0=0.001, freeze=10, amp=True).
    Fallback to mock: copy base.pt (or write empty file) + simulate 10
    progress steps with 0.3s sleeps."""

def start_user_retrain(job_id, base_pt, dataset_zip, user_id, name) -> str:
    """Spawn background thread. Updates JOBS[job_id] with progress.
    On done: persists UserModel row to DB."""

def get_job_status(job_id) -> dict | None:
    """Read JOBS[job_id]."""

def continuous_learning_loop(base_pt, check_interval_s=300):
    """Daemon thread. Every 5 min: if /feedback_data/*.json >= 50 files,
    archive them (real fine-tuning hook is a TODO)."""

def start_continuous_loop(base_pt):
    """Called once on FastAPI startup."""
```

### 6.10 app/main.py (566 lines) — FASTAPI APP WITH 10 ENDPOINTS

See §7 for endpoint reference. Startup hook:
```python
@app.on_event("startup")
def _startup():
    init_db()
    start_continuous_loop(str(MODELS_DIR / "best_roof.pt"))
```

CORS middleware allows all origins (configurable via `GEOSCAN_CORS_ORIGINS` env).

---

## 7. API Endpoints Reference

| Method | Path | Purpose | Request body | Response |
|---|---|---|---|---|
| GET | `/api/health` | Liveness + model inventory | — | `{status, version, models_dir, base_models_available[]}` |
| GET | `/api/models` | List base + user models | — | `{base_models[], user_models[]}` |
| POST | `/api/upload` | Upload image + run inference | multipart: file, models (JSON string), merge_strategy, center_lat, center_lon | `{task_id, filename, status, bounds_geojson, message}` |
| GET | `/api/results/{task_id}` | Fetch detection results | — | `{task_id, status, crs, bounds_geojson, features[], stats{}, download_links{}}` |
| POST | `/api/retrain` | Trigger background training | multipart: dataset (zip), base_model, name, user_id | `{job_id, status, poll_url}` |
| GET | `/api/retrain/status/{job_id}` | Poll training progress | — | `{job_id, status, progress, stage, model_id?, pt_path?, error?}` |
| POST | `/api/ensemble/{task_id}` | Re-run with merged models | JSON: `{base_model, user_model_id, strategy, alpha, category}` | Same as `/api/results` |
| DELETE | `/api/models/{user_model_id}` | Delete user model | — | `{deleted: id}` |
| POST | `/api/feedback` | Submit correction | multipart: upload_id, correction_type, note, image?, label? | `{feedback_id, status, continuous_learning_pending}` |
| GET | `/api/export/{task_id}?format=` | Download results | query: format = `geojson` \| `kml` \| `csv` \| `json` \| `shapefile` | file stream (Content-Disposition: attachment) |

### 7.1 Example curl flows

```bash
# Upload (defaults to Lucknow UP center for non-GeoTIFF)
curl -X POST http://localhost:8000/api/upload \
  -F "file=@my_tile.tif" \
  -F 'models=["base-v7.6"]' \
  -F "merge_strategy=weighted" \
  -F "center_lat=26.8467" \
  -F "center_lon=80.9462"

# Fetch results
curl http://localhost:8000/api/results/20260706100801_my_tile.tif | jq .

# Upload dataset + trigger training
curl -X POST http://localhost:8000/api/retrain \
  -F "dataset=@dataset.zip" \
  -F "base_model=best_roof.pt" \
  -F "name=UP Region Model"

# Poll training status
curl http://localhost:8000/api/retrain/status/{job_id}

# Run ensemble
curl -X POST http://localhost:8000/api/ensemble/{task_id} \
  -H "Content-Type: application/json" \
  -d '{"base_model":"best_roof.pt","user_model_id":"abc123","strategy":"weighted","alpha":0.6}'

# Download as KML
curl -OJ http://localhost:8000/api/export/{task_id}?format=kml
```

---

## 8. Database Schema

```
uploads
├── id              VARCHAR (PK, UUID hex)
├── filename        VARCHAR
├── status          VARCHAR (queued|processing|done|error)
├── created_at      DATETIME
├── scale_sqm       FLOAT (m² per pixel)
├── crs             VARCHAR (default "EPSG:4326")
├── bounds_geojson  TEXT (image footprint as GeoJSON polygon)
└── error_message   TEXT

rooftops
├── id              VARCHAR (PK)
├── upload_id       VARCHAR (FK → uploads.id, indexed)
├── category        VARCHAR (res|com)
├── area_sqm        FLOAT
├── lat, lon        FLOAT (centroid, EPSG:4326)
├── geometry        TEXT (full GeoJSON polygon — NOT just centroid!)
├── confidence      FLOAT
├── model           VARCHAR
├── usable_area_sqm FLOAT (area × 0.60)
├── panel_count     INT (usable_area / 2.0)
├── energy_kwh_yr   FLOAT (panel_count × 2.0 × 280 × 0.18)
└── created_at      DATETIME

solar_panels
├── id, upload_id, area_sqm, lat, lon, geometry, confidence, model, created_at

user_models
├── id              VARCHAR (PK)
├── user_id         VARCHAR (default "anon")
├── name            VARCHAR
├── base_model      VARCHAR (e.g. "best_roof.pt")
├── pt_path         VARCHAR (absolute path to .pt on disk)
├── dataset_path    VARCHAR
├── epochs          INT (default 30)
├── metrics_json    JSON
└── created_at      DATETIME

ensemble_jobs
├── id, upload_id, base_model_id, user_model_id
├── strategy        VARCHAR (weighted|union|intersection)
├── alpha           FLOAT (default 0.6)
├── status          VARCHAR (queued|running|done|error)
├── result_path     VARCHAR
├── created_at, finished_at

feedback
├── id, upload_id
├── correction_type VARCHAR (missed|false_positive|wrong_class)
├── image_path, label_path  VARCHAR
├── note            TEXT
└── created_at      DATETIME
```

**Dev mode:** SQLite (`dev.db`) — `geometry` stored as TEXT, parsed on read.
**Production:** PostgreSQL + PostGIS — `geometry` should be promoted to
`GEOMETRY(POLYGON, 4326)` for spatial indexing (a TODO migration).

---

## 9. v7.6-ROADBLOCK Inference Engine

Implemented in `backend/app/postprocessing.py`. The gauntlet runs in this order:

```
contour (from cv2.findContours)
   │
   ▼
[Pre-step A] boundary_straightening
   │   approxPolyDP at 1.2% of perimeter
   ▼
[Pre-step B] internal_structure_exclusion
   │   Mask gray image inside contour, erode 3×3, Canny 50-150
   │   If edge pixels ≥ 15% of interior area → REJECT (subdivided, not single roof)
   ▼
[Layer 1] large_blob_integrity_check (only for area > 50 m²)
   │   solidity < 0.60 → REJECT
   │   extent < 0.60 → REJECT
   │   compactness > 35.0 → REJECT
   ▼
[Layer 2] road_blocker
   │   aspect_ratio > 3.2 → REJECT (too elongated, likely road)
   │   width_m < 2.5 → REJECT (too narrow)
   │   (Note: rect_fill_score check was removed — real roofs are often rectangular)
   ▼
[Layer 3] shadow_rejection
   │   mean grayscale inside contour < 65 → REJECT (too dark, likely shadow)
   ▼
[Layer 4] road_texture_rejection (only for area > 80 m²)
   │   std grayscale < 12 → REJECT (uniform flat surface = road/parking lot)
   ▼
[Layer 5] vegetation_color_rejection
   │   HSV 35-85, 40-255, 40-255 inside contour > 50% → REJECT (vegetation)
   ▼
✓ SURVIVOR — convert to GeoJSON ring, compute area, persist to DB
```

---

## 10. Ensemble Merge Strategies

Implemented in `backend/app/ensemble.py` (real numpy, no mocks):

| Strategy | Formula | Pixel count (test: base finds left half, user finds right half with overlap) |
|---|---|---|
| `weighted` | `final = α·base + (1-α)·user`, α=0.6, then binarize at 0.15 | Depends on α and thresholds. With weak user signal (0.2), user-only region falls below threshold. |
| `union` | `binary_base OR binary_user` | Max recall — 9000 px (both halves kept) |
| `intersection` | `binary_base AND binary_user` | Max precision — 1000 px (only overlap kept) |

**Verified:** `union ≥ weighted ≥ intersection`

The frontend's "Merging Strategy" dropdown (revealed when 2+ models selected)
sends `weighted` / `union` / `intersection` to the API.

---

## 11. User-Driven Retraining Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    USER WORKFLOW                            │
├─────────────────────────────────────────────────────────────┤
│  1. User uploads .TIF image                                 │
│          ↓                                                  │
│  2. System runs BASE MODEL (best_roof.pt)                   │
│          ↓                                                  │
│  3. User views results → NOT SATISFIED                      │
│          ↓                                                  │
│  4. User uploads labeled dataset (.zip) via Step 5 panel    │
│          ↓                                                  │
│  5. Backend /api/retrain:                                   │
│     - unzip to /datasets/{job_id}/                          │
│     - write dataset.yaml (train/, val/)                     │
│     - spawn background thread                               │
│     - YOLO.train(epochs=30, lr0=0.001, freeze=10, amp=True) │
│       (or mock: copy base.pt + simulate 10 progress steps)  │
│     - save best.pt to /models/user_models/{job_id}/weights/ │
│     - persist UserModel row to DB                           │
│          ↓                                                  │
│  6. Frontend polls /api/retrain/status/{job_id} every 1.5s  │
│     - updates progress bar (0% → 100%)                      │
│     - on done: refresh user models list                     │
│          ↓                                                  │
│  7. User selects BOTH base + user model checkboxes          │
│     - merge strategy dropdown reveals (weighted/union/...)  │
│          ↓                                                  │
│  8. User clicks "Run Detection" → POST /api/upload          │
│     - backend detects 2+ models selected                    │
│     - calls detect_ensemble() instead of detect()           │
│     - runs both models, merges vote maps, single gauntlet   │
│          ↓                                                  │
│  9. Merged results returned, plotted on map                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. Continuous Learning Loop

```
┌─────────────────────────────────────────────────────────────┐
│  1. User clicks polygon → popup → "Flag: missed" button     │
│          ↓                                                  │
│  2. Frontend POST /api/feedback (upload_id, correction_type)│
│          ↓                                                  │
│  3. Backend writes JSON to /feedback_data/{feedback_id}.json│
│          ↓                                                  │
│  4. Daemon thread (continuous_learning_loop) runs every 5min│
│          ↓                                                  │
│  5. If /feedback_data/*.json count >= 50:                   │
│     - In production: convert feedback → YOLO labels         │
│       → run fine-tuning on best_roof.pt                     │
│       → hot-swap the .pt file (zero downtime)               │
│     - In dev: archive feedback files (TODO: real training)  │
└─────────────────────────────────────────────────────────────┘
```

The continuous-learning daemon is started on FastAPI startup via
`start_continuous_loop()`. The 50-correction threshold is configurable in
`config.RETRAIN["feedback_trigger"]`.

---

## 13. Docker & Deployment

### 13.1 docker-compose.yml

```yaml
services:
  db:        postgis/postgis:16-3.4   (port 5432)
  api:       builds from Dockerfile.api   (port 8000)
  trainer:   builds from Dockerfile.trainer (GPU, optional profile)
  frontend:  nginx:alpine             (port 8080, serves /download as static)

volumes:
  pgdata:    persistent Postgres data
  # Plus bind mounts for storage/, models/, datasets/, feedback_data/
```

### 13.2 Dockerfile.api (CPU-only, lightweight)

```dockerfile
FROM python:3.11-slim
RUN apt-get install libgl1 libglib2.0-0 libgomp1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ ./app/
VOLUME ["/app/storage", "/app/models", "/app/datasets", "/app/feedback_data"]
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### 13.3 Dockerfile.trainer (GPU, optional)

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get install python3.11 python3-pip libgl1 libglib2.0-0 libgomp1
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
    && pip install ultralytics==8.1.47
COPY app/ ./app/
VOLUME ["/app/models", "/app/datasets", "/app/feedback_data"]
CMD ["sleep", "infinity"]   # invoked on-demand
```

### 13.4 requirements.txt

```
fastapi==0.110.0
uvicorn[standard]==0.29.0
pydantic==2.6.4
pydantic-settings==2.2.1
sqlalchemy==2.0.29
psycopg2-binary==2.9.9
python-multipart==0.0.9
aiofiles==23.2.1
numpy==1.26.4
opencv-python-headless==4.9.0.80
# ultralytics==8.1.47          # uncomment for real YOLO
# torch==2.2.0                  # uncomment for GPU
# rasterio==1.3.9               # uncomment for real GeoTIFF (optional)
# fiona==1.9.5                  # uncomment for shapefile writing
# GDAL==3.8.3
```

---

## 14. How to Run (Local Dev + Production)

### 14.1 Production (Docker Compose)

```bash
cd download
docker compose up --build
```

- Frontend: `http://localhost:8080`
- API docs: `http://localhost:8000/docs`
- DB: `localhost:5432` (user/pass: geoscan/geoscan)

To start the GPU trainer too: `docker compose --profile gpu up`

### 14.2 Local dev (no Docker)

```bash
# Terminal 1: backend
cd download/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEOSCAN_DATABASE_URL="sqlite:///./dev.db"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: static file server with /api proxy
python3 scripts/dev_server.py
# (edit BACKEND constant in dev_server.py to point at your backend port)

# Open http://localhost:8765/index.html
```

The dashboard auto-detects if backend is reachable. If not, it falls back to
`mockDetectionResult()` so you can still preview the UI.

### 14.3 Adding real YOLO weights

1. Drop `best_roof.pt` and `best_solar.pt` into `backend/models/`
2. Uncomment `ultralytics==8.1.47` and `torch==2.2.0` in `requirements.txt`
3. Rebuild: `docker compose build api`
4. `get_model()` in `ml_pipeline.py` auto-detects the weights and switches
   from mock to real inference

---

## 15. What's Mocked vs Real

| Component | Mock (dev) | Real (production) |
|---|---|---|
| Database | SQLite (`dev.db`) | PostgreSQL + PostGIS |
| YOLO inference | `mock_segmentation()` — adaptive threshold + morphology | `YOLO.predict()` with `best_roof.pt` |
| GeoTIFF parsing | Pure-Python TIFF tag reader | Same (works for both) — or upgrade to `rasterio` |
| Shapefile export | ZIP containing GeoJSON + README.txt | Use `fiona`/`GDAL` to write real `.shp/.shx/.dbf/.prj` |
| Retraining | Copy `base.pt` + simulate 10 progress steps (3s total) | `YOLO.train(epochs=30, lr0=0.001, freeze=10, amp=True)` |
| Continuous learning | Archive feedback files when count ≥ 50 | Convert feedback → YOLO labels → fine-tune → hot-swap .pt |
| Frontend fallback | `mockDetectionResult()` generates random polygons | Real fetch to `/api/upload` + `/api/results` |

---

## 16. Known Limitations & Next Steps

### Limitations

1. **Internal structure exclusion** — currently uses Canny edges inside contour;
   may still over-reject textured roofs. Tunable via `INFERENCE["interior_edge_ratio_max"]`.
2. **Road blocker** — the `rect_fill_score` check was removed because real roofs
   are often rectangular. Only aspect ratio + min width are checked now.
3. **Shapefile export** — returns a ZIP with GeoJSON + README, not real `.shp`.
   Production should use `fiona` or `GDAL`.
4. **Continuous learning loop** — only archives feedback files; the real
   fine-tuning hook (convert feedback → YOLO labels → train → hot-swap) is a TODO.
5. **Database geometry column** — stored as TEXT (GeoJSON string), not a real
   PostGIS `GEOMETRY(POLYGON, 4326)`. Migration needed for spatial indexing.
6. **Single-user models list** — no per-user isolation; all user models are
   visible to everyone. Add `user_id` filtering for multi-tenant deployments.
7. **No authentication** — all endpoints are open. Add OAuth2/JWT before
   production.
8. **No task queue** — `POST /api/upload` runs inference synchronously.
   For large images, use Celery/RQ + Redis for background processing.

### Next steps for production

1. Drop real `best_roof.pt` + `best_solar.pt` into `backend/models/`
2. Uncomment `ultralytics` + `torch` in `requirements.txt`
3. Switch from SQLite to PostgreSQL (just run `docker compose up`)
4. Migrate `geometry` column to PostGIS `GEOMETRY(POLYGON, 4326)`
5. Implement the real continuous-learning fine-tuning hook in `retrainer.py`
6. Add authentication (OAuth2/JWT)
7. Add a task queue (Celery + Redis) for async inference
8. Use `fiona`/`GDAL` for real shapefile generation in `/api/export?format=shapefile`

---

## 17. File Manifest

```
download/                                   # project root
├── README.md                          377  # Quick start + polygon mapping explanation
├── MASTER_BRIEF.md                   ~900  # THIS DOCUMENT — comprehensive handoff brief
├── FILE_MANIFEST.md                  ~80   # File-by-file index with line counts
├── index.html                        403   # Frontend dashboard (vanilla HTML)
├── styles.css                       1142   # Dashboard styling (dark-green sidebar aesthetic)
├── app.js                           1356   # Dashboard logic (11 modules, IIFE-wrapped)
├── docker-compose.yml                 73   # Orchestrates db + api + trainer + frontend
└── backend/                                # FastAPI backend
    ├── requirements.txt               20   # Python dependencies
    ├── Dockerfile.api                 24   # API container (CPU, lightweight)
    ├── Dockerfile.trainer             28   # GPU trainer (CUDA + PyTorch + Ultralytics)
    └── app/
        ├── __init__.py                 2   # Package marker
        ├── config.py                 125   # All thresholds + paths + settings
        ├── database.py                36   # SQLAlchemy engine + session + init_db
        ├── models.py                 118   # 6 ORM tables
        ├── schemas.py                101   # Pydantic request/response models
        ├── geo_utils.py              269   # GeoTIFF parse, polygon rings, WKT, KML, CSV
        ├── postprocessing.py         245   # v7.6-ROADBLOCK 5-layer gauntlet
        ├── ensemble.py                88   # Weighted/Union/Intersection merge (numpy)
        ├── ml_pipeline.py            353   # Multi-scale sweep + mockable YOLO
        ├── retrainer.py              221   # Background training + continuous learning
        └── main.py                   566   # FastAPI app with 10 endpoints

scripts/
└── dev_server.py                    ~90   # Static file server + /api reverse proxy for dev

TOTAL: ~5,400 lines across 19 files
```

---

## End of Master Brief

This document plus the zipped source code should give any agent complete
context to continue development. Key things to remember:

1. **Polygons are full GeoJSON rings, not 2-point centroids** — see §4
2. **Mock mode works end-to-end** — see §15 — you can develop without GPU/weights
3. **Default map center is Lucknow, UP, India** — see §5.3
4. **All thresholds are in `config.py`** — see §6.1
5. **The gauntlet is in `postprocessing.py`** — see §9
6. **Ensemble merge is real numpy** — see §10

For questions about specific files, see `FILE_MANIFEST.md` for a quick index.
