"""FastAPI application — exposes all 8 endpoints from the master plan §4.4.

Endpoints
---------
GET  /api/health                          → liveness probe + model inventory
GET  /api/models                          → list base + user models
POST /api/upload                          → upload .tif/.jpg/.png/.ecw + run base inference
GET  /api/results/{task_id}               → fetch detection results (GeoJSON)
POST /api/retrain                         → upload dataset .zip, trigger training
GET  /api/retrain/status/{job_id}         → poll training progress
POST /api/ensemble/{task_id}              → re-run inference with merged models
DELETE /api/models/{user_model_id}        → delete a user-trained model
POST /api/feedback                        → submit a correction (continuous learning)
GET  /api/export/{task_id}?format=...     → download shapefile/geojson/kml/csv/json

All polygon geometries returned are FULL GeoJSON Polygon rings in EPSG:4326,
not just centroids. The frontend renders them via L.polygon().
"""
from __future__ import annotations
import math
import requests
import io
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from sqlalchemy.orm import Session

from . import config, geo_utils, ml_pipeline, retrainer
from .config import settings, INFERENCE, MODELS_DIR, USER_MODELS_DIR, STORAGE_DIR, FEEDBACK_DIR, DATASETS_DIR, ENERGY
from .database import engine, SessionLocal, init_db, get_db
from .models import Upload, Rooftop, SolarPanel, UserModel, EnsembleJob, Feedback, Dataset, RetrainJob
from .schemas import (
    HealthResponse, ModelsListResponse, UploadResponse, DetectionResponse,
    RetrainRequest, RetrainStatusResponse, EnsembleRequest,
    FeedbackRequest, ExportResponse,
)
from .retrainer import (
    start_user_retrain, get_job_status, start_continuous_loop,
    start_merged_retrain, get_merged_retrain_status, count_images_and_labels,
)


# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description="GeoScan.AI — v7.6-ROADBLOCK + ensemble retraining backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    init_db()
    # Kick off continuous-learning watcher (no-op without feedback files)
    base_pt = str(MODELS_DIR / "best_roof.pt")
    start_continuous_loop(base_pt)


@app.get("/api/health", response_model=HealthResponse)
def health():
    base_models = []
    for f in MODELS_DIR.glob("*.pt"):
        base_models.append(f.name)
    return HealthResponse(
        status="ok",
        version="2.0.0",
        models_dir=str(MODELS_DIR),
        base_models_available=base_models,
    )


# ---------------------------------------------------------------------
# GET /api/models — list base + user-trained models
# ---------------------------------------------------------------------
@app.get("/api/models", response_model=ModelsListResponse)
def list_models(db: Session = Depends(get_db)):
    base = [{"id": f.stem, "name": f.name, "path": str(f),
             "type": "base"} for f in MODELS_DIR.glob("*.pt")]
    user_models = db.query(UserModel).order_by(UserModel.created_at.desc()).all()
    user = [{
        "id": um.id, "name": um.name, "base_model": um.base_model,
        "pt_path": um.pt_path, "epochs": um.epochs,
        "created_at": um.created_at.isoformat(), "type": "user",
        "metrics": um.metrics_json,
    } for um in user_models]
    return ModelsListResponse(base_models=base, user_models=user)


# ---------------------------------------------------------------------
# POST /api/upload — accept file + run base inference
# ---------------------------------------------------------------------
@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...),
                 models: str = Form('["base-v7.6"]'),
                 merge_strategy: str = Form("weighted"),
                 center_lat: float = Form(26.8467),  # default = Lucknow, UP, India
                 center_lon: float = Form(80.9462),
                 db: Session = Depends(get_db)):
    # --- validate extension ---
    name = (file.filename or "").lower()
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(415, f"Unsupported file format: {ext}. "
                                 f"Allowed: {settings.allowed_extensions}")

    # --- validate size ---
    contents = await file.read()
    if len(contents) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {settings.max_upload_mb} MB.")

    # --- save to /storage ---
    task_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + name
    save_path = STORAGE_DIR / task_id
    save_path.write_bytes(contents)
    upload_filename = file.filename or task_id

    # --- create upload row ---
    upload_row = Upload(
        id=task_id,
        filename=upload_filename,
        status="processing",
        crs="EPSG:4326",
    )
    db.add(upload_row)
    db.commit()

    # --- run inference (synchronous for simplicity) ---
    try:
        # Try rasterio first (handles all GeoTIFFs with any CRS)
        try:
            img, transform, crs = geo_utils.read_geotiff_rasterio(str(save_path))
            if transform is not None:
                print(f"[upload] GeoTIFF (rasterio): origin=({transform.origin_lon:.6f}, {transform.origin_lat:.6f})")
            else:
                print("[upload] rasterio read but no transform — using anchor")
        except Exception as e:
            print(f"[upload] rasterio failed ({e}), falling back to cv2")
            transform = geo_utils.parse_geotiff_transform(str(save_path))
            if transform is None:
                print(f"[upload] No GeoTIFF tags. Using anchor: lat={center_lat}, lon={center_lon}")
                img = cv2.imread(str(save_path))
                h, w = img.shape[:2] if img is not None else (1024, 1024)
                transform = geo_utils.synthetic_transform(center_lat, center_lon, w, h, dpi=96)
            else:
                img = cv2.imread(str(save_path))

        upload_row.scale_sqm = abs(transform.lon_per_px * transform.lat_per_px)
        upload_row.bounds_geojson = json.dumps({
            "type": "Polygon",
            "coordinates": [transform.to_geojson_ring(
                img.shape[1] if img is not None else 1024,
                img.shape[0] if img is not None else 1024
            )]
        })

        # Decide which model(s) to run
        model_list = json.loads(models) if models else ["base-v7.6"]
        base_roof_pt = str(MODELS_DIR / "best_roof.pt")
        base_solar_pt = str(MODELS_DIR / "best_solar.pt")
        user_pts = []
        for m_id in model_list:
            if m_id == "base-v7.6" or m_id == "best_roof":
                continue
            um = db.query(UserModel).filter(UserModel.id == m_id).first()
            if um:
                user_pts.append(um.pt_path)

        # Run BOTH roof and solar detection (like Colab)
        all_features = []
        total_roofs = 0
        total_panels = 0
        total_energy = 0

        # ROOF DETECTION
        if user_pts:
            roof_result = ml_pipeline.detect_ensemble(
                img_path=str(save_path), transform=transform,
                base_weights=base_roof_pt, user_weights=user_pts[0],
                strategy=merge_strategy, category="rooftop", image_bgr=img)
        else:
            roof_result = ml_pipeline.detect(
                img_path=str(save_path), transform=transform,
                weights_path=base_roof_pt, model_name="base-v7.6",
                category="rooftop", image_bgr=img)

        all_features.extend(roof_result.features)
        total_roofs = len(roof_result.features)
        total_energy = sum(f["properties"].get("energy_kwh_yr", 0) for f in roof_result.features)
        print(f"[upload] Roof detection complete: {total_roofs} rooftops found")

        # SOLAR PANEL DETECTION (always run if best_solar.pt exists)
        # ============================================================
        # FIXED: Added comprehensive logging to diagnose issues
        # ============================================================
        print(f"[upload] Solar model path: {base_solar_pt}")
        print(f"[upload] MODELS_DIR exists: {MODELS_DIR.exists()}")
        print(f"[upload] Files in MODELS_DIR: {[f.name for f in MODELS_DIR.glob('*.pt')]}")
        
        if os.path.isfile(base_solar_pt):
            print(f"[upload] ✓ best_solar.pt FOUND at {base_solar_pt} — running solar detection...")
            try:
                solar_result = ml_pipeline.detect(
                    img_path=str(save_path), transform=transform,
                    weights_path=base_solar_pt, model_name="base-v7.6-solar",
                    category="solar_panel", image_bgr=img)
                
                # Log detailed results
                print(f"[upload] Solar detection returned {len(solar_result.features)} panels")
                if solar_result.features:
                    for i, feat in enumerate(solar_result.features[:3]):  # Log first 3
                        conf = feat["properties"].get("confidence", 0)
                        area = feat["properties"].get("area_m2", 0)
                        print(f"[upload]   Panel {i+1}: confidence={conf:.3f}, area={area:.2f}m²")
                    if len(solar_result.features) > 3:
                        print(f"[upload]   ... and {len(solar_result.features) - 3} more panels")
                
                all_features.extend(solar_result.features)
                total_panels = len(solar_result.features)
                print(f"[upload] ✓ Solar detection complete: {total_panels} panels saved")
                
            except Exception as solar_err:
                import traceback
                print(f"[upload] ✗ Solar detection FAILED with error: {solar_err}")
                print(f"[upload] Full traceback:")
                traceback.print_exc()
                total_panels = 0
        else:
            print(f"[upload] ✗ best_solar.pt NOT FOUND at {base_solar_pt}")
            print(f"[upload]   Searched directory: {MODELS_DIR}")
            print(f"[upload]   Available .pt files: {[f.name for f in MODELS_DIR.glob('*.pt')]}")
            print(f"[upload]   Skipping solar panel detection (will only detect rooftops)")
            total_panels = 0

        # Persist rooftops and solar panels to DB
        for feat in all_features:
            ring = feat["geometry"]["coordinates"][0]
            lat, lon = geo_utils.ring_centroid(ring)
            props = feat["properties"]
            if props["type"] == "rooftop":
                db.add(Rooftop(
                    upload_id=task_id, category="res",
                    area_sqm=props.get("area_m2", 0), lat=lat, lon=lon,
                    geometry=json.dumps(feat["geometry"]),
                    confidence=props.get("confidence", 0),
                    model=props.get("model", "base-v7.6"),
                    usable_area_sqm=props.get("usable_area_sqm", 0),
                    panel_count=props.get("panel_count", 0),
                    energy_kwh_yr=props.get("energy_kwh_yr", 0),
                ))
            else:  # solar_panel
                db.add(SolarPanel(
                    upload_id=task_id, area_sqm=props.get("area_m2", 0),
                    lat=lat, lon=lon, geometry=json.dumps(feat["geometry"]),
                    confidence=props.get("confidence", 0),
                    model=props.get("model", "base-v7.6-solar"),
                ))

        upload_row.status = "done"
        db.commit()

        msg = f"Detected {total_roofs} rooftops + {total_panels} solar panels in {roof_result.elapsed_s:.1f}s"
        print(f"[upload] ✓ {msg}")
        return UploadResponse(
            task_id=task_id, filename=upload_filename, status="done",
            bounds_geojson=json.loads(upload_row.bounds_geojson) if upload_row.bounds_geojson else None,
            message=msg
        )
    except Exception as exc:
        import traceback
        upload_row.status = "error"
        upload_row.error_message = str(exc)
        db.commit()
        print(f"[upload] ✗ Detection failed: {exc}")
        traceback.print_exc()
        raise HTTPException(500, f"Detection failed: {exc}")


# ---------------------------------------------------------------------
# GET /api/results/{task_id} — return full GeoJSON of detections
# FIXED: Now includes BOTH rooftops AND solar panels
# ---------------------------------------------------------------------
@app.get("/api/results/{task_id}", response_model=DetectionResponse)
def get_results(task_id: str, db: Session = Depends(get_db)):
    up = db.query(Upload).filter(Upload.id == task_id).first()
    if not up:
        raise HTTPException(404, f"Task {task_id} not found")
    if up.status != "done":
        return DetectionResponse(
            task_id=task_id, status=up.status,
            message="Processing not finished",
        )

    # FIXED: Query BOTH Rooftop AND SolarPanel tables
    roofs = db.query(Rooftop).filter(Rooftop.upload_id == task_id).all()
    panels = db.query(SolarPanel).filter(SolarPanel.upload_id == task_id).all()
    
    print(f"[results] task_id={task_id}: {len(roofs)} rooftops, {len(panels)} solar panels in DB")
    
    features = []
    total_area = 0.0
    total_energy = 0.0
    total_panel_area = 0.0
    
    # Add rooftop features
    for r in roofs:
        geom = json.loads(r.geometry)
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "type": "rooftop",
                "area_m2": r.area_sqm,
                "confidence": r.confidence,
                "model": r.model,
                "usable_area_sqm": r.usable_area_sqm,
                "panel_count": r.panel_count,
                "energy_kwh_yr": r.energy_kwh_yr,
                "centroid": [r.lat, r.lon],
                "category": r.category,
            }
        })
        total_area += r.area_sqm
        total_energy += r.energy_kwh_yr
    
    # FIXED: Add solar panel features
    for p in panels:
        geom = json.loads(p.geometry)
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "type": "solar_panel",
                "area_m2": p.area_sqm,
                "confidence": p.confidence,
                "model": p.model,
                "centroid": [p.lat, p.lon],
                "category": "solar_panel",
            }
        })
        total_panel_area += p.area_sqm

    stats = {
        "feature_count": len(features),
        "total_area_m2": round(total_area, 2),
        "rooftops": len(roofs),
        "solar_panels": len(panels),  # FIXED: Added solar panel count
        "total_panel_area_m2": round(total_panel_area, 2),  # FIXED: Added panel area
        "total_energy_kwh_yr": round(total_energy, 0),
        "models": list(set(f["properties"]["model"] for f in features)),
        "crs": "EPSG:4326",
    }

    download_links = {
        "geojson": f"/api/export/{task_id}?format=geojson",
        "kml":     f"/api/export/{task_id}?format=kml",
        "csv":     f"/api/export/{task_id}?format=csv",
        "json":    f"/api/export/{task_id}?format=json",
        "shapefile": f"/api/export/{task_id}?format=shapefile",
    }

    bounds = json.loads(up.bounds_geojson) if up.bounds_geojson else None
    return DetectionResponse(
        task_id=task_id, status="done", crs="EPSG:4326",
        bounds_geojson=bounds, features=features, stats=stats,
        download_links=download_links,
    )


# ---------------------------------------------------------------------
# POST /api/retrain — upload dataset .zip, trigger training
# ---------------------------------------------------------------------
@app.post("/api/retrain")
async def retrain(background_tasks: BackgroundTasks,
                  base_model: str = Form("best_roof.pt"),
                  name: Optional[str] = Form(None),
                  user_id: str = Form("anon"),
                  dataset: UploadFile = File(...),
                  db: Session = Depends(get_db)):
    # save uploaded zip
    zip_path = STORAGE_DIR / f"dataset_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.zip"
    zip_path.write_bytes(await dataset.read())

    base_pt = str(MODELS_DIR / base_model)
    if not os.path.isfile(base_pt):
        raise HTTPException(404, f"Base model not found: {base_model}")

    job_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    start_user_retrain(job_id, base_pt, str(zip_path), user_id=user_id, name=name)

    return {"job_id": job_id, "status": "started",
            "poll_url": f"/api/retrain/status/{job_id}"}


# ---------------------------------------------------------------------
# GET /api/retrain/status/{job_id}
# ---------------------------------------------------------------------
@app.get("/api/retrain/status/{job_id}", response_model=RetrainStatusResponse)
def retrain_status(job_id: str):
    s = get_job_status(job_id)
    if not s:
        raise HTTPException(404, f"Job {job_id} not found")
    return RetrainStatusResponse(job_id=job_id, **s)


# ---------------------------------------------------------------------
# POST /api/ensemble/{task_id} — re-run inference with merged models
# ---------------------------------------------------------------------
@app.post("/api/ensemble/{task_id}", response_model=DetectionResponse)
def ensemble(task_id: str, req: EnsembleRequest,
             db: Session = Depends(get_db)):
    up = db.query(Upload).filter(Upload.id == task_id).first()
    if not up:
        raise HTTPException(404, f"Task {task_id} not found")

    img_path = str(STORAGE_DIR / task_id)
    if not os.path.isfile(img_path):
        raise HTTPException(404, "Original image not found on disk")

    # Reconstruct transform
    transform = geo_utils.parse_geotiff_transform(img_path)
    if transform is None:
        transform = geo_utils.synthetic_transform(26.8467, 80.9462, 1024, 1024)

    base_pt = str(MODELS_DIR / req.base_model)
    um = db.query(UserModel).filter(UserModel.id == req.user_model_id).first()
    if not um:
        raise HTTPException(404, f"User model {req.user_model_id} not found")

    job = EnsembleJob(
        upload_id=task_id,
        base_model_id=req.base_model,
        user_model_id=um.id,
        strategy=req.strategy,
        alpha=req.alpha,
        status="running",
    )
    db.add(job)
    db.commit()

    try:
        result = ml_pipeline.detect_ensemble(
            img_path=img_path,
            transform=transform,
            base_weights=base_pt,
            user_weights=um.pt_path,
            strategy=req.strategy,
            alpha=req.alpha,
            category=req.category,
        )
        job.status = "done"
        job.finished_at = datetime.utcnow()
        db.commit()

        # Update rooftops: wipe old, insert new
        db.query(Rooftop).filter(Rooftop.upload_id == task_id).delete()
        for feat in result.features:
            ring = feat["geometry"]["coordinates"][0]
            lat, lon = geo_utils.ring_centroid(ring)
            props = feat["properties"]
            db.add(Rooftop(
                upload_id=task_id, category="res",
                area_sqm=props.get("area_m2", 0), lat=lat, lon=lon,
                geometry=json.dumps(feat["geometry"]),
                confidence=props.get("confidence", 0),
                model=props.get("model", "ensemble"),
                usable_area_sqm=props.get("usable_area_sqm", 0),
                panel_count=props.get("panel_count", 0),
                energy_kwh_yr=props.get("energy_kwh_yr", 0),
            ))
        db.commit()
        return get_results(task_id, db)
    except Exception as exc:
        job.status = "error"
        db.commit()
        raise HTTPException(500, f"Ensemble failed: {exc}")


# ---------------------------------------------------------------------
# DELETE /api/models/{user_model_id}
# ---------------------------------------------------------------------
@app.delete("/api/models/{user_model_id}")
def delete_model(user_model_id: str, db: Session = Depends(get_db)):
    um = db.query(UserModel).filter(UserModel.id == user_model_id).first()
    if not um:
        raise HTTPException(404, "User model not found")
    try:
        if os.path.isfile(um.pt_path):
            os.remove(um.pt_path)
    except Exception:
        pass
    db.delete(um)
    db.commit()
    return {"deleted": user_model_id}


# ---------------------------------------------------------------------
# POST /api/feedback — submit a correction
# ---------------------------------------------------------------------
@app.post("/api/feedback")
async def feedback(upload_id: str = Form(...),
                   correction_type: str = Form("missed"),
                   note: str = Form(""),
                   image: Optional[UploadFile] = File(None),
                   label: Optional[UploadFile] = File(None),
                   db: Session = Depends(get_db)):
    """Submit a correction. Multipart form-data (because of optional image/label files).

    Fields:
      upload_id (str, required) — the task_id from POST /api/upload
      correction_type (str) — 'missed' | 'false_positive' | 'wrong_class'
      note (str) — free-text note
      image (file, optional) — corrected image
      label (file, optional) — corrected label
    """
    up = db.query(Upload).filter(Upload.id == upload_id).first()
    if not up:
        raise HTTPException(404, "Upload not found")

    img_path = lab_path = ""
    if image:
        img_path = str(FEEDBACK_DIR / f"{upload_id}_img_{image.filename}")
        Path(img_path).write_bytes(await image.read())
    if label:
        lab_path = str(FEEDBACK_DIR / f"{upload_id}_label_{label.filename}")
        Path(lab_path).write_bytes(await label.read())

    fb = Feedback(
        upload_id=upload_id,
        correction_type=correction_type,
        image_path=img_path,
        label_path=lab_path,
        note=note,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)

    # Also write a JSON summary the continuous-learning loop can scan
    summary = {
        "id": fb.id, "upload_id": upload_id,
        "correction_type": correction_type, "note": note,
        "image_path": img_path, "label_path": lab_path,
        "created_at": fb.created_at.isoformat(),
    }
    (FEEDBACK_DIR / f"{fb.id}.json").write_text(json.dumps(summary, indent=2))

    return {"feedback_id": fb.id, "status": "saved",
            "continuous_learning_pending": len(list(FEEDBACK_DIR.glob("*.json")))}


# ---------------------------------------------------------------------
# GET /api/export/{task_id}?format=...
# FIXED: Now includes BOTH rooftops AND solar panels in exports
# ---------------------------------------------------------------------
@app.get("/api/export/{task_id}")
def export(task_id: str, format: str = "geojson",
           db: Session = Depends(get_db)):
    up = db.query(Upload).filter(Upload.id == task_id).first()
    if not up:
        raise HTTPException(404, "Task not found")

    # FIXED: Query BOTH tables
    roofs = db.query(Rooftop).filter(Rooftop.upload_id == task_id).all()
    panels = db.query(SolarPanel).filter(SolarPanel.upload_id == task_id).all()
    
    features = []
    
    # Add rooftop features
    for r in roofs:
        features.append({
            "type": "Feature",
            "geometry": json.loads(r.geometry),
            "properties": {
                "type": "rooftop", "area_m2": r.area_sqm,
                "confidence": r.confidence, "model": r.model,
                "usable_area_sqm": r.usable_area_sqm,
                "panel_count": r.panel_count,
                "energy_kwh_yr": r.energy_kwh_yr,
                "centroid_lat": r.lat, "centroid_lon": r.lon,
            }
        })
    
    # FIXED: Add solar panel features
    for p in panels:
        features.append({
            "type": "Feature",
            "geometry": json.loads(p.geometry),
            "properties": {
                "type": "solar_panel", "area_m2": p.area_sqm,
                "confidence": p.confidence, "model": p.model,
                "centroid_lat": p.lat, "centroid_lon": p.lon,
            }
        })

    fmt = format.lower()
    if fmt == "geojson":
        fc = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
            "features": features,
        }
        blob = json.dumps(fc, indent=2).encode()
        return StreamingResponse(io.BytesIO(blob),
            media_type="application/geo+json",
            headers={"Content-Disposition": f"attachment; filename=detections_{task_id}.geojson"})

    if fmt == "kml":
        kml = geo_utils.features_to_kml(features)
        return StreamingResponse(io.BytesIO(kml.encode()),
            media_type="application/vnd.google-earth.kml+xml",
            headers={"Content-Disposition": f"attachment; filename=detections_{task_id}.kml"})

    if fmt == "csv":
        csv_text = geo_utils.features_to_csv(features)
        return StreamingResponse(io.BytesIO(csv_text.encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=report_{task_id}.csv"})

    if fmt == "json":
        report = {
            "task_id": task_id, "filename": up.filename,
            "generated_at": datetime.utcnow().isoformat(),
            "srs": "EPSG:4326",
            "stats": {
                "feature_count": len(features),
                "rooftops": len(roofs),
                "solar_panels": len(panels),  # FIXED
                "total_area_m2": sum(f["properties"]["area_m2"] for f in features),
                "total_energy_kwh_yr": sum(f["properties"].get("energy_kwh_yr", 0) for f in features),
            },
            "features": features,
        }
        blob = json.dumps(report, indent=2).encode()
        return StreamingResponse(io.BytesIO(blob),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=metadata_{task_id}.json"})

    if fmt == "shapefile":
        # Real shapefile generation requires GDAL/fiona. We return a zip
        # with the GeoJSON + a README explaining how to convert to .shp.
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("detections.geojson", json.dumps({
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
                "features": features,
            }, indent=2))
            zf.writestr("README.txt",
                "Backend produces this GeoJSON. To convert to ESRI Shapefile,\n"
                "use: ogr2ogr -f 'ESRI Shapefile' out.shp detections.geojson\n"
                "(requires GDAL). Coordinates are EPSG:4326 (WGS84).\n")
        zip_buf.seek(0)
        return StreamingResponse(zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=shapefile_{task_id}.zip"})

    raise HTTPException(400, f"Unknown format: {format}")


# ============================================================
# DATASET POOL — accumulate datasets for merge-and-retrain
# ============================================================
# This is the "LLM-style fine-tuning" flow:
#   1. User uploads datasets over time → they accumulate in a pool
#   2. User clicks "Merge & Retrain"
#   3. ALL datasets in the pool are merged into one YOLO dataset
#   4. The base model is RETRAINED on the merged data
#   5. The new .pt file REPLACES the old base model (hot-swap)
# ============================================================


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...),
                         name: str = Form(""),
                         user_id: str = Form("anon"),
                         notes: str = Form(""),
                         db: Session = Depends(get_db)):
    """Upload a dataset .zip to the user's pool.

    The .zip should contain images (.jpg/.png) and labels (.txt) in either:
      - train/ + val/ subfolders (standard YOLO layout), OR
      - images + labels at the root level

    The dataset is extracted to /datasets/{dataset_id}/ and registered
    in the database. It will be included in the next "Merge & Retrain" job.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(415, "Dataset must be a .zip file")

    contents = await file.read()
    if len(contents) > 500 * 1024 * 1024:  # 500 MB max per dataset
        raise HTTPException(413, "Dataset too large (max 500 MB)")

    dataset_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + (file.filename or "dataset.zip")[:20]
    ds_dir = DATASETS_DIR / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    zip_path = ds_dir / "source.zip"
    zip_path.write_bytes(contents)

    # Extract
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ds_dir)
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid .zip file")
    # Remove the source zip to save space
    zip_path.unlink(missing_ok=True)

    # Count images and labels
    img_count, lbl_count = count_images_and_labels(ds_dir)

    # Persist to database
    ds_row = Dataset(
        id=dataset_id,
        user_id=user_id,
        name=name or (file.filename or "Unnamed Dataset").replace(".zip", ""),
        original_filename=file.filename or "",
        path=str(ds_dir),
        image_count=img_count,
        label_count=lbl_count,
        notes=notes,
    )
    db.add(ds_row)
    db.commit()
    db.refresh(ds_row)

    return {
        "dataset_id": ds_row.id,
        "name": ds_row.name,
        "image_count": img_count,
        "label_count": lbl_count,
        "message": f"Dataset added to pool. {img_count} images, {lbl_count} labels.",
    }


@app.get("/api/datasets")
def list_datasets(user_id: str = "anon", db: Session = Depends(get_db)):
    """List all datasets in the user's pool."""
    datasets = db.query(Dataset).filter(
        Dataset.user_id == user_id
    ).order_by(Dataset.created_at.desc()).all()
    return {
        "datasets": [{
            "id": d.id,
            "name": d.name,
            "image_count": d.image_count,
            "label_count": d.label_count,
            "notes": d.notes,
            "created_at": d.created_at.isoformat(),
        } for d in datasets],
        "total": len(datasets),
        "total_images": sum(d.image_count for d in datasets),
    }


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Remove a dataset from the pool and delete its files."""
    import shutil
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    # Delete files
    try:
        if os.path.isdir(ds.path):
            shutil.rmtree(ds.path)
    except Exception as e:
        print(f"[delete_dataset] Could not remove {ds.path}: {e}")
    db.delete(ds)
    db.commit()
    return {"deleted": dataset_id}


@app.post("/api/retrain/merge")
def retrain_merge(base_model: str = Form("best_roof.pt"),
                  user_id: str = Form("anon"),
                  epochs: int = Form(30),
                  merge_first: bool = Form(False),
                  db: Session = Depends(get_db)):
    """Trigger merge-and-retrain: retrain the base model on ALL datasets in
    the pool, then hot-swap the .pt file.

    Args:
        merge_first: If True, physically merges all datasets into one folder
                     before training (slower, uses more disk, but creates a
                     unified train/val split). If False (default), trains
                     directly on the original dataset locations via a
                     multi-path data.yaml — no file copying, faster, less
                     disk usage. YOLOv8 supports list-valued train/val paths.

    This is the "LLM-style fine-tuning" flow:
      - Accumulates datasets (no data is lost)
      - (Optional) Merges them into one YOLO dataset
      - Retrains the base model (transfer learning, 30 epochs)
      - Hot-swaps the new .pt (old model backed up to .pt.bak)

    No ensemble — the retrained model directly replaces the old one.
    """
    base_pt = str(MODELS_DIR / base_model)
    if not os.path.isfile(base_pt):
        raise HTTPException(404, f"Base model not found: {base_model}")

    # Get all datasets in the user's pool
    datasets = db.query(Dataset).filter(Dataset.user_id == user_id).all()
    if not datasets:
        raise HTTPException(400, "No datasets in pool. Upload datasets first via /api/datasets/upload")

    dataset_paths = [d.path for d in datasets]
    dataset_ids = [d.id for d in datasets]
    total_images = sum(d.image_count for d in datasets)

    job_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + f"_{len(dataset_paths)}ds"

    # Persist job to database
    job_row = RetrainJob(
        id=job_id,
        user_id=user_id,
        base_model_path=base_pt,
        dataset_ids=json.dumps(dataset_ids),
        total_images=total_images,
        epochs=epochs,
        status="running",
        stage="queued",
    )
    db.add(job_row)
    db.commit()

    # Kick off background thread
    start_merged_retrain(job_id, base_pt, dataset_paths,
                         user_id=user_id, epochs=epochs,
                         merge_first=merge_first)

    mode_label = "merge-then-train" if merge_first else "multi-path (no merge)"
    return {
        "job_id": job_id,
        "status": "started",
        "mode": mode_label,
        "merge_first": merge_first,
        "total_datasets": len(dataset_paths),
        "total_images": total_images,
        "poll_url": f"/api/retrain/merge/status/{job_id}",
        "message": f"Retraining on {len(dataset_paths)} datasets ({total_images} images) "
                   f"using {mode_label} mode. Base model will be hot-swapped on completion.",
    }


@app.get("/api/retrain/merge/status/{job_id}")
def retrain_merge_status(job_id: str):
    """Poll the progress of a merge-and-retrain job."""
    s = get_merged_retrain_status(job_id)
    if not s:
        raise HTTPException(404, f"Job {job_id} not found")
    return s


# ---------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Helper: Fetch satellite tiles for a bounding box
# ---------------------------------------------------------------------
def fetch_map_tiles(west, south, east, north, zoom=18):
    """Downloads Esri satellite tiles for a bounding box and stitches them."""
    def lonlat_to_tile(lat, lon, z):
        n = 2.0 ** z
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        return xtile, ytile

    def tile_to_top_left_lonlat(xtile, ytile, z):
        n = 2.0 ** z
        lon = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat = math.degrees(lat_rad)
        return lat, lon

    x_min, y_max = lonlat_to_tile(north, west, zoom)
    x_max, y_min = lonlat_to_tile(south, east, zoom)

    # Limit to 5x5 tiles max to prevent memory crashes
    if (x_max - x_min) > 4 or (y_max - y_min) > 4:
        raise HTTPException(400, "Map area is too large. Please zoom in closer (zoom level 17+ recommended).")

    tile_size = 256
    stitched_width = (x_max - x_min + 1) * tile_size
    stitched_height = (y_max - y_min + 1) * tile_size
    stitched_img = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    img_array = np.asarray(bytearray(r.content), dtype=np.uint8)
                    tile_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    px = (x - x_min) * tile_size
                    py = (y - y_min) * tile_size
                    stitched_img[py:py+tile_size, px:px+tile_size] = tile_img
            except Exception as e:
                print(f"Failed to download tile {x},{y}: {e}")

    top_left_lat, top_left_lon = tile_to_top_left_lonlat(x_min, y_min, zoom)
    bottom_right_lat, bottom_right_lon = tile_to_top_left_lonlat(x_max + 1, y_max + 1, zoom)

    lon_per_px = (bottom_right_lon - top_left_lon) / stitched_width
    lat_per_px = (bottom_right_lat - top_left_lat) / stitched_height

    transform = geo_utils.GeoTransform(
        origin_lon=top_left_lon,
        origin_lat=top_left_lat,
        lon_per_px=lon_per_px,
        lat_per_px=lat_per_px,
        crs="EPSG:4326"
    )
    return stitched_img, transform

# ---------------------------------------------------------------------
# POST /api/detect_map — Run detection on current map view
# ---------------------------------------------------------------------
@app.post("/api/detect_map", response_model=DetectionResponse)
def detect_map(west: float = Form(...), south: float = Form(...), 
               east: float = Form(...), north: float = Form(...), 
               zoom: int = Form(18),
               db: Session = Depends(get_db)):
    
    print(f"[detect_map] Fetching imagery for bbox: W={west}, S={south}, E={east}, N={north} zoom={zoom}")
    
    try:
        img, transform = fetch_map_tiles(west, south, east, north, zoom)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch map tiles: {e}")

    # Create a fake upload record so we can store results
    task_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_mapview"
    upload_row = Upload(id=task_id, filename=f"MapView_{zoom}", status="processing", crs="EPSG:4326")
    db.add(upload_row)
    db.commit()

    try:
        base_roof_pt = str(MODELS_DIR / "best_roof.pt")
        base_solar_pt = str(MODELS_DIR / "best_solar.pt")

        all_features = []
        
        # 1. ROOF DETECTION
        roof_result = ml_pipeline.detect(
            img_path="", transform=transform,
            weights_path=base_roof_pt, model_name="base-v7.6",
            category="rooftop", image_bgr=img)
        all_features.extend(roof_result.features)

        # 2. SOLAR PANEL DETECTION
        if os.path.isfile(base_solar_pt):
            solar_result = ml_pipeline.detect(
                img_path="", transform=transform,
                weights_path=base_solar_pt, model_name="base-v7.6-solar",
                category="solar_panel", image_bgr=img)
            all_features.extend(solar_result.features)

        # Save to DB
        for feat in all_features:
            ring = feat["geometry"]["coordinates"][0]
            lat, lon = geo_utils.ring_centroid(ring)
            props = feat["properties"]
            if props["type"] == "rooftop":
                db.add(Rooftop(
                    upload_id=task_id, category="res", area_sqm=props.get("area_m2", 0),
                    lat=lat, lon=lon, geometry=json.dumps(feat["geometry"]),
                    confidence=props.get("confidence", 0), model=props.get("model", "base-v7.6"),
                    usable_area_sqm=props.get("usable_area_sqm", 0), panel_count=props.get("panel_count", 0),
                    energy_kwh_yr=props.get("energy_kwh_yr", 0),
                ))
            else:
                db.add(SolarPanel(
                    upload_id=task_id, area_sqm=props.get("area_m2", 0),
                    lat=lat, lon=lon, geometry=json.dumps(feat["geometry"]),
                    confidence=props.get("confidence", 0), model=props.get("model", "base-v7.6-solar"),
                ))

        upload_row.status = "done"
        db.commit()
        return get_results(task_id, db)

    except Exception as exc:
        upload_row.status = "error"
        upload_row.error_message = str(exc)
        db.commit()
        raise HTTPException(500, f"Map detection failed: {exc}")
@app.get("/")
def root():
    return {"app": settings.app_name, "version": "2.1.0",
            "docs": "/docs", "health": "/api/health"}   


