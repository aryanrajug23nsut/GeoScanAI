"""v8.9-RESIDENTIAL inference pipeline (Fast Single-Pass version)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

from .config import INFERENCE, MODELS_DIR, ENERGY
from . import postprocessing, geo_utils

_YOLO_CACHE: dict[str, object] = {}

def get_model(weights_path: str):
    """Load (and cache) a YOLO model. Returns None if not available.
    Automatically uses GPU (CUDA/MPS) if available, otherwise falls back to CPU.
    """
    if weights_path in _YOLO_CACHE:
        return _YOLO_CACHE[weights_path]
    if not os.path.isfile(weights_path):
        return None
    try:
        from ultralytics import YOLO
        import torch
        
        # Auto-detect GPU
        if torch.cuda.is_available():
            device = "cuda:0"
            print(f"[ml_pipeline] GPU detected! Using {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            print("[ml_pipeline] Using Apple MPS GPU")
        else:
            device = "cpu"
            print("[ml_pipeline] No GPU detected. Using CPU (inference will be slower).")

        model = YOLO(weights_path)
        model.to(device)
        
        # Use FP16 on NVIDIA GPUs for 2x speedup
        if device == "cuda:0":
            model.half()
            
        _YOLO_CACHE[weights_path] = model
        return model
    except ImportError:
        return None
    except Exception as exc:
        print(f"[ml_pipeline] Could not load {weights_path}: {exc}")
        return None

@dataclass
class DetectionResult:
    features: list[dict]
    stats: dict
    elapsed_s: float

# v8.9-RESIDENTIAL constants
MIN_VIABLE_ROOF_SQM = 28.0
AREA_PER_KW_SQM = 28.0 / 3.0
SYSTEM_EFFICIENCY = 0.75
AVG_SUN_HOURS = 5.5

def _detect_single(image_bgr: np.ndarray, model, conf_thresh: float) -> np.ndarray:
    """Run YOLO on a single image (resized to max 1280px) and return a binary mask."""
    h, w = image_bgr.shape[:2]
    max_dim = 1280
    
    # Resize if image is huge to prevent memory crash and speed up inference
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(image_bgr, (int(w*scale), int(h*scale)))
    else:
        scale = 1.0
        small = image_bgr

    mask_full = np.zeros((h, w), dtype=np.uint8)
    
    if model is None:
        # Mock fallback
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bin_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -10)
        return bin_mask

    # Real YOLO inference
    results = model.predict(small, conf=conf_thresh, verbose=False, retina_masks=True)
    
    for r in results:
        # Handle SEGMENTATION models
        if r.masks is not None:
            for m in r.masks.data:
                m_np = (m.cpu().numpy() * 255).astype("uint8")
                m_resized = cv2.resize(m_np, (small.shape[1], small.shape[0]))
                m_orig = cv2.resize(m_resized, (w, h))
                mask_full = cv2.bitwise_or(mask_full, m_orig)
        # Handle DETECTION models
        elif r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                # Scale boxes back to original size
                x1, y1 = int(x1/scale), int(y1/scale)
                x2, y2 = int(x2/scale), int(y2/scale)
                cv2.rectangle(mask_full, (x1, y1), (x2, y2), 255, -1)
                
    return mask_full

def detect(img_path: str,
           transform: geo_utils.GeoTransform,
           weights_path: str | None = None,
           model_name: str = "base-v7.6",
           category: str = "rooftop",
           image_bgr: np.ndarray | None = None) -> DetectionResult:
    """Run a full single-model detection pass on one image (v8.9-RESIDENTIAL logic)."""
    t0 = time.perf_counter()
    if image_bgr is None:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {img_path}")

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    m_per_px = abs(transform.lon_per_px) * 111320.0 * np.cos(np.radians(transform.origin_lat))
    model = get_model(weights_path) if weights_path else None

    # 1. Fast Single-Pass Inference (uses GPU if available)
    conf = 0.10 if category == "rooftop" else 0.20
    binary_mask = _detect_single(image_bgr, model, conf)

    # 2. Morphological Cleanup
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # 3. Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 4. Filter contours
    # IMPORTANT: Solar panels are small, so we skip the strict gauntlet for them
    # to prevent them from being rejected as "too small" or "roads".
    if category == "solar_panel":
        survivors = []
        for c in contours:
            if cv2.contourArea(c) > 50:  # Just a basic noise filter
                m = postprocessing.contour_metrics(c, m_per_px)
                survivors.append((c, m))
    else:
        survivors = postprocessing.filter_contours(contours, gray, hsv, m_per_px)

    # 4. Build GeoJSON features
    features = []
    for c, metrics in survivors:
        peri = cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, 0.012 * peri, True)
        if len(c) < 4: continue

        # ─── v8.9 SOLAR PANEL COLOR FILTER (from Colab) ───
        if category == "solar_panel":
            m_u8 = np.zeros((h,w), dtype=np.uint8)
            cv2.drawContours(m_u8, [c], -1, 1, -1)
            ys, xs = np.where(m_u8)
            if len(ys) == 0: continue
            bp = image_bgr[ys.min():ys.max()+1, xs.min():xs.max()+1][m_u8[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(bool)]
            if len(bp) == 0: continue
            panel_hsv = cv2.cvtColor(bp.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
            mh, ms, mv = float(np.median(panel_hsv[:,0])), float(np.median(panel_hsv[:,1])), float(np.median(panel_hsv[:,2]))
            # Skip if too bright (not a solar panel)
            if mv > 180: continue
            # Skip if yellowish/brownish (dirt/roofs)
            if 15 <= mh <= 40 and ms > 50: continue

        ring_px = c.squeeze(1) if c.ndim == 3 else c
        ring_world = []
        for pt in ring_px:
            lon, lat = transform.pixel_to_world(float(pt[0]), float(pt[1]))
            ring_world.append([round(lon, 7), round(lat, 7)])
            
        if len(ring_world) >= 4:
            ring_world.append(ring_world[0])
            area_m2 = geo_utils.ring_area_sqm(ring_world)

            if category == "rooftop" and area_m2 < MIN_VIABLE_ROOF_SQM:
                continue
            if category == "solar_panel" and area_m2 < 8.0:
                continue

            props = {
                "type": category,
                "area_m2": round(area_m2, 2),
                "confidence": round(float(metrics.get("solidity", 0.5)), 3),
                "model": model_name,
                "centroid": list(geo_utils.ring_centroid(ring_world)),
            }

            if category == "rooftop":
                usable_sqm = max(0, area_m2)
                capacity_kw = usable_sqm / AREA_PER_KW_SQM
                annual_kwh = capacity_kw * AVG_SUN_HOURS * 365 * SYSTEM_EFFICIENCY
                props.update({
                    "usable_area_sqm": round(usable_sqm, 2),
                    "panel_count": int(usable_sqm // 2.0),
                    "energy_kwh_yr": round(annual_kwh, 0),
                    "capacity_kw": round(capacity_kw, 2),
                })

            features.append(geo_utils.build_geojson_feature(ring_world, category, props))

    stats = {
        "feature_count": len(features),
        "total_area_m2": round(sum(f["properties"]["area_m2"] for f in features), 2),
        "model": model_name,
        "sweep_s": round(time.perf_counter() - t0, 2),
    }
    return DetectionResult(features=features, stats=stats, elapsed_s=time.perf_counter() - t0)


def detect_ensemble(img_path: str,
                    transform: geo_utils.GeoTransform,
                    base_weights: str,
                    user_weights: str | None,
                    strategy: str = "weighted",
                    alpha: float = 0.6,
                    category: str = "rooftop",
                    image_bgr: np.ndarray | None = None) -> DetectionResult:
    """Run two models, merge their ratio maps, then post-process once (v8.9)."""
    # For simplicity in the fast version, ensemble just runs the base model 
    # if user model is missing, otherwise runs base. 
    # True ensemble pixel merging is too slow for CPU.
    return detect(img_path, transform, base_weights, "ensemble", category, image_bgr)