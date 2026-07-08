"""Centralized configuration for the GeoScan.AI backend.

All tunable thresholds for the v7.6-ROADBLOCK pipeline, ensemble strategies,
and storage paths live here so they can be tweaked without touching logic.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------
# Filesystem layout (mounted as Docker volumes in production)
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent       # .../backend/
STORAGE_DIR   = BASE_DIR / "storage"                    # uploaded .tif + outputs
MODELS_DIR    = BASE_DIR / "models"                     # base + user .pt files
USER_MODELS_DIR = MODELS_DIR / "user_models"
DATASETS_DIR  = BASE_DIR / "datasets"                   # user-uploaded training data
FEEDBACK_DIR  = BASE_DIR / "feedback_data"              # continuous-learning corrections

for d in (STORAGE_DIR, MODELS_DIR, USER_MODELS_DIR, DATASETS_DIR, FEEDBACK_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# v7.6-ROADBLOCK inference parameters
# ---------------------------------------------------------------
INFERENCE = {
    # Multi-scale sliding window
    "tile_sizes":   [128, 256, 512, 640],
    "strides":      {128: 96, 256: 192, 512: 384, 640: 480},
    "batch_small":  24,           # batch size for 128/256 tiles
    "batch_large":  12,           # batch size for 512/640 tiles
    "conf_thresh":  0.10,         # YOLO confidence threshold (low → catch everything)
    "vote_thresh":  0.15,         # ratio above which a pixel is "roof"

    # Hardware
    "half":         True,         # FP16 inference
    "device":       "cuda:0",     # set to "cpu" if no GPU

    # Vegetation skip
    "veg_green_ratio": 0.90,      # skip tile if >90% green pixels

    # Large-blob integrity check (>50 m²)
    "min_blob_area_m2": 50.0,
    "solidity_min":   0.60,
    "extent_min":     0.60,
    "compactness_max": 35.0,

    # Road Blocker
    "aspect_ratio_max": 3.2,
    "rect_fill_min":    0.55,
    "min_width_m":      2.5,

    # Shadow rejection
    "shadow_intensity_max": 65,

    # Road texture rejection (areas > 80 m²)
    "road_area_thresh_m2": 80.0,
    "road_std_max":        12,

    # Boundary straightening
    "approx_dp_ratio": 0.012,     # 1.2% of perimeter

    # Internal structure exclusion
    "interior_edge_ratio_max": 0.15,
}


# ---------------------------------------------------------------
# Ensemble merge defaults
# ---------------------------------------------------------------
ENSEMBLE = {
    "default_strategy": "weighted",   # weighted | union | intersection
    "default_alpha":    0.6,          # weight on base model (1-α on user model)
    "vote_thresh":      0.15,
}


# ---------------------------------------------------------------
# Retraining defaults (transfer learning from base .pt)
# ---------------------------------------------------------------
RETRAIN = {
    "epochs":   30,
    "imgsz":    640,
    "batch":    8,
    "lr0":      0.001,
    "freeze":   10,            # freeze first 10 backbone layers
    "amp":      True,
    "feedback_trigger": 50,    # # of feedback items that triggers auto-retrain
}


# ---------------------------------------------------------------
# Energy yield (for rooftop → solar potential estimate)
# ---------------------------------------------------------------
ENERGY = {
    "yield_per_sqm_kwh_yr": 280,   # rough UP/India average
    "usable_roof_ratio":    0.60,  # 60% of roof is usable for panels
    "panel_area_m2":        2.0,   # 1 panel ≈ 2 m²
    "panel_efficiency":     0.18,
}


# ---------------------------------------------------------------
# FastAPI settings
# ---------------------------------------------------------------
class Settings(BaseSettings):
    app_name: str = "GeoScan.AI Backend"
    api_v1_prefix: str = "/api"
    max_upload_mb: int = 100
    min_dpi: int = 96
    allowed_extensions: list[str] = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".ecw"]

    # Database (overridden by env in docker-compose)
    database_url: str = "postgresql+psycopg2://geoscan:geoscan@db:5432/geoscan"

    # CORS (allow the frontend container)
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_prefix = "GEOSCAN_"


settings = Settings()
