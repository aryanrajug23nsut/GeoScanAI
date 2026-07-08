"""Pydantic schemas — request/response models for the API."""
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    models_dir: str
    base_models_available: list[str]


class ModelsListResponse(BaseModel):
    base_models: list[dict]
    user_models: list[dict]


# ---------------------------------------------------------------------
# Upload / detection
# ---------------------------------------------------------------------
class DetectionStats(BaseModel):
    feature_count: int = 0
    total_area_m2: float = 0.0
    rooftops: int = 0
    solar_panels: int = 0
    total_energy_kwh_yr: float = 0.0
    models: list[str] = Field(default_factory=list)
    strategy: Optional[str] = None
    sweep_s: float = 0.0


class UploadResponse(BaseModel):
    task_id: str
    filename: str
    status: str
    bounds_geojson: Optional[dict] = None
    message: str = ""


class DetectionResponse(BaseModel):
    task_id: str
    status: str
    crs: str = "EPSG:4326"
    bounds_geojson: Optional[dict] = None
    features: list[dict] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
    elapsed_s: float = 0.0
    download_links: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------
class RetrainRequest(BaseModel):
    base_model: str = "best_roof.pt"
    name: Optional[str] = None
    user_id: str = "anon"


class RetrainStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    stage: str
    model_id: Optional[str] = None
    pt_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None


# ---------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------
class EnsembleRequest(BaseModel):
    base_model: str = "best_roof.pt"
    user_model_id: str
    strategy: str = "weighted"   # weighted | union | intersection
    alpha: float = 0.6
    category: str = "rooftop"


# ---------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    upload_id: str
    correction_type: str = "missed"  # missed | false_positive | wrong_class
    note: str = ""


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
class ExportResponse(BaseModel):
    format: str
    download_url: str
    size_bytes: int
