"""SQLAlchemy ORM models — the 6 tables from the master plan §10.

All geometry columns are stored as GeoJSON strings in EPSG:4326.
PostGIS could be used for spatial queries, but for portability we
keep plain TEXT and parse on read. A migration to PostGIS geometry
columns is straightforward later (see README).
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON,
)
from sqlalchemy.orm import relationship

from .database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


class Upload(Base):
    __tablename__ = "uploads"

    id            = Column(String, primary_key=True, default=_uuid)
    filename      = Column(String, nullable=False)
    status        = Column(String, default="queued")   # queued|processing|done|error
    created_at    = Column(DateTime, default=datetime.utcnow)
    scale_sqm     = Column(Float, default=0.0)         # m² per pixel
    crs           = Column(String, default="EPSG:4326")
    bounds_geojson= Column(Text,   default="")         # image footprint as GeoJSON polygon
    error_message = Column(Text,   default="")

    rooftops  = relationship("Rooftop",  back_populates="upload", cascade="all, delete-orphan")
    panels    = relationship("SolarPanel", back_populates="upload", cascade="all, delete-orphan")
    feedback  = relationship("Feedback",  back_populates="upload", cascade="all, delete-orphan")


class Rooftop(Base):
    __tablename__ = "rooftops"

    id          = Column(String, primary_key=True, default=_uuid)
    upload_id   = Column(String, ForeignKey("uploads.id"), nullable=False, index=True)
    category    = Column(String, default="res")           # res | com
    area_sqm    = Column(Float, default=0.0)
    lat         = Column(Float, default=0.0)              # centroid lat (EPSG:4326)
    lon         = Column(Float, default=0.0)              # centroid lon
    geometry    = Column(Text, nullable=False)            # full polygon as GeoJSON
    confidence  = Column(Float, default=0.0)
    model       = Column(String, default="base-v7.6")
    usable_area_sqm = Column(Float, default=0.0)
    panel_count = Column(Integer, default=0)
    energy_kwh_yr = Column(Float, default=0.0)
    created_at  = Column(DateTime, default=datetime.utcnow)

    upload = relationship("Upload", back_populates="rooftops")


class SolarPanel(Base):
    __tablename__ = "solar_panels"

    id          = Column(String, primary_key=True, default=_uuid)
    upload_id   = Column(String, ForeignKey("uploads.id"), nullable=False, index=True)
    area_sqm    = Column(Float, default=0.0)
    lat         = Column(Float, default=0.0)
    lon         = Column(Float, default=0.0)
    geometry    = Column(Text, nullable=False)            # full polygon as GeoJSON
    confidence  = Column(Float, default=0.0)
    model       = Column(String, default="base-v7.6")
    created_at  = Column(DateTime, default=datetime.utcnow)

    upload = relationship("Upload", back_populates="panels")


class UserModel(Base):
    """A user-trained .pt file (transfer-learned from a base model)."""
    __tablename__ = "user_models"

    id            = Column(String, primary_key=True, default=_uuid)
    user_id       = Column(String, default="anon")
    name          = Column(String, nullable=False)
    base_model    = Column(String, default="best_roof.pt")
    pt_path       = Column(String, nullable=False)        # absolute path on disk
    dataset_path  = Column(String, default="")
    epochs        = Column(Integer, default=30)
    metrics_json  = Column(JSON, default=dict)
    created_at    = Column(DateTime, default=datetime.utcnow)


class EnsembleJob(Base):
    """Records a request to re-run inference with multiple models merged."""
    __tablename__ = "ensemble_jobs"

    id              = Column(String, primary_key=True, default=_uuid)
    upload_id       = Column(String, ForeignKey("uploads.id"), nullable=False, index=True)
    base_model_id   = Column(String, default="base-v7.6")
    user_model_id   = Column(String, ForeignKey("user_models.id"), nullable=True)
    strategy        = Column(String, default="weighted")  # weighted|union|intersection
    alpha           = Column(Float, default=0.6)
    status          = Column(String, default="queued")    # queued|running|done|error
    result_path     = Column(String, default="")          # path to merged results JSON
    created_at      = Column(DateTime, default=datetime.utcnow)
    finished_at     = Column(DateTime, nullable=True)


class Dataset(Base):
    """An uploaded training dataset in the user's pool.

    Datasets accumulate over time. When the user clicks "Merge & Retrain",
    ALL datasets in their pool are merged into a single YOLO dataset
    and the base model is retrained on the combined data. The new .pt
    file hot-swaps the old one (no ensemble — direct replacement).
    """
    __tablename__ = "datasets"

    id              = Column(String, primary_key=True, default=_uuid)
    user_id         = Column(String, default="anon", index=True)
    name            = Column(String, nullable=False)
    original_filename = Column(String, default="")
    path            = Column(String, nullable=False)       # path to extracted dataset dir
    image_count     = Column(Integer, default=0)
    label_count     = Column(Integer, default=0)
    notes           = Column(Text, default="")
    created_at      = Column(DateTime, default=datetime.utcnow)


class RetrainJob(Base):
    """Tracks a merge-and-retrain job (accumulated dataset → retrained model).

    This is DIFFERENT from EnsembleJob. Here:
      - Multiple datasets are MERGED into one
      - The base model is RETRAINED on the merged data
      - The new .pt file REPLACES the old base model (hot-swap)
      - Future detections use the retrained model directly (no ensemble)
    """
    __tablename__ = "retrain_jobs"

    id              = Column(String, primary_key=True, default=_uuid)
    user_id         = Column(String, default="anon", index=True)
    base_model_path = Column(String, nullable=False)        # starting .pt
    new_model_path  = Column(String, default="")            # resulting .pt (hot-swapped)
    dataset_ids     = Column(Text, default="[]")            # JSON list of Dataset.id
    total_images    = Column(Integer, default=0)
    epochs          = Column(Integer, default=30)
    status          = Column(String, default="queued")      # queued|running|done|error
    progress        = Column(Integer, default=0)
    stage           = Column(String, default="queued")
    error           = Column(Text, default="")
    created_at      = Column(DateTime, default=datetime.utcnow)
    finished_at     = Column(DateTime, nullable=True)


class Feedback(Base):
    """User-flagged corrections — feeds the continuous-learning loop."""
    __tablename__ = "feedback"

    id              = Column(String, primary_key=True, default=_uuid)
    upload_id       = Column(String, ForeignKey("uploads.id"), nullable=False, index=True)
    correction_type = Column(String, default="missed")    # missed | false_positive | wrong_class
    image_path      = Column(String, default="")
    label_path      = Column(String, default="")
    note            = Column(Text, default="")
    created_at      = Column(DateTime, default=datetime.utcnow)

    upload = relationship("Upload", back_populates="feedback")
