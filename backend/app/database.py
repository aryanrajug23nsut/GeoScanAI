"""SQLAlchemy database setup.

A single shared engine + sessionmaker. Tables are created on startup
via `init_db()` so the app is usable without running Alembic migrations
in development.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from .config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a scoped session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Idempotent — safe to call on every startup."""
    from . import models  # noqa: F401 — ensures models are registered with Base
    Base.metadata.create_all(bind=engine)
