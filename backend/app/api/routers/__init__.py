"""FastAPI router package barrel exporting all 5 local REST routers."""

from __future__ import annotations

from backend.app.api.routers.health import router as health_router
from backend.app.api.routers.metrics import router as metrics_router
from backend.app.api.routers.predict import router as predict_router
from backend.app.api.routers.presets import router as presets_router
from backend.app.api.routers.watershed import router as watershed_router

__all__ = [
    "health_router",
    "metrics_router",
    "predict_router",
    "presets_router",
    "watershed_router",
]
