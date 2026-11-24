"""FastAPI cached scientific audit & metrics router (``GET /api/metrics``)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.app.api.dependencies import AppState, get_app_state
from backend.app.api.schemas import MetricsResponse
from backend.app.api.services.metrics_service import METRICS_CACHE_CONTROL

router = APIRouter(prefix="/api", tags=["metrics"])
AppStateDep = Annotated[AppState, Depends(get_app_state)]


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(state: AppStateDep) -> JSONResponse:
    """Serve startup-cached read-only evaluation, curation, and PDI calibration audit JSONs."""
    return JSONResponse(
        content=state.metrics_payload,
        headers={"Cache-Control": METRICS_CACHE_CONTROL},
    )


__all__ = ["router"]
