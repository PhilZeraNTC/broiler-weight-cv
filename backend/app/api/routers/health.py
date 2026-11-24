"""FastAPI health endpoint router (``GET /api/health``)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import AppState, get_app_state
from backend.app.api.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])
AppStateDep = Annotated[AppState, Depends(get_app_state)]


@router.get("/health", response_model=HealthResponse)
def get_health(state: AppStateDep) -> HealthResponse:
    """Return runtime model metadata, locked PDI hash, and preset catalog counts."""
    return HealthResponse(
        status="ok",
        model_version=str(state.bundle["model_version"]),
        winning_model_name=str(state.bundle["winning_model_name"]),
        winning_ablation_condition=str(state.bundle["winning_ablation_condition"]),
        pdi_config_hash=state.pdi_config_hash,
        n_features=len(state.bundle["feature_names"]),
        single_bird_presets_count=len(state.preset_store.single_bird_by_id),
        watershed_presets_count=len(state.preset_store.watershed_by_id),
        classical_pdi_only=True,
    )


__all__ = ["router"]
