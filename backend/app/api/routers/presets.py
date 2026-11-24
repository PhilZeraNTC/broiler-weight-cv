"""FastAPI presets & cached thumbnail router (``GET /api/presets``, ``.../thumbnail``)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from backend.app.api.dependencies import AppState, get_app_state
from backend.app.api.schemas import PresetsResponse
from backend.app.api.services.preset_service import THUMBNAIL_CACHE_CONTROL

router = APIRouter(prefix="/api", tags=["presets"])
AppStateDep = Annotated[AppState, Depends(get_app_state)]


@router.get("/presets", response_model=PresetsResponse)
def get_presets(state: AppStateDep) -> PresetsResponse:
    """Return curated single-bird and Watershed presets enriched with ``thumbnail_url``."""
    return state.preset_store.catalog


@router.get("/presets/{preset_id}/thumbnail")
async def get_preset_thumbnail(preset_id: str, state: AppStateDep) -> Response:
    """Serve a pre-warmed 320 px EXIF-transposed JPEG thumbnail strictly by allowlisted ID."""
    jpeg_bytes = state.preset_store.thumbnails.get(preset_id)
    if jpeg_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="Preset não encontrado no catálogo auditado.",
        )
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": THUMBNAIL_CACHE_CONTROL},
    )


__all__ = ["router"]
