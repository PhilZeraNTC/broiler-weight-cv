"""FastAPI isolated Marker-based Watershed demo router (``POST /api/watershed-demo``)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.api.dependencies import AppState, get_app_state
from backend.app.api.schemas import WatershedResponse
from backend.app.api.services.inference_service import build_optical_override
from backend.app.api.services.upload_guard import read_and_validate_upload
from backend.app.api.services.watershed_service import (
    run_watershed_service,
    validate_watershed_params,
)

router = APIRouter(prefix="/api", tags=["watershed"])
AppStateDep = Annotated[AppState, Depends(get_app_state)]


@router.post("/watershed-demo", response_model=WatershedResponse)
async def post_watershed_demo(
    state: AppStateDep,
    file: Annotated[UploadFile | None, File()] = None,
    preset_id: Annotated[str | None, Form()] = None,
    h_rel: Annotated[float | None, Form()] = None,
    min_distance_rel: Annotated[float | None, Form()] = None,
    min_area_ratio: Annotated[float | None, Form()] = None,
    focal_length_mm: Annotated[float | None, Form()] = None,
    zoom_multiple: Annotated[float | None, Form()] = None,
) -> WatershedResponse:
    """Run classical Marker-based Watershed separation & per-bird illustrative scoring."""
    if preset_id is not None and not preset_id.strip():
        raise HTTPException(
            status_code=422,
            detail="Informe um 'preset_id' válido ou envie apenas um arquivo de imagem.",
        )
    clean_preset_id = preset_id.strip() if preset_id is not None else None
    if (file is None) == (clean_preset_id is None):
        raise HTTPException(
            status_code=422,
            detail="Envie exatamente uma origem por requisição: 'file' ou 'preset_id'.",
        )

    h_val, min_dist_val, min_area_val = validate_watershed_params(
        h_rel,
        min_distance_rel,
        min_area_ratio,
        state.pdi_config,
    )
    optical_override = build_optical_override(focal_length_mm, zoom_multiple)

    if clean_preset_id is not None:
        preset = state.preset_store.watershed_by_id.get(clean_preset_id)
        preset_path = state.preset_store.watershed_paths.get(clean_preset_id)
        if preset is None or preset_path is None:
            raise HTTPException(
                status_code=404,
                detail="Preset Watershed não encontrado no catálogo auditado.",
            )
        return await run_in_threadpool(
            run_watershed_service,
            state,
            source=preset_path,
            source_type="preset",
            preset=preset,
            h_rel=h_val,
            min_distance_rel=min_dist_val,
            min_area_ratio=min_area_val,
            optical_override=optical_override,
        )

    assert file is not None
    raw_bytes = await read_and_validate_upload(file)
    return await run_in_threadpool(
        run_watershed_service,
        state,
        source=raw_bytes,
        source_type="upload",
        preset=None,
        h_rel=h_val,
        min_distance_rel=min_dist_val,
        min_area_ratio=min_area_val,
        optical_override=optical_override,
    )


__all__ = ["router"]
