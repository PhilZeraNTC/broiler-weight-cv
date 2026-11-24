"""FastAPI single-bird classical PDI inference router (``POST /api/predict``)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.api.dependencies import AppState, get_app_state
from backend.app.api.schemas import PredictResponse
from backend.app.api.services.inference_service import (
    build_optical_override,
    run_predict_service,
)
from backend.app.api.services.upload_guard import read_and_validate_upload

router = APIRouter(prefix="/api", tags=["predict"])
AppStateDep = Annotated[AppState, Depends(get_app_state)]


@router.post("/predict", response_model=PredictResponse)
async def post_predict(
    state: AppStateDep,
    file: Annotated[UploadFile | None, File()] = None,
    preset_id: Annotated[str | None, Form()] = None,
    focal_length_mm: Annotated[float | None, Form()] = None,
    zoom_multiple: Annotated[float | None, Form()] = None,
) -> PredictResponse:
    """Run EXIF-calibrated classical PDI X-Ray & weight regression on upload XOR preset."""
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

    optical_override = build_optical_override(focal_length_mm, zoom_multiple)

    if clean_preset_id is not None:
        preset = state.preset_store.single_bird_by_id.get(clean_preset_id)
        preset_path = state.preset_store.single_bird_paths.get(clean_preset_id)
        if preset is None or preset_path is None:
            raise HTTPException(
                status_code=404,
                detail="Preset de ave única não encontrado no catálogo auditado.",
            )
        return await run_in_threadpool(
            run_predict_service,
            state,
            source=preset_path,
            source_type="preset",
            preset=preset,
            optical_override=optical_override,
        )

    assert file is not None
    raw_bytes = await read_and_validate_upload(file)
    return await run_in_threadpool(
        run_predict_service,
        state,
        source=raw_bytes,
        source_type="upload",
        preset=None,
        optical_override=optical_override,
    )


__all__ = ["router"]
