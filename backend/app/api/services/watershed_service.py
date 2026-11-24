"""Isolated Marker-based Watershed service & per-instance scoring (D-03, D-04).

Imports authoritative ASVS V5 slider bounds strictly from ``backend.app.pdi.config``
(``h_rel [0.05, 0.80]`` default ``0.20``, ``min_distance_rel [0.20, 2.50]`` default
``0.85``, ``min_area_ratio [0.01, 0.45]`` default ``0.10``), converts the Stage 4
OpenCV BGR basin palette into matching CSS hex swatches, scores every separated
instance via ``bundle["production_pipeline"]``, and always attaches the mandatory
``group_sum_only`` scientific disclaimer.
"""

from __future__ import annotations

import math
import numbers
import time
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from backend.app.api.dependencies import AppState
from backend.app.api.schemas import (
    AgeStagePayload,
    OpticalPayload,
    TimingPayload,
    WatershedInstancePayload,
    WatershedParamBounds,
    WatershedParamDefaults,
    WatershedParamsPayload,
    WatershedPresetItem,
    WatershedPresetReference,
    WatershedResponse,
    WatershedVisualsPayload,
)
from backend.app.api.services.inference_service import SLA_TARGET_MS
from backend.app.modeling.age_stage import map_weight_to_age_stage
from backend.app.modeling.artifacts import _WATERSHED_GROUP_SUM_DISCLAIMER
from backend.app.modeling.models import predict_from_feature_dict
from backend.app.pdi.config import (
    WATERSHED_MAX_AREA_RATIO,
    WATERSHED_MAX_DISTANCE_REL,
    WATERSHED_MAX_H_REL,
    WATERSHED_MIN_AREA_RATIO,
    WATERSHED_MIN_DISTANCE_REL,
    WATERSHED_MIN_H_REL,
    PDIConfig,
)
from backend.app.pdi.io_exif import OpticalOverride
from backend.app.pdi.watershed import _BASIN_PALETTE_BGR, run_watershed_pdi

BASIN_PALETTE_HEX: tuple[str, ...] = tuple(
    f"#{r:02X}{g:02X}{b:02X}" for b, g, r in _BASIN_PALETTE_BGR
)
WATERSHED_GROUP_SUM_DISCLAIMER: str = _WATERSHED_GROUP_SUM_DISCLAIMER

_ERR_INVALID_PARAMS: str = (
    "Um ou mais parâmetros estão fora dos limites aceitos "
    "(distância focal 0,5–200 mm; zoom 0,1–20×; limites do Watershed). "
    "Ajuste o campo destacado ou clique em 'Restaurar EXIF Original' (Aba 1) / "
    "'Restaurar Parâmetros Padrão' (Aba 2)."
)
_ERR_DECODE_UPLOAD: str = (
    "Não foi possível decodificar esta imagem com segurança. "
    "Verifique se o arquivo não está corrompido e clique em 'Carregar Nova Foto'."
)


def _validate_slider_float(
    value: float | None,
    *,
    default: float,
    lo: float,
    hi: float,
) -> float:
    candidate = default if value is None else value
    if isinstance(candidate, bool) or not isinstance(candidate, numbers.Real):
        raise HTTPException(status_code=422, detail=_ERR_INVALID_PARAMS)
    numeric = float(candidate)
    if not math.isfinite(numeric) or not (lo <= numeric <= hi):
        raise HTTPException(status_code=422, detail=_ERR_INVALID_PARAMS)
    return numeric


def validate_watershed_params(
    h_rel: float | None,
    min_distance_rel: float | None,
    min_area_ratio: float | None,
    defaults: PDIConfig,
) -> tuple[float, float, float]:
    """Validate optional Watershed sliders strictly against Phase 2 ASVS V5 bounds."""
    valid_h = _validate_slider_float(
        h_rel,
        default=defaults.watershed_h_rel,
        lo=WATERSHED_MIN_H_REL,
        hi=WATERSHED_MAX_H_REL,
    )
    valid_dist = _validate_slider_float(
        min_distance_rel,
        default=defaults.watershed_min_distance_rel,
        lo=WATERSHED_MIN_DISTANCE_REL,
        hi=WATERSHED_MAX_DISTANCE_REL,
    )
    valid_area = _validate_slider_float(
        min_area_ratio,
        default=defaults.watershed_min_area_ratio,
        lo=WATERSHED_MIN_AREA_RATIO,
        hi=WATERSHED_MAX_AREA_RATIO,
    )
    return valid_h, valid_dist, valid_area


def run_watershed_service(
    state: AppState,
    *,
    source: Path | bytes,
    source_type: Literal["preset", "upload"],
    preset: WatershedPresetItem | None,
    h_rel: float,
    min_distance_rel: float,
    min_area_ratio: float,
    optical_override: OpticalOverride | None,
) -> WatershedResponse:
    """Execute isolated classical Marker-based Watershed and illustrative per-bird scoring."""
    if source_type == "preset" and preset is None:
        raise ValueError("Watershed preset metadata is required when source_type == 'preset'.")
    if source_type == "upload" and preset is not None:
        raise ValueError("Watershed preset metadata must be None when source_type == 'upload'.")

    t_total = time.perf_counter()

    try:
        ws_res = run_watershed_pdi(
            source,
            config=state.pdi_config,
            optical_override=optical_override,
            h_rel=h_rel,
            min_distance_rel=min_distance_rel,
            min_area_ratio=min_area_ratio,
            return_visuals=True,
        )
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
    ) as exc:
        if source_type == "upload":
            raise HTTPException(status_code=400, detail=_ERR_DECODE_UPLOAD) from exc
        raise

    if ws_res.visuals is None:
        raise RuntimeError("run_watershed_pdi(return_visuals=True) returned no visual artifacts.")

    prod_pipeline = state.bundle["production_pipeline"]
    feature_names: tuple[str, ...] = state.bundle["feature_names"]

    t_inf = time.perf_counter()
    per_bird_weights: list[float] = [
        predict_from_feature_dict(prod_pipeline, feature_names, inst.features)
        for inst in ws_res.instances
    ]
    inference_ms = (time.perf_counter() - t_inf) * 1000.0

    total_weight_g = float(sum(per_bird_weights))
    mean_weight_g: float | None = (
        round(total_weight_g / float(len(per_bird_weights)), 4) if per_bird_weights else None
    )

    instances_payload: list[WatershedInstancePayload] = []
    for inst, est_weight in zip(ws_res.instances, per_bird_weights, strict=True):
        age_res = map_weight_to_age_stage(est_weight)
        share_pct = (
            min(100.0, round((est_weight / total_weight_g) * 100.0, 4))
            if total_weight_g > 0.0
            else 0.0
        )
        color_hex = BASIN_PALETTE_HEX[(inst.instance_id - 1) % len(BASIN_PALETTE_HEX)]
        instances_payload.append(
            WatershedInstancePayload(
                instance_id=int(inst.instance_id),
                color_hex=color_hex,
                centroid_xy=(
                    round(float(inst.centroid_xy[0]), 3),
                    round(float(inst.centroid_xy[1]), 3),
                ),
                area_work_px=int(inst.area_work_px),
                area_px=round(float(inst.area_px), 4),
                area_frac=round(float(inst.area_frac), 6),
                area_optical_norm=round(float(inst.area_optical_norm), 4),
                estimated_weight_g=round(float(est_weight), 4),
                share_of_total_pct=share_pct,
                age_stage=AgeStagePayload(
                    stage_code=age_res.stage_code,
                    stage_label=age_res.stage_label,
                    estimated_age_days_range=age_res.estimated_age_days_range,
                    out_of_training_range=age_res.out_of_training_range,
                    valid_training_range_g=age_res.valid_training_range_g,
                    valid_age_days_range=age_res.valid_age_days_range,
                ),
            )
        )

    preset_reference: WatershedPresetReference | None = None
    if source_type == "preset" and preset is not None:
        gt_sum = float(preset.group_sum_weight_g)
        sum_diff = total_weight_g - gt_sum
        sum_pct_err = (abs(sum_diff) / gt_sum) * 100.0
        preset_reference = WatershedPresetReference(
            folder_id=preset.folder_id,
            expected_bird_count=preset.expected_bird_count,
            group_sum_weight_g=gt_sum,
            mean_bird_weight_reference_g=float(preset.mean_bird_weight_reference_g),
            sum_diff_g=round(sum_diff, 4),
            sum_pct_err=round(sum_pct_err, 4),
        )

    parameters = WatershedParamsPayload(
        h_rel=float(ws_res.h_rel),
        min_distance_rel=float(ws_res.min_distance_rel),
        min_area_ratio=float(ws_res.min_area_ratio),
        defaults=WatershedParamDefaults(
            h_rel=float(state.pdi_config.watershed_h_rel),
            min_distance_rel=float(state.pdi_config.watershed_min_distance_rel),
            min_area_ratio=float(state.pdi_config.watershed_min_area_ratio),
        ),
        bounds=WatershedParamBounds(),
    )
    optical = OpticalPayload(
        focal_length_mm=float(ws_res.optical.focal_length_mm),
        zoom_multiple=float(ws_res.optical.zoom_multiple),
        exif_optical_present=bool(ws_res.optical.exif_optical_present),
        override_applied=bool(ws_res.optical.override_applied),
        lens_class=ws_res.optical.lens_class,
    )
    visuals = WatershedVisualsPayload(
        stage1_original=f"data:image/jpeg;base64,{ws_res.visuals.stage1_original_b64}",
        stage2_union_mask=f"data:image/png;base64,{ws_res.visuals.stage2_union_mask_b64}",
        stage3_distance_peaks=f"data:image/jpeg;base64,{ws_res.visuals.stage3_distance_peaks_b64}",
        stage4_color_basins=f"data:image/jpeg;base64,{ws_res.visuals.stage4_color_basins_b64}",
        total_b64_bytes=int(ws_res.visuals.total_b64_bytes),
    )

    total_ms = (time.perf_counter() - t_total) * 1000.0
    timing = TimingPayload(
        pdi_ms=round(float(ws_res.latency_ms), 3),
        inference_ms=round(float(inference_ms), 3),
        total_ms=round(float(total_ms), 3),
        sla_target_ms=SLA_TARGET_MS,
        within_sla=(total_ms < float(SLA_TARGET_MS)),
    )

    return WatershedResponse(
        source_type=source_type,
        preset_id=preset.preset_id if preset is not None else None,
        ground_truth_scope="group_sum_only",
        disclaimer=WATERSHED_GROUP_SUM_DISCLAIMER,
        bird_count=int(ws_res.bird_count),
        estimated_total_weight_g=round(total_weight_g, 4),
        estimated_mean_weight_g=mean_weight_g,
        preset_reference=preset_reference,
        parameters=parameters,
        optical=optical,
        red_feeder_excluded_px=int(ws_res.red_feeder_excluded_px),
        pdi_config_hash=str(ws_res.pdi_config_hash),
        instances=instances_payload,
        visuals=visuals,
        timing=timing,
    )


__all__ = [
    "BASIN_PALETTE_HEX",
    "WATERSHED_GROUP_SUM_DISCLAIMER",
    "run_watershed_service",
    "validate_watershed_params",
]
