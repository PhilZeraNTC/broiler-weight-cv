"""Single-bird classical PDI + honest OOF vs. production scoring service (D-02, D-03).

Routes curated single-bird presets strictly through ``bundle["fold_pipelines"][fold]``
(``oof_evaluation=True``) and user uploads through ``bundle["production_pipeline"]``
(``oof_evaluation=False``), using the single-source Phase 2 ``run_single_bird_pdi``
engine and Phase 3 ``predict_from_feature_dict`` / ``map_weight_to_age_stage`` helpers.
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
    BandErrorInterval,
    OofAuditPayload,
    OpticalPayload,
    PredictResponse,
    SegmentationAuditPayload,
    SingleBirdPresetItem,
    TimingPayload,
    WeightBand,
    XRayPayload,
)
from backend.app.ingestion.decoding import (
    BAND_LIGHT_MAX_EXCLUSIVE_G,
    BAND_MEDIUM_MAX_INCLUSIVE_G,
)
from backend.app.modeling.age_stage import map_weight_to_age_stage
from backend.app.modeling.models import predict_from_feature_dict
from backend.app.pdi.io_exif import (
    MAX_VALID_FOCAL_LENGTH_MM,
    MAX_VALID_ZOOM_MULTIPLE,
    MIN_VALID_FOCAL_LENGTH_MM,
    MIN_VALID_ZOOM_MULTIPLE,
    OpticalOverride,
)
from backend.app.pdi.pipeline import run_single_bird_pdi

SLA_TARGET_MS: int = 500

FEATURES_SUMMARY_KEYS: tuple[str, ...] = (
    "area_optical_norm",
    "area_px",
    "area_frac",
    "major_axis_optical_norm",
    "minor_axis_optical_norm",
    "solidity",
    "eccentricity",
    "circularity",
    "yellow_down_ratio",
    "white_plumage_ratio",
    "pink_skin_ratio",
    "feather_grad_mean",
    "lab_a_mean",
    "lab_b_mean",
)

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


def classify_weight_band(weight_g: float) -> WeightBand:
    """Classify a positive finite live weight (g) into the 3 coarse PRD weight bands."""
    if isinstance(weight_g, bool) or not isinstance(weight_g, numbers.Real):
        raise ValueError(f"weight_g must be a finite positive number, got {weight_g!r}")
    numeric = float(weight_g)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"weight_g must be finite and > 0, got {weight_g!r}")
    if numeric < float(BAND_LIGHT_MAX_EXCLUSIVE_G):
        return "<500g"
    if numeric <= float(BAND_MEDIUM_MAX_INCLUSIVE_G):
        return "500-1000g"
    return ">1000g"


def _validate_optional_optical_float(
    value: float | None,
    *,
    lo: float,
    hi: float,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise HTTPException(status_code=422, detail=_ERR_INVALID_PARAMS)
    numeric = float(value)
    if not math.isfinite(numeric) or not (lo <= numeric <= hi):
        raise HTTPException(status_code=422, detail=_ERR_INVALID_PARAMS)
    return numeric


def build_optical_override(
    focal_length_mm: float | None,
    zoom_multiple: float | None,
) -> OpticalOverride | None:
    """Validate optional manual optical override floats (ASVS V5) and construct override."""
    valid_focal = _validate_optional_optical_float(
        focal_length_mm,
        lo=MIN_VALID_FOCAL_LENGTH_MM,
        hi=MAX_VALID_FOCAL_LENGTH_MM,
    )
    valid_zoom = _validate_optional_optical_float(
        zoom_multiple,
        lo=MIN_VALID_ZOOM_MULTIPLE,
        hi=MAX_VALID_ZOOM_MULTIPLE,
    )
    if valid_focal is None and valid_zoom is None:
        return None
    return OpticalOverride(focal_length_mm=valid_focal, zoom_multiple=valid_zoom)


def run_predict_service(
    state: AppState,
    *,
    source: Path | bytes,
    source_type: Literal["preset", "upload"],
    preset: SingleBirdPresetItem | None,
    optical_override: OpticalOverride | None,
) -> PredictResponse:
    """Run classical single-bird PDI, dual-bundle scoring, age mapping, and X-Ray assembly."""
    if source_type == "preset" and preset is None:
        raise ValueError("Curated preset metadata is required when source_type == 'preset'.")
    if source_type == "upload" and preset is not None:
        raise ValueError("Preset metadata must be None when source_type == 'upload'.")

    t_total = time.perf_counter()

    try:
        pdi_res = run_single_bird_pdi(
            source,
            config=state.pdi_config,
            optical_override=optical_override,
            return_xray=True,
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

    if pdi_res.xray is None:
        raise RuntimeError("run_single_bird_pdi(return_xray=True) returned no X-Ray artifacts.")

    if source_type == "preset":
        assert preset is not None
        pipeline = state.bundle["fold_pipelines"][preset.fold]
        oof_evaluation = True
    else:
        pipeline = state.bundle["production_pipeline"]
        oof_evaluation = False

    feature_names: tuple[str, ...] = state.bundle["feature_names"]
    t_inf = time.perf_counter()
    pred_g = predict_from_feature_dict(pipeline, feature_names, pdi_res.features)
    inference_ms = (time.perf_counter() - t_inf) * 1000.0

    weight_band = classify_weight_band(pred_g)
    band_interval = BandErrorInterval(**state.bundle["band_error_intervals"][weight_band])

    age_res = map_weight_to_age_stage(pred_g)
    age_stage = AgeStagePayload(
        stage_code=age_res.stage_code,
        stage_label=age_res.stage_label,
        estimated_age_days_range=age_res.estimated_age_days_range,
        out_of_training_range=age_res.out_of_training_range,
        valid_training_range_g=age_res.valid_training_range_g,
        valid_age_days_range=age_res.valid_age_days_range,
    )

    oof_audit: OofAuditPayload | None = None
    if source_type == "preset" and preset is not None:
        gt_weight = float(preset.ground_truth_weight_g)
        signed_err = pred_g - gt_weight
        abs_err = abs(signed_err)
        pct_err = (abs_err / gt_weight) * 100.0
        oof_audit = OofAuditPayload(
            fold=preset.fold,
            session_id=preset.session_id,
            ground_truth_weight_g=gt_weight,
            abs_err_g=round(abs_err, 4),
            pct_err=round(pct_err, 4),
            signed_err_g=round(signed_err, 4),
            pair_group_id=preset.pair_group_id,
        )

    optical = OpticalPayload(
        focal_length_mm=float(pdi_res.optical.focal_length_mm),
        zoom_multiple=float(pdi_res.optical.zoom_multiple),
        exif_optical_present=bool(pdi_res.optical.exif_optical_present),
        override_applied=bool(pdi_res.optical.override_applied),
        lens_class=pdi_res.optical.lens_class,
    )
    segmentation = SegmentationAuditPayload(
        scene_type=pdi_res.segmentation.scene_type,
        scene_is_bucket=bool(pdi_res.segmentation.scene_is_bucket),
        lcd_excluded_px=int(pdi_res.segmentation.lcd_excluded_px),
        lcd_core_count=int(pdi_res.segmentation.lcd_core_count),
        segmentation_fallback=bool(pdi_res.segmentation.segmentation_fallback),
        mask_area_frac=round(float(pdi_res.segmentation.mask_area_frac), 6),
        ellipse_from_min_area_rect=bool(pdi_res.segmentation.ellipse_from_min_area_rect),
    )
    xray = XRayPayload(
        stage1_original=f"data:image/jpeg;base64,{pdi_res.xray.stage1_original_b64}",
        stage2_clahe=f"data:image/jpeg;base64,{pdi_res.xray.stage2_clahe_b64}",
        stage3_mask=f"data:image/png;base64,{pdi_res.xray.stage3_mask_b64}",
        stage4_overlay=f"data:image/jpeg;base64,{pdi_res.xray.stage4_overlay_b64}",
        display_w=int(pdi_res.xray.display_w),
        display_h=int(pdi_res.xray.display_h),
        total_b64_bytes=int(pdi_res.xray.total_b64_bytes),
    )

    features_summary = {
        key: round(float(pdi_res.features[key]), 6) for key in FEATURES_SUMMARY_KEYS
    }
    model_features = {col: round(float(pdi_res.features[col]), 6) for col in feature_names}

    total_ms = (time.perf_counter() - t_total) * 1000.0
    timing = TimingPayload(
        pdi_ms=round(float(pdi_res.latency_ms), 3),
        inference_ms=round(float(inference_ms), 3),
        total_ms=round(float(total_ms), 3),
        sla_target_ms=SLA_TARGET_MS,
        within_sla=(total_ms < float(SLA_TARGET_MS)),
    )

    return PredictResponse(
        source_type=source_type,
        preset_id=preset.preset_id if preset is not None else None,
        oof_evaluation=oof_evaluation,
        model_version=str(state.bundle["model_version"]),
        winning_model_name=str(state.bundle["winning_model_name"]),
        pdi_config_hash=str(pdi_res.pdi_config_hash),
        predicted_weight_g=round(pred_g, 4),
        predicted_weight_kg=round(pred_g / 1000.0, 6),
        weight_band=weight_band,
        band_error_interval=band_interval,
        age_stage=age_stage,
        oof_audit=oof_audit,
        optical=optical,
        segmentation=segmentation,
        xray=xray,
        features_summary=features_summary,
        model_features=model_features,
        timing=timing,
    )


__all__ = [
    "FEATURES_SUMMARY_KEYS",
    "SLA_TARGET_MS",
    "build_optical_override",
    "classify_weight_band",
    "run_predict_service",
]
