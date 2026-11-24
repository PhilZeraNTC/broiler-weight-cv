"""Strict Pydantic v2 REST API schemas for AvianScale CV (WEB-01, D-01..D-05).

Imports authoritative optical bounds from ``backend.app.pdi.io_exif`` and
authoritative Watershed ASVS V5 bounds from ``backend.app.pdi.config``:
- ``h_rel`` in ``[0.05, 0.80]`` (default ``0.20``)
- ``min_distance_rel`` in ``[0.20, 2.50]`` (default ``0.85``)
- ``min_area_ratio`` in ``[0.01, 0.45]`` (default ``0.10``)

These Phase 2 locked constants reconcile and supersede the narrower draft values
in ``04-CONTEXT.md`` (D-04 / D-10) and align verbatim with approved ``04-UI-SPEC.md``.
All strict contract models forbid extra fields (``extra="forbid"``), forbid ``NaN``/``Inf``
floats, and permanently omit ``rel_path`` and raw dataset filenames (PROIBIÇÃO 4).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.pdi.config import (
    WATERSHED_MAX_AREA_RATIO,
    WATERSHED_MAX_DISTANCE_REL,
    WATERSHED_MAX_H_REL,
    WATERSHED_MIN_AREA_RATIO,
    WATERSHED_MIN_DISTANCE_REL,
    WATERSHED_MIN_H_REL,
)
from backend.app.pdi.io_exif import (
    MAX_VALID_FOCAL_LENGTH_MM,
    MAX_VALID_ZOOM_MULTIPLE,
    MIN_VALID_FOCAL_LENGTH_MM,
    MIN_VALID_ZOOM_MULTIPLE,
)
from backend.app.pdi.segmentation import SceneType

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
LensClass = Literal["wide", "ultrawide", "unknown"]
SourceType = Literal["preset", "upload"]
WeightBand = Literal["<500g", "500-1000g", ">1000g"]


class _StrictSchema(BaseModel):
    """Base schema rejecting undeclared fields (prevents accidental ``rel_path`` leaks)."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(_StrictSchema):
    """Payload returned by ``GET /api/health``."""

    status: Literal["ok"] = "ok"
    model_version: str
    winning_model_name: str
    winning_ablation_condition: str
    pdi_config_hash: str
    n_features: int = Field(..., ge=1)
    single_bird_presets_count: int = Field(..., ge=0)
    watershed_presets_count: int = Field(..., ge=0)
    classical_pdi_only: Literal[True] = True


class SingleBirdPresetItem(_StrictSchema):
    """Curated out-of-fold single-bird preset entry (opaque IDs only; zero ``rel_path``)."""

    preset_id: str
    label: str
    category: str
    pair_group_id: str | None = None
    image_id: str
    session_id: str
    fold: int = Field(..., ge=0, le=4)
    ground_truth_weight_g: FiniteFloat = Field(..., gt=0.0)
    weight_band: WeightBand
    lens_class: LensClass
    scene_type: SceneType
    oof_predicted_g: FiniteFloat = Field(..., gt=0.0)
    oof_abs_err_g: FiniteFloat = Field(..., ge=0.0)
    oof_pct_err: FiniteFloat = Field(..., ge=0.0)
    thumbnail_url: str


class WatershedPresetItem(_StrictSchema):
    """Curated multi-bird Watershed preset entry with mandatory group-sum disclaimer."""

    preset_id: str
    label: str
    folder_id: str
    image_id: str
    lens_class: LensClass
    expected_bird_count: int = Field(..., ge=1)
    detected_bird_count_default: int = Field(..., ge=0)
    group_sum_weight_g: FiniteFloat = Field(..., gt=0.0)
    mean_bird_weight_reference_g: FiniteFloat = Field(..., gt=0.0)
    ground_truth_scope: Literal["group_sum_only"] = "group_sum_only"
    disclaimer: str
    thumbnail_url: str


class PresetsResponse(_StrictSchema):
    """Payload returned by ``GET /api/presets``."""

    schema_version: str
    pdi_config_hash: str
    winning_model_name: str
    winning_ablation_condition: str
    single_bird_presets: list[SingleBirdPresetItem]
    watershed_presets: list[WatershedPresetItem]


class BandErrorInterval(_StrictSchema):
    """Band-stratified OOF empirical error interval from the serialized model bundle."""

    n: int = Field(..., ge=1)
    mae_g: FiniteFloat = Field(..., ge=0.0)
    mape_pct: FiniteFloat = Field(..., ge=0.0)
    median_abs_err_g: FiniteFloat = Field(..., ge=0.0)
    p68_abs_err_g: FiniteFloat = Field(..., ge=0.0)
    p90_abs_err_g: FiniteFloat = Field(..., ge=0.0)
    mbe_g: FiniteFloat


class AgeStagePayload(_StrictSchema):
    """Cobb Avian 48 developmental age-stage mapping and training coverage disclosure."""

    stage_code: str
    stage_label: str
    estimated_age_days_range: str
    out_of_training_range: bool
    valid_training_range_g: tuple[int, int]
    valid_age_days_range: str


class OofAuditPayload(_StrictSchema):
    """Honest out-of-fold audit comparison populated strictly for curated single-bird presets."""

    fold: int = Field(..., ge=0, le=4)
    session_id: str
    ground_truth_weight_g: FiniteFloat = Field(..., gt=0.0)
    abs_err_g: FiniteFloat = Field(..., ge=0.0)
    pct_err: FiniteFloat = Field(..., ge=0.0)
    signed_err_g: FiniteFloat
    pair_group_id: str | None = None


class OpticalPayload(_StrictSchema):
    """Resolved optical calibration metadata after override -> EXIF -> default precedence."""

    focal_length_mm: FiniteFloat = Field(
        ...,
        ge=MIN_VALID_FOCAL_LENGTH_MM,
        le=MAX_VALID_FOCAL_LENGTH_MM,
    )
    zoom_multiple: FiniteFloat = Field(
        ...,
        ge=MIN_VALID_ZOOM_MULTIPLE,
        le=MAX_VALID_ZOOM_MULTIPLE,
    )
    exif_optical_present: bool
    override_applied: bool
    lens_class: LensClass


class SegmentationAuditPayload(_StrictSchema):
    """Single-bird classical segmentation audit metadata (LCD suppression & fallback)."""

    scene_type: SceneType
    scene_is_bucket: bool
    lcd_excluded_px: int = Field(..., ge=0)
    lcd_core_count: int = Field(..., ge=0)
    segmentation_fallback: bool
    mask_area_frac: FiniteFloat = Field(..., ge=0.0, le=1.0)
    ellipse_from_min_area_rect: bool


class XRayPayload(_StrictSchema):
    """Complete RFC 2397 Data URIs for all 4 single-bird classical PDI X-Ray stages."""

    stage1_original: str
    stage2_clahe: str
    stage3_mask: str
    stage4_overlay: str
    display_w: int = Field(..., ge=1)
    display_h: int = Field(..., ge=1)
    total_b64_bytes: int = Field(..., ge=1)


class TimingPayload(_StrictSchema):
    """End-to-end server-side latency breakdown and `< 500 ms` SLA status."""

    pdi_ms: FiniteFloat = Field(..., ge=0.0)
    inference_ms: FiniteFloat = Field(..., ge=0.0)
    total_ms: FiniteFloat = Field(..., ge=0.0)
    sla_target_ms: int = Field(default=500, ge=1)
    within_sla: bool


class PredictResponse(_StrictSchema):
    """Payload returned by ``POST /api/predict``."""

    source_type: SourceType
    preset_id: str | None = None
    oof_evaluation: bool
    model_version: str
    winning_model_name: str
    pdi_config_hash: str
    predicted_weight_g: FiniteFloat = Field(..., gt=0.0)
    predicted_weight_kg: FiniteFloat = Field(..., gt=0.0)
    weight_band: WeightBand
    band_error_interval: BandErrorInterval
    age_stage: AgeStagePayload
    oof_audit: OofAuditPayload | None = None
    optical: OpticalPayload
    segmentation: SegmentationAuditPayload
    xray: XRayPayload
    features_summary: dict[str, FiniteFloat]
    model_features: dict[str, FiniteFloat]
    timing: TimingPayload


class WatershedParamBounds(_StrictSchema):
    """Authoritative ASVS V5 inclusive bounds for interactive Watershed tuning sliders."""

    h_rel: tuple[FiniteFloat, FiniteFloat] = (WATERSHED_MIN_H_REL, WATERSHED_MAX_H_REL)
    min_distance_rel: tuple[FiniteFloat, FiniteFloat] = (
        WATERSHED_MIN_DISTANCE_REL,
        WATERSHED_MAX_DISTANCE_REL,
    )
    min_area_ratio: tuple[FiniteFloat, FiniteFloat] = (
        WATERSHED_MIN_AREA_RATIO,
        WATERSHED_MAX_AREA_RATIO,
    )


class WatershedParamDefaults(_StrictSchema):
    """Locked Phase 2 PDIConfig default slider values for Marker-based Watershed."""

    h_rel: FiniteFloat = Field(..., ge=WATERSHED_MIN_H_REL, le=WATERSHED_MAX_H_REL)
    min_distance_rel: FiniteFloat = Field(
        ...,
        ge=WATERSHED_MIN_DISTANCE_REL,
        le=WATERSHED_MAX_DISTANCE_REL,
    )
    min_area_ratio: FiniteFloat = Field(
        ...,
        ge=WATERSHED_MIN_AREA_RATIO,
        le=WATERSHED_MAX_AREA_RATIO,
    )


class WatershedParamsPayload(_StrictSchema):
    """Effective Watershed parameters paired with canonical defaults and ASVS V5 bounds."""

    h_rel: FiniteFloat = Field(..., ge=WATERSHED_MIN_H_REL, le=WATERSHED_MAX_H_REL)
    min_distance_rel: FiniteFloat = Field(
        ...,
        ge=WATERSHED_MIN_DISTANCE_REL,
        le=WATERSHED_MAX_DISTANCE_REL,
    )
    min_area_ratio: FiniteFloat = Field(
        ...,
        ge=WATERSHED_MIN_AREA_RATIO,
        le=WATERSHED_MAX_AREA_RATIO,
    )
    defaults: WatershedParamDefaults
    bounds: WatershedParamBounds


class WatershedPresetReference(_StrictSchema):
    """Group-sum dataset reference populated strictly for curated Watershed presets."""

    folder_id: str
    expected_bird_count: int = Field(..., ge=1)
    group_sum_weight_g: FiniteFloat = Field(..., gt=0.0)
    mean_bird_weight_reference_g: FiniteFloat = Field(..., gt=0.0)
    sum_diff_g: FiniteFloat
    sum_pct_err: FiniteFloat = Field(..., ge=0.0)


class WatershedInstancePayload(_StrictSchema):
    """Per-instance separated bird metrics and illustrative tabular weight estimate."""

    instance_id: int = Field(..., ge=1)
    color_hex: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    centroid_xy: tuple[FiniteFloat, FiniteFloat]
    area_work_px: int = Field(..., ge=1)
    area_px: FiniteFloat = Field(..., gt=0.0)
    area_frac: FiniteFloat = Field(..., ge=0.0, le=1.0)
    area_optical_norm: FiniteFloat = Field(..., ge=0.0)
    estimated_weight_g: FiniteFloat = Field(..., gt=0.0)
    share_of_total_pct: FiniteFloat = Field(..., ge=0.0, le=100.0)
    age_stage: AgeStagePayload


class WatershedVisualsPayload(_StrictSchema):
    """Complete RFC 2397 Data URIs for all 4 classical Watershed inspection stages."""

    stage1_original: str
    stage2_union_mask: str
    stage3_distance_peaks: str
    stage4_color_basins: str
    total_b64_bytes: int = Field(..., ge=1)


class WatershedResponse(_StrictSchema):
    """Payload returned by ``POST /api/watershed-demo``."""

    source_type: SourceType
    preset_id: str | None = None
    ground_truth_scope: Literal["group_sum_only"] = "group_sum_only"
    disclaimer: str
    bird_count: int = Field(..., ge=0)
    estimated_total_weight_g: FiniteFloat = Field(..., ge=0.0)
    estimated_mean_weight_g: FiniteFloat | None = None
    preset_reference: WatershedPresetReference | None = None
    parameters: WatershedParamsPayload
    optical: OpticalPayload
    red_feeder_excluded_px: int = Field(..., ge=0)
    pdi_config_hash: str
    instances: list[WatershedInstancePayload]
    visuals: WatershedVisualsPayload
    timing: TimingPayload


class MetricsResponse(_StrictSchema):
    """Unified read-only payload returned by ``GET /api/metrics``."""

    schema_version: str
    pdi_config_hash: str
    evaluation_report: dict[str, Any]
    dataset_audit_summary: dict[str, Any]
    pdi_calibration_audit: dict[str, Any]


__all__ = [
    "AgeStagePayload",
    "BandErrorInterval",
    "FiniteFloat",
    "HealthResponse",
    "LensClass",
    "MetricsResponse",
    "OofAuditPayload",
    "OpticalPayload",
    "PredictResponse",
    "PresetsResponse",
    "SceneType",
    "SegmentationAuditPayload",
    "SingleBirdPresetItem",
    "SourceType",
    "TimingPayload",
    "WatershedInstancePayload",
    "WatershedParamBounds",
    "WatershedParamDefaults",
    "WatershedParamsPayload",
    "WatershedPresetItem",
    "WatershedPresetReference",
    "WatershedResponse",
    "WatershedVisualsPayload",
    "WeightBand",
    "XRayPayload",
]
