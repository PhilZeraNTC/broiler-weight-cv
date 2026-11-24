"""Verified manifest+features dataset loader and 4-condition ablation specs (D-02, D-04, D-11).

Enforces strict Phase 3 scientific integrity gates before any model fits:
- Exact 1:1 row parity between ``manifest_clean.parquet`` and ``features_clean.parquet``
  on ``("image_id", "fold", "session_id", "weight_g", "weight_band", "lens_class")``.
- Uniform ``pdi_config_hash == PDIConfig().config_hash()`` lock (``43a3cfaa3208019e``).
- Zero cross-fold ``session_id`` overlap via ``verify_fold_integrity()`` (PROIBIÇÃO 3).
- Zero ``NaN``/``Inf`` feature values and zero ``BANNED_FEATURE_COLUMNS`` or background/audit
  metadata inside any regressor feature specification (PROIBIÇÃO 2).
- Never retains ``rel_path`` in ``LoadedModelingDataset.df`` (PROIBIÇÃO 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.config import CLEAN_MANIFEST_PATH, FEATURES_CLEAN_PATH
from backend.app.ingestion.splits import MIN_HEAVY_SESSIONS_PER_FOLD, verify_fold_integrity
from backend.app.pdi.config import PDIConfig
from backend.app.pdi.features import (
    BANNED_FEATURE_COLUMNS,
    CHROMATIC_TEXTURE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MORPHOMETRIC_FEATURE_COLUMNS,
    OPTICAL_FEATURE_COLUMNS,
)

ABLATION_CONDITION_ORDER: tuple[str, ...] = (
    "C1_uncalibrated_pdi",
    "C2_physics_normalized_strict",
    "C3_physics_plus_exif_meta",
    "C4_naive_double_count_fxz",
)

DIMENSIONLESS_SHAPE_COLUMNS: tuple[str, ...] = (
    "aspect_ratio_ellipse",
    "eccentricity",
    "solidity",
    "circularity",
    "extent_rotrect",
    "log_hu_1",
    "log_hu_2",
    "log_hu_3",
)

OPTICAL_NORM_5_COLUMNS: tuple[str, ...] = (
    "area_optical_norm",
    "sqrt_area_optical_norm",
    "perim_optical_norm",
    "major_axis_optical_norm",
    "minor_axis_optical_norm",
)

NAIVE_FXZ_5_COLUMNS: tuple[str, ...] = (
    "area_naive_fxz",
    "sqrt_area_naive_fxz",
    "perim_naive_fxz",
    "major_naive_fxz",
    "minor_naive_fxz",
)

NON_FEATURE_AUDIT_COLUMNS: frozenset[str] = frozenset(
    {
        "image_id",
        "rel_path",
        "fold",
        "session_id",
        "weight_g",
        "weight_band",
        "lens_class",
        "scene_type",
        "scene_is_bucket",
        "lcd_excluded_px",
        "segmentation_fallback",
        "pdi_config_hash",
    }
)

_C2_FORBIDDEN_RAW_COLUMNS: frozenset[str] = frozenset(
    {
        "focal_length_mm",
        "zoom_multiple",
        "exif_optical_present",
        "area_px",
        "perimeter_px",
        "major_axis_px",
        "minor_axis_px",
        "convex_area_px",
        "area_frac",
        "area_frac_optical_norm",
    }
)

_ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "image_id",
    "fold",
    "session_id",
    "weight_g",
    "weight_band",
    "lens_class",
)

_STRING_ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "image_id",
    "session_id",
    "weight_band",
    "lens_class",
)

_NUMERIC_ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "fold",
    "weight_g",
)

_FEATURES_JOIN_AUDIT_COLUMNS: tuple[str, ...] = (
    "image_id",
    "fold",
    "session_id",
    "weight_g",
    "weight_band",
    "lens_class",
    "scene_type",
    "scene_is_bucket",
    "lcd_excluded_px",
    "segmentation_fallback",
    "pdi_config_hash",
)

_EXPECTED_CONDITION_LENGTHS: dict[str, int] = {
    "C1_uncalibrated_pdi": 40,
    "C2_physics_normalized_strict": 39,
    "C3_physics_plus_exif_meta": 49,
    "C4_naive_double_count_fxz": 39,
}


def _validate_ablation_feature_specs(specs: dict[str, tuple[str, ...]]) -> None:
    if tuple(specs.keys()) != ABLATION_CONDITION_ORDER:
        raise RuntimeError(
            f"Ablation spec keys {tuple(specs.keys())!r} != {ABLATION_CONDITION_ORDER!r}"
        )
    if not set(DIMENSIONLESS_SHAPE_COLUMNS).issubset(MORPHOMETRIC_FEATURE_COLUMNS):
        raise RuntimeError(
            "DIMENSIONLESS_SHAPE_COLUMNS must be a subset of MORPHOMETRIC_FEATURE_COLUMNS."
        )
    if not set(OPTICAL_NORM_5_COLUMNS).issubset(OPTICAL_FEATURE_COLUMNS):
        raise RuntimeError(
            "OPTICAL_NORM_5_COLUMNS must be a subset of OPTICAL_FEATURE_COLUMNS."
        )

    prohibited = BANNED_FEATURE_COLUMNS | NON_FEATURE_AUDIT_COLUMNS
    for cond, cols in specs.items():
        if len(cols) != len(set(cols)):
            raise RuntimeError(
                f"Duplicate feature columns detected in ablation condition {cond!r}."
            )
        expected_len = _EXPECTED_CONDITION_LENGTHS[cond]
        if len(cols) != expected_len:
            raise RuntimeError(
                f"Ablation condition {cond!r} expected {expected_len} columns, got {len(cols)}."
            )
        overlap = prohibited.intersection(cols)
        if overlap:
            raise RuntimeError(
                f"PROIBIÇÃO 2 violation in {cond!r}: prohibited columns {sorted(overlap)}"
            )

    c1_optical = set(specs["C1_uncalibrated_pdi"]).intersection(OPTICAL_FEATURE_COLUMNS)
    if c1_optical:
        raise RuntimeError(
            f"C1_uncalibrated_pdi must exclude optical columns, found {sorted(c1_optical)}"
        )

    c2_raw = set(specs["C2_physics_normalized_strict"]).intersection(_C2_FORBIDDEN_RAW_COLUMNS)
    if c2_raw:
        raise RuntimeError(
            f"C2_physics_normalized_strict must exclude raw EXIF/pixel columns: {sorted(c2_raw)}"
        )

    c4_forbidden = set(specs["C4_naive_double_count_fxz"]).intersection(
        _C2_FORBIDDEN_RAW_COLUMNS | set(OPTICAL_NORM_5_COLUMNS)
    )
    if c4_forbidden:
        raise RuntimeError(
            f"C4_naive_double_count_fxz contains disallowed columns: {sorted(c4_forbidden)}"
        )


def build_ablation_feature_specs() -> dict[str, tuple[str, ...]]:
    """Return strictly whitelisted feature column tuples for all 4 EXIF ablation conditions."""
    specs: dict[str, tuple[str, ...]] = {
        "C1_uncalibrated_pdi": (
            MORPHOMETRIC_FEATURE_COLUMNS + CHROMATIC_TEXTURE_FEATURE_COLUMNS
        ),
        "C2_physics_normalized_strict": (
            DIMENSIONLESS_SHAPE_COLUMNS
            + CHROMATIC_TEXTURE_FEATURE_COLUMNS
            + OPTICAL_NORM_5_COLUMNS
        ),
        "C3_physics_plus_exif_meta": FEATURE_COLUMNS,
        "C4_naive_double_count_fxz": (
            DIMENSIONLESS_SHAPE_COLUMNS
            + CHROMATIC_TEXTURE_FEATURE_COLUMNS
            + NAIVE_FXZ_5_COLUMNS
        ),
    }
    _validate_ablation_feature_specs(specs)
    return specs


# Validate immutable feature specifications immediately on module import.
_MODULE_VERIFIED_ABLATION_SPECS: dict[str, tuple[str, ...]] = build_ablation_feature_specs()


@dataclass(frozen=True, slots=True)
class LoadedModelingDataset:
    """Verified, aligned, hash-locked Phase 3 modeling table and ablation feature specs."""

    df: pd.DataFrame
    pdi_config: PDIConfig
    pdi_config_hash: str
    fold_audit: dict[str, Any]
    ablation_columns: dict[str, tuple[str, ...]]


def load_modeling_dataset(
    manifest_path: Path = CLEAN_MANIFEST_PATH,
    features_path: Path = FEATURES_CLEAN_PATH,
    *,
    expected_config: PDIConfig | None = None,
    min_heavy_sessions_per_fold: int | None = MIN_HEAVY_SESSIONS_PER_FOLD,
) -> LoadedModelingDataset:
    """Load, align, audit, and augment clean manifest + PDI features for Phase 3 modeling."""
    manifest_file = Path(manifest_path)
    features_file = Path(features_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Clean manifest Parquet not found: {manifest_file}")
    if not features_file.is_file():
        raise FileNotFoundError(f"Clean features Parquet not found: {features_file}")

    manifest_df = pd.read_parquet(manifest_file)
    features_df = pd.read_parquet(features_file)

    missing_manifest = [col for col in _ALIGNMENT_COLUMNS if col not in manifest_df.columns]
    if missing_manifest:
        raise ValueError(f"Manifest Parquet missing required columns: {missing_manifest}")

    banned_in_features = BANNED_FEATURE_COLUMNS.intersection(features_df.columns)
    if banned_in_features:
        raise ValueError(
            "PROIBIÇÃO 2 violation: banned feature columns present in features Parquet: "
            f"{sorted(banned_in_features)}"
        )

    required_features_cols = _FEATURES_JOIN_AUDIT_COLUMNS + FEATURE_COLUMNS
    missing_features = [col for col in required_features_cols if col not in features_df.columns]
    if missing_features:
        raise ValueError(f"Features Parquet missing required columns: {missing_features}")

    if manifest_df.empty or features_df.empty:
        raise ValueError("Cannot load modeling dataset from empty manifest or features Parquet.")

    if len(manifest_df) != len(features_df):
        raise ValueError(
            f"Row count mismatch between manifest ({len(manifest_df)}) "
            f"and features ({len(features_df)})."
        )

    for col in _ALIGNMENT_COLUMNS:
        if manifest_df[col].isna().any() or features_df[col].isna().any():
            raise ValueError(f"Missing/NaN values detected in alignment column {col!r}.")

    for col in _FEATURES_JOIN_AUDIT_COLUMNS:
        if features_df[col].isna().any():
            raise ValueError(f"Missing/NaN values detected in audit column {col!r}.")

    if manifest_df["image_id"].astype(str).nunique() != len(manifest_df):
        raise ValueError("Duplicate image_id values detected in manifest Parquet.")
    if features_df["image_id"].astype(str).nunique() != len(features_df):
        raise ValueError("Duplicate image_id values detected in features Parquet.")

    for col in _STRING_ALIGNMENT_COLUMNS:
        m_vals = manifest_df[col].astype(str).tolist()
        f_vals = features_df[col].astype(str).tolist()
        if m_vals != f_vals:
            raise ValueError(
                f"1:1 row alignment mismatch between manifest and features on column {col!r}."
            )

    for col in _NUMERIC_ALIGNMENT_COLUMNS:
        m_num = manifest_df[col].to_numpy()
        f_num = features_df[col].to_numpy()
        if not np.array_equal(m_num, f_num):
            raise ValueError(
                f"1:1 row alignment mismatch between manifest and features on column {col!r}."
            )

    cfg = expected_config if expected_config is not None else PDIConfig()
    expected_hash = cfg.config_hash()
    if features_df["pdi_config_hash"].isna().any():
        raise ValueError("Missing/NaN values detected in pdi_config_hash column.")
    observed_hashes = features_df["pdi_config_hash"].astype(str).unique().tolist()
    if observed_hashes != [expected_hash]:
        raise ValueError(
            f"PDIConfig hash lock mismatch: expected uniform {expected_hash!r}, "
            f"got {observed_hashes!r}."
        )

    # Select strictly the verified audit/join + 49 frozen PDI columns; never retain rel_path.
    merged = (
        features_df.loc[:, list(required_features_cols)].copy().reset_index(drop=True)
    )
    if "rel_path" in merged.columns:
        raise RuntimeError(
            "PROIBIÇÃO 4 invariant violated: rel_path leaked into modeling DataFrame."
        )

    # PROIBIÇÃO 3: verify zero cross-fold session_id overlap and heavy-session floor.
    fold_audit = verify_fold_integrity(
        merged,
        min_heavy_sessions_per_fold=min_heavy_sessions_per_fold,
    )

    weights = merged["weight_g"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("weight_g must be finite and strictly positive for all clean rows.")

    feature_matrix = merged.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError("Non-finite (NaN or Inf) value detected in 49-feature matrix.")

    focal = merged["focal_length_mm"].to_numpy(dtype=np.float64)
    zoom = merged["zoom_multiple"].to_numpy(dtype=np.float64)
    if np.any(focal <= 0.0) or np.any(zoom <= 0.0):
        raise ValueError("focal_length_mm and zoom_multiple must be strictly positive.")

    # Derive the 5 offline-only C4 naive (focal * zoom) double-count ablation columns
    # using Pandas 3.1 Copy-on-Write safe single-step .loc assignment.
    fxz = focal * zoom
    area_px = merged["area_px"].to_numpy(dtype=np.float64)
    perimeter_px = merged["perimeter_px"].to_numpy(dtype=np.float64)
    major_px = merged["major_axis_px"].to_numpy(dtype=np.float64)
    minor_px = merged["minor_axis_px"].to_numpy(dtype=np.float64)

    area_naive = area_px / (fxz ** 2)
    naive_cols_map: dict[str, np.ndarray] = {
        "area_naive_fxz": area_naive,
        "sqrt_area_naive_fxz": np.sqrt(np.clip(area_naive, 0.0, None)),
        "perim_naive_fxz": perimeter_px / fxz,
        "major_naive_fxz": major_px / fxz,
        "minor_naive_fxz": minor_px / fxz,
    }
    for col in NAIVE_FXZ_5_COLUMNS:
        vals = naive_cols_map[col]
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Non-finite value computed for offline C4 column {col!r}.")
        merged.loc[:, col] = vals

    ablation_columns = build_ablation_feature_specs()
    return LoadedModelingDataset(
        df=merged,
        pdi_config=cfg,
        pdi_config_hash=expected_hash,
        fold_audit=fold_audit,
        ablation_columns=ablation_columns,
    )


__all__ = [
    "ABLATION_CONDITION_ORDER",
    "DIMENSIONLESS_SHAPE_COLUMNS",
    "OPTICAL_NORM_5_COLUMNS",
    "NAIVE_FXZ_5_COLUMNS",
    "NON_FEATURE_AUDIT_COLUMNS",
    "LoadedModelingDataset",
    "build_ablation_feature_specs",
    "load_modeling_dataset",
]
