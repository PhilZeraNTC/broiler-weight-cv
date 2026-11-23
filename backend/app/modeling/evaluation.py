"""Leak-free 5-fold OOF evaluation, 4-condition optical ablation & anti-confounding gate.

Implements the Phase 3 scientific evaluation engine (D-01..D-06, D-09):
- Strictly fold-isolated 5-fold ``session_id`` cross-validation across all 4 mandatory
  classical tabular families (``RidgeLog``, ``HuberLog``, ``RandomForestLog``,
  ``LightGBMLog``) x all 4 EXIF optical ablation conditions (``C1``..``C4``).
- Unweighted out-of-fold global, session-averaged (``session_averaged_r2``), and
  stratified metrics (per fold, weight band, lens, visual scene, band x scene,
  band x lens, and per-band error quantiles ``p50``/``p68``/``p90``).
- Within-session dual-lens consistency across mixed-lens sessions (``wide`` vs.
  ``ultrawide`` OOF predictions).
- Dual-mode counterfactual lens-flip audit (``focal_only`` + ``physical_lens_swap``
  propagating Redmi 10 ``4.25/1.0 <-> 1.65/0.6`` and sensor area ratio ``1.5625``)
  paired with the ``<500g`` ultrawide degradation check to automatically detect and
  disqualify lens-ID shortcuts in ``C3`` and ``C4`` while proving mathematical ``0.0%``
  invariance on ``C2_physics_normalized_strict``.
- Deterministic ``D-06`` winner selection minimizing ``balanced_band_mape_pct`` among
  gate-passing candidates, plus 2x2 weighting x smearing and linear allometric-log
  sensitivity diagnostics.

PROIBIÇÃO 1-4 compliance: zero deep learning/OCR libraries, zero random/ungrouped
splitters, zero validation-metric weighting, zero ``scene_type``/``lens_class`` features
in ``X``, and zero raw filesystem paths or JPEG filenames.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from backend.app.ingestion.decoding import (
    BAND_LIGHT_MAX_EXCLUSIVE_G,
    BAND_MEDIUM_MAX_INCLUSIVE_G,
    assign_weight_band,
)
from backend.app.modeling.datasets import ABLATION_CONDITION_ORDER, LoadedModelingDataset
from backend.app.modeling.models import (
    DEFAULT_WEIGHTING_MODE,
    MODEL_FAMILY_NAMES,
    build_regressor_pipeline,
    compute_sample_weights,
    extract_feature_importances,
)

WEIGHT_BAND_ORDER: tuple[str, ...] = ("<500g", "500-1000g", ">1000g")
LENS_CLASS_ORDER: tuple[str, ...] = ("wide", "ultrawide")
SCENE_TYPE_ORDER: tuple[str, ...] = ("bucket_dorsal", "scale_or_litter")

COUNTERFACTUAL_FOCAL_MEDIAN_THRESHOLD_PCT: float = 10.0
COUNTERFACTUAL_PHYS_MEDIAN_THRESHOLD_PCT: float = 10.0
COUNTERFACTUAL_PHYS_P90_THRESHOLD_PCT: float = 15.0
WIDE_OVER_UW_SENSOR_AREA_RATIO: float = 1.5625
LITERATURE_R2_REFERENCE: float = 0.90

_WIDE_FOCAL_MM: float = 4.25
_UW_FOCAL_MM: float = 1.65
_WIDE_ZOOM: float = 1.0
_UW_ZOOM: float = 0.6
_N_FOLDS: int = 5

_LENS_DEPENDENT_PHYSICAL_SWAP_COLUMNS: frozenset[str] = frozenset(
    {
        "focal_length_mm",
        "zoom_multiple",
        "area_px",
        "convex_area_px",
        "perimeter_px",
        "major_axis_px",
        "minor_axis_px",
        "area_frac",
        "area_frac_optical_norm",
        "area_naive_fxz",
        "sqrt_area_naive_fxz",
        "perim_naive_fxz",
        "major_naive_fxz",
        "minor_naive_fxz",
    }
)


def _round4(value: float) -> float:
    """Round deterministically to 4 decimals and normalize ``-0.0`` to ``0.0``."""
    rounded = round(float(value), 4)
    return 0.0 if rounded == 0.0 else rounded


def _round6(value: float) -> float:
    """Round deterministically to 6 decimals and normalize ``-0.0`` to ``0.0``."""
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0.0 else rounded


def _contains_bool_array(raw: np.ndarray) -> bool:
    if raw.dtype == np.bool_:
        return True
    if raw.dtype == object:
        return any(isinstance(v, bool) for v in raw.ravel())
    return False


def _coerce_1d_float_array(
    values: Sequence[float] | np.ndarray | pd.Series,
    name: str,
) -> np.ndarray:
    """Validate and coerce an array-like numeric sequence into a 1D ``float64`` ndarray."""
    raw = np.asarray(values)
    if _contains_bool_array(raw):
        raise ValueError(f"{name} must be numeric, not boolean.")
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to float64.") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape!r}")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite (NaN or Inf) values.")
    return arr


def _weight_band_for_value(weight_g: float) -> str:
    """Assign weight band aligned identically with Phase 1 ``assign_weight_band()``."""
    if float(weight_g).is_integer():
        return assign_weight_band(int(weight_g))
    if weight_g < BAND_LIGHT_MAX_EXCLUSIVE_G:
        return "<500g"
    if weight_g <= BAND_MEDIUM_MAX_INCLUSIVE_G:
        return "500-1000g"
    return ">1000g"


def _summarize_shifts(shifts: np.ndarray) -> dict[str, float | None]:
    if shifts.size == 0:
        return {
            "median_shift_pct": None,
            "mean_shift_pct": None,
            "p90_shift_pct": None,
        }
    return {
        "median_shift_pct": _round4(float(np.median(shifts))),
        "mean_shift_pct": _round4(float(np.mean(shifts))),
        "p90_shift_pct": _round4(float(np.percentile(shifts, 90.0))),
    }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Complete leak-free 5-fold OOF evaluation for one candidate configuration."""

    model_name: str
    ablation_condition: str
    feature_names: tuple[str, ...]
    weighting_mode: str
    duan_smearing_enabled: bool
    oof_predictions: np.ndarray
    fold_pipelines: dict[int, Pipeline]
    global_metrics: dict[str, Any]
    session_level_metrics: dict[str, Any]
    balanced_band_mape_pct: float
    per_fold: list[dict[str, Any]]
    by_weight_band: dict[str, dict[str, Any]]
    by_lens: dict[str, dict[str, Any]]
    by_scene: dict[str, dict[str, Any]]
    by_band_and_scene: dict[str, dict[str, Any]]
    by_band_and_lens: dict[str, dict[str, Any]]
    band_error_intervals: dict[str, dict[str, Any]]
    dual_lens_consistency: dict[str, Any]
    counterfactual_lens_flip: dict[str, Any]
    lens_confounding_detected: bool
    confounding_reasons: list[str]
    eligible_for_production: bool
    is_winner: bool


@dataclass(frozen=True, slots=True)
class FullEvaluationResult:
    """Aggregated Phase 3 5-fold CV, 4-condition ablation, mechanics, and winner audit."""

    candidates: list[CandidateEvaluation]
    winner: CandidateEvaluation
    mechanics_ablation: list[dict[str, Any]]
    allometric_log_ablation: list[dict[str, Any]]
    optical_ablation: dict[str, Any]
    feature_importances: list[dict[str, Any]]
    oof_scatter_points: list[dict[str, Any]]


def compute_regression_metrics(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Compute strictly unweighted regression metrics in grams with safe empty/single guards."""
    y_t = _coerce_1d_float_array(y_true, "y_true")
    y_p = _coerce_1d_float_array(y_pred, "y_pred")
    if y_t.shape != y_p.shape:
        raise ValueError(
            f"y_true ({y_t.shape[0]}) and y_pred ({y_p.shape[0]}) must have equal length."
        )

    n = int(y_t.size)
    if n == 0:
        return {
            "n": 0,
            "mae_g": None,
            "rmse_g": None,
            "mape_pct": None,
            "r2": None,
            "mbe_g": None,
            "median_abs_err_g": None,
            "p68_abs_err_g": None,
            "p90_abs_err_g": None,
        }

    err = y_p - y_t
    abs_err = np.abs(err)
    mae_g = float(mean_absolute_error(y_t, y_p))
    rmse_g = float(root_mean_squared_error(y_t, y_p))
    mape_pct = float(np.mean(abs_err / np.maximum(y_t, 1e-6)) * 100.0)
    mbe_g = float(np.mean(err))
    p50, p68, p90 = np.percentile(abs_err, (50.0, 68.0, 90.0))

    r2_val: float | None = None
    if n >= 2 and float(np.ptp(y_t)) > 0.0:
        r2_val = _round4(float(r2_score(y_t, y_p)))

    return {
        "n": n,
        "mae_g": _round4(mae_g),
        "rmse_g": _round4(rmse_g),
        "mape_pct": _round4(mape_pct),
        "r2": r2_val,
        "mbe_g": _round4(mbe_g),
        "median_abs_err_g": _round4(float(p50)),
        "p68_abs_err_g": _round4(float(p68)),
        "p90_abs_err_g": _round4(float(p90)),
    }


def compute_session_level_metrics(
    df: pd.DataFrame,
    oof_pred: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Aggregate OOF actual and predicted weights per ``session_id`` for session-level metrics."""
    pred_arr = _coerce_1d_float_array(oof_pred, "oof_pred")
    if len(df) != pred_arr.size:
        raise ValueError(
            f"df length ({len(df)}) does not match oof_pred length ({pred_arr.size})."
        )

    if df.empty:
        return {
            "n_sessions": 0,
            "session_averaged_r2": None,
            "mae_g": None,
            "rmse_g": None,
            "mape_pct": None,
            "mbe_g": None,
            "median_abs_err_g": None,
            "p68_abs_err_g": None,
            "p90_abs_err_g": None,
        }

    y_true_arr = _coerce_1d_float_array(df["weight_g"].to_numpy(dtype=np.float64), "weight_g")
    grouped_indices = df.groupby(["session_id"], sort=True).indices

    sess_true: list[float] = []
    sess_pred: list[float] = []
    for _, raw_idx in sorted(grouped_indices.items(), key=lambda kv: kv[0]):
        idx = np.asarray(raw_idx, dtype=np.int64)
        sess_true.append(float(np.mean(y_true_arr[idx])))
        sess_pred.append(float(np.mean(pred_arr[idx])))

    metrics = compute_regression_metrics(sess_true, sess_pred)
    return {
        "n_sessions": int(metrics["n"]),
        "session_averaged_r2": metrics["r2"],
        "mae_g": metrics["mae_g"],
        "rmse_g": metrics["rmse_g"],
        "mape_pct": metrics["mape_pct"],
        "mbe_g": metrics["mbe_g"],
        "median_abs_err_g": metrics["median_abs_err_g"],
        "p68_abs_err_g": metrics["p68_abs_err_g"],
        "p90_abs_err_g": metrics["p90_abs_err_g"],
    }


def compute_stratified_metrics(
    df: pd.DataFrame,
    oof_pred: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Compute unweighted stratified OOF metrics across folds, bands, lenses, and scenes."""
    pred_arr = _coerce_1d_float_array(oof_pred, "oof_pred")
    if len(df) != pred_arr.size:
        raise ValueError(
            f"df length ({len(df)}) does not match oof_pred length ({pred_arr.size})."
        )

    y_true_arr = (
        _coerce_1d_float_array(df["weight_g"].to_numpy(dtype=np.float64), "weight_g")
        if not df.empty
        else np.asarray([], dtype=np.float64)
    )
    folds_arr = (
        df["fold"].to_numpy(dtype=np.int64)
        if not df.empty
        else np.asarray([], dtype=np.int64)
    )
    band_arr = np.asarray(
        [_weight_band_for_value(float(w)) for w in y_true_arr],
        dtype=object,
    )
    lens_arr = (
        df["lens_class"].astype(str).to_numpy()
        if not df.empty
        else np.asarray([], dtype=object)
    )
    scene_arr = (
        df["scene_type"].astype(str).to_numpy()
        if not df.empty
        else np.asarray([], dtype=object)
    )
    sessions_arr = (
        df["session_id"].astype(str).to_numpy()
        if not df.empty
        else np.asarray([], dtype=object)
    )

    per_fold: list[dict[str, Any]] = []
    for fold_idx in range(_N_FOLDS):
        mask = folds_arr == fold_idx
        fold_metrics = compute_regression_metrics(y_true_arr[mask], pred_arr[mask])
        n_sessions = int(len(set(sessions_arr[mask].tolist()))) if np.any(mask) else 0
        per_fold.append(
            {
                "fold": int(fold_idx),
                "n_sessions": n_sessions,
                **fold_metrics,
            }
        )

    by_weight_band: dict[str, dict[str, Any]] = {}
    band_error_intervals: dict[str, dict[str, Any]] = {}
    for band in WEIGHT_BAND_ORDER:
        mask = band_arr == band
        m = compute_regression_metrics(y_true_arr[mask], pred_arr[mask])
        by_weight_band[band] = m
        band_error_intervals[band] = {
            "n": m["n"],
            "mae_g": m["mae_g"],
            "mape_pct": m["mape_pct"],
            "median_abs_err_g": m["median_abs_err_g"],
            "p68_abs_err_g": m["p68_abs_err_g"],
            "p90_abs_err_g": m["p90_abs_err_g"],
            "mbe_g": m["mbe_g"],
        }

    by_lens: dict[str, dict[str, Any]] = {}
    for lens in LENS_CLASS_ORDER:
        mask = lens_arr == lens
        by_lens[lens] = compute_regression_metrics(y_true_arr[mask], pred_arr[mask])

    by_scene: dict[str, dict[str, Any]] = {}
    for scene in SCENE_TYPE_ORDER:
        mask = scene_arr == scene
        by_scene[scene] = compute_regression_metrics(y_true_arr[mask], pred_arr[mask])

    by_band_and_scene: dict[str, dict[str, Any]] = {}
    for band in WEIGHT_BAND_ORDER:
        for scene in SCENE_TYPE_ORDER:
            mask = (band_arr == band) & (scene_arr == scene)
            by_band_and_scene[f"{band}__{scene}"] = compute_regression_metrics(
                y_true_arr[mask],
                pred_arr[mask],
            )

    by_band_and_lens: dict[str, dict[str, Any]] = {}
    for band in WEIGHT_BAND_ORDER:
        for lens in LENS_CLASS_ORDER:
            mask = (band_arr == band) & (lens_arr == lens)
            by_band_and_lens[f"{band}__{lens}"] = compute_regression_metrics(
                y_true_arr[mask],
                pred_arr[mask],
            )

    non_none_band_mapes = [
        float(by_weight_band[band]["mape_pct"])
        for band in WEIGHT_BAND_ORDER
        if by_weight_band[band]["mape_pct"] is not None
    ]
    balanced_band_mape_pct: float | None = (
        _round4(float(np.mean(non_none_band_mapes))) if non_none_band_mapes else None
    )

    return {
        "per_fold": per_fold,
        "by_weight_band": by_weight_band,
        "by_lens": by_lens,
        "by_scene": by_scene,
        "by_band_and_scene": by_band_and_scene,
        "by_band_and_lens": by_band_and_lens,
        "band_error_intervals": band_error_intervals,
        "balanced_band_mape_pct": balanced_band_mape_pct,
    }


def compute_within_session_dual_lens_consistency(
    df: pd.DataFrame,
    oof_pred: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Pair ``wide`` vs. ``ultrawide`` OOF predictions per ``session_id`` (D-05)."""
    pred_arr = _coerce_1d_float_array(oof_pred, "oof_pred")
    if len(df) != pred_arr.size:
        raise ValueError(
            f"df length ({len(df)}) does not match oof_pred length ({pred_arr.size})."
        )

    abs_diffs: list[float] = []
    rel_diffs_pct: list[float] = []
    signed_diffs: list[float] = []

    if not df.empty:
        lens_arr = df["lens_class"].astype(str).to_numpy()
        y_true_arr = _coerce_1d_float_array(df["weight_g"].to_numpy(dtype=np.float64), "weight_g")
        grouped_indices = df.groupby(["session_id"], sort=True).indices

        for _, raw_idx in sorted(grouped_indices.items(), key=lambda kv: kv[0]):
            idx = np.asarray(raw_idx, dtype=np.int64)
            grp_lens = lens_arr[idx]
            w_idx = idx[grp_lens == "wide"]
            u_idx = idx[grp_lens == "ultrawide"]
            if w_idx.size == 0 or u_idx.size == 0:
                continue

            y_w = float(np.mean(pred_arr[w_idx]))
            y_u = float(np.mean(pred_arr[u_idx]))
            w_true = float(np.mean(y_true_arr[idx]))

            abs_diff = abs(y_w - y_u)
            rel_diff_pct = (abs_diff / max(w_true, 1e-6)) * 100.0
            signed_diff = y_w - y_u

            abs_diffs.append(abs_diff)
            rel_diffs_pct.append(rel_diff_pct)
            signed_diffs.append(signed_diff)

    n_mixed = len(abs_diffs)
    if n_mixed == 0:
        return {
            "n_mixed_lens_sessions": 0,
            "mean_abs_diff_g": None,
            "median_abs_diff_g": None,
            "mean_rel_diff_pct": None,
            "median_rel_diff_pct": None,
            "mean_signed_wide_minus_uw_g": None,
            "median_signed_wide_minus_uw_g": None,
        }

    abs_arr = np.asarray(abs_diffs, dtype=np.float64)
    rel_arr = np.asarray(rel_diffs_pct, dtype=np.float64)
    signed_arr = np.asarray(signed_diffs, dtype=np.float64)
    return {
        "n_mixed_lens_sessions": int(n_mixed),
        "mean_abs_diff_g": _round4(float(np.mean(abs_arr))),
        "median_abs_diff_g": _round4(float(np.median(abs_arr))),
        "mean_rel_diff_pct": _round4(float(np.mean(rel_arr))),
        "median_rel_diff_pct": _round4(float(np.median(rel_arr))),
        "mean_signed_wide_minus_uw_g": _round4(float(np.mean(signed_arr))),
        "median_signed_wide_minus_uw_g": _round4(float(np.median(signed_arr))),
    }


def compute_counterfactual_lens_flip(
    df: pd.DataFrame,
    condition: str,
    feature_names: Sequence[str],
    fold_pipelines: Mapping[int, Pipeline],
) -> dict[str, Any]:
    """Run dual-mode OOF counterfactual lens-flip audit (``focal_only`` + ``physical_lens_swap``).

    Evaluates strictly on held-out fold validation rows with ``lens_class in ("wide", "ultrawide")``
    using Pandas 3.1 Copy-on-Write safe ``.copy()`` + single-step ``.loc[...] = ...`` mutations.
    """
    cols = [str(c) for c in feature_names]
    col_set = set(cols)

    if condition == "C2_physics_normalized_strict":
        leaked = _LENS_DEPENDENT_PHYSICAL_SWAP_COLUMNS.intersection(col_set)
        if leaked:
            raise RuntimeError(
                "C2_physics_normalized_strict must contain zero lens-dependent columns, "
                f"found {sorted(leaked)}"
            )

    if df.empty or not fold_pipelines:
        return {
            "n_evaluated": 0,
            "focal_only": _summarize_shifts(np.asarray([], dtype=np.float64)),
            "physical_lens_swap": _summarize_shifts(np.asarray([], dtype=np.float64)),
        }

    folds_arr = df["fold"].to_numpy(dtype=np.int64)
    lens_all = df["lens_class"].astype(str).to_numpy()
    valid_lens_mask = (lens_all == "wide") | (lens_all == "ultrawide")

    focal_shifts_parts: list[np.ndarray] = []
    phys_shifts_parts: list[np.ndarray] = []

    for fold_idx in sorted(int(k) for k in fold_pipelines):
        va_mask = (folds_arr == fold_idx) & valid_lens_mask
        if not np.any(va_mask):
            continue

        df_val = df.loc[va_mask].reset_index(drop=True)
        X_val = df_val.loc[:, cols].copy()
        pipe = fold_pipelines[fold_idx]
        y_orig = np.asarray(pipe.predict(X_val), dtype=np.float64)
        denom = np.maximum(y_orig, 1e-6)

        val_lens = df_val["lens_class"].astype(str).to_numpy()
        is_wide = val_lens == "wide"
        is_uw = val_lens == "ultrawide"

        # Mode 1: focal_only (flip focal_length_mm 4.25 <-> 1.65 keeping all else fixed).
        if "focal_length_mm" in col_set:
            X_cf_focal = X_val.copy()
            X_cf_focal.loc[is_wide, "focal_length_mm"] = _UW_FOCAL_MM
            X_cf_focal.loc[is_uw, "focal_length_mm"] = _WIDE_FOCAL_MM
            y_cf_focal = np.asarray(pipe.predict(X_cf_focal), dtype=np.float64)
            shift_focal = (np.abs(y_cf_focal - y_orig) / denom) * 100.0
        else:
            shift_focal = np.zeros(len(df_val), dtype=np.float64)

        # Mode 2: physical_lens_swap (hold area_optical_norm + dimensionless + chroma constant,
        # swap Redmi 10 4.25/1.0 <-> 1.65/0.6 and sensor area ratio 1.5625 across lens-dependent
        # features consumed by this ablation condition).
        if condition == "C2_physics_normalized_strict":
            shift_phys = np.zeros(len(df_val), dtype=np.float64)
        else:
            X_cf_phys = X_val.copy()
            f_old = df_val["focal_length_mm"].to_numpy(dtype=np.float64)
            z_old = df_val["zoom_multiple"].to_numpy(dtype=np.float64)
            f_new = np.where(is_wide, _UW_FOCAL_MM, _WIDE_FOCAL_MM)
            z_new = np.where(is_wide, _UW_ZOOM, _WIDE_ZOOM)
            sensor_ratio = np.where(
                is_wide,
                1.0 / WIDE_OVER_UW_SENSOR_AREA_RATIO,
                WIDE_OVER_UW_SENSOR_AREA_RATIO,
            )
            z_ratio = z_new / z_old
            f_ratio = f_old / f_new

            if "focal_length_mm" in col_set:
                X_cf_phys.loc[:, "focal_length_mm"] = f_new
            if "zoom_multiple" in col_set:
                X_cf_phys.loc[:, "zoom_multiple"] = z_new

            for area_col in ("area_px", "convex_area_px"):
                if area_col in col_set:
                    X_cf_phys.loc[:, area_col] = X_cf_phys[area_col].to_numpy(dtype=np.float64) * (
                        z_ratio ** 2
                    )

            for lin_col in ("perimeter_px", "major_axis_px", "minor_axis_px"):
                if lin_col in col_set:
                    X_cf_phys.loc[:, lin_col] = (
                        X_cf_phys[lin_col].to_numpy(dtype=np.float64) * z_ratio
                    )

            if "area_frac" in col_set:
                X_cf_phys.loc[:, "area_frac"] = X_cf_phys["area_frac"].to_numpy(
                    dtype=np.float64
                ) * ((z_ratio ** 2) / sensor_ratio)

            if "area_frac_optical_norm" in col_set:
                X_cf_phys.loc[:, "area_frac_optical_norm"] = X_cf_phys[
                    "area_frac_optical_norm"
                ].to_numpy(dtype=np.float64) * (1.0 / sensor_ratio)

            if "area_naive_fxz" in col_set:
                X_cf_phys.loc[:, "area_naive_fxz"] = X_cf_phys["area_naive_fxz"].to_numpy(
                    dtype=np.float64
                ) * (f_ratio ** 2)

            for naive_lin_col in (
                "sqrt_area_naive_fxz",
                "perim_naive_fxz",
                "major_naive_fxz",
                "minor_naive_fxz",
            ):
                if naive_lin_col in col_set:
                    X_cf_phys.loc[:, naive_lin_col] = (
                        X_cf_phys[naive_lin_col].to_numpy(dtype=np.float64) * f_ratio
                    )

            y_cf_phys = np.asarray(pipe.predict(X_cf_phys), dtype=np.float64)
            shift_phys = (np.abs(y_cf_phys - y_orig) / denom) * 100.0

        focal_shifts_parts.append(shift_focal)
        phys_shifts_parts.append(shift_phys)

    if not focal_shifts_parts:
        return {
            "n_evaluated": 0,
            "focal_only": _summarize_shifts(np.asarray([], dtype=np.float64)),
            "physical_lens_swap": _summarize_shifts(np.asarray([], dtype=np.float64)),
        }

    focal_all = np.concatenate(focal_shifts_parts)
    phys_all = np.concatenate(phys_shifts_parts)
    return {
        "n_evaluated": int(focal_all.size),
        "focal_only": _summarize_shifts(focal_all),
        "physical_lens_swap": _summarize_shifts(phys_all),
    }


def evaluate_anti_confounding_gate(
    condition: str,
    counterfactual: Mapping[str, Any],
    by_band_and_lens: Mapping[str, Mapping[str, Any]],
    c2_by_band_and_lens: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[bool, list[str]]:
    """Evaluate D-05 anti-confounding gate returning ``(lens_confounding_detected, reasons)``."""
    focal_info = counterfactual.get("focal_only", {})
    phys_info = counterfactual.get("physical_lens_swap", {})
    n_eval = int(counterfactual.get("n_evaluated", 0) or 0)

    if condition == "C2_physics_normalized_strict":
        if n_eval > 0:
            focal_med = focal_info.get("median_shift_pct")
            phys_med = phys_info.get("median_shift_pct")
            if focal_med != 0.0 or phys_med != 0.0:
                raise RuntimeError(
                    "C2_physics_normalized_strict violated strict 0.0% lens invariance: "
                    f"focal_med={focal_med!r}, phys_med={phys_med!r}"
                )
        return False, []

    if condition == "C1_uncalibrated_pdi":
        # Baseline without EXIF features: zero EXIF confounding shortcut (physical zoom
        # sensitivity is still logged in counterfactual_lens_flip for ablation transparency).
        return False, []

    reasons: list[str] = []

    focal_med = focal_info.get("median_shift_pct")
    if focal_med is not None and float(focal_med) > COUNTERFACTUAL_FOCAL_MEDIAN_THRESHOLD_PCT:
        reasons.append(
            f"focal_only median shift {float(focal_med):.2f}% > "
            f"{COUNTERFACTUAL_FOCAL_MEDIAN_THRESHOLD_PCT:.1f}%"
        )

    phys_med = phys_info.get("median_shift_pct")
    if phys_med is not None and float(phys_med) > COUNTERFACTUAL_PHYS_MEDIAN_THRESHOLD_PCT:
        reasons.append(
            f"physical_lens_swap median shift {float(phys_med):.2f}% > "
            f"{COUNTERFACTUAL_PHYS_MEDIAN_THRESHOLD_PCT:.1f}%"
        )

    phys_p90 = phys_info.get("p90_shift_pct")
    if phys_p90 is not None and float(phys_p90) > COUNTERFACTUAL_PHYS_P90_THRESHOLD_PCT:
        reasons.append(
            f"physical_lens_swap p90 shift {float(phys_p90):.2f}% > "
            f"{COUNTERFACTUAL_PHYS_P90_THRESHOLD_PCT:.1f}%"
        )

    if c2_by_band_and_lens is not None:
        slice_key = "<500g__ultrawide"
        cand_slice = by_band_and_lens.get(slice_key, {})
        c2_slice = c2_by_band_and_lens.get(slice_key, {})
        cand_n = int(cand_slice.get("n", 0) or 0)
        c2_n = int(c2_slice.get("n", 0) or 0)
        cand_mae = cand_slice.get("mae_g")
        cand_mbe = cand_slice.get("mbe_g")
        c2_mae = c2_slice.get("mae_g")
        c2_mbe = c2_slice.get("mbe_g")
        if (
            cand_n > 0
            and c2_n > 0
            and cand_mae is not None
            and cand_mbe is not None
            and c2_mae is not None
            and c2_mbe is not None
            and float(cand_mae) > float(c2_mae)
            and abs(float(cand_mbe)) > abs(float(c2_mbe))
        ):
            reasons.append(
                f"<500g ultrawide degraded vs C2 (mae {float(cand_mae):.2f}g > "
                f"{float(c2_mae):.2f}g and |mbe| {abs(float(cand_mbe)):.2f}g > "
                f"{abs(float(c2_mbe)):.2f}g)"
            )

    return len(reasons) > 0, reasons


def evaluate_single_candidate(
    dataset: LoadedModelingDataset,
    model_name: str,
    condition: str,
    *,
    use_sample_weight: bool = True,
    use_smearing: bool = False,
    use_allometric_log: bool = True,
    n_jobs: int = 4,
    c2_by_band_and_lens: Mapping[str, Mapping[str, Any]] | None = None,
) -> CandidateEvaluation:
    """Run strictly leak-free 5-fold OOF CV for one ``(model_name, condition, mechanics)`` setup."""
    if model_name not in MODEL_FAMILY_NAMES:
        raise ValueError(
            f"Unknown model_name {model_name!r}; expected one of {MODEL_FAMILY_NAMES!r}."
        )
    if condition not in dataset.ablation_columns:
        raise ValueError(
            f"Unknown ablation condition {condition!r}; expected one of "
            f"{tuple(dataset.ablation_columns.keys())!r}."
        )

    df = dataset.df
    feature_names = tuple(dataset.ablation_columns[condition])
    col_list = list(feature_names)
    folds_arr = df["fold"].to_numpy(dtype=np.int64)
    y_all = df["weight_g"].to_numpy(dtype=np.float64)

    oof_pred = np.full(len(df), np.nan, dtype=np.float64)
    fold_pipelines: dict[int, Pipeline] = {}

    for fold_idx in range(_N_FOLDS):
        tr_mask = folds_arr != fold_idx
        va_mask = folds_arr == fold_idx
        if not np.any(tr_mask) or not np.any(va_mask):
            raise ValueError(f"Fold {fold_idx} has an empty train or validation split.")

        X_tr = df.loc[tr_mask, col_list]
        y_tr = y_all[tr_mask]
        X_va = df.loc[va_mask, col_list]

        pipe = build_regressor_pipeline(
            model_name,
            use_smearing=use_smearing,
            n_jobs=n_jobs,
            use_allometric_log=use_allometric_log,
        )
        if use_sample_weight:
            sw = compute_sample_weights(
                df.loc[tr_mask, "session_id"],
                df.loc[tr_mask, "weight_band"],
            )
            pipe.fit(X_tr, y_tr, reg__sample_weight=sw)
        else:
            pipe.fit(X_tr, y_tr)

        va_pred = np.asarray(pipe.predict(X_va), dtype=np.float64)
        expected_n = int(np.count_nonzero(va_mask))
        if (
            va_pred.shape != (expected_n,)
            or not np.all(np.isfinite(va_pred))
            or np.any(va_pred <= 0.0)
        ):
            raise ValueError(
                f"Invalid OOF predictions for {model_name}/{condition} fold {fold_idx}."
            )

        oof_pred[va_mask] = va_pred
        fold_pipelines[fold_idx] = pipe

    if not np.all(np.isfinite(oof_pred)) or np.any(oof_pred <= 0.0):
        raise ValueError(
            f"Incomplete or non-positive OOF prediction vector for {model_name}/{condition}."
        )

    global_metrics = compute_regression_metrics(y_all, oof_pred)
    session_level_metrics = compute_session_level_metrics(df, oof_pred)
    global_metrics["session_averaged_r2"] = session_level_metrics["session_averaged_r2"]

    strat = compute_stratified_metrics(df, oof_pred)
    balanced_mape = strat["balanced_band_mape_pct"]
    if balanced_mape is None:
        raise ValueError(
            f"Unable to compute balanced_band_mape_pct for {model_name}/{condition}: empty bands."
        )

    dual_lens = compute_within_session_dual_lens_consistency(df, oof_pred)
    counterfactual = compute_counterfactual_lens_flip(df, condition, feature_names, fold_pipelines)
    confounded, reasons = evaluate_anti_confounding_gate(
        condition,
        counterfactual,
        strat["by_band_and_lens"],
        c2_by_band_and_lens,
    )

    weighting_mode = DEFAULT_WEIGHTING_MODE if use_sample_weight else "unweighted"
    return CandidateEvaluation(
        model_name=model_name,
        ablation_condition=condition,
        feature_names=feature_names,
        weighting_mode=weighting_mode,
        duan_smearing_enabled=bool(use_smearing),
        oof_predictions=oof_pred,
        fold_pipelines=fold_pipelines,
        global_metrics=global_metrics,
        session_level_metrics=session_level_metrics,
        balanced_band_mape_pct=float(balanced_mape),
        per_fold=strat["per_fold"],
        by_weight_band=strat["by_weight_band"],
        by_lens=strat["by_lens"],
        by_scene=strat["by_scene"],
        by_band_and_scene=strat["by_band_and_scene"],
        by_band_and_lens=strat["by_band_and_lens"],
        band_error_intervals=strat["band_error_intervals"],
        dual_lens_consistency=dual_lens,
        counterfactual_lens_flip=counterfactual,
        lens_confounding_detected=bool(confounded),
        confounding_reasons=reasons,
        eligible_for_production=not bool(confounded),
        is_winner=False,
    )


def _winner_sort_key(cand: CandidateEvaluation) -> tuple[float, float, float, float, str, str]:
    global_mape = cand.global_metrics.get("mape_pct")
    global_mae = cand.global_metrics.get("mae_g")
    dl_med = cand.dual_lens_consistency.get("median_abs_diff_g")
    return (
        float(cand.balanced_band_mape_pct),
        float(global_mape) if global_mape is not None else math.inf,
        float(global_mae) if global_mae is not None else math.inf,
        float(dl_med) if dl_med is not None else math.inf,
        str(cand.ablation_condition),
        str(cand.model_name),
    )


def select_winning_candidate(candidates: Sequence[CandidateEvaluation]) -> CandidateEvaluation:
    """Deterministically select winning gate-passing candidate per D-06 total order."""
    eligible = [
        cand
        for cand in candidates
        if cand.eligible_for_production and not cand.lens_confounding_detected
    ]
    if not eligible:
        raise RuntimeError("No production-eligible candidates passed the anti-confounding gate.")
    return min(eligible, key=_winner_sort_key)


def _average_fold_feature_importances(
    fold_pipelines: Mapping[int, Pipeline],
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Average fold-level feature importances across all 5 folds with deterministic tie-breaks."""
    ordered_names = [str(name) for name in feature_names]
    pct_sums: dict[str, float] = {name: 0.0 for name in ordered_names}
    raw_sums: dict[str, float] = {name: 0.0 for name in ordered_names}
    n_folds = len(fold_pipelines)
    if n_folds == 0:
        raise ValueError("Cannot extract feature importances from empty fold_pipelines.")

    for fold_idx in sorted(fold_pipelines):
        fold_items = extract_feature_importances(fold_pipelines[fold_idx], ordered_names)
        for item in fold_items:
            feat = str(item["feature"])
            pct_sums[feat] += float(item["importance_pct"])
            raw_sums[feat] += float(item["raw_importance"])

    mean_pcts = np.asarray([pct_sums[name] / n_folds for name in ordered_names], dtype=np.float64)
    total_pct = float(np.sum(mean_pcts))
    norm_pcts = (mean_pcts / total_pct) * 100.0 if total_pct > 0.0 else np.zeros_like(mean_pcts)

    items: list[dict[str, Any]] = [
        {
            "feature": name,
            "importance_pct": _round4(float(pct)),
            "raw_importance": _round6(raw_sums[name] / n_folds),
        }
        for name, pct in zip(ordered_names, norm_pcts, strict=True)
    ]
    items.sort(key=lambda item: (-float(item["importance_pct"]), str(item["feature"])))
    return items


def _build_oof_scatter_points(
    df: pd.DataFrame,
    oof_pred: np.ndarray,
) -> list[dict[str, Any]]:
    """Build anonymized OOF scatter points sorted stably by opaque ``image_id`` (D-09)."""
    image_ids = df["image_id"].astype(str).to_numpy()
    session_ids = df["session_id"].astype(str).to_numpy()
    folds = df["fold"].to_numpy(dtype=np.int64)
    weights = df["weight_g"].to_numpy(dtype=np.float64)
    lenses = df["lens_class"].astype(str).to_numpy()
    scenes = df["scene_type"].astype(str).to_numpy()

    points: list[dict[str, Any]] = []
    for idx in range(len(df)):
        actual = float(weights[idx])
        pred = float(oof_pred[idx])
        abs_err = abs(pred - actual)
        pct_err = (abs_err / max(actual, 1e-6)) * 100.0
        actual_out: int | float = int(actual) if actual.is_integer() else _round4(actual)
        points.append(
            {
                "image_id": str(image_ids[idx]),
                "session_id": str(session_ids[idx]),
                "fold": int(folds[idx]),
                "actual_g": actual_out,
                "predicted_g": _round4(pred),
                "abs_err_g": _round4(abs_err),
                "pct_err": _round4(pct_err),
                "weight_band": _weight_band_for_value(actual),
                "lens_class": str(lenses[idx]),
                "scene_type": str(scenes[idx]),
            }
        )

    points.sort(key=lambda item: str(item["image_id"]))
    return points


def _build_optical_ablation_summary(
    candidates: Sequence[CandidateEvaluation],
    winner: CandidateEvaluation,
) -> dict[str, Any]:
    """Assemble the 4-condition optical ablation table, family means, and deltas vs. C1."""
    by_cond_and_model: dict[tuple[str, str], CandidateEvaluation] = {
        (cand.ablation_condition, cand.model_name): cand for cand in candidates
    }
    winner_family_by_cond: dict[str, CandidateEvaluation] = {
        cond: by_cond_and_model[(cond, winner.model_name)] for cond in ABLATION_CONDITION_ORDER
    }
    c1_ref = winner_family_by_cond["C1_uncalibrated_pdi"]

    conditions_summary: dict[str, dict[str, Any]] = {}
    dual_lens_summary: dict[str, dict[str, Any]] = {}
    counterfactual_summary: dict[str, dict[str, Any]] = {}
    for cond in ABLATION_CONDITION_ORDER:
        cand = winner_family_by_cond[cond]
        lt500_uw = dict(cand.by_band_and_lens.get("<500g__ultrawide", {}))
        conditions_summary[cond] = {
            "n_features": int(len(cand.feature_names)),
            "global": dict(cand.global_metrics),
            "session_level": dict(cand.session_level_metrics),
            "balanced_band_mape_pct": cand.balanced_band_mape_pct,
            "by_weight_band": {b: dict(m) for b, m in cand.by_weight_band.items()},
            "lt500_ultrawide": lt500_uw,
            "dual_lens_consistency": dict(cand.dual_lens_consistency),
            "counterfactual_lens_flip": dict(cand.counterfactual_lens_flip),
            "lens_confounding_detected": bool(cand.lens_confounding_detected),
            "confounding_reasons": list(cand.confounding_reasons),
            "eligible_for_production": bool(cand.eligible_for_production),
        }
        dual_lens_summary[cond] = dict(cand.dual_lens_consistency)
        counterfactual_summary[cond] = {
            **dict(cand.counterfactual_lens_flip),
            "lt500_ultrawide_mae_g": lt500_uw.get("mae_g"),
            "lt500_ultrawide_mbe_g": lt500_uw.get("mbe_g"),
            "lens_confounding_detected": bool(cand.lens_confounding_detected),
            "confounding_reasons": list(cand.confounding_reasons),
        }

    family_means_by_condition: dict[str, dict[str, Any]] = {}
    for cond in ABLATION_CONDITION_ORDER:
        cond_cands = [cand for cand in candidates if cand.ablation_condition == cond]
        r2_vals = [
            float(c.global_metrics["r2"])
            for c in cond_cands
            if c.global_metrics["r2"] is not None
        ]
        family_means_by_condition[cond] = {
            "n_models": int(len(cond_cands)),
            "mean_mae_g": _round4(
                float(np.mean([float(c.global_metrics["mae_g"]) for c in cond_cands]))
            ),
            "mean_rmse_g": _round4(
                float(np.mean([float(c.global_metrics["rmse_g"]) for c in cond_cands]))
            ),
            "mean_mape_pct": _round4(
                float(np.mean([float(c.global_metrics["mape_pct"]) for c in cond_cands]))
            ),
            "mean_balanced_band_mape_pct": _round4(
                float(np.mean([float(c.balanced_band_mape_pct) for c in cond_cands]))
            ),
            "mean_r2": (
                _round4(float(np.mean(r2_vals)))
                if len(r2_vals) == len(cond_cands)
                else None
            ),
        }

    deltas_vs_c1: dict[str, dict[str, float | None]] = {}
    for cond in ABLATION_CONDITION_ORDER:
        if cond == "C1_uncalibrated_pdi":
            continue
        cand = winner_family_by_cond[cond]
        cand_r2 = cand.global_metrics["r2"]
        c1_r2 = c1_ref.global_metrics["r2"]
        cand_sess_r2 = cand.session_level_metrics["session_averaged_r2"]
        c1_sess_r2 = c1_ref.session_level_metrics["session_averaged_r2"]
        deltas_vs_c1[cond] = {
            "delta_mae_g": _round4(
                float(cand.global_metrics["mae_g"]) - float(c1_ref.global_metrics["mae_g"])
            ),
            "delta_rmse_g": _round4(
                float(cand.global_metrics["rmse_g"]) - float(c1_ref.global_metrics["rmse_g"])
            ),
            "delta_mape_pct": _round4(
                float(cand.global_metrics["mape_pct"]) - float(c1_ref.global_metrics["mape_pct"])
            ),
            "delta_balanced_band_mape_pct": _round4(
                float(cand.balanced_band_mape_pct) - float(c1_ref.balanced_band_mape_pct)
            ),
            "delta_r2": (
                _round4(float(cand_r2) - float(c1_r2))
                if cand_r2 is not None and c1_r2 is not None
                else None
            ),
            "delta_session_averaged_r2": (
                _round4(float(cand_sess_r2) - float(c1_sess_r2))
                if cand_sess_r2 is not None and c1_sess_r2 is not None
                else None
            ),
        }

    return {
        "reference_model_name": winner.model_name,
        "conditions": conditions_summary,
        "family_means_by_condition": family_means_by_condition,
        "deltas_vs_c1": deltas_vs_c1,
        "dual_lens_consistency": dual_lens_summary,
        "counterfactual_lens_flip": counterfactual_summary,
    }


def _mean_smearing_factor(fold_pipelines: Mapping[int, Pipeline]) -> float:
    factors = [
        float(fold_pipelines[fold_idx].named_steps["reg"].smearing_factor_)
        for fold_idx in sorted(fold_pipelines)
    ]
    return _round6(float(np.mean(factors)))


def run_full_evaluation(
    dataset: LoadedModelingDataset,
    *,
    n_jobs: int = 4,
) -> FullEvaluationResult:
    """Execute complete 4x4 5-fold CV, anti-confounding audit, mechanics sweeps & D-06 selection."""
    # Step 1: Evaluate C2_physics_normalized_strict first across all 4 families so C3/C4
    # anti-confounding gates receive their exact same-model C2 <500g ultrawide baselines.
    c2_by_model: dict[str, CandidateEvaluation] = {}
    for model_name in MODEL_FAMILY_NAMES:
        c2_by_model[model_name] = evaluate_single_candidate(
            dataset,
            model_name,
            "C2_physics_normalized_strict",
            use_sample_weight=True,
            use_smearing=False,
            use_allometric_log=True,
            n_jobs=n_jobs,
        )

    # Step 2: Evaluate all remaining ablation conditions and store all 16 candidates
    # in canonical (ablation_condition, model_name) order.
    evaluated_map: dict[tuple[str, str], CandidateEvaluation] = {
        ("C2_physics_normalized_strict", model_name): cand
        for model_name, cand in c2_by_model.items()
    }
    for condition in ABLATION_CONDITION_ORDER:
        if condition == "C2_physics_normalized_strict":
            continue
        for model_name in MODEL_FAMILY_NAMES:
            evaluated_map[(condition, model_name)] = evaluate_single_candidate(
                dataset,
                model_name,
                condition,
                use_sample_weight=True,
                use_smearing=False,
                use_allometric_log=True,
                n_jobs=n_jobs,
                c2_by_band_and_lens=c2_by_model[model_name].by_band_and_lens,
            )

    raw_candidates = [
        evaluated_map[(condition, model_name)]
        for condition in ABLATION_CONDITION_ORDER
        for model_name in MODEL_FAMILY_NAMES
    ]

    # Step 3: Select production winner deterministically per D-06.
    raw_winner = select_winning_candidate(raw_candidates)
    candidates: list[CandidateEvaluation] = [
        replace(cand, is_winner=(cand is raw_winner)) for cand in raw_candidates
    ]
    winner = next(cand for cand in candidates if cand.is_winner)

    # Step 4: 2x2 weighting x Duan's smearing sensitivity on (winner.model_name, C2).
    mechanics_specs: tuple[tuple[bool, bool], ...] = (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    )
    mechanics_cands: list[CandidateEvaluation] = []
    for use_sw, use_smear in mechanics_specs:
        if use_sw and not use_smear:
            mechanics_cands.append(c2_by_model[winner.model_name])
        else:
            mechanics_cands.append(
                evaluate_single_candidate(
                    dataset,
                    winner.model_name,
                    "C2_physics_normalized_strict",
                    use_sample_weight=use_sw,
                    use_smearing=use_smear,
                    use_allometric_log=True,
                    n_jobs=n_jobs,
                )
            )

    best_mech = select_winning_candidate(mechanics_cands)
    mechanics_ablation: list[dict[str, Any]] = [
        {
            "model_name": cand.model_name,
            "ablation_condition": cand.ablation_condition,
            "weighting_mode": cand.weighting_mode,
            "duan_smearing_enabled": bool(cand.duan_smearing_enabled),
            "mean_smearing_factor": _mean_smearing_factor(cand.fold_pipelines),
            "balanced_band_mape_pct": cand.balanced_band_mape_pct,
            "global": dict(cand.global_metrics),
            "session_level": dict(cand.session_level_metrics),
            "by_weight_band": {b: dict(m) for b, m in cand.by_weight_band.items()},
            "is_best_by_bal_mape": bool(cand is best_mech),
            "is_selected": bool(
                cand.weighting_mode == winner.weighting_mode
                and cand.duan_smearing_enabled == winner.duan_smearing_enabled
            ),
        }
        for cand in mechanics_cands
    ]

    # Step 5: Linear AllometricLogTransformer ON vs OFF diagnostic on C2 (RidgeLog, HuberLog).
    allometric_log_ablation: list[dict[str, Any]] = []
    for lin_model in ("RidgeLog", "HuberLog"):
        cand_on = c2_by_model[lin_model]
        cand_off = evaluate_single_candidate(
            dataset,
            lin_model,
            "C2_physics_normalized_strict",
            use_sample_weight=True,
            use_smearing=False,
            use_allometric_log=False,
            n_jobs=n_jobs,
        )
        for use_allo, cand in ((True, cand_on), (False, cand_off)):
            allometric_log_ablation.append(
                {
                    "model_name": cand.model_name,
                    "ablation_condition": cand.ablation_condition,
                    "use_allometric_log": bool(use_allo),
                    "weighting_mode": cand.weighting_mode,
                    "duan_smearing_enabled": bool(cand.duan_smearing_enabled),
                    "balanced_band_mape_pct": cand.balanced_band_mape_pct,
                    "global": dict(cand.global_metrics),
                    "session_level": dict(cand.session_level_metrics),
                    "by_weight_band": {b: dict(m) for b, m in cand.by_weight_band.items()},
                }
            )

    optical_ablation = _build_optical_ablation_summary(candidates, winner)
    feature_importances = _average_fold_feature_importances(
        winner.fold_pipelines,
        winner.feature_names,
    )
    oof_scatter_points = _build_oof_scatter_points(dataset.df, winner.oof_predictions)

    return FullEvaluationResult(
        candidates=candidates,
        winner=winner,
        mechanics_ablation=mechanics_ablation,
        allometric_log_ablation=allometric_log_ablation,
        optical_ablation=optical_ablation,
        feature_importances=feature_importances,
        oof_scatter_points=oof_scatter_points,
    )


__all__ = [
    "WEIGHT_BAND_ORDER",
    "LENS_CLASS_ORDER",
    "SCENE_TYPE_ORDER",
    "COUNTERFACTUAL_FOCAL_MEDIAN_THRESHOLD_PCT",
    "COUNTERFACTUAL_PHYS_MEDIAN_THRESHOLD_PCT",
    "COUNTERFACTUAL_PHYS_P90_THRESHOLD_PCT",
    "WIDE_OVER_UW_SENSOR_AREA_RATIO",
    "LITERATURE_R2_REFERENCE",
    "CandidateEvaluation",
    "FullEvaluationResult",
    "compute_regression_metrics",
    "compute_session_level_metrics",
    "compute_stratified_metrics",
    "compute_within_session_dual_lens_consistency",
    "compute_counterfactual_lens_flip",
    "evaluate_anti_confounding_gate",
    "evaluate_single_candidate",
    "select_winning_candidate",
    "run_full_evaluation",
]
