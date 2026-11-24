"""Atomic production model bundling, evaluation reporting & opaque-ID preset publishing.

Implements Phase 3 persistence contracts (D-07, D-09, D-10, D-11):
- Dual-bundle ``models/weight_regressor.joblib`` packing the refitted 100%-clean
  ``production_pipeline``, all 5 held-out ``fold_pipelines`` (``{0..4}``), locked
  ``PDIConfig`` + 16-hex SHA-256 hash, exact ordered ``feature_names`` tuple,
  per-band error intervals, and Cobb Avian 48 training coverage bounds.
- Byte-deterministic, zero-path-leakage ``models/evaluation_report.json`` containing
  dataset summary, literature benchmark, selection criteria, mechanics & allometric-log
  ablations, 16-row model comparison, 4-condition optical ablation + dual-lens and
  counterfactual audits, full winning-model diagnostics (including ``session_averaged_r2``),
  and anonymized ``oof_scatter_points`` sorted stably by ``image_id``.
- Byte-deterministic, zero-path-leakage ``models/presets_catalog.json`` storing ONLY
  opaque ``preset_id`` + ``image_id`` (PROIBIÇÃO 4 & ASVS V4) across 6 non-fallback
  median-tier single-bird OOF presets spanning all 5 folds + 4 multi-bird Watershed
  presets verified via non-symlink ``RAW_DATASET_DIR.resolve()`` containment and
  ``run_watershed_pdi(..., return_visuals=False)``.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.ingestion.decoding import MULTI_BIRD_FOLDERS, assign_weight_band
from backend.app.modeling.age_stage import (
    VALID_AGE_DAYS_RANGE,
    VALID_TRAINING_WEIGHT_MAX_G,
    VALID_TRAINING_WEIGHT_MIN_G,
)
from backend.app.modeling.datasets import ABLATION_CONDITION_ORDER, LoadedModelingDataset
from backend.app.modeling.evaluation import (
    COUNTERFACTUAL_FOCAL_MEDIAN_THRESHOLD_PCT,
    COUNTERFACTUAL_PHYS_MEDIAN_THRESHOLD_PCT,
    COUNTERFACTUAL_PHYS_P90_THRESHOLD_PCT,
    LITERATURE_R2_REFERENCE,
    WEIGHT_BAND_ORDER,
    CandidateEvaluation,
    FullEvaluationResult,
)
from backend.app.modeling.models import (
    DEFAULT_WEIGHTING_MODE,
    MODEL_FAMILY_NAMES,
    build_regressor_pipeline,
    compute_sample_weights,
)
from backend.app.pdi.config import PDIConfig
from backend.app.pdi.watershed import run_watershed_pdi

MODEL_BUNDLE_SCHEMA_VERSION: str = "1.0.0"
EVALUATION_REPORT_SCHEMA_VERSION: str = "1.0.0"
PRESETS_CATALOG_SCHEMA_VERSION: str = "1.0.0"
MODEL_VERSION: str = "avianscale-cv-v1.0.0"

_FORBIDDEN_LEAK_SUBSTRINGS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    "rel_path",
    "/home/",
    "data/raw",
)

WATERSHED_FOLDER_SPECS: tuple[tuple[str, int, int], ...] = (
    ("417", 2, 417),
    ("757", 2, 757),
    ("823", 4, 823),
    ("1520", 2, 1520),
)

if {folder_id for folder_id, _, _ in WATERSHED_FOLDER_SPECS} != MULTI_BIRD_FOLDERS:
    raise RuntimeError(
        "WATERSHED_FOLDER_SPECS folder IDs diverged from MULTI_BIRD_FOLDERS: "
        f"{sorted(fid for fid, _, _ in WATERSHED_FOLDER_SPECS)} != {sorted(MULTI_BIRD_FOLDERS)}"
    )

REQUIRED_BUNDLE_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "model_version",
        "pdi_config",
        "pdi_config_hash",
        "winning_model_name",
        "winning_ablation_condition",
        "feature_names",
        "weighting_mode",
        "duan_smearing_enabled",
        "production_pipeline",
        "fold_pipelines",
        "band_error_intervals",
        "valid_weight_range_g",
        "valid_age_days_range",
    }
)

_EXPECTED_FOLDS: frozenset[int] = frozenset({0, 1, 2, 3, 4})
_EXPECTED_WEIGHT_BANDS: frozenset[str] = frozenset(WEIGHT_BAND_ORDER)
_REQUIRED_BAND_INTERVAL_KEYS: frozenset[str] = frozenset(
    {
        "n",
        "mae_g",
        "mape_pct",
        "median_abs_err_g",
        "p68_abs_err_g",
        "p90_abs_err_g",
        "mbe_g",
    }
)
_HEX_LOWER_CHARS: frozenset[str] = frozenset("0123456789abcdef")

_MIN_MEDIAN_TIER_PCT_ERR: float = 3.0

_REQUIRED_QUARANTINE_COLUMNS: tuple[str, ...] = (
    "image_id",
    "rel_path",
    "folder_chain",
    "quarantine_reason",
    "lens_class",
)

_WATERSHED_GROUP_SUM_DISCLAIMER: str = (
    "Pesagem de grupo na base pública: apenas o peso total conjunto das aves foi "
    "registrado; estimativas individuais por ave obtidas via Watershed são ilustrativas "
    "e não possuem gabarito individual."
)

_LITERATURE_BENCHMARK_NOTE: str = (
    "Fixed ceiling-camera controlled-height benchmarks (constant Z); handheld "
    "uncalibrated-Z single-frame vs. session-averaged context."
)

_SINGLE_BIRD_SLOT_ORDER: tuple[str, ...] = (
    "sb_young_chick",
    "sb_intermediate_grower",
    "sb_heavy_broiler",
    "sb_hard_white_bucket",
    "dual_lens_pair",
)


def _round4(value: float) -> float:
    rounded = round(float(value), 4)
    return 0.0 if rounded == 0.0 else rounded


def _json_normalize(value: Any) -> Any:
    """Convert NumPy scalars, tuples, and dataclasses to strict RFC 8259 JSON-native values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_normalize(asdict(value))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_json_normalize(v) for v in value.tolist()]
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite float {value!r} cannot be serialized to strict RFC 8259 JSON."
            )
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        return {str(k): _json_normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(v) for v in value]
    return value


def _assert_no_proibicao4_leaks(serialized_text: str, *, context_label: str) -> None:
    """Reject forbidden raw-filename or filesystem-path tokens before touching disk."""
    lowered = serialized_text.lower()
    for token in _FORBIDDEN_LEAK_SUBSTRINGS:
        if token in lowered:
            raise ValueError(
                f"PROIBIÇÃO 4 violation: forbidden token {token!r} detected in {context_label}."
            )


def _serialize_deterministic_json(payload: Mapping[str, Any], *, context_label: str) -> str:
    """Serialize mapping to byte-deterministic RFC 8259 UTF-8 JSON with leak validation."""
    normalized = _json_normalize(payload)
    serialized = (
        json.dumps(
            normalized,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    _assert_no_proibicao4_leaks(serialized, context_label=context_label)
    return serialized


def _atomic_write_text(content: str, path: Path) -> None:
    """Write UTF-8 text atomically via sibling temp file and ``os.replace()`` (D-11)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_joblib(payload: Any, path: Path, *, compress: int = 3) -> None:
    """Write compressed joblib artifact atomically via sibling temp file + ``os.replace()``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        joblib.dump(payload, tmp_path, compress=compress)
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _is_contained_non_symlink(path: Path, canonical_root: Path) -> bool:
    """Return True iff ``path`` is not a symlink and resolves strictly inside ``canonical_root``."""
    if path.is_symlink():
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved.is_relative_to(canonical_root)


def _resolve_safe_image_path(rel_path: Any, canonical_root: Path) -> Path:
    """Resolve a manifest ``rel_path`` under ``canonical_root`` enforcing ASVS V4 containment."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("ASVS V4 path containment violation: empty or non-string rel_path.")
    if "\\" in rel_path or "\x00" in rel_path:
        raise ValueError("ASVS V4 path containment violation: illegal path characters.")

    rel = Path(rel_path)
    if rel.is_absolute():
        raise ValueError("ASVS V4 path containment violation: absolute path rejected.")
    if len(rel.parts) == 0 or ".." in rel.parts:
        raise ValueError("ASVS V4 path containment violation: parent traversal segment rejected.")

    cur = canonical_root
    for part in rel.parts:
        if part in ("", ".", ".."):
            raise ValueError("ASVS V4 path containment violation: invalid path segment.")
        cur = cur / part
        if not _is_contained_non_symlink(cur, canonical_root):
            raise ValueError(
                "ASVS V4 path containment violation: symlink or out-of-root target rejected."
            )

    resolved = cur.resolve()
    if not resolved.is_relative_to(canonical_root):
        raise ValueError("ASVS V4 path containment violation: resolved path escapes root.")
    if not resolved.is_file():
        raise FileNotFoundError(
            "Manifest references an image file that does not exist inside dataset_root."
        )
    return resolved


def validate_model_bundle(
    bundle: Any,
    *,
    expected_pdi_config_hash: str | None = None,
) -> None:
    """Validate D-07 dual-bundle schema, hash lock, feature tuple, and 5 fold pipelines."""
    if not isinstance(bundle, dict):
        raise ValueError(f"Model bundle must be a dict, got {type(bundle).__name__}.")

    missing = sorted(REQUIRED_BUNDLE_KEYS - set(bundle.keys()))
    if missing:
        raise ValueError(f"Model bundle missing required key(s): {missing}")

    schema_ver = bundle["schema_version"]
    if schema_ver != MODEL_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model bundle schema_version {schema_ver!r}; "
            f"expected {MODEL_BUNDLE_SCHEMA_VERSION!r}."
        )

    model_ver = bundle["model_version"]
    if not isinstance(model_ver, str) or not model_ver.strip():
        raise ValueError(f"model_version must be a non-empty string, got {model_ver!r}.")

    pdi_cfg = bundle["pdi_config"]
    if not isinstance(pdi_cfg, dict) or not pdi_cfg:
        raise ValueError("pdi_config must be a non-empty dictionary.")

    cfg_hash = bundle["pdi_config_hash"]
    if (
        not isinstance(cfg_hash, str)
        or len(cfg_hash) != 16
        or not set(cfg_hash).issubset(_HEX_LOWER_CHARS)
    ):
        raise ValueError(
            f"pdi_config_hash must be a 16-char lowercase hex string, got {cfg_hash!r}"
        )
    if expected_pdi_config_hash is not None and cfg_hash != expected_pdi_config_hash:
        raise ValueError(
            f"pdi_config_hash mismatch: bundle has {cfg_hash!r}, "
            f"expected {expected_pdi_config_hash!r}."
        )

    winning_model = bundle["winning_model_name"]
    if not isinstance(winning_model, str) or winning_model not in MODEL_FAMILY_NAMES:
        raise ValueError(
            f"winning_model_name must be one of {MODEL_FAMILY_NAMES!r}, got {winning_model!r}."
        )

    winning_cond = bundle["winning_ablation_condition"]
    if not isinstance(winning_cond, str) or winning_cond not in ABLATION_CONDITION_ORDER:
        raise ValueError(
            f"winning_ablation_condition must be one of {ABLATION_CONDITION_ORDER!r}, "
            f"got {winning_cond!r}."
        )

    feature_names = bundle["feature_names"]
    if not isinstance(feature_names, tuple) or len(feature_names) == 0:
        raise ValueError("feature_names must be a non-empty tuple of strings.")
    if any(not isinstance(col, str) or not col.strip() for col in feature_names):
        raise ValueError("feature_names must contain strictly non-empty string column names.")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names must not contain duplicate column names.")

    weighting_mode = bundle["weighting_mode"]
    if not isinstance(weighting_mode, str) or not weighting_mode.strip():
        raise ValueError(f"weighting_mode must be a non-empty string, got {weighting_mode!r}.")

    if not isinstance(bundle["duan_smearing_enabled"], bool):
        raise ValueError(
            f"duan_smearing_enabled must be a bool, got {bundle['duan_smearing_enabled']!r}."
        )

    prod_pipe = bundle["production_pipeline"]
    if prod_pipe is None or not callable(getattr(prod_pipe, "predict", None)):
        raise ValueError("production_pipeline must expose a callable predict() method.")

    fold_pipelines = bundle["fold_pipelines"]
    if not isinstance(fold_pipelines, dict):
        raise ValueError("fold_pipelines must be a dict mapping fold ints 0..4 to pipelines.")
    if any(isinstance(k, bool) or not isinstance(k, int) for k in fold_pipelines):
        raise ValueError("fold_pipelines keys must be non-bool integers 0..4.")
    if set(fold_pipelines.keys()) != _EXPECTED_FOLDS:
        raise ValueError(
            f"fold_pipelines must contain exact keys {sorted(_EXPECTED_FOLDS)}, "
            f"got {sorted(fold_pipelines.keys())}."
        )
    for fold_idx in sorted(fold_pipelines):
        fpipe = fold_pipelines[fold_idx]
        if fpipe is None or not callable(getattr(fpipe, "predict", None)):
            raise ValueError(
                f"fold_pipelines[{fold_idx}] must expose a callable predict() method."
            )

    band_intervals = bundle["band_error_intervals"]
    if not isinstance(band_intervals, dict) or set(band_intervals.keys()) != _EXPECTED_WEIGHT_BANDS:
        raise ValueError(
            f"band_error_intervals must contain exact keys {list(WEIGHT_BAND_ORDER)}, "
            f"got {sorted(band_intervals.keys()) if isinstance(band_intervals, dict) else None}."
        )
    for band in WEIGHT_BAND_ORDER:
        entry = band_intervals[band]
        if not isinstance(entry, dict) or not _REQUIRED_BAND_INTERVAL_KEYS.issubset(entry.keys()):
            raise ValueError(
                f"band_error_intervals[{band!r}] missing required keys "
                f"{sorted(_REQUIRED_BAND_INTERVAL_KEYS)}."
            )

    weight_range = bundle["valid_weight_range_g"]
    if not isinstance(weight_range, dict) or set(weight_range.keys()) != {"min", "max"}:
        raise ValueError("valid_weight_range_g must be a dict with exact keys {'min', 'max'}.")
    w_min = weight_range["min"]
    w_max = weight_range["max"]
    if (
        isinstance(w_min, bool)
        or isinstance(w_max, bool)
        or not isinstance(w_min, (int, float))
        or not isinstance(w_max, (int, float))
        or not math.isfinite(float(w_min))
        or not math.isfinite(float(w_max))
        or float(w_min) <= 0.0
        or float(w_min) >= float(w_max)
    ):
        raise ValueError(f"Invalid valid_weight_range_g bounds: {weight_range!r}.")

    age_range = bundle["valid_age_days_range"]
    if not isinstance(age_range, str) or not age_range.strip():
        raise ValueError(f"valid_age_days_range must be a non-empty string, got {age_range!r}.")


def build_model_bundle(
    dataset: LoadedModelingDataset,
    eval_result: FullEvaluationResult,
    *,
    n_jobs: int = 4,
) -> dict[str, Any]:
    """Refit winning pipeline on 100% clean data and pack the D-07 dual-bundle dictionary."""
    winner = eval_result.winner
    feature_names = tuple(winner.feature_names)
    col_list = list(feature_names)

    X_all = dataset.df.loc[:, col_list]
    y_all = dataset.df["weight_g"].to_numpy(dtype=np.float64)

    production_pipeline = build_regressor_pipeline(
        winner.model_name,
        use_smearing=winner.duan_smearing_enabled,
        n_jobs=n_jobs,
    )
    if winner.weighting_mode == DEFAULT_WEIGHTING_MODE:
        sw_all = compute_sample_weights(
            dataset.df["session_id"],
            dataset.df["weight_band"],
        )
        production_pipeline.fit(X_all, y_all, reg__sample_weight=sw_all)
    else:
        production_pipeline.fit(X_all, y_all)

    fold_pipelines = {
        int(fold_idx): winner.fold_pipelines[fold_idx]
        for fold_idx in sorted(winner.fold_pipelines)
    }

    bundle: dict[str, Any] = {
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "pdi_config": _json_normalize(asdict(dataset.pdi_config)),
        "pdi_config_hash": str(dataset.pdi_config_hash),
        "winning_model_name": str(winner.model_name),
        "winning_ablation_condition": str(winner.ablation_condition),
        "feature_names": feature_names,
        "weighting_mode": str(winner.weighting_mode),
        "duan_smearing_enabled": bool(winner.duan_smearing_enabled),
        "production_pipeline": production_pipeline,
        "fold_pipelines": fold_pipelines,
        "band_error_intervals": _json_normalize(winner.band_error_intervals),
        "valid_weight_range_g": {
            "min": VALID_TRAINING_WEIGHT_MIN_G,
            "max": VALID_TRAINING_WEIGHT_MAX_G,
        },
        "valid_age_days_range": VALID_AGE_DAYS_RANGE,
    }
    validate_model_bundle(bundle, expected_pdi_config_hash=dataset.pdi_config_hash)
    return bundle


def save_model_bundle(
    bundle: dict[str, Any],
    path: Path,
    *,
    expected_pdi_config_hash: str | None = None,
    compress: int = 3,
) -> None:
    """Validate and atomically serialize the D-07 dual-bundle via compressed joblib."""
    validate_model_bundle(bundle, expected_pdi_config_hash=expected_pdi_config_hash)
    _atomic_write_joblib(bundle, path, compress=compress)


def load_model_bundle(
    path: Path,
    *,
    expected_pdi_config_hash: str | None = None,
) -> dict[str, Any]:
    """Load and strictly validate a serialized D-07 dual-bundle joblib artifact."""
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    bundle = joblib.load(bundle_path)
    validate_model_bundle(bundle, expected_pdi_config_hash=expected_pdi_config_hash)
    return bundle


def _format_candidate_comparison_row(cand: CandidateEvaluation) -> dict[str, Any]:
    return {
        "model_name": str(cand.model_name),
        "ablation_condition": str(cand.ablation_condition),
        "n_features": int(len(cand.feature_names)),
        "weighting_mode": str(cand.weighting_mode),
        "duan_smearing_enabled": bool(cand.duan_smearing_enabled),
        "global": _json_normalize(cand.global_metrics),
        "session_level": _json_normalize(cand.session_level_metrics),
        "balanced_band_mape_pct": _json_normalize(cand.balanced_band_mape_pct),
        "by_weight_band": _json_normalize(cand.by_weight_band),
        "dual_lens_consistency": _json_normalize(cand.dual_lens_consistency),
        "counterfactual_lens_flip": _json_normalize(cand.counterfactual_lens_flip),
        "lens_confounding_detected": bool(cand.lens_confounding_detected),
        "confounding_reasons": list(cand.confounding_reasons),
        "eligible_for_production": bool(cand.eligible_for_production),
        "is_winner": bool(cand.is_winner),
    }


def build_evaluation_report(
    dataset: LoadedModelingDataset,
    eval_result: FullEvaluationResult,
) -> dict[str, Any]:
    """Build byte-deterministic, zero-path-leakage D-09 evaluation report dictionary."""
    df = dataset.df
    winner = eval_result.winner

    if "r2" not in winner.global_metrics or "session_averaged_r2" not in winner.global_metrics:
        raise ValueError(
            "Winner global_metrics must expose both 'r2' and 'session_averaged_r2' (D-09)."
        )

    dataset_summary: dict[str, Any] = {
        "n_clean_images": int(len(df)),
        "n_unique_weights": int(df["weight_g"].nunique()),
        "n_sessions": int(df["session_id"].nunique()),
        "n_folds": int(df["fold"].nunique()),
        "cross_fold_session_overlap": int(dataset.fold_audit["cross_fold_session_overlap"]),
        "n_mixed_lens_sessions": int(
            winner.dual_lens_consistency.get("n_mixed_lens_sessions", 0) or 0
        ),
        "weight_min_g": int(df["weight_g"].min()),
        "weight_max_g": int(df["weight_g"].max()),
        "valid_age_days_range": VALID_AGE_DAYS_RANGE,
    }

    report: dict[str, Any] = {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "pdi_config_hash": str(dataset.pdi_config_hash),
        "dataset_summary": dataset_summary,
        "literature_benchmark": {
            "r2_reference": LITERATURE_R2_REFERENCE,
            "note": _LITERATURE_BENCHMARK_NOTE,
        },
        "selection_criteria": {
            "primary_metric": "balanced_band_mape_pct",
            "tie_breakers": [
                "global_mape_pct",
                "global_mae_g",
                "dual_lens_median_abs_diff_g",
                "ablation_condition",
                "model_name",
            ],
            "anti_confounding_gate": {
                "focal_only_median_threshold_pct": COUNTERFACTUAL_FOCAL_MEDIAN_THRESHOLD_PCT,
                "physical_lens_swap_median_threshold_pct": (
                    COUNTERFACTUAL_PHYS_MEDIAN_THRESHOLD_PCT
                ),
                "physical_lens_swap_p90_threshold_pct": COUNTERFACTUAL_PHYS_P90_THRESHOLD_PCT,
                "lt500_ultrawide_vs_c2_degradation_check": True,
                "strict_inequality": True,
            },
        },
        "mechanics_ablation": _json_normalize(eval_result.mechanics_ablation),
        "allometric_log_ablation": _json_normalize(eval_result.allometric_log_ablation),
        "model_comparison": [
            _format_candidate_comparison_row(cand) for cand in eval_result.candidates
        ],
        "optical_ablation": _json_normalize(eval_result.optical_ablation),
        "winning_model": {
            "model_name": str(winner.model_name),
            "ablation_condition": str(winner.ablation_condition),
            "n_features": int(len(winner.feature_names)),
            "feature_names": list(winner.feature_names),
            "weighting_mode": str(winner.weighting_mode),
            "duan_smearing_enabled": bool(winner.duan_smearing_enabled),
            "global": _json_normalize(winner.global_metrics),
            "session_level": _json_normalize(winner.session_level_metrics),
            "balanced_band_mape_pct": _json_normalize(winner.balanced_band_mape_pct),
            "per_fold": _json_normalize(winner.per_fold),
            "by_weight_band": _json_normalize(winner.by_weight_band),
            "by_lens": _json_normalize(winner.by_lens),
            "by_scene": _json_normalize(winner.by_scene),
            "by_band_and_scene": _json_normalize(winner.by_band_and_scene),
            "by_band_and_lens": _json_normalize(winner.by_band_and_lens),
            "band_error_intervals": _json_normalize(winner.band_error_intervals),
            "dual_lens_consistency": _json_normalize(winner.dual_lens_consistency),
            "counterfactual_lens_flip": _json_normalize(winner.counterfactual_lens_flip),
            "feature_importances": _json_normalize(eval_result.feature_importances),
        },
        "oof_scatter_points": _json_normalize(eval_result.oof_scatter_points),
    }
    return report


def write_evaluation_report(report: Mapping[str, Any], path: Path) -> None:
    """Validate PROIBIÇÃO 4 compliance and atomically write ``evaluation_report.json``."""
    serialized = _serialize_deterministic_json(report, context_label="evaluation_report.json")
    _atomic_write_text(serialized, path)


def _format_single_bird_preset(
    row: Mapping[str, Any],
    *,
    preset_id: str,
    label: str,
    category: str,
    pair_group_id: str | None,
) -> dict[str, Any]:
    weight_val = float(row["weight_g"])
    gt_weight: int | float = int(weight_val) if weight_val.is_integer() else _round4(weight_val)
    return {
        "preset_id": preset_id,
        "label": label,
        "category": category,
        "pair_group_id": pair_group_id,
        "image_id": str(row["image_id"]),
        "session_id": str(row["session_id"]),
        "fold": int(row["fold"]),
        "ground_truth_weight_g": gt_weight,
        "weight_band": str(row["weight_band"]),
        "lens_class": str(row["lens_class"]),
        "scene_type": str(row["scene_type"]),
        "oof_predicted_g": _round4(float(row["oof_predicted_g"])),
        "oof_abs_err_g": _round4(float(row["oof_abs_err_g"])),
        "oof_pct_err": _round4(float(row["oof_pct_err"])),
    }


def _target_median_pct_err(pool: pd.DataFrame) -> float:
    pct_vals = pool["oof_pct_err"].to_numpy(dtype=np.float64)
    median_tier = pct_vals[pct_vals >= _MIN_MEDIAN_TIER_PCT_ERR]
    if median_tier.size > 0:
        return float(np.median(median_tier))
    return float(np.median(pct_vals))


def _rank_single_slot_candidates_by_fold(
    pool: pd.DataFrame,
    *,
    preferred_mask: np.ndarray,
    target_weight_g: float,
    slot_name: str,
) -> dict[int, tuple[tuple[int, int, float, float, str], dict[str, Any]]]:
    """Select the single best deterministic candidate per available fold for one slot."""
    if pool.empty:
        raise ValueError(f"Empty single-bird preset candidate pool for slot {slot_name!r}.")

    pref_pool = pool.loc[preferred_mask] if np.any(preferred_mask) else pool
    target_median = _target_median_pct_err(pref_pool)

    records = pool.to_dict(orient="records")
    pref_flags = preferred_mask.tolist()
    best_by_fold: dict[int, tuple[tuple[int, int, float, float, str], dict[str, Any]]] = {}

    for rec, is_pref in zip(records, pref_flags, strict=True):
        fold_idx = int(rec["fold"])
        pct_err = float(rec["oof_pct_err"])
        weight_g = float(rec["weight_g"])
        image_id = str(rec["image_id"])

        rank_key = (
            0 if is_pref else 1,
            0 if pct_err >= _MIN_MEDIAN_TIER_PCT_ERR else 1,
            round(abs(pct_err - target_median), 6),
            round(abs(weight_g - target_weight_g), 6),
            image_id,
        )
        existing = best_by_fold.get(fold_idx)
        if existing is None or rank_key < existing[0]:
            best_by_fold[fold_idx] = (rank_key, rec)

    return best_by_fold


def _rank_dual_lens_pairs_by_fold(
    clean_non_fallback: pd.DataFrame,
) -> dict[
    int,
    tuple[tuple[int, int, float, float, str, str], tuple[dict[str, Any], dict[str, Any]]],
]:
    """Select the best same-session same-scene Wide + Ultrawide pair per available fold."""
    if clean_non_fallback.empty:
        raise ValueError("Empty single-bird candidate pool for slot 'dual_lens_pair'.")

    # Gather all valid same-session same-scene groups containing both wide and ultrawide frames.
    candidate_groups: list[tuple[int, str, str, pd.DataFrame, pd.DataFrame]] = []
    grouped_indices = clean_non_fallback.groupby(
        ["session_id", "scene_type"],
        sort=True,
    ).indices
    lens_arr = clean_non_fallback["lens_class"].astype(str).to_numpy()

    for (sess_id, scene_type), raw_idx in sorted(grouped_indices.items(), key=lambda kv: kv[0]):
        idx = np.asarray(raw_idx, dtype=np.int64)
        grp_lens = lens_arr[idx]
        w_idx = idx[grp_lens == "wide"]
        u_idx = idx[grp_lens == "ultrawide"]
        if w_idx.size == 0 or u_idx.size == 0:
            continue
        w_df = clean_non_fallback.iloc[w_idx]
        u_df = clean_non_fallback.iloc[u_idx]
        fold_idx = int(w_df["fold"].iloc[0])
        candidate_groups.append((fold_idx, str(sess_id), str(scene_type), w_df, u_df))

    if not candidate_groups:
        raise ValueError(
            "Empty single-bird preset candidate pool for slot 'dual_lens_pair': "
            "no same-session same-scene wide+ultrawide pairs found."
        )

    all_pair_rows = pd.concat(
        [pd.concat([w_df, u_df], ignore_index=True) for _, _, _, w_df, u_df in candidate_groups],
        ignore_index=True,
    )
    target_median = _target_median_pct_err(all_pair_rows)
    target_weight_g = float(np.median(all_pair_rows["weight_g"].to_numpy(dtype=np.float64)))

    best_by_fold: dict[
        int,
        tuple[tuple[int, int, float, float, str, str], tuple[dict[str, Any], dict[str, Any]]],
    ] = {}

    for fold_idx, _sess_id, _scene_type, w_df, u_df in candidate_groups:
        w_best = min(
            w_df.to_dict(orient="records"),
            key=lambda r: (
                0 if float(r["oof_pct_err"]) >= _MIN_MEDIAN_TIER_PCT_ERR else 1,
                round(abs(float(r["oof_pct_err"]) - target_median), 6),
                str(r["image_id"]),
            ),
        )
        u_best = min(
            u_df.to_dict(orient="records"),
            key=lambda r: (
                0 if float(r["oof_pct_err"]) >= _MIN_MEDIAN_TIER_PCT_ERR else 1,
                round(abs(float(r["oof_pct_err"]) - target_median), 6),
                str(r["image_id"]),
            ),
        )
        w_pct = float(w_best["oof_pct_err"])
        u_pct = float(u_best["oof_pct_err"])
        both_median_tier = (
            w_pct >= _MIN_MEDIAN_TIER_PCT_ERR and u_pct >= _MIN_MEDIAN_TIER_PCT_ERR
        )
        pair_mean_pct = 0.5 * (w_pct + u_pct)
        weight_g = float(w_best["weight_g"])

        rank_key = (
            0,  # Always in preferred window for same-session same-scene mixed-lens pairs
            0 if both_median_tier else 1,
            round(abs(pair_mean_pct - target_median), 6),
            round(abs(weight_g - target_weight_g), 6),
            str(w_best["image_id"]),
            str(u_best["image_id"]),
        )
        existing = best_by_fold.get(fold_idx)
        if existing is None or rank_key < existing[0]:
            best_by_fold[fold_idx] = (rank_key, (w_best, u_best))

    return best_by_fold


def _select_single_bird_presets(
    dataset: LoadedModelingDataset,
    eval_result: FullEvaluationResult,
) -> list[dict[str, Any]]:
    """Deterministically select 6 non-fallback median-tier OOF presets spanning all 5 folds."""
    df = dataset.df
    oof_pred = np.asarray(eval_result.winner.oof_predictions, dtype=np.float64)
    if df.empty or len(df) != oof_pred.size:
        raise ValueError("Cannot build single-bird presets from empty or misaligned OOF dataset.")

    enriched = df.loc[
        :,
        [
            "image_id",
            "session_id",
            "fold",
            "weight_g",
            "weight_band",
            "lens_class",
            "scene_type",
            "segmentation_fallback",
        ],
    ].copy()
    y_true = enriched["weight_g"].to_numpy(dtype=np.float64)
    abs_err = np.abs(oof_pred - y_true)
    pct_err = (abs_err / np.maximum(y_true, 1e-6)) * 100.0

    enriched.loc[:, "oof_predicted_g"] = oof_pred
    enriched.loc[:, "oof_abs_err_g"] = abs_err
    enriched.loc[:, "oof_pct_err"] = pct_err
    # Normalize weight_band consistently via Phase 1 assign_weight_band.
    enriched.loc[:, "weight_band"] = [
        assign_weight_band(int(round(float(w)))) for w in y_true
    ]

    non_fallback = enriched.loc[~enriched["segmentation_fallback"].astype(bool)].reset_index(
        drop=True
    )
    if non_fallback.empty:
        raise ValueError(
            "Empty single-bird preset candidate pool: all rows have segmentation_fallback=True."
        )

    band_arr = non_fallback["weight_band"].astype(str).to_numpy()
    scene_arr = non_fallback["scene_type"].astype(str).to_numpy()

    # Slot 1: Young chick (<500g, preferred [130, 240] g, target 180 g).
    young_pool = non_fallback.loc[band_arr == "<500g"].reset_index(drop=True)
    young_w = young_pool["weight_g"].to_numpy(dtype=np.float64)
    young_by_fold = _rank_single_slot_candidates_by_fold(
        young_pool,
        preferred_mask=(young_w >= 130.0) & (young_w <= 240.0),
        target_weight_g=180.0,
        slot_name="sb_young_chick",
    )

    # Slot 2: Intermediate grower (500-1000g, preferred [480, 650] g, target 550 g).
    grower_pool = non_fallback.loc[band_arr == "500-1000g"].reset_index(drop=True)
    grower_w = grower_pool["weight_g"].to_numpy(dtype=np.float64)
    grower_by_fold = _rank_single_slot_candidates_by_fold(
        grower_pool,
        preferred_mask=(grower_w >= 480.0) & (grower_w <= 650.0),
        target_weight_g=550.0,
        slot_name="sb_intermediate_grower",
    )

    # Slot 3: Heavy broiler (>1000g, preferred [1400, 1850] g & scale_or_litter, target 1600 g).
    heavy_pool = non_fallback.loc[band_arr == ">1000g"].reset_index(drop=True)
    heavy_w = heavy_pool["weight_g"].to_numpy(dtype=np.float64)
    heavy_scene = heavy_pool["scene_type"].astype(str).to_numpy()
    heavy_pref = (heavy_w >= 1400.0) & (heavy_w <= 1850.0) & (heavy_scene == "scale_or_litter")
    if not np.any(heavy_pref):
        heavy_pref = (heavy_w >= 1400.0) & (heavy_w <= 1850.0)
    heavy_by_fold = _rank_single_slot_candidates_by_fold(
        heavy_pool,
        preferred_mask=heavy_pref,
        target_weight_g=1600.0,
        slot_name="sb_heavy_broiler",
    )

    # Slot 4: Hard white-on-white dorsal bucket (>1000g & bucket_dorsal, target 1450 g).
    bucket_mask = (band_arr == ">1000g") & (scene_arr == "bucket_dorsal")
    bucket_pool = non_fallback.loc[bucket_mask].reset_index(drop=True)
    bucket_by_fold = _rank_single_slot_candidates_by_fold(
        bucket_pool,
        preferred_mask=np.ones(len(bucket_pool), dtype=bool),
        target_weight_g=1450.0,
        slot_name="sb_hard_white_bucket",
    )

    # Slot 5-6: Paired same-session same-scene dual-lens Wide + Ultrawide pair.
    dual_by_fold = _rank_dual_lens_pairs_by_fold(non_fallback)

    # Exact 5-fold bijection over the 5 fold-consuming slot groups ordered scarcest-pool-first.
    slot_candidates: dict[str, dict[int, tuple[tuple[Any, ...], Any]]] = {
        "sb_young_chick": young_by_fold,
        "sb_intermediate_grower": grower_by_fold,
        "sb_heavy_broiler": heavy_by_fold,
        "sb_hard_white_bucket": bucket_by_fold,
        "dual_lens_pair": dual_by_fold,
    }
    ordered_by_scarcity = sorted(
        _SINGLE_BIRD_SLOT_ORDER,
        key=lambda slot: (len(slot_candidates[slot]), slot),
    )

    best_assignment_cost: tuple[Any, ...] | None = None
    best_assignment: dict[str, int] | None = None

    for perm in itertools.permutations((0, 1, 2, 3, 4)):
        assignment = dict(zip(ordered_by_scarcity, perm, strict=True))
        if not all(
            assignment[slot] in slot_candidates[slot] for slot in ordered_by_scarcity
        ):
            continue

        # Enforce distinct session_id across all 5 slot groups (defense-in-depth beyond folds).
        used_sessions: set[str] = set()
        duplicate_session = False
        pref_penalties = 0
        tier_penalties = 0
        pct_cost_sum = 0.0
        weight_cost_sum = 0.0
        tie_ids: list[str] = []

        for slot in _SINGLE_BIRD_SLOT_ORDER:
            fold_idx = assignment[slot]
            rank_key, payload = slot_candidates[slot][fold_idx]
            if slot == "dual_lens_pair":
                w_rec, u_rec = payload
                sid = str(w_rec["session_id"])
                tie_ids.append(f"{w_rec['image_id']}:{u_rec['image_id']}")
            else:
                sid = str(payload["session_id"])
                tie_ids.append(str(payload["image_id"]))

            if sid in used_sessions:
                duplicate_session = True
                break
            used_sessions.add(sid)

            pref_penalties += int(rank_key[0])
            tier_penalties += int(rank_key[1])
            pct_cost_sum += float(rank_key[2])
            weight_cost_sum += float(rank_key[3])

        if duplicate_session:
            continue

        cost = (
            pref_penalties,
            tier_penalties,
            round(pct_cost_sum, 6),
            round(weight_cost_sum, 6),
            tuple(tie_ids),
        )
        if best_assignment_cost is None or cost < best_assignment_cost:
            best_assignment_cost = cost
            best_assignment = assignment

    if best_assignment is None:
        pool_folds = {
            slot: sorted(slot_candidates[slot].keys()) for slot in _SINGLE_BIRD_SLOT_ORDER
        }
        raise ValueError(
            "Unable to find a valid 5-fold distinct assignment across the 5 single-bird "
            f"preset slots; available folds per slot: {pool_folds}"
        )

    young_rec = young_by_fold[best_assignment["sb_young_chick"]][1]
    grower_rec = grower_by_fold[best_assignment["sb_intermediate_grower"]][1]
    heavy_rec = heavy_by_fold[best_assignment["sb_heavy_broiler"]][1]
    bucket_rec = bucket_by_fold[best_assignment["sb_hard_white_bucket"]][1]
    dual_wide_rec, dual_uw_rec = dual_by_fold[best_assignment["dual_lens_pair"]][1]

    return [
        _format_single_bird_preset(
            young_rec,
            preset_id="sb_young_chick",
            label="Pintinho Jovem (~180 g)",
            category="young_chick",
            pair_group_id=None,
        ),
        _format_single_bird_preset(
            grower_rec,
            preset_id="sb_intermediate_grower",
            label="Frango em Crescimento (~550 g)",
            category="intermediate_grower",
            pair_group_id=None,
        ),
        _format_single_bird_preset(
            heavy_rec,
            preset_id="sb_heavy_broiler",
            label="Frango Pesado (~1.600 g)",
            category="heavy_broiler",
            pair_group_id=None,
        ),
        _format_single_bird_preset(
            bucket_rec,
            preset_id="sb_hard_white_bucket",
            label="Caso Difícil — Balde Branco (>1.000 g)",
            category="hard_white_on_white_bucket",
            pair_group_id=None,
        ),
        _format_single_bird_preset(
            dual_wide_rec,
            preset_id="sb_dual_lens_wide",
            label="Par Dual-Lens — Lente Wide (4,25 mm)",
            category="dual_lens_pair_wide",
            pair_group_id="dual_lens_pair",
        ),
        _format_single_bird_preset(
            dual_uw_rec,
            preset_id="sb_dual_lens_ultrawide",
            label="Par Dual-Lens — Lente Ultrawide (1,65 mm)",
            category="dual_lens_pair_ultrawide",
            pair_group_id="dual_lens_pair",
        ),
    ]


def _select_watershed_presets(
    quarantine_df: pd.DataFrame,
    dataset_root: Path,
    cfg: PDIConfig,
) -> list[dict[str, Any]]:
    """Select 4 deterministic opaque-ID multi-bird Watershed presets enforcing ASVS V4."""
    missing = [col for col in _REQUIRED_QUARANTINE_COLUMNS if col not in quarantine_df.columns]
    if missing:
        raise ValueError(f"Quarantine manifest missing required columns: {missing}")
    if quarantine_df.empty:
        raise ValueError("Cannot build Watershed presets from an empty quarantine manifest.")

    canonical_root = Path(dataset_root).resolve()
    if not canonical_root.exists() or not canonical_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist or is not a directory: {dataset_root}"
        )

    multi_df = quarantine_df.loc[
        quarantine_df["quarantine_reason"].astype(str) == "multi_bird"
    ].copy()
    if multi_df.empty:
        raise ValueError(
            "Empty Watershed candidate pool: no rows with quarantine_reason == 'multi_bird'."
        )

    leaf_folders = multi_df["folder_chain"].astype(str).str.split("/").str[-1]
    presets: list[dict[str, Any]] = []

    for folder_id, expected_count, group_sum_weight_g in WATERSHED_FOLDER_SPECS:
        folder_df = multi_df.loc[leaf_folders == folder_id]
        if folder_df.empty:
            raise ValueError(
                f"Empty Watershed candidate pool for multi-bird folder {folder_id!r}."
            )

        ranked_candidates: list[
            tuple[tuple[int, float, str], dict[str, Any], int]
        ] = []
        for record in folder_df.loc[:, list(_REQUIRED_QUARANTINE_COLUMNS)].to_dict(
            orient="records"
        ):
            safe_path = _resolve_safe_image_path(record["rel_path"], canonical_root)
            ws_res = run_watershed_pdi(safe_path, config=cfg, return_visuals=False)

            areas = [
                float(inst.area_work_px) for inst in ws_res.instances if inst.area_work_px > 0
            ]
            balance_ratio = (
                min(areas) / max(areas) if len(areas) >= 2 and max(areas) > 0.0 else 0.0
            )

            image_id = str(record["image_id"])
            sort_key = (
                abs(int(ws_res.bird_count) - int(expected_count)),
                -round(balance_ratio, 6),
                image_id,
            )
            ranked_candidates.append((sort_key, record, int(ws_res.bird_count)))

        ranked_candidates.sort(key=lambda item: item[0])
        _best_key, best_record, best_detected_count = ranked_candidates[0]

        presets.append(
            {
                "preset_id": f"ws_{folder_id}",
                "label": f"Grupo {expected_count} aves — Soma {group_sum_weight_g} g",
                "folder_id": str(folder_id),
                "image_id": str(best_record["image_id"]),
                "lens_class": str(best_record["lens_class"]),
                "expected_bird_count": int(expected_count),
                "detected_bird_count_default": int(best_detected_count),
                "group_sum_weight_g": int(group_sum_weight_g),
                "mean_bird_weight_reference_g": _round4(
                    float(group_sum_weight_g) / float(expected_count)
                ),
                "ground_truth_scope": "group_sum_only",
                "disclaimer": _WATERSHED_GROUP_SUM_DISCLAIMER,
            }
        )

    return presets


def build_presets_catalog(
    dataset: LoadedModelingDataset,
    eval_result: FullEvaluationResult,
    quarantine_df: pd.DataFrame,
    dataset_root: Path,
    *,
    config: PDIConfig | None = None,
) -> dict[str, Any]:
    """Build byte-deterministic, opaque-ID-only D-10 curated presets catalog."""
    cfg = config if config is not None else dataset.pdi_config
    single_bird_presets = _select_single_bird_presets(dataset, eval_result)
    watershed_presets = _select_watershed_presets(quarantine_df, dataset_root, cfg)

    return {
        "schema_version": PRESETS_CATALOG_SCHEMA_VERSION,
        "pdi_config_hash": str(dataset.pdi_config_hash),
        "winning_model_name": str(eval_result.winner.model_name),
        "winning_ablation_condition": str(eval_result.winner.ablation_condition),
        "single_bird_presets": single_bird_presets,
        "watershed_presets": watershed_presets,
    }


def write_presets_catalog(catalog: Mapping[str, Any], path: Path) -> None:
    """Validate PROIBIÇÃO 4 compliance and atomically write ``presets_catalog.json``."""
    serialized = _serialize_deterministic_json(catalog, context_label="presets_catalog.json")
    _atomic_write_text(serialized, path)


__all__ = [
    "MODEL_BUNDLE_SCHEMA_VERSION",
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "PRESETS_CATALOG_SCHEMA_VERSION",
    "MODEL_VERSION",
    "WATERSHED_FOLDER_SPECS",
    "REQUIRED_BUNDLE_KEYS",
    "build_model_bundle",
    "validate_model_bundle",
    "save_model_bundle",
    "load_model_bundle",
    "build_evaluation_report",
    "write_evaluation_report",
    "build_presets_catalog",
    "write_presets_catalog",
]
