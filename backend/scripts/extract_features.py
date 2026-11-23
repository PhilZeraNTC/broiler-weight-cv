"""Capped-concurrency offline batch PDI feature extractor and dual-lens calibration audit CLI.

Reads ``data/processed/manifest_clean.parquet`` strictly read-only, enforces ASVS V4
path/symlink containment against ``dataset_root.resolve()`` (T-02-10), runs the
single-source classical PDI pipeline ``run_single_bird_pdi(..., return_xray=False)``
across a strictly capped worker pool (``MAX_WORKERS_CAP = 4``, ``cv2.setNumThreads(1)``
per worker, D-10 / T-02-11), writes the frozen 60-column typed PyArrow Parquet
``data/processed/features_clean.parquet`` (D-08), and publishes the byte-deterministic
zero-path-leakage aggregate calibration audit ``models/pdi_calibration_audit.json``
(D-09, PROIBIÇÃO 4, T-02-12).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa

from backend.app.config import (
    CLEAN_MANIFEST_PATH,
    FEATURES_CLEAN_PATH,
    PDI_CALIBRATION_AUDIT_PATH,
    RAW_DATASET_DIR,
)
from backend.app.pdi.config import PDIConfig
from backend.app.pdi.features import BANNED_FEATURE_COLUMNS, FEATURE_COLUMNS
from backend.app.pdi.pipeline import run_single_bird_pdi

CALIBRATION_AUDIT_SCHEMA_VERSION: str = "1.0.0"
MAX_WORKERS_CAP: int = 4

_FORBIDDEN_LEAK_SUBSTRINGS: tuple[str, ...] = (".jpg", "rel_path", "/home/", "data/raw")

_WINNING_FORMULA: str = "area_px / (zoom_multiple ** 2)"
_REJECTED_DOUBLE_COUNT_FORMULA: str = "area_px / ((focal_length_mm * zoom_multiple) ** 2)"
_RESOLUTION_AREA_RATIO_WIDE_OVER_UW: float = 1.5625
_EXPECTED_ZOOM_AREA_RATIO_WIDE_OVER_UW: float = 2.7778

_CANDIDATE_FORMULAS: tuple[str, ...] = (
    "area_frac",
    "area_px",
    "area_px / (zoom_multiple ** 2)",
    "area_frac / (zoom_multiple ** 2)",
    "area_frac / (focal_length_mm ** 2)",
    "area_px / ((focal_length_mm * zoom_multiple) ** 2)",
)

# Frozen 11 join/audit columns matching manifest_clean.parquet dtypes (D-10).
JOIN_AUDIT_COLUMNS: tuple[str, ...] = (
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

# Frozen 60-column Parquet schema = 11 join/audit + 49 strictly mask-gated features (D-08).
FEATURES_PARQUET_COLUMNS: tuple[str, ...] = JOIN_AUDIT_COLUMNS + FEATURE_COLUMNS

if not BANNED_FEATURE_COLUMNS.isdisjoint(FEATURES_PARQUET_COLUMNS):
    _overlap = sorted(BANNED_FEATURE_COLUMNS.intersection(FEATURES_PARQUET_COLUMNS))
    raise RuntimeError(
        f"PROIBIÇÃO 2 schema violation: banned columns present in FEATURES_PARQUET_COLUMNS: "
        f"{_overlap}"
    )

_REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "rel_path",
    "fold",
    "session_id",
    "weight_g",
    "weight_band",
    "lens_class",
)

_INT8_COLUMNS: tuple[str, ...] = ("fold",)
_INT32_COLUMNS: tuple[str, ...] = ("weight_g", "lcd_excluded_px")
_BOOL_COLUMNS: tuple[str, ...] = ("scene_is_bucket", "segmentation_fallback")
_STRING_COLUMNS: tuple[str, ...] = (
    "image_id",
    "session_id",
    "weight_band",
    "lens_class",
    "scene_type",
    "pdi_config_hash",
)

_FEATURES_ARROW_SCHEMA: pa.Schema = pa.schema(
    [
        ("image_id", pa.string()),
        ("fold", pa.int8()),
        ("session_id", pa.string()),
        ("weight_g", pa.int32()),
        ("weight_band", pa.string()),
        ("lens_class", pa.string()),
        ("scene_type", pa.string()),
        ("scene_is_bucket", pa.bool_()),
        ("lcd_excluded_px", pa.int32()),
        ("segmentation_fallback", pa.bool_()),
        ("pdi_config_hash", pa.string()),
        *((col, pa.float64()) for col in FEATURE_COLUMNS),
    ]
)


def _resolve_effective_workers(workers: int) -> int:
    """Validate and strictly clamp worker concurrency to ``[1, MAX_WORKERS_CAP]`` (T-02-11)."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"--workers must be a positive integer >= 1, got {workers!r}")
    return min(workers, MAX_WORKERS_CAP)


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


def _init_worker() -> None:
    """Cap OpenCV per-worker thread count to 1 to avoid thread oversubscription (D-10)."""
    cv2.setNumThreads(1)


def _extract_single_row(task: tuple[str, dict[str, Any], PDIConfig]) -> dict[str, Any]:
    """Worker entrypoint extracting the 60-column row dict for one clean image."""
    abs_path_str, manifest_meta, cfg = task
    result = run_single_bird_pdi(Path(abs_path_str), config=cfg, return_xray=False)

    row: dict[str, Any] = {
        "image_id": str(manifest_meta["image_id"]),
        "fold": int(manifest_meta["fold"]),
        "session_id": str(manifest_meta["session_id"]),
        "weight_g": int(manifest_meta["weight_g"]),
        "weight_band": str(manifest_meta["weight_band"]),
        "lens_class": str(manifest_meta["lens_class"]),
        "scene_type": str(result.segmentation.scene_type),
        "scene_is_bucket": bool(result.segmentation.scene_is_bucket),
        "lcd_excluded_px": int(result.segmentation.lcd_excluded_px),
        "segmentation_fallback": bool(result.segmentation.segmentation_fallback),
        "pdi_config_hash": str(result.pdi_config_hash),
    }

    for col in FEATURE_COLUMNS:
        val = float(result.features[col])
        if not math.isfinite(val):
            raise ValueError(f"Non-finite feature value {val!r} extracted for column {col!r}.")
        row[col] = val

    return row


def _json_normalize(value: Any) -> Any:
    """Convert dataclass tuples into JSON-native lists for deterministic equality."""
    if isinstance(value, dict):
        return {str(k): _json_normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(v) for v in value]
    return value


def _enforce_features_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce exact 60-column ordering, Pandas 3.1 CoW-safe dtypes, and finite feature guards."""
    banned_present = BANNED_FEATURE_COLUMNS.intersection(df.columns)
    if banned_present:
        raise ValueError(
            f"PROIBIÇÃO 2 violation: banned feature columns present: {sorted(banned_present)}"
        )

    missing = [col for col in FEATURES_PARQUET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features_clean columns: {missing}")

    if df.empty:
        empty_cols: dict[str, pd.Series[Any]] = {}
        for col in FEATURES_PARQUET_COLUMNS:
            if col in _INT8_COLUMNS:
                empty_cols[col] = pd.Series([], dtype="int8")
            elif col in _INT32_COLUMNS:
                empty_cols[col] = pd.Series([], dtype="int32")
            elif col in _BOOL_COLUMNS:
                empty_cols[col] = pd.Series([], dtype="bool")
            elif col in _STRING_COLUMNS:
                empty_cols[col] = pd.Series([], dtype="str")
            else:
                empty_cols[col] = pd.Series([], dtype="float64")
        return pd.DataFrame(empty_cols, columns=list(FEATURES_PARQUET_COLUMNS))

    out = df.loc[:, list(FEATURES_PARQUET_COLUMNS)].copy()

    for col in _INT8_COLUMNS:
        out[col] = out[col].astype("int8")
    for col in _INT32_COLUMNS:
        out[col] = out[col].astype("int32")
    for col in _BOOL_COLUMNS:
        out[col] = out[col].astype("bool")
    for col in _STRING_COLUMNS:
        out[col] = out[col].astype("str")
    for col in FEATURE_COLUMNS:
        out[col] = out[col].astype("float64")

    feature_matrix = out.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError("Non-finite (NaN or Inf) value detected in extracted feature matrix.")

    return out


def _compute_candidate_formula_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute vectorized float64 arrays for all 6 D-09 optical normalization candidates."""
    area_px = df["area_px"].to_numpy(dtype=np.float64)
    area_frac = df["area_frac"].to_numpy(dtype=np.float64)
    zoom = df["zoom_multiple"].to_numpy(dtype=np.float64)
    focal = df["focal_length_mm"].to_numpy(dtype=np.float64)

    zoom_sq = zoom * zoom
    focal_sq = focal * focal
    fz_sq = (focal * zoom) ** 2

    return {
        "area_frac": area_frac,
        "area_px": area_px,
        "area_px / (zoom_multiple ** 2)": area_px / zoom_sq,
        "area_frac / (zoom_multiple ** 2)": area_frac / zoom_sq,
        "area_frac / (focal_length_mm ** 2)": area_frac / focal_sq,
        "area_px / ((focal_length_mm * zoom_multiple) ** 2)": area_px / fz_sq,
    }


def _evaluate_paired_scope(df: pd.DataFrame, group_cols: list[str]) -> dict[str, Any]:
    """Evaluate paired Wide/Ultrawide within-group median ratios across all 6 candidate formulas."""
    ratios_by_formula: dict[str, list[float]] = {name: [] for name in _CANDIDATE_FORMULAS}
    n_pairs = 0

    if not df.empty:
        formula_arrays = _compute_candidate_formula_arrays(df)
        lens_arr = df["lens_class"].astype(str).to_numpy()
        grouped_indices = df.groupby(group_cols, sort=True).indices

        for _, raw_idx in sorted(grouped_indices.items(), key=lambda kv: kv[0]):
            idx = np.asarray(raw_idx, dtype=np.int64)
            grp_lens = lens_arr[idx]
            w_idx = idx[grp_lens == "wide"]
            u_idx = idx[grp_lens == "ultrawide"]
            if w_idx.size == 0 or u_idx.size == 0:
                continue

            n_pairs += 1
            for formula_name in _CANDIDATE_FORMULAS:
                arr = formula_arrays[formula_name]
                med_w = float(np.median(arr[w_idx]))
                med_u = float(np.median(arr[u_idx]))
                if med_w > 0.0 and med_u > 0.0 and math.isfinite(med_w) and math.isfinite(med_u):
                    ratio = med_w / med_u
                    if ratio > 0.0 and math.isfinite(ratio):
                        ratios_by_formula[formula_name].append(ratio)

    formula_summaries: dict[str, dict[str, float | None]] = {}
    for formula_name in _CANDIDATE_FORMULAS:
        vals = np.asarray(ratios_by_formula[formula_name], dtype=np.float64)
        if vals.size == 0:
            formula_summaries[formula_name] = {
                "median_wide_over_uw": None,
                "iqr_25": None,
                "iqr_75": None,
                "median_abs_log_ratio": None,
            }
        else:
            p25, p50, p75 = np.percentile(vals, (25.0, 50.0, 75.0))
            med_abs_log = float(np.median(np.abs(np.log(vals))))
            formula_summaries[formula_name] = {
                "median_wide_over_uw": round(float(p50), 6),
                "iqr_25": round(float(p25), 6),
                "iqr_75": round(float(p75), 6),
                "median_abs_log_ratio": round(med_abs_log, 6),
            }

    return {
        "n_pairs": int(n_pairs),
        "formulas": formula_summaries,
    }


def build_calibration_audit(
    features_df: pd.DataFrame,
    config: PDIConfig,
    *,
    workers: int,
) -> dict[str, Any]:
    """Build byte-deterministic aggregate-only extraction & dual-lens calibration audit (D-09)."""
    effective_workers = _resolve_effective_workers(workers)
    typed_df = _enforce_features_dtypes(features_df)

    n_clean = int(len(typed_df))
    is_bucket = typed_df["scene_is_bucket"].to_numpy(dtype=bool)
    n_bucket_dorsal = int(np.count_nonzero(is_bucket))
    n_scale_or_litter = int(np.count_nonzero(~is_bucket))

    fallback_mask = typed_df["segmentation_fallback"].to_numpy(dtype=bool)
    primary_fallback_count = int(np.count_nonzero(fallback_mask))
    primary_fallback_pct = (
        round(100.0 * primary_fallback_count / n_clean, 4) if n_clean > 0 else 0.0
    )

    lcd_triggered = typed_df["lcd_excluded_px"].to_numpy(dtype=np.int64) > 0
    lcd_suppression_triggered_count = int(np.count_nonzero(lcd_triggered))
    lcd_suppression_bucket_false_positives = int(np.count_nonzero(lcd_triggered & is_bucket))

    bucket_df = typed_df.loc[is_bucket].reset_index(drop=True)
    scale_litter_df = typed_df.loc[~is_bucket].reset_index(drop=True)

    scopes = {
        "all_mixed_lens_sessions": _evaluate_paired_scope(typed_df, ["session_id"]),
        "same_scene_pairs": _evaluate_paired_scope(typed_df, ["session_id", "scene_type"]),
        "bucket_only_pairs": _evaluate_paired_scope(bucket_df, ["session_id"]),
        "scale_or_litter_pairs": _evaluate_paired_scope(scale_litter_df, ["session_id"]),
    }

    return {
        "schema_version": CALIBRATION_AUDIT_SCHEMA_VERSION,
        "pdi_config_hash": config.config_hash(),
        "pdi_config": _json_normalize(asdict(config)),
        "dataset_extraction_summary": {
            "n_clean_images": n_clean,
            "n_bucket_dorsal": n_bucket_dorsal,
            "n_scale_or_litter": n_scale_or_litter,
            "primary_fallback_count": primary_fallback_count,
            "primary_fallback_pct": primary_fallback_pct,
            "lcd_suppression_triggered_count": lcd_suppression_triggered_count,
            "lcd_suppression_bucket_false_positives": lcd_suppression_bucket_false_positives,
            "workers": int(effective_workers),
        },
        "paired_dual_lens_calibration": {
            "winning_formula": _WINNING_FORMULA,
            "rejected_double_count_formula": _REJECTED_DOUBLE_COUNT_FORMULA,
            "resolution_area_ratio_wide_over_uw": _RESOLUTION_AREA_RATIO_WIDE_OVER_UW,
            "expected_zoom_area_ratio_wide_over_uw": _EXPECTED_ZOOM_AREA_RATIO_WIDE_OVER_UW,
            "scopes": scopes,
        },
    }


def write_features_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist the 60-column feature table strictly as typed PyArrow Parquet (never CSV)."""
    typed_df = _enforce_features_dtypes(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    typed_df.to_parquet(path, engine="pyarrow", index=False, schema=_FEATURES_ARROW_SCHEMA)


def write_calibration_audit(audit: dict[str, Any], path: Path) -> None:
    """Serialize aggregate calibration audit to byte-deterministic UTF-8 JSON (PROIBIÇÃO 4)."""
    serialized = json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    lowered = serialized.lower()
    for token in _FORBIDDEN_LEAK_SUBSTRINGS:
        if token in lowered:
            raise ValueError(
                f"PROIBIÇÃO 4 violation: forbidden path/filename marker {token!r} "
                "detected in calibration audit payload."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for capped-concurrency PDI batch feature extraction."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract strictly mask-gated classical PDI features for AvianScale CV "
            "and generate the aggregate dual-lens optical calibration audit."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=CLEAN_MANIFEST_PATH,
        help="Path to materialized manifest_clean.parquet.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=RAW_DATASET_DIR,
        help="Root directory containing raw broiler weight JPEG folders (read-only).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=FEATURES_CLEAN_PATH,
        help="Destination path for typed 60-column features_clean.parquet (git-ignored).",
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=PDI_CALIBRATION_AUDIT_PATH,
        help="Destination path for git-tracked aggregate pdi_calibration_audit.json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS_CAP,
        help=f"Number of parallel worker processes (strictly clamped to 1..{MAX_WORKERS_CAP}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full extraction and calibration audit without writing any files.",
    )
    args = parser.parse_args(argv)
    args.workers = _resolve_effective_workers(args.workers)
    return args


def run_extract_features(
    manifest_path: Path,
    dataset_root: Path,
    output_path: Path,
    audit_path: Path,
    *,
    workers: int = MAX_WORKERS_CAP,
    dry_run: bool = False,
    config: PDIConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run capped-concurrency batch PDI extraction, 60-col Parquet write, and calibration audit."""
    t0 = time.perf_counter()
    effective_workers = _resolve_effective_workers(workers)
    cfg = config if config is not None else PDIConfig()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Clean manifest Parquet not found: {manifest_path}")

    canonical_root = dataset_root.resolve()
    if not canonical_root.exists() or not canonical_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist or is not a directory: {dataset_root}"
        )

    manifest_df = pd.read_parquet(manifest_path)
    missing_cols = [col for col in _REQUIRED_MANIFEST_COLUMNS if col not in manifest_df.columns]
    if missing_cols:
        raise ValueError(f"Manifest Parquet missing required columns: {missing_cols}")

    # Validate every rel_path against ASVS V4 containment BEFORE starting worker pool (T-02-10).
    tasks: list[tuple[str, dict[str, Any], PDIConfig]] = []
    for record in manifest_df.loc[:, list(_REQUIRED_MANIFEST_COLUMNS)].to_dict(orient="records"):
        safe_path = _resolve_safe_image_path(record["rel_path"], canonical_root)
        manifest_meta = {
            "image_id": str(record["image_id"]),
            "fold": int(record["fold"]),
            "session_id": str(record["session_id"]),
            "weight_g": int(record["weight_g"]),
            "weight_band": str(record["weight_band"]),
            "lens_class": str(record["lens_class"]),
        }
        tasks.append((str(safe_path), manifest_meta, cfg))

    if not tasks:
        rows: list[dict[str, Any]] = []
    elif effective_workers == 1:
        rows = [_extract_single_row(task) for task in tasks]
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp_ctx,
            initializer=_init_worker,
        ) as executor:
            rows = list(executor.map(_extract_single_row, tasks, chunksize=8))

    features_df = _enforce_features_dtypes(
        pd.DataFrame(rows, columns=list(FEATURES_PARQUET_COLUMNS))
    )

    expected_ids = manifest_df["image_id"].astype(str).tolist()
    actual_ids = features_df["image_id"].astype(str).tolist()
    if actual_ids != expected_ids:
        raise RuntimeError(
            "Extracted feature rows failed exact 1:1 image_id alignment with manifest."
        )

    audit = build_calibration_audit(features_df, cfg, workers=effective_workers)

    if not dry_run:
        write_features_parquet(features_df, output_path)
        write_calibration_audit(audit, audit_path)

    wall_clock_sec = time.perf_counter() - t0
    summary = audit["dataset_extraction_summary"]
    bucket_scope = audit["paired_dual_lens_calibration"]["scopes"]["bucket_only_pairs"]
    bucket_med = bucket_scope["formulas"][_WINNING_FORMULA]["median_wide_over_uw"]

    # PROIBIÇÃO 4: emit strictly aggregate metrics — never raw filenames or host paths.
    print(
        "Feature extraction complete: "
        f"n_clean_images={summary['n_clean_images']} "
        f"n_bucket_dorsal={summary['n_bucket_dorsal']} "
        f"n_scale_or_litter={summary['n_scale_or_litter']} "
        f"primary_fallback_count={summary['primary_fallback_count']} "
        f"primary_fallback_pct={summary['primary_fallback_pct']:.4f}% "
        f"lcd_triggered={summary['lcd_suppression_triggered_count']} "
        f"lcd_bucket_fp={summary['lcd_suppression_bucket_false_positives']} "
        f"bucket_wide_over_uw={bucket_med} "
        f"workers={effective_workers} "
        f"pdi_config_hash={audit['pdi_config_hash']} "
        f"wall_clock_sec={wall_clock_sec:.2f} "
        f"dry_run={dry_run}"
    )
    return features_df, audit


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    run_extract_features(
        manifest_path=args.manifest_path,
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        audit_path=args.audit_path,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    return 0


__all__ = [
    "CALIBRATION_AUDIT_SCHEMA_VERSION",
    "MAX_WORKERS_CAP",
    "JOIN_AUDIT_COLUMNS",
    "FEATURES_PARQUET_COLUMNS",
    "parse_args",
    "build_calibration_audit",
    "write_features_parquet",
    "write_calibration_audit",
    "run_extract_features",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
