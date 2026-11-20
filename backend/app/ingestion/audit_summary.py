"""Deterministic, zero-path-leakage aggregate audit summary builder (D-08).

Constructs the git-tracked ``models/dataset_audit_summary.json`` payload
consumed by Phase 4's ``WEB-05`` Forensic Audit Dashboard strictly from
aggregate counters, audited anomaly folder identifiers, and static Mendeley
published claims — never raw image paths, ``rel_path`` entries, host filesystem
paths, or volatile wall-clock timestamps (PROIBIÇÃO 4, T-01-07).
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.ingestion.decoding import (
    LCD_CORRECTIONS,
    MTIME_OFFSET_SEC,
    SESSION_GAP_SECONDS,
)
from backend.app.ingestion.manifest import enforce_manifest_dtypes
from backend.app.ingestion.splits import N_SPLITS, RANDOM_STATE, verify_fold_integrity

AUDIT_SUMMARY_SCHEMA_VERSION: str = "1.0.0"

DATASET_IDENTITY: dict[str, str] = {
    "name": "Broiler weights of Cobb Avian",
    "doi": "10.17632/zrs8kk9dvr.1",
    "breed": "Cobb Avian 48",
    "capture_device": "Xiaomi Redmi 10",
    "scale_model": "SF-400",
}

MENDELEY_PUBLISHED_CLAIMS: dict[str, Any] = {
    "published_images": 1772,
    "published_duration_claim": "daily for 45 days",
    "published_weight_note": "no curated quarantine or EXIF optical calibration disclosed",
}

SPARSE_GAP_LOWER_EXCLUSIVE_G: int = 801
SPARSE_GAP_UPPER_EXCLUSIVE_G: int = 1216

_CP866_AUDIT_FOLDERS: frozenset[str] = frozenset({"547", "562"})
_WEIGHT_BAND_ORDER: tuple[str, ...] = ("<500g", "500-1000g", ">1000g")
_FORBIDDEN_LEAK_SUBSTRINGS: tuple[str, ...] = (".jpg", "rel_path", "/home/", "data/raw")


def _sorted_dates(series: pd.Series[Any]) -> list[date]:
    values: list[date] = []
    for raw in series.dropna().unique().tolist():
        if isinstance(raw, date):
            values.append(raw)
        else:
            values.append(pd.Timestamp(raw).date())
    return sorted(set(values))


def _folder_count_map(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    counts = df["folder_chain"].value_counts()
    return {str(folder): int(count) for folder, count in sorted(counts.items())}


def _compute_temporal_hiatus(clean_dates: list[date]) -> dict[str, Any]:
    if len(clean_dates) < 2:
        return {
            "max_gap_days": 0,
            "gap_start_date": None,
            "gap_end_date": None,
        }
    max_gap = -1
    gap_start: date | None = None
    gap_end: date | None = None
    for prev_d, curr_d in zip(clean_dates, clean_dates[1:], strict=False):
        delta = (curr_d - prev_d).days
        if delta > max_gap:
            max_gap = delta
            gap_start = prev_d
            gap_end = curr_d
    return {
        "max_gap_days": int(max_gap),
        "gap_start_date": gap_start.isoformat() if gap_start is not None else None,
        "gap_end_date": gap_end.isoformat() if gap_end is not None else None,
    }


def _count_variable_digital_zoom(zoom_series: pd.Series[Any]) -> int:
    zoom_arr = zoom_series.to_numpy(dtype=float)
    finite_zooms = zoom_arr[np.isfinite(zoom_arr)]
    return int(
        sum(
            1
            for z in finite_zooms
            if not (math.isclose(z, 0.6, abs_tol=1e-6) or math.isclose(z, 1.0, abs_tol=1e-6))
        )
    )


def build_audit_summary(
    clean_df: pd.DataFrame,
    quarantine_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build deterministic aggregate audit JSON dict from clean and quarantine manifests."""
    typed_clean = enforce_manifest_dtypes(clean_df)
    typed_quarantine = enforce_manifest_dtypes(quarantine_df)

    if typed_quarantine.empty:
        full_df = typed_clean.copy()
    elif typed_clean.empty:
        full_df = typed_quarantine.copy()
    else:
        full_df = pd.concat([typed_clean, typed_quarantine], ignore_index=True)

    raw_images = int(len(full_df))
    raw_leaf_folders = int(full_df["folder_chain"].nunique())
    raw_unique_weights = int(full_df["raw_weight_g"].nunique())
    raw_sessions = int(full_df["session_id"].nunique())

    all_dates = _sorted_dates(full_df["capture_date"])
    date_min = all_dates[0] if all_dates else None
    date_max = all_dates[-1] if all_dates else None
    calendar_span_days = int((date_max - date_min).days + 1) if date_min and date_max else 0

    clean_images = int(len(typed_clean))
    clean_weight_min_g = int(typed_clean["weight_g"].min()) if clean_images > 0 else None
    clean_weight_max_g = int(typed_clean["weight_g"].max()) if clean_images > 0 else None

    mendeley_vs_disk_reconciliation: dict[str, Any] = {
        "published_claims": dict(MENDELEY_PUBLISHED_CLAIMS),
        "disk_audit": {
            "raw_images": raw_images,
            "raw_leaf_folders": raw_leaf_folders,
            "raw_unique_weights": raw_unique_weights,
            "raw_sessions": raw_sessions,
            "distinct_capture_dates": int(len(all_dates)),
            "date_min": date_min.isoformat() if date_min is not None else None,
            "date_max": date_max.isoformat() if date_max is not None else None,
            "calendar_span_days": calendar_span_days,
            "estimated_developmental_age_days_range": "~7-29 days",
            "clean_weight_min_g": clean_weight_min_g,
            "clean_weight_max_g": clean_weight_max_g,
            "missing_vs_mendeley_images": int(
                MENDELEY_PUBLISHED_CLAIMS["published_images"] - raw_images
            ),
        },
    }

    mb_df = typed_quarantine.loc[typed_quarantine["is_multi_bird"]]
    sm_df = typed_quarantine.loc[typed_quarantine["is_severe_mislabel"]]
    lcd_df = typed_clean.loc[typed_clean["was_lcd_corrected"]]
    lcd_applied_folders = set(lcd_df["folder_chain"].astype(str).unique().tolist())

    curation_funnel: dict[str, Any] = {
        "raw_images": raw_images,
        "raw_sessions": raw_sessions,
        "quarantined_total": int(len(typed_quarantine)),
        "quarantined_multi_bird": {
            "count": int(len(mb_df)),
            "folders": sorted(str(f) for f in mb_df["folder_chain"].unique().tolist()),
            "per_folder_counts": _folder_count_map(mb_df),
        },
        "quarantined_severe_mislabel": {
            "count": int(len(sm_df)),
            "folders": sorted(str(f) for f in sm_df["folder_chain"].unique().tolist()),
            "per_folder_counts": _folder_count_map(sm_df),
        },
        "lcd_corrected_images": {
            "count": int(len(lcd_df)),
            "corrections_applied": {
                str(folder): int(corrected)
                for folder, corrected in sorted(LCD_CORRECTIONS.items())
                if folder in lcd_applied_folders
            },
        },
        "clean_images": clean_images,
        "clean_leaf_folders": int(typed_clean["folder_chain"].nunique()),
        "clean_unique_weights": int(typed_clean["weight_g"].nunique()),
        "clean_sessions": int(typed_clean["session_id"].nunique()),
    }

    rotated_count = int((full_df["exif_orientation"] != 1).sum())
    rotated_pct = round(100.0 * rotated_count / raw_images, 2) if raw_images > 0 else 0.0
    orientation_counts = {
        str(int(orient)): int(cnt)
        for orient, cnt in sorted(
            full_df["exif_orientation"].value_counts().items(),
            key=lambda kv: str(int(kv[0])),
        )
    }
    filename_pattern_counts = {
        str(pattern): int(cnt)
        for pattern, cnt in sorted(full_df["filename_pattern"].value_counts().items())
    }
    cp866_folder_df = full_df.loc[full_df["folder_chain"].isin(_CP866_AUDIT_FOLDERS)]
    cp866_mtime_fallback_count = int(
        (cp866_folder_df["filename_pattern"] != "img_mojibake_cp866").sum()
    )
    nested_level2_folders = sorted(
        str(fc) for fc in full_df.loc[full_df["was_nested_folder"], "folder_chain"].unique()
    )
    suffix_sanitized_folders = sorted(
        str(fc)
        for fc in full_df["folder_chain"].unique().tolist()
        if not str(fc).rsplit("/", 1)[-1].isdigit()
    )

    encoding_and_exif_recovery: dict[str, Any] = {
        "exif_orientation_rotated_count": rotated_count,
        "exif_orientation_rotated_pct": rotated_pct,
        "orientation_counts": orientation_counts,
        "filename_pattern_counts": filename_pattern_counts,
        "cp866_recovered_count": int(full_df["was_cp866_recovered"].sum()),
        "cp866_mtime_fallback_count": cp866_mtime_fallback_count,
        "year_2024_capture_count": int((full_df["capture_dt"].dt.year == 2024).sum()),
        "nested_level2_folders": nested_level2_folders,
        "nested_level2_images_count": int(full_df["was_nested_folder"].sum()),
        "suffix_sanitized_folders": suffix_sanitized_folders,
        "mtime_fallback_count": int((full_df["filename_pattern"] == "weight_renamed").sum()),
        "mtime_offset_sec": int(MTIME_OFFSET_SEC),
        "variable_digital_zoom_count": _count_variable_digital_zoom(full_df["zoom_multiple"]),
    }

    weight_bands: dict[str, dict[str, Any]] = {}
    for band in _WEIGHT_BAND_ORDER:
        band_df = typed_clean.loc[typed_clean["weight_band"] == band]
        n_band = int(len(band_df))
        weight_bands[band] = {
            "n_images": n_band,
            "pct_images": round(100.0 * n_band / clean_images, 2) if clean_images > 0 else 0.0,
            "n_sessions": int(band_df["session_id"].nunique()),
            "n_wide": int((band_df["lens_class"] == "wide").sum()),
            "n_ultrawide": int((band_df["lens_class"] == "ultrawide").sum()),
        }

    lenses: dict[str, dict[str, Any]] = {}
    for lens_name in ("wide", "ultrawide", "unknown"):
        lens_df = typed_clean.loc[typed_clean["lens_class"] == lens_name]
        n_lens = int(len(lens_df))
        if lens_name == "unknown" and n_lens == 0:
            continue
        lenses[lens_name] = {
            "n_images": n_lens,
            "pct_images": round(100.0 * n_lens / clean_images, 2) if clean_images > 0 else 0.0,
            "n_sessions": int(lens_df["session_id"].nunique()),
        }

    sparse_df = typed_clean.loc[
        (typed_clean["weight_g"] > SPARSE_GAP_LOWER_EXCLUSIVE_G)
        & (typed_clean["weight_g"] < SPARSE_GAP_UPPER_EXCLUSIVE_G)
    ]
    clean_dates = _sorted_dates(typed_clean["capture_date"])
    timeline_by_date: list[dict[str, Any]] = []
    for cap_date in clean_dates:
        day_df = typed_clean.loc[typed_clean["capture_date"] == cap_date]
        timeline_by_date.append(
            {
                "date": cap_date.isoformat(),
                "n_images": int(len(day_df)),
                "n_sessions": int(day_df["session_id"].nunique()),
                "weight_min_g": int(day_df["weight_g"].min()),
                "weight_max_g": int(day_df["weight_g"].max()),
            }
        )

    distributions: dict[str, Any] = {
        "weight_bands": weight_bands,
        "lenses": lenses,
        "coverage_gaps": {
            "weight_sparse_interval": {
                "lower_exclusive_g": int(SPARSE_GAP_LOWER_EXCLUSIVE_G),
                "upper_exclusive_g": int(SPARSE_GAP_UPPER_EXCLUSIVE_G),
                "span_g": int(SPARSE_GAP_UPPER_EXCLUSIVE_G - SPARSE_GAP_LOWER_EXCLUSIVE_G),
                "clean_images_inside": int(len(sparse_df)),
                "clean_unique_weights_inside": int(sparse_df["weight_g"].nunique()),
                "clean_weights_inside": sorted(
                    int(w) for w in sparse_df["weight_g"].unique().tolist()
                ),
                "clean_sessions_inside": int(sparse_df["session_id"].nunique()),
            },
            "temporal_hiatus": _compute_temporal_hiatus(clean_dates),
        },
        "timeline_by_date": timeline_by_date,
    }

    fold_audit = verify_fold_integrity(typed_clean)
    split_audit: dict[str, Any] = {
        "strategy": "StratifiedGroupKFold",
        "n_splits": int(N_SPLITS),
        "shuffle": True,
        "random_state": int(RANDOM_STATE),
        "group_column": "session_id",
        "session_gap_threshold_seconds": int(SESSION_GAP_SECONDS),
        "stratify_column": "weight_band",
        "cross_fold_session_overlap": int(fold_audit["cross_fold_session_overlap"]),
        "folds": fold_audit["folds"],
    }

    return {
        "schema_version": AUDIT_SUMMARY_SCHEMA_VERSION,
        "dataset_identity": dict(DATASET_IDENTITY),
        "mendeley_vs_disk_reconciliation": mendeley_vs_disk_reconciliation,
        "curation_funnel": curation_funnel,
        "encoding_and_exif_recovery": encoding_and_exif_recovery,
        "distributions": distributions,
        "split_audit": split_audit,
    }


def write_audit_summary(summary: dict[str, Any], path: Path) -> None:
    """Serialize aggregate audit summary to byte-deterministic UTF-8 JSON (PROIBIÇÃO 4)."""
    serialized = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    lowered = serialized.lower()
    for token in _FORBIDDEN_LEAK_SUBSTRINGS:
        if token in lowered:
            raise ValueError(
                f"PROIBIÇÃO 4 violation: forbidden path/filename marker {token!r} "
                "detected in aggregate audit summary payload."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


__all__ = [
    "AUDIT_SUMMARY_SCHEMA_VERSION",
    "DATASET_IDENTITY",
    "MENDELEY_PUBLISHED_CLAIMS",
    "SPARSE_GAP_LOWER_EXCLUSIVE_G",
    "SPARSE_GAP_UPPER_EXCLUSIVE_G",
    "build_audit_summary",
    "write_audit_summary",
]
