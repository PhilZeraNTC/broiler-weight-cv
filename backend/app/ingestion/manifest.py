"""Read-only forensic dataset crawler and 24-column Parquet manifest builder.

Crawls ``data/raw/Broiler weights of Cobb Avian/`` strictly read-only with
ASVS L1 symlink/path containment (T-01-04), decodes Audit §9 anomalies (D-04),
extracts EXIF metadata via the single shared Pillow loader (D-07), clusters
sequential captures into ``session_id`` values (gap > 600.0 s), and enforces
the immutable 24-column typed PyArrow Parquet contract (D-03).
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from backend.app.ingestion.decoding import (
    LCD_CORRECTIONS,
    MTIME_OFFSET_SEC,
    MULTI_BIRD_FOLDERS,
    SESSION_GAP_SECONDS,
    SEVERE_MISLABEL_FOLDERS,
    assign_weight_band,
    classify_lens,
    compute_image_id,
    decode_filename,
    parse_folder_weight_g,
)
from backend.app.pdi.io_exif import read_exif_metadata

# Immutable 24-column Parquet manifest contract (D-03)
RAW_BASE_COLUMNS: tuple[str, ...] = (
    "rel_path",
    "folder_chain",
    "raw_weight_g",
    "weight_g",
    "capture_dt",
    "capture_date",
    "oriented_w",
    "oriented_h",
    "exif_orientation",
    "focal_length_mm",
    "zoom_multiple",
    "is_multi_bird",
    "is_severe_mislabel",
    "is_single_bird_clean",
    "session_id",
)

DERIVED_COLUMNS: tuple[str, ...] = (
    "image_id",
    "filename_pattern",
    "was_cp866_recovered",
    "was_nested_folder",
    "was_lcd_corrected",
    "lens_class",
    "weight_band",
    "quarantine_reason",
    "fold",
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "rel_path",
    "folder_chain",
    "raw_weight_g",
    "weight_g",
    "capture_dt",
    "capture_date",
    "oriented_w",
    "oriented_h",
    "exif_orientation",
    "focal_length_mm",
    "zoom_multiple",
    "is_multi_bird",
    "is_severe_mislabel",
    "is_single_bird_clean",
    "session_id",
    "filename_pattern",
    "was_cp866_recovered",
    "was_nested_folder",
    "was_lcd_corrected",
    "lens_class",
    "weight_band",
    "quarantine_reason",
    "fold",
)

_SORT_KEYS: list[str] = ["folder_chain", "capture_dt", "rel_path"]
_INT32_COLUMNS: tuple[str, ...] = ("raw_weight_g", "weight_g", "oriented_w", "oriented_h")
_INT8_COLUMNS: tuple[str, ...] = ("exif_orientation", "fold")
_FLOAT64_COLUMNS: tuple[str, ...] = ("focal_length_mm", "zoom_multiple")
_BOOL_COLUMNS: tuple[str, ...] = (
    "is_multi_bird",
    "is_severe_mislabel",
    "is_single_bird_clean",
    "was_cp866_recovered",
    "was_nested_folder",
    "was_lcd_corrected",
)
_STRING_COLUMNS: tuple[str, ...] = (
    "image_id",
    "rel_path",
    "folder_chain",
    "session_id",
    "filename_pattern",
    "lens_class",
    "weight_band",
    "quarantine_reason",
)

_MANIFEST_ARROW_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("rel_path", pa.string()),
        ("folder_chain", pa.string()),
        ("raw_weight_g", pa.int32()),
        ("weight_g", pa.int32()),
        ("capture_dt", pa.timestamp("us")),
        ("capture_date", pa.date32()),
        ("oriented_w", pa.int32()),
        ("oriented_h", pa.int32()),
        ("exif_orientation", pa.int8()),
        ("focal_length_mm", pa.float64()),
        ("zoom_multiple", pa.float64()),
        ("is_multi_bird", pa.bool_()),
        ("is_severe_mislabel", pa.bool_()),
        ("is_single_bird_clean", pa.bool_()),
        ("session_id", pa.string()),
        ("filename_pattern", pa.string()),
        ("was_cp866_recovered", pa.bool_()),
        ("was_nested_folder", pa.bool_()),
        ("was_lcd_corrected", pa.bool_()),
        ("lens_class", pa.string()),
        ("weight_band", pa.string()),
        ("quarantine_reason", pa.string()),
        ("fold", pa.int8()),
    ]
)


def _normalize_date_value(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def enforce_manifest_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce exact 24-column ordering and Pandas 3.1 / PyArrow dtypes (CoW-safe)."""
    missing = [col for col in MANIFEST_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required manifest columns: {missing}")

    if df.empty:
        empty_cols: dict[str, pd.Series[Any]] = {}
        for col in MANIFEST_COLUMNS:
            if col in _INT32_COLUMNS:
                empty_cols[col] = pd.Series([], dtype="int32")
            elif col in _INT8_COLUMNS:
                empty_cols[col] = pd.Series([], dtype="int8")
            elif col in _FLOAT64_COLUMNS:
                empty_cols[col] = pd.Series([], dtype="float64")
            elif col in _BOOL_COLUMNS:
                empty_cols[col] = pd.Series([], dtype="bool")
            elif col == "capture_dt":
                empty_cols[col] = pd.Series([], dtype="datetime64[us]")
            elif col == "capture_date":
                empty_cols[col] = pd.Series([], dtype="object")
            else:
                empty_cols[col] = pd.Series([], dtype="str")
        return pd.DataFrame(empty_cols, columns=list(MANIFEST_COLUMNS))

    out = df.loc[:, list(MANIFEST_COLUMNS)].copy()

    dt_series = pd.to_datetime(out["capture_dt"], errors="raise")
    if dt_series.dt.tz is not None:
        raise ValueError("capture_dt must be timezone-naive local datetime matching Audit §9.")
    out["capture_dt"] = dt_series.astype("datetime64[us]")

    out["capture_date"] = pd.Series(
        [_normalize_date_value(v) for v in out["capture_date"]],
        index=out.index,
        dtype="object",
    )

    for col in _INT32_COLUMNS:
        out[col] = out[col].astype("int32")
    for col in _INT8_COLUMNS:
        out[col] = out[col].astype("int8")
    for col in _FLOAT64_COLUMNS:
        out[col] = out[col].astype("float64")
    for col in _BOOL_COLUMNS:
        out[col] = out[col].astype("bool")
    for col in _STRING_COLUMNS:
        out[col] = out[col].astype("str")

    return out


def _is_contained_non_symlink(path: Path, canonical_root: Path) -> bool:
    """Return True iff ``path`` is not a symlink and resolves strictly inside ``canonical_root``."""
    if path.is_symlink():
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved.is_relative_to(canonical_root)


def scan_dataset_records(dataset_root: Path) -> pd.DataFrame:
    """Recursively crawl ``dataset_root`` read-only and return the typed 24-column manifest."""
    root = dataset_root.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist or is not a directory: {dataset_root}"
        )

    rows: list[dict[str, Any]] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dir_p = Path(dirpath)
        if dir_p != root and not _is_contained_non_symlink(dir_p, root):
            dirnames[:] = []
            continue

        # Sort and prune symlinked or escaping subdirectories before descending.
        safe_dirnames: list[str] = []
        for dname in sorted(dirnames):
            if _is_contained_non_symlink(dir_p / dname, root):
                safe_dirnames.append(dname)
        dirnames[:] = safe_dirnames

        jpg_names = sorted(f for f in filenames if f.lower().endswith(".jpg"))
        if not jpg_names:
            continue

        valid_jpg_paths = [
            dir_p / fname
            for fname in jpg_names
            if _is_contained_non_symlink(dir_p / fname, root)
        ]
        if not valid_jpg_paths:
            continue

        rel_dir = dir_p.relative_to(root)
        if ".." in rel_dir.parts:
            continue
        folder_chain = rel_dir.as_posix()

        raw_weight_g = parse_folder_weight_g(dir_p.name)
        weight_g = LCD_CORRECTIONS.get(folder_chain, raw_weight_g)
        is_multi_bird = folder_chain in MULTI_BIRD_FOLDERS
        is_severe_mislabel = folder_chain in SEVERE_MISLABEL_FOLDERS
        is_single_bird_clean = (not is_multi_bird) and (not is_severe_mislabel)
        if is_multi_bird:
            quarantine_reason: str | None = "multi_bird"
        elif is_severe_mislabel:
            quarantine_reason = "severe_mislabel"
        else:
            quarantine_reason = None

        was_nested_folder = "/" in folder_chain
        was_lcd_corrected = folder_chain in LCD_CORRECTIONS
        weight_band = assign_weight_band(weight_g)

        for fp in valid_jpg_paths:
            rel_fp = fp.relative_to(root)
            if ".." in rel_fp.parts:
                continue
            rel_path = rel_fp.as_posix()

            decoded = decode_filename(fp.name)
            if decoded.capture_dt_from_name is not None:
                capture_dt = decoded.capture_dt_from_name
            else:
                # Verbatim Audit §9 calibration: naive local mtime minus 14,401 s.
                mtime_dt = datetime.fromtimestamp(fp.stat().st_mtime)
                capture_dt = mtime_dt - timedelta(seconds=MTIME_OFFSET_SEC)

            exif_meta = read_exif_metadata(fp)
            rows.append(
                {
                    "image_id": compute_image_id(rel_path),
                    "rel_path": rel_path,
                    "folder_chain": folder_chain,
                    "raw_weight_g": raw_weight_g,
                    "weight_g": weight_g,
                    "capture_dt": capture_dt,
                    "capture_date": capture_dt.date(),
                    "oriented_w": exif_meta.oriented_w,
                    "oriented_h": exif_meta.oriented_h,
                    "exif_orientation": exif_meta.exif_orientation,
                    "focal_length_mm": exif_meta.focal_length_mm,
                    "zoom_multiple": exif_meta.zoom_multiple,
                    "is_multi_bird": is_multi_bird,
                    "is_severe_mislabel": is_severe_mislabel,
                    "is_single_bird_clean": is_single_bird_clean,
                    "session_id": "",
                    "filename_pattern": decoded.filename_pattern,
                    "was_cp866_recovered": decoded.was_cp866_recovered,
                    "was_nested_folder": was_nested_folder,
                    "was_lcd_corrected": was_lcd_corrected,
                    "lens_class": classify_lens(exif_meta.focal_length_mm),
                    "weight_band": weight_band,
                    "quarantine_reason": quarantine_reason,
                    "fold": np.int8(-1),
                }
            )

    if not rows:
        return enforce_manifest_dtypes(pd.DataFrame(columns=list(MANIFEST_COLUMNS)))

    df = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    df["capture_dt"] = pd.to_datetime(df["capture_dt"], errors="raise").astype("datetime64[us]")
    df = df.sort_values(_SORT_KEYS, kind="mergesort").reset_index(drop=True)

    # Cluster sequential captures inside each folder_chain (strict gap > 600.0 s).
    session_ids: list[str] = []
    for folder_chain, group in df.groupby("folder_chain", sort=False):
        sid = 0
        prev_t: pd.Timestamp | None = None
        for ts in group["capture_dt"]:
            if prev_t is not None and (ts - prev_t).total_seconds() > SESSION_GAP_SECONDS:
                sid += 1
            session_ids.append(f"{folder_chain}__s{sid}")
            prev_t = ts

    df.loc[:, "session_id"] = session_ids
    return enforce_manifest_dtypes(df)


def split_clean_and_quarantine(full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition full manifest into independent clean and quarantine DataFrames."""
    typed_df = enforce_manifest_dtypes(full_df)
    clean_mask = typed_df["is_single_bird_clean"].to_numpy(dtype=bool)

    clean_df = (
        typed_df.loc[clean_mask]
        .sort_values(_SORT_KEYS, kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    quarantine_df = (
        typed_df.loc[~clean_mask]
        .sort_values(_SORT_KEYS, kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    if not quarantine_df.empty:
        quarantine_df.loc[:, "fold"] = np.int8(-1)

    return enforce_manifest_dtypes(clean_df), enforce_manifest_dtypes(quarantine_df)


def write_manifest_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a 24-column manifest strictly as a typed PyArrow Parquet file (never CSV)."""
    typed_df = enforce_manifest_dtypes(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    typed_df.to_parquet(path, engine="pyarrow", index=False, schema=_MANIFEST_ARROW_SCHEMA)
