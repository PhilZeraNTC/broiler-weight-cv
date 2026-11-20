"""Public package barrel for AvianScale CV forensic ingestion and anti-leakage splits."""

from __future__ import annotations

from backend.app.ingestion.decoding import (
    ARABIC_TO_ASCII,
    BAND_LIGHT_MAX_EXCLUSIVE_G,
    BAND_MEDIUM_MAX_INCLUSIVE_G,
    LCD_CORRECTIONS,
    MTIME_OFFSET_SEC,
    MULTI_BIRD_FOLDERS,
    SESSION_GAP_SECONDS,
    SEVERE_MISLABEL_FOLDERS,
    WIDE_FOCAL_THRESHOLD_MM,
    DecodedFilename,
    assign_weight_band,
    classify_lens,
    compute_image_id,
    decode_filename,
    parse_folder_weight_g,
)
from backend.app.ingestion.manifest import (
    DERIVED_COLUMNS,
    MANIFEST_COLUMNS,
    RAW_BASE_COLUMNS,
    enforce_manifest_dtypes,
    scan_dataset_records,
    split_clean_and_quarantine,
    write_manifest_parquet,
)
from backend.app.ingestion.splits import (
    MIN_HEAVY_SESSIONS_PER_FOLD,
    N_SPLITS,
    RANDOM_STATE,
    SORT_KEYS,
    assign_session_folds,
    verify_fold_integrity,
)

__all__ = [
    "ARABIC_TO_ASCII",
    "BAND_LIGHT_MAX_EXCLUSIVE_G",
    "BAND_MEDIUM_MAX_INCLUSIVE_G",
    "DERIVED_COLUMNS",
    "DecodedFilename",
    "LCD_CORRECTIONS",
    "MANIFEST_COLUMNS",
    "MIN_HEAVY_SESSIONS_PER_FOLD",
    "MTIME_OFFSET_SEC",
    "MULTI_BIRD_FOLDERS",
    "N_SPLITS",
    "RANDOM_STATE",
    "RAW_BASE_COLUMNS",
    "SESSION_GAP_SECONDS",
    "SEVERE_MISLABEL_FOLDERS",
    "SORT_KEYS",
    "WIDE_FOCAL_THRESHOLD_MM",
    "assign_session_folds",
    "assign_weight_band",
    "classify_lens",
    "compute_image_id",
    "decode_filename",
    "enforce_manifest_dtypes",
    "parse_folder_weight_g",
    "scan_dataset_records",
    "split_clean_and_quarantine",
    "verify_fold_integrity",
    "write_manifest_parquet",
]
