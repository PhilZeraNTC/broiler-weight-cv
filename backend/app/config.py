"""Canonical repository-root-relative filesystem paths for AvianScale CV.

All paths are anchored to ``Path(__file__).resolve().parents[2]`` so CLI
scripts, pytest runs, and future FastAPI workers resolve identical paths
regardless of the process current working directory.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATASET_DIR: Path = DATA_DIR / "raw" / "Broiler weights of Cobb Avian"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = REPO_ROOT / "models"
CLEAN_MANIFEST_PATH: Path = PROCESSED_DIR / "manifest_clean.parquet"
QUARANTINE_MANIFEST_PATH: Path = PROCESSED_DIR / "manifest_quarantine.parquet"
AUDIT_SUMMARY_PATH: Path = MODELS_DIR / "dataset_audit_summary.json"
FEATURES_CLEAN_PATH: Path = PROCESSED_DIR / "features_clean.parquet"
PDI_CALIBRATION_AUDIT_PATH: Path = MODELS_DIR / "pdi_calibration_audit.json"
MODEL_BUNDLE_PATH: Path = MODELS_DIR / "weight_regressor.joblib"
EVALUATION_REPORT_PATH: Path = MODELS_DIR / "evaluation_report.json"
PRESETS_CATALOG_PATH: Path = MODELS_DIR / "presets_catalog.json"

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "RAW_DATASET_DIR",
    "PROCESSED_DIR",
    "MODELS_DIR",
    "CLEAN_MANIFEST_PATH",
    "QUARANTINE_MANIFEST_PATH",
    "AUDIT_SUMMARY_PATH",
    "FEATURES_CLEAN_PATH",
    "PDI_CALIBRATION_AUDIT_PATH",
    "MODEL_BUNDLE_PATH",
    "EVALUATION_REPORT_PATH",
    "PRESETS_CATALOG_PATH",
]
