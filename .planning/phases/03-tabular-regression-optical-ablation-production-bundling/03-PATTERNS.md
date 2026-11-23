# Phase 3: Tabular Regression, Optical Ablation & Production Bundling - Pattern Map

**Mapped:** 2026-11-23
**Files analyzed:** 12
**Analogs found:** 10 / 12

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `backend/app/config.py` *(modify)* | config | file-I/O | `backend/app/config.py` | exact |
| `backend/app/modeling/__init__.py` *(new)* | config | transform | `backend/app/pdi/__init__.py` | exact |
| `backend/app/modeling/age_stage.py` *(new)* | utility | transform | `backend/app/ingestion/decoding.py` (+ `backend/app/pdi/config.py`) | exact |
| `backend/app/modeling/datasets.py` *(new)* | service | file-I/O | `backend/app/ingestion/splits.py` (+ `backend/app/pdi/features.py`, `backend/scripts/extract_features.py`) | role-match |
| `backend/app/modeling/wrappers.py` *(new)* | model | transform | *(none — first scikit-learn custom estimator/transformer domain; see Shared Patterns + `03-RESEARCH.md`)* | none |
| `backend/app/modeling/models.py` *(new)* | model | transform | *(none — first tabular regressor factory & fold sample-weighting domain; see Shared Patterns + `03-RESEARCH.md`)* | none |
| `backend/app/modeling/evaluation.py` *(new)* | service | batch | `backend/scripts/extract_features.py` (+ `backend/app/ingestion/audit_summary.py`, `backend/app/ingestion/splits.py`) | role-match |
| `backend/app/modeling/artifacts.py` *(new)* | service | file-I/O | `backend/app/ingestion/audit_summary.py` (+ `backend/scripts/extract_features.py`, `backend/app/pdi/config.py`, `backend/app/pdi/watershed.py`) | exact |
| `backend/scripts/train_and_evaluate.py` *(new)* | utility | batch | `backend/scripts/extract_features.py` | exact |
| `backend/tests/test_modeling_wrappers_and_age_stage.py` *(new)* | test | transform | `backend/tests/test_io_exif.py` (+ `backend/tests/test_extract_features_integration.py`) | exact |
| `backend/tests/test_modeling_datasets_and_ablation.py` *(new)* | test | transform | `backend/tests/test_splits.py` (+ `backend/tests/test_extract_features_integration.py`) | exact |
| `backend/tests/test_train_and_evaluate_integration.py` *(new)* | test | batch | `backend/tests/test_extract_features_integration.py` (+ `backend/tests/test_io_exif.py`) | exact |

*(Generated runtime artifacts — not hand-edited source modules: git-ignored `models/weight_regressor.joblib` serialized atomically via `joblib.dump(..., compress=3)` + `os.replace()`, and git-tracked, byte-deterministic, zero-path-leakage `models/evaluation_report.json` and `models/presets_catalog.json` serialized via the `PROIBIÇÃO 4` JSON writer pattern upgraded to atomic `os.replace()`.)*

## Pattern Assignments

### `backend/app/config.py` (config, file-I/O) — MODIFY

**Analog:** `backend/app/config.py` (self-extension)

**Path constant & `__all__` export pattern** (`backend/app/config.py` lines 8-34):
```python
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
]
```
**Planner guidance:** Append the three Phase 3 canonical `Path` constants (`D-11`):
- `MODEL_BUNDLE_PATH: Path = MODELS_DIR / "weight_regressor.joblib"`
- `EVALUATION_REPORT_PATH: Path = MODELS_DIR / "evaluation_report.json"`
- `PRESETS_CATALOG_PATH: Path = MODELS_DIR / "presets_catalog.json"`
Add all three symbols to `__all__`. Never alter any existing Phase 1/2 constant (guarded by `backend/tests/test_io_exif.py` lines 80-108).

---

### `backend/app/modeling/__init__.py` (config, transform) — NEW

**Analog:** `backend/app/pdi/__init__.py`

**Package barrel & alphabetically sorted `__all__` pattern** (`backend/app/pdi/__init__.py` lines 1-14, 44-79):
```python
"""Public package barrel for AvianScale CV classical PDI and Watershed engines."""

from __future__ import annotations

from backend.app.pdi.config import PDIConfig
from backend.app.pdi.features import (
    BANNED_FEATURE_COLUMNS,
    CHROMATIC_TEXTURE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MORPHOMETRIC_FEATURE_COLUMNS,
    OPTICAL_FEATURE_COLUMNS,
    extract_mask_gated_features,
)
...
__all__ = [
    "BANNED_FEATURE_COLUMNS",
    "CHROMATIC_TEXTURE_FEATURE_COLUMNS",
    ...
    "suppress_lcd_and_platter",
]
```
**Planner guidance:** Re-export the public modeling API across all 6 submodules (`AgeStageResult`, `map_weight_to_age_stage`, `ALLOMETRIC_LOG_COLUMNS`, `AllometricLogTransformer`, `LogDuanRegressor`, `ABLATION_CONDITION_ORDER`, `LoadedModelingDataset`, `load_modeling_dataset`, `MODEL_FAMILY_NAMES`, `compute_sample_weights`, `build_regressor_pipeline`, `predict_from_feature_dict`, `run_full_evaluation`, `MODEL_BUNDLE_SCHEMA_VERSION`, `EVALUATION_REPORT_SCHEMA_VERSION`, `PRESETS_CATALOG_SCHEMA_VERSION`, `build_model_bundle`, `save_model_bundle`, `load_model_bundle`, `write_evaluation_report`, `build_presets_catalog`, `write_presets_catalog`) with explicit alphabetically sorted `__all__` and `from __future__ import annotations`.

---

### `backend/app/modeling/age_stage.py` (utility, transform) — NEW

**Analog:** `backend/app/ingestion/decoding.py` (frozen result dataclass & domain weight-band mapper) + `backend/app/pdi/config.py` (ASVS V5 finite-float/bool guard)

**Frozen slotted dataclass & pure threshold mapper pattern** (`backend/app/ingestion/decoding.py` lines 27-29, 37-44, 111-117):
```python
WIDE_FOCAL_THRESHOLD_MM: float = 3.0
BAND_LIGHT_MAX_EXCLUSIVE_G: int = 500
BAND_MEDIUM_MAX_INCLUSIVE_G: int = 1000
...
@dataclass(frozen=True, slots=True)
class DecodedFilename:
    """Result of decoding a raw broiler image filename under Audit §9 rules."""

    capture_dt_from_name: datetime | None
    filename_pattern: str
    was_cp866_recovered: bool
...
def assign_weight_band(weight_g: int) -> str:
    """Map integer gram weight into the 3 coarse PRD weight bands (D-05)."""
    if weight_g < BAND_LIGHT_MAX_EXCLUSIVE_G:
        return "<500g"
    if weight_g <= BAND_MEDIUM_MAX_INCLUSIVE_G:
        return "500-1000g"
    return ">1000g"
```

**Strict ASVS V5 finite-float & non-bool validation guard** (`backend/app/pdi/config.py` lines 41-65):
```python
def _validate_finite_float(
    name: str,
    value: object,
    *,
    min_value: float | None = None,
    min_inclusive: bool = True,
    max_value: float | None = None,
    max_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None:
        if min_inclusive and numeric < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {value!r}")
        if not min_inclusive and numeric <= min_value:
            raise ValueError(f"{name} must be > {min_value}, got {value!r}")
    ...
    return numeric
```
**Planner guidance:** Keep `age_stage.py` pure stdlib (zero OpenCV/sklearn/pandas imports so Phase 4 services can import it cheaply). Define constants `VALID_TRAINING_WEIGHT_MIN_G: int = 116`, `VALID_TRAINING_WEIGHT_MAX_G: int = 2093`, `VALID_AGE_DAYS_RANGE: str = "~7-29 days"`, frozen slotted `@dataclass(frozen=True, slots=True) class AgeStageResult`, and `map_weight_to_age_stage(weight_g: float) -> AgeStageResult` implementing the exact 4 `D-08` Cobb Avian 48 thresholds (`< 350.0`, `350.0..700.0`, `700.0..1200.0`, `> 1200.0`) and `out_of_training_range = (w < 116.0) or (w > 2093.0)`. Reject `bool`, `NaN`, `Inf`, and `<= 0.0` via `ValueError` (ASVS V5).

---

### `backend/app/modeling/datasets.py` (service, file-I/O) — NEW

**Analog:** `backend/app/ingestion/splits.py` (fold integrity gate & required-column validation) + `backend/app/pdi/features.py` (canonical feature tuple constants) + `backend/scripts/extract_features.py` (1:1 alignment & finite feature matrix guards)

**Canonical feature tuple imports (zero string duplication)** (`backend/app/pdi/features.py` lines 39-103):
```python
MORPHOMETRIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "area_px",
    "perimeter_px",
    "major_axis_px",
    "minor_axis_px",
    "convex_area_px",
    "area_frac",
    "aspect_ratio_ellipse",
    "eccentricity",
    "solidity",
    "circularity",
    "extent_rotrect",
    "log_hu_1",
    "log_hu_2",
    "log_hu_3",
)
CHROMATIC_TEXTURE_FEATURE_COLUMNS: tuple[str, ...] = ( ... )  # 26 cols
OPTICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "focal_length_mm",
    "zoom_multiple",
    "exif_optical_present",
    "area_optical_norm",
    "sqrt_area_optical_norm",
    "perim_optical_norm",
    "major_axis_optical_norm",
    "minor_axis_optical_norm",
    "area_frac_optical_norm",
)
FEATURE_COLUMNS: tuple[str, ...] = (
    MORPHOMETRIC_FEATURE_COLUMNS + CHROMATIC_TEXTURE_FEATURE_COLUMNS + OPTICAL_FEATURE_COLUMNS
)
BANNED_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {"oriented_w", "oriented_h", "megapixels", "bbox_w", "bbox_h", "raw_w", "raw_h"}
)
```

**Fold integrity & zero-session-overlap gate** (`backend/app/ingestion/splits.py` lines 35-85, 133-136):
```python
def verify_fold_integrity(
    clean_df: pd.DataFrame,
    *,
    min_heavy_sessions_per_fold: int | None = None,
) -> dict[str, Any]:
    """Audit 5-fold session assignments for zero leakage, completeness, and heavy balance."""
    missing = sorted(_VERIFY_REQUIRED_COLS - set(clean_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for fold verification: {missing}")
    ...
    return {
        "cross_fold_session_overlap": 0,
        "folds": fold_summaries,
    }
```

**1:1 alignment, `pdi_config_hash` lock & finite matrix guards** (`backend/scripts/extract_features.py` lines 258-260, 528-533):
```python
    feature_matrix = out.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError("Non-finite (NaN or Inf) value detected in extracted feature matrix.")
...
    expected_ids = manifest_df["image_id"].astype(str).tolist()
    actual_ids = features_df["image_id"].astype(str).tolist()
    if actual_ids != expected_ids:
        raise RuntimeError(
            "Extracted feature rows failed exact 1:1 image_id alignment with manifest."
        )
```
**Planner guidance:** Define `@dataclass(frozen=True, slots=True) class LoadedModelingDataset` holding validated `df`, `pdi_config`, `pdi_config_hash`, `fold_audit`, and `ablation_columns`. Derive the 4 ablation column tuples programmatically from `backend.app.pdi.features` (`C1_uncalibrated_pdi`: 40 cols; `C2_physics_normalized_strict`: 39 cols = 8 dimensionless shape + 26 chromatic/texture + 5 `*_optical_norm`; `C3_physics_plus_exif_meta`: 49 cols = `FEATURE_COLUMNS`; `C4_naive_double_count_fxz`: 39 cols with offline-only `area_naive_fxz`..`minor_naive_fxz` derived on an isolated copy inside `datasets.py`). Assert at module load AND dataset load: (1) zero `BANNED_FEATURE_COLUMNS` or non-feature audit columns (`scene_type`, `scene_is_bucket`, `lcd_excluded_px`, `segmentation_fallback`, `lens_class`, `fold`, `session_id`, `weight_g`, `weight_band`, `image_id`, `rel_path`) inside any ablation feature list (`D-02`); (2) `C2` strictly disjoint from raw EXIF/area columns (`focal_length_mm`, `zoom_multiple`, `exif_optical_present`, `area_px`, `perimeter_px`, `major_axis_px`, `minor_axis_px`, `convex_area_px`, `area_frac`, `area_frac_optical_norm`); (3) exact 1:1 row parity between `manifest_clean.parquet` and `features_clean.parquet` on `("image_id", "fold", "session_id", "weight_g", "weight_band", "lens_class")`; (4) uniform `pdi_config_hash == cfg.config_hash()` (`"43a3cfaa3208019e"`); (5) `verify_fold_integrity(df, min_heavy_sessions_per_fold=MIN_HEAVY_SESSIONS_PER_FOLD)`. Always use CoW-safe `df.copy()` and single-step `.loc[:, col] = ...`.

---

### `backend/app/modeling/evaluation.py` (service, batch) — NEW

**Analog:** `backend/scripts/extract_features.py` (paired Wide/Ultrawide within-session groupby evaluation) + `backend/app/ingestion/audit_summary.py` (deterministic stratified slicing) + `backend/app/ingestion/splits.py` (5-fold contract)

**Deterministic paired Wide/Ultrawide within-session evaluation pattern** (`backend/scripts/extract_features.py` lines 286-337):
```python
def _evaluate_paired_scope(df: pd.DataFrame, group_cols: list[str]) -> dict[str, Any]:
    """Evaluate paired Wide/Ultrawide within-group median ratios across all 6 candidate formulas."""
    ...
    if not df.empty:
        lens_arr = df["lens_class"].astype(str).to_numpy()
        grouped_indices = df.groupby(group_cols, sort=True).indices

        for _, raw_idx in sorted(grouped_indices.items(), key=lambda kv: kv[0]):
            idx = np.asarray(raw_idx, dtype=np.int64)
            grp_lens = lens_arr[idx]
            w_idx = idx[grp_lens == "wide"]
            u_idx = idx[grp_lens == "ultrawide"]
            if w_idx.size == 0 or u_idx.size == 0:
                continue
            ...
```

**Deterministic ordered stratum slicing pattern** (`backend/app/ingestion/audit_summary.py` lines 49, 229-251):
```python
_WEIGHT_BAND_ORDER: tuple[str, ...] = ("<500g", "500-1000g", ">1000g")
...
    weight_bands: dict[str, dict[str, Any]] = {}
    for band in _WEIGHT_BAND_ORDER:
        band_df = typed_clean.loc[typed_clean["weight_band"] == band]
        ...
    lenses: dict[str, dict[str, Any]] = {}
    for lens_name in ("wide", "ultrawide", "unknown"):
        lens_df = typed_clean.loc[typed_clean["lens_class"] == lens_name]
        ...
```
**Planner guidance:** Mirror `_evaluate_paired_scope` (`df.groupby(["session_id"], sort=True).indices`, sorted keys, `w_idx` vs `u_idx` mean/median OOF predictions) to compute within-session dual-lens consistency across all `127` mixed-lens sessions (`|ŷ_wide - ŷ_uw|` in g and `%`, plus signed `ŷ_wide - ŷ_uw`). Mirror `_WEIGHT_BAND_ORDER` (`"<500g"`, `"500-1000g"`, `">1000g"`), `("wide", "ultrawide")`, and `("bucket_dorsal", "scale_or_litter")` for deterministic stratified slices (`global`, `session_level`, `per_fold`, `by_weight_band`, `by_lens`, `by_scene`, `by_band_and_scene`, `by_band_and_lens`). For 5-fold CV training, unweighted OOF metrics (`sklearn.metrics.mean_absolute_error`, `root_mean_squared_error`, `r2_score`), 2-mode counterfactual lens-flip (`focal_only` + `physical_lens_swap`), `<500g` ultrawide anti-confounding gate, `2×2` weighting×smearing sensitivity, and `D-06` deterministic winner selection, follow `03-RESEARCH.md` lines 335-353 and 415-420. Always copy validation slices (`X_cf = X_val.copy()`) and mutate via `.loc[mask, col] = ...` (Pandas 3.1 CoW).

---

### `backend/app/modeling/artifacts.py` (service, file-I/O) — NEW

**Analog:** `backend/app/ingestion/audit_summary.py` (byte-deterministic `PROIBIÇÃO 4` JSON serialization) + `backend/scripts/extract_features.py` (ASVS V4 path containment & `_json_normalize`) + `backend/app/pdi/watershed.py` (`run_watershed_pdi` contract)

**`PROIBIÇÃO 4` leak guard & deterministic JSON serializer** (`backend/scripts/extract_features.py` lines 44, 209-215, 406-417):
```python
_FORBIDDEN_LEAK_SUBSTRINGS: tuple[str, ...] = (".jpg", "rel_path", "/home/", "data/raw")
...
def _json_normalize(value: Any) -> Any:
    """Convert dataclass tuples into JSON-native lists for deterministic equality."""
    if isinstance(value, dict):
        return {str(k): _json_normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(v) for v in value]
    return value
...
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
```

**ASVS V4 non-symlink path containment for Watershed preset verification** (`backend/scripts/extract_features.py` lines 132-173):
```python
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
```

**Isolated Watershed preset audit call contract** (`backend/app/pdi/watershed.py` lines 89-120, 435-444):
```python
@dataclass(frozen=True, slots=True)
class WatershedInstance:
    instance_id: int
    centroid_xy: tuple[float, float]
    area_work_px: int
    area_px: float
    area_frac: float
    area_optical_norm: float
    bbox_work: tuple[int, int, int, int]
    features: dict[str, float]
...
def run_watershed_pdi(
    source: Path | bytes,
    *,
    config: PDIConfig | None = None,
    optical_override: OpticalOverride | None = None,
    h_rel: float | None = None,
    min_distance_rel: float | None = None,
    min_area_ratio: float | None = None,
    return_visuals: bool = True,
) -> WatershedResult: ...
```
**Planner guidance:**
1. **Expand leak guard (`03-RESEARCH.md` line 399):** Use `_FORBIDDEN_LEAK_SUBSTRINGS: tuple[str, ...] = (".jpg", ".jpeg", ".png", "rel_path", "/home/", "data/raw")` on both `write_evaluation_report()` and `write_presets_catalog()`. Never emit volatile timestamps or wall-clock runtimes inside either JSON payload.
2. **Upgrade writers to atomic same-directory temp file + `os.replace()` (`D-11`):** Unlike `audit_summary.py`/`extract_features.py` (which call `path.write_text`), write `models/weight_regressor.joblib` (`joblib.dump(bundle, tmp_path, compress=3)`), `models/evaluation_report.json`, and `models/presets_catalog.json` via `tempfile.NamedTemporaryFile(dir=path.parent, delete=False)` + flush/close + `os.replace(tmp_path, path)` with `try/except` cleanup of `tmp_path` on failure.
3. **Dual-bundle contract (`D-07`):** Implement `build_model_bundle()`, `save_model_bundle()`, and `load_model_bundle()` validating `schema_version == "1.0.0"`, `pdi_config_hash == PDIConfig().config_hash()`, exact required keys, `set(fold_pipelines.keys()) == {0, 1, 2, 3, 4}`, and `tuple` `feature_names`.
4. **Opaque-ID Presets (`D-10`):** Select 6 median-tier (`pct_err >= 3.0%`, `segmentation_fallback == False`, spanning all 5 folds including a same-session Wide/UW pair) single-bird OOF presets + 4 Watershed presets across `MULTI_BIRD_FOLDERS` (`417`, `757`, `823`, `1520`) via `run_watershed_pdi(safe_path, config=cfg, return_visuals=False)` sorted deterministically by `(abs(detected - expected), -(min_area / max_area), image_id)`. Store **ONLY opaque `preset_id` + `image_id`** (never `rel_path` or `.jpg`).

---

### `backend/scripts/train_and_evaluate.py` (utility, batch) — NEW

**Analog:** `backend/scripts/extract_features.py`

**CLI argparse, orchestrator, aggregate-only stdout & entrypoint pattern** (`backend/scripts/extract_features.py` lines 13-40, 420-477, 537-594):
```python
from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
...
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for capped-concurrency PDI batch feature extraction."""
    parser = argparse.ArgumentParser(
        description=(...),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=CLEAN_MANIFEST_PATH,
        help="Path to materialized manifest_clean.parquet.",
    )
    ...
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full extraction and calibration audit without writing any files.",
    )
    args = parser.parse_args(argv)
    return args
...
    if not dry_run:
        write_features_parquet(features_df, output_path)
        write_calibration_audit(audit, audit_path)

    wall_clock_sec = time.perf_counter() - t0
    ...
    # PROIBIÇÃO 4: emit strictly aggregate metrics — never raw filenames or host paths.
    print(
        "Feature extraction complete: "
        f"n_clean_images={summary['n_clean_images']} "
        ...
        f"pdi_config_hash={audit['pdi_config_hash']} "
        f"wall_clock_sec={wall_clock_sec:.2f} "
        f"dry_run={dry_run}"
    )
...
def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    run_extract_features(...)
    return 0


__all__ = [...]


if __name__ == "__main__":
    raise SystemExit(main())
```
**Planner guidance:** Mirror `extract_features.py` verbatim for `train_and_evaluate.py`:
- CLI flags defaulting strictly to `backend.app.config` repo-root-anchored constants: `--manifest-path` (`CLEAN_MANIFEST_PATH`), `--quarantine-manifest-path` (`QUARANTINE_MANIFEST_PATH`), `--features-path` (`FEATURES_CLEAN_PATH`), `--dataset-root` (`RAW_DATASET_DIR`), `--bundle-path` (`MODEL_BUNDLE_PATH`), `--report-path` (`EVALUATION_REPORT_PATH`), `--presets-path` (`PRESETS_CATALOG_PATH`), `--n-jobs` (clamped `1..4` via `_resolve_effective_n_jobs`, default `4`), `--dry-run` (`action="store_true"`).
- **Critical:** Keep ALL custom estimators/transformers (`LogDuanRegressor`, `AllometricLogTransformer`) imported from `backend.app.modeling.wrappers` — NEVER define classes inside `train_and_evaluate.py` (`__main__` pickling hazard).
- On `--dry-run`, run full dataset verification, 5-fold CV, 4-condition ablation, counterfactual checks, bundle construction, and preset selection without creating any output directories or files.
- Log `wall_clock_sec` ONLY to stdout (never inside git-tracked JSON files), keeping stdout strictly aggregate-only (`PROIBIÇÃO 4`).

---

### `backend/tests/test_modeling_wrappers_and_age_stage.py` (test, transform) — NEW

**Analog:** `backend/tests/test_io_exif.py` (parametrized boundary & invalid-input rejection) + `backend/tests/test_extract_features_integration.py` (subprocess verification style)

**Invalid-input & boundary assertion style** (`backend/tests/test_io_exif.py` lines 205-215):
```python
def test_parse_focal_length_handles_zero_denominator_and_invalid_inputs() -> None:
    assert _parse_focal_length(IFDRational(425, 100)) == pytest.approx(4.25)
    ...
    assert _parse_focal_length(-4.25) is None
    assert _parse_focal_length(0.0) is None
    assert _parse_focal_length(math.nan) is None
    assert _parse_focal_length(math.inf) is None
    assert _parse_focal_length(True) is None
```

**Clean subprocess verification style** (`backend/tests/test_extract_features_integration.py` lines 628-634):
```python
    tracked = subprocess.run(
        ["git", "ls-files", "data/raw", "data/processed"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
```
**Planner guidance:** Test fast synthetic data only (zero `data/raw/` reads):
1. `LogDuanRegressor`: exact `log1p`/`expm1` round-trip, fold-internal Duan's smearing (`> 1.0` on noisy log residuals when `use_smearing=True`, exact `1.0` when `False`), `Pipeline` `reg__sample_weight` forwarding, rejection of `<= 0` / `NaN` / `Inf` targets, and **fresh subprocess** `subprocess.run([sys.executable, "-c", ...], cwd=tmp_path, check=True)` executing `joblib.load()` + single-row prediction (`< 5 ms`, zero `UserWarning`s via `-W error::UserWarning`).
2. `AllometricLogTransformer`: transforms strictly `ALLOMETRIC_LOG_COLUMNS`, leaves chromatic/dimensionless columns bit-identical, clips negatives to `0.0`.
3. `compute_sample_weights()`: verifies `max_w / min_w <= 5.0 + 1e-9`, `abs(mean - 1.0) < 1e-9`, positive finite floats, invalid `max_weight_ratio < 1.0` rejection, and determinism.
4. `map_weight_to_age_stage()`: parametrized assertions across all 4 Cobb Avian 48 boundaries (`115.0`, `116.0`, `349.9`, `350.0`, `700.0`, `700.1`, `1200.0`, `1200.1`, `2093.0`, `2094.0`) + `pytest.raises(ValueError)` on `True`, `False`, `0.0`, `-10.0`, `math.nan`, `math.inf`, `"500"`.

---

### `backend/tests/test_modeling_datasets_and_ablation.py` (test, transform) — NEW

**Analog:** `backend/tests/test_splits.py` (synthetic dataset builder, injected leakage failure gates & AST splitter prohibition scan) + `backend/tests/test_extract_features_integration.py` (schema/banned-column checks)

**Injected leakage & integrity failure assertions** (`backend/tests/test_splits.py` lines 236-264):
```python
def test_verify_fold_integrity_catches_injected_leakage_and_invalid_folds() -> None:
    ...
    tampered = assigned.copy()
    target_sid = str(tampered.iloc[0]["session_id"])
    sid_indices = tampered.index[tampered["session_id"] == target_sid].tolist()
    orig_fold = int(tampered.loc[sid_indices[0], "fold"])
    tampered.loc[sid_indices[0], "fold"] = np.int8((orig_fold + 1) % N_SPLITS)

    with pytest.raises(RuntimeError, match="Session leakage"):
        verify_fold_integrity(tampered)
```

**AST anti-leakage splitter prohibition scan (`PROIBIÇÃO 3`)** (`backend/tests/test_splits.py` lines 266-300):
```python
def test_ast_scan_prohibits_random_and_ungrouped_splitters_in_ingestion() -> None:
    ingestion_dir = REPO_ROOT / "backend" / "app" / "ingestion"
    py_files = sorted(ingestion_dir.rglob("*.py"))
    assert py_files

    banned_splitters = {
        "train_test_split",
        "KFold",
        "StratifiedKFold",
        "ShuffleSplit",
        "StratifiedShuffleSplit",
        "GroupShuffleSplit",
    }
    violations: list[str] = []
    for py_file in py_files:
        rel_file = py_file.relative_to(REPO_ROOT)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in banned_splitters:
                        violations.append(f"{rel_file}:{node.lineno}:import {alias.name}")
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in banned_splitters:
                    violations.append(f"{rel_file}:{node.lineno}:call {func.id}")
                elif isinstance(func, ast.Attribute) and func.attr in banned_splitters:
                    violations.append(f"{rel_file}:{node.lineno}:call {func.attr}")

    assert not violations, f"PROIBIÇÃO 3 violation — banned splitters found: {violations}"
```
**Planner guidance:** Mirror both patterns for `backend/app/modeling/` and `backend/scripts/train_and_evaluate.py`:
1. Extend the AST scan over `backend/app/modeling/*.py` and `backend/scripts/train_and_evaluate.py` prohibiting both `PROIBIÇÃO 3` banned splitters AND `PROIBIÇÃO 1` deep learning / OCR imports (`torch`, `tensorflow`, `keras`, `ultralytics`, `timm`, `onnxruntime`, `paddleocr`, `easyocr`).
2. Assert exact ablation column counts (`C1=40`, `C2=39`, `C3=49`, `C4=39`), zero `BANNED_FEATURE_COLUMNS`, zero audit/background columns (`scene_type`, `scene_is_bucket`, `lcd_excluded_px`, `segmentation_fallback`, `lens_class`, `fold`, `session_id`, `rel_path`), and `C2` disjoint from raw EXIF/area columns.
3. Inject mismatched `pdi_config_hash`, misaligned `image_id`, `NaN` feature, and cross-fold `session_id` overlap into synthetic Parquet fixtures — asserting loud `ValueError`/`RuntimeError` rejection.
4. Build a synthetic lens-confounded dataset where `C3`/`C4` models cheat on `ultrawide ↔ >1000g` while `C2` stays strictly lens-invariant (`0.00%` counterfactual shift) — asserting the dual-mode gate sets `lens_confounding_detected = True` on `C3`/`C4` and selects `C2`.

---

### `backend/tests/test_train_and_evaluate_integration.py` (test, batch) — NEW

**Analog:** `backend/tests/test_extract_features_integration.py` (CLI `--dry-run`, `PROIBIÇÃO 4` leak tests, SHA-256 determinism, real-corpus gates & git hygiene) + `backend/tests/test_io_exif.py` (config path CWD-independence)

**CLI `--dry-run`, aggregate-only stdout & SHA-256 byte-determinism pattern** (`backend/tests/test_extract_features_integration.py` lines 37, 326-363, 463-465):
```python
_FORBIDDEN_LEAK_MARKERS: tuple[str, ...] = (".jpg", "rel_path", "/home/", "data/raw")
...
def test_cli_dry_run_writes_zero_files_and_emits_zero_jpg_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ...
    rc = extract_features.main([... , "--dry-run"])
    captured = capsys.readouterr()

    assert rc == 0
    assert not out_parquet.exists()
    assert not out_parquet.parent.exists()
    assert not out_audit.exists()
    assert not out_audit.parent.exists()

    # PROIBIÇÃO 4: stdout logs strictly aggregate metrics, zero filenames/paths.
    lowered_out = captured.out.lower()
    for forbidden in _FORBIDDEN_LEAK_MARKERS:
        assert forbidden not in lowered_out
...
    digest_1 = hashlib.sha256(out_audit_1.read_bytes()).hexdigest()
    digest_2 = hashlib.sha256(out_audit_2.read_bytes()).hexdigest()
    assert digest_1 == digest_2
```

**Defense-in-depth `PROIBIÇÃO 4` rejection & git hygiene pattern** (`backend/tests/test_extract_features_integration.py` lines 478-485, 626-648):
```python
def test_write_calibration_audit_rejects_proibicao_4_leak_markers(
    tmp_path: Path,
    bad_payload: dict[str, Any],
) -> None:
    target = tmp_path / "rejected_audit.json"
    with pytest.raises(ValueError, match="PROIBIÇÃO 4"):
        extract_features.write_calibration_audit(bad_payload, target)
    assert not target.exists()
...
def test_git_hygiene_excludes_features_parquet_while_tracking_calibration_audit() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "data/raw", "data/processed"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout.strip() == ""

    ignored_check = subprocess.run(
        ["git", "check-ignore", "models/pdi_calibration_audit.json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored_check.returncode != 0
```
**Planner guidance:** Mirror all 5 verification gates for Phase 3 using expanded `_FORBIDDEN_LEAK_MARKERS = (".jpg", ".jpeg", ".png", "rel_path", "/home/", "data/raw")`:
1. `parse_args([])` defaults match `config.py` constants; `--n-jobs` clamps to `1..4` and rejects `<= 0`; `--dry-run` writes zero files/dirs and emits zero forbidden tokens to stdout.
2. Defense-in-depth `PROIBIÇÃO 4` rejection on `write_evaluation_report()` and `write_presets_catalog()`, plus atomic write guarantee (target untouched on failure).
3. Materialized `models/weight_regressor.joblib` gate: fresh subprocess `joblib.load()` + single-row prediction (`< 5 ms`, zero `UserWarning`s), `schema_version == "1.0.0"`, `pdi_config_hash == "43a3cfaa3208019e"`, `winning_ablation_condition == "C2_physics_normalized_strict"`, exact 5 `fold_pipelines` (`{0, 1, 2, 3, 4}`), exact 39 `feature_names`, and all 3 `band_error_intervals`.
4. Materialized `models/evaluation_report.json` & `models/presets_catalog.json` gates: zero forbidden leak substrings, exact `D-09`/`D-10` schema contracts (`16` comparison rows, `127` mixed-lens sessions, `1,738` anonymized `oof_scatter_points`, `6` non-fallback median-tier OOF presets spanning all 5 folds + paired dual-lens session, `4` Watershed presets `417`/`757`/`823`/`1520` whose `image_id`s exist in `manifest_quarantine.parquet`), and **byte-identical SHA-256** across consecutive runs.
5. Git hygiene gate: `git check-ignore models/weight_regressor.joblib` returns `0` (ignored by `.gitignore:4`), while `git check-ignore models/evaluation_report.json` and `git check-ignore models/presets_catalog.json` both return non-zero (git-trackable) and `git ls-files data/raw data/processed models/*.joblib` is empty.

---

## Shared Patterns

### 1. Module Header, Frozen Slotted Dataclasses & Ruff/Pytest Configuration
**Source:** `backend/app/pdi/config.py` lines 1-16, 68-70 & `pyproject.toml` lines 35-49
**Apply to:** All 9 production modules and 3 test modules in Phase 3
```python
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ExampleResult:
    ...
```
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["backend/tests"]
addopts = ["-ra", "-q"]

[tool.ruff]
target-version = "py312"
line-length = 100
src = ["."]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.ruff.lint.isort]
known-first-party = ["backend"]
```
- Every file opens with a module docstring + `from __future__ import annotations`.
- Every public module defines explicit `__all__`.
- Never access `self.__dict__` on `slots=True` dataclasses — use `dataclasses.asdict(self)` + `_json_normalize(...)`.
- **Zero dependency edits:** Never alter `pyproject.toml` or `uv.lock` (guarded by `backend/tests/test_io_exif.py` lines 289-309).

### 2. Atomic Disk Persistence + `PROIBIÇÃO 4` Zero-Path-Leakage Guard
**Source:** `backend/scripts/extract_features.py` lines 44, 406-417 (upgraded to atomic `os.replace()` per `D-11`)
**Apply to:** `backend/app/modeling/artifacts.py` (`save_model_bundle`, `write_evaluation_report`, `write_presets_catalog`) and `backend/scripts/train_and_evaluate.py`
```python
_FORBIDDEN_LEAK_SUBSTRINGS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    "rel_path",
    "/home/",
    "data/raw",
)
```
- Serialize JSON strictly via `json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"` with **zero volatile timestamps or wall-clock timings** inside git-tracked JSON payloads.
- Scan `serialized.lower()` against `_FORBIDDEN_LEAK_SUBSTRINGS` BEFORE creating parent directories or temp files; raise `ValueError("PROIBIÇÃO 4 violation: ...")` on any match.
- Write all 3 Phase 3 artifacts (`weight_regressor.joblib`, `evaluation_report.json`, `presets_catalog.json`) to a sibling temp file in `path.parent` and commit via `os.replace(tmp_path, path)`, unlinking `tmp_path` in `except` on failure.
- CLI stdout prints strictly aggregate counts/metrics (never raw filenames or `rel_path`).

### 3. Pandas 3.1 Copy-on-Write & Arrow `str` Discipline
**Source:** `backend/scripts/extract_features.py` lines 245-262 & `backend/app/ingestion/splits.py` lines 66-70, 178-205
**Apply to:** `datasets.py`, `models.py`, `evaluation.py`, `artifacts.py`, and all test helpers
```python
    out = df.loc[:, list(FEATURES_PARQUET_COLUMNS)].copy()
    ...
    feature_matrix = out.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
```
- Always slice with explicit column lists (`df.loc[:, list(cols)]`) and `.copy()` before mutating counterfactual or fold frames.
- Mutate exclusively with single-step `.loc[mask, col] = value` (never chained `df[mask][col] = value`, which raises `ChainedAssignmentError` in Pandas 3.1).
- Never use `df.select_dtypes(include=["object"])` (Pandas 3.1 uses Arrow-backed `str` dtype).
- Always preserve exact ordered `feature_names` when constructing single-row or batch `pd.DataFrame` inputs into `sklearn`/`LightGBM` pipelines.

### 4. Repo-Wide AST, Thread-Cap & Security Invariants (`PROIBIÇÃO 1–4`, ASVS L1)
**Source:** `backend/tests/test_io_exif.py` lines 264-309, `backend/tests/test_splits.py` lines 266-300, `backend/scripts/extract_features.py` lines 42, 125-173
**Apply to:** All modules under `backend/app/modeling/` and `backend/scripts/train_and_evaluate.py`
- **No raw OpenCV image loading:** `backend/tests/test_io_exif.py` lines 264-286 scans ALL `backend/app/**/*.py` for `imread`/`imdecode` — never introduce them anywhere under `backend/app/modeling/`.
- **Bounded CPU concurrency (`7.1 GiB` RAM host):** Cap `n_jobs` strictly to `min(n_jobs, 4)` (never `n_jobs=-1`) on `RandomForestRegressor` and `LGBMRegressor`, with `deterministic=True, force_col_wise=True, random_state=42, verbosity=-1` on `LGBMRegressor` for bit-identical repeatability.
- **ASVS V4 path containment:** When `artifacts.py` loads quarantined multi-bird images from `manifest_quarantine.parquet` for `run_watershed_pdi()`, validate every `rel_path` through `_resolve_safe_image_path(rel_path, dataset_root.resolve())` (rejecting `".."`, absolute paths, null bytes, backslashes, and symlinks) and never write `rel_path` into `models/presets_catalog.json`.
- **ASVS V5 input validation:** Explicitly reject `isinstance(x, bool)`, `NaN`/`Inf`, and out-of-range values across `age_stage.py`, `wrappers.py`, `models.py`, `datasets.py`, and `artifacts.py`.

---

## No Analog Found

Files introducing brand-new scikit-learn / LightGBM estimator and fold sample-weighting domains with no prior algorithmic analog in the Phase 1–2 codebase (planner MUST apply **Shared Patterns #1, #3, #4** for typing, Pandas 3 CoW, security, and thread-cap rules, and source the exact estimator/weighting code directly from `03-RESEARCH.md`):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `backend/app/modeling/wrappers.py` | model | transform | First custom `sklearn.base.BaseEstimator` / `TransformerMixin` / `RegressorMixin` module in the repository (`AllometricLogTransformer`, `LogDuanRegressor`, `ALLOMETRIC_LOG_COLUMNS`); must live in an importable module (never `__main__` or `lambda`) for clean Phase 4 `joblib.load()`. Follow `03-RESEARCH.md` Pattern 1 (lines 327-334) and Code Example #1 (lines 478-588). |
| `backend/app/modeling/models.py` | model | transform | First tabular regressor pipeline factory (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog` with `SimpleImputer(strategy="median")` first step to prevent LightGBM feature-name warnings), fold-only capped `compute_sample_weights()` (`max_w/min_w <= 5.0`), and shared `predict_from_feature_dict()` column-ordering helper. Follow `03-RESEARCH.md` Code Examples #2–#3 (lines 590-624) and Pitfalls 3–4 (lines 455-462). |

## Metadata

**Analog search scope:** `backend/app/`, `backend/scripts/`, `backend/tests/`, `pyproject.toml`, `.gitignore` (all 14 analog paths verified git-tracked via `git ls-files -- <path>`)
**Files scanned:** 14 tracked files
**Pattern extraction date:** 2026-11-23
