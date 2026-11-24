# Phase 2: Classical PDI Engine, Optical Calibration & Watershed - Pattern Map

**Mapped:** 2026-11-22
**Files analyzed:** 17
**Analogs found:** 12 / 17

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `backend/app/config.py` *(modify)* | config | file-I/O | `backend/app/config.py` | exact |
| `backend/app/pdi/__init__.py` *(new)* | config | transform | `backend/app/ingestion/__init__.py` | exact |
| `backend/app/pdi/config.py` *(new)* | config | transform | `backend/app/pdi/io_exif.py` (+ `backend/app/ingestion/decoding.py`) | role-match |
| `backend/app/pdi/io_exif.py` *(modify)* | service | file-I/O | `backend/app/pdi/io_exif.py` (+ `backend/app/ingestion/decoding.py`) | exact |
| `backend/app/pdi/preprocess.py` *(new)* | utility | transform | *(none — new OpenCV filter domain; see Shared Patterns + RESEARCH.md)* | none |
| `backend/app/pdi/lcd_suppression.py` *(new)* | service | transform | *(none — new OpenCV morphology/contour domain; see Shared Patterns + RESEARCH.md)* | none |
| `backend/app/pdi/segmentation.py` *(new)* | service | transform | *(none — new classical segmentation domain; see Shared Patterns + RESEARCH.md)* | none |
| `backend/app/pdi/features.py` *(new)* | service | transform | `backend/app/ingestion/manifest.py` (+ `backend/app/pdi/io_exif.py`) | role-match |
| `backend/app/pdi/xray.py` *(new)* | utility | transform | *(none — new OpenCV/Base64 visual renderer domain; see Shared Patterns + RESEARCH.md)* | none |
| `backend/app/pdi/pipeline.py` *(new)* | service | transform | `backend/scripts/build_manifest.py` (+ `backend/app/pdi/io_exif.py`) | role-match |
| `backend/app/pdi/watershed.py` *(new)* | service | transform | *(none — new scikit-image/SciPy Watershed domain; see Shared Patterns + RESEARCH.md)* | none |
| `backend/scripts/extract_features.py` *(new)* | utility | batch | `backend/scripts/build_manifest.py` (+ `backend/app/ingestion/manifest.py`, `backend/app/ingestion/audit_summary.py`) | exact |
| `backend/tests/test_pdi_preprocess_and_lcd.py` *(new)* | test | transform | `backend/tests/test_io_exif.py` (+ `backend/tests/conftest.py`) | exact |
| `backend/tests/test_pdi_features_invariance.py` *(new)* | test | transform | `backend/tests/test_decoding.py` (+ `backend/tests/test_io_exif.py`) | exact |
| `backend/tests/test_pdi_pipeline_and_xray.py` *(new)* | test | file-I/O | `backend/tests/test_io_exif.py` (+ `backend/tests/conftest.py`) | exact |
| `backend/tests/test_pdi_watershed.py` *(new)* | test | transform | `backend/tests/test_io_exif.py` (+ `backend/tests/test_manifest_integration.py`) | role-match |
| `backend/tests/test_extract_features_integration.py` *(new)* | test | batch | `backend/tests/test_manifest_integration.py` (+ `backend/tests/test_io_exif.py`) | exact |

*(Generated runtime artifacts — not hand-edited source modules: git-ignored `data/processed/features_clean.parquet` written via `backend/app/ingestion/manifest.py` PyArrow schema pattern, and git-tracked `models/pdi_calibration_audit.json` written via `backend/app/ingestion/audit_summary.py` deterministic zero-path-leakage JSON pattern.)*

## Pattern Assignments

### `backend/app/config.py` (config, file-I/O) — MODIFY

**Analog:** `backend/app/config.py` (self-extension)

**Path constant & `__all__` export pattern** (`backend/app/config.py` lines 8-30):
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

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "RAW_DATASET_DIR",
    "PROCESSED_DIR",
    "MODELS_DIR",
    "CLEAN_MANIFEST_PATH",
    "QUARANTINE_MANIFEST_PATH",
    "AUDIT_SUMMARY_PATH",
]
```
**Planner guidance:** Append `FEATURES_CLEAN_PATH: Path = PROCESSED_DIR / "features_clean.parquet"` and `PDI_CALIBRATION_AUDIT_PATH: Path = MODELS_DIR / "pdi_calibration_audit.json"` using the exact same `Path` type annotations and add both names to `__all__` without modifying any existing Phase 1 constant (guarded by `backend/tests/test_io_exif.py` lines 80-108).

---

### `backend/app/pdi/__init__.py` (config, transform) — NEW

**Analog:** `backend/app/ingestion/__init__.py`

**Package barrel & sorted `__all__` pattern** (`backend/app/ingestion/__init__.py` lines 1-45):
```python
"""Public package barrel for AvianScale CV forensic ingestion and anti-leakage splits."""

from __future__ import annotations

from backend.app.ingestion.decoding import (
    ARABIC_TO_ASCII,
    ...
    DecodedFilename,
    assign_weight_band,
    classify_lens,
)
from backend.app.ingestion.manifest import (
    DERIVED_COLUMNS,
    MANIFEST_COLUMNS,
    ...
)

__all__ = [
    "ARABIC_TO_ASCII",
    ...
]
```
**Planner guidance:** Re-export the public PDI contract (`PDIConfig`, `ExifMetadata`, `OrientedImage`, `OpticalOverride`, `OpticalContext`, `WorkingImage`, `read_exif_metadata`, `load_oriented_rgb`, `load_working_rgb`, `resolve_optical_context`, `FEATURE_COLUMNS`, `extract_mask_gated_features`, `PDIResult`, `XRayArtifacts`, `run_single_bird_pdi`, `WatershedInstance`, `WatershedVisuals`, `WatershedResult`, `run_watershed_pdi`) with explicit alphabetically sorted `__all__` and `from __future__ import annotations`.

---

### `backend/app/pdi/config.py` (config, transform) — NEW

**Analog:** `backend/app/pdi/io_exif.py` (frozen dataclass convention) + `backend/app/ingestion/decoding.py` (16-hex SHA-256 helper)

**Frozen slots dataclass pattern** (`backend/app/pdi/io_exif.py` lines 32-44):
```python
@dataclass(frozen=True, slots=True)
class ExifMetadata:
    """Header-level EXIF orientation and optical calibration metadata."""

    raw_w: int
    raw_h: int
    oriented_w: int
    oriented_h: int
    exif_orientation: int
    focal_length_mm: float | None
    zoom_multiple: float | None
    has_exif_optical: bool
```

**Deterministic 16-hex SHA-256 fingerprint pattern** (`backend/app/ingestion/decoding.py` lines 12, 120-122):
```python
import hashlib
...
def compute_image_id(rel_path: str) -> str:
    """Compute a deterministic 16-hex SHA-256 identifier from a POSIX ``rel_path``."""
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]
```
**Planner guidance:** Implement `PDIConfig` as `@dataclass(frozen=True, slots=True)` using the exact default fields in `02-RESEARCH.md` lines 390-427. Use `dataclasses.asdict(self)` (never `self.__dict__`, which does not exist on `slots=True` dataclasses) with `json.dumps(..., sort_keys=True, separators=(",", ":"))` and `hashlib.sha256(...).hexdigest()[:16]`. Validate critical bounds in `__post_init__` if needed while preserving immutability.

---

### `backend/app/pdi/io_exif.py` (service, file-I/O) — MODIFY

**Analog:** `backend/app/pdi/io_exif.py` (self-extension) + `backend/app/ingestion/decoding.py` (`classify_lens`)

**Imports & bounded security constants pattern** (`backend/app/pdi/io_exif.py` lines 10-30):
```python
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

EXIF_TAG_ORIENTATION = 274
EXIF_IFD_POINTER = 0x8769
EXIF_TAG_FOCAL_LENGTH = 37386
XIAOMI_TAG_JSON = 39321
MAX_XIAOMI_TAG_BYTES = 16_384
MIN_VALID_ZOOM_MULTIPLE = 0.1
MAX_VALID_ZOOM_MULTIPLE = 20.0
TRANSPOSE_SWAP_ORIENTATIONS = frozenset({5, 6, 7, 8})
```

**Finite numeric validation guard pattern** (`backend/app/pdi/io_exif.py` lines 54-65):
```python
def _parse_focal_length(raw: Any) -> float | None:
    """Safely convert an EXIF FocalLength (e.g., IFDRational) to positive finite mm."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value
```

**Single Pillow image-opening & EXIF-before-transpose pattern** (`backend/app/pdi/io_exif.py` lines 137-156):
```python
def _open_image_source(source: Path | bytes) -> Image.Image:
    """Open a Path or raw image bytes using Pillow without altering MAX_IMAGE_PIXELS."""
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    return Image.open(source)


def read_exif_metadata(source: Path | bytes) -> ExifMetadata:
    """Header-only EXIF & post-transpose dimension reader (does not decode pixel data)."""
    with _open_image_source(source) as im:
        return _extract_exif_from_open_image(im)


def load_oriented_rgb(source: Path | bytes) -> OrientedImage:
    """Extract EXIF metadata BEFORE applying ``ImageOps.exif_transpose().convert('RGB')``."""
    with _open_image_source(source) as im:
        metadata = _extract_exif_from_open_image(im)
        oriented = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.asarray(oriented, dtype=np.uint8)
    return OrientedImage(rgb=rgb, metadata=metadata)
```

**Single-source lens classification pattern** (`backend/app/ingestion/decoding.py` lines 27, 102-108):
```python
WIDE_FOCAL_THRESHOLD_MM: float = 3.0
...
def classify_lens(focal_length_mm: float | None) -> str:
    """Classify optical lens family from EXIF focal length (``>= 3.0 mm`` wide vs ultrawide)."""
    if focal_length_mm is None or not math.isfinite(focal_length_mm):
        return "unknown"
    if focal_length_mm >= WIDE_FOCAL_THRESHOLD_MM:
        return "wide"
    return "ultrawide"
```
**Planner guidance:** Keep `ExifMetadata`, `OrientedImage`, `read_exif_metadata()`, and `load_oriented_rgb()` 100% backward-compatible. Add `OpticalOverride`, `OpticalContext`, `WorkingImage`, `resolve_optical_context()` (validating finite bounds `0.5 <= focal_length_mm <= 200.0` and `0.1 <= zoom_multiple <= 20.0` per ASVS V5, reusing `classify_lens` from `backend.app.ingestion.decoding` with zero circular imports), and `load_working_rgb()` calling `_extract_exif_from_open_image(im)` strictly **before** aspect-preserving `im.draft("RGB", draft_box)` and `cv2.resize(..., interpolation=cv2.INTER_AREA)` (see `02-RESEARCH.md` lines 700-742). Never call `cv2.imread` or `cv2.imdecode` and never alter `Image.MAX_IMAGE_PIXELS`.

---

### `backend/app/pdi/features.py` (service, transform) — NEW

**Analog:** `backend/app/ingestion/manifest.py` (immutable ordered schema-column tuple contract) + `backend/app/pdi/io_exif.py` (frozen dataclass & finite numeric guards)

**Immutable ordered schema-tuple pattern** (`backend/app/ingestion/manifest.py` lines 66-91):
```python
MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "rel_path",
    "folder_chain",
    ...
    "quarantine_reason",
    "fold",
)
```
**Planner guidance:** Define immutable tuples `MORPHOMETRIC_FEATURE_COLUMNS`, `CHROMATIC_TEXTURE_FEATURE_COLUMNS`, `OPTICAL_FEATURE_COLUMNS`, and their concatenation `FEATURE_COLUMNS: tuple[str, ...]` (exact 49 snake_case keys from `02-RESEARCH.md` lines 560-595) plus `BANNED_FEATURE_COLUMNS: frozenset[str] = frozenset({"oriented_w", "oriented_h", "megapixels", "bbox_w", "bbox_h", "raw_w", "raw_h"})`. Implement pure `extract_mask_gated_features(bgr_work, bird_mask, scale_factor, optical, contour=None, ellipse=None) -> dict[str, float]` strictly indexing pixel-local `cv2.cvtColor(bgr_work, ...)` at `bird_mask == 255` and 3-px eroded interior (`cv2.erode(..., 7x7 ellipse) > 0`) for `Scharr`/`Laplacian` so outside-mask perturbations produce exact `0.0` diff (`PROIBIÇÃO 2`). Copy exact CV math from `02-RESEARCH.md` lines 560-595 and 630-642.

---

### `backend/app/pdi/pipeline.py` (service, transform) — NEW

**Analog:** `backend/scripts/build_manifest.py` (`run_build_manifest` pure orchestrator flow) + `backend/app/pdi/io_exif.py` (frozen result dataclass)

**Linear stage orchestration pattern** (`backend/scripts/build_manifest.py` lines 100-127):
```python
def run_build_manifest(
    dataset_root: Path,
    output_dir: Path,
    summary_path: Path,
    *,
    dry_run: bool = False,
    min_heavy_sessions_per_fold: int | None = None,
) -> dict[str, Any]:
    """Execute end-to-end forensic curation, 5-fold splitting, and aggregate summary export."""
    full_df = scan_dataset_records(dataset_root)
    clean_df, quarantine_df = split_clean_and_quarantine(full_df)
    ...
```
**Planner guidance:** Implement `PDIResult` as `@dataclass(frozen=True, slots=True)` (`02-RESEARCH.md` lines 496-504) and `run_single_bird_pdi(source: Path | bytes, *, config: PDIConfig | None = None, optical_override: OpticalOverride | None = None, return_xray: bool = False) -> PDIResult` as a stateless pure orchestrator chaining `load_working_rgb` → `resolve_optical_context` → `preprocess_working_rgb` → `segment_single_bird` → `extract_mask_gated_features` → optional `render_xray_artifacts` (strictly skipped when `return_xray=False`), timing with `time.perf_counter()` to populate `latency_ms` and `pdi_config_hash = cfg.config_hash()`.

---

### `backend/scripts/extract_features.py` (utility, batch) — NEW

**Analog:** `backend/scripts/build_manifest.py` (CLI structure, `--dry-run`, aggregate-only stdout) + `backend/app/ingestion/manifest.py` (PyArrow schema write + `_is_contained_non_symlink` ASVS V4 containment) + `backend/app/ingestion/audit_summary.py` (`write_audit_summary` + `_FORBIDDEN_LEAK_SUBSTRINGS`)

**CLI argument parsing & entrypoint pattern** (`backend/scripts/build_manifest.py` lines 12-77, 149-166):
```python
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import AUDIT_SUMMARY_PATH, PROCESSED_DIR, RAW_DATASET_DIR
...
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for forensic manifest building."""
    parser = argparse.ArgumentParser(
        description="Build AvianScale CV forensic Parquet manifests and aggregate audit summary.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=RAW_DATASET_DIR,
        help="Root directory containing raw broiler weight JPEG folders (read-only).",
    )
    ...
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full curation and split verification without writing any files or directories.",
    )
    return parser.parse_args(argv)
...
def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    run_build_manifest(...)
    return 0


__all__ = ["parse_args", "run_build_manifest", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
```

**Aggregate-only stdout logging (`PROIBIÇÃO 4`)** (`backend/scripts/build_manifest.py` lines 135-146):
```python
    # PROIBIÇÃO 4: log strictly aggregate metrics — never raw filenames or host paths.
    print(
        "Manifest build complete: "
        f"raw_images={funnel['raw_images']} "
        f"clean_images={funnel['clean_images']} "
        ...
        f"dry_run={dry_run}"
    )
```

**ASVS V4 path/symlink containment guard** (`backend/app/ingestion/manifest.py` lines 208-216):
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
```

**Explicit PyArrow schema & CoW-safe Parquet persistence** (`backend/app/ingestion/manifest.py` lines 116-143, 156-205, 364-368):
```python
_MANIFEST_ARROW_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ...
        ("fold", pa.int8()),
    ]
)
...
def write_manifest_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a 24-column manifest strictly as a typed PyArrow Parquet file (never CSV)."""
    typed_df = enforce_manifest_dtypes(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    typed_df.to_parquet(path, engine="pyarrow", index=False, schema=_MANIFEST_ARROW_SCHEMA)
```

**Byte-deterministic zero-path-leakage JSON persistence** (`backend/app/ingestion/audit_summary.py` lines 50, 315-326):
```python
_FORBIDDEN_LEAK_SUBSTRINGS: tuple[str, ...] = (".jpg", "rel_path", "/home/", "data/raw")
...
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
```
**Planner guidance:** Build `extract_features.py` with `parse_args`, `run_extract_features`, `build_calibration_audit`, `write_features_parquet`, `write_calibration_audit`, and `main`. Validate every manifest `rel_path` (`".." not in Path(rel_path).parts` + `_is_contained_non_symlink(dataset_root / rel_path, canonical_root)`), execute `ProcessPoolExecutor(max_workers=min(workers, 4), initializer=cv2.setNumThreads, initargs=(1,))` calling `run_single_bird_pdi(..., return_xray=False)`, preserve exact `manifest_clean.parquet` 1:1 row order (`image_id`), write the 60-column typed PyArrow schema (`11` join/audit + `49` `pa.float64()` features), and write aggregate-only `models/pdi_calibration_audit.json` (skipping both writes on `--dry-run`). Note: consider omitting volatile `batch_wall_clock_sec` or isolating it if exact byte-reproducibility is desired, or follow `02-RESEARCH.md` lines 599-628 (`batch_wall_clock_sec` recorded as float, deterministic audit sections verified separately).

---

### `backend/tests/test_pdi_preprocess_and_lcd.py` (test, transform) — NEW

**Analog:** `backend/tests/test_io_exif.py` + `backend/tests/conftest.py`

**Imports & parametrized `Path`-vs-`bytes` parity test pattern** (`backend/tests/test_io_exif.py` lines 1-29, 110-146):
```python
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from backend.app import config
from backend.app.pdi import io_exif
from backend.tests.conftest import make_synthetic_jpeg_bytes, make_synthetic_png_bytes
...
@pytest.mark.parametrize("orientation", [1, 3, 5, 6, 7, 8])
def test_exif_metadata_and_oriented_rgb_path_bytes_parity(
    tmp_path: Path,
    orientation: int,
) -> None:
    raw_bytes = make_synthetic_jpeg_bytes(
        size=(40, 20),
        orientation=orientation,
        focal_length_mm=4.25,
        zoom_multiple=1.0,
        draw_corner_marker=True,
    )
    img_path = tmp_path / f"orient_{orientation}.jpg"
    img_path.write_bytes(raw_bytes)
    ...
    assert np.array_equal(loaded_path.rgb, loaded_bytes.rgb)
```
**Planner guidance:** Mirror `test_io_exif.py` parametrization and synthetic fixture usage to test: `PDIConfig.config_hash()` determinism & mutation sensitivity (`slots=True` safe), `load_working_rgb` aspect-preserving `im.draft()` pre-draft metadata preservation + `Path`-vs-`bytes` bit-identical parity on large synthetic `20:9` JPEGs, `OpticalOverride` ASVS V5 bounds rejection, `preprocess_working_rgb` bilateral/median + Lab CLAHE `uint8` bounds, `suppress_lcd_and_platter` rotation-invariant dark LCD + specular housing zeroing, `fit_ellipse_safe` `<5`-point/collinear fallback to `minAreaRect`, and Scenario B smooth-wall peeling on synthetic white-on-white bucket frames.

---

### `backend/tests/test_pdi_features_invariance.py` (test, transform) — NEW

**Analog:** `backend/tests/test_decoding.py` (schema tuple & boundary assertions) + `backend/tests/test_io_exif.py` (guard tests)

**Immutable schema-tuple & boundary assertion pattern** (`backend/tests/test_decoding.py` lines 45-58, 128-154):
```python
def test_audit_section_9_constants_match_verbatim() -> None:
    ...
    assert len(RAW_BASE_COLUMNS) == 15
    assert len(DERIVED_COLUMNS) == 9
    assert len(MANIFEST_COLUMNS) == 24
    assert set(MANIFEST_COLUMNS) == set(RAW_BASE_COLUMNS) | set(DERIVED_COLUMNS)
...
def test_boundary_weight_bands_lens_and_image_id() -> None:
    ...
    assert classify_lens(4.25) == "wide"
    assert classify_lens(1.65) == "ultrawide"
    assert classify_lens(None) == "unknown"
```
**Planner guidance:** Combine this constant/contract style with the strict outside-mask perturbation test in `02-RESEARCH.md` lines 796-816: verify `len(FEATURE_COLUMNS) == 49`, verify zero banned raw-dimension or axis-aligned bbox columns (`oriented_w`, `oriented_h`, `megapixels`, `bbox_w`, `bbox_h`), assert **exact `0.0` diff** across all 49 features when arbitrary LCD digits/neon pixels are painted on `bird_mask == 0` (`PROIBIÇÃO 2`), verify 90°/180° rotation invariance of morphometrics & chromatic descriptors, verify Hue `S < 25` gating returns `0.0` on neutral white silhouettes, and verify `area_optical_norm = area_px / (zoom_multiple ** 2)` (no `focal_length_mm * zoom_multiple` double-counting).

---

### `backend/tests/test_pdi_pipeline_and_xray.py` (test, file-I/O) — NEW

**Analog:** `backend/tests/test_io_exif.py` (graceful degradation, decompression bomb & AST ban checks) + `backend/tests/conftest.py`

**Graceful upload degradation pattern** (`backend/tests/test_io_exif.py` lines 176-203):
```python
def test_stripped_exif_jpeg_and_rgba_png_degrade_gracefully(tmp_path: Path) -> None:
    stripped_jpeg = make_synthetic_jpeg_bytes(
        size=(32, 24),
        orientation=None,
        focal_length_mm=None,
        zoom_multiple=None,
        draw_corner_marker=False,
    )
    ...
    rgba_png = make_synthetic_png_bytes(size=(24, 16), mode="RGBA")
    png_loaded = load_oriented_rgb(rgba_png)
    assert png_loaded.metadata.has_exif_optical is False
    assert png_loaded.rgb.shape == (16, 24, 3)
    assert png_loaded.rgb.dtype == np.uint8
```
**Planner guidance:** Follow this upload-robustness pattern for end-to-end `run_single_bird_pdi()` tests: EXIF-stripped JPEG & RGBA/grayscale PNG uploads (falling back to `4.25 mm / 1.0×`, `exif_optical_present == 0.0`), `Path`-vs-`bytes` `PDIResult` exact feature parity, `return_xray=False` returning `xray is None`, `return_xray=True` decoding all 4 Base64 artifacts via Pillow (`JPEG` stages 1/2/4 + 1-channel `PNG` stage 3 at `<= 640 px`, combined payload `< 250 KB`), synthetic full-pipeline SF-400 LCD exclusion check (`np.count_nonzero(seg.bird_mask & lcd_gt_mask) == 0`), and real native `4080×1840` Wide JPEG latency check (`< 400 ms`, guarded if `CLEAN_MANIFEST_PATH` exists, never printing raw filenames).

---

### `backend/tests/test_pdi_watershed.py` (test, transform) — NEW

**Analog:** `backend/tests/test_io_exif.py` (synthetic unit tests & parameter bounds) + `backend/tests/test_manifest_integration.py` (read-only real quarantine manifest audit without filename leaks)

**Read-only quarantine manifest loading pattern** (`backend/tests/test_manifest_integration.py` lines 52-57, 186-195):
```python
def _load_real_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    assert CLEAN_MANIFEST_PATH.is_file(), "manifest_clean.parquet must be materialized"
    assert QUARANTINE_MANIFEST_PATH.is_file(), "manifest_quarantine.parquet must be materialized"
    clean_df = pd.read_parquet(CLEAN_MANIFEST_PATH)
    quarantine_df = pd.read_parquet(QUARANTINE_MANIFEST_PATH)
    return clean_df, quarantine_df
...
    multi_bird = quarantine_df.loc[quarantine_df["quarantine_reason"] == "multi_bird"]
    assert len(multi_bird) == 13
    assert multi_bird["folder_chain"].value_counts().to_dict() == {
        "417": 4,
        "757": 1,
        "823": 4,
        "1520": 4,
    }
```
**Planner guidance:** Split into fast synthetic unit tests (touching binary/RGB ellipses split into `bird_count == 2`, red-feeder chromatic pre-suppression, empty-mask zero-crash guard returning `bird_count == 0`, ASVS V5 parameter bounds validation, per-instance 49-key `FEATURE_COLUMNS` parity + 4-stage Base64 visuals) AND a read-only integration audit across all `13` quarantined multi-bird rows in `manifest_quarantine.parquet` (`417`, `757`, `823`, `1520`) asserting zero crashes, valid per-instance features, and `PROIBIÇÃO 4`-clean aggregate assertions (never printing `rel_path`).

---

### `backend/tests/test_extract_features_integration.py` (test, batch) — NEW

**Analog:** `backend/tests/test_manifest_integration.py` (PyArrow schema check, JSON zero-path-leakage & git-hygiene checks) + `backend/tests/test_io_exif.py` (CLI `--dry-run` & stdout hygiene tests)

**PyArrow exact schema contract check** (`backend/tests/test_manifest_integration.py` lines 91-100):
```python
    for parquet_path in (CLEAN_MANIFEST_PATH, QUARANTINE_MANIFEST_PATH):
        schema = pq.read_schema(parquet_path)
        assert schema.names == list(MANIFEST_COLUMNS)
        assert len(schema.names) == 24
        for col_name, expected_type in expected_arrow_types.items():
            assert schema.field(col_name).type == expected_type, (
                f"Schema mismatch on column {col_name!r}: "
                f"{schema.field(col_name).type} != {expected_type}"
            )
```

**JSON zero-path-leakage & defense-in-depth rejection check** (`backend/tests/test_manifest_integration.py` lines 49, 245-254, 329-332):
```python
_FORBIDDEN_LEAK_MARKERS: tuple[str, ...] = (".jpg", "rel_path", "/home/", "data/raw")
...
    raw_text = AUDIT_SUMMARY_PATH.read_text(encoding="utf-8")
    lowered_text = raw_text.lower()
    for forbidden in _FORBIDDEN_LEAK_MARKERS:
        assert forbidden not in lowered_text, (
            f"PROIBIÇÃO 4 / T-01-07 violation: forbidden marker {forbidden!r} in audit summary"
        )
...
    with pytest.raises(ValueError, match="PROIBIÇÃO 4"):
        write_audit_summary({"bad": "116/116 (1).jpg"}, tmp_path / "rejected.json")
```

**Git-hygiene verification pattern** (`backend/tests/test_manifest_integration.py` lines 334-356):
```python
def test_git_hygiene_excludes_raw_and_processed_while_tracking_audit_summary() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "data/raw", "data/processed"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout.strip() == "", (
        "PROIBIÇÃO 4 / T-01-08 violation: data/raw or data/processed files are tracked by Git."
    )

    ignored_check = subprocess.run(
        ["git", "check-ignore", "models/dataset_audit_summary.json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored_check.returncode != 0, (
        "models/dataset_audit_summary.json must NOT be ignored by .gitignore."
    )
```

**CLI `--dry-run` zero-writes & zero-`.jpg`-stdout pattern** (`backend/tests/test_io_exif.py` lines 311-336):
```python
def test_build_manifest_cli_dry_run_writes_zero_files(
    synthetic_dataset_tree: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "out_dry"
    summary_path = tmp_path / "summary_dry.json"

    rc = build_manifest.main(
        [
            "--dataset-root",
            str(synthetic_dataset_tree),
            "--output-dir",
            str(out_dir),
            "--summary-path",
            str(summary_path),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert not out_dir.exists()
    assert not summary_path.exists()
    # PROIBIÇÃO 4: stdout must never dump individual JPEG filenames.
    assert ".jpg" not in captured.out.lower()
```
**Planner guidance:** Mirror all 4 checks verbatim for Phase 2: (1) CLI `--dry-run` + synthetic manifest run verifying symlink-escape rejection and zero `.jpg` stdout; (2) materialized `data/processed/features_clean.parquet` exact 60-column PyArrow schema (`1,738` rows, 1:1 `image_id` alignment with `manifest_clean.parquet`, zero NaN/inf in all 49 features, banned raw-dimension columns absent, `segmentation_fallback.mean() < 0.02`); (3) materialized `models/pdi_calibration_audit.json` zero `_FORBIDDEN_LEAK_MARKERS`, winning formula `area_px / (zoom_multiple ** 2)` collapsing bucket Wide/UW ratio near `1.0` and outperforming rejected double-counted `area_px / ((focal_length_mm * zoom_multiple) ** 2)`; (4) `git check-ignore models/pdi_calibration_audit.json` returns non-zero while `git ls-files data/raw data/processed` stays empty.

---

## Shared Patterns

### 1. Frozen Slots Dataclasses & Python 3.12 Style / Ruff Contract
**Source:** `backend/app/pdi/io_exif.py` lines 10-52 & `pyproject.toml` lines 40-49
**Apply to:** All 12 production modules under `backend/app/` and `backend/scripts/`
```python
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class OrientedImage:
    """EXIF-transposed RGB uint8 image paired with pre-transpose EXIF metadata."""

    rgb: np.ndarray
    metadata: ExifMetadata
```
```toml
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["."]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.ruff.lint.isort]
known-first-party = ["backend"]
```
- Every file begins with a module docstring + `from __future__ import annotations`.
- Public modules define explicit `__all__`.
- Never access `self.__dict__` on `slots=True` dataclasses — use `dataclasses.asdict(self)`.
- Never modify `pyproject.toml` or `uv.lock` dependencies (enforced by `backend/tests/test_io_exif.py` lines 289-309).

### 2. Repo-Wide AST Ban on `cv2.imread` / `cv2.imdecode` & Decompression-Bomb Guard (`PROIBIÇÃO 1 & 2`)
**Source:** `backend/tests/test_io_exif.py` lines 258-287
**Apply to:** Every module under `backend/app/pdi/` (`io_exif.py`, `preprocess.py`, `lcd_suppression.py`, `segmentation.py`, `features.py`, `xray.py`, `pipeline.py`, `watershed.py`)
```python
def test_ast_scan_confirms_zero_raw_opencv_image_loading_under_backend_app() -> None:
    backend_app_dir = config.REPO_ROOT / "backend" / "app"
    py_files = sorted(backend_app_dir.rglob("*.py"))
    ...
    banned_calls = {"imread", "imdecode"}
    ...
```
- **Rule:** Never call or import `cv2.imread` or `cv2.imdecode` anywhere under `backend/app/` (encoding via `cv2.imencode` in `xray.py`/`watershed.py` is allowed). All image loading flows strictly through `backend/app/pdi/io_exif.py`, keeping `PIL.Image.MAX_IMAGE_PIXELS` enabled.

### 3. Input Bounds & Finite-Float Validation (ASVS V5)
**Source:** `backend/app/pdi/io_exif.py` lines 54-103 & `backend/app/ingestion/splits.py` lines 35-54
**Apply to:** `io_exif.py` (`OpticalOverride`), `config.py` (`PDIConfig`), `watershed.py` (`h_rel`, `min_distance_rel`, `min_area_ratio`), `extract_features.py` (CLI workers & manifest columns)
```python
    if raw is None or isinstance(raw, bool):
        return None
    ...
    if not math.isfinite(zoom) or not (
        MIN_VALID_ZOOM_MULTIPLE <= zoom <= MAX_VALID_ZOOM_MULTIPLE
    ):
        return None
```
- Explicitly reject `bool` (`isinstance(x, bool)` is a subclass of `int` in Python), `NaN`/`Inf` (`not math.isfinite(x)`), and out-of-range overrides/parameters with clear `ValueError` (for public API/override params) or `None` degradation (for untrusted raw EXIF tags).

### 4. ASVS V4 Path/Symlink Containment & `PROIBIÇÃO 4` Zero-Path-Leakage Output Guard
**Source:** `backend/app/ingestion/manifest.py` lines 208-216 & `backend/app/ingestion/audit_summary.py` lines 50, 315-326
**Apply to:** `backend/scripts/extract_features.py` and `backend/tests/test_extract_features_integration.py`
```python
_FORBIDDEN_LEAK_SUBSTRINGS: tuple[str, ...] = (".jpg", "rel_path", "/home/", "data/raw")
```
- Canonicalize `dataset_root.resolve()`, reject `".."` segments and symlinks via `_is_contained_non_symlink()`, log strictly aggregate counts to stdout, and validate serialized JSON against `_FORBIDDEN_LEAK_SUBSTRINGS` before writing `models/pdi_calibration_audit.json`.

### 5. Pandas 3.1 Copy-on-Write & Typed PyArrow Parquet Contract
**Source:** `backend/app/ingestion/manifest.py` lines 116-205, 364-368
**Apply to:** `backend/scripts/extract_features.py` (`features_clean.parquet`)
- Always construct or copy frames explicitly (`out = df.loc[:, cols].copy()`), mutate via single-step `.loc[mask, col] = val`, cast text columns to `str`, and write via `df.to_parquet(path, engine="pyarrow", index=False, schema=_FEATURES_ARROW_SCHEMA)`.

### 6. Shared Synthetic Image Test Helpers (Zero `data/raw/` Reads in Unit Tests)
**Source:** `backend/tests/conftest.py` lines 21-77
**Apply to:** All 5 new Phase 2 test suites under `backend/tests/`
```python
def make_synthetic_jpeg_bytes(
    size: tuple[int, int] = (40, 20),
    orientation: int | None = 6,
    focal_length_mm: float | None = 4.25,
    zoom_multiple: float | None = 1.0,
    raw_xiaomi_tag: str | bytes | None = None,
    draw_corner_marker: bool = True,
) -> bytes: ...

def make_synthetic_png_bytes(
    size: tuple[int, int] = (24, 16),
    mode: str = "RGBA",
) -> bytes: ...
```
- Unit tests synthesize RGB/BGR images and JPEG/PNG bytes in memory or `tmp_path`. Integration tests reading `data/processed/manifest_*.parquet` or `data/raw/` stay strictly read-only and never print raw filenames or `rel_path` values.

## No Analog Found

Files introducing brand-new classical CV, visual rendering, or Watershed domains with no prior algorithmic analog in the Phase 1 codebase (planner MUST apply **Shared Patterns #1–#3** for dataclass/typing/security conventions and source the CV algorithms directly from `02-RESEARCH.md`):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `backend/app/pdi/preprocess.py` | utility | transform | First OpenCV filtering/color-space module (`cv2.bilateralFilter`, `cv2.medianBlur`, `cv2.createCLAHE` on Lab `L`, HSV, Scharr gradients); follow `02-RESEARCH.md` lines 118, 456-466 & `D-03`. |
| `backend/app/pdi/lcd_suppression.py` | service | transform | First OpenCV contour/morphology exclusion module (rotation-invariant dark-neutral rectangular LCD bezel + connected specular white SF-400 housing suppression); follow `02-RESEARCH.md` Spike 3 lines 175-180, 467-472 & `D-05`. |
| `backend/app/pdi/segmentation.py` | service | transform | First classical 3-scenario segmentation module (bimodal border-ring detector, Scenario B smooth-wall peeling + Black-Hat barrier, Scenarios A/C bird-vs-straw, `fit_ellipse_safe`, `[0.015, 0.72]` adaptive fallback); follow `02-RESEARCH.md` Spike 3 lines 160-196, 473-485 & Code Example #2 lines 745-769. |
| `backend/app/pdi/xray.py` | utility | transform | First 4-stage visual annotation & Base64 JPEG/PNG encoder (`cv2.imencode` @ `640 px`, 3× JPEG `Q=82` + 1× 1-ch PNG, `< 250 KB`); follow `02-RESEARCH.md` Spike 1 lines 108-118, 486-495 & `D-04`. |
| `backend/app/pdi/watershed.py` | service | transform | First isolated scikit-image/SciPy Marker-based Watershed module (red-feeder pre-suppression, union z-score mask, `distance_transform_edt`, bbox-cropped `h_maxima`, modern `peak_local_max`, `skimage.segmentation.watershed`, per-instance `extract_mask_gated_features`, 4 Base64 visual stages); follow `02-RESEARCH.md` Spike 4 lines 200-221, 506-539 & Code Example #3 lines 771-793. |

## Metadata

**Analog search scope:** `backend/app/`, `backend/scripts/`, `backend/tests/`, `pyproject.toml`, `.gitignore` (all analog paths verified git-tracked via `git ls-files -- <path>`)
**Files scanned:** 14 tracked files
**Pattern extraction date:** 2026-11-22
