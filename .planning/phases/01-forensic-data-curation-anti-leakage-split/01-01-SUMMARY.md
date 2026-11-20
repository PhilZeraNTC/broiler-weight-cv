---
phase: 01-forensic-data-curation-anti-leakage-split
plan: 01
subsystem: infra
tags: [python312, uv, pillow, exif, pyarrow, parquet, pytest, ruff]

# Dependency graph
requires: []
provides:
  - Isolated uv-managed CPython 3.12.14 binary-only workspace (.python-version, pyproject.toml, uv.lock) locking all 17 D-02 packages with zero banned DL/OCR/GUI-OpenCV libraries
  - Single shared Pillow EXIF-first metadata and orientation loader (backend/app/pdi/io_exif.py) supporting Path | bytes with bounded Xiaomi tag 39321 parsing and decompression-bomb protection
  - Repository-root-anchored path configuration (backend/app/config.py) decoupled from process cwd
  - End-to-end runnable CLI-to-Parquet tracer (backend/scripts/build_manifest.py) with read-only symlink-safe traversal and --dry-run mode
  - Reusable synthetic JPEG+EXIF and Audit §9 dataset tree fixtures (backend/tests/conftest.py) and 17 passing security/parity/prohibition tests (backend/tests/test_io_exif.py)
affects: [01-02, 01-03, phase-2-classical-pdi, phase-3-modeling, phase-4-web-mvp]

actuals:
  tokens: 31478
  tasks: 2
  commits: 2

tech-stack:
  added:
    - uv 0.13.9 (managed CPython 3.12.14, no-build binary wheels)
    - opencv-python-headless 4.14.0.94
    - scikit-image 0.26.0
    - scipy 1.18.1
    - pillow 13.0.0
    - numpy 2.5.5
    - pandas 3.1.1
    - pyarrow 26.0.1
    - scikit-learn 1.9.1
    - lightgbm 4.7.0
    - joblib 1.6.0
    - fastapi 0.165.0
    - uvicorn 0.54.0
    - pydantic 2.14.4
    - python-multipart 0.0.34
    - pytest 9.2.0
    - httpx 0.28.1
    - ruff 0.17.6
  patterns:
    - PEP 420 namespace packages under repo-root pythonpath=["."] using backend.app.* and backend.scripts.* imports
    - Single Pillow EXIF-first loader extracting Orientation/FocalLength/Xiaomi tag 39321 strictly before ImageOps.exif_transpose()
    - Path-anchored repo config via Path(__file__).resolve().parents[2]
    - Read-only symlink-rejecting dataset crawl with is_relative_to containment check

key-files:
  created:
    - .python-version
    - pyproject.toml
    - uv.lock
    - backend/app/config.py
    - backend/app/pdi/io_exif.py
    - backend/scripts/build_manifest.py
    - backend/tests/conftest.py
    - backend/tests/test_io_exif.py
  modified: []

key-decisions:
  - "Locked root pyproject.toml with [tool.uv] package=false, no-build=true, python-preference='only-managed' so CPython 3.12.14 binary wheels resolve identically across CLI, pytest, and future API workers without invoking host Python 3.14 or building sdists (D-01, D-02, T-01-SC)."
  - "Enforced strict UTF-8 decoding with MAX_XIAOMI_TAG_BYTES=16384 cap, bool/NaN/Inf rejection, and [0.1, 20.0] range bounds in _parse_xiaomi_zoom plus IFDRational zero-division guards in _parse_focal_length (D-07, T-01-01)."
  - "Kept PEP 420 namespace directories for backend/, backend/app/, backend/app/pdi/, and backend/scripts/ per plan contract, reserving explicit __init__.py files for barrel packages such as backend/app/ingestion/ in Plan 01-02."

patterns-established:
  - "Pattern 1 (EXIF-first I/O): Always read EXIF/IFD 0x8769 metadata via backend.app.pdi.io_exif before ImageOps.exif_transpose(); never call cv2.imread/cv2.imdecode anywhere under backend/app/ (verified by AST scan)."
  - "Pattern 2 (CWD-independent paths): Always import dataset, manifest, and audit summary defaults from backend.app.config rather than relative strings or Path.cwd()."
  - "Pattern 3 (Zero-raw-coupling tests): Synthesize EXIF-bearing JPEGs, RGBA PNGs, and miniature multi-band dataset trees in backend/tests/conftest.py without reading data/raw/ or logging raw filenames."

requirements-completed:
  - DATA-01

coverage:
  - id: D1
    description: "Isolated CPython 3.12.14 uv workspace with binary-only wheel locking (no-build=true) for all 17 D-02 dependencies and zero banned DL/OCR/GUI-OpenCV packages"
    requirement: DATA-01
    verification:
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions"
        status: pass
      - kind: integration
        ref: "uv sync --frozen --no-build --managed-python && uv run python --version"
        status: pass
    human_judgment: false
  - id: D2
    description: "Shared Pillow EXIF-first loader (io_exif.py) with Path|bytes parity across orientations 1,3,5,6,7,8, missing-EXIF graceful fallback, bounded Xiaomi tag 39321 parsing, IFDRational zero-division safety, MAX_IMAGE_PIXELS guard, and zero cv2.imread/imdecode AST compliance"
    requirement: DATA-01
    verification:
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_exif_metadata_and_oriented_rgb_path_bytes_parity"
        status: pass
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_orientation_6_extracts_exif_before_transpose"
        status: pass
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_stripped_exif_jpeg_and_rgba_png_degrade_gracefully"
        status: pass
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_parse_focal_length_handles_zero_denominator_and_invalid_inputs"
        status: pass
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_parse_xiaomi_zoom_enforces_security_bounds_and_null_termination"
        status: pass
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_decompression_bomb_guard_remains_enabled"
        status: pass
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_ast_scan_confirms_zero_raw_opencv_image_loading_under_backend_app"
        status: pass
    human_judgment: false
  - id: D3
    description: "CWD-independent repo-root path config (backend/app/config.py) and end-to-end runnable CLI-to-Parquet tracer (backend/scripts/build_manifest.py) with symlink-escape rejection and --dry-run mode"
    requirement: DATA-01
    verification:
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_config_paths_are_repo_root_anchored_and_cwd_independent"
        status: pass
      - kind: integration
        ref: "backend/tests/test_io_exif.py#test_build_manifest_cli_dry_run_writes_zero_files"
        status: pass
      - kind: integration
        ref: "backend/tests/test_io_exif.py#test_build_manifest_cli_writes_parquet_and_rejects_symlink_escapes"
        status: pass
    human_judgment: false

duration: 5min
completed: 2026-11-20
status: complete
---

# Phase 01 Plan 01: Workspace Bootstrap, Shared Pillow EXIF Loader & CLI Tracer Summary

**Isolated CPython 3.12.14 binary-only `uv` workspace, hardened Pillow EXIF-first loader (`backend/app/pdi/io_exif.py`), repo-root path config (`backend/app/config.py`), and verified CLI-to-Parquet tracer (`backend/scripts/build_manifest.py`) backed by 17 synthetic parity and security tests**

## Performance

- **Duration:** 5 min
- **Started:** 2026-11-20T21:33:53Z
- **Completed:** 2026-11-20T21:38:44Z
- **Tasks:** 2 (1 tracer + 1 TDD)
- **Files modified:** 8 created, 0 modified

## Accomplishments

- Bootstrapped the root `uv` Python 3.12 workspace (`.python-version`, `pyproject.toml`, `uv.lock`) resolving all 49 distributions strictly from pure binary wheels (`no-build = true`, `python-preference = "only-managed"`, CPython `3.12.14`) with zero Deep Learning, OCR, or GUI OpenCV packages.
- Built `backend/app/pdi/io_exif.py` as the single repository-wide Pillow EXIF-first image loader supporting identical `Path | bytes` inputs, extracting `Orientation` (`274`), `FocalLength` (`37386`), and Xiaomi vendor JSON tag `39321` (`zoomMultiple`) prior to `ImageOps.exif_transpose()`, with ASVS L1 bounds (`MAX_XIAOMI_TAG_BYTES = 16_384`, `IFDRational` zero-division guard, `[0.1, 20.0]` zoom clamp, and active `Image.MAX_IMAGE_PIXELS` guard).
- Delivered the working end-to-end tracer `backend/scripts/build_manifest.py` wired to repo-root-anchored defaults from `backend/app/config.py`, safely crawling JPEGs read-only with symlink-escape rejection and writing typed PyArrow Parquet manifests.
- Created reusable offline synthetic JPEG/PNG and Audit §9 multi-band dataset fixtures in `backend/tests/conftest.py` (supplying `>= 5` clean sessions per weight band for Plans `01-02` and `01-03`) and verified all 17 tests in `backend/tests/test_io_exif.py` plus zero `ruff` lint diagnostics across `backend/`.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end Python 3.12 (uv) workspace, shared Pillow EXIF-first loader, and CLI-to-Parquet tracer slice** - `5c2210e` (feat)
2. **Task 2: Add repo-root path config and harden shared Pillow EXIF loader with synthetic fixtures and security/parity tests** - `de88c33` (feat)

## Files Created/Modified

- `.python-version` - Pins uv runtime selector to `3.12`.
- `pyproject.toml` - Root uv workspace declaring all 17 locked `D-02` dependencies, `[tool.uv]` binary-only/managed-python settings, pytest `pythonpath = ["."]`, and ruff `py312` rules.
- `uv.lock` - Deterministic hash-locked binary wheel resolution (49 packages, CPython 3.12.14).
- `backend/app/config.py` - Repo-root-anchored path constants (`REPO_ROOT`, `DATA_DIR`, `RAW_DATASET_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, `CLEAN_MANIFEST_PATH`, `QUARANTINE_MANIFEST_PATH`, `AUDIT_SUMMARY_PATH`).
- `backend/app/pdi/io_exif.py` - Shared Pillow EXIF-first header reader (`read_exif_metadata`) and transposing RGB loader (`load_oriented_rgb`) with `ExifMetadata`/`OrientedImage` dataclasses and hardened tag parsers.
- `backend/scripts/build_manifest.py` - Runnable CLI tracer entrypoint (`parse_args`, `run_tracer_ingest`, `main`) with `--dataset-root`, `--output-dir`, `--summary-path`, and `--dry-run`.
- `backend/tests/conftest.py` - Reusable synthetic EXIF JPEG builder (`make_synthetic_jpeg_bytes`), RGBA PNG builder (`make_synthetic_png_bytes`), and Audit §9 multi-band `synthetic_dataset_tree` pytest fixture.
- `backend/tests/test_io_exif.py` - 17 unit, security, AST prohibition, supply-chain `pyproject.toml`/`uv.lock`, and CLI tracer integration tests.

## Decisions Made

- Used strict UTF-8 decoding (returning `None` on `UnicodeDecodeError`) inside `_parse_xiaomi_zoom` so corrupted non-UTF-8 byte payloads fail safely instead of silently stripping invalid bytes into accidental JSON.
- Painted dual red (top-left) and green (top-right) corner blocks with 4:4:4 chroma subsampling (`quality=95, subsampling=0`) in `make_synthetic_jpeg_bytes` so all 6 tested orientations (`1, 3, 5, 6, 7, 8`) verify distinct physical pixel transposition coordinates rather than only swapped array dimensions.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Initial `uv run ruff check backend/` pass after Task 2 flagged line-length (`E501`) and Yoda-condition (`SIM300`, `config.CONST == expected_var`) rules in the newly authored test files; reformatted assertions and comments cleanly and re-verified `17 passed` + `All checks passed!`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Ready for Plan `01-02`: `uv` Python 3.12 environment, `backend/app/pdi/io_exif.py`, `backend/app/config.py`, and the multi-band `synthetic_dataset_tree` fixture (already containing CP866, Arabic-Indic, `_1` burst, Windows-renamed `116 (1).jpg`, nested `171/173/`, `367` LCD correction, `1340_` suffix, quarantined `417`/`1351`, and `>= 5` clean sessions per weight band) are verified and in place.
- Note for Plan `01-03`: `backend/tests/test_io_exif.py` invokes `build_manifest.main(...)` exclusively and avoids asserting raw counts that include quarantined folders, so all 17 tests will remain green without modification when `build_manifest.py` is upgraded from the tracer to the full Audit §9 curation/split/summary pipeline.

## Self-Check: PASSED

- Verified all 8 created artifacts exist on disk (`.python-version`, `pyproject.toml`, `uv.lock`, `backend/app/config.py`, `backend/app/pdi/io_exif.py`, `backend/scripts/build_manifest.py`, `backend/tests/conftest.py`, `backend/tests/test_io_exif.py`).
- Verified atomic task commits `5c2210e` and `de88c33` in `git log`.
- Verified `uv sync --frozen --no-build --managed-python` (`Python 3.12.14`), `uv run pytest backend/tests/test_io_exif.py -v` (`17 passed in 0.48s`), and `uv run ruff check backend/` (`All checks passed!`).

---
*Phase: 01-forensic-data-curation-anti-leakage-split*
*Completed: 2026-11-20*
