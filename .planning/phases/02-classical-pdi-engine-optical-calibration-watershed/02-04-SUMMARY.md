---
phase: 02-classical-pdi-engine-optical-calibration-watershed
plan: 04
subsystem: pdi-engine
tags: [batch-extraction, pyarrow, parquet, optical-calibration, dual-lens, multiprocessing, asvs-v4, audit, performance]

# Dependency graph
requires:
  - phase: 01-forensic-data-curation-anti-leakage-split
    provides: Materialized 1,738x24 data/processed/manifest_clean.parquet with locked 5-fold StratifiedGroupKFold session_id splits and read-only raw dataset
  - phase: 02-classical-pdi-engine-optical-calibration-watershed
    provides: Frozen PDIConfig, single-source run_single_bird_pdi() pipeline, 49-key FEATURE_COLUMNS / BANNED_FEATURE_COLUMNS schema, and complete backend.app.pdi barrel (Plans 02-01, 02-02, 02-03)
provides:
  - Capped 4-worker offline batch extraction CLI (backend/scripts/extract_features.py) with ASVS V4 path/symlink containment, ProcessPoolExecutor(max_workers=min(workers,4), initializer=cv2.setNumThreads(1)), 60-column typed PyArrow Parquet writer, --dry-run support, aggregate-only stdout, and byte-deterministic 4-scope dual-lens calibration audit generator
  - Materialized git-ignored data/processed/features_clean.parquet (1,738 rows x 60 typed PyArrow columns: 11 join/audit + 49 finite mask-gated float64 features, exact 1:1 row alignment with manifest_clean.parquet, 0.2301% primary fallback = 4/1,738, 0 LCD bucket false positives, 57.08 s batch wall-clock < 60 s)
  - Git-tracked byte-idempotent models/pdi_calibration_audit.json (SHA-256 3c2029095c4b276799d9c28ec210d105e2973e525c1d7293c7971c1fc6d8eba6, zero forbidden leak tokens) empirically proving area_px / (zoom_multiple ** 2) collapses Wide/UW ratio to 0.963757 on bucket pairs and beats double-counted area_px / ((focal_length_mm * zoom_multiple) ** 2) across all 4 paired scopes
  - 17-test CLI, security, schema, calibration, PROIBIÇÃO 4, and git-hygiene integration suite (backend/tests/test_extract_features_integration.py)
affects: [phase-3-modeling, phase-4-web-mvp]

# Plan ledger
plan_head_before: 501b4dc4a3fe832d8baaab96db3be866f8cf3217
commits:
  - af22902
  - bcb6b9e
  - 150f6f7

# Actuals
actuals:
  tokens: 21522
  tasks: 2
  commits: 3

tech-stack:
  added: []
  patterns:
    - Capped-concurrency spawn ProcessPoolExecutor(max_workers=min(workers, 4), initializer=_init_worker) with cv2.setNumThreads(1) preserving exact manifest row order via executor.map(..., chunksize=8)
    - Explicit 60-column PyArrow schema serialization (11 manifest/audit columns + 49 float64 FEATURE_COLUMNS) with runtime BANNED_FEATURE_COLUMNS and finite-matrix guards
    - Byte-deterministic UTF-8 JSON calibration audit omitting volatile wall-clock fields (logged to stdout only) and guarded at write time against PROIBIÇÃO 4 leak tokens (.jpg, rel_path, /home/, data/raw)
    - Bit-identical classical PDI micro-optimizations (contour-fill hole filling, bbox-cropped Lab/HSV/eroded-texture, histogram uint8 quantiles, vectorized diamond-offset scanline convex-hull area, uint8 np.where scalars) cutting 4-worker batch wall-clock by 33.2% (85.41 s -> 57.08 s) with exact 10-digit feature & SHA-256 audit parity

key-files:
  created:
    - backend/scripts/extract_features.py
    - models/pdi_calibration_audit.json
    - backend/tests/test_extract_features_integration.py
  modified:
    - backend/app/pdi/features.py
    - backend/app/pdi/io_exif.py
    - backend/app/pdi/lcd_suppression.py
    - backend/app/pdi/segmentation.py

key-decisions:
  - "Omitted volatile batch_wall_clock_sec from models/pdi_calibration_audit.json (logging elapsed wall-clock strictly to stdout while recording capped workers=4 in JSON) so consecutive runs yield byte-identical SHA-256 digests (3c2029095c4b276799d9c28ec210d105e2973e525c1d7293c7971c1fc6d8eba6)."
  - "Eliminated redundant full-frame allocations and skimage.measure.regionprops convex_hull_image rasterization overhead via 100% bit-identical equivalents (cv2.findContours external hole fill, tight-bbox Lab/HSV & 4-px zero-padded texture, bincount uint8 quantiles, single-pass cv2.moments, vectorized diamond-offset scanline exact convex-hull area, integer chroma_sq, and uint8 np.where scalars) to bring 4-worker batch extraction from 85.41 s down to 57.08 s (< 60 s D-10 SLA) with zero numerical drift."

patterns-established:
  - "Pattern 1 (ASVS V4 manifest path containment): Pre-validate every manifest rel_path against canonical dataset_root.resolve() rejecting absolute paths, parent traversal ('..'), and symlinks before spawning any worker pool."
  - "Pattern 2 (Byte-idempotent aggregate calibration audit): Build 4-scope dual-lens calibration reports deterministically from features_clean.parquet and scan serialized JSON against _FORBIDDEN_LEAK_SUBSTRINGS before writing to models/."
  - "Pattern 3 (Zero-copy tight-bbox mask-gated extraction): Crop pixel-local color conversions and 4-px zero-padded morphological texture stencils strictly to cv2.boundingRect(mask_u8) so CPU and L3 cache scale cleanly across 4 workers while preserving exact 0.0 outside-mask invariance."

requirements-completed:
  - PDI-03

coverage:
  - id: D1
    description: "Capped 4-worker CLI (backend/scripts/extract_features.py) with default repo-anchored paths, 1..4 worker clamping, ASVS V4 path/symlink containment, --dry-run zero-write behavior, and zero .jpg stdout emission"
    requirement: PDI-03
    verification:
      - kind: unit
        ref: "backend/tests/test_extract_features_integration.py#test_parse_args_defaults_worker_clamping_and_invalid_workers_rejection"
        status: pass
      - kind: unit
        ref: "backend/tests/test_extract_features_integration.py#test_asvs_v4_path_containment_rejects_traversal_and_symlinks"
        status: pass
      - kind: unit
        ref: "backend/tests/test_extract_features_integration.py#test_cli_dry_run_writes_zero_files_and_emits_zero_jpg_stdout"
        status: pass
      - kind: unit
        ref: "backend/tests/test_extract_features_integration.py#test_cli_full_run_mixed_lens_schema_order_clamped_workers_and_byte_idempotency"
        status: pass
    human_judgment: false
  - id: D2
    description: "Materialized git-ignored data/processed/features_clean.parquet (1,738 x 60 typed PyArrow schema, exact 1:1 manifest alignment, zero NaN/Inf, zero BANNED_FEATURE_COLUMNS, uniform pdi_config_hash, 4/1,738 = 0.2301% primary fallback < 2.0%, 0 bucket LCD false positives, 57.08 s batch wall-clock < 60 s)"
    requirement: PDI-03
    verification:
      - kind: unit
        ref: "backend/tests/test_extract_features_integration.py#test_schema_constants_have_exact_60_columns_and_exclude_banned_columns"
        status: pass
      - kind: integration
        ref: "backend/tests/test_extract_features_integration.py#test_real_features_clean_parquet_schema_alignment_finite_values_and_fallback_gate"
        status: pass
    human_judgment: false
  - id: D3
    description: "Git-tracked byte-idempotent models/pdi_calibration_audit.json (zero forbidden PROIBIÇÃO 4 tokens) quantitatively proving area_px / (zoom_multiple ** 2) collapses bucket Wide/UW ratio to 0.963757 and outperforms double-counted area_px / ((focal_length_mm * zoom_multiple) ** 2) across all 4 paired dual-lens scopes"
    requirement: PDI-03
    verification:
      - kind: unit
        ref: "backend/tests/test_extract_features_integration.py#test_write_calibration_audit_rejects_proibicao_4_leak_markers"
        status: pass
      - kind: integration
        ref: "backend/tests/test_extract_features_integration.py#test_real_pdi_calibration_audit_json_zero_leakage_and_4_scope_optical_proof"
        status: pass
    human_judgment: false
  - id: D4
    description: "Git hygiene intact: data/raw and data/processed remain completely untracked while models/pdi_calibration_audit.json is git-tracked and not ignored"
    requirement: PDI-03
    verification:
      - kind: integration
        ref: "backend/tests/test_extract_features_integration.py#test_git_hygiene_excludes_features_parquet_while_tracking_calibration_audit"
        status: pass
    human_judgment: false

duration: 35min
completed: 2026-11-23
status: complete
---

# Phase 02 Plan 04: Offline Batch Feature Extraction & Dual-Lens Optical Calibration Audit Summary

**Materialized the frozen `1,738 × 60` typed PyArrow `features_clean.parquet` (`0.2301%` primary fallback = `4 / 1,738`, `0` bucket LCD false positives, `57.08 s` on `4` workers) and byte-idempotent `models/pdi_calibration_audit.json` empirically proving `area_px / (zoom_multiple ** 2)` collapses Wide/Ultrawide ratio to `0.963757` on bucket pairs and beats double-counted `area_px / ((focal_length_mm * zoom_multiple) ** 2)` across all 4 paired scopes**

## Performance

- **Duration:** ~35 min (across Task 1 initial session + Task 2 continuation)
- **Started:** 2026-11-22T13:35:00-03:00
- **Completed:** 2026-11-23T16:02:36-03:00
- **Tasks:** 2 (Task 1 TDD CLI & unit/security tests, Task 2 real-corpus materialization, `< 60 s` perf SLA fix & integration gate)
- **Files modified:** 3 created (`backend/scripts/extract_features.py`, `models/pdi_calibration_audit.json`, `backend/tests/test_extract_features_integration.py`), 4 modified for bit-identical performance optimization (`backend/app/pdi/features.py`, `backend/app/pdi/io_exif.py`, `backend/app/pdi/lcd_suppression.py`, `backend/app/pdi/segmentation.py`)

## Accomplishments

- Built `backend/scripts/extract_features.py` enforcing ASVS V4 pre-pool path/symlink containment (`_resolve_safe_image_path`, `T-02-10`), strict worker concurrency capping (`MAX_WORKERS_CAP = 4`, `cv2.setNumThreads(1)` via `_init_worker`, `T-02-11`), `--dry-run` zero-file execution, frozen 60-column typed PyArrow schema (`_FEATURES_ARROW_SCHEMA`, `D-08` / `D-10`), exact 1:1 `image_id` alignment with `manifest_clean.parquet`, runtime `PROIBIÇÃO 4` leak guards (`_FORBIDDEN_LEAK_SUBSTRINGS`), and strictly aggregate stdout (`T-02-12`).
- Materialized local git-ignored `data/processed/features_clean.parquet` (`656 KB`, exact `1,738 × 60` PyArrow schema, `0` NaN/Inf across all `1,738 × 49` features, `0` banned columns, uniform `pdi_config_hash = "43a3cfaa3208019e"`, `377` dorsal bucket + `1,361` scale/litter scenes, **`4 / 1,738 = 0.2301%`** primary fallback vs `< 2.0%` `D-07` gate, `786` LCD suppressions with **`0`** bucket false positives) in **`57.08 s`** (`< 60.0 s` `D-10` SLA) on `4` workers.
- Published git-tracked, byte-idempotent `models/pdi_calibration_audit.json` (`SHA-256 = 3c2029095c4b276799d9c28ec210d105e2973e525c1d7293c7971c1fc6d8eba6`, verified identical across consecutive runs with zero `.jpg`/`rel_path`/`/home/`/`data/raw` tokens), proving across all 4 paired dual-lens scopes that `area_px / (zoom_multiple ** 2)` outperforms all other formulas — and specifically crushes the rejected double-counted `area_px / ((focal_length_mm * zoom_multiple) ** 2)`:
  - `bucket_only_pairs` (`N = 76`): winning `med(W/UW) = 0.963757`, `med|log| = 0.413364` vs. rejected `0.145264`, `1.929227`
  - `same_scene_pairs` (`N = 175`): winning `med(W/UW) = 0.932643`, `med|log| = 0.474557` vs. rejected `0.140574`, `1.962020`
  - `scale_or_litter_pairs` (`N = 99`): winning `med(W/UW) = 0.922404`, `med|log| = 0.510654` vs. rejected `0.139031`, `1.973059`
  - `all_mixed_lens_sessions` (`N = 127`): winning `med(W/UW) = 0.748935`, `med|log| = 0.482596` vs. rejected `0.112884`, `2.181391`
- Expanded `backend/tests/test_extract_features_integration.py` to 17 passing tests covering CLI clamping/dry-run/containment, synthetic mixed-lens extraction + byte-idempotency, real `1,738 × 60` Parquet schema/alignment/finite/`< 2%`-fallback verification, 4-scope calibration audit contract & `PROIBIÇÃO 4` leak guards, and git hygiene (`git ls-files data/raw data/processed` empty; `git check-ignore models/pdi_calibration_audit.json` exits `1`).

## Task Commits

Each task (plus the bit-identical `< 60 s` SLA performance fix) was committed atomically:

1. **Task 1: Build 4-worker batch extraction CLI (`extract_features.py`), typed 60-column Parquet writer, and 4-scope dual-lens calibration audit** - `af22902` (feat)
2. **Auto-fix (Rule 1/3 perf SLA): Eliminate bit-identical PDI hotspots to meet `< 60 s` 4-worker batch SLA** - `bcb6b9e` (perf)
3. **Task 2: Materialize real `features_clean.parquet` (`1,738 x 60`) and `models/pdi_calibration_audit.json` from read-only corpus and pass full Phase 2 integration & git-hygiene gate** - `150f6f7` (feat)

## Files Created/Modified

- `backend/scripts/extract_features.py` - Capped 4-worker batch PDI feature extractor CLI, ASVS V4 path containment, frozen 60-column PyArrow Parquet writer, and byte-deterministic 4-scope calibration audit generator.
- `models/pdi_calibration_audit.json` - Git-tracked aggregate extraction & dual-lens calibration audit report (`1,738` clean images, zero raw paths).
- `backend/tests/test_extract_features_integration.py` - 17 CLI, ASVS V4 security, synthetic mixed-lens, real-corpus `1,738 × 60` Parquet, 4-scope calibration audit, `PROIBIÇÃO 4`, and git-hygiene tests.
- `backend/app/pdi/features.py` - Bit-identical tight-bbox Lab/HSV/4-px zero-padded texture extraction, histogram `uint8` quantiles, single-pass `cv2.moments` eccentricity/Hu, and vectorized diamond-offset scanline exact convex-hull area (`13×` speedup on `area_convex`, `0` pixel diffs).
- `backend/app/pdi/segmentation.py` - Bit-identical external contour hole fill (`_fill_holes_u8`), ring-only `classify_scene_border_ring`, bbox-sliced `_select_dominant_component`, histogram `uint8` medians, and `uint8` `np.where` scalars.
- `backend/app/pdi/lcd_suppression.py` - Exact integer squared-chroma (`chroma_sq`) and `uint8` `np.where` scalars avoiding full-frame `float32`/`int64` temporaries.
- `backend/app/pdi/io_exif.py` - Skip redundant `.convert("RGB")` full-buffer copy when `ImageOps.exif_transpose()` already yields mode `"RGB"`.

## Decisions Made

- Kept volatile `wall_clock_sec` in aggregate stdout logs while omitting it from `models/pdi_calibration_audit.json`, ensuring reproducible SHA-256 byte-idempotency (`3c2029095c4b276799d9c28ec210d105e2973e525c1d7293c7971c1fc6d8eba6`) across consecutive runs.
- Preserved `MAX_WORKERS_CAP = 4` (`D-10` / `T-02-11`) strictly and met the `< 60 s` batch SLA by optimizing pure-CPU full-frame allocation and `regionprops.area_convex` hotspots bit-identically inside `backend/app/pdi/`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 / Rule 3 - Performance SLA] 4-worker batch extraction initially took `85.41 s` (`> 60.0 s` `D-10` SLA)**
- **Found during:** Task 2 Step 1 (real-corpus batch verification run of `uv run python -m backend.scripts.extract_features --workers 4`)
- **Issue:** Full 49-feature extraction + segmentation averaged `110.1 ms/image` single-core and scaled `2.42×` across 4 workers on the 15W Ryzen 7 5700U host (`85.41 s` total) due to `skimage.measure.regionprops.area_convex` (`9.2 ms/img`), `scipy.ndimage.binary_fill_holes` (`6.5 ms/img`), full-frame `np.where(..., 255, 0)` `int64` intermediates, full-frame Lab/HSV/chroma/Scharr conversions, and redundant `.convert("RGB")` copies.
- **Fix:** Replaced each hotspot with its exact mathematical, 100% bit-identical equivalent (`_area_convex_exact` vectorized diamond-offset scanline `9.20 ms → 0.71 ms`, `_fill_holes_u8` via external contours `4.48 ms → 0.48 ms`, tight-bbox Lab/HSV & 4-px zero-padded eroded texture, `np.bincount` `uint8` quantiles/medians, bbox-sliced dominant component prior, integer `chroma_sq`, `uint8` `np.where` scalars, and conditional `.convert("RGB")`).
- **Files modified:** `backend/app/pdi/features.py`, `backend/app/pdi/segmentation.py`, `backend/app/pdi/lcd_suppression.py`, `backend/app/pdi/io_exif.py`
- **Verification:** Re-running `uv run python -m backend.scripts.extract_features --workers 4` dropped wall-clock time from `85.41 s` to **`57.08 s`** (`-33.2%`, `< 60.0 s` SLA met), produced the **exact same SHA-256** (`3c2029095c4b276799d9c28ec210d105e2973e525c1d7293c7971c1fc6d8eba6`) for `models/pdi_calibration_audit.json`, matched all 10-digit feature-table aggregates verbatim (`area_px_sum = 2800406052.214844`, `convex_sum = 3991665487.054688`, `ecc_sum = 1319.3819569315`, `fallbacks = 4`, `bucket = 377`, `lcd_trig = 786`), and passed all `130 / 130` backend tests + `ruff check backend/`.
- **Committed in:** `bcb6b9e` (isolated atomic `perf(02-04)` commit immediately preceding Task 2 `150f6f7`)

---

**Total deviations:** 1 auto-fixed (1 Rule 1/Rule 3 bit-identical performance SLA fix)
**Impact on plan:** Essential to satisfy `D-10` / `02-04-PLAN.md` `< 60 s` batch SLA without violating `MAX_WORKERS_CAP = 4`, with zero numerical drift (`0` SHA-256 diff), zero config hash drift (`43a3cfaa3208019e`), and faster online Phase 4 `/api/predict` inference as a bonus.

## Issues Encountered

- Prior executor session hit an upstream API 429 quota error right after committing Task 1 (`af22902`) and generating the initial `features_clean.parquet` and `models/pdi_calibration_audit.json`; continuation resumed cleanly from `af22902`, verified byte-idempotency against the prior run's `H1` digest, resolved the `< 60 s` SLA gap, and completed Task 2 verification.

## User Setup Required

None - zero new external dependencies added (`pyproject.toml` and `uv.lock` untouched).

## Next Phase Readiness

- **Phase 2 (`PDI-01`..`PDI-05`) is 100% complete (`4 / 4` plans executed).**
- Ready for **Phase 3 (`Tabular Regression, Optical Ablation & Production Bundling`, `MOD-01`..`MOD-03`)**:
  - `data/processed/features_clean.parquet` (`1,738 × 60`, git-ignored, zero NaN/Inf, `0.2301%` fallback, locked 5-fold `session_id` splits, `pdi_config_hash = "43a3cfaa3208019e"`) is materialized and validated on disk.
  - `models/pdi_calibration_audit.json` (git-tracked, zero raw paths) is ready for Phase 3 optical ablation comparison and Phase 4 `/api/metrics` dashboard display.
  - All `130` backend tests and `uv run ruff check backend/` pass with zero failures.

## Self-Check: PASSED

- Verified all created/modified files exist on disk (`backend/scripts/extract_features.py`, `models/pdi_calibration_audit.json`, `backend/tests/test_extract_features_integration.py`, `data/processed/features_clean.parquet`, `backend/app/pdi/features.py`, `backend/app/pdi/segmentation.py`, `backend/app/pdi/lcd_suppression.py`, `backend/app/pdi/io_exif.py`).
- Verified all 3 atomic commits (`af22902`, `bcb6b9e`, `150f6f7`) in `git log` after `plan_head_before = 501b4dc4a3fe832d8baaab96db3be866f8cf3217`.
- Verified live 4-worker batch extraction (`wall_clock_sec = 57.08 < 60.0 s`, zero `.jpg` in stdout, byte-identical SHA-256 `3c2029095c4b276799d9c28ec210d105e2973e525c1d7293c7971c1fc6d8eba6`).
- Verified `uv run pytest backend/tests/test_extract_features_integration.py -v` (`17 passed in 4.02s`), full suite `uv run pytest -v` (`130 passed in 14.47s`), and `uv run ruff check backend/` (`All checks passed!`).
- Verified git hygiene (`git ls-files data/raw data/processed` empty; `git check-ignore models/pdi_calibration_audit.json` exit `1`; `PROIBIÇÃO 1–4` compliant).

---
*Phase: 02-classical-pdi-engine-optical-calibration-watershed*
*Completed: 2026-11-23*
