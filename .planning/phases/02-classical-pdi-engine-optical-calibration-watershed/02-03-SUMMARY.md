---
phase: 02-classical-pdi-engine-optical-calibration-watershed
plan: 03
subsystem: pdi-engine
tags: [watershed, scikit-image, scipy, opencv, multi-bird, red-feeder-suppression, h-maxima, base64, barrel]

# Dependency graph
requires:
  - phase: 02-classical-pdi-engine-optical-calibration-watershed
    provides: Frozen PDIConfig, Pillow EXIF-first working loader (load_working_rgb, resolve_optical_context), Bilateral/Lab-CLAHE preprocessor (preprocess_working_rgb), 49-key strictly mask-gated feature extractor (extract_mask_gated_features), 4-stage X-Ray renderer, and unified run_single_bird_pdi() pipeline (Plans 02-01 & 02-02)
provides:
  - Isolated classical Marker-based Watershed multi-bird separation engine (backend/app/pdi/watershed.py) with ASVS V5 bounded tunable relative parameters (h_rel, min_distance_rel, min_area_ratio), chromatic red-feeder pre-suppression, positive-score Otsu robust z-score union foreground mask with wall-strip rejection, exact float64 EDT + scale-adaptive Gaussian smoothing, grouped-bbox h_maxima (bit-identical to full-frame, 15x-45x faster), modern coordinate-only peak_local_max + disconnected-component rescue, skimage.segmentation.watershed, tiny-basin pruning, deterministic (-area_work_px, round(cy, 3), round(cx, 3)) ordering, strictly mask-gated 49-key per-instance feature extraction, and on-demand 4-stage Base64 visual rendering
  - Complete alphabetically sorted public Phase 2 PDI & Watershed package barrel (backend/app/pdi/__init__.py)
  - 37-test suite (backend/tests/test_pdi_watershed.py) covering all 5 PDI-05 deterministic edge probes (boundary, adjacency, empty, ordering, precision), red-feeder pre-suppression, wall-strip rejection, barrel export integrity, and read-only real-corpus audit across all 13 quarantined multi-bird images (9/13 exact, 12/13 within +-1, 4/4 exact on folder 417, p95=234.5 ms, max=260.5 ms)
affects: [02-04, phase-3-modeling, phase-4-web-mvp]

# Plan ledger
plan_head_before: 92f5f13dc0ec2ffcdc0f8d2b0eeaebe40c3e9bd7
commits:
  - 88b916c
  - a236c34

# Actuals
actuals:
  tokens: 13606
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - Positive-score Otsu thresholding on robust median/MAD Lab-L/Lab-a/Scharr-gradient z-scores (lo=0.10, hi=p99.0, Otsu strictly on score > 0.10 floored at 78) avoiding zero-clipping Dirac spike collapse on cluttered straw scenes
    - 5x5-dilated component-grouped +-2 px padded bbox h_maxima achieving bit-identical parity with full-frame h_maxima (verified across h_rel in [0.05, 0.80] and all 13 real scenes) while cutting peak-detection latency from 119-305 ms down to 5.1-31.5 ms
    - Strictly isolated per-basin extract_mask_gated_features(frame.bgr_work, inst_mask, scale_factor, optical) calls guaranteeing zero cross-bird feature bleed and zero folder-name hardcoding

key-files:
  created:
    - backend/app/pdi/watershed.py
    - backend/app/pdi/__init__.py
    - backend/tests/test_pdi_watershed.py
  modified: []

key-decisions:
  - "Rescaled the fused robust z-score strictly above positive excess floor (_SCORE_POS_FLOOR=0.10 to p99.0) and computed Otsu over positive valid pixels (score > 0.10, floored at 78) so the background negative-score Dirac spike cannot collapse Otsu downward into straw clutter."
  - "Implemented _compute_bbox_h_maxima() over 5x5-dilated foreground component groups (+-2 px padded) rather than one full-union bounding box so dispersed multi-bird scenes (e.g. 4 chicks spanning the frame) run h_maxima in 5-31 ms instead of 130-165 ms while remaining 100% bit-identical to full-frame h_maxima."
  - "Included OrientedImage alongside all 33 Plan 02-03 barrel exports in backend/app/pdi/__init__.py (34 total alphabetically sorted symbols) to preserve full alignment with 02-PATTERNS.md and Phase 1 io_exif consumers."

patterns-established:
  - "Pattern 1 (Zero-hardcoding multi-bird separation): Watershed accepts arbitrary Path | bytes images with tunable ASVS V5 bounded relative parameters (h_rel, min_distance_rel, min_area_ratio) and contains zero folder-name or preset-ID branching."
  - "Pattern 2 (Grouped-bbox bit-identical h_maxima): Grouping foreground components via 5x5 dilation before +-2 px bbox cropping guarantees mathematical bit-identity with full-frame h_maxima even for closely-spaced blobs while preserving sub-350 ms end-to-end CPU latency."
  - "Pattern 3 (Per-instance mask-gated features): Every separated WatershedInstance carries the full 49-key FEATURE_COLUMNS dict computed strictly on its own mutually exclusive binary mask over pixel-local frame.bgr_work, enabling illustrative per-bird weight inference in Phase 4 WEB-04."

requirements-completed:
  - PDI-05

coverage:
  - id: D1
    description: "ASVS V5 bounded parameter validation on run_watershed_pdi() accepting exact min/max boundaries (h_rel in [0.05, 0.80], min_distance_rel in [0.20, 2.50], min_area_ratio in [0.01, 0.45]) and rejecting one-step-outside, bool, NaN, and Inf values with ValueError"
    requirement: PDI-05
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_accepts_exact_min_and_max_parameter_bounds"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_rejects_out_of_bounds_bool_and_non_finite_params"
        status: pass
    human_judgment: false
  - id: D2
    description: "Touching-bird Marker-based Watershed separation splitting fused single-component silhouettes into mutually exclusive basins (bird_count == 2) while preserving isolated single birds, merging sub-threshold humps, rejecting border wall strips, and suppressing saturated red plastic feeders"
    requirement: PDI-05
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_separates_touching_fused_birds_with_disjoint_basins"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_preserves_isolated_single_bird_and_merges_subthreshold_humps"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_suppresses_red_feeder_and_excludes_it_from_bird_basins"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_rejects_border_architectural_wall_strip"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_basin_masks_are_mutually_exclusive_and_empty_h_maxima_safe"
        status: pass
    human_judgment: false
  - id: D3
    description: "Empty/uniform dark and all-red-feeder scenes return WatershedResult(bird_count=0, instances=()) cleanly without crashing, with valid Pillow-decodable visuals when return_visuals=True"
    requirement: PDI-05
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_handles_empty_uniform_dark_and_all_red_scenes_cleanly"
        status: pass
    human_judgment: false
  - id: D4
    description: "Deterministic instance ordering by (-area_work_px, round(centroid_y, 3), round(centroid_x, 3)) with contiguous instance_id 1..bird_count and exact Path-vs-bytes parity"
    requirement: PDI-05
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_deterministic_ordering_and_path_bytes_parity"
        status: pass
    human_judgment: false
  - id: D5
    description: "Exact float64 EDT/Gaussian + bit-identical bbox-cropped h_maxima parity, per-instance 49-key finite mask-gated features (zero banned columns), 4-stage Pillow-decodable Base64 visuals (<= 640 px, < 300 KB), and zero folder-name hardcoding"
    requirement: PDI-05
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_bbox_cropped_h_maxima_matches_full_frame_bit_identically"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_per_instance_49_features_optical_override_and_4_stage_visuals"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_watershed_module_contains_zero_folder_name_hardcoding"
        status: pass
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_ast_scan_confirms_zero_raw_opencv_image_loading_under_backend_app"
        status: pass
    human_judgment: false
  - id: D6
    description: "Complete alphabetically sorted backend/app/pdi/__init__.py public barrel and read-only audit across all 13 quarantined multi-bird images (9/13 exact, 12/13 within +-1, 4/4 exact on folder 417 with red-feeder suppression active, max latency 260.5 ms < 350 ms, zero raw filename leakage)"
    requirement: PDI-05
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_watershed.py#test_pdi_package_barrel_exports_complete_sorted_public_api"
        status: pass
      - kind: integration
        ref: "backend/tests/test_pdi_watershed.py#test_real_13_quarantined_multi_bird_images_read_only_watershed_audit"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-11-22
status: complete
---

# Phase 02 Plan 03: Isolated Marker-Based Watershed Multi-Bird Engine & Public PDI Barrel Summary

**Delivered the zero-hardcoding classical Marker-based Watershed engine (`watershed.py`) and complete sorted `backend.app.pdi` package barrel, hitting `9 / 13` exact counts (`12 / 13` within `±1`, `4 / 4` exact on red-feeder folder `417`, MAE = `0.38` birds) at `p95 = 234.5 ms` / `max = 260.5 ms` and `max = 145.1 KB` 4-stage Base64 visuals across all 13 quarantined multi-bird scenes**

## Performance

- **Duration:** 25 min
- **Started:** 2026-11-22T14:42:42Z
- **Completed:** 2026-11-22T15:07:33Z
- **Tasks:** 2 (Task 1 TDD, Task 2 barrel + real-corpus integration audit)
- **Files modified:** 3 created, 0 modified (`+1,403` lines)

## Accomplishments

- Implemented `backend/app/pdi/watershed.py` (`WatershedInstance`, `WatershedVisuals`, `WatershedResult`, `run_watershed_pdi()`) strictly isolated from single-bird training with zero DL/OCR dependencies, zero raw OpenCV image loading, and zero folder-name hardcoding (`D-11`, `D-12`, `PDI-05`).
- Enforced ASVS V5 parameter validation (`0.05 <= h_rel <= 0.80`, `0.20 <= min_distance_rel <= 2.50`, `0.01 <= min_area_ratio <= 0.45`, rejecting `bool`/`NaN`/`Inf` and out-of-range inputs with `ValueError`) plus deterministic minimum peak distance (`>= 10 px`), prominence (`>= 1.0`), and basin pruning floors (`T-02-07`).
- Combined chromatic red-feeder pre-suppression (`Lab a* > 146 & S > 65` / red-hue halo + `11×11` ellipse dilation), positive-score Otsu robust median/MAD z-score union segmentation with border wall-strip rejection, exact `float64` EDT + scale-adaptive Gaussian smoothing, 5×5-grouped `±2 px` bbox-cropped `h_maxima` (`100%` bit-identical to full-frame, cutting `h_maxima` from `119–305 ms` to `5.1–31.5 ms`), modern coordinate-only `peak_local_max` + disconnected-component seed rescue, `skimage.segmentation.watershed`, and tiny-basin pruning.
- Computed full 49-key strictly mask-gated feature vectors per separated basin via `extract_mask_gated_features(frame.bgr_work, inst_mask, working.scale_factor, optical)` with deterministic `(-area_work_px, round(cy, 3), round(cx, 3))` sorting (`T-02-08`), alongside on-demand 4-stage `<= 640 px` Base64 visual rendering (`stage1_original_b64`, 1-ch PNG `stage2_union_mask_b64`, `COLORMAP_INFERNO` `stage3_distance_peaks_b64`, alpha-blended `#id` `stage4_color_basins_b64`).
- Published `backend/app/pdi/__init__.py` re-exporting all 34 public Phase 2 PDI and Watershed symbols in an alphabetically sorted `__all__`.
- Verified all 5 deterministic `PDI-05` edge probes plus the read-only real 13-image multi-bird corpus audit (`manifest_quarantine.parquet`, zero raw filenames logged per `PROIBIÇÃO 4` / `T-02-09`): `9 / 13` exact (`417`: `[2, 2, 2, 2]`, `823`: `[5, 4, 4, 4]`, `1520`: `[2, 2, 3, 3]`, `757`: `[4]` backstop for Phase 4 `WEB-04` sliders), `12 / 13` within `±1`, `p50 = 183.9 ms`, `p95 = 234.5 ms`, `max = 260.5 ms` (`< 350 ms`), and combined Base64 visuals `p50 = 131.9 KB`, `max = 145.1 KB` (`< 300 KB`).

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement isolated Marker-based Watershed engine (`watershed.py`) and synthetic edge-probe unit test suite** - `88b916c` (feat)
2. **Task 2: Publish `backend/app/pdi/__init__.py` barrel and add read-only 13-image quarantined multi-bird audit (`417`, `757`, `823`, `1520`)** - `a236c34` (feat)

## Files Created/Modified

- `backend/app/pdi/watershed.py` - Isolated classical Marker-based Watershed engine with ASVS V5 parameter guards, chromatic red-feeder pre-suppression, grouped-bbox bit-identical `h_maxima`, per-instance 49-key mask-gated feature extraction, and 4-stage Base64 visuals.
- `backend/app/pdi/__init__.py` - Public package barrel re-exporting the locked Phase 2 single-bird PDI and Watershed APIs with an alphabetically sorted `__all__`.
- `backend/tests/test_pdi_watershed.py` - 37 tests covering ASVS V5 parameter boundaries, touching-bird separation, isolated-bird & sub-threshold hump preservation, red-feeder pre-suppression, border wall-strip rejection, empty/uniform scenes, equal-area deterministic ordering + `Path`/`bytes` parity, bbox-cropped vs full-frame `h_maxima` bit-equality, 49-key per-instance features, 4-stage Base64 Pillow decoding, zero-hardcoding guard, barrel export contract, and read-only 13-image multi-bird corpus audit.

## Decisions Made

- Used positive-excess z-score rescaling (`_SCORE_POS_FLOOR = 0.10` to `p99.0`) and computed Otsu strictly over positive valid pixels (`score > 0.10`, floored at `78.0`) so background negative-score pixels do not form a Dirac spike at `0` that drags Otsu downward into straw clutter.
- Upgraded `_compute_bbox_h_maxima()` from a single foreground-union bounding box to `5×5`-dilated component groups (`±2 px`-padded) so dispersed multi-bird scenes execute `h_maxima` in `5.1–31.5 ms` instead of `130–165 ms` while preserving exact `np.array_equal` bit-identity with full-frame `h_maxima`.
- Included `OrientedImage` alongside all 33 Plan `02-03` exports in `backend/app/pdi/__init__.py` (`34` alphabetically sorted symbols) so downstream callers can import all `io_exif` types directly from `backend.app.pdi`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Otsu foreground over-inclusion and full-union bbox `h_maxima` latency on dispersed multi-bird scenes**
- **Found during:** Task 1 verification & real-corpus prototyping (`backend/app/pdi/watershed.py`)
- **Issue:** Rescaling the full `[p1, p99.5]` z-score range mapped median background (`score ≈ 0`) above `100/255`, flooding `~30–35%` of straw into the foreground union mask (`exact = 3/13`, `max_lat = 505 ms`), and a single union bounding box across dispersed birds in folder `823` still spanned `~90%` of the frame (`130–165 ms` in `h_maxima`).
- **Fix:** Rescaled `score` above `_SCORE_POS_FLOOR = 0.10` to `p99.0`, computed Otsu strictly on positive valid pixels (`score > 0.10`, floored at `78.0`), and grouped foreground components via `5×5` dilation before running `±2 px`-padded bbox `h_maxima` (`bit-identical = True` on all 13 real scenes and synthetic `h_rel ∈ {0.05, 0.20, 0.80}` probes).
- **Files modified:** `backend/app/pdi/watershed.py`
- **Verification:** Achieved `9 / 13` exact (`417`: `4/4`, `823`: `3/4`, `1520`: `2/4`), `12 / 13` within `±1`, `max = 260.5 ms` (`< 350 ms`), and `37 / 37` passing tests in `backend/tests/test_pdi_watershed.py`.
- **Committed in:** `88b916c` (Task 1 commit)

**2. [Rule 1 - Bug] Corrected synthetic straw background color in `test_pdi_watershed.py` to avoid false-positive red-feeder triggering**
- **Found during:** Task 1 TDD verification (`backend/tests/test_pdi_watershed.py`)
- **Issue:** Initial synthetic straw helper used reddish-brown `BGR = (56, 84, 114)` (`Lab a* = 142 ± noise`), causing ~50% of straw noise pixels to exceed the red-feeder threshold `Lab a* > 146 & S > 65`.
- **Fix:** Updated `_make_straw_background()` to realistic golden-brown wood-shaving bedding `BGR = (48, 92, 112)` (`Lab a* ≈ 130, b* ≈ 148`), leaving straw completely untouched by red-feeder suppression while saturated red feeders (`BGR = (18, 22, 228)`) trigger `> 20,000 px` exclusion cleanly.
- **Files modified:** `backend/tests/test_pdi_watershed.py`
- **Verification:** All 35 synthetic unit tests passed immediately after the fix.
- **Committed in:** `88b916c` (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Essential correctness and latency fixes delivering exact Spike 4 count accuracy (`9/13` exact, `12/13` within `±1`, `4/4` on `417`) and sub-261 ms runtime with zero folder hardcoding and zero scope creep.

## Issues Encountered

None beyond the two Rule 1 issues diagnosed and fixed during Task 1 above. (Note: `scikit-image 0.26.0` emits an upstream `DeprecationWarning` inside `skimage/morphology/grayreconstruct.py:216` under NumPy 2.5 — non-blocking, zero behavioral impact.)

## User Setup Required

None - zero new dependencies installed (`pyproject.toml` and `uv.lock` untouched).

## Next Phase Readiness

- Ready for Wave 4 (`02-04`: `backend/scripts/extract_features.py` capped-concurrency 4-worker batch extraction across all `1,738` clean images into `data/processed/features_clean.parquet`, aggregate `models/pdi_calibration_audit.json`, and full Phase 2 integration tests).
- All 113 backend tests (`39` Phase 1 + `24` Plan `02-01` + `13` Plan `02-02` + `37` Plan `02-03`) and `uv run ruff check backend/` pass cleanly.

## Self-Check: PASSED

- Verified all 3 created files exist on disk (`backend/app/pdi/watershed.py`, `backend/app/pdi/__init__.py`, `backend/tests/test_pdi_watershed.py`).
- Verified both atomic task commits (`88b916c`, `a236c34`) in `git log` after `plan_head_before = 92f5f13dc0ec2ffcdc0f8d2b0eeaebe40c3e9bd7`.
- Verified `uv run pytest backend/tests/test_pdi_watershed.py backend/tests/test_io_exif.py -v` (`54 passed in 8.66s`), full suite `uv run pytest -v` (`113 passed in 11.42s`), and `uv run ruff check backend/` (`All checks passed!`).
- Verified `PROIBIÇÃO 1-4` compliance: zero DL/OCR imports, zero `cv2.imread`/`cv2.imdecode` calls under `backend/app/` (AST-checked), zero folder-name hardcoding in `watershed.py`, zero `pyproject.toml`/`uv.lock` edits, zero `data/raw/` mutations, zero raw filenames logged.

---
*Phase: 02-classical-pdi-engine-optical-calibration-watershed*
*Completed: 2026-11-22*
