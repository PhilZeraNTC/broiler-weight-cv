---
phase: 02-classical-pdi-engine-optical-calibration-watershed
plan: 02
subsystem: pdi-engine
tags: [opencv, scikit-image, pillow, feature-extraction, anti-leakage, optical-calibration, xray-base64, pipeline]

# Dependency graph
requires:
  - phase: 02-classical-pdi-engine-optical-calibration-watershed
    provides: Frozen PDIConfig, acyclic aspect-preserving Pillow EXIF loader (load_working_rgb, resolve_optical_context), Bilateral/Lab-CLAHE preprocessor, rotation-invariant SF-400 LCD suppressor, and 3-scenario classical single-bird segmenter with orientation-normalized fit_ellipse_safe (Plan 02-01)
provides:
  - Frozen 49-column immutable FEATURE_COLUMNS schema tuple (14 morphometrics + 26 pixel-local chromatic & eroded interior texture + 9 EXIF optical & calibrated descriptors) with explicit BANNED_FEATURE_COLUMNS exclusion and strictly mask-gated extract_mask_gated_features() (backend/app/pdi/features.py)
  - 4-stage classical PDI "X-Ray" visual annotation and raw RFC 4648 ASCII Base64 renderer capped at 640 px long edge and < 250 KB payload (backend/app/pdi/xray.py)
  - Unified single-source PDIResult and run_single_bird_pdi() orchestrator shared verbatim by offline batch extraction (return_xray=False) and Phase 4 FastAPI serving (return_xray=True), stamping pdi_config_hash and routing pixel-local bgr_work into feature extraction (backend/app/pdi/pipeline.py)
  - 8-test PROIBIÇÃO 2 outside-mask bit-invariance (0.0 diff), 90/180-deg rotation invariance, Hue S>=25 gating, tiny/empty-mask, and D-09 optical calibration test suite (backend/tests/test_pdi_features_invariance.py)
  - 5-test end-to-end pipeline, 4-stage Pillow Base64 decoding, Path-vs-bytes parity, EXIF-stripped/RGBA/grayscale upload degradation, synthetic full-pipeline SF-400 anti-LCD, and real native 4080x1840 < 400 ms latency SLA suite (backend/tests/test_pdi_pipeline_and_xray.py)
affects: [02-03, 02-04, phase-3-modeling, phase-4-web-mvp]

# Plan ledger
plan_head_before: 5ec98d5e9bbc1fcec5f746899dfc35192b96c14e
commits:
  - 6296f5e
  - 898cf74

# Actuals
actuals:
  tokens: 13853
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - Strictly pixel-local cv2.cvtColor(bgr_work, ...) indexed at bird_mask == 255 (never tile-coupled bgr_clahe or spatial bgr_smooth)
    - 3-px eroded interior (7x7 ellipse, BORDER_CONSTANT=0) mask-zeroed Scharr/Laplacian texture evaluated via bit-deterministic float64 np.hypot(gx[interior], gy[interior])
    - Empirically verified Spike 2 optical normalization area_optical_norm = area_px / (zoom_multiple ** 2) paired with resize-agnostic area_frac_optical_norm = area_frac / (zoom_multiple ** 2) (zero focal*zoom double-counting)
    - 4-stage 640 px X-Ray encoding (3x JPEG Q=82 + 1x 1-channel PNG compression=6) as raw RFC 4648 ASCII Base64 skipped completely when return_xray=False

key-files:
  created:
    - backend/app/pdi/features.py
    - backend/app/pdi/xray.py
    - backend/app/pdi/pipeline.py
    - backend/tests/test_pdi_features_invariance.py
    - backend/tests/test_pdi_pipeline_and_xray.py
  modified: []

key-decisions:
  - "Replaced OpenCV float32 cv2.magnitude(gx, gy) with strictly masked float64 np.hypot(gx[interior], gy[interior]) inside extract_mask_gated_features() after TDD caught 1-ULP (~1.8e-8) SIMD non-determinism across identical calls on OpenCV 4.14.0."
  - "Applied explicit borderType=cv2.BORDER_CONSTANT, borderValue=0 during 7x7 elliptical erosion of bird_mask so 3x3 Scharr/Laplacian stencils stay strictly inside both mask boundaries and image borders even when a silhouette touches frame edges."
  - "Rendered Stage 4 contour, fitted ellipse, and orthogonal amber/cyan major/minor axes at canonical working scale before aspect-preserving INTER_AREA downscale to 640 px so axis geometry matches fit_ellipse_safe() without coordinate rescaling drift."

patterns-established:
  - "Pattern 1 (PROIBIÇÃO 2 bit-identical feature gate): Always route raw pixel-local frame.bgr_work (never frame.bgr_clahe or frame.bgr_smooth) into extract_mask_gated_features() and compute derivative magnitudes via float64 np.hypot on 3-px eroded interior pixels for exact 0.0 diff under outside-mask perturbation."
  - "Pattern 2 (Single-source PDIResult contract): Both offline batch extraction (return_xray=False) and online API inference (return_xray=True) execute run_single_bird_pdi(), stamping pdi_config_hash and keeping X-Ray encoding zero-overhead during batch extraction."
  - "Pattern 3 (Raw Base64 X-Ray framing): render_xray_artifacts() returns raw RFC 4648 ASCII Base64 strings without data:image/... URI prefixes so Phase 4 FastAPI schemas own transport framing cleanly."

requirements-completed:
  - PDI-03
  - PDI-04

coverage:
  - id: D1
    description: "Immutable 49-column FEATURE_COLUMNS tuple (14 morphometric + 26 chromatic/texture + 9 optical) excluding all BANNED_FEATURE_COLUMNS (oriented_w, oriented_h, megapixels, bbox_w, bbox_h, raw_w, raw_h)"
    requirement: PDI-03
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_features_invariance.py#test_feature_schema_has_exact_49_unique_columns_and_excludes_banned_columns"
        status: pass
    human_judgment: false
  - id: D2
    description: "Strictly mask-gated pixel-local feature extraction with exact 0.0 outside-mask perturbation diff (PROIBIÇÃO 2), 90/180-deg rotation invariance, Hue S>=25 gating (<10 px -> 0.0), and empty/tiny-mask finite guards"
    requirement: PDI-03
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_features_invariance.py#test_outside_mask_perturbation_yields_bit_identical_features"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_features_invariance.py#test_rotation_invariance_90_and_180_degrees"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_features_invariance.py#test_achromatic_hue_gating_tiny_mask_and_empty_mask_guards"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_features_invariance.py#test_extract_mask_gated_features_rejects_invalid_inputs"
        status: pass
    human_judgment: false
  - id: D3
    description: "Empirically verified optical calibration area_optical_norm = area_px / (zoom_multiple ** 2) and area_frac_optical_norm = area_frac / (zoom_multiple ** 2) with zero focal_length_mm * zoom_multiple double-counting"
    requirement: PDI-03
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_features_invariance.py#test_optical_calibration_formulas_and_no_focal_zoom_double_counting"
        status: pass
    human_judgment: false
  - id: D4
    description: "4-stage X-Ray Base64 visual renderer capped at 640 px long edge (3x JPEG Q=82 + 1x 1-channel PNG, < 250 KB payload, Pillow-decodable, ellipse + orthogonal major/minor axis alignment)"
    requirement: PDI-04
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_pipeline_and_xray.py#test_render_xray_artifacts_decodes_all_four_stages_within_budget_and_axis_alignment"
        status: pass
    human_judgment: false
  - id: D5
    description: "Unified single-source run_single_bird_pdi() orchestrator with Path-vs-bytes parity, return_xray toggle, graceful EXIF-stripped/RGBA/grayscale upload fallback + OpticalOverride, full-pipeline SF-400 anti-LCD exclusion, and < 400 ms real 4080x1840 latency SLA"
    requirement: PDI-04
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_pipeline_and_xray.py#test_run_single_bird_pdi_path_vs_bytes_parity_and_xray_toggle"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_pipeline_and_xray.py#test_run_single_bird_pdi_handles_stripped_jpeg_rgba_grayscale_png_and_override"
        status: pass
      - kind: integration
        ref: "backend/tests/test_pdi_pipeline_and_xray.py#test_full_pipeline_sf400_anti_lcd_excludes_digits_completely"
        status: pass
      - kind: integration
        ref: "backend/tests/test_pdi_pipeline_and_xray.py#test_real_native_4080x1840_wide_jpeg_meets_latency_and_payload_sla"
        status: pass
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_ast_scan_confirms_zero_raw_opencv_image_loading_under_backend_app"
        status: pass
    human_judgment: false

duration: 8min
completed: 2026-11-22
status: complete
---

# Phase 02 Plan 02: Strictly Mask-Gated 49-Feature Extractor, 4-Stage X-Ray Renderer & Unified Single-Bird PDI Pipeline Summary

**Locked the immutable 49-column strictly mask-gated feature extractor (`0.0` outside-mask perturbation diff, `area_px / zoom_multiple**2` optical calibration), 4-stage `640 px` Base64 X-Ray renderer (`< 250 KB`), and unified `run_single_bird_pdi()` orchestrator verified at `p95 = 182.7 ms` / `max = 206.2 KB` across 10 real native `4080×1840` Wide JPEGs**

## Performance

- **Duration:** 8 min
- **Started:** 2026-11-22T14:33:26Z
- **Completed:** 2026-11-22T14:40:49Z
- **Tasks:** 2 (both TDD)
- **Files modified:** 5 created, 0 modified (`+1,332` lines)

## Accomplishments

- Implemented `backend/app/pdi/features.py` locking the immutable 49-column `FEATURE_COLUMNS` tuple (`14` rotation-invariant morphometrics + `26` pixel-local Lab/HSV plumage chroma & 3-px eroded interior feather texture + `9` EXIF optical & calibrated descriptors) and banning raw pixel/bbox columns via `BANNED_FEATURE_COLUMNS` (`D-08`, `D-09`).
- Proved exact `0.0` outside-mask perturbation bit-invariance (`PROIBIÇÃO 2`) across all 49 features (both centered and border-touching masks) by slicing strictly pixel-local `cv2.cvtColor(bgr_work, ...)` at `bird_mask == 255`, gating circular Hue (`hsv_h_sin_mean`, `hsv_h_cos_mean`) strictly to `S >= 25` (`< 10` px → `0.0`), and evaluating Scharr/Laplacian texture strictly inside a `7×7` eroded interior (`BORDER_CONSTANT=0`) via bit-deterministic float64 `np.hypot`.
- Implemented `backend/app/pdi/xray.py` (`XRayArtifacts`, `render_xray_artifacts()`) downscaling all 4 inspection stages aspect-preserving to `config.xray_display_long_edge = 640` — Stages 1, 2, 4 as `JPEG Q=82` and Stage 3 binary bird mask as 1-channel `"L"` `PNG` — with Stage 4 drawing translucent silhouette tint, contour, fitted ellipse, and orthogonal amber/cyan major/minor axes geometrically aligned to `fit_ellipse_safe()`.
- Implemented `backend/app/pdi/pipeline.py` (`PDIResult`, `run_single_bird_pdi()`) chaining `load_working_rgb -> resolve_optical_context -> preprocess_working_rgb -> segment_single_bird -> extract_mask_gated_features(frame.bgr_work, ...) -> optional render_xray_artifacts()` identically across `Path` and `bytes`, skipping X-Ray encoding completely when `return_xray=False`, stamping `pdi_config_hash`, and degrading EXIF-stripped JPEG and RGBA/grayscale PNG uploads cleanly to `4.25 mm / 1.0x`.
- Verified end-to-end SLA across a read-only 10-image benchmark of real native `4080×1840` Wide JPEGs spanning orientations `1, 3, 6, 8` (zero raw filenames logged): `lat_p50 = 131.0 ms`, `lat_p95 = 182.7 ms`, `lat_max = 208.7 ms` (well below `< 400 ms` target / `< 500 ms` SLA) and combined 4-stage Base64 payload `p50 = 146.7 KB`, `max = 206.2 KB` (`< 250 KB`).

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement strictly mask-gated 49-feature extractor (`features.py`) and PROIBIÇÃO 2 invariance test suite** - `6296f5e` (feat)
2. **Task 2: Implement 4-stage "X-Ray" Base64 renderer (`xray.py`), unified `run_single_bird_pdi()` pipeline (`pipeline.py`), and end-to-end latency/upload/anti-LCD tests** - `898cf74` (feat)

## Files Created/Modified

- `backend/app/pdi/features.py` - Frozen 49-column `FEATURE_COLUMNS` tuple, `BANNED_FEATURE_COLUMNS`, and strictly mask-gated pixel-local `extract_mask_gated_features()`.
- `backend/app/pdi/xray.py` - `XRayArtifacts`, `_draw_stage4_overlay()`, and `render_xray_artifacts()` emitting 4 Pillow-decodable raw Base64 stages (`<= 640 px`, `< 250 KB`).
- `backend/app/pdi/pipeline.py` - Unified single-source `PDIResult` and `run_single_bird_pdi()` orchestrator.
- `backend/tests/test_pdi_features_invariance.py` - 8 tests verifying 49-key schema & banned-column exclusion, exact `0.0`-diff outside-mask perturbation invariance, 90°/180° rotation invariance, achromatic Hue `S >= 25` gating, tiny/empty-mask guards, D-09 optical formulas, and invalid-input errors.
- `backend/tests/test_pdi_pipeline_and_xray.py` - 5 tests verifying 4-stage X-Ray Pillow decoding & axis alignment, `Path`-vs-`bytes` parity & `return_xray` toggle, EXIF-stripped/RGBA/grayscale uploads + `OpticalOverride`, full-pipeline SF-400 anti-LCD exclusion, and real `4080×1840` Wide JPEG latency & payload SLA.

## Decisions Made

- Used strictly masked float64 `np.hypot(gx[interior].astype(np.float64), gy[interior].astype(np.float64))` instead of full-frame float32 `cv2.magnitude(gx, gy)` inside `extract_mask_gated_features()` to eliminate pointer-alignment SIMD rounding drift and guarantee 100% bit-deterministic `0.0` perturbation diff.
- Eroded `bird_mask` with `borderType=cv2.BORDER_CONSTANT, borderValue=0` (`7×7` ellipse) so interior `3×3` Scharr/Laplacian stencils never touch either outside-mask pixels or frame-border reflections.
- Kept X-Ray Stage 1–4 payloads as raw RFC 4648 ASCII Base64 strings without `data:image/...;base64,` URI prefixes (matching `02-RESEARCH.md` and Plan `02-02` Flagged Assumption PDI-04).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Eliminated 1-ULP SIMD non-determinism in `feather_grad_mean` (`cv2.magnitude`)**
- **Found during:** Task 1 (`test_outside_mask_perturbation_yields_bit_identical_features`)
- **Issue:** OpenCV 4.14.0's full-frame `cv2.magnitude(gx, gy)` exhibited ~`1.8e-8` (`1 ULP` float32) non-determinism across repeated identical calls on bit-identical `l_masked` arrays, causing strict `f_a["feather_grad_mean"] == f_b["feather_grad_mean"]` equality to fail (`136.64865582346775 != 136.64865580536716`).
- **Fix:** Replaced `cv2.magnitude(gx, gy)` with strictly masked float64 `np.hypot(gx[interior].astype(np.float64), gy[interior].astype(np.float64))` in `backend/app/pdi/features.py`, also avoiding redundant magnitude computation on outside-mask pixels.
- **Files modified:** `backend/app/pdi/features.py`
- **Verification:** Verified `1` unique value across 20 repeated calls, 5/5 consecutive runs of `uv run pytest backend/tests/test_pdi_features_invariance.py -q`, and exact `0.0` diff across all 49 features.
- **Committed in:** `6296f5e` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug)
**Impact on plan:** Critical correctness fix guaranteeing strict bit-identical `0.0` diff under `PROIBIÇÃO 2` perturbation testing; zero scope creep.

## Issues Encountered

None beyond the OpenCV 4.14.0 `cv2.magnitude` SIMD non-determinism diagnosed and resolved during Task 1 TDD above.

## User Setup Required

None - zero new dependencies installed (`pyproject.toml` and `uv.lock` untouched).

## Next Phase Readiness

- Ready for Wave 3 (`02-03`: isolated Marker-based Watershed `backend/app/pdi/watershed.py` reusing `extract_mask_gated_features()` per separated instance + `backend/app/pdi/__init__.py` public barrel) and Wave 4 (`02-04`: `backend/scripts/extract_features.py` batch runner invoking `run_single_bird_pdi(..., return_xray=False)` across all `1,738` clean images + `models/pdi_calibration_audit.json`).
- All 76 backend tests (`39` Phase 1 + `24` Plan `02-01` + `13` Plan `02-02`) and `ruff check backend/` pass with zero warnings.

## Self-Check: PASSED

- Verified all 5 created files exist on disk (`backend/app/pdi/features.py`, `backend/app/pdi/xray.py`, `backend/app/pdi/pipeline.py`, `backend/tests/test_pdi_features_invariance.py`, `backend/tests/test_pdi_pipeline_and_xray.py`).
- Verified both atomic task commits (`6296f5e`, `898cf74`) in `git log` after `plan_head_before = 5ec98d5e9bbc1fcec5f746899dfc35192b96c14e`.
- Verified `uv run pytest backend/tests/test_pdi_pipeline_and_xray.py backend/tests/test_pdi_features_invariance.py backend/tests/test_io_exif.py -v` (`30 passed in 3.35s`), full suite `uv run pytest -v` (`76 passed in 4.63s`), and `uv run ruff check backend/` (`All checks passed!`).
- Verified `PROIBIÇÃO 1-4` compliance: zero DL/OCR imports, zero `cv2.imread`/`cv2.imdecode` calls under `backend/app/` (AST-checked), exact `0.0` outside-mask perturbation diff across all 49 features, zero `pyproject.toml`/`uv.lock` edits, zero `data/raw/` mutations, zero raw filenames logged.

---
*Phase: 02-classical-pdi-engine-optical-calibration-watershed*
*Completed: 2026-11-22*
