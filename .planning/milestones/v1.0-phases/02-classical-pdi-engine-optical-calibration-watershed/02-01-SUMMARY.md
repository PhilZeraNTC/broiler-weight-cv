---
phase: 02-classical-pdi-engine-optical-calibration-watershed
plan: 01
subsystem: pdi-engine
tags: [opencv, pillow, scipy, exif, clahe, bilateral, lcd-suppression, segmentation, classical-cv]

# Dependency graph
requires:
  - phase: 01-forensic-data-curation-anti-leakage-split
    provides: Shared Pillow EXIF-first loader (backend/app/pdi/io_exif.py), repo-root path config (backend/app/config.py), AST prohibition gate, and materialized clean/quarantine manifests
provides:
  - Extended repo-root path configuration exporting FEATURES_CLEAN_PATH and PDI_CALIBRATION_AUDIT_PATH (backend/app/config.py)
  - Immutable frozen slotted PDIConfig registry with ASVS V5 validation and deterministic 16-hex SHA-256 config_hash() using dataclasses.asdict (backend/app/pdi/config.py)
  - Acyclic EXIF-first working-resolution image loader with aspect-preserving PIL.Image.draft(), local classify_lens(), bounded OpticalOverride/OpticalContext resolution, and bit-exact Path-vs-bytes parity (backend/app/pdi/io_exif.py)
  - Classical OpenCV edge-preserving Bilateral/Median smoother, CIE Lab L-channel CLAHE, HSV, and float32 Scharr gradient maps (backend/app/pdi/preprocess.py)
  - Dedicated rotation-invariant SF-400 dark LCD bezel and connected smooth plastic housing suppressor with zero hardcoded strip coordinates (backend/app/pdi/lcd_suppression.py)
  - 3-scenario classical single-bird segmenter (bimodal border-ring classifier, Scenario B smooth-wall peeling + Black-Hat shadow barrier, Scenarios A/C bird-vs-straw with pre-closing LCD suppression and post-hole-fill re-zeroing, orientation-preserving fit_ellipse_safe, deterministic component tie-breaking, and [0.015, 0.72] adaptive fallback) (backend/app/pdi/segmentation.py)
  - 24-test synthetic unit and edge-probe suite covering PDIConfig, acyclic io_exif, preprocessing, rotation-invariant LCD exclusion, 3-scenario segmentation, ellipse orientation alignment, empty-frame fallback, and deterministic tie-breaking (backend/tests/test_pdi_preprocess_and_lcd.py)
affects: [02-02, 02-03, 02-04, phase-3-modeling, phase-4-web-mvp]

# Plan ledger
plan_head_before: 9b4b312db9974e690e5bbf7db151e07f874fa9fc
commits:
  - 8d2a55b
  - 98ffcdd
  - ef5ca5a

# Actuals
actuals:
  tokens: 18793
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - Frozen slotted PDIConfig hashed via dataclasses.asdict() -> deterministic 16-hex SHA-256
    - Acyclic backend.app.pdi.io_exif with local WIDE_FOCAL_THRESHOLD_MM/classify_lens and aspect-preserving PIL.Image.draft() strictly after header EXIF read
    - Rotation-invariant SF-400 LCD bezel + connected smooth housing exclusion executed strictly before morphological closing and re-zeroed after binary_fill_holes
    - Orientation-preserving ellipse axis normalization (major >= minor, 0 <= angle_deg < 180 with +90 deg wrap on axis swap) across both cv2.fitEllipse and cv2.minAreaRect fallback
    - Strict OpenCV THRESH_OTSU greater-than boundary semantics (l_f > otsu_int) inside Scenario B interior plumage fusion

key-files:
  created:
    - backend/app/pdi/config.py
    - backend/app/pdi/preprocess.py
    - backend/app/pdi/lcd_suppression.py
    - backend/app/pdi/segmentation.py
    - backend/tests/test_pdi_preprocess_and_lcd.py
  modified:
    - backend/app/config.py
    - backend/app/pdi/io_exif.py

key-decisions:
  - "Defined WIDE_FOCAL_THRESHOLD_MM = 3.0 and classify_lens() locally inside backend/app/pdi/io_exif.py (matching backend/app/ingestion/decoding.py verbatim) so backend/app/pdi/ remains strictly acyclic regardless of import order."
  - "Enforced OpenCV cv2.THRESH_OTSU strict greater-than semantics (l_f > otsu_int & l_f >= l_p50) inside Scenario B bucket interior cue_white so constant-luminance shaded bucket floors are never misclassified as foreground."
  - "Rounded component centrality score to 6 decimal places using float64 mean reduction before sorting by (-score, -area, round(cy, 4), round(cx, 4)) so equal-centrality symmetric candidates break ties deterministically."

patterns-established:
  - "Pattern 1 (Pre-draft EXIF + aspect-preserving draft): Always call _extract_exif_from_open_image(im) before im.draft('RGB', (int(w*sc), int(h*sc))) so native pre-draft dimensions stay intact while triggering 1/4 libjpeg DCT shrink-on-load."
  - "Pattern 2 (PROIBIÇÃO 2 anti-LCD sandwich): Always subtract lcd_res.exclusion_mask before MORPH_OPEN/MORPH_CLOSE AND re-zero after binary_fill_holes so closing/hole-filling can never bridge back onto SF-400 LCD pixels."
  - "Pattern 3 (Canonical ellipse contract): fit_ellipse_safe() always returns ((cx, cy), (major, minor), angle_deg) with major >= minor, 0.0 <= angle_deg < 180.0, and major aligned along angle_deg for cv2.ellipse() and downstream axis overlays."

requirements-completed:
  - PDI-01
  - PDI-02

coverage:
  - id: D1
    description: "Frozen slotted PDIConfig registry with ASVS V5 bounds validation, deterministic 16-char SHA-256 config_hash(), and extended backend/app/config.py path constants"
    requirement: PDI-01
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_pdi_config_hash_determinism_immutability_and_validation"
        status: pass
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_config_paths_are_repo_root_anchored_and_cwd_independent"
        status: pass
    human_judgment: false
  - id: D2
    description: "Acyclic Pillow EXIF-first working-resolution loader (load_working_rgb) with aspect-preserving Image.draft(), Path-vs-bytes parity across orientations 1/3/6/8, EXIF-stripped/RGBA/grayscale degradation, and ASVS V5 OpticalOverride bounds"
    requirement: PDI-01
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_io_exif_is_acyclic_and_matches_ingestion_classify_lens"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_load_working_rgb_preserves_pre_draft_dims_and_path_bytes_parity"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_load_working_rgb_and_optical_context_graceful_degradation_and_bounds"
        status: pass
      - kind: unit
        ref: "backend/tests/test_io_exif.py#test_ast_scan_confirms_zero_raw_opencv_image_loading_under_backend_app"
        status: pass
    human_judgment: false
  - id: D3
    description: "Classical OpenCV edge-preserving Bilateral/Median smoother, CIE Lab L-channel CLAHE, HSV, and float32 Scharr gradient maps (preprocess_working_rgb)"
    requirement: PDI-01
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_preprocess_working_rgb_bilateral_and_median_contracts"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_preprocess_working_rgb_rejects_invalid_inputs"
        status: pass
    human_judgment: false
  - id: D4
    description: "Dedicated rotation-invariant SF-400 dark LCD bezel and connected smooth white plastic housing suppressor (suppress_lcd_and_platter) with zero hardcoded strip rules and zero bucket false positives"
    requirement: PDI-02
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_suppress_lcd_and_platter_is_rotation_invariant_and_spares_bird"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_suppress_lcd_and_platter_has_zero_false_positives_on_clean_white_bucket"
        status: pass
    human_judgment: false
  - id: D5
    description: "3-scenario classical single-bird segmenter (classify_scene_border_ring, Scenario B white-on-white bucket smooth-wall peeling + Black-Hat shadow barrier, Scenarios A/C bird-vs-straw with touching-platter zero-LCD-leakage, orientation-preserving fit_ellipse_safe, empty-frame fallback, and deterministic symmetric tie-breaking)"
    requirement: PDI-02
    verification:
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_classify_scene_border_ring_separates_bucket_from_straw"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_segment_single_bird_scenario_b_white_on_white_bucket_peels_walls"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_segment_single_bird_scenario_a_touching_platter_has_zero_lcd_leakage"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_fit_ellipse_safe_preserves_major_axis_orientation_and_iou"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_fit_ellipse_safe_degenerate_contours_and_min_area_rect_alignment"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_segment_single_bird_empty_uniform_frame_triggers_safe_fallback"
        status: pass
      - kind: unit
        ref: "backend/tests/test_pdi_preprocess_and_lcd.py#test_segment_single_bird_deterministic_ordering_for_symmetric_candidates"
        status: pass
    human_judgment: false

duration: 11min
completed: 2026-11-22
status: complete
---

# Phase 02 Plan 01: Core Classical PDI Config, Working EXIF Loader, Preprocessing, Anti-LCD Suppression & 3-Scenario Segmentation Summary

**Locked frozen `PDIConfig` (`768 px` working scale, SHA-256 fingerprint), acyclic aspect-preserving Pillow EXIF working loader, Bilateral/Lab-CLAHE preprocessor, rotation-invariant SF-400 LCD/housing suppressor, and 3-scenario classical single-bird segmenter with orientation-preserving ellipse fitting verified by 24 new synthetic/edge-probe tests and all 39 Phase 1 regression tests**

## Performance

- **Duration:** 11 min
- **Started:** 2026-11-22T14:20:05Z
- **Completed:** 2026-11-22T14:30:39Z
- **Tasks:** 3 (1 tracer + 2 TDD)
- **Files modified:** 5 created, 2 modified (`+1,868 / -6` lines)

## Accomplishments

- Extended `backend/app/config.py` with `FEATURES_CLEAN_PATH` and `PDI_CALIBRATION_AUDIT_PATH`, and created frozen slotted `PDIConfig` (`backend/app/pdi/config.py`) validating all structural/ASVS V5 bounds and hashing via `dataclasses.asdict(self)` into a deterministic 16-hex SHA-256 `config_hash()`.
- Extended `backend/app/pdi/io_exif.py` while keeping it 100% backward-compatible and strictly acyclic (zero `backend.app.ingestion` imports; local `WIDE_FOCAL_THRESHOLD_MM = 3.0` and `classify_lens()` matching `decoding.py` verbatim), adding pre-draft EXIF preservation with aspect-preserving `PIL.Image.draft("RGB", draft_box)`, `cv2.INTER_AREA` downscale, bit-identical `Path`-vs-`bytes` parity, graceful RGBA/grayscale/stripped-EXIF degradation, and ASVS V5 bounded `OpticalOverride` / `resolve_optical_context()`.
- Built `backend/app/pdi/preprocess.py` implementing parameterized edge-preserving Bilateral/Median smoothing, CIE Lab `L`-channel CLAHE, OpenCV `uint8` HSV, and `float32` Scharr gradient magnitude maps.
- Built dedicated `backend/app/pdi/lcd_suppression.py` (`PROIBIÇÃO 2` / `D-05`) suppressing dark neutral-chroma rectangular SF-400 LCD bezels via rotation-invariant `cv2.minAreaRect` geometry and connected smooth cool-white plastic housing/platter with zero hardcoded frame strips and zero white-bucket false positives.
- Built `backend/app/pdi/segmentation.py` (`D-06`, `D-07`) with bimodal outer `13%` border-ring scene classification, Scenario B border-connected smooth-wall peeling gated by a `17×17` Black-Hat contact-shadow barrier, Scenarios A/C bird-vs-straw fusion with pre-closing LCD suppression + `MORPH_OPEN(9×9)` bridge severing + post-hole-fill LCD re-zeroing, orientation-preserving `fit_ellipse_safe()` (`+90° mod 180°` swap alignment verified at `IoU >= 0.988`), deterministic `(-score, -area, cy, cx)` component tie-breaking, and deterministic `[0.015, 0.72]` adaptive fallback.
- Verified aggregate real-dataset sanity across a stratified 60-image read-only sample (`<500g`, `500-1000g`, `>1000g`, zero raw filenames logged): `0/60` fallbacks (`0.0%`), `21` bucket classifications, `24` LCD suppressions, `0` bucket LCD false positives, `p50 = 77.1 ms`, `p95 = 94.6 ms`.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end frozen PDIConfig, aspect-preserving EXIF-first working loader, and classical OpenCV preprocessing tracer slice** - `8d2a55b` (feat)
2. **Task 2: Build dedicated rotation-invariant SF-400 LCD & platter suppressor (`lcd_suppression.py`) and core preprocessing/LCD test suite** - `98ffcdd` (feat)
3. **Task 3: Build 3-scenario classical single-bird segmenter (`segmentation.py`) with Scenario B smooth-wall peeling, guarded ellipse fitting, and edge-probe tests** - `ef5ca5a` (feat)

## Files Created/Modified

- `backend/app/config.py` - Appended `FEATURES_CLEAN_PATH` and `PDI_CALIBRATION_AUDIT_PATH` to repo-root path constants and `__all__`.
- `backend/app/pdi/config.py` - Frozen slotted `PDIConfig` with ASVS V5 parameter validation and deterministic 16-char SHA-256 `config_hash()`.
- `backend/app/pdi/io_exif.py` - Added acyclic local `classify_lens()`, `OpticalOverride`, `OpticalContext`, `WorkingImage`, `resolve_optical_context()`, and aspect-preserving `load_working_rgb()` preserving full Phase 1 compatibility.
- `backend/app/pdi/preprocess.py` - `PreprocessedFrame` and `preprocess_working_rgb()` implementing Bilateral/Median smoothing, Lab `L`-channel CLAHE, HSV, and Scharr gradient maps.
- `backend/app/pdi/lcd_suppression.py` - `LcdSuppressionResult` and rotation-invariant `suppress_lcd_and_platter()` for SF-400 LCD bezels and connected smooth housing.
- `backend/app/pdi/segmentation.py` - `SingleBirdSegmentation`, `classify_scene_border_ring()`, `fit_ellipse_safe()`, `_select_dominant_component()`, Scenario B smooth-wall peeling, Scenarios A/C anti-LCD bird-vs-straw segmentation, and `[0.015, 0.72]` adaptive fallback.
- `backend/tests/test_pdi_preprocess_and_lcd.py` - 24 synthetic unit and edge-probe tests covering all `PDI-01` and `PDI-02` contracts.

## Decisions Made

- Kept `backend/app/pdi/io_exif.py` strictly independent of `backend.app.ingestion` by defining `WIDE_FOCAL_THRESHOLD_MM = 3.0` and `classify_lens()` locally (enforced via an AST test and parity check against `backend.app.ingestion.decoding.classify_lens`).
- Used `l_f > otsu_int` alongside `l_f >= l_p50` inside Scenario B `cue_white` to honor OpenCV's exact `THRESH_BINARY + THRESH_OTSU` strict-greater-than foreground partition.
- Computed component centrality means in `float64` rounded to 6 decimals inside `_select_dominant_component()` so symmetric equal-centrality blobs fall through cleanly to deterministic `(-area, round(cy, 4), round(cx, 4))` tie-breaking.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Enforced strict greater-than OpenCV Otsu boundary in Scenario B interior cue**
- **Found during:** Task 3 (`segment_single_bird` TDD verification on `test_segment_single_bird_scenario_b_white_on_white_bucket_peels_walls`)
- **Issue:** OpenCV `cv2.threshold(..., THRESH_BINARY + THRESH_OTSU)` returns the inclusive upper bound of the lower/background histogram mode (`src > thr` is foreground). Using `l_f >= max(otsu_int, l_p50)` included constant-intensity shaded bucket floor pixels equal to `otsu_int` (`L == 181`), collapsing synthetic white-on-white bucket IoU to `0.344`.
- **Fix:** Updated `cue_white` in `_segment_bucket_dorsal` (`backend/app/pdi/segmentation.py`) to `(l_f > otsu_int) & (l_f >= l_p50) & (frame.grad_l_clahe_blur <= 260.0) & (a_f <= 133.5)`.
- **Files modified:** `backend/app/pdi/segmentation.py`
- **Verification:** `test_segment_single_bird_scenario_b_white_on_white_bucket_peels_walls` passed cleanly (`wall_bleed == 0`, `IoU >= 0.70`), and a 60-image stratified read-only real dataset sanity sweep recorded `0/60` fallbacks (`21` bucket scenes, `0` bucket LCD false positives).
- **Committed in:** `ef5ca5a` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug)
**Impact on plan:** Essential correctness fix matching OpenCV Otsu binary partition semantics; zero scope creep.

## Issues Encountered

- Minor Ruff `E501` line-length (`> 100` chars) formatting issues flagged during Task 1, Task 2, and Task 3 verifications were immediately wrapped cleanly before each atomic commit (`uv run ruff check backend/` exited `0` on every commit).

## User Setup Required

None - zero new dependencies installed (`pyproject.toml` and `uv.lock` untouched).

## Next Phase Readiness

- Ready for Wave 2 (`02-02` mask-gated features, 4-stage X-Ray renderer, and unified `run_single_bird_pdi()` pipeline): `PDIConfig`, `load_working_rgb()`, `resolve_optical_context()`, `preprocess_working_rgb()`, `suppress_lcd_and_platter()`, `segment_single_bird()`, and orientation-normalized `fit_ellipse_safe()` are verified and exported cleanly.
- All 63 backend tests (`39` Phase 1 + `24` Plan `02-01`) and `ruff check backend/` pass with zero warnings.

## Self-Check: PASSED

- Verified all 7 created/modified files exist on disk (`backend/app/config.py`, `backend/app/pdi/config.py`, `backend/app/pdi/io_exif.py`, `backend/app/pdi/preprocess.py`, `backend/app/pdi/lcd_suppression.py`, `backend/app/pdi/segmentation.py`, `backend/tests/test_pdi_preprocess_and_lcd.py`).
- Verified all 3 atomic task commits (`8d2a55b`, `98ffcdd`, `ef5ca5a`) in `git log` after `plan_head_before = 9b4b312db9974e690e5bbf7db151e07f874fa9fc`.
- Verified `uv run pytest backend/tests/test_pdi_preprocess_and_lcd.py backend/tests/test_io_exif.py -v` (`41 passed in 2.35s`), full suite `uv run pytest -v` (`63 passed in 3.27s`), and `uv run ruff check backend/` (`All checks passed!`).
- Verified `PROIBIÇÃO 1-4` compliance: zero DL/OCR imports, zero `cv2.imread`/`cv2.imdecode` calls under `backend/app/` (AST-checked), zero `pyproject.toml`/`uv.lock` edits, zero `data/raw/` mutations, zero raw filenames logged.

---
*Phase: 02-classical-pdi-engine-optical-calibration-watershed*
*Completed: 2026-11-22*
