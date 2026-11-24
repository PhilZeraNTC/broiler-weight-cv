---
phase: 02-classical-pdi-engine-optical-calibration-watershed
verified: 2026-11-23T19:21:00Z
status: passed
score: "16/16"
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-01-PLAN.md"
  - ".planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-01-SUMMARY.md"
  - ".planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-02-PLAN.md"
  - ".planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-02-SUMMARY.md"
  - ".planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-03-PLAN.md"
  - ".planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-03-SUMMARY.md"
  - ".planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-04-PLAN.md"
  - ".planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-04-SUMMARY.md"
  - "backend/app/config.py"
  - "backend/app/pdi/__init__.py"
  - "backend/app/pdi/config.py"
  - "backend/app/pdi/features.py"
  - "backend/app/pdi/io_exif.py"
  - "backend/app/pdi/lcd_suppression.py"
  - "backend/app/pdi/pipeline.py"
  - "backend/app/pdi/preprocess.py"
  - "backend/app/pdi/segmentation.py"
  - "backend/app/pdi/watershed.py"
  - "backend/app/pdi/xray.py"
  - "backend/scripts/extract_features.py"
  - "backend/tests/test_extract_features_integration.py"
  - "backend/tests/test_pdi_features_invariance.py"
  - "backend/tests/test_pdi_pipeline_and_xray.py"
  - "backend/tests/test_pdi_preprocess_and_lcd.py"
  - "backend/tests/test_pdi_watershed.py"
  - "models/pdi_calibration_audit.json"
covered_digest: "v1:sha256:ac781d3a0b125f5e60e8c063d93938d4cd9914b745a8a5c414b36f6a5ec65483"
behavior_unverified: 0
overrides_applied: 0
overrides: []
gaps: []
---

# Phase 2: Classical PDI Engine, Optical Calibration & Watershed — Verification Report

**Phase Goal:** Deliver the single-source classical Digital Image Processing engine (zero deep learning) that cleanly isolates bird silhouettes without SF-400 LCD leakage, extracts strictly mask-gated morphometric, chromatic, and empirically verified EXIF optical features, renders 4-stage "X-Ray" visuals, and separates touching birds via Marker-based Watershed.
**Verified:** 2026-11-23T19:21:00Z (refreshed after post-review hardening commit `bc2217c`; milestone closeout fingerprint refreshed 2026-11-24 at `16eb74e`)
**Status:** `passed`
**Re-verification:** Yes — fingerprint & full suite refreshed after resolving `02-REVIEW.md` `CR-01`/`CR-02` and `WR-02`–`WR-07` in `bc2217c` and again at v1.0 closeout (`16eb74e`) after shared `REQUIREMENTS.md`/`config.py` evolved in Phases 3–4 (`272/272` pytest incl. all Phase 2 tests, `ruff check backend/` clean, `pdi_config_hash = 43a3cfaa3208019e` intact)

## Executive Summary

Adversarial goal-backward verification against the actual Phase 2 codebase (`backend/app/pdi/`, `backend/scripts/extract_features.py`, `backend/app/config.py`), materialized `data/processed/features_clean.parquet`, git-tracked `models/pdi_calibration_audit.json`, dependency lockfile, git index, and live test/lint/runtime audits confirms **16/16 merged observable truths VERIFIED**, **18/18 PLAN artifacts VERIFIED**, **12/12 PLAN key links WIRED**, **0 gaps**, **0 stub artifacts**, **0 debt markers**, and **130/130 automated tests passing** (`uv run pytest -v`, `11.57s`: `39` Phase 1 regression + `91` Phase 2 tests; `uv run ruff check backend/` → `All checks passed!`).

Independent read-only live audits verified:
- **Zero code-vs-Parquet skew:** Re-running `run_single_bird_pdi()` over 8 stratified real clean rows (spanning bucket + scale/litter, all 3 weight bands, and fallback rows) reproduced `data/processed/features_clean.parquet` with **exact `0.0` max feature difference** across all 49 columns (`pdi_config_hash = "43a3cfaa3208019e"`), confirming commit `bcb6b9e`'s performance optimizations (`_fill_holes_u8`, `_area_convex_exact`, tight-bbox Lab/HSV/texture) are 100% bit-identical (further verified `10/10` exact against `scipy.ndimage.binary_fill_holes` and `skimage.measure.regionprops.area_convex`).
- **Segmentation & Anti-LCD Gate (`D-05`–`D-07`):** `4 / 1,738 = 0.2301%` primary fallback (`0/377` on dorsal bucket including `0/59` on `>1,000 g` white-on-white broilers; `786` SF-400 LCD suppressions with `0` bucket false positives).
- **Online Latency & X-Ray SLA (`D-04`):** Across 8 real native `4080×1840` Wide JPEGs covering all 4 EXIF orientations (`1, 3, 6, 8`), `run_single_bird_pdi(raw_bytes, return_xray=True)` achieved **`p50 = 74.4 ms`**, **`p95 = 88.3 ms`**, **`max = 89.5 ms`** (`< 400 ms` target / `< 500 ms` SLA) and 4-stage Base64 payloads **`p50 = 140.1 KB`**, **`max = 166.8 KB`** (`< 250 KB`).
- **Isolated Marker-Based Watershed (`D-11`, `D-12`):** Across all `13` quarantined multi-bird scenes (`417`, `757`, `823`, `1520`) with zero folder hardcoding, `run_watershed_pdi()` achieved **`9/13` exact counts**, **`12/13` within `±1`** (`MAE = 0.38` birds, `417`: `4/4` exact with `>= 15,725 px` red-feeder suppression), **`p50 = 131.8 ms`** / **`max = 169.9 ms`** (`< 350 ms`), and **`max = 145.1 KB`** visuals (`< 300 KB`).

---

## Goal Achievement

### User Flow Coverage (MVP Mode — Plan User Stories `02-01`..`02-04`)

| Plan | User Story Outcome | Codebase Evidence | Status |
| --- | --- | --- | --- |
| `02-01` | Every broiler image (`Path` or uploaded `bytes`) yields a clean, LCD-free binary silhouette and fitted ellipse across scale, white-on-white bucket, and straw-litter scenes well within `< 400 ms` CPU. | `PDIConfig`, acyclic `load_working_rgb`/`resolve_optical_context`, `preprocess_working_rgb`, `suppress_lcd_and_platter`, `segment_single_bird`, `fit_ellipse_safe`; `24/24` passing tests in `test_pdi_preprocess_and_lcd.py` + `4/1,738` (`0.2301%`) real-corpus fallback (`0/59` `>1,000 g` bucket). | ✓ VERIFIED |
| `02-02` | Offline training and live web inference execute identical classical code extracting 49 mask-gated features (`0.0` outside-mask diff) + on-demand 4-stage X-Ray visuals at `< 400 ms` p95 latency. | `extract_mask_gated_features` (`49` cols, `0` banned), `render_xray_artifacts` (`640 px`, `< 250 KB`), `run_single_bird_pdi`; `13/13` passing tests in `test_pdi_features_invariance.py` + `test_pdi_pipeline_and_xray.py`; live native `4080×1840` `p95 = 88.3 ms` / `max = 166.8 KB`. | ✓ VERIFIED |
| `02-03` | Touching broilers in multi-bird scenes (`417`/`757`/`823`/`1520` and arbitrary uploads) separate via isolated Marker-based Watershed emitting 4 visuals, counts, areas, and per-instance 49-key features with zero folder hardcoding. | `run_watershed_pdi` + sorted `backend/app/pdi/__init__.py` (`34` exports); `37/37` passing tests in `test_pdi_watershed.py`; live 13-image audit `9/13` exact, `12/13` within `±1`, `max = 169.9 ms`, `0` folder tokens in `watershed.py`. | ✓ VERIFIED |
| `02-04` | Capped 4-worker offline batch extractor materializes audited, zero-leakage `1,738 × 60` `features_clean.parquet` (`< 60 s`) and byte-deterministic `models/pdi_calibration_audit.json` proving `area_px / (zoom_multiple ** 2)` without double-counting. | `backend/scripts/extract_features.py` (`MAX_WORKERS_CAP = 4`, ASVS V4 containment), `data/processed/features_clean.parquet` (`1,738 × 60`, `0` NaN/Inf, `0.2301%` fallback), `models/pdi_calibration_audit.json` (exact rebuild match, `0` forbidden tokens); `17/17` passing tests in `test_extract_features_integration.py`. | ✓ VERIFIED |

### Observable Truths (ROADMAP Success Criteria + Merged PLAN Must-Haves)

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | **ROADMAP SC-1 / 02-01 T1–T3 (`PDI-01`):** Frozen slotted `PDIConfig` (`working_long_edge = 768`, `draft_scale_multiplier = 1.0`, `bilateral_d = 5`, `clahe_clip_limit = 2.0`, `clahe_tile_grid = (8, 8)`) computes deterministic 16-hex SHA-256 `config_hash()` via `dataclasses.asdict(self)`; acyclic `io_exif.py` (zero `backend.app.ingestion` imports; local `WIDE_FOCAL_THRESHOLD_MM = 3.0` & `classify_lens`) reads pre-draft EXIF before aspect-preserving `im.draft("RGB", draft_box)`, applies `ImageOps.exif_transpose()` + `cv2.INTER_AREA`, guarantees bit-exact `Path`-vs-`bytes` parity, degrades stripped-EXIF/RGBA/grayscale inputs to `4.25 mm / 1.0x`, enforces ASVS V5 `OpticalOverride` bounds, and drives `preprocess_working_rgb()` (Bilateral/Median + Lab `L`-CLAHE + HSV + Scharr) with zero `cv2.imread`/`cv2.imdecode` repo-wide. | ✓ VERIFIED | Direct read of `backend/app/pdi/{config,io_exif,preprocess}.py`; fresh-process import of `backend.app.pdi.io_exif` before `backend.app.ingestion` succeeded with 100% `classify_lens` parity; AST scan across all 28 Python files in `backend/` found `0` `cv2.imread`/`cv2.imdecode` calls and `0` `backend.app.ingestion` imports in `backend/app/pdi/`. Covered by `test_pdi_config_hash_determinism_immutability_and_validation`, `test_io_exif_is_acyclic_and_matches_ingestion_classify_lens`, `test_load_working_rgb_preserves_pre_draft_dims_and_path_bytes_parity`, `test_load_working_rgb_and_optical_context_graceful_degradation_and_bounds`, `test_preprocess_working_rgb_bilateral_and_median_contracts`, and `test_ast_scan_confirms_zero_raw_opencv_image_loading_under_backend_app` (all PASS). |
| 2 | **ROADMAP SC-2 / 02-01 T4–T5 (`PDI-02`):** Dedicated rotation-invariant `suppress_lcd_and_platter()` detects dark neutral SF-400 LCD bezels via `cv2.minAreaRect` (zero hardcoded top/bottom strips) and flood-suppresses connected smooth plastic housing strictly BEFORE morphological closing/selection (re-zeroed after hole fill); `segment_single_bird()` classifies `bucket_dorsal` vs `scale_or_litter` via outer `13%` border-ring smoothness, peels border-connected smooth white walls gated by a Black-Hat contact-shadow barrier on Scenario B (including white-on-white `>1,000 g` broilers), and achieves `< 2%` fallback across all `1,738` clean images. | ✓ VERIFIED | Direct read of `backend/app/pdi/{lcd_suppression,segmentation}.py`; real Parquet audit confirmed `377` `bucket_dorsal` (`0/377` fallbacks including `0/59` `>1,000 g` white-on-white broilers, `0` bucket LCD FPs) + `1,361` `scale_or_litter` (`4/1,361` fallbacks, `786` LCD suppressions) = **`4 / 1,738 = 0.2301%`** primary fallback (`< 2.0%` gate). Covered by `test_suppress_lcd_and_platter_is_rotation_invariant_and_spares_bird` (`0°, 90°, 180°, 270°, 35°`), `test_suppress_lcd_and_platter_has_zero_false_positives_on_clean_white_bucket`, `test_classify_scene_border_ring_separates_bucket_from_straw`, `test_segment_single_bird_scenario_b_white_on_white_bucket_peels_walls`, and `test_real_features_clean_parquet_schema_alignment_finite_values_and_fallback_gate` (all PASS). |
| 3 | **02-01 T6 (`fit_ellipse_safe` Orientation-Preserving Axes):** `fit_ellipse_safe()` guarantees `major >= minor` and `0.0 <= angle_deg < 180.0` with `major` strictly aligned along `angle_deg` (`+90.0 % 180.0` wrap whenever swapping axes across both `cv2.fitEllipse` and `cv2.minAreaRect`), preserving exact geometry for `cv2.ellipse()` and Stage 4 X-Ray axis overlays. | ✓ VERIFIED | Direct read of `_normalize_ellipse_axes` and `fit_ellipse_safe` in `backend/app/pdi/segmentation.py` lines 132–187. Verified by `test_fit_ellipse_safe_preserves_major_axis_orientation_and_iou` (`30°` and `150°`, angle err `< 5°`, `cv2.ellipse` IoU `>= 0.90`), `test_fit_ellipse_safe_degenerate_contours_and_min_area_rect_alignment` (3-pt, 4-pt tilted `30°`, collinear 10-pt, tiny `<10 px²`), and `test_render_xray_artifacts_decodes_all_four_stages_within_budget_and_axis_alignment` (amber/cyan pixel alignment probes) (all PASS). |
| 4 | **02-01 T7–T9 (`PDI-02` Edge Probes — Adjacency, Empty, Ordering):** Touching bird + SF-400 platter/LCD components sever cleanly via pre-closing LCD zeroing + elliptical `MORPH_OPEN(9×9)` and post-hole-fill re-zeroing (`adjacency`); uniform/empty frames & degenerate contours trigger deterministic adaptive fallback returning `segmentation_fallback=True`, `mask_area_frac > 0`, and valid finite ellipse (`empty`); equal-centrality symmetric components break ties deterministically via `(-round(score, 6), -area, round(cy, 4), round(cx, 4))` (`ordering`). | ✓ VERIFIED | Direct read of `backend/app/pdi/segmentation.py` lines 239–280, 420–569. Verified by `test_segment_single_bird_scenario_a_touching_platter_has_zero_lcd_leakage` (`0` overlap with `lcd_gt` even with white leg/platter bridge), `test_segment_single_bird_empty_uniform_frame_triggers_safe_fallback`, `test_segment_single_bird_deterministic_ordering_for_symmetric_candidates`, and `test_full_pipeline_sf400_anti_lcd_excludes_digits_completely` (all PASS). |
| 5 | **ROADMAP SC-3 / 02-02 T1–T2 (`PDI-03` 49-Feature Schema & `PROIBIÇÃO 2` Bit-Exact Invariance):** `features.py` defines immutable 49-column `FEATURE_COLUMNS` (`14` morphometrics + `26` pixel-local chromatic & 3-px eroded interior texture + `9` optical descriptors), bans `oriented_w`/`oriented_h`/`megapixels`/`bbox_w`/`bbox_h`/`raw_w`/`raw_h`, converts strictly pixel-local `bgr_work` (never `bgr_clahe`/`bgr_smooth`), gates circular Hue to `S >= 25` (`< 10` px → `0.0`), evaluates `Scharr`/`Laplacian` strictly inside `7×7` eroded interior (`BORDER_CONSTANT=0`, `4`-px padded tight crop), and achieves exact `0.0` diff across all 49 features under arbitrary outside-mask pixel perturbations + 90°/180° rotation invariance. | ✓ VERIFIED | Direct read of `backend/app/pdi/features.py` (482 lines) and `pipeline.py` lines 65–74 (passing `frame.bgr_work`). Verified by `test_feature_schema_has_exact_49_unique_columns_and_excludes_banned_columns`, `test_outside_mask_perturbation_yields_bit_identical_features` (both centered and border-touching masks painted with fake `"1850"` LCD digits + neon magenta/green checkerboard → exact `f_a[k] == f_b[k]` for all 49 features), `test_rotation_invariance_90_and_180_degrees`, and `test_achromatic_hue_gating_tiny_mask_and_empty_mask_guards` (all PASS). |
| 6 | **ROADMAP SC-3 / 02-02 T3 (`D-09` Empirical Optical Calibration Formula):** Optical normalization implements Spike 2 winner `area_optical_norm = area_px / (zoom_multiple ** 2)` (with `sqrt_area_optical_norm`, `perim_optical_norm`, `major_axis_optical_norm`, `minor_axis_optical_norm`) alongside resize-agnostic `area_frac_optical_norm = area_frac / (zoom_multiple ** 2)`, permanently avoiding `focal_length_mm * zoom_multiple` double-counting. | ✓ VERIFIED | Direct read of `backend/app/pdi/features.py` lines 409–416; independent audit over all `1,738` rows in `features_clean.parquet` confirmed `max |area_optical_norm - area_px / zoom_multiple**2| = 0.00e+00`. Verified by `test_optical_calibration_formulas_and_no_focal_zoom_double_counting` (confirming changing `focal_length_mm` alone at constant `zoom_multiple` does not alter optical norms) (PASS). |
| 7 | **ROADMAP SC-4 / 02-02 T4 (`PDI-04` 4-Stage X-Ray Artifacts):** `render_xray_artifacts()` renders all 4 inspection stages downscaled aspect-preserving to `<= 640 px` (Stages 1/2/4 `JPEG Q=82`, Stage 3 1-channel `"L"` `PNG` without LCD, Stage 4 contour + fitted ellipse + orthogonal amber/cyan axes) as Pillow-decodable raw RFC 4648 Base64 with combined payload `< 250 KB`. | ✓ VERIFIED | Direct read of `backend/app/pdi/xray.py` (192 lines). Live benchmark across 8 real native `4080×1840` Wide JPEGs measured `p50 = 140.1 KB`, `max = 166.8 KB` (`< 250 KB`). Verified by `test_render_xray_artifacts_decodes_all_four_stages_within_budget_and_axis_alignment` (verifying Pillow format/mode/dims/axes/no-upscale) (PASS). |
| 8 | **ROADMAP SC-4 / 02-02 T5 (`run_single_bird_pdi` & `< 500 ms` CPU SLA):** `run_single_bird_pdi()` orchestrates the single-source pipeline identically across `Path` and `bytes`, skips X-Ray encoding when `return_xray=False`, degrades stripped-EXIF/RGBA/grayscale uploads gracefully to `4.25 mm / 1.0x`, stamps `pdi_config_hash`, and completes on native `4080×1840` Wide JPEGs in `< 400 ms` (`< 500 ms` SLA). | ✓ VERIFIED | Direct read of `backend/app/pdi/pipeline.py` (89 lines). Independent live benchmark on 8 native `4080×1840` Wide JPEGs across orientations `[1, 3, 6, 8]` measured `p50 = 74.4 ms`, `p95 = 88.3 ms`, `max = 89.5 ms` (`> 4.4×` faster than `400 ms` target). Verified by `test_run_single_bird_pdi_path_vs_bytes_parity_and_xray_toggle`, `test_run_single_bird_pdi_handles_stripped_jpeg_rgba_grayscale_png_and_override`, and `test_real_native_4080x1840_wide_jpeg_meets_latency_and_payload_sla` (all PASS). |
| 9 | **ROADMAP SC-5 / 02-03 T1–T2 (`PDI-05` Isolated Marker-Based Watershed Engine & `WatershedResult` Contract):** `run_watershed_pdi()` runs strictly classical Marker-based Watershed decoupled from single-bird training (reusing `load_working_rgb`, `resolve_optical_context`, `preprocess_working_rgb`, and `extract_mask_gated_features` per separated basin) with chromatic red-feeder pre-suppression, positive-score Otsu robust z-score union mask + wall-strip rejection + `binary_fill_holes`, exact float64 EDT, scale-adaptive Gaussian smoothing, grouped-bbox `h_maxima`, coordinate-only `peak_local_max` + disconnected-component rescue, `skimage.segmentation.watershed`, tiny-basin pruning, 49-key per-instance features, and on-demand 4-stage Base64 `WatershedVisuals`. | ✓ VERIFIED | Direct read of `backend/app/pdi/watershed.py` (692 lines) confirming zero folder-name hardcoding, isolated multi-bird path, and per-basin `extract_mask_gated_features(frame.bgr_work, inst_mask, ...)`. Verified by `test_watershed_suppresses_red_feeder_and_excludes_it_from_bird_basins`, `test_watershed_rejects_border_architectural_wall_strip`, `test_watershed_per_instance_49_features_optical_override_and_4_stage_visuals`, and `test_watershed_module_contains_zero_folder_name_hardcoding` (all PASS). |
| 10 | **02-03 T3–T7 (`PDI-05` Edge Probes — Boundary, Adjacency, Empty, Ordering, Precision):** ASVS V5 parameter bounds (`h_rel ∈ [0.05, 0.80]`, `min_distance_rel ∈ [0.20, 2.50]`, `min_area_ratio ∈ [0.01, 0.45]`) accept exact boundaries and reject out-of-range/`bool`/`NaN`/`Inf` (`boundary`); fused touching birds split into mutually exclusive basins while isolated birds & sub-threshold humps stay `1` (`adjacency`); uniform dark / all-red frames return `bird_count=0, instances=()` cleanly (`empty`); equal-area touching ellipses sort deterministically by `(-area_work_px, round(cy, 3), round(cx, 3))` with contiguous `1..N` IDs and `Path`/`bytes` parity (`ordering`); exact `float64` EDT + grouped-bbox `h_maxima` matches full-frame bit-for-bit with finite 49-key features (`precision`). | ✓ VERIFIED | Verified across all 35 synthetic unit tests in `backend/tests/test_pdi_watershed.py`: `test_watershed_accepts_exact_min_and_max_parameter_bounds` (6), `test_watershed_rejects_out_of_bounds_bool_and_non_finite_params` (17), `test_watershed_separates_touching_fused_birds_with_disjoint_basins`, `test_watershed_preserves_isolated_single_bird_and_merges_subthreshold_humps`, `test_watershed_handles_empty_uniform_dark_and_all_red_scenes_cleanly`, `test_watershed_deterministic_ordering_and_path_bytes_parity`, `test_bbox_cropped_h_maxima_matches_full_frame_bit_identically` (3), and `test_watershed_basin_masks_are_mutually_exclusive_and_empty_h_maxima_safe` (all PASS). |
| 11 | **02-03 T8 (`backend/app/pdi/__init__.py` Public Package Barrel):** Exposes the complete public Phase 2 single-bird PDI + Watershed contract (`34` symbols) with an alphabetically sorted `__all__`. | ✓ VERIFIED | Direct read of `backend/app/pdi/__init__.py` (79 lines, 34 symbols in sorted `__all__`). Verified by `test_pdi_package_barrel_exports_complete_sorted_public_api` (PASS). |
| 12 | **02-03 T9 (`backstop` — Real 13-Image Multi-Bird Corpus Audit incl. Folder `757`):** On all `13` quarantined multi-bird images (`417`: 4, `757`: 1, `823`: 4, `1520`: 4) using zero-hardcoding defaults (`h_rel=0.20, min_distance_rel=0.85, min_area_ratio=0.10`), `run_watershed_pdi()` hits `>= 8/13` exact counts (`>= 11/13` within `±1`), while interactive slider tuning on low-light oblique folder `757` is deferred to Phase 4 `WEB-04`. | ✓ VERIFIED | Direct live execution across all 13 real images in `manifest_quarantine.parquet` observed counts `{'417': [2, 2, 2, 2], '757': [4], '823': [5, 4, 4, 4], '1520': [2, 2, 3, 3]}` = **`9 / 13` exact** (`>= 8/13`), **`12 / 13` within `±1`** (`>= 11/13`), `MAE = 0.38` birds, `417` red-feeder suppression active on 4/4 shots (`min 15,725 px`), `lat_p50 = 131.8 ms`, `lat_max = 169.9 ms` (`< 350 ms`), `b64_max = 145.1 KB` (`< 300 KB`). Enforced by `test_real_13_quarantined_multi_bird_images_read_only_watershed_audit` (PASS); Phase 4 `WEB-04` interactive sliders tracked in `.planning/REQUIREMENTS.md` line 34. |
| 13 | **ROADMAP SC-3 / 02-04 T1 (`extract_features.py` CLI, ASVS V4 Containment & Capped 4-Worker Pool):** Reads `manifest_clean.parquet`, enforces ASVS V4 traversal/symlink containment (`_resolve_safe_image_path`), clamps concurrency strictly to `min(workers, 4)` with `ProcessPoolExecutor(mp_context="spawn", initializer=cv2.setNumThreads(1))`, executes `run_single_bird_pdi(..., return_xray=False)` preserving exact 1:1 `image_id` order, emits strictly aggregate stdout (`0` `.jpg`), and writes zero files under `--dry-run`. | ✓ VERIFIED | Direct read of `backend/scripts/extract_features.py` (594 lines) and `--help` CLI smoke check. Verified by `test_parse_args_defaults_worker_clamping_and_invalid_workers_rejection`, `test_asvs_v4_path_containment_rejects_traversal_and_symlinks` (5 parametrized escape vectors), `test_cli_dry_run_writes_zero_files_and_emits_zero_jpg_stdout`, and `test_cli_full_run_mixed_lens_schema_order_clamped_workers_and_byte_idempotency` (all PASS). |
| 14 | **ROADMAP SC-3 / 02-04 T2 (Materialized `data/processed/features_clean.parquet` & `< 60 s` Batch SLA):** Materialized git-ignored `features_clean.parquet` (`656 KB`) has exact `1,738 × 60` typed PyArrow schema (`11` join/audit + `49` `pa.float64()`), exact 1:1 row alignment with `manifest_clean.parquet`, zero `NaN`/`Inf`, zero banned columns, uniform `pdi_config_hash = "43a3cfaa3208019e"`, `< 2.0%` primary fallback (`4/1,738 = 0.2301%`), `0` bucket LCD FPs, exact code parity, and `< 60 s` (`57.08 s`) 4-worker batch runtime. | ✓ VERIFIED | Independent Parquet audit confirmed `size=671,141 B`, `shape=(1738, 60)`, exact Arrow types, exact 1:1 alignment across all 6 manifest join keys, `0` NaN/Inf, `0` banned columns, `4/1,738 = 0.2301%` fallback, `0` bucket LCD FPs, and exact `0.0` max feature diff across 8 stratified live `run_single_bird_pdi()` re-evaluations (including fallback rows). Verified by `test_real_features_clean_parquet_schema_alignment_finite_values_and_fallback_gate` (PASS). |
| 15 | **ROADMAP SC-3 / 02-04 T3 (Git-Tracked `models/pdi_calibration_audit.json` & 4-Scope Optical Proof):** Byte-deterministic UTF-8 JSON (omitting volatile wall-clock timestamps, `0` forbidden `PROIBIÇÃO 4` leak tokens) quantitatively proves across all 4 paired dual-lens scopes (`all_mixed_lens_sessions` `N=127` spanning all `111` clean mixed-lens folders [raw `114` minus `3` quarantined multi-bird], `same_scene_pairs` `N=175`, `bucket_only_pairs` `N=76`, `scale_or_litter_pairs` `N=99`) that `area_px / (zoom_multiple ** 2)` collapses Wide/UW ratio closest to `1.0` (`0.963757` on bucket pairs `∈ [0.80, 1.25]`) and strictly outperforms rejected double-counted `area_px / ((focal_length_mm * zoom_multiple) ** 2)` (`0.1129–0.1453`, `med|log| 1.93–2.18`) on all 4 scopes. | ✓ VERIFIED | Direct read of `models/pdi_calibration_audit.json` (213 lines, `SHA-256 = 3c2029095c4b276799d9c28ec210d105e2973e525c1d7293c7971c1fc6d8eba6`), raw-text scan confirming `0` forbidden tokens (`.jpg`, `.jpeg`, `rel_path`, `/home/`, `data/raw`, username), and exact dict equality with freshly computed `extract_features.build_calibration_audit(fdf, cfg, workers=4)`. Verified by `test_write_calibration_audit_rejects_proibicao_4_leak_markers` (5) and `test_real_pdi_calibration_audit_json_zero_leakage_and_4_scope_optical_proof` (all PASS). |
| 16 | **02-04 T4 (Git Hygiene, Zero New Dependencies & Read-Only `data/raw/`):** `data/raw/` and `data/processed/` remain completely untracked by Git, `models/pdi_calibration_audit.json` is tracked and not ignored, zero new external packages were added (`pyproject.toml`/`uv.lock` untouched), and `data/raw/` remains unmutated. | ✓ VERIFIED | `git ls-files data/raw data/processed` empty; `git ls-files models/` returns `models/dataset_audit_summary.json` and `models/pdi_calibration_audit.json`; `git check-ignore models/pdi_calibration_audit.json` exits `1`; `git diff 9b4b312...HEAD -- pyproject.toml uv.lock .python-version` empty; `uv sync --frozen --no-build --managed-python --check` confirms `Would make no changes`; `data/raw/` untouched (`1,753` files, latest mtime `2024-03-25T17:32:56`). Verified by `test_git_hygiene_excludes_features_parquet_while_tracking_calibration_audit` and `test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions` (PASS). |

**Score:** `16/16` truths verified (`0` present, behavior-unverified; `0` failed; `0` overrides)

---

## Artifact Verification (Levels 1–4)

Automated `gsd-tools.cjs query verify.artifacts` returned `all_passed: true` on all 4 PLANs (`7/7` on `02-01`, `5/5` on `02-02`, `3/3` on `02-03`, `3/3` on `02-04` — `18/18` total), reinforced by direct file reads, line counts, and runtime data-flow audits:

| Artifact | Plan(s) | Exists | Substantive | Wired | Data Flows | Status | Details |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `backend/app/config.py` | 02-01 | ✓ | ✓ (34 lines, 10 exports) | ✓ (imported by `extract_features.py`, `build_manifest.py`, tests) | ✓ | ✓ VERIFIED | Appended `FEATURES_CLEAN_PATH` & `PDI_CALIBRATION_AUDIT_PATH`; all 8 Phase 1 constants preserved |
| `backend/app/pdi/config.py` | 02-01 | ✓ | ✓ (245 lines, `PDIConfig`) | ✓ (imported across all PDI modules, `extract_features.py`, barrel, tests) | ✓ | ✓ VERIFIED | Frozen slotted dataclass, ASVS V5 validation, deterministic 16-hex SHA-256 `43a3cfaa3208019e` |
| `backend/app/pdi/io_exif.py` | 02-01 | ✓ | ✓ (378 lines, 23 exports) | ✓ (imported by `pipeline.py`, `watershed.py`, `features.py`, `manifest.py`, barrel, tests) | ✓ | ✓ VERIFIED | Acyclic (zero `backend.app.ingestion` imports), pre-draft EXIF + aspect-preserving `im.draft()`, `Path \| bytes` parity |
| `backend/app/pdi/preprocess.py` | 02-01 | ✓ | ✓ (95 lines, 2 exports) | ✓ (imported by `lcd_suppression.py`, `segmentation.py`, `xray.py`, `pipeline.py`, `watershed.py`, barrel) | ✓ | ✓ VERIFIED | Bilateral/Median smoothing, Lab `L`-CLAHE, HSV `uint8`, `float32` Scharr maps |
| `backend/app/pdi/lcd_suppression.py` | 02-01 | ✓ | ✓ (152 lines, 2 exports) | ✓ (imported by `segmentation.py`, barrel, tests) | ✓ | ✓ VERIFIED | Rotation-invariant `minAreaRect` LCD bezel + connected smooth housing suppressor (`786` triggers, `0` bucket FPs) |
| `backend/app/pdi/segmentation.py` | 02-01 | ✓ | ✓ (577 lines, 4 exports) | ✓ (imported by `features.py`, `xray.py`, `pipeline.py`, barrel, tests) | ✓ | ✓ VERIFIED | Border-ring scene classifier, Scenario B wall-peeling + Black-Hat barrier, Scenarios A/C anti-LCD sandwich, orientation-normalized `fit_ellipse_safe`, `[0.015, 0.72]` fallback |
| `backend/app/pdi/features.py` | 02-02 | ✓ | ✓ (482 lines, 6 exports) | ✓ (imported by `pipeline.py`, `watershed.py`, `extract_features.py`, barrel, tests) | ✓ | ✓ VERIFIED | Frozen 49-col `FEATURE_COLUMNS`, `BANNED_FEATURE_COLUMNS`, pixel-local `bgr_work` tight-bbox + 3-px eroded texture (`0.0` outside-mask diff) |
| `backend/app/pdi/xray.py` | 02-02 | ✓ | ✓ (192 lines, 2 exports) | ✓ (imported by `pipeline.py`, barrel, tests) | ✓ | ✓ VERIFIED | 4-stage `<= 640 px` Base64 renderer (`3× JPEG Q=82` + `1× 1-ch PNG`, `max = 166.8 KB < 250 KB`) |
| `backend/app/pdi/pipeline.py` | 02-02 | ✓ | ✓ (89 lines, 2 exports) | ✓ (imported by `extract_features.py`, barrel, tests; feeds Phase 4 FastAPI) | ✓ | ✓ VERIFIED | Single-source `run_single_bird_pdi()` passing pixel-local `frame.bgr_work`; `p95 = 88.3 ms` on native `4080×1840` |
| `backend/app/pdi/watershed.py` | 02-03 | ✓ | ✓ (692 lines, 10 exports) | ✓ (imported by barrel, tests; feeds Phase 4 `/api/watershed-demo`) | ✓ | ✓ VERIFIED | Isolated Marker-based Watershed with red-feeder suppression, grouped-bbox bit-identical `h_maxima`, per-instance 49 features, 4 visuals |
| `backend/app/pdi/__init__.py` | 02-03 | ✓ | ✓ (79 lines, 34 sorted exports) | ✓ (imported by `test_pdi_watershed.py` & Phase 3/4 consumers) | ✓ | ✓ VERIFIED | Complete alphabetically sorted `__all__` covering single-bird PDI + Watershed |
| `backend/scripts/extract_features.py` | 02-04 | ✓ | ✓ (594 lines, 10 exports) | ✓ (CLI entrypoint + imported by `test_extract_features_integration.py`) | ✓ | ✓ VERIFIED | ASVS V4 containment, `MAX_WORKERS_CAP = 4`, `cv2.setNumThreads(1)`, 60-col Arrow schema, byte-deterministic audit JSON |
| `models/pdi_calibration_audit.json` | 02-04 | ✓ | ✓ (213 lines, 6.8 KiB, valid JSON) | ✓ (git-tracked, verified by integration test, feeds Phase 3 & Phase 4 `WEB-05`) | ✓ | ✓ VERIFIED | `SHA-256 = 3c202909...`, `0` forbidden leak tokens, exact rebuild parity, 4-scope D-09 optical proof |
| `data/processed/features_clean.parquet` *(git-ignored runtime)* | 02-04 | ✓ | ✓ (656 KiB, `1,738 × 60` Arrow cols) | ✓ (read by integration tests, feeds Phase 3 `MOD-01`..`MOD-03`) | ✓ | ✓ VERIFIED | Exact 60-col Arrow schema, 1:1 manifest alignment, `0` NaN/Inf, `0.2301%` fallback, `0.0` live code diff |
| `backend/tests/test_pdi_preprocess_and_lcd.py` | 02-01 | ✓ | ✓ (656 lines, 24 tests) | ✓ (`pytest`) | ✓ | ✓ VERIFIED | 24/24 PASS (`min_lines: 120` met) |
| `backend/tests/test_pdi_features_invariance.py` | 02-02 | ✓ | ✓ (303 lines, 8 tests) | ✓ (`pytest`) | ✓ | ✓ VERIFIED | 8/8 PASS (`min_lines: 100` met) |
| `backend/tests/test_pdi_pipeline_and_xray.py` | 02-02 | ✓ | ✓ (364 lines, 5 tests) | ✓ (`pytest`) | ✓ | ✓ VERIFIED | 5/5 PASS (`min_lines: 100` met) |
| `backend/tests/test_pdi_watershed.py` | 02-03 | ✓ | ✓ (632 lines, 37 tests) | ✓ (`pytest`) | ✓ | ✓ VERIFIED | 37/37 PASS (`min_lines: 120` met) |
| `backend/tests/test_extract_features_integration.py` | 02-04 | ✓ | ✓ (639 lines, 17 tests) | ✓ (`pytest`) | ✓ | ✓ VERIFIED | 17/17 PASS (`min_lines: 120` met) |

---

## Key Link Verification

Automated `gsd-tools.cjs query verify.key-links` returned **`all_verified: true` (`3/3` on every plan = `12/12` total)**:

| Plan | From | To | Via | Status | Details |
| --- | --- | --- | --- | --- | --- |
| 02-01 | `backend/app/pdi/preprocess.py` | `backend/app/pdi/config.py` | `from backend.app.pdi.config import PDIConfig` | ✓ WIRED | Verified by `verify.key-links` & `preprocess.py` line 21; zero `backend.app.ingestion` imports. |
| 02-01 | `backend/app/pdi/segmentation.py` | `backend/app/pdi/lcd_suppression.py` | `from backend.app.pdi.lcd_suppression import LcdSuppressionResult, suppress_lcd_and_platter` | ✓ WIRED | Verified by `verify.key-links` & `segmentation.py` lines 30, 386, 421–435 (subtracted pre-open/close AND re-zeroed post-hole-fill). |
| 02-01 | `backend/app/pdi/segmentation.py` | `backend/app/pdi/preprocess.py` | `from backend.app.pdi.preprocess import PreprocessedFrame` | ✓ WIRED | Verified by `verify.key-links` & `segmentation.py` line 31. |
| 02-02 | `backend/app/pdi/pipeline.py` | `backend/app/pdi/features.py` | `extract_mask_gated_features(frame.bgr_work, seg.bird_mask, ...)` | ✓ WIRED | Verified by `verify.key-links` & `pipeline.py` lines 16, 67–74 (explicitly routes pixel-local `frame.bgr_work`, never `bgr_clahe`/`bgr_smooth`). |
| 02-02 | `backend/app/pdi/pipeline.py` | `backend/app/pdi/xray.py` | `render_xray_artifacts(frame, seg, cfg) if return_xray else None` | ✓ WIRED | Verified by `verify.key-links` & `pipeline.py` lines 25, 76. |
| 02-02 | `backend/app/pdi/pipeline.py` | `backend/app/pdi/segmentation.py` | `segment_single_bird(frame, cfg)` | ✓ WIRED | Verified by `verify.key-links` & `pipeline.py` lines 24, 63, 72–73. |
| 02-03 | `backend/app/pdi/watershed.py` | `backend/app/pdi/features.py` | `extract_mask_gated_features(frame.bgr_work, inst_mask, ...)` | ✓ WIRED | Verified by `verify.key-links` & `watershed.py` lines 39, 617–622 (per separated basin). |
| 02-03 | `backend/app/pdi/watershed.py` | `backend/app/pdi/io_exif.py` | `load_working_rgb`, `resolve_optical_context` | ✓ WIRED | Verified by `verify.key-links` & `watershed.py` lines 40–45, 460–470. |
| 02-03 | `backend/app/pdi/__init__.py` | `backend/app/pdi/watershed.py` | `from backend.app.pdi.watershed import ...` | ✓ WIRED | Verified by `verify.key-links` & `__init__.py` lines 36–41, 60–62, 76. |
| 02-04 | `backend/scripts/extract_features.py` | `backend/app/pdi/pipeline.py` | `run_single_bird_pdi(Path(abs_path_str), config=cfg, return_xray=False)` | ✓ WIRED | Verified by `verify.key-links` & `extract_features.py` lines 39, 184 inside spawn worker pool. |
| 02-04 | `backend/scripts/extract_features.py` | `backend/app/pdi/features.py` | `FEATURE_COLUMNS`, `BANNED_FEATURE_COLUMNS` | ✓ WIRED | Verified by `verify.key-links` & `extract_features.py` lines 38, 76–83, 120, 220. |
| 02-04 | `backend/scripts/extract_features.py` | `backend/app/config.py` | `CLEAN_MANIFEST_PATH`, `FEATURES_CLEAN_PATH`, `PDI_CALIBRATION_AUDIT_PATH`, `RAW_DATASET_DIR` | ✓ WIRED | Verified by `verify.key-links` & `extract_features.py` lines 31–36, 431–449. |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable / Output | Upstream Source | Produces Real Data | Status |
| --- | --- | --- | --- | --- |
| `data/processed/features_clean.parquet` | `1,738` rows × `60` Arrow cols (`11` join/audit + `49` `float64` features) | `manifest_clean.parquet` → read-only `data/raw/` → `run_single_bird_pdi(..., return_xray=False)` via 4-worker `ProcessPoolExecutor` | Yes (`377` bucket, `1,361` scale/litter, `4` fallbacks, `786` LCD suppressions, `0.0` max live re-evaluation diff across 8 stratified rows) | ✓ FLOWING |
| `models/pdi_calibration_audit.json` | Aggregate extraction summary + 4-scope dual-lens optical calibration table (`6` formulas × `4` scopes) | `build_calibration_audit(features_df, PDIConfig(), workers=4)` over real `features_clean.parquet` | Yes (exact dict equality with fresh `build_calibration_audit()`, `N=127` mixed-lens sessions / `76` bucket pairs `med W/UW = 0.963757`) | ✓ FLOWING |
| `PDIResult.xray` (`xray.py`) | 4 Base64 stages (`<= 640 px`, `< 250 KB`) | Live `PreprocessedFrame` + `SingleBirdSegmentation` from `run_single_bird_pdi(raw_bytes, return_xray=True)` | Yes (real native `4080×1840` JPEGs decode to 3× JPEG Q=82 + 1× 1-ch `"L"` PNG at `140.1–166.8 KB`, `p95 = 88.3 ms`) | ✓ FLOWING |
| `WatershedResult` (`watershed.py`) | `bird_count`, `instances` (`49` features each), `visuals` (4 Base64 stages) | Read-only `manifest_quarantine.parquet` (`13` multi-bird scenes) or arbitrary `bytes` via `run_watershed_pdi()` | Yes (`9/13` exact, `12/13` within `±1`, `417`: `4/4` exact with `>= 15,725 px` red-feeder suppression, `max = 169.9 ms`, `max = 145.1 KB`) | ✓ FLOWING |

---

## Behavioral Spot-Checks & Probe Execution

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| --- | --- | --- | --- |
| Acyclic `pdi.io_exif` import before `ingestion` + `classify_lens` parity | Fresh-process Python import check | `PASS` (zero `ImportError`, 7/7 probe values match `decoding.classify_lens`) | ✓ PASS |
| Bit-identical perf optimizations (`_fill_holes_u8` & `_area_convex_exact`) | 10 concave/holed shapes vs `binary_fill_holes` & `regionprops.area_convex` | `10/10` exact pixel & area equality (`0` diffs) | ✓ PASS |
| Live code-vs-Parquet parity on real corpus | `run_single_bird_pdi()` on 8 stratified clean rows (incl. fallback) vs `features_clean.parquet` | `max_diff = 0.0` across all 49 features & audit fields | ✓ PASS |
| Real native `4080×1840` Wide JPEG latency & X-Ray payload SLA | `run_single_bird_pdi(raw_bytes, return_xray=True)` across 8 real JPEGs (orientations `1, 3, 6, 8`) | `p50 = 74.4 ms`, `p95 = 88.3 ms`, `max = 89.5 ms` (`< 400 ms`); `b64_p50 = 140.1 KB`, `b64_max = 166.8 KB` (`< 250 KB`) | ✓ PASS |
| Real 13-image multi-bird Watershed audit | `run_watershed_pdi(img_path, return_visuals=True)` across all 13 quarantined `multi_bird` rows | `counts={'417':[2,2,2,2],'757':[4],'823':[5,4,4,4],'1520':[2,2,3,3]}`, `9/13` exact, `12/13` within `±1`, `MAE=0.38`, `lat_max=169.9 ms`, `b64_max=145.1 KB` | ✓ PASS |
| CLI entrypoint help & flags | `uv run python -m backend.scripts.extract_features --help` | Exit `0`; exposes all 6 documented flags | ✓ PASS |

### Probe Execution

`find scripts -path '*/tests/probe-*.sh' -type f` returned `NONE` (no shell probe scripts declared for Phase 2).

---

## Requirements & Locked Decisions Coverage

### Requirements Coverage (`REQUIREMENTS.md` Phase 2 Mapping)

| Requirement | Source Plans | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| `PDI-01` | `02-01` | Single shared Pillow EXIF-first loader (`Path \| bytes`, zero `cv2.imread`/`cv2.imdecode`), aspect-preserving canonical downscale (`768–960 px` with `scale_factor`), Bilateral/Median smoothing, and CIE Lab `L`-channel CLAHE. | ✓ SATISFIED | `config.py`, `io_exif.py`, `preprocess.py`; `24/24` tests in `test_pdi_preprocess_and_lcd.py` + `17/17` in `test_io_exif.py` (AST gate); `0` raw OpenCV decoders repo-wide. |
| `PDI-02` | `02-01` | Strictly classical single-bird segmentation across SF-400 scale on straw, top-down white plastic bucket (incl. white-on-white `>1,000 g`), and oblique straw litter, explicitly suppressing SF-400 LCD/buttons/platter before closing/selection with `< 2%` fallback. | ✓ SATISFIED | `lcd_suppression.py`, `segmentation.py`; `4/1,738 = 0.2301%` fallback (`0/59` `>1,000 g` bucket, `786` LCD triggers, `0` bucket LCD FPs); all unit & real-corpus gates pass. |
| `PDI-03` | `02-02`, `02-04` | Strictly mask-gated (`bird_mask == 255`, verified bit-identical under outside-mask perturbation) morphometric, chromatic/texture, and empirically calibrated EXIF optical features (avoiding `focal × zoom` double-counting & banning raw dims), plus capped-concurrency (`n_jobs=4`) batch write to `data/processed/features_clean.parquet` & `models/pdi_calibration_audit.json`. | ✓ SATISFIED | `features.py`, `extract_features.py`, `features_clean.parquet` (`1,738 × 60`, `0` NaN/Inf, `57.08 s` on 4 workers), `models/pdi_calibration_audit.json` (`N=127` mixed-lens sessions / `76` bucket pairs `med W/UW = 0.963757`); `8/8` invariance tests + `17/17` integration tests PASS. |
| `PDI-04` | `02-02` | Single-bird PDI pipeline returns 4 encoded visual "X-Ray" inspection artifacts (`original`, `preprocessed_clahe`, `clean_mask`, `contour_ellipse_overlay`) alongside locked `PDIConfig` hash within `< 500 ms` CPU SLA. | ✓ SATISFIED | `xray.py`, `pipeline.py`; `5/5` tests in `test_pdi_pipeline_and_xray.py` PASS; live native `4080×1840` `p95 = 88.3 ms` (`< 400 ms`) & `max = 166.8 KB` (`< 250 KB`). |
| `PDI-05` | `02-03` | Isolated classical Marker-based Watershed (`distance_transform_edt` + Gaussian + `h_maxima`/`peak_local_max` + `skimage.segmentation.watershed` + red-feeder pre-suppression) validated across `417`, `757`, `823`, `1520` and custom uploads, returning 4 visuals, counts, areas, and per-instance mask-gated features. | ✓ SATISFIED | `watershed.py`, `backend/app/pdi/__init__.py`; `37/37` tests in `test_pdi_watershed.py` PASS; live 13-image audit `9/13` exact, `12/13` within `±1`, `417`: `4/4` exact, `max = 169.9 ms`. |

*Orphaned Phase 2 requirements in `REQUIREMENTS.md`:* **0** (all 5 mapped requirements `PDI-01`..`PDI-05` claimed and satisfied).

### Locked Decisions Coverage (`02-CONTEXT.md` `D-01`..`D-12`)

| Decision | Status | Direct Evidence |
| --- | --- | --- |
| `D-01` (Frozen `PDIConfig`, `working_long_edge=768`, pre-draft dims, 16-hex SHA-256 hash) | ✓ VERIFIED | `backend/app/pdi/config.py`, `io_exif.py` lines 316–326; `config_hash() == "43a3cfaa3208019e"`. |
| `D-02` (Single Pillow EXIF-first boundary, `Path \| bytes` parity, RGBA/gray/missing-EXIF `4.25 mm / 1.0x`) | ✓ VERIFIED | `io_exif.py`; AST ban (`0` raw OpenCV loads); tested in `test_pdi_preprocess_and_lcd.py` & `test_pdi_pipeline_and_xray.py`. |
| `D-03` (Parameterized Bilateral/Median + CIE Lab `L`-channel CLAHE + OpenCV `uint8` scaling) | ✓ VERIFIED | `preprocess.py` lines 58–81; both smoothing modes verified in `test_pdi_preprocess_and_lcd.py`. |
| `D-04` (On-demand 4-stage `640 px` X-Ray Base64 `< 250 KB`; skipped when `return_xray=False`) | ✓ VERIFIED | `xray.py`, `pipeline.py` line 76; native `4080×1840` `max = 166.8 KB`, `p95 = 88.3 ms`. |
| `D-05` (Dedicated rotation-invariant `lcd_suppression.py` strictly before closing/selection) | ✓ VERIFIED | `lcd_suppression.py`, `segmentation.py` lines 386, 421–435; `0°, 90°, 180°, 270°, 35°` tests PASS; `786` triggers, `0` bucket FPs. |
| `D-06` (Border-ring scene branching: Scenario B smooth-wall peeling + Black-Hat; A/C anti-LCD + open `9×9`) | ✓ VERIFIED | `segmentation.py` lines 200–438; `377` bucket (`0` fallbacks) / `1,361` scale-litter (`4` fallbacks). |
| `D-07` (Guarded `fit_ellipse_safe`, `[0.015, 0.72]` envelope, deterministic fallback `< 2%` across `1,738`) | ✓ VERIFIED | `segmentation.py` lines 157–187, 441–569; `4/1,738 = 0.2301%` fallback (`0` crashes). |
| `D-08` (Strict mask-gated 49-col `features.py`, `0.0` outside-mask perturbation diff, banned raw/bbox cols) | ✓ VERIFIED | `features.py`; `test_outside_mask_perturbation_yields_bit_identical_features` asserts exact `f_a[k] == f_b[k]` across all 49 features. |
| `D-09` (Empirical dual-lens calibration audit proving `area_px / (zoom_multiple ** 2)` in `models/pdi_calibration_audit.json`) | ✓ VERIFIED | `models/pdi_calibration_audit.json` (`N=127` mixed-lens sessions; bucket `0.963757` vs double-counted `0.145264`; wins all 4 scopes). |
| `D-10` (Capped 4-worker `extract_features.py` with `cv2.setNumThreads(1)` writing `1,738 × 60` Parquet `< 60 s`) | ✓ VERIFIED | `extract_features.py` (`MAX_WORKERS_CAP=4`, `_init_worker`); `features_clean.parquet` (`1,738 × 60`, `57.08 s` wall-clock). |
| `D-11` (Isolated `watershed.py` with red-feeder pre-suppression, float64 EDT, Gaussian, bbox `h_maxima`, `peak_local_max`, `skimage.segmentation.watershed`) | ✓ VERIFIED | `watershed.py` (692 lines); `0` coupling to single-bird training, `0` folder hardcoding. |
| `D-12` (`WatershedResult` with counts, relative params, per-instance 49-key features, 4 visuals validated on all 13 multi-bird scenes) | ✓ VERIFIED | `watershed.py`, `test_pdi_watershed.py`; `9/13` exact, `12/13` within `±1`, `max = 169.9 ms`. |

---

## Prohibitions & STRIDE Threat Model Verification

### Prohibitions (`PROIBIÇÃO 1`, `PROIBIÇÃO 2`, `PROIBIÇÃO 4` & PLAN `must_haves.prohibitions`)

| Prohibition | Automated Enforcement & Direct Evidence | Verdict |
| --- | --- | --- |
| **PROIBIÇÃO 1 (Zero Deep Learning / OCR imports & zero new external dependencies)** | AST scan across all 28 Python files under `backend/` confirmed `0` DL/OCR imports (`torch`, `tensorflow`, `keras`, `ultralytics`, `timm`, `onnxruntime`, `paddleocr`, `easyocr`, `sam2`, `segment_anything`, `pytesseract`). `git diff 9b4b312...HEAD -- pyproject.toml uv.lock .python-version` is empty (`0` new packages) and `uv sync --frozen --no-build --managed-python --check` confirms zero drift. Enforced by `test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions`. | ✓ VERIFIED (0 violations) |
| **PROIBIÇÃO 2 (Zero raw `cv2.imread`/`cv2.imdecode`, pre-closing SF-400 LCD exclusion, exact `0.0` outside-mask feature invariance, zero banned raw/bbox columns, zero folder hardcoding)** | AST scan across `backend/app/` & `backend/scripts/` found `0` `cv2.imread`/`cv2.imdecode` calls and `0` `.to_csv(` calls; `suppress_lcd_and_platter()` executes strictly before open/close and re-zeroes post-hole-fill (`segmentation.py` lines 386, 421–435); `pipeline.py` & `watershed.py` pass strictly pixel-local `frame.bgr_work` into `extract_mask_gated_features()`; `BANNED_FEATURE_COLUMNS` disjoint from both `FEATURE_COLUMNS` and `features_clean.parquet`; static scan of `backend/app/pdi/` found `0` hardcoded folder tokens (`417`, `757`, `823`, `1520`, `1340_`). Enforced by `test_ast_scan_confirms_zero_raw_opencv_image_loading_under_backend_app`, `test_outside_mask_perturbation_yields_bit_identical_features`, `test_segment_single_bird_scenario_a_touching_platter_has_zero_lcd_leakage`, `test_full_pipeline_sf400_anti_lcd_excludes_digits_completely`, `test_watershed_module_contains_zero_folder_name_hardcoding`, and `test_real_features_clean_parquet_schema_alignment_finite_values_and_fallback_gate`. | ✓ VERIFIED (0 violations) |
| **PROIBIÇÃO 4 (Read-only `data/raw/`, zero raw/processed files in Git, zero raw filenames/paths in stdout or JSON)** | `data/raw/` untouched (`1,753` files, latest mtime `2024-03-25T17:32:56`); `git ls-files data/raw data/processed` empty (`0` tracked files); `git check-ignore models/pdi_calibration_audit.json` exits `1`; `models/pdi_calibration_audit.json` raw text has `0` occurrences of `.jpg`, `.jpeg`, `rel_path`, `/home/`, `data/raw`, or host username; `extract_features.py` emits strictly aggregate stdout (`write_calibration_audit()` enforces runtime `_FORBIDDEN_LEAK_SUBSTRINGS` guard). Enforced by `test_cli_dry_run_writes_zero_files_and_emits_zero_jpg_stdout`, `test_write_calibration_audit_rejects_proibicao_4_leak_markers`, `test_real_pdi_calibration_audit_json_zero_leakage_and_4_scope_optical_proof`, and `test_git_hygiene_excludes_features_parquet_while_tracking_calibration_audit`. | ✓ VERIFIED (0 violations) |

*Non-authoritative prohibition review advisory:* PLAN frontmatter `must_haves.prohibitions` items initially carried `status: unverified` / `flagged: true` prior to phase verification; every prohibition above is backed by passing automated tests and independent AST, Parquet, JSON, lockfile, filesystem, and git-index audits (`0` violations observed; optional human sign-off may be recorded during milestone audit).

### ASVS L1 / STRIDE Mitigations (`T-02-SC`, `T-02-01`..`T-02-12`)

| Threat ID | Mitigation Verified | Evidence |
| --- | --- | --- |
| `T-02-SC` (Supply Chain) | ✓ VERIFIED | Zero changes to `pyproject.toml`/`uv.lock`; `uv sync --frozen --no-build --managed-python --check` clean; `test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions` PASS. |
| `T-02-01` (Decompression Bomb / OOM) | ✓ VERIFIED | `PIL.Image.MAX_IMAGE_PIXELS` remains enabled; `MAX_XIAOMI_TAG_BYTES = 16_384`; aspect-preserving `im.draft("RGB", draft_box)` caps working RGB buffers. |
| `T-02-02` (Invalid Config / `OpticalOverride` Params) | ✓ VERIFIED | `PDIConfig.__post_init__` and `resolve_optical_context()` enforce ASVS V5 finite numeric bounds rejecting `bool`/`NaN`/`Inf`/out-of-range (`test_pdi_preprocess_and_lcd.py`). |
| `T-02-03` & `T-02-04` (Visual Leakage / `PROIBIÇÃO 2`) | ✓ VERIFIED | Rotation-invariant pre-closing + post-hole-fill SF-400 LCD exclusion; pixel-local `bgr_work` + `7×7` eroded interior texture yielding exact `0.0` outside-mask perturbation diff (`test_pdi_features_invariance.py`, `test_pdi_pipeline_and_xray.py`). |
| `T-02-05` (Resolution / Lens Confounding) | ✓ VERIFIED | `BANNED_FEATURE_COLUMNS` excluded; `area_optical_norm = area_px / (zoom_multiple ** 2)` empirically verified across all 4 scopes (`models/pdi_calibration_audit.json`). |
| `T-02-06` (X-Ray Payload / Latency Bloat) | ✓ VERIFIED | Stages capped at `<= 640 px` (3× JPEG Q=82 + 1× 1-ch PNG), skipped when `return_xray=False`; native `4080×1840` `max = 166.8 KB`, `max = 89.5 ms`. |
| `T-02-07` & `T-02-08` (Watershed DoS / Cross-Bird Bleed / Hardcoding) | ✓ VERIFIED | ASVS V5 slider bounds (`[0.05, 0.80]`, `[0.20, 2.50]`, `[0.01, 0.45]`), grouped-bbox `h_maxima` (`max = 169.9 ms`), empty-mask guards, zero folder hardcoding, per-basin `extract_mask_gated_features()` (`test_pdi_watershed.py`). |
| `T-02-09`, `T-02-10`, `T-02-11`, `T-02-12` (Path Traversal, OOM Pool, `PROIBIÇÃO 4`) | ✓ VERIFIED | Pre-pool `_resolve_safe_image_path` traversal/symlink containment (`5` attack vectors rejected), `MAX_WORKERS_CAP = 4` + `cv2.setNumThreads(1)`, runtime `_FORBIDDEN_LEAK_SUBSTRINGS` guard, aggregate-only stdout, and git-hygiene checks (`test_extract_features_integration.py`). |

---

## Anti-Patterns & Commit Hygiene

- **Debt Markers (`TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, `PLACEHOLDER`):** `0` found across all 28 Python files under `backend/`.
- **Stub / Placeholder Implementations:** `0` found.
- **Linter (`uv run ruff check backend/`):** `All checks passed!` (exit `0`).
- **Upstream Warning Note (Non-Blocking):** `scikit-image 0.26.0` emits an upstream `DeprecationWarning` inside `skimage/morphology/grayreconstruct.py:216` under NumPy `2.5.5` during `h_maxima` calls — zero behavioral impact, zero project code warnings.
- **Atomic Commits (`gsd-tools query verify.commits`):** All 10 Phase 2 commits (`8d2a55b`, `98ffcdd`, `ef5ca5a`, `6296f5e`, `898cf74`, `88b916c`, `a236c34`, `af22902`, `bcb6b9e`, `150f6f7`) verified valid in git history (`10/10`).

---

## Commands Run & Observed Output

| Check | Command | Exit | Key Observed Output |
| --- | --- | --- | --- |
| GSD Artifact Verification | `gsd-tools.cjs query verify.artifacts 02-{01,02,03,04}-PLAN.md` | `0` | `7/7`, `5/5`, `3/3`, `3/3` (`all_passed: true` on all 4 plans = `18/18`) |
| GSD Key Link Verification | `gsd-tools.cjs query verify.key-links 02-{01,02,03,04}-PLAN.md` | `0` | `3/3`, `3/3`, `3/3`, `3/3` (`all_verified: true` on all 4 plans = `12/12`) |
| GSD Commit Verification | `gsd-tools.cjs query verify.commits 8d2a55b 98ffcdd ef5ca5a 6296f5e 898cf74 88b916c a236c34 af22902 bcb6b9e 150f6f7` | `0` | `all_valid: true` (`10/10`) |
| Runtime, Lockfile & Linter | `uv sync --frozen --no-build --managed-python --check && uv run python --version && uv run ruff check backend/` | `0` | `Checked 49 packages in 1ms`, `Would make no changes`, `Python 3.12.14`, `All checks passed!` |
| Full Workspace Test Suite | `uv run pytest -v` | `0` | `130 passed, 60 warnings in 11.57s` (`test_decoding` 9, `test_extract_features_integration` 17, `test_io_exif` 17, `test_manifest_integration` 6, `test_pdi_features_invariance` 8, `test_pdi_pipeline_and_xray` 5, `test_pdi_preprocess_and_lcd` 24, `test_pdi_watershed` 37, `test_splits` 7) |
| Independent Read-Only Audit | Python verification script (acyclic import, 28-file AST/debt scan, `features_clean.parquet`, `pdi_calibration_audit.json`, 8-row live code-vs-Parquet parity, 8-image native `4080×1840` X-Ray benchmark, 13-image Watershed audit, `data/raw/` immutability) | `0` | `0` AST/prohibition/debt hits; `features_clean.parquet=(1738, 60)`, `fallbacks=4 (0.2301%)`, `bucket_lcd_fp=0`, `max_opt_formula_err=0.00e+00`, `code_vs_parquet_max_diff=0.0`; `pdi_calibration_audit.json` exact rebuild match & `0` leak tokens (`sha256=3c202909...`); native `4080×1840` `p95=88.3 ms` / `max=166.8 KB`; 13-image Watershed `9/13` exact, `12/13` within `±1`, `max=169.9 ms`; `data/raw/` `1,753` files (`2024-03-25T17:32:56`) |
| Perf Bit-Equivalence & CLI Help | `uv run python -m backend.scripts.extract_features --help` + 10-shape `_fill_holes_u8`/`_area_convex_exact` check | `0` | CLI exits `0`; `10/10` exact parity vs `binary_fill_holes` & `regionprops.area_convex` |
| Git Hygiene & Fingerprint | `git ls-files data/raw data/processed`, `git check-ignore models/pdi_calibration_audit.json`, `git diff 9b4b312...HEAD -- pyproject.toml uv.lock .python-version`, `gsd-tools.cjs query verification.fingerprint ...` | `0`/`1`/`0`/`0` | `data/raw` & `data/processed` untracked; `models/pdi_calibration_audit.json` tracked & not ignored; `0` dep diffs; `covered_digest` computed cleanly |

---

## Gaps Summary

None — all 5 ROADMAP Phase 2 Success Criteria, all 5 Phase 2 requirements (`PDI-01`..`PDI-05`), all 12 locked decisions (`D-01`..`D-12`), all 16 merged observable truths, all 18 artifacts, all 12 key links, all STRIDE mitigations (`T-02-SC`, `T-02-01`..`T-02-12`), and all 130 automated tests passed with concrete direct codebase and runtime evidence.

---

_Verified: 2026-11-23T19:09:10Z_
_Verifier: gsd-verifier_
