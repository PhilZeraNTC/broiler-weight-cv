---
phase: 03-tabular-regression-optical-ablation-production-bundling
plan: 01
subsystem: modeling
tags: [scikit-learn, lightgbm, joblib, pandas, pyarrow, tabular-regression, optical-ablation, anti-leakage]

requires:
  - phase: 01-forensic-data-curation-anti-leakage-split
    provides: "Locked 5-fold StratifiedGroupKFold session splits, manifest_clean.parquet (1,738 x 24), verify_fold_integrity(), and assign_weight_band()."
  - phase: 02-classical-pdi-engine-optical-calibration-watershed
    provides: "Frozen PDIConfig (hash 43a3cfaa3208019e), frozen 49-key FEATURE_COLUMNS & BANNED_FEATURE_COLUMNS, and materialized features_clean.parquet (1,738 x 60)."

provides:
  - "Extended backend/app/config.py exporting repo-root-anchored MODEL_BUNDLE_PATH, EVALUATION_REPORT_PATH, and PRESETS_CATALOG_PATH (D-11)."
  - "Importable pickling-safe sklearn wrappers AllometricLogTransformer and LogDuanRegressor with fold-internal Duan smearing in backend/app/modeling/wrappers.py (D-01, D-02)."
  - "Verified manifest+features join loader, PDIConfig hash lock, fold-integrity gate, and strictly whitelisted 4-condition ablation column specs (C1=40, C2=39, C3=49, C4=39) in backend/app/modeling/datasets.py (D-02, D-04, D-11)."
  - "4-family classical tabular regressor factory (RidgeLog, HuberLog, RandomForestLog, LightGBMLog), fold-only capped sample weighting (max_w/min_w <= 5.0), strict column-ordered single-row predictor, and deterministic importance extractor in backend/app/modeling/models.py (D-01..D-03)."
  - "Pure stdlib Cobb Avian 48 developmental age-stage mapper with [116, 2093] g training-range disclosure in backend/app/modeling/age_stage.py (D-08)."
  - "Comprehensive unit, boundary, fresh-subprocess joblib pickling, injected-leakage rejection, and PROIBIÇÃO 1/3 AST test suites."

affects:
  - 03-02-PLAN
  - 03-03-PLAN
  - 04-web-api-xray-pdi-studio

actuals:
  tokens: 18734
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Importable sklearn estimator wrappers (AllometricLogTransformer, LogDuanRegressor) with zero lambda or __main__ classes for portable joblib deserialization"
    - "Upstream SimpleImputer(strategy='median') before LightGBMLog eliminating feature-name warnings while preserving pipeline.feature_names_in_"
    - "Programmatic ablation column derivation from frozen FEATURE_COLUMNS with module-load and dataset-load PROIBIÇÃO 2/3 assertions"
    - "Fast 2D float64 ndarray-backed single-row DataFrame construction inside predict_from_feature_dict() delivering ~2.1 ms LightGBMLog inference"
    - "Pure stdlib domain mapping module (age_stage.py) decoupled from CV/ML runtimes for Phase 4 reuse"

key-files:
  created:
    - backend/app/modeling/wrappers.py
    - backend/app/modeling/datasets.py
    - backend/app/modeling/models.py
    - backend/app/modeling/age_stage.py
    - backend/tests/test_modeling_wrappers_and_age_stage.py
    - backend/tests/test_modeling_datasets_and_ablation.py
  modified:
    - backend/app/config.py

key-decisions:
  - "Locked conservative empirically verified hyperparameters across all 4 mandatory PRD families (RidgeLog, HuberLog, RandomForestLog, LightGBMLog) with random_state=42 and MAX_N_JOBS_CAP=4."
  - "Kept AllometricLogTransformer strictly scoped to 17 positive geometric area/linear columns (ALLOMETRIC_LOG_COLUMNS) while leaving chromatic/texture and dimensionless columns bit-identical."
  - "Used pre-allocated (1, n_features) float64 ndarray inside pd.DataFrame(..., columns=ordered_names) in predict_from_feature_dict() for 6x faster DataFrame construction (~2.16 ms end-to-end single-row LightGBMLog inference)."
  - "Asserted bit-identical dict-order invariance on linear/LightGBM pipelines and abs=1e-9 parity on RandomForestLog (accounting for 1-2 ULP parallel reduction across n_jobs=4 threads)."

patterns-established:
  - "Every clean-subprocess joblib verification test sets cwd=tmp_path and env={**os.environ, 'PYTHONPATH': str(REPO_ROOT)} under [tool.uv] package = false with -W error::UserWarning."
  - "LoadedModelingDataset strips rel_path and BANNED_FEATURE_COLUMNS completely before returning any modeling DataFrame."

requirements-completed:
  - MOD-01
  - MOD-03

coverage:
  - id: D1
    description: "Repo-root-anchored Phase 3 artifact path constants (MODEL_BUNDLE_PATH, EVALUATION_REPORT_PATH, PRESETS_CATALOG_PATH) backward-compatible with Phases 1-2."
    requirement: MOD-03
    verification:
      - kind: unit
        ref: backend/tests/test_io_exif.py#test_config_paths_are_repo_root_anchored_and_cwd_independent
        status: pass
    human_judgment: false
  - id: D2
    description: "Pickling-safe AllometricLogTransformer and LogDuanRegressor round-tripping through clean-subprocess joblib.load() in < 5 ms with zero UserWarnings."
    requirement: MOD-01
    verification:
      - kind: unit
        ref: backend/tests/test_modeling_wrappers_and_age_stage.py#test_all_four_model_families_fresh_subprocess_joblib_roundtrip_and_latency
        status: pass
    human_judgment: false
  - id: D3
    description: "Verified 1,738-row manifest+features dataset loader with PDIConfig hash lock (43a3cfaa3208019e), zero cross-fold session_id overlap, 4-condition ablation specs (40/39/49/39 cols), and PROIBIÇÃO 1/2/3 enforcement."
    requirement: MOD-01
    verification:
      - kind: unit
        ref: backend/tests/test_modeling_datasets_and_ablation.py#test_load_modeling_dataset_materialized_parquets_contract
        status: pass
      - kind: unit
        ref: backend/tests/test_modeling_datasets_and_ablation.py#test_ast_scan_prohibits_dl_ocr_and_banned_splitters_in_modeling
        status: pass
    human_judgment: false
  - id: D4
    description: "4-family classical regressor factory, fold-only capped sample weighting (max_w/min_w <= 5.0), column-order-enforcing feature-dict predictor, and deterministic tie-broken feature importances."
    requirement: MOD-01
    verification:
      - kind: unit
        ref: backend/tests/test_modeling_wrappers_and_age_stage.py#test_all_four_families_fit_predict_ordering_importances_and_validation
        status: pass
      - kind: unit
        ref: backend/tests/test_modeling_wrappers_and_age_stage.py#test_compute_sample_weights_cap_normalization_and_validation
        status: pass
    human_judgment: false
  - id: D5
    description: "Pure stdlib Cobb Avian 48 age-stage mapper covering all exact weight boundaries (115.99..2093.01 g) and ASVS V5 invalid-input rejections."
    requirement: MOD-03
    verification:
      - kind: unit
        ref: backend/tests/test_modeling_wrappers_and_age_stage.py#test_map_weight_to_age_stage_exact_boundaries
        status: pass
      - kind: unit
        ref: backend/tests/test_modeling_wrappers_and_age_stage.py#test_age_stage_module_uses_strictly_stdlib_imports
        status: pass
    human_judgment: false

duration: 11min
completed: 2026-11-23
status: complete
---

# Phase 03 Plan 01: Modeling Foundation, Wrappers, Ablation Dataset Loader & Age-Stage Mapper Summary

**Established the Phase 3 tabular modeling foundation: locked artifact path constants, pickling-safe `AllometricLogTransformer` + `LogDuanRegressor` wrappers, hash-locked zero-overlap 4-condition ablation dataset loader, 4-family classical regressor factory with 5x-capped fold weighting, and pure stdlib Cobb Avian 48 age-stage mapper.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-11-23T23:30:44Z
- **Completed:** 2026-11-23T23:41:14Z
- **Tasks:** 3 / 3 completed
- **Files modified:** 7 (1 modified, 6 created)

## Accomplishments
- Extended `backend/app/config.py` with `MODEL_BUNDLE_PATH`, `EVALUATION_REPORT_PATH`, and `PRESETS_CATALOG_PATH` while keeping all Phase 1–2 constants intact.
- Implemented `backend/app/modeling/wrappers.py` (`ALLOMETRIC_LOG_COLUMNS`, `AllometricLogTransformer`, `LogDuanRegressor`) with full ASVS V5 numeric/non-bool validation, fold-internal weighted/unweighted Duan's smearing, and clean-subprocess `joblib` round-trip verified at `2.158 ms` single-row inference with zero `UserWarning`s.
- Implemented `backend/app/modeling/datasets.py` (`load_modeling_dataset`, `build_ablation_feature_specs`, `LoadedModelingDataset`) verifying 1:1 row parity across `1,738` clean rows, `pdi_config_hash == 43a3cfaa3208019e`, `0` cross-fold `session_id` overlap, strict `C1`–`C4` feature column disjointness (`40`, `39`, `49`, `39` cols), offline `NAIVE_FXZ_5_COLUMNS` derivation, and complete removal of `rel_path` and `BANNED_FEATURE_COLUMNS`.
- Implemented `backend/app/modeling/models.py` (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog`, `MAX_N_JOBS_CAP = 4`, fold-only capped `compute_sample_weights`, strict column-ordering `predict_from_feature_dict`, and deterministic tie-broken `extract_feature_importances`).
- Implemented pure stdlib `backend/app/modeling/age_stage.py` (`map_weight_to_age_stage`, `AgeStageResult`) with exact 4-stage Cobb Avian 48 thresholds and `[116, 2093] g` training-range disclosure.
- Delivered `32` new Phase 3 tests (`27` in `test_modeling_wrappers_and_age_stage.py` + `5` in `test_modeling_datasets_and_ablation.py`); full repository suite passes `168 / 168` tests and `ruff check` exits `0` with zero diagnostics.

## Task Commits

Each task was committed atomically:

1. **Task 1 (tracer): Config paths, pickling-safe wrappers, dataset/ablation loader, and 4-family regressor factory** - `900041f` (feat)
2. **Task 2 (TDD): Cobb Avian 48 age-stage mapper and wrappers/weighting/pickling unit test suite** - `076a104` (feat)
3. **Task 3 (TDD): Dataset hygiene, 4-condition ablation contract, injected-leakage rejection, and PROIBIÇÃO 1/3 AST tests** - `f92dc8e` (test)

## Files Created/Modified
- `backend/app/config.py` - Added `MODEL_BUNDLE_PATH`, `EVALUATION_REPORT_PATH`, and `PRESETS_CATALOG_PATH` to module constants and `__all__`.
- `backend/app/modeling/wrappers.py` - Added `ALLOMETRIC_LOG_COLUMNS` (17 cols), `AllometricLogTransformer`, and `LogDuanRegressor` with ASVS V5 validation and fold-internal Duan smearing.
- `backend/app/modeling/datasets.py` - Added `LoadedModelingDataset`, `build_ablation_feature_specs()`, and `load_modeling_dataset()` with 1:1 alignment, `PDIConfig` hash lock, `verify_fold_integrity()`, `PROIBIÇÃO 2–4` guards, and offline `C4` naive `f*z` feature derivation.
- `backend/app/modeling/models.py` - Added `MODEL_FAMILY_NAMES`, `MAX_N_JOBS_CAP`, `DEFAULT_WEIGHTING_MODE`, `compute_sample_weights()`, `build_regressor_pipeline()`, `predict_from_feature_dict()`, and `extract_feature_importances()`.
- `backend/app/modeling/age_stage.py` - Added pure stdlib `AgeStageResult` and `map_weight_to_age_stage()` implementing `D-08` Cobb Avian 48 stages and `[116, 2093] g` out-of-range flag.
- `backend/tests/test_modeling_wrappers_and_age_stage.py` - 27 unit, boundary, validation, and clean-subprocess `joblib` pickling/latency tests.
- `backend/tests/test_modeling_datasets_and_ablation.py` - 5 dataset hygiene, 4-condition ablation disjointness, injected-leakage rejection, real Parquet verification, and `PROIBIÇÃO 1 & 3` AST scan tests (with reusable synthetic Parquet fixture factory for Plan `03-02`).

## Verification Results

- **Task 1 End-to-End Tracer (real 1,738-row Parquets + fold-0 LightGBMLog + clean-subprocess joblib.load):** `PHASE3_TRACER_OK 606.9814957171029 2.158` (`2.158 ms` single-row inference `< 5.0 ms`, bit-identical `pred_inproc == pred_sub`, zero `UserWarning`s under `-W error::UserWarning`).
- **Plan 03-01 Targeted Pytest Gate (`test_modeling_wrappers_and_age_stage.py`, `test_modeling_datasets_and_ablation.py`, `test_io_exif.py`):** `49 passed in 5.19s` (`0` warnings).
- **Full Backend Pytest Regression Suite:** `168 passed in 15.95s` (only 60 pre-existing upstream `skimage` NumPy 2.5 warnings in `test_pdi_watershed.py`).
- **Ruff Lint Gate (`backend/`):** `All checks passed!` (`0` diagnostics).

## Decisions Made
- Constructed single-row DataFrames in `predict_from_feature_dict()` from a pre-allocated `(1, n_features)` `float64` NumPy array (`pd.DataFrame(ordered_values, columns=ordered_names)`) instead of a list-of-dicts, reducing 1-row DataFrame construction overhead by `~6x` (`0.057 ms` vs `0.357 ms`) while preserving exact column names and ordering.

## Deviations from Plan

### Notes on Inline Tracer Verification Precision

**1. Tracer inline print precision vs. `1e-9` parity assertion**
- **Found during:** Task 1 (tracer verification)
- **Observation:** The plan's inline tracer snippet formatted the subprocess prediction as `{y:.6f}` (truncating at `1e-6` precision) and then asserted `abs(pred_inproc - pred_sub) < 1e-9`. Re-running with full-precision float output (`{y!r}`) confirmed exact bit-identical equality (`606.9814957171029 == 606.9814957171029`, difference `0.0`).
- **Impact:** Zero production code deviation; full-precision JSON serialization is used in `test_modeling_wrappers_and_age_stage.py`.

None — all production files, public APIs, security mitigations, and prohibitions were executed exactly as specified in `03-01-PLAN.md`.

## Issues Encountered
- Initial dict-order test asserted exact float `==` on `RandomForestLog` (`n_jobs=4`), exposing a `1.8e-12` (2-ULP) parallel-reduction rounding difference across 4 threads in `RandomForestRegressor.predict()`. Updated the test to assert `abs=1e-9` across all 4 families while retaining exact `==` for `RidgeLog`, `HuberLog`, and `LightGBMLog`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Ready for **Plan 03-02**: `backend/app/modeling/evaluation.py` can directly import `load_modeling_dataset`, `ABLATION_CONDITION_ORDER`, `MODEL_FAMILY_NAMES`, `build_regressor_pipeline`, `compute_sample_weights`, and `extract_feature_importances`, and extend `_write_synthetic_modeling_parquets` in `backend/tests/test_modeling_datasets_and_ablation.py` with counterfactual and anti-confounding gate tests.
- Ready for **Plan 03-03**: `MODEL_BUNDLE_PATH`, `EVALUATION_REPORT_PATH`, `PRESETS_CATALOG_PATH`, and `map_weight_to_age_stage()` are locked and verified.

## Self-Check: PASSED

- Verified all 7 created/modified files exist on disk (`backend/app/config.py`, `backend/app/modeling/wrappers.py`, `backend/app/modeling/datasets.py`, `backend/app/modeling/models.py`, `backend/app/modeling/age_stage.py`, `backend/tests/test_modeling_wrappers_and_age_stage.py`, `backend/tests/test_modeling_datasets_and_ablation.py`).
- Verified all 3 task commits exist in git history (`900041f`, `076a104`, `f92dc8e`).
- Verified `PROIBIÇÃO 1–4` compliance (zero DL/OCR imports, zero banned columns or splitters, zero raw JPEG filenames or filesystem path leaks, zero dependency or raw-data modifications).

---
*Phase: 03-tabular-regression-optical-ablation-production-bundling*
*Completed: 2026-11-23*
