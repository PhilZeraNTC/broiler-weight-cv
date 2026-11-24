---
phase: 03-tabular-regression-optical-ablation-production-bundling
plan: 03
subsystem: modeling
tags: [joblib, scikit-learn, lightgbm, watershed, cli, artifacts, anti-leakage, byte-determinism, asvs-v4]

requires:
  - phase: 03-tabular-regression-optical-ablation-production-bundling
    provides: "Locked Phase 3 artifact paths, pickling-safe AllometricLogTransformer & LogDuanRegressor wrappers, hash-locked 4-condition ablation dataset loader, 4-family regressor factory, Cobb Avian 48 age-stage mapper, and leak-free 5-fold OOF & dual-mode counterfactual evaluation engine from Plans 03-01 and 03-02."
  - phase: 02-classical-pdi-engine-optical-calibration-watershed
    provides: "Frozen PDIConfig (hash 43a3cfaa3208019e), run_watershed_pdi(), and materialized features_clean.parquet (1,738 x 60)."

provides:
  - "Atomic dual-bundle joblib packer/validator/loader (build_model_bundle, validate_model_bundle, save_model_bundle, load_model_bundle), byte-deterministic evaluation_report.json builder/writer, ASVS V4 non-symlink Watershed preset selector, and opaque-ID presets_catalog.json builder/writer in backend/app/modeling/artifacts.py (D-07, D-09, D-10, D-11)."
  - "Clean 21-symbol public modeling barrel with alphabetically sorted __all__ in backend/app/modeling/__init__.py."
  - "Single deterministic CLI orchestrator backend/scripts/train_and_evaluate.py (--dry-run, --n-jobs clamped 1..4, zero __main__ classes, aggregate-only stdout) executing end-to-end in ~37 s (D-11)."
  - "Materialized git-ignored dual-bundle models/weight_regressor.joblib (3,336,668 bytes; 100%-clean production_pipeline + 5 held-out fold_pipelines, verified in clean subprocess joblib.load() with -W error::UserWarning in 2.34 ms / 1.77 ms)."
  - "Materialized git-tracked byte-deterministic models/evaluation_report.json (SHA-256 b8d21b2dddac67609c4f8f610b8a422963fb80c19692c23f12f5869c6ffb0c5b) and models/presets_catalog.json (SHA-256 3a0c7e5005ae2e9d52b9e3baaf5cf8b93b2ab5dd6f4d80c767a6d60e290e1e29)."
  - "27 CLI, ASVS V4 symlink/traversal, PROIBIÇÃO 4 leak-guard, atomic write, synthetic two-run byte-determinism, real materialized artifact/subprocess-joblib, and git-hygiene integration tests in backend/tests/test_train_and_evaluate_integration.py."

affects:
  - 04-web-api-xray-pdi-studio

actuals:
  tokens: 24911
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Sibling NamedTemporaryFile + fsync + atomic os.replace() persistence with except cleanup across joblib and UTF-8 JSON writers"
    - "Pre-disk RFC 8259 strict JSON normalization (rejecting NaN/Inf, normalizing -0.0 to 0.0) and PROIBIÇÃO 4 forbidden-token scanning before directory or file creation"
    - "Exact 5-fold scarcest-pool-first bijection matching for single-bird OOF presets guaranteeing distinct folds {0..4}, distinct sessions, non-fallback masks, and realistic median-tier (>= 3.0%) errors including a same-session Wide/Ultrawide pair"
    - "ASVS V4 non-symlink segment-by-segment path containment before invoking run_watershed_pdi(..., return_visuals=False) over quarantined multi-bird images, emitting strictly opaque preset_id + image_id references"
    - "Zero-class CLI entrypoint importing exclusively from backend.app.modeling barrel to prevent __main__ joblib pickling hazards"

key-files:
  created:
    - backend/app/modeling/artifacts.py
    - backend/app/modeling/__init__.py
    - backend/scripts/train_and_evaluate.py
    - backend/tests/test_train_and_evaluate_integration.py
    - models/evaluation_report.json
    - models/presets_catalog.json
  modified: []

key-decisions:
  - "Used exact 5-fold permutation matching ordered scarcest-pool-first over all 5 fold-consuming single-bird preset slots (young_chick, intermediate_grower, heavy_broiler, hard_white_bucket, dual_lens_pair), guaranteeing zero greedy dead-ends while deterministically selecting representative median-tier OOF samples spanning all 5 folds and 5 distinct sessions."
  - "Enforced pre-write RFC 8259 non-finite float rejection and PROIBIÇÃO 4 forbidden-substring scanning strictly before creating parent directories or sibling temp files, ensuring rejected writes leave zero filesystem artifacts and never clobber existing valid targets."
  - "Kept wall_clock_sec strictly on CLI stdout and excluded volatile timestamps from both evaluation_report.json and presets_catalog.json so back-to-back runs over the real 1,738-image corpus produce bit-identical SHA-256 digests."

patterns-established:
  - "All Phase 3 persisted outputs (weight_regressor.joblib, evaluation_report.json, presets_catalog.json) commit atomically via sibling temp files and os.replace()."
  - "Phase 4 FastAPI services can consume models/weight_regressor.joblib via load_model_bundle(..., expected_pdi_config_hash=PDIConfig().config_hash()), serve models/evaluation_report.json verbatim on GET /api/metrics, and resolve opaque image_id keys from models/presets_catalog.json strictly at runtime."

requirements-completed:
  - MOD-01
  - MOD-02
  - MOD-03

coverage:
  - id: D1
    description: "Dual-bundle joblib packer, strict schema/hash/fold validator, atomic os.replace() persistence, and clean-subprocess < 5 ms loading verified on materialized models/weight_regressor.joblib."
    requirement: MOD-03
    verification:
      - kind: integration
        ref: backend/tests/test_train_and_evaluate_integration.py#test_real_materialized_model_bundle_subprocess_joblib_and_latency
        status: pass
      - kind: unit
        ref: backend/tests/test_train_and_evaluate_integration.py#test_load_model_bundle_rejects_corrupt_schema_hash_folds_and_features
        status: pass
    human_judgment: false
  - id: D2
    description: "Byte-deterministic zero-path-leakage models/evaluation_report.json exposing 16 model comparisons, 4-condition EXIF ablation, deltas_vs_c1, 127-session dual-lens consistency, dual-mode counterfactual audit, session_averaged_r2, stratified breakdowns, and 1,738 anonymized OOF scatter points."
    requirement: MOD-02
    verification:
      - kind: integration
        ref: backend/tests/test_train_and_evaluate_integration.py#test_real_materialized_evaluation_report_json_contract_and_zero_leakage
        status: pass
      - kind: integration
        ref: backend/tests/test_train_and_evaluate_integration.py#test_cli_two_run_byte_identical_sha256_and_subprocess_joblib_load
        status: pass
    human_judgment: false
  - id: D3
    description: "Byte-deterministic opaque-ID models/presets_catalog.json containing 6 non-fallback median-tier single-bird OOF presets spanning all 5 folds (including same-session Wide/Ultrawide pair verified against live fold_pipelines) and 4 ASVS V4-verified multi-bird Watershed presets."
    requirement: MOD-03
    verification:
      - kind: integration
        ref: backend/tests/test_train_and_evaluate_integration.py#test_real_materialized_presets_catalog_json_contract_and_live_oof_parity
        status: pass
      - kind: unit
        ref: backend/tests/test_train_and_evaluate_integration.py#test_build_presets_catalog_rejects_empty_single_bird_and_watershed_pools
        status: pass
    human_judgment: false
  - id: D4
    description: "Single deterministic CLI train_and_evaluate.py with --dry-run, clamped --n-jobs (1..4), aggregate-only stdout, ASVS V4 non-symlink quarantine containment, PROIBIÇÃO 1-4 AST/leak guards, and clean git hygiene."
    requirement: MOD-01
    verification:
      - kind: integration
        ref: backend/tests/test_train_and_evaluate_integration.py#test_cli_dry_run_writes_zero_files_and_emits_aggregate_only_stdout
        status: pass
      - kind: unit
        ref: backend/tests/test_train_and_evaluate_integration.py#test_asvs_v4_build_presets_catalog_rejects_malicious_quarantine_manifest_rows
        status: pass
      - kind: unit
        ref: backend/tests/test_train_and_evaluate_integration.py#test_ast_scan_train_and_evaluate_prohibitions_and_no_main_classes
        status: pass
      - kind: integration
        ref: backend/tests/test_train_and_evaluate_integration.py#test_git_hygiene_ignores_joblib_tracks_phase3_json_and_excludes_raw_processed
        status: pass
    human_judgment: false

duration: 14min
completed: 2026-11-24
status: complete
---

# Phase 03 Plan 03: Production Artifact Bundling, Deterministic CLI & Materialized Phase 3 Gates Summary

**Delivered atomic dual-bundle model publishing, byte-deterministic evaluation reporting, opaque-ID OOF & Watershed preset selection, the single reproducible training CLI, and materialized verified Phase 3 artifacts ready for Phase 4 FastAPI serving.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-11-23T23:56:22Z
- **Completed:** 2026-11-24T00:10:02Z
- **Tasks:** 3 / 3 completed
- **Files created/modified:** 6 tracked files created (+ 1 git-ignored runtime dual bundle materialized locally)

## Accomplishments
- Implemented `backend/app/modeling/artifacts.py` (`1,111` lines) providing:
  - Schema-versioned dual-bundle builder, strict validator, and atomic joblib reader/writer (`MODEL_BUNDLE_SCHEMA_VERSION = "1.0.0"`, `MODEL_VERSION = "avianscale-cv-v1.0.0"`), refitting `production_pipeline` on 100% of clean images while preserving all 5 held-out `fold_pipelines` (`{0, 1, 2, 3, 4}`), locked `PDIConfig` + hash `43a3cfaa3208019e`, 39-column `feature_names` tuple, per-band error intervals, and Cobb Avian 48 coverage bounds.
  - Byte-deterministic `build_evaluation_report()` and `write_evaluation_report()` enforcing strict RFC 8259 JSON normalization, pre-write `PROIBIÇÃO 4` forbidden-token rejection, and atomic sibling temp-file replacement.
  - Deterministic `build_presets_catalog()` and `write_presets_catalog()` using exact 5-fold scarcest-pool-first matching for 6 non-fallback median-tier (`oof_pct_err >= 3.0%`) single-bird OOF presets across distinct folds/sessions (including same-session `541__s0` Wide `383.22 g` vs. Ultrawide `384.16 g` demonstrating `< 1 g` cross-lens consistency) plus 4 ASVS V4 non-symlink-validated multi-bird Watershed presets (`ws_417`, `ws_757`, `ws_823`, `ws_1520`) storing strictly opaque identifiers.
- Created public modeling barrel `backend/app/modeling/__init__.py` exporting all 21 public Phase 3 symbols with alphabetically sorted `__all__`.
- Created single deterministic CLI `backend/scripts/train_and_evaluate.py` (`--dry-run`, `--n-jobs` clamped `1..4`, zero custom classes in `__main__`, aggregate-only stdout).
- Executed the production CLI end-to-end twice over the real 1,738-image clean and 15-image quarantine corpus (`36.96 s` / `37.23 s`, both `< 60 s`), proving **exact consecutive-run byte-identical SHA-256 parity**:
  - `models/evaluation_report.json`: `b8d21b2dddac67609c4f8f610b8a422963fb80c19692c23f12f5869c6ffb0c5b` (`624,113` bytes; `LightGBMLog / C2_physics_normalized_strict`: `BalMAPE = 35.1129%`, `MAE = 158.1076 g`, `RMSE = 264.0242 g`, `R² = 0.5582`, `session_averaged_r2 = 0.6905`; `C2` strictly beats `C1` by `-6.1613 g` MAE / `-11.1664 g` RMSE / `-1.0810 pp` BalMAPE / `+0.0381` R² with exact `0.0%` counterfactual shift while `C3` and `C4` are flagged and disqualified).
  - `models/presets_catalog.json`: `3a0c7e5005ae2e9d52b9e3baaf5cf8b93b2ab5dd6f4d80c767a6d60e290e1e29` (`5,636` bytes; 6 OOF single-bird + 4 Watershed presets, live `fold_pipelines` OOF parity verified within `0.05 g`).
  - `models/weight_regressor.joblib` (git-ignored): `3,336,668` bytes (`3.18 MiB`), cold load `84.33 ms`, single-row prediction `2.343 ms` (`production_pipeline`) and `1.765 ms` (`fold_pipelines[0]`) under `-W error::UserWarning` in a clean subprocess.
- Passed the entire repository verification gate: **202 / 202 pytest tests passed** across Phases 1, 2, and 3 (`27` new tests in `backend/tests/test_train_and_evaluate_integration.py`) and **0 ruff diagnostics**.

## Task Commits

Each task was committed atomically:

1. **Task 1: Dual-bundle artifacts publisher, deterministic JSON writers, and modeling barrel** - `8cbd7b1` (feat)
2. **Task 2: Deterministic train_and_evaluate CLI and synthetic security/determinism integration tests** - `36fcd07` (feat)
3. **Task 3: Materialized evaluation report, presets catalog, and real-corpus artifact & git-hygiene gates** - `c130aec` (feat)

## Files Created/Modified
- `backend/app/modeling/artifacts.py` - Dual-bundle joblib builder/validator/loader, atomic persistence, RFC 8259 byte-deterministic JSON writers with pre-disk `PROIBIÇÃO 4` leak checks, exact 5-fold single-bird OOF preset selector, and ASVS V4 non-symlink Watershed preset selector.
- `backend/app/modeling/__init__.py` - 21-symbol public modeling package barrel with alphabetically sorted `__all__`.
- `backend/scripts/train_and_evaluate.py` - Single zero-class deterministic Phase 3 CLI with `--dry-run`, bounded `--n-jobs <= 4`, and aggregate-only stdout.
- `backend/tests/test_train_and_evaluate_integration.py` - 27 synthetic CLI, ASVS V4 traversal/symlink, `PROIBIÇÃO 4` leak-guard, atomic write, corruption rejection, empty-pool rejection, AST scan, real materialized artifact/subprocess-joblib, and git-hygiene integration tests.
- `models/evaluation_report.json` - Git-tracked, byte-deterministic, zero-path-leakage 5-fold OOF evaluation and 4-condition EXIF ablation report for Phase 4 `GET /api/metrics` and Tab 3 (`WEB-05`).
- `models/presets_catalog.json` - Git-tracked, byte-deterministic, opaque-ID curated 6 single-bird OOF + 4 multi-bird Watershed preset catalog for Phase 4 `GET /api/presets`, `WEB-03`, and `WEB-04`.

## Decisions Made
- Applied exact 5-fold permutation matching ordered scarcest-pool-first for the 5 single-bird preset slot groups so rare strata (`sb_hard_white_bucket`) never hit greedy fold conflicts while preserving full determinism.
- Rejected non-finite floats and forbidden path/filename markers prior to creating directories or temp files, guaranteeing zero residual filesystem state on validation errors.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Initial ruff lint pass flagged 4 minor formatting diagnostics in `artifacts.py` (two 101-char lines, one unused variable, one `SIM108` ternary) and 1 import-sort order diagnostic (`I001`) in `test_train_and_evaluate_integration.py`; all resolved immediately with zero remaining diagnostics.
- `tempfile.NamedTemporaryFile` creates files with `0600` mode; normalized permissions on git-tracked `models/evaluation_report.json` and `models/presets_catalog.json` to `0644` matching Phase 1–2 tracked audit JSONs while preserving exact SHA-256 byte identity.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 3 is 100% complete and verified across all requirements (`MOD-01`, `MOD-02`, `MOD-03`):
  - Phase 4 FastAPI `lifespan` can load `models/weight_regressor.joblib` via `backend.app.modeling.load_model_bundle()` with hash lock `43a3cfaa3208019e`, routing uploads to `production_pipeline`, curated presets to `fold_pipelines[fold]`, and weights to `map_weight_to_age_stage()`.
  - Phase 4 `GET /api/metrics` and `GET /api/presets` can directly serve `models/evaluation_report.json` and `models/presets_catalog.json` with zero path-leakage risk.

## Self-Check: PASSED

- Verified all 6 git-tracked created files exist on disk (`backend/app/modeling/artifacts.py`, `backend/app/modeling/__init__.py`, `backend/scripts/train_and_evaluate.py`, `backend/tests/test_train_and_evaluate_integration.py`, `models/evaluation_report.json`, `models/presets_catalog.json`) plus local git-ignored `models/weight_regressor.joblib`.
- Verified all 3 task commits exist in git history (`8cbd7b1`, `36fcd07`, `c130aec`).
- Verified consecutive-run SHA-256 byte-identity on real `models/evaluation_report.json` (`b8d21b2dddac67609c4f8f610b8a422963fb80c19692c23f12f5869c6ffb0c5b`) and `models/presets_catalog.json` (`3a0c7e5005ae2e9d52b9e3baaf5cf8b93b2ab5dd6f4d80c767a6d60e290e1e29`).
- Verified full repository pytest (`202 passed`) and ruff (`All checks passed!`) gates, `PROIBIÇÃO 1–4` compliance, ASVS V4 non-symlink containment, and clean git hygiene.

---
*Phase: 03-tabular-regression-optical-ablation-production-bundling*
*Completed: 2026-11-24*
