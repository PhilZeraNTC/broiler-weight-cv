---
phase: 03-tabular-regression-optical-ablation-production-bundling
verified: 2026-11-24T00:32:00Z
status: passed
score: "16/16"
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/03-tabular-regression-optical-ablation-production-bundling/03-01-PLAN.md"
  - ".planning/phases/03-tabular-regression-optical-ablation-production-bundling/03-01-SUMMARY.md"
  - ".planning/phases/03-tabular-regression-optical-ablation-production-bundling/03-02-PLAN.md"
  - ".planning/phases/03-tabular-regression-optical-ablation-production-bundling/03-02-SUMMARY.md"
  - ".planning/phases/03-tabular-regression-optical-ablation-production-bundling/03-03-PLAN.md"
  - ".planning/phases/03-tabular-regression-optical-ablation-production-bundling/03-03-SUMMARY.md"
  - ".planning/phases/03-tabular-regression-optical-ablation-production-bundling/03-REVIEW.md"
  - "backend/app/config.py"
  - "backend/app/modeling/__init__.py"
  - "backend/app/modeling/age_stage.py"
  - "backend/app/modeling/artifacts.py"
  - "backend/app/modeling/datasets.py"
  - "backend/app/modeling/evaluation.py"
  - "backend/app/modeling/models.py"
  - "backend/app/modeling/wrappers.py"
  - "backend/scripts/train_and_evaluate.py"
  - "backend/tests/test_modeling_datasets_and_ablation.py"
  - "backend/tests/test_modeling_wrappers_and_age_stage.py"
  - "backend/tests/test_train_and_evaluate_integration.py"
  - "models/evaluation_report.json"
  - "models/presets_catalog.json"
covered_digest: "v1:sha256:e111a66385816c70baad9ad598d57f4ca9639f5616321b7dd31ac3f0891bd2c3"
behavior_unverified: 0
overrides_applied: 0
overrides: []
gaps: []
---

# Phase 3: Tabular Regression, Optical Ablation & Production Bundling — Verification Report

**Phase Goal:** Train and rigorously audit interpretable tabular weight regressors over the locked 5-fold `session_id` splits, quantitatively prove the impact of EXIF optical calibration without lens-confounding shortcuts, and publish immutable production and out-of-fold pitch artifacts.
**Verified:** 2026-11-24T00:32:00Z (at post-review hardened HEAD `1fc10c41ec926b3860d17102aa12d74c989a7d81`; milestone closeout fingerprint refreshed 2026-11-24 at `16eb74e`)
**Status:** `passed`
**Re-verification:** Yes — digest refreshed at v1.0 closeout (`16eb74e`) after shared `REQUIREMENTS.md`/`config.py` evolved in Phase 4; full regression suite passed (`272/272` pytest incl. all 69 Phase 3 tests, `ruff check backend/` clean)

## Executive Summary

Adversarial goal-backward verification directly against the Phase 3 codebase (`backend/app/config.py`, `backend/app/modeling/*`, `backend/scripts/train_and_evaluate.py`), all three Phase 3 test suites (`69` new tests), git-tracked `models/evaluation_report.json` and `models/presets_catalog.json`, local git-ignored `models/weight_regressor.joblib`, git index, AST scans, and live subprocess/runtime audits confirms **16/16 merged observable truths VERIFIED**, **14/14 unique PLAN artifacts VERIFIED** (`15/15` across `03-01`..`03-03`), **8/8 PLAN key links WIRED**, **3/3 requirements SATISFIED (`MOD-01`, `MOD-02`, `MOD-03`)**, **11/11 locked decisions (`D-01`..`D-11`) VERIFIED**, **11/11 `03-REVIEW.md` findings (`CR-01`, `WR-01`..`WR-06`, `IN-01`..`IN-04`) RESOLVED**, **0 gaps**, **0 stub artifacts**, **0 debt markers**, and **205/205 automated tests passing** (`uv run pytest -v`, `61.96s`; `uv run ruff check backend/` → `All checks passed!`).

Independent live audits verified directly on disk:
- **Post-Review Hardening (`1fc10c4`):** Confirmed `models/weight_regressor.joblib` (`3,336,661` bytes) was regenerated after `CR-01`: `bundle["pdi_config"]["clahe_tile_grid"]` deserializes as `tuple` `(8, 8)`, `PDIConfig(**bundle["pdi_config"]) == PDIConfig()`, and `validate_model_bundle()` reconstructs and hash-verifies `PDIConfig(**pdi_cfg).config_hash() == bundle["pdi_config_hash"]` (`43a3cfaa3208019e`). Independently triggered and verified `WR-01` (swapped DataFrame columns + stale `feature_names_in_`), `WR-02` (`y_true <= 0`), `WR-03` (`np.complex128`, `np.bool_`), `WR-04` (`np.int32`/`np.int64`/`np.float32` acceptance in `age_stage.py`), `WR-05` (winner-condition mechanics scoping), `WR-06` (`C2` `mean`/`p90` zero-shift gate), and `IN-01`..`IN-04`.
- **Clean-Subprocess Dual-Bundle SLA (`D-07`):** Executed `load_model_bundle()` from an isolated `cwd=tmp_path` with `PYTHONPATH=REPO_ROOT` and `-W error::UserWarning` (`0` warnings): cold bundle load `75.245 ms`, single-row prediction `p50 = 2.430 ms` / `max = 3.566 ms` (`production_pipeline`) and `p50 = 2.398 ms` / `max = 4.152 ms` (`fold_pipelines[0]`) — well under the `< 5.0 ms` SLA.
- **Zero Code-vs-Artifact OOF Skew:** Live inference of `bundle["fold_pipelines"][f]` against `data/processed/features_clean.parquet` matched **all `1,738` `oof_scatter_points` in `models/evaluation_report.json` with exact `0.0` max diff** and **all `6` `single_bird_presets` in `models/presets_catalog.json` with exact `0.0` max diff**.
- **EXIF Optical Calibration Gain & Anti-Confounding Gate (`D-04`, `D-05`, `D-06`):** `LightGBMLog / C2_physics_normalized_strict` (`39` features, `session_band_capped_5x`, Duan smearing OFF) wins deterministically (`BalMAPE = 35.1129%`, `MAE = 158.1076 g`, `RMSE = 264.0242 g`, `MAPE = 37.0876%`, `R² = 0.5582`, `session_averaged_r2 = 0.6905`, `127` mixed-lens sessions median `|Δ| = 66.1006 g`), beating `C1_uncalibrated_pdi` by **`-6.1613 g` MAE / `-11.1664 g` RMSE / `-1.0810 pp` BalMAPE / `+0.0381` R² / `+0.0327` session R²**, with **exact `0.0%` shift** across `focal_only` and `physical_lens_swap` (`median`/`mean`/`p90`), while **all 8 `C3` and `C4` candidates** are automatically flagged `lens_confounding_detected = True` and disqualified from production.

---

## Goal Achievement

### User Flow Coverage (MVP Mode — Plan User Stories `03-01`..`03-03`)

| Plan | User Story Outcome | Codebase Evidence | Status |
| --- | --- | --- | --- |
| `03-01` | Locked Phase 3 artifact config paths, pickling-safe `AllometricLogTransformer` + `LogDuanRegressor`, hash-locked zero-overlap 4-condition ablation dataset loader, 4-family classical regressor factory (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog`) with 5×-capped fold weighting, and pure stdlib Cobb Avian 48 age-stage mapper. | `backend/app/config.py`, `backend/app/modeling/{wrappers,datasets,models,age_stage}.py`; `41/41` passing tests across `test_modeling_wrappers_and_age_stage.py` (`28`) + `test_modeling_datasets_and_ablation.py` (`13`). | ✓ VERIFIED |
| `03-02` | Leak-free 5-fold `session_id` OOF evaluation across 4 families × 4 EXIF ablation conditions, `session_averaged_r2`, stratified slicing, 127-session dual-lens consistency, dual-mode counterfactual anti-confounding gate, mechanics/allometric sensitivity sweeps, and deterministic `D-06` winner selector. | `backend/app/modeling/evaluation.py` (`1,231` lines), `test_modeling_datasets_and_ablation.py` (`13/13` passing including synthetic confounding disqualification, strict inequality boundaries, empty/single/constant guards, and permutation-invariant tie-breaks). | ✓ VERIFIED |
| `03-03` | Single deterministic CLI (`train_and_evaluate.py`) atomically publishing git-ignored dual-bundle `models/weight_regressor.joblib`, byte-deterministic zero-path-leakage `models/evaluation_report.json`, and opaque-ID `models/presets_catalog.json` ready for Phase 4 FastAPI serving. | `backend/app/modeling/{artifacts,__init__}.py`, `backend/scripts/train_and_evaluate.py`, `models/{evaluation_report,presets_catalog}.json`, local `models/weight_regressor.joblib`; `28/28` passing tests in `test_train_and_evaluate_integration.py` + live subprocess & `0.0` OOF parity checks. | ✓ VERIFIED |

### Observable Truths (ROADMAP Success Criteria + Merged PLAN Must-Haves)

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | **03-01 T1 (`D-11` Config Path Anchoring):** `backend/app/config.py` exports repo-root-anchored `MODEL_BUNDLE_PATH`, `EVALUATION_REPORT_PATH`, and `PRESETS_CATALOG_PATH` alongside all Phase 1–2 constants. | ✓ VERIFIED | Direct read of `backend/app/config.py:22-40` (`13` exports in `__all__`). Verified by `test_config_paths_are_repo_root_anchored_and_cwd_independent` (`test_io_exif.py`) and `test_parse_args_defaults_n_jobs_clamping_and_invalid_rejection` (PASS). |
| 2 | **ROADMAP SC-1 / 03-01 T2 (`MOD-01`, `D-01`, `D-02`, `WR-01` Pickling-Safe Estimators):** `wrappers.py` defines `ALLOMETRIC_LOG_COLUMNS` (`17` geometric cols), `AllometricLogTransformer` (column-name/order validation + stale `feature_names_in_` cleanup per `WR-01`), and `LogDuanRegressor` (`log1p`/`expm1`, fold-internal weighted/unweighted Duan's smearing, zero `lambda`/`__main__`) round-tripping through clean-subprocess `joblib.load()` in `< 5 ms` with `-W error::UserWarning`. | ✓ VERIFIED | Direct read of `backend/app/modeling/wrappers.py:21-209`; live test confirmed swapped DataFrame columns raise `ValueError` and ndarray refit clears `feature_names_in_`. Verified by `test_allometric_log_transformer_targets_geometric_cols_and_clips_negatives`, `test_log_duan_regressor_roundtrip_smearing_weights_and_validation`, `test_all_four_model_families_fresh_subprocess_joblib_roundtrip_and_latency`, and `test_review_hardening_wrappers_complex_and_numpy_age_scalars` (all PASS). |
| 3 | **ROADMAP SC-1 / 03-01 T3 (`MOD-01`, `D-02`, `D-04`, `D-11`, `IN-04` Leak-Free Dataset Loader & 4-Condition Specs):** `datasets.py` joins `manifest_clean.parquet` (`1,738 × 24`) and `features_clean.parquet` (`1,738 × 60`) with exact 1:1 parity, locks `pdi_config_hash == "43a3cfaa3208019e"`, runs `verify_fold_integrity()` (`0` session overlap, `>= 4` heavy sessions/fold), checks NaNs across all audit/feature columns (`IN-04`), strips `rel_path`, bans `BANNED_FEATURE_COLUMNS` & `NON_FEATURE_AUDIT_COLUMNS`, and builds exact `C1` (`40`), `C2` (`39`, disjoint from raw EXIF/pixel cols), `C3` (`49`), and `C4` (`39`, offline `area_px / (f*z)²`) specs. | ✓ VERIFIED | Direct read of `backend/app/modeling/datasets.py:33-383`; live load confirmed `1,738` rows, `0` cross-fold overlap, `pdi_config_hash == "43a3cfaa3208019e"`, zero `rel_path`. Verified by `test_ablation_feature_specs_counts_disjointness_and_hygiene`, `test_load_modeling_dataset_synthetic_valid_and_naive_fxz_math`, `test_load_modeling_dataset_rejects_mismatched_hash_misalignment_nan_banned_and_leakage`, `test_load_modeling_dataset_materialized_parquets_contract`, and `test_review_hardening_nonpositive_metrics_c2_p90_gate_and_audit_nan` (all PASS). |
| 4 | **ROADMAP SC-1 / 03-01 T4 & T7 (`MOD-01`, `D-01`..`D-03`, `WR-03` 4-Family Classical Regressor Factory & Weighting):** `models.py` exposes `RidgeLog`, `HuberLog`, `RandomForestLog`, and `LightGBMLog` (zero deep learning, `random_state=42`, `MAX_N_JOBS_CAP = 4`, upstream `SimpleImputer`), fold-only `compute_sample_weights()` (`max/min <= 5.0`, `mean == 1.0`, empty/single guards), `predict_from_feature_dict()` (strict column ordering; rejects missing/`bool`/`np.bool_`/`np.complex*`/non-finite per `WR-03`), and deterministic tie-broken `extract_feature_importances()`. | ✓ VERIFIED | Direct read of `backend/app/modeling/models.py:28-302`; live test verified rejection of `np.complex128(1+2j)`, `np.bool_(True)`, and `True`. Covered by `test_compute_sample_weights_cap_normalization_and_validation`, `test_all_four_families_fit_predict_ordering_importances_and_validation`, `test_extract_feature_importances_deterministic_tie_break_and_zero_sum`, and `test_review_hardening_wrappers_complex_and_numpy_age_scalars` (all PASS). |
| 5 | **03-01 T5 & T6 (`MOD-03`, `D-08`, `WR-04` Pure Stdlib Cobb Avian 48 Age-Stage Mapper):** `age_stage.py` uses strictly stdlib imports (`math`, `numbers`, `dataclasses`), maps exact 4-stage boundary adjacencies (`115.99`/`116.0`/`349.99`/`350.0`/`700.0`/`700.01`/`1200.0`/`1200.01`/`2093.0`/`2093.01`), accepts NumPy real scalars (`np.int32`/`np.int64`/`np.float32`, `WR-04`), and rejects `bool`/`np.bool_`/`complex`/`NaN`/`Inf`/`<= 0`. | ✓ VERIFIED | Direct read of `backend/app/modeling/age_stage.py:12-86`; live check verified `np.int32(500)`, `np.int64(1500)`, `np.float32(200.5)` pass and `np.bool_(True)`/`np.complex128`/`0`/`-10` raise `ValueError`. Covered by `test_age_stage_module_uses_strictly_stdlib_imports`, `test_map_weight_to_age_stage_exact_boundaries` (10 parametrized cases), `test_map_weight_to_age_stage_rejects_invalid_inputs` (10 cases), and `test_review_hardening_wrappers_complex_and_numpy_age_scalars` (all PASS). |
| 6 | **ROADMAP SC-1 & SC-2 / 03-02 T1 (`MOD-01`, `MOD-02`, `WR-02` Leak-Free 5-Fold CV Across 16 Candidates):** `evaluation.py` executes strictly fold-isolated 5-fold `session_id` OOF CV across all 4 families × 4 conditions (`80` fold fits), keeping preprocessing, capped `sample_weight`, and Duan's smearing strictly inside `fold != k`, scoring strictly unweighted gram metrics, and rejecting `y_true <= 0` (`WR-02`). | ✓ VERIFIED | Direct read of `backend/app/modeling/evaluation.py:194-245, 719-839`; live check confirmed `compute_regression_metrics([0.0], [100.0])` and `[-50.0, 500.0]` raise `ValueError`. All 16 candidates verified present in `models/evaluation_report.json`. Covered by `test_compute_regression_metrics_normal_empty_single_constant_and_validation` and `test_review_hardening_nonpositive_metrics_c2_p90_gate_and_audit_nan` (all PASS). |
| 7 | **ROADMAP SC-2 / 03-02 T2, T6, T7 (`MOD-02`, `D-02`, `D-09` Session `R²`, Stratified Slicing & Edge Guards):** `compute_session_level_metrics()` exposes `session_averaged_r2` (`0.6905` vs frame `0.5582`) inside both `winning_model.global` and `session_level`; `compute_stratified_metrics()` computes unweighted `per_fold`, `by_weight_band` (aligned with `assign_weight_band()` at `499`/`500`/`1000`/`1001`), `by_lens`, `by_scene`, `by_band_and_scene`, `by_band_and_lens`, `band_error_intervals`, and `balanced_band_mape_pct`, handling empty (`n == 0`), single (`n == 1`), and zero-variance slices cleanly. | ✓ VERIFIED | Direct read of `backend/app/modeling/evaluation.py:127-136, 248-413, 794-798`; verified in materialized `models/evaluation_report.json` (`global.r2 = 0.5582`, `global.session_averaged_r2 = 0.6905`, `session_level.n_sessions = 188`). Covered by `test_compute_session_level_metrics_improves_r2_on_frame_noise_and_handles_empty` and `test_compute_stratified_metrics_weight_band_adjacency_and_empty_slices` (all PASS). |
| 8 | **ROADMAP SC-2 / 03-02 T3 (`MOD-02`, `D-05` Within-Session Dual-Lens Consistency):** `compute_within_session_dual_lens_consistency()` pairs `wide` vs `ultrawide` OOF predictions across all `127` mixed-lens sessions via sorted `session_id` groups, returning `0`/`None` safely when no mixed-lens session exists. | ✓ VERIFIED | Direct read of `backend/app/modeling/evaluation.py:415-478`; materialized `evaluation_report.json` confirms `n_mixed_lens_sessions = 127` (`C2`: median `|Δ| = 66.1006 g`, `19.4206%`, mean `|Δ| = 118.9414 g`). Covered by `test_compute_within_session_dual_lens_consistency_mixed_and_single_lens` (PASS). |
| 9 | **ROADMAP SC-2 / 03-02 T4 (`MOD-02`, `D-05`, `WR-06`, `IN-03` Dual-Mode Counterfactual Audit & Anti-Confounding Gate):** `compute_counterfactual_lens_flip()` executes `focal_only` and `physical_lens_swap` (`4.25/1.0 <-> 1.65/0.6`, sensor ratio `1.5625`, CoW-safe `.copy()` + `.loc[...] = ...`, positive finite optics guard per `IN-03`); `evaluate_anti_confounding_gate()` asserts strict `0.0%` invariance across `median`, `mean`, AND `p90` on `C2` (`WR-06`), applies strict `>` thresholds (`10.0%`/`15.0%`), and flags all 8 `C3`/`C4` candidates as confounded. | ✓ VERIFIED | Direct read of `backend/app/modeling/evaluation.py:481-717`; live check confirmed `C2` with `median=0.0, p90=5.0` raises `RuntimeError`; materialized report shows `C2` exact `0.0%` across all stats while `LightGBMLog/C3` (`phys p90 = 23.5244% > 15%`) and `LightGBMLog/C4` (`phys med = 12.7381%`, `p90 = 39.4276%`, `<500g UW` degraded vs `C2`) are flagged & disqualified. Covered by `test_evaluate_anti_confounding_gate_exact_strict_inequality_boundaries`, `test_synthetic_lens_confounded_dataset_flags_c3_and_c4_and_selects_c2`, and `test_review_hardening_nonpositive_metrics_c2_p90_gate_and_audit_nan` (all PASS). |
| 10 | **03-02 T5 & T8 (`MOD-01`, `MOD-02`, `D-01`..`D-04`, `D-06`, `WR-05` Mechanics/Allometric Sweeps & Winner Selection):** `run_full_evaluation()` executes the `2×2` weighting × smearing sweep on `winner.ablation_condition` (`WR-05`), linear `AllometricLogTransformer` ON/OFF ablation, explicit `deltas_vs_c1`, and permutation-invariant `select_winning_candidate()` deterministically picking `LightGBMLog / C2_physics_normalized_strict`. | ✓ VERIFIED | Direct read of `backend/app/modeling/evaluation.py:842-866, 1063-1208`; materialized report confirms `session_band_capped_5x` + `duan=False` (`BalMAPE = 35.1129%`) beats unweighted (`35.7632%`), unweighted+Duan (`35.7496%`), and weighted+Duan (`35.1330%`), and `is_selected` checks `ablation_condition`, `weighting_mode`, and `duan_smearing_enabled`. Covered by `test_select_winning_candidate_deterministic_tie_breaks_and_empty_eligible` (PASS). |
| 11 | **03-03 T1 & T6 (`MOD-03`, `D-07`, `D-09`..`D-11`, `CR-01`, `IN-01`, `IN-02` Atomic Artifact Builders & Corruption Guards):** `artifacts.py` commits all 3 outputs atomically via sibling temp-file + `os.replace()`, preserves tuple `pdi_config` via `asdict()` and reconstructs + hash-verifies `PDIConfig(**pdi_cfg)` inside `validate_model_bundle()` (`CR-01`), validates `weighting_mode` explicitly (`IN-01`), aligns float weight-band cutoffs (`IN-02`), and rejects corrupt bundles or empty preset pools with `ValueError`. | ✓ VERIFIED | Direct read of `backend/app/modeling/artifacts.py:207-521, 828-831`; covered by `test_atomic_write_cleans_up_temp_file_and_preserves_target_on_replace_failure`, `test_load_model_bundle_rejects_corrupt_schema_hash_folds_and_features`, `test_build_presets_catalog_rejects_empty_single_bird_and_watershed_pools`, and `test_real_model_bundle_pdi_config_reconstructs_cleanly` (all PASS). |
| 12 | **ROADMAP SC-3 / 03-03 T2 (`MOD-03`, `D-07` Materialized Dual-Bundle `models/weight_regressor.joblib`):** Materialized git-ignored bundle (`3,336,661` bytes) passes clean-subprocess `load_model_bundle()` (`cwd=tmp_path`, `-W error::UserWarning`) with `PDIConfig(**bundle["pdi_config"]) == PDIConfig()`, `pdi_config_hash == "43a3cfaa3208019e"`, 39-col `feature_names` tuple, 100%-clean `production_pipeline`, all 5 held-out `fold_pipelines` `{0..4}`, `band_error_intervals`, `[116, 2093] g` / `"~7-29 days"`, `< 5.0 ms` single-row inference, and `0.0` OOF parity across all `1,738` rows. | ✓ VERIFIED | Live subprocess audit: `load_ms = 75.245 ms`, `prod_p50_ms = 2.430 ms` (`max = 3.566 ms`), `fold0_p50_ms = 2.398 ms` (`max = 4.152 ms`), `isinstance(clahe_tile_grid, tuple) == True`, `ALL_1738_SCATTER_LIVE_OOF_MAX_DIFF = 0.0`, `PRESETS_LIVE_OOF_MAX_DIFF = 0.0`. Covered by `test_real_materialized_model_bundle_subprocess_joblib_and_latency` and `test_real_model_bundle_pdi_config_reconstructs_cleanly` (all PASS). |
| 13 | **ROADMAP SC-2 / 03-03 T3 (`MOD-02`, `D-09` Materialized Git-Tracked `models/evaluation_report.json`):** Byte-deterministic UTF-8 JSON (`624,113` bytes, SHA-256 `b8d21b2dddac67609c4f8f610b8a422963fb80c19692c23f12f5869c6ffb0c5b`), zero forbidden leak substrings, complete `D-09` contract (`16` model rows, `deltas_vs_c1` proving `C2` beats `C1`, `127` mixed-lens sessions, `C2` `0.0%` vs `C3`/`C4` flagged, `39` feature importances summing to `99.9997%`, `1,738` sorted anonymized scatter points). | ✓ VERIFIED | Live programmatic inspection of `models/evaluation_report.json` confirmed every field, zero forbidden substrings, exact `C2` vs `C1` gains (`-6.1613 g` MAE, `-11.1664 g` RMSE, `-1.0810 pp` BalMAPE, `+0.0381` R²). Covered by `test_real_materialized_evaluation_report_json_contract_and_zero_leakage` and `test_cli_two_run_byte_identical_sha256_and_subprocess_joblib_load` (all PASS). |
| 14 | **ROADMAP SC-4 / 03-03 T4 (`MOD-03`, `D-10` Materialized Git-Tracked `models/presets_catalog.json`):** Byte-deterministic UTF-8 JSON (`5,636` bytes, SHA-256 `3a0c7e5005ae2e9d52b9e3baaf5cf8b93b2ab5dd6f4d80c767a6d60e290e1e29`), strictly opaque `preset_id` + `image_id` ONLY (zero `.jpg`/`rel_path`/`/home/`/`data/raw`), `6` non-fallback median-tier (`>= 3.0%`) single-bird OOF presets spanning all 5 folds `{0..4}` & 5 distinct sessions (`180 g`, `576 g`, `1,555 g`, `1,986 g` `bucket_dorsal`, same-session `541__s0` Wide `383.22 g` vs Ultrawide `384.16 g`) with `0.0` live `fold_pipelines` diff, plus `4` ASVS V4-verified multi-bird Watershed presets (`ws_417`: 2/2, `ws_757`: 3/2, `ws_823`: 4/4, `ws_1520`: 2/2) carrying `group_sum_only` disclaimer. | ✓ VERIFIED | Live programmatic inspection of `models/presets_catalog.json` + quarantine manifest ID check + live `fold_pipelines` OOF check (`max_diff = 0.0`). Covered by `test_real_materialized_presets_catalog_json_contract_and_live_oof_parity` and `test_asvs_v4_build_presets_catalog_rejects_malicious_quarantine_manifest_rows` (all PASS). |
| 15 | **03-03 T5 (`D-11` Public Barrel & Single Deterministic CLI):** `backend/app/modeling/__init__.py` exports all 21 symbols in alphabetically sorted `__all__`; `backend/scripts/train_and_evaluate.py` provides zero-class CLI (`--dry-run` writes zero files, `--n-jobs` clamped `1..4`, aggregate-only stdout, `< 60 s`). | ✓ VERIFIED | Direct read of `backend/app/modeling/__init__.py` (`21` sorted exports) and `backend/scripts/train_and_evaluate.py` (`235` lines, `0` classes). Covered by `test_parse_args_defaults_n_jobs_clamping_and_invalid_rejection`, `test_cli_dry_run_writes_zero_files_and_emits_aggregate_only_stdout`, `test_cli_two_run_byte_identical_sha256_and_subprocess_joblib_load`, and `test_ast_scan_train_and_evaluate_prohibitions_and_no_main_classes` (all PASS). |
| 16 | **03-03 T7 (`PROIBIÇÃO 1–4` & Git Hygiene):** `weight_regressor.joblib` git-ignored, `evaluation_report.json` & `presets_catalog.json` git-tracked, zero raw/processed/joblib files in git index, zero DL/OCR/GUI-OpenCV packages in `pyproject.toml`, zero banned imports/splitters/`cv2.imread` via AST scan, `data/raw/` untouched (`1,753` JPEGs), and full repo pytest + ruff green. | ✓ VERIFIED | Live git commands (`git check-ignore`, `git ls-files`), `pyproject.toml` audit (`17` deps, `opencv-python-headless` present, `0` banned), independent AST scan (`8` files, `0` violations), raw JPEG count `= 1753`, `uv run pytest -v` → `205 passed in 61.96s`, `uv run ruff check backend/` → `All checks passed!`. Covered by `test_git_hygiene_ignores_joblib_tracks_phase3_json_and_excludes_raw_processed` (PASS). |

---

## Artifact Verification (4-Level Check + `gsd-tools query verify.artifacts`)

`gsd-tools.cjs query verify.artifacts` returned `all_passed: true` across all three plans (`03-01`: `7/7`, `03-02`: `2/2`, `03-03`: `6/6`). Every file was independently inspected for existence, substance, wiring, and real data flow:

| Artifact Path | Exists | Substantive | Wired | Data Flows | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `backend/app/config.py` | ✓ | ✓ (40 lines) | ✓ | ✓ | ✓ VERIFIED | Exports `MODEL_BUNDLE_PATH`, `EVALUATION_REPORT_PATH`, `PRESETS_CATALOG_PATH` + all Phase 1–2 paths. |
| `backend/app/modeling/wrappers.py` | ✓ | ✓ (209 lines) | ✓ | ✓ | ✓ VERIFIED | `ALLOMETRIC_LOG_COLUMNS` (17), `AllometricLogTransformer` (`WR-01` hardened), `LogDuanRegressor`. |
| `backend/app/modeling/datasets.py` | ✓ | ✓ (383 lines) | ✓ | ✓ | ✓ VERIFIED | 1:1 Parquet join, hash lock, `verify_fold_integrity()`, `IN-04` audit NaN guard, `C1`..`C4` specs. |
| `backend/app/modeling/models.py` | ✓ | ✓ (302 lines) | ✓ | ✓ | ✓ VERIFIED | 4 classical families, `MAX_N_JOBS_CAP = 4`, capped fold weights, `WR-03` hardened predictor, importances. |
| `backend/app/modeling/age_stage.py` | ✓ | ✓ (86 lines) | ✓ | ✓ | ✓ VERIFIED | Pure stdlib (`math`, `numbers`, `dataclasses`), exact 4-stage boundaries, `WR-04` NumPy scalar support. |
| `backend/app/modeling/evaluation.py` | ✓ | ✓ (1,231 lines) | ✓ | ✓ | ✓ VERIFIED | 5-fold CV, `session_averaged_r2`, stratified slicing, 127-session dual-lens, dual-mode counterfactual, `WR-02`/`WR-05`/`WR-06`/`IN-03` hardened. |
| `backend/app/modeling/artifacts.py` | ✓ | ✓ (1,147 lines) | ✓ | ✓ | ✓ VERIFIED | Atomic dual-bundle + JSON writers, `CR-01` tuple `PDIConfig` validation, `IN-01`/`IN-02` hardened, ASVS V4 quarantine containment. |
| `backend/app/modeling/__init__.py` | ✓ | ✓ (50 lines) | ✓ | ✓ | ✓ VERIFIED | 21-symbol public barrel, alphabetically sorted `__all__`, consumed directly by CLI. |
| `backend/scripts/train_and_evaluate.py` | ✓ | ✓ (235 lines) | ✓ | ✓ | ✓ VERIFIED | Zero-class deterministic CLI, `--dry-run`, `--n-jobs` clamped `1..4`, aggregate-only stdout. |
| `backend/tests/test_modeling_wrappers_and_age_stage.py` | ✓ | ✓ (473 lines) | ✓ | ✓ | ✓ VERIFIED | `28/28` tests passing (exceeds `min_lines: 120`). |
| `backend/tests/test_modeling_datasets_and_ablation.py` | ✓ | ✓ (1,115 lines) | ✓ | ✓ | ✓ VERIFIED | `13/13` tests passing (exceeds `min_lines: 220`). |
| `backend/tests/test_train_and_evaluate_integration.py` | ✓ | ✓ (1,143 lines) | ✓ | ✓ | ✓ VERIFIED | `28/28` tests passing (exceeds `min_lines: 180`). |
| `models/evaluation_report.json` | ✓ | ✓ (624,113 B) | ✓ | ✓ | ✓ VERIFIED | Git-tracked; SHA-256 `b8d21b2dddac67609c4f8f610b8a422963fb80c19692c23f12f5869c6ffb0c5b`; `0` leaks; `0.0` live OOF diff. |
| `models/presets_catalog.json` | ✓ | ✓ (5,636 B) | ✓ | ✓ | ✓ VERIFIED | Git-tracked; SHA-256 `3a0c7e5005ae2e9d52b9e3baaf5cf8b93b2ab5dd6f4d80c767a6d60e290e1e29`; opaque IDs only; `0.0` live OOF diff. |
| `models/weight_regressor.joblib` *(git-ignored runtime output)* | ✓ | ✓ (3,336,661 B) | ✓ | ✓ | ✓ VERIFIED | Local dual-bundle (`production_pipeline` + 5 `fold_pipelines`); passes clean-subprocess `-W error::UserWarning` load & `< 5 ms` inference. |

---

## Key Link Verification (`gsd-tools query verify.key-links`)

All **8/8** PLAN key links returned `verified: true` (`03-01`: `3/3`, `03-02`: `2/2`, `03-03`: `3/3`) and were confirmed via direct source inspection:

| Plan | From | To | Via | Status |
| --- | --- | --- | --- | --- |
| `03-01` | `backend/app/modeling/datasets.py` | `backend/app/pdi/features.py` | Derives `C1`..`C4` tuples programmatically from frozen `FEATURE_COLUMNS` / `MORPHOMETRIC_*` / `CHROMATIC_*` / `OPTICAL_*` / `BANNED_*` | ✓ WIRED |
| `03-01` | `backend/app/modeling/datasets.py` | `backend/app/ingestion/splits.py` | Enforces `0` cross-fold `session_id` overlap & `>= 4` heavy sessions/fold via `verify_fold_integrity()` | ✓ WIRED |
| `03-01` | `backend/app/modeling/models.py` | `backend/app/modeling/wrappers.py` | Wraps all 4 regressor families in importable `LogDuanRegressor` and linear pipelines in `AllometricLogTransformer` | ✓ WIRED |
| `03-02` | `backend/app/modeling/evaluation.py` | `backend/app/modeling/datasets.py` | Consumes `LoadedModelingDataset`, `ABLATION_CONDITION_ORDER`, and whitelisted `C1`..`C4` column specs | ✓ WIRED |
| `03-02` | `backend/app/modeling/evaluation.py` | `backend/app/modeling/models.py` | Instantiates fresh per-fold `build_regressor_pipeline()`, fold-only `compute_sample_weights()`, and `extract_feature_importances()` | ✓ WIRED |
| `03-03` | `backend/app/modeling/artifacts.py` | `backend/app/pdi/watershed.py` | Calls `run_watershed_pdi(safe_path, config=cfg, return_visuals=False)` over ASVS V4-validated quarantine images | ✓ WIRED |
| `03-03` | `backend/scripts/train_and_evaluate.py` | `backend/app/modeling/__init__.py` | Imports dataset loader, evaluator, and atomic builders/writers strictly from public `backend.app.modeling` barrel | ✓ WIRED |
| `03-03` | `backend/scripts/train_and_evaluate.py` | `backend/app/config.py` | Anchors all default CLI paths to `CLEAN_MANIFEST_PATH`, `QUARANTINE_MANIFEST_PATH`, `FEATURES_CLEAN_PATH`, `RAW_DATASET_DIR`, `MODEL_BUNDLE_PATH`, `EVALUATION_REPORT_PATH`, `PRESETS_CATALOG_PATH` | ✓ WIRED |

---

## Requirements Coverage (`MOD-01`, `MOD-02`, `MOD-03`) & Locked Decisions (`D-01`..`D-11`)

Zero orphaned Phase 3 requirements in `.planning/REQUIREMENTS.md` (`grep "Phase 3" .planning/REQUIREMENTS.md` returns exclusively `MOD-01`, `MOD-02`, `MOD-03`).

| Req ID | Description Summary | Mapped Plans | Status | Concrete Evidence |
| --- | --- | --- | --- | --- |
| **MOD-01** | Fit & compare interpretable tabular regressors (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog` — zero DL) across locked 5-fold `session_id` splits using strictly PDI/optical features, evaluating capped session/stratum weighting and fold-internal Duan smearing. | `03-01`, `03-02`, `03-03` | ✓ SATISFIED | `wrappers.py`, `datasets.py`, `models.py`, `evaluation.py`, `train_and_evaluate.py`; 16 candidates + 4-row `mechanics_ablation` + 4-row `allometric_log_ablation` materialized in `models/evaluation_report.json`; zero cross-fold session overlap; `69` Phase 3 tests passing. |
| **MOD-02** | Score all models out-of-fold and write `models/evaluation_report.json` with MAE, RMSE, MAPE, $R^2$, MBE globally & stratified by weight band/lens/scenario, 4-condition WITH vs. WITHOUT EXIF ablation, and within-session dual-lens consistency + counterfactual audit. | `03-02`, `03-03` | ✓ SATISFIED | `models/evaluation_report.json` (`624,113` B, `0` leaks) exposes all global + `session_averaged_r2` + stratified slices, 4-condition ablation (`C2` improves over `C1` by `-6.1613 g` MAE / `-11.1664 g` RMSE / `-1.0810 pp` BalMAPE / `+0.0381` R²), 127-session dual-lens consistency, and dual-mode counterfactual audit disqualifying `C3`/`C4`. |
| **MOD-03** | Export dual-bundle `models/weight_regressor.joblib` (`production_pipeline` + 5 `fold_pipelines` + locked `PDIConfig` hash + `feature_names` + version metadata + per-band error intervals) and curated out-of-fold `models/presets_catalog.json`. | `03-01`, `03-03` | ✓ SATISFIED | `models/weight_regressor.joblib` (`3,336,661` B, git-ignored) loads cleanly in isolated subprocess (`75.2 ms`, `< 5 ms` inference, `PDIConfig` round-trips identically, `0.0` OOF diff across `1,738` rows & `6` presets); `models/presets_catalog.json` (`5,636` B, git-tracked) exposes 6 OOF single-bird + 4 Watershed opaque-ID presets; `age_stage.py` pure stdlib mapper ready for Phase 4. |

**Locked Decision Compliance (`D-01`..`D-11`):**
- `D-01` (4 classical families + `LogDuanRegressor` + Duan ON/OFF comparison): ✓ VERIFIED (`mechanics_ablation` proves weighted/smearing=False optimal at `BalMAPE = 35.1129%`).
- `D-02` (strict 49-key feature hygiene + `AllometricLogTransformer` on linear models): ✓ VERIFIED (zero `scene_type`/`scene_is_bucket`/`lens_class`/`BANNED_FEATURE_COLUMNS` in `X`; `allometric_log_ablation` logged).
- `D-03` (fold-only capped `<= 5.0` session×band `sample_weight`, unweighted validation metrics): ✓ VERIFIED (`compute_sample_weights` strictly inside training fold; unweighted OOF metrics).
- `D-04` (4-condition `C1`/`C2`/`C3`/`C4` EXIF optical ablation): ✓ VERIFIED (exact `40`/`39`/`49`/`39` cols; `C2` strictly disjoint from raw EXIF/pixel cols).
- `D-05` (127-session dual-lens consistency + dual-mode counterfactual anti-confounding gate): ✓ VERIFIED (`C2` exact `0.0%` across `median`/`mean`/`p90`; all 8 `C3`/`C4` candidates flagged & disqualified).
- `D-06` (deterministic winner selection by `balanced_band_mape_pct` + total-order tie-breakers): ✓ VERIFIED (`LightGBMLog / C2_physics_normalized_strict` selected).
- `D-07` (`models/weight_regressor.joblib` `1.0.0` dual-bundle schema + clean-subprocess `< 5 ms` load): ✓ VERIFIED (`75.2 ms` load, `2.43 ms` prod / `2.40 ms` fold-0 inference, `PDIConfig(**pdi_config)` exact match).
- `D-08` (pure stdlib `map_weight_to_age_stage()` with 4 Cobb Avian 48 stages & `[116, 2093] g` flag): ✓ VERIFIED.
- `D-09` (byte-deterministic git-tracked `models/evaluation_report.json` with `0` leaks & `1,738` scatter points): ✓ VERIFIED.
- `D-10` (byte-deterministic git-tracked opaque-ID `models/presets_catalog.json`: `6` OOF single-bird across 5 folds incl. dual-lens pair + `4` Watershed presets): ✓ VERIFIED.
- `D-11` (single deterministic CLI `train_and_evaluate.py`, config paths, atomic `os.replace()`): ✓ VERIFIED.

---

## Code Review Hardening Verification (`03-REVIEW.md` at HEAD `1fc10c4`)

All **11/11** findings from `03-REVIEW.md` (`1` Critical, `6` Warning, `4` Info) were independently verified fixed in code, covered by regression tests, and empirically validated on live inputs:

| Finding | Severity | Fix Location | Independent Verification |
| --- | --- | --- | --- |
| `CR-01` | Critical | `artifacts.py:324-347, 477` + regenerated `models/weight_regressor.joblib` | Live clean subprocess verified `isinstance(b["pdi_config"]["clahe_tile_grid"], tuple)`, `PDIConfig(**b["pdi_config"]) == PDIConfig()`, and `validate_model_bundle()` checks reconstructed hash; `test_real_model_bundle_pdi_config_reconstructs_cleanly` PASS. |
| `WR-01` | Warning | `wrappers.py:76-77, 84-91` | Live test verified swapped DataFrame columns raise `ValueError` and ndarray refit deletes `feature_names_in_`; `test_review_hardening_wrappers_complex_and_numpy_age_scalars` PASS. |
| `WR-02` | Warning | `evaluation.py:220-227` | Live test verified `y_true = [0.0]` and `[-50.0, 500.0]` raise `ValueError`; `test_review_hardening_nonpositive_metrics_c2_p90_gate_and_audit_nan` PASS. |
| `WR-03` | Warning | `models.py:222` | Live test verified `np.complex128(1+2j)`, `np.bool_(True)`, and `True` raise `ValueError` in `predict_from_feature_dict()`; PASS. |
| `WR-04` | Warning | `age_stage.py:15, 42` | Live test verified `np.int32(500)`, `np.int64(1500)`, `np.float32(200.5)` succeed while `np.bool_(True)` and `np.complex128` raise `ValueError`, preserving pure stdlib imports; PASS. |
| `WR-05` | Warning | `evaluation.py:1124-1160` | Direct read verified `mechanics_ablation` sweeps `winner.ablation_condition` and checks `cand.ablation_condition == winner.ablation_condition` for `is_selected`. |
| `WR-06` | Warning | `evaluation.py:647-660` | Live test confirmed `C2` gate raises `RuntimeError` when `median == 0.0` but `mean`/`p90 > 0.0`; `test_review_hardening_nonpositive_metrics_c2_p90_gate_and_audit_nan` PASS. |
| `IN-01` | Info | `artifacts.py:455-467` | Direct read verified explicit `elif winner.weighting_mode == "unweighted"` and `else: raise ValueError(...)`. |
| `IN-02` | Info | `artifacts.py:828-831` | Direct read verified exact float cutoffs (`< 500.0`, `<= 1000.0`, `> 1000.0`) matching `evaluation.py::_weight_band_for_value`. |
| `IN-03` | Info | `evaluation.py:551-560` | Direct read verified positive finite check on `f_old` and `z_old` before division in `compute_counterfactual_lens_flip()`. |
| `IN-04` | Info | `datasets.py:278-280` | Direct read + `test_review_hardening_nonpositive_metrics_c2_p90_gate_and_audit_nan` verified NaN rejection across all `_FEATURES_JOIN_AUDIT_COLUMNS`. |

---

## Prohibitions, Anti-Patterns & STRIDE Threat Model Verification

| Check / Threat ID | Rule / Mitigation | Direct Verification Evidence | Status |
| --- | --- | --- | --- |
| **PROIBIÇÃO 1 (`T-03-07`)** | Zero DL/OCR libraries (`torch`, `tensorflow`, `keras`, `ultralytics`, `timm`, `onnxruntime`, `paddleocr`, `easyocr`); zero edits to `pyproject.toml`/`uv.lock`; `n_jobs <= 4`. | `git diff --stat HEAD~10 -- pyproject.toml uv.lock` empty; `pyproject.toml` has `17` deps (`opencv-python-headless`, `0` banned); AST scan across all `8` Phase 3 files found `0` banned imports; `MAX_N_JOBS_CAP = 4` enforced in `models.py` and `train_and_evaluate.py`. | ✓ VERIFIED |
| **PROIBIÇÃO 2 (`T-03-02`)** | Zero `BANNED_FEATURE_COLUMNS` or non-mask background/audit metadata (`scene_type`, `scene_is_bucket`, `lcd_excluded_px`, `segmentation_fallback`, `lens_class`, `fold`, `session_id`, `rel_path`) in `X`; zero `cv2.imread`/`imdecode`. | Module-load and runtime assertions in `datasets.py`; AST scan confirmed `0` `cv2.imread`/`imdecode`; `test_ablation_feature_specs_counts_disjointness_and_hygiene` + `test_io_exif.py` PASS. | ✓ VERIFIED |
| **PROIBIÇÃO 3 (`T-03-01`)** | Zero random/ungrouped splitters (`train_test_split`, `KFold`, `StratifiedKFold`, `ShuffleSplit`, etc.); fold-isolated preprocessing/weighting/smearing; unweighted OOF scoring. | Independent AST scan across all `8` Phase 3 files found `0` banned splitters; `verify_fold_integrity()` guarantees `0` session overlap; `test_ast_scan_prohibits_dl_ocr_and_banned_splitters_in_modeling` + `test_ast_scan_train_and_evaluate_prohibitions_and_no_main_classes` PASS. | ✓ VERIFIED |
| **PROIBIÇÃO 4 (`T-03-03`)** | Zero `.jpg`/`.jpeg`/`.png`/`rel_path`/`/home/`/`data/raw` tokens in JSON artifacts or stdout; `weight_regressor.joblib` git-ignored; `data/raw/` untouched. | Programmatic scan of `models/evaluation_report.json` (`624,113` B) and `models/presets_catalog.json` (`5,636` B) confirmed `0` forbidden tokens; `git check-ignore models/weight_regressor.joblib` exited `0`; `git ls-files data/raw data/processed "models/*.joblib"` empty; `data/raw/` retains exact `1,753` JPEGs. | ✓ VERIFIED |
| **ASVS V4 (`T-03-04`)** | Non-symlink segment-by-segment path containment in `_resolve_safe_image_path()` for quarantine Watershed preset resolution. | Direct read of `artifacts.py:253-294`; `10` parametrized traversal/symlink/null-byte/backslash tests in `test_train_and_evaluate_integration.py` PASS. | ✓ VERIFIED |
| **ASVS V5 & Integrity (`T-03-05a`, `T-03-05b`, `T-03-06`, `T-03-08`)** | Atomic sibling temp-file + `os.replace()` writes, zero `__main__` classes, strict numeric/bool/complex validation, dual-mode counterfactual gate, CoW-safe Pandas 3.1 mutations. | Verified in source code and live tests; `0` debt markers (`TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`) across all Phase 3 source files; all `9` Phase 3 commits verified valid via `gsd-tools query verify.commits`. | ✓ VERIFIED |

---

## Empirical Benchmark Snapshot (Materialized `models/evaluation_report.json`)

| Condition | Model | Features | BalMAPE (%) | Global MAE (g) | Global RMSE (g) | Frame $R^2$ | Session $R^2$ | Confounded | Winner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `C1_uncalibrated_pdi` | `RidgeLog` | 40 | 48.1927 | 217.5007 | 329.1092 | 0.3136 | 0.4631 | No | — |
| `C1_uncalibrated_pdi` | `HuberLog` | 40 | 48.8280 | 220.4285 | 330.9104 | 0.3061 | 0.4973 | No | — |
| `C1_uncalibrated_pdi` | `RandomForestLog` | 40 | 38.7983 | 174.8468 | 289.2905 | 0.4696 | 0.5701 | No | — |
| `C1_uncalibrated_pdi` | `LightGBMLog` | 40 | 36.1939 | 164.2689 | 275.1906 | 0.5201 | 0.6578 | No | — |
| `C2_physics_normalized_strict` | `RidgeLog` | 39 | 47.1571 | 213.0279 | 320.0847 | 0.3507 | 0.4984 | No | — |
| `C2_physics_normalized_strict` | `HuberLog` | 39 | 47.8765 | 216.1671 | 321.3550 | 0.3456 | 0.5417 | No | — |
| `C2_physics_normalized_strict` | `RandomForestLog` | 39 | 38.5133 | 171.9993 | 282.3725 | 0.4947 | 0.6025 | No | — |
| **`C2_physics_normalized_strict`** | **`LightGBMLog`** | **39** | **35.1129** | **158.1076** | **264.0242** | **0.5582** | **0.6905** | **No (`0.0%` shift)** | **✓** |
| `C3_physics_plus_exif_meta` | `LightGBMLog` *(all 4 flagged)* | 49 | 34.7900 | 156.5607 | 263.1701 | 0.5611 | 0.6956 | **Yes** (`p90=23.52%`) | Disqualified |
| `C4_naive_double_count_fxz` | `LightGBMLog` *(all 4 flagged)* | 39 | 34.4819 | 155.8352 | 260.1403 | 0.5711 | 0.7024 | **Yes** (`med=12.74%`, `p90=39.43%`, `<500g UW` degraded) | Disqualified |

---

## Commands Run Log

1. `uv run pytest -v` → **exit 0** (`205 passed, 94 warnings in 61.96s` — all warnings are upstream `skimage.morphology.grayreconstruct` NumPy 2.5 `DeprecationWarning`; `0` failures, `0` skipped; includes `69` Phase 3 tests: `28` wrappers/age-stage + `13` datasets/ablation + `28` integration).
2. `uv run ruff check backend/` → **exit 0** (`All checks passed!`).
3. `git rev-parse HEAD && git status --short && git check-ignore models/weight_regressor.joblib && (! git check-ignore models/evaluation_report.json) && (! git check-ignore models/presets_catalog.json) && git ls-files models/evaluation_report.json models/presets_catalog.json && git ls-files data/raw data/processed "models/*.joblib"` → **exit 0** (HEAD `1fc10c41ec926b3860d17102aa12d74c989a7d81`; joblib ignored; both JSONs tracked; zero raw/processed/joblib tracked).
4. `node gsd-tools.cjs query verify.artifacts` (`03-01`, `03-02`, `03-03`) → **exit 0** (`7/7`, `2/2`, `6/6` passed).
5. `node gsd-tools.cjs query verify.key-links` (`03-01`, `03-02`, `03-03`) → **exit 0** (`3/3`, `2/2`, `3/3` verified).
6. `node gsd-tools.cjs query verify.commits 900041f 076a104 f92dc8e f314d77 0a12900 8cbd7b1 36fcd07 c130aec 1fc10c4` → **exit 0** (`9/9` valid).
7. Independent read-only Python empirical audit (JSON leak scan, schema & metrics check, clean-subprocess `load_model_bundle` `-W error::UserWarning`, `< 5 ms` latency, `1,738`-row + `6`-preset `0.0` live OOF parity, and live `CR-01`/`WR-01`..`WR-06` behavioral probes) → **exit 0** (`ALL_EMPIRICAL_AUDITS_PASSED`).
8. Independent AST prohibition scan (`8` Phase 3 files) + `pyproject.toml` dependency audit + raw JPEG count check + debt-marker `grep` → **exit 0** (`AST_SCAN_OK`, `PYPROJECT_OK deps=17`, `RAW_JPEG_COUNT=1753`, `ZERO_DEBT_MARKERS`).
9. `node gsd-tools.cjs query verification.fingerprint 3 ...` → **exit 0** (`covered_digest = v1:sha256:d3fcf7d564b0a76eab77e3c9d42842014335e14364ef45597f4491ef47661c74`).
