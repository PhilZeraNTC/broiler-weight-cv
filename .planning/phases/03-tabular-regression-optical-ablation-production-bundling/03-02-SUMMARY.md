---
phase: 03-tabular-regression-optical-ablation-production-bundling
plan: 02
subsystem: modeling
tags: [scikit-learn, lightgbm, pandas, cross-validation, optical-ablation, counterfactual-audit, anti-confounding]

requires:
  - phase: 03-tabular-regression-optical-ablation-production-bundling
    provides: "LoadedModelingDataset, 4-condition ablation specs (C1..C4), 4-family regressor factory, fold-only capped compute_sample_weights(), and pickling-safe AllometricLogTransformer/LogDuanRegressor from Plan 03-01."

provides:
  - "Leak-free 5-fold session_id OOF evaluation, 4x4 optical ablation engine, unweighted global/session/stratified metrics (including session_averaged_r2), 127-session dual-lens consistency, dual-mode counterfactual lens-flip audit (focal_only + physical_lens_swap), anti-confounding gate, 2x2 weighting x smearing & linear allometric-log sensitivity diagnostics, and deterministic D-06 winner selector in backend/app/modeling/evaluation.py."
  - "Expanded unit, edge-case (empty/single/constant/adjacency/ordering), synthetic lens-confounding disqualification, and PROIBIÇÃO 1/3 AST test suite in backend/tests/test_modeling_datasets_and_ablation.py (12 tests total; 39 Phase 3 tests passing in 6.23s)."

affects:
  - 03-03-PLAN
  - 04-web-api-xray-pdi-studio

actuals:
  tokens: 19677
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Strict fold-isolated fitting loop instantiating fresh pipelines, imputers, scalers, AllometricLogTransformer, capped sample_weight, and Duan smearing exclusively inside fold != k while scoring unweighted OOF predictions in grams"
    - "Dual-mode OOF counterfactual lens-flip audit (focal_only + physical_lens_swap with Redmi 10 4.25/1.0 <-> 1.65/0.6 and 1.5625 sensor area ratio) catching both linear focal-length shortcuts and tree lens-ID reconstruction while asserting 0.0% invariance on C2_physics_normalized_strict"
    - "Total-order permutation-invariant D-06 winner selection key (balanced_band_mape_pct, global_mape_pct, global_mae_g, dual_lens_median_abs_diff_g_or_inf, ablation_condition, model_name)"
    - "Zero-negative-zero deterministic 4-decimal / 6-decimal rounding helpers for byte-reproducible downstream JSON serialization"

key-files:
  created:
    - backend/app/modeling/evaluation.py
  modified:
    - backend/tests/test_modeling_datasets_and_ablation.py

key-decisions:
  - "Embedded session_averaged_r2 directly into global_metrics alongside frame-level r2 (in addition to full session_level_metrics) so Phase 4 WEB-05 displays both metrics side-by-side inside winning_model.global."
  - "Aligned compute_stratified_metrics() weight-band partitioning directly with Phase 1 assign_weight_band() (<500g / 500-1000g / >1000g) and enforced strict '>' inequality on anti-confounding thresholds (>10.0% median / >15.0% p90)."
  - "Evaluated C2_physics_normalized_strict across all 4 families first inside run_full_evaluation() so C3/C4 anti-confounding gates receive their exact same-model C2 <500g ultrawide MAE/MBE baselines, and reused already-fitted C2 candidates inside mechanics_ablation and allometric_log_ablation to avoid redundant fits (34.01 s end-to-end on real 1,738-row corpus)."

patterns-established:
  - "Pandas 3.1 CoW-safe counterfactual slicing via X_val.copy() and single-step .loc[mask, col] = ... mutations strictly on held-out validation folds."
  - "Ordered deterministic stratum slicing across WEIGHT_BAND_ORDER, LENS_CLASS_ORDER, SCENE_TYPE_ORDER, and sorted groupby(['session_id'], sort=True).indices."

requirements-completed:
  - MOD-01
  - MOD-02

coverage:
  - id: D1
    description: "Leak-free 5-fold session_id OOF evaluation across 4 classical families x 4 optical ablation conditions with unweighted global, session_averaged_r2, and stratified metrics plus safe empty/single/constant/adjacency edge guards."
    requirement: MOD-01
    verification:
      - kind: unit
        ref: backend/tests/test_modeling_datasets_and_ablation.py#test_compute_regression_metrics_normal_empty_single_constant_and_validation
        status: pass
      - kind: unit
        ref: backend/tests/test_modeling_datasets_and_ablation.py#test_compute_session_level_metrics_improves_r2_on_frame_noise_and_handles_empty
        status: pass
      - kind: unit
        ref: backend/tests/test_modeling_datasets_and_ablation.py#test_compute_stratified_metrics_weight_band_adjacency_and_empty_slices
        status: pass
    human_judgment: false
  - id: D2
    description: "Within-session dual-lens consistency (127 mixed-lens sessions on real corpus), dual-mode counterfactual lens-flip audit (focal_only + physical_lens_swap), and strict-inequality anti-confounding gate proving 0.0% C2 invariance while flagging C3 and C4 shortcuts."
    requirement: MOD-02
    verification:
      - kind: unit
        ref: backend/tests/test_modeling_datasets_and_ablation.py#test_compute_within_session_dual_lens_consistency_mixed_and_single_lens
        status: pass
      - kind: unit
        ref: backend/tests/test_modeling_datasets_and_ablation.py#test_evaluate_anti_confounding_gate_exact_strict_inequality_boundaries
        status: pass
      - kind: unit
        ref: backend/tests/test_modeling_datasets_and_ablation.py#test_synthetic_lens_confounded_dataset_flags_c3_and_c4_and_selects_c2
        status: pass
    human_judgment: false
  - id: D3
    description: "Deterministic permutation-invariant D-06 winner selector and PROIBIÇÃO 1/3 AST verification covering evaluation.py."
    requirement: MOD-02
    verification:
      - kind: unit
        ref: backend/tests/test_modeling_datasets_and_ablation.py#test_select_winning_candidate_deterministic_tie_breaks_and_empty_eligible
        status: pass
      - kind: unit
        ref: backend/tests/test_modeling_datasets_and_ablation.py#test_ast_scan_prohibits_dl_ocr_and_banned_splitters_in_modeling
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-11-23
status: complete
---

# Phase 03 Plan 02: Leak-Free 5-Fold Evaluation, 4-Condition Optical Ablation & Anti-Confounding Audit Summary

**Implemented the complete leak-free 5-fold `session_id` evaluation, 4×4 EXIF optical ablation, 127-session dual-lens consistency, dual-mode counterfactual anti-confounding gate, and deterministic D-06 winner selector in `backend/app/modeling/evaluation.py`, verified against both synthetic confounding/edge-case tests and the full 1,738-row corpus.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-11-23T23:43:23Z
- **Completed:** 2026-11-23T23:54:28Z
- **Tasks:** 2 / 2 completed
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments
- Implemented `backend/app/modeling/evaluation.py` (`1,211` lines) delivering:
  - Strictly unweighted `compute_regression_metrics()` with safe `None` returns for empty (`n == 0`), single-row (`n == 1`), and constant-target slices.
  - `compute_session_level_metrics()` computing session-averaged `session_averaged_r2` (exposed both in `session_level_metrics` and embedded inside `global_metrics["session_averaged_r2"]` alongside image-level `r2`).
  - `compute_stratified_metrics()` covering `per_fold` (`0..4`), `by_weight_band` (aligned identically with Phase 1 `assign_weight_band()`), `by_lens`, `by_scene`, `by_band_and_scene`, `by_band_and_lens`, `band_error_intervals` (`D-07`), and `balanced_band_mape_pct`.
  - `compute_within_session_dual_lens_consistency()` pairing `wide` vs. `ultrawide` OOF predictions across all mixed-lens sessions (`127` on real data) using sorted `df.groupby(["session_id"], sort=True).indices`.
  - `compute_counterfactual_lens_flip()` executing both `focal_only` (`4.25 <-> 1.65`) and `physical_lens_swap` (Redmi 10 `4.25/1.0 <-> 1.65/0.6` with sensor area ratio `1.5625` via Pandas 3.1 CoW-safe `.copy()` + `.loc[...] = ...`) strictly on held-out fold pipelines, proving exact `0.0%` shift on `C2_physics_normalized_strict`.
  - `evaluate_anti_confounding_gate()` enforcing strict `>` thresholds (`>10.0%` focal median, `>10.0%` physical median, `>15.0%` physical p90, and `<500g` ultrawide MAE+`|MBE|` degradation vs. `C2`).
  - `evaluate_single_candidate()`, permutation-invariant `select_winning_candidate()`, and `run_full_evaluation()` executing all 16 candidates (`4` families × `4` conditions), `2×2` weighting × smearing `mechanics_ablation`, linear `allometric_log_ablation` (`RidgeLog`/`HuberLog`), `optical_ablation` (with `deltas_vs_c1`), averaged fold `feature_importances`, and anonymized `oof_scatter_points`.
- Verified live on the real `1,738`-row corpus (`34.01 s`, zero `UserWarning`s under `-W error::UserWarning`):
  - Winner: **`LightGBMLog / C2_physics_normalized_strict`** (`BalMAPE = 35.1129%`, `MAE = 158.1076 g`, `RMSE = 264.0242 g`, `MAPE = 37.0876%`, `R² = 0.5582`, `session_averaged_r2 = 0.6905`, `127` mixed-lens sessions).
  - `C2` strictly beats `C1` (`delta_mae_g = -6.1613 g`, `delta_rmse_g = -11.1664 g`, `delta_balanced_band_mape_pct = -1.0810 pp`, `delta_r2 = +0.0381`, `delta_session_averaged_r2 = +0.0327`).
  - `C2` achieves exact `0.0%` shift under both counterfactual modes while `C3` (`p90 = 23.52% > 15.0%`) and `C4` (`median = 12.74% > 10.0%`, `p90 = 39.43% > 15.0%`, `<500g UW` degraded vs. `C2`) are automatically flagged and disqualified.
- Expanded `backend/tests/test_modeling_datasets_and_ablation.py` with `7` new deterministic tests (`12` total in file; `39` across Phase 3 test files passing in `6.23 s`; `175 / 175` total repo tests passing; `0` ruff diagnostics).

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD): Leak-free 5-fold evaluation, stratified/session metrics, dual-lens consistency, dual-mode counterfactual audit, and D-06 winner selector** - `f314d77` (feat)
2. **Task 2 (TDD): Expanded edge-case, tie-break, synthetic lens-confounding disqualification & AST test suite** - `0a12900` (test)

## Files Created/Modified
- `backend/app/modeling/evaluation.py` - Complete leak-free 5-fold CV runner, global/session/stratified metrics, dual-lens consistency, dual-mode counterfactual lens-flip audit, anti-confounding gate, mechanics/allometric diagnostics, and deterministic `D-06` winner selector.
- `backend/tests/test_modeling_datasets_and_ablation.py` - Added 7 unit and synthetic lens-confounding tests verifying empty/single-row/constant metric guards, exact `499/500/1000/1001` weight-band adjacency, exact `10.0/10.01` and `15.0/15.01` strict inequality gate boundaries, permutation-invariant tie-breaking, synthetic `C3`/`C4` disqualification vs. `0.0%` `C2` invariance, and explicit AST coverage of `evaluation.py`.

## Decisions Made
- Embedded `session_averaged_r2` into `CandidateEvaluation.global_metrics` in addition to `session_level_metrics` so Plan `03-03` (`build_evaluation_report`) and Phase 4 (`WEB-05`) read both `winning_model.global.r2` and `winning_model.global.session_averaged_r2` directly.
- Normalized `-0.0` to `0.0` in deterministic rounding helpers (`_round4`, `_round6`) to guarantee byte-identical JSON rendering in Plan `03-03`.
- Reused already-computed `C2` candidates across `mechanics_ablation` and `allometric_log_ablation` to keep end-to-end 1,738-row evaluation at `34.01 s` well within the `60 s` CPU budget.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Initial ruff checks caught 3 line-length (`E501`) lines in `evaluation.py` and 9 in `test_modeling_datasets_and_ablation.py`; wrapped cleanly to `<= 100` chars with zero remaining diagnostics.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Ready for **Plan 03-03**: `backend/app/modeling/artifacts.py`, `backend/app/modeling/__init__.py`, and `backend/scripts/train_and_evaluate.py` can directly consume `run_full_evaluation()`, `FullEvaluationResult`, `CandidateEvaluation`, `LITERATURE_R2_REFERENCE`, and `WEIGHT_BAND_ORDER` to publish `models/weight_regressor.joblib`, `models/evaluation_report.json`, and `models/presets_catalog.json`.

## Self-Check: PASSED

- Verified both created/modified files exist on disk (`backend/app/modeling/evaluation.py`, `backend/tests/test_modeling_datasets_and_ablation.py`).
- Verified both task commits exist in git history (`f314d77`, `0a12900`).
- Verified targeted Phase 3 pytest (`39 passed in 6.23s`), full repository pytest (`175 passed`), real-corpus `run_full_evaluation()` (`34.01 s`, `0` warnings under `-W error::UserWarning`), and `ruff check backend/` (`All checks passed!`).
- Verified `PROIBIÇÃO 1–4` compliance (zero DL/OCR imports, zero banned splitters, zero validation weighting, zero `scene_type`/`lens_class` features in `X`, zero raw JPEG filenames or path leaks).

---
*Phase: 03-tabular-regression-optical-ablation-production-bundling*
*Completed: 2026-11-23*
