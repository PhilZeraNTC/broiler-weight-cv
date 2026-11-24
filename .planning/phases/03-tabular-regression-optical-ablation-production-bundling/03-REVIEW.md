---
phase: 03-tabular-regression-optical-ablation-production-bundling
reviewed: 2026-11-24T00:20:18Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - backend/app/config.py
  - backend/app/modeling/__init__.py
  - backend/app/modeling/wrappers.py
  - backend/app/modeling/datasets.py
  - backend/app/modeling/models.py
  - backend/app/modeling/age_stage.py
  - backend/app/modeling/evaluation.py
  - backend/app/modeling/artifacts.py
  - backend/scripts/train_and_evaluate.py
  - backend/tests/test_modeling_wrappers_and_age_stage.py
  - backend/tests/test_modeling_datasets_and_ablation.py
  - backend/tests/test_train_and_evaluate_integration.py
  - models/presets_catalog.json
  - models/evaluation_report.json
findings:
  critical: 1
  warning: 6
  info: 4
  total: 11
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-11-24T00:20:18Z
**Depth:** standard
**Files Reviewed:** 14
**Status:** issues_found

## Summary

Reviewed all Phase 3 modeling modules (`backend/app/modeling/*`), configuration updates (`backend/app/config.py`), deterministic training CLI (`backend/scripts/train_and_evaluate.py`), all three Phase 3 test suites (66 passed, `ruff check` clean), and the materialized artifacts (`models/weight_regressor.joblib`, `models/evaluation_report.json`, `models/presets_catalog.json`).

**Strengths verified:**
- **Zero data leakage across folds (`PROIBIÇÃO 3`):** Imputation (`SimpleImputer`), scaling (`StandardScaler`), log-allometry, `LogDuanRegressor` target transform + training-fold-only Duan smearing, capped inverse session×band `sample_weight`, OOF counterfactual evaluation, and fold-averaged feature importances are strictly isolated inside each `session_id` training fold (`0` cross-fold session overlap; `0.00000000` max diff across all 1,738 OOF scatter predictions vs. live `fold_pipelines`).
- **Optical ablation & physical lens-swap algebra (`MOD-02`):** `C1`..`C4` feature whitelists match D-04 verbatim; `physical_lens_swap` propagation (`z_ratio²`, `z_ratio`, `sensor_ratio = 1.5625`, `f_ratio²`, `f_ratio`) is algebraically exact even for digital-zoom frames while holding `area_optical_norm` invariant.
- **PROIBIÇÃO 1, 2, 4 & ASVS L1 compliance:** Zero deep-learning/OCR imports, zero banned or background-derived features (`scene_is_bucket`, `scene_type`, `BANNED_FEATURE_COLUMNS`) in `X`, zero raw image filenames or host paths in `models/*.json` or stdout, robust ASVS V4 non-symlink path containment in `_resolve_safe_image_path`, and clean atomic temp-file cleanup on failure.

**Key defects found:**
- **1 Critical (`CR-01`):** `build_model_bundle` serializes `pdi_config["clahe_tile_grid"]` as a `list` (`[8, 8]`) instead of a `tuple` (`(8, 8)`), crashing `PDIConfig(**bundle["pdi_config"])` on Phase 4 startup, while `validate_model_bundle` fails to reconstruct or hash-verify `pdi_config`.
- **6 Warnings (`WR-01`..`WR-06`):** Silent feature-column order blindness in `AllometricLogTransformer.transform()`, explosive `1e9%` MAPE on `y_true <= 0` in `compute_regression_metrics`, `np.complex*` acceptance in `predict_from_feature_dict`, NumPy integer/float32 rejection in `map_weight_to_age_stage`, hardcoded `C2` target in Step 4 `mechanics_ablation`, and median-only invariance check in `evaluate_anti_confounding_gate`.
- **4 Info (`IN-01`..`IN-04`):** Defensive hardening around `weighting_mode`, fractional weight-band rounding parity, standalone `compute_counterfactual_lens_flip` positive-optics validation, and audit-column NaN checks.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: `build_model_bundle` converts `pdi_config["clahe_tile_grid"]` from `tuple` to `list`, breaking `PDIConfig(**bundle["pdi_config"])`, and `validate_model_bundle` skips `pdi_config` hash/type verification

**File:** `backend/app/modeling/artifacts.py:321-338` and `backend/app/modeling/artifacts.py:463`
**Issue:** `build_model_bundle` packs `pdi_config` via `_json_normalize(asdict(dataset.pdi_config))` (line 463). Because `_json_normalize` (lines 175–176) converts all tuples to lists, `bundle["pdi_config"]["clahe_tile_grid"]` is persisted in `models/weight_regressor.joblib` as `[8, 8]` (`list`) instead of `(8, 8)` (`tuple[int, int]`), violating D-07 (`asdict(PDIConfig())`). When verified directly against the materialized `models/weight_regressor.joblib`, calling `PDIConfig(**bundle["pdi_config"])` crashes immediately with `ValueError: clahe_tile_grid must be a 2-tuple of positive ints, got [8, 8]` (`backend/app/pdi/config.py:159`). Simultaneously, `validate_model_bundle` (lines 321–338) only checks `isinstance(pdi_cfg, dict) and bool(pdi_cfg)` — it never reconstructs `PDIConfig(**pdi_cfg)` or checks `PDIConfig(**pdi_cfg).config_hash() == bundle["pdi_config_hash"]`, allowing both this broken `clahe_tile_grid` type and any tampered `pdi_config` dictionary to pass startup validation silently. This is a direct Phase 4 FastAPI `lifespan` crash hazard on a one-way serialized bundle contract.
**Fix:**
```python
# backend/app/modeling/artifacts.py — in build_model_bundle (line 463):
        "pdi_config": asdict(dataset.pdi_config),
```
```python
# backend/app/modeling/artifacts.py — in validate_model_bundle (lines 321-339):
    pdi_cfg = bundle["pdi_config"]
    if not isinstance(pdi_cfg, dict) or not pdi_cfg:
        raise ValueError("pdi_config must be a non-empty dictionary.")
    try:
        reconstructed_cfg = PDIConfig(**pdi_cfg)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pdi_config is not a valid PDIConfig dictionary: {exc}") from exc

    cfg_hash = bundle["pdi_config_hash"]
    if (
        not isinstance(cfg_hash, str)
        or len(cfg_hash) != 16
        or not set(cfg_hash).issubset(_HEX_LOWER_CHARS)
    ):
        raise ValueError(
            f"pdi_config_hash must be a 16-char lowercase hex string, got {cfg_hash!r}"
        )
    if reconstructed_cfg.config_hash() != cfg_hash:
        raise ValueError(
            f"pdi_config internal hash mismatch: reconstructed {reconstructed_cfg.config_hash()!r} "
            f"!= bundle pdi_config_hash {cfg_hash!r}."
        )
    if expected_pdi_config_hash is not None and cfg_hash != expected_pdi_config_hash:
        raise ValueError(
            f"pdi_config_hash mismatch: bundle has {cfg_hash!r}, "
            f"expected {expected_pdi_config_hash!r}."
        )
```
Update the synthetic stub in `backend/tests/test_train_and_evaluate_integration.py:653` to pass `asdict(PDIConfig())`, add a negative test for mismatched `pdi_config` vs. `pdi_config_hash`, and re-run `backend/scripts/train_and_evaluate.py` to regenerate `models/weight_regressor.joblib`.

## Warnings

### WR-01: `AllometricLogTransformer.transform()` ignores `feature_names_in_` on DataFrames and strips column names, causing silent catastrophic misprediction on reordered columns

**File:** `backend/app/modeling/wrappers.py:58-96`
**Issue:** `AllometricLogTransformer.fit()` records `self.feature_names_in_` when fit on a `pd.DataFrame` (line 67) and stores positional `self.log_indices_`. However, `transform()` (lines 80–96) never compares `X.columns` against `self.feature_names_in_` before converting `X` to a positional `np.ndarray` and applying `np.log1p` to `self.log_indices_`. Because it outputs a nameless `np.ndarray`, the downstream `SimpleImputer` and `StandardScaler` in `RidgeLog`/`HuberLog` pipelines never receive column names either. Verified live: passing a DataFrame with swapped columns `["solidity", "area_optical_norm"]` to a fitted `RidgeLog` pipeline silently outputs `485,165,194 g` instead of raising `ValueError` (whereas `LightGBMLog`/`RandomForestLog` properly raise via `SimpleImputer`). Additionally, refitting the same transformer instance on an `np.ndarray` after a `DataFrame` leaves a stale `feature_names_in_` attribute on the estimator.
**Fix:**
```python
# backend/app/modeling/wrappers.py — in AllometricLogTransformer.fit() else branch:
        else:
            if hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D input array, got shape {arr.shape!r}")
            self.n_features_in_ = int(arr.shape[1])
            self.log_indices_ = ()
```
```python
# backend/app/modeling/wrappers.py — at top of AllometricLogTransformer.transform():
        check_is_fitted(self, "log_indices_")
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            observed = [str(c) for c in X.columns]
            expected = self.feature_names_in_.tolist()
            if observed != expected:
                raise ValueError(
                    "Feature names and column order must match those passed during fit."
                )
```

### WR-02: `compute_regression_metrics` silently returns explosive `1e9%` MAPE on zero or negative `y_true`

**File:** `backend/app/modeling/evaluation.py:220-231`
**Issue:** Line 224 computes `mape_pct = float(np.mean(abs_err / np.maximum(y_t, 1e-6)) * 100.0)` without validating `y_t > 0.0`. Verified live: `compute_regression_metrics([0.0], [10.0])` silently returns `mape_pct = 1000000000.0` (1 billion %), and `compute_regression_metrics([-100.0, 500.0], [100.0, 500.0])` clamps `-100.0` to `1e-6`, returning `mape_pct = 10000000000.0`. Unlike `datasets.py:326` and `wrappers.py:144`, this exported metric function neither rejects non-positive ground-truth weights nor masks undefined percentage denominators as `None`.
**Fix:**
```python
# backend/app/modeling/evaluation.py — inside compute_regression_metrics (around line 220):
    if np.any(y_t <= 0.0):
        raise ValueError("y_true must contain strictly positive weights (> 0) for gram metrics.")
    err = y_p - y_t
    abs_err = np.abs(err)
    mae_g = float(mean_absolute_error(y_t, y_p))
    rmse_g = float(root_mean_squared_error(y_t, y_p))
    mape_pct = float(np.mean(abs_err / y_t) * 100.0)
```
*(Alternatively, if callers may pass `y_true == 0.0` intentionally, set `mape_pct = None` when `np.any(y_t <= 0.0)` rather than dividing by `1e-6`.)*

### WR-03: `predict_from_feature_dict` accepts `np.complex*` feature values, silently dropping the imaginary part with `ComplexWarning`

**File:** `backend/app/modeling/models.py:221-225`
**Issue:** Line 221 checks `if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float, np.number)):`. Because `np.complex64` and `np.complex128` subclass `np.number`, passing `{"a": np.complex128(1 + 2j), ...}` bypasses validation and calls `float(raw_val)`, which in NumPy silently discards the imaginary part while emitting a `ComplexWarning` instead of raising `ValueError`.
**Fix:**
```python
# backend/app/modeling/models.py (line 221):
        if isinstance(raw_val, (bool, np.bool_)) or not isinstance(
            raw_val, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"Feature {name!r} must be a finite real numeric value, got {raw_val!r}"
            )
```

### WR-04: `map_weight_to_age_stage` rejects NumPy `int32`/`int64`/`float32` scalars (such as Parquet `weight_g` `int32` values)

**File:** `backend/app/modeling/age_stage.py:41`
**Issue:** Line 41 checks `if isinstance(weight_g, bool) or not isinstance(weight_g, (int, float)):`. While `np.float64` subclasses Python `float`, `np.int32`, `np.int64`, and `np.float32` do not. Verified live: `map_weight_to_age_stage(np.int32(500))` raises `ValueError`, even though `manifest_clean.parquet` and `features_clean.parquet` store `weight_g` with dtype `int32`. Phase 4 code passing a Parquet row's `weight_g` directly to `map_weight_to_age_stage()` will fail unless manually cast with `float(...)`. Using stdlib `numbers.Real` accepts all Python and NumPy real scalars (`int`, `float`, `np.int32`, `np.int64`, `np.float32`, `np.float64`) while preserving 100% stdlib purity and still rejecting `bool`, `np.bool_`, `complex`, `str`, and `None`.
**Fix:**
```python
# backend/app/modeling/age_stage.py:
import math
import numbers
...
def map_weight_to_age_stage(weight_g: float) -> AgeStageResult:
    """Map a strictly positive finite live weight (g) into a Cobb Avian 48 stage (D-08)."""
    if isinstance(weight_g, bool) or not isinstance(weight_g, numbers.Real):
        raise ValueError(f"weight_g must be a finite positive number, got {weight_g!r}")
```

### WR-05: `run_full_evaluation` hardcodes `"C2_physics_normalized_strict"` in `mechanics_ablation` and marks `is_selected=True` without checking `ablation_condition`

**File:** `backend/app/modeling/evaluation.py:1100-1143`
**Issue:** Both `C1_uncalibrated_pdi` and `C2_physics_normalized_strict` pass the anti-confounding gate and are eligible to win `select_winning_candidate` (line 648). However, Step 4 (`mechanics_ablation`, lines 1108–1122) hardcodes `"C2_physics_normalized_strict"` and reuses `c2_by_model[winner.model_name]` rather than `winner.ablation_condition`, and lines 1137–1140 compute `"is_selected"` without verifying `cand.ablation_condition == winner.ablation_condition`. On any dataset where `C1_uncalibrated_pdi` wins, `mechanics_ablation` silently audits `C2` and falsely marks a non-winning `C2` candidate as `"is_selected": true`.
**Fix:**
```python
# backend/app/modeling/evaluation.py (lines 1108-1143):
    for use_sw, use_smear in mechanics_specs:
        if use_sw and not use_smear:
            mechanics_cands.append(winner)
        else:
            mechanics_cands.append(
                evaluate_single_candidate(
                    dataset,
                    winner.model_name,
                    winner.ablation_condition,
                    use_sample_weight=use_sw,
                    use_smearing=use_smear,
                    use_allometric_log=True,
                    n_jobs=n_jobs,
                    c2_by_band_and_lens=c2_by_model[winner.model_name].by_band_and_lens,
                )
            )
...
            "is_selected": bool(
                cand.ablation_condition == winner.ablation_condition
                and cand.weighting_mode == winner.weighting_mode
                and cand.duan_smearing_enabled == winner.duan_smearing_enabled
            ),
```

### WR-06: `evaluate_anti_confounding_gate` verifies only `median_shift_pct == 0.0` (ignoring `mean` and `p90`) for `C2` 0.0% lens invariance

**File:** `backend/app/modeling/evaluation.py:634-643`
**Issue:** Lines 636–638 assert `C2_physics_normalized_strict` lens invariance solely via `if focal_med != 0.0 or phys_med != 0.0: raise RuntimeError(...)`. Because the median is the 50th percentile, up to 49.9% of evaluated predictions can shift arbitrarily (`mean_shift_pct > 0`, `p90_shift_pct > 0`) while `median_shift_pct` stays `0.0`. Checking `mean_shift_pct` and `p90_shift_pct` alongside `median_shift_pct` closes this logical gap because shifts are strictly non-negative.
**Fix:**
```python
# backend/app/modeling/evaluation.py (lines 634-643):
    if condition == "C2_physics_normalized_strict":
        if n_eval > 0:
            shift_vals = (
                focal_info.get("median_shift_pct"),
                focal_info.get("mean_shift_pct"),
                focal_info.get("p90_shift_pct"),
                phys_info.get("median_shift_pct"),
                phys_info.get("mean_shift_pct"),
                phys_info.get("p90_shift_pct"),
            )
            if any(v != 0.0 for v in shift_vals):
                raise RuntimeError(
                    "C2_physics_normalized_strict violated strict 0.0% lens invariance: "
                    f"focal={focal_info!r}, phys={phys_info!r}"
                )
        return False, []
```

## Info

### IN-01: `build_model_bundle` silently falls back to unweighted training on unrecognized `weighting_mode`

**File:** `backend/app/modeling/artifacts.py:446-454`
**Issue:** `if winner.weighting_mode == DEFAULT_WEIGHTING_MODE:` uses `sample_weight`, while `else:` silently fits unweighted without asserting `winner.weighting_mode == "unweighted"`. An unrecognized weighting mode would silently refit `production_pipeline` unweighted while recording the unrecognized name in the bundle.
**Fix:** Replace `else:` with `elif winner.weighting_mode == "unweighted": production_pipeline.fit(X_all, y_all)` and `else: raise ValueError(f"Unsupported weighting_mode: {winner.weighting_mode!r}")`.

### IN-02: Fractional weight-band rounding inconsistency between `artifacts.py` and `evaluation.py`

**File:** `backend/app/modeling/artifacts.py:814-816`
**Issue:** `artifacts.py` classifies `weight_band` via `assign_weight_band(int(round(float(w))))`, whereas `evaluation.py:127-135` (`_weight_band_for_value`) compares float weights directly (`< 500`, `<= 1000`). On fractional weights like `499.6`, `artifacts.py` assigns `"500-1000g"` while `evaluation.py` assigns `"<500g"`. Harmless on the current `int32` dataset, but sharing `_weight_band_for_value` removes subtle boundary divergence.
**Fix:** Reuse a single shared weight-band helper between `evaluation.py` and `artifacts.py`.

### IN-03: `compute_counterfactual_lens_flip` does not validate `focal_length_mm > 0` / `zoom_multiple > 0` when called standalone

**File:** `backend/app/modeling/evaluation.py:546-556`
**Issue:** `load_modeling_dataset` enforces `focal > 0` and `zoom > 0`, so full runs are guarded, but direct callers passing non-positive or non-finite `zoom_multiple` / `focal_length_mm` trigger `ZeroDivisionError`/`inf`/`nan` in `z_ratio = z_new / z_old`.
**Fix:** Add a finite strictly-positive check on `f_old` and `z_old` before dividing in `compute_counterfactual_lens_flip`.

### IN-04: `load_modeling_dataset` does not check non-alignment audit columns for NaN values

**File:** `backend/app/modeling/datasets.py:274-277`
**Issue:** Only `_ALIGNMENT_COLUMNS`, `pdi_config_hash`, and `FEATURE_COLUMNS` are checked for missing/NaN values; `scene_type`, `scene_is_bucket`, `lcd_excluded_px`, and `segmentation_fallback` are not. If regenerated Parquets contained NaN in `scene_type` or `segmentation_fallback`, `astype(str)` / `astype(bool)` would silently convert them to `"nan"` / `True` instead of failing fast. (Verified the currently materialized `features_clean.parquet` has zero NaNs across all 60 columns.)
**Fix:** Extend the NaN validation loop to cover all columns in `_FEATURES_JOIN_AUDIT_COLUMNS` on `features_df`.

---

_Reviewed: 2026-11-24T00:20:18Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
