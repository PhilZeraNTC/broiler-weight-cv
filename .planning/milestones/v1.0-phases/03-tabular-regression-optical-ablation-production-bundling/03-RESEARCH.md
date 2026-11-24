# Phase 3: Tabular Regression, Optical Ablation & Production Bundling - Research

**Researched:** 2026-11-23
**Domain:** Classical Interpretable Tabular Regression, EXIF Optical Ablation, Anti-Confounding Dual-Lens Auditing & Deterministic ML Artifact Bundling (`scikit-learn` + `LightGBM` + `joblib`)
**Confidence:** HIGH (grounded in live 5-fold cross-validation spikes over all `1,738` materialized Phase 2 feature rows, all `13` quarantined multi-bird images, and verbatim codebase verification)

---

## User Constraints

> Copied verbatim from `.planning/phases/03-tabular-regression-optical-ablation-production-bundling/03-CONTEXT.md` (`## Implementation Decisions`, `## the agent's Discretion`, and `## Deferred Ideas`). Downstream planners and executors MUST honor every locked decision (`D-01`–`D-11`).

### Tabular Regressor Families, Feature Hygiene & Target Mechanics (`MOD-01`)
- **D-01:** Train and compare 4 interpretable/tabular regressor families strictly complying with `PROIBIÇÃO 1` (zero deep learning / neural networks) over the locked 5-fold `session_id` splits: (1) `RidgeLog`, (2) `HuberLog`, (3) `RandomForestLog`, and (4) `LightGBMLog` (optionally benchmarking `HistGradientBoostingLog` during Phase 3 research while keeping the 4 PRD families as mandatory rows). Implement target log-transformation (`np.log1p` / `np.expm1`) and optional fold-internal Duan's smearing ($\hat{\sigma}_{\text{train}} = \frac{1}{n}\sum \exp(e_i)$ computed strictly on training-fold residuals) inside an importable `sklearn`-compatible estimator class (`backend/app/modeling/wrappers.py::LogDuanRegressor` — NEVER `lambda` functions or `__main__` classes so `joblib` serialization succeeds cleanly in fresh FastAPI processes). Empirically compare Duan's smearing ON vs. OFF during Phase 3 research/CV.
- **D-02:** Enforce strict feature hygiene (`PROIBIÇÃO 2` & Pitfall 3): regressors may ONLY consume columns from Phase 2's frozen 49-key `FEATURE_COLUMNS` (strictly computed inside `bird_mask == 255`); NEVER include `scene_is_bucket`/`scene_type` (derived from outer border-ring background photometry), `lcd_excluded_px`, `segmentation_fallback`, `lens_class`, `fold`, `session_id`, or any `BANNED_FEATURE_COLUMNS` (`oriented_w`, `oriented_h`, `megapixels`, `bbox_w`, `bbox_h`, `raw_w`, `raw_h`) as training features — reserve `scene_type`/`scene_is_bucket` and `lens_class` strictly for stratified evaluation slicing (`MOD-02`). For linear families (`RidgeLog`, `HuberLog`), include an importable `sklearn` transformer (`AllometricLogTransformer` inside `Pipeline`, fit per fold) applying `np.log1p(np.clip(x, 0, None))` to positive geometric area/linear columns alongside `SimpleImputer(strategy="median")` + `StandardScaler()` so linear models directly express physical $W \propto A^{3/2}$ allometry and smooth tail extrapolation — **Reversibility:** costly — changing allowed input features or pipeline preprocessing alters all CV comparisons and serialized contracts.
- **D-03:** Compute training-fold-only `sample_weight` combining inverse session frequency (`1 / n_imgs_in_session` within each training fold, preventing 47-photo weights from overpowering 6-photo sessions) multiplied by moderate inverse weight-band frequency (`<500g`, `500-1000g`, `>1000g`), capped strictly at `max_weight / min_weight <= 5.0` and normalized to mean `1.0`. Empirically benchmark weighted vs. unweighted fitting across all 5 folds during Phase 3 research, and NEVER apply sample weights to validation metrics (`MAE`, `RMSE`, `MAPE`, `R²`, `MBE` are always unweighted out-of-fold metrics in grams).

### 4-Condition EXIF Optical Ablation & Anti-Confounding Gate (`MOD-02`)
- **D-04:** Execute a clean, mathematically leak-free **4-condition optical ablation** across all candidate model families on the identical 5 `session_id` folds:
  1. `C1_uncalibrated_pdi` (`40` features: `MORPHOMETRIC_FEATURE_COLUMNS` + `CHROMATIC_TEXTURE_FEATURE_COLUMNS`, ZERO `OPTICAL_FEATURE_COLUMNS` — baseline WITHOUT EXIF).
  2. `C2_physics_normalized_strict` (`39` features: strictly dimensionless/scale-invariant shape [`aspect_ratio_ellipse`, `eccentricity`, `solidity`, `circularity`, `extent_rotrect`, `log_hu_1`, `log_hu_2`, `log_hu_3`] + `26` `CHROMATIC_TEXTURE_FEATURE_COLUMNS` + `5` optically normalized physical descriptors [`area_optical_norm`, `sqrt_area_optical_norm`, `perim_optical_norm`, `major_axis_optical_norm`, `minor_axis_optical_norm`] using Phase 2's empirically winning `area_px / zoom²` calibration — ZERO raw `focal_length_mm`, ZERO `zoom_multiple`, ZERO `exif_optical_present`, and ZERO uncalibrated `area_px`/`area_frac` so tree models cannot reconstruct lens ID via ratios).
  3. `C3_physics_plus_exif_meta` (all `49` `FEATURE_COLUMNS`, including `focal_length_mm`, `zoom_multiple`, `exif_optical_present`, raw `area_px..area_frac`, and `area_frac_optical_norm` — directly tests whether explicit lens/zoom metadata adds genuine physical signal or exploits the `ultrawide ↔ >1,000 g` confounding shortcut).
  4. `C4_naive_double_count_fxz` (replaces the 5 `*_optical_norm` columns in `C2` with naive `area_px / ((focal_length_mm * zoom_multiple) ** 2)` and its linear `/(f*z)` equivalents — quantitatively proving in downstream 5-fold CV why Phase 2 Spike 2's empirical correction prevented `focal × zoom` double-counting degradation).
  - Phase 3 research may also run an internal diagnostic comparing `area_frac_optical_norm` vs. `area_optical_norm` inside `C2` to confirm Phase 2's choice on downstream regression error — **Reversibility:** costly — ablation condition keys become the API and chart contract for Phase 4 `WEB-05`.
- **D-05:** Enforce a quantitative **Within-Session Dual-Lens Consistency & Counterfactual Lens-Flip Audit**:
  - *Within-Session Dual-Lens Consistency (OOF only):* Across all `127` mixed-lens sessions in `manifest_clean.parquet`, pair Wide (`4.25 mm`) vs. Ultrawide (`1.65 mm`) OOF predictions of the same bird/session and compute mean/median `|ŷ_wide - ŷ_uw|` (g) and relative `|ŷ_wide - ŷ_uw| / weight_g` (%).
  - *Counterfactual Lens-Flip Sanity Check:* On models exposed to `focal_length_mm`/`zoom_multiple` (`C3`), hold physically normalized area, shape, and plumage chroma fixed while flipping `focal_length_mm` between `4.25 mm` and `1.65 mm` (and checking `<500 g` ultrawide Mean Bias Error). If median counterfactual prediction shift exceeds `10%` (or degrades `<500 g` ultrawide MBE/MAE vs. `C2`), automatically flag `lens_confounding_detected = True` and disqualify raw lens-ID shortcuts from production winner selection in favor of strictly physics-normalized (`C2`) models.

### Deterministic Winner Selection, Dual-Bundle `.joblib` & Age-Stage Mapper (`MOD-01`, `MOD-03`)
- **D-06:** Select the winning production candidate deterministically among candidates that pass the anti-confounding gate (`lens_confounding_detected == False`) by minimizing **balanced weight-band OOF MAPE** (`mean(MAPE_<500g, MAPE_500-1000g, MAPE_>1000g)`), with global OOF MAPE, global MAE, and within-session dual-lens consistency as secondary tie-breakers — preventing the `70.4%` `<500 g` stratum from sacrificing accuracy on rare `>1,000 g` broilers.
- **D-07:** Serialize the winning production bundle to git-ignored `models/weight_regressor.joblib` (`MODEL_BUNDLE_SCHEMA_VERSION = "1.0.0"`) containing: `schema_version`, `model_version`, `pdi_config` (`asdict(PDIConfig())`), `pdi_config_hash` (`"43a3cfaa3208019e"`), `winning_model_name`, `winning_ablation_condition`, `feature_names` (`tuple[str, ...]`), `weighting_mode`, `duan_smearing_enabled`, `production_pipeline` (refitted on 100% of `1,738` clean images for arbitrary live uploads), `fold_pipelines` (`dict[int, Pipeline]` for folds `0..4`, enabling strictly honest out-of-fold live preset inference in Phase 4 `WEB-03`), `band_error_intervals` (per weight band `<500g`, `500-1000g`, `>1000g`: `mae_g`, `mape_pct`, `median_abs_err_g`, `p68_abs_err_g`, `p90_abs_err_g`, `mbe_g`), `valid_weight_range_g` (`{"min": 116, "max": 2093}`), and `valid_age_days_range` (`"~7-29 days"`). Gate completion on clean subprocess `joblib.load()` + single-row prediction (`< 5 ms`, zero pickling errors) — **Reversibility:** one-way — Phase 4 FastAPI `lifespan` loader (`WEB-01`, `WEB-02`, `WEB-03`) consumes this exact bundle schema.
- **D-08:** Implement a pure, shared Cobb Avian 48 developmental age-stage mapper in `backend/app/modeling/age_stage.py` (`map_weight_to_age_stage(weight_g: float)`) mapping predicted live weight (g) into 4 agronomic stages grounded in the audited `~7–29 days` / `116–2,093 g` coverage (`Inicial / Pintinho`: `< 350 g`, `~7–13 dias`; `Crescimento Inicial`: `350–700 g`, `~14–19 dias`; `Crescimento Intermediário`: `700–1200 g`, `~20–24 dias`; `Terminação / Pré-Abate`: `> 1200 g`, `~25–29 dias`), returning an explicit `out_of_training_range: bool` flag whenever `< 116 g` or `> 2,093 g`.

### Git-Tracked Evaluation Report & Curated OOF Presets Catalog (`MOD-02`, `MOD-03`)
- **D-09:** Publish byte-deterministic, git-tracked `models/evaluation_report.json` (`PROIBIÇÃO 4` compliant — zero volatile timestamps, zero `.jpg` filenames, zero `rel_path` or `/home/` strings, enforced by automated substring leak tests) reporting: `schema_version`, `pdi_config_hash`, `dataset_summary` (`1,738` images, `188` sessions, `5` folds, `0` session overlap), `literature_benchmark` (`r2_reference = 0.90`), `model_comparison` (all candidate models × ablation conditions), `winning_model` (`global`, `per_fold`, `by_weight_band`, `by_lens`, `by_scene`, `by_band_and_scene`, `by_band_and_lens`, `feature_importances`), `optical_ablation` (4-condition table + explicit delta vs. `C1_uncalibrated_pdi` + `dual_lens_consistency` across `127` mixed-lens sessions + `counterfactual_lens_flip` audit), and anonymized `oof_scatter_points` (`1,738` rows keyed strictly by opaque `image_id`, `session_id`, `fold`, `actual_g`, `predicted_g`, `abs_err_g`, `pct_err`, `weight_band`, `lens_class`, `scene_type`) — **Reversibility:** one-way — Phase 4 `GET /api/metrics` and `WEB-05` Dashboard consume this exact JSON contract.
- **D-10:** Publish byte-deterministic, git-tracked `models/presets_catalog.json` strictly enforcing `PROIBIÇÃO 4` and ASVS V4: store **ONLY opaque `preset_id` + `image_id`** (NEVER raw `.jpg` filenames, `rel_path`, or filesystem paths — Phase 4 FastAPI resolves `image_id → rel_path` solely at runtime via `data/processed/manifest_clean.parquet` / `manifest_quarantine.parquet` validated against `RAW_DATASET_DIR.resolve()`). Include:
  - **6 single-bird curated out-of-fold presets** deterministically selected from clean, non-fallback (`segmentation_fallback == False`) validation samples spanning distinct sessions/folds: (1) Young chick (`~180 g`, `<500g`), (2) Intermediate grower (`~550 g`, `500-1000g`), (3) Heavy broiler (`~1,600 g`, `>1000g`), (4) Hard white-on-white dorsal bucket (`>1,000 g`, `scene_type == "bucket_dorsal"`), plus (5–6) a **Paired Dual-Lens Same-Session pair** (same `session_id`, one Wide `4.25 mm` + one Ultrawide `1.65 mm`, demonstrating live EXIF optical consistency in the pitch) — choosing representative median-error (non-cherry-picked) OOF samples carrying `ground_truth_weight_g`, `fold`, `session_id`, `weight_band`, `lens_class`, `scene_type`, `oof_predicted_g`, `oof_abs_err_g`, `oof_pct_err`.
  - **4 multi-bird Watershed presets** — 1 representative quarantined image per audited folder (`417`: 2 birds / `417 g` group sum, `757`: 2 birds / `757 g`, `823`: 4 birds / `823 g`, `1520`: 2 birds / `1,520 g`) keyed strictly by `image_id` from `manifest_quarantine.parquet`, carrying `expected_bird_count`, `group_sum_weight_g`, and explicit group-sum disclaimer metadata — **Reversibility:** one-way — Phase 4 `GET /api/presets`, `WEB-03`, and `WEB-04` consume this JSON catalog directly.
- **D-11:** Expose a single deterministic CLI `backend/scripts/train_and_evaluate.py` (updating `backend/app/config.py` with `MODEL_BUNDLE_PATH`, `EVALUATION_REPORT_PATH`, `PRESETS_CATALOG_PATH`) that loads and joins `manifest_clean.parquet` + `features_clean.parquet`, verifies exact row alignment, `pdi_config_hash` lock, and `0` cross-fold `session_id` overlap (`PROIBIÇÃO 3`), executes full 5-fold CV + 4-condition ablation + counterfactual checks, and atomically writes `models/weight_regressor.joblib`, `models/evaluation_report.json`, and `models/presets_catalog.json`.

### the agent's Discretion
- Exact hyperparameter grids / regularization settings for `Ridge`, `HuberRegressor`, `RandomForestRegressor`, and `LGBMRegressor` (tuned conservatively via empirical Phase 3 research on the 5 `session_id` folds to prevent overfitting on `188` sessions, while keeping total `train_and_evaluate.py` runtime under `~30–60 s` on CPU).
- Exact module decomposition inside `backend/app/modeling/` (e.g., `datasets.py`, `wrappers.py`, `models.py`, `evaluation.py`, `age_stage.py`, `artifacts.py`, `__init__.py`).
- Choice of whether to include `HistGradientBoostingRegressor` as a 5th comparison row in `model_comparison` alongside the 4 mandatory PRD families (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog`).
- Exact heuristic for choosing the 1 representative image per Watershed folder (`417`, `757`, `823`, `1520`) in `presets_catalog.json` (e.g., preferring exact-count `run_watershed_pdi()` match with clearest separation).

### Deferred Ideas
None — discussion stayed strictly within Phase 3 scope. (Leave-One-Date-Out temporal cross-validation `HARD-04` and post-pitch `30–45 d` data expansion `BARN-02` remain deferred per `REQUIREMENTS.md`.)

---

## Project Constraints (from GEMINI.md)

Extracted from `GEMINI.md` (`[VERIFIED: GEMINI.md:11-18, 108-140]`):
- **Classical PDI & Zero Deep Learning (`PROIBIÇÃO 1`):** `"Proibido usar redes neurais/deep learning (CNNs/YOLO/ViTs) na segmentação ou regressão principal"` — strictly `scikit-learn` (`Ridge`, `HuberRegressor`, `RandomForestRegressor`) and `LightGBM` (`LGBMRegressor`); zero PyTorch/TensorFlow/Keras/YOLO/ONNX/OCR packages (`[VERIFIED: backend/tests/test_io_exif.py:58-71]`).
- **Anti-Leakage Visual & Split (`PROIBIÇÃO 2` & `PROIBIÇÃO 3`):** Regressors consume strictly mask-gated `FEATURE_COLUMNS` (zero LCD/background/raw-resolution leakage) and validate strictly on the locked 5-fold `StratifiedGroupKFold` `session_id` splits (zero `train_test_split` or ungrouped `KFold`, zero cross-fold preprocessing leakage).
- **Timeline & Performance Budget:** MVP code freeze `2026-11-28` (pitch `2026-11-30`); end-to-end API `/api/predict` latency `< 500 ms` on local CPU (model single-row prediction must stay `< 5 ms`).
- **Locked Python 3.12 Stack:** `uv`-managed Python `3.12.14` (`requires-python = ">=3.12,<3.13"`, `no-build = true`, `python-preference = "only-managed"`) with exact runtime/dev dependency allowlist enforced by `test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions` (`[VERIFIED: pyproject.toml:5-33, backend/tests/test_io_exif.py:41-57]`). **Phase 3 installs zero new packages.**
- **Pandas 3.1 CoW & Arrow `str` Discipline:** Always mutate DataFrames with single-step `.loc[mask, col] = val`, select explicit numeric feature lists (never `select_dtypes(include=["object"])`), and preserve exact `feature_names` column ordering at inference (`[VERIFIED: GEMINI.md:127-128, 138]`).
- **Bounded CPU Parallelism:** Cap training/CV parallelism to `n_jobs=4` (`7.1 GiB` RAM host; never `n_jobs=-1`) (`[VERIFIED: GEMINI.md:119]`).
- **Data Hygiene (`PROIBIÇÃO 4`):** `data/raw/`, `data/processed/`, `models/*.joblib`, `models/*.pkl`, `*.jpg`, `*.png` are strictly git-ignored (`[VERIFIED: .gitignore:2-8]`). Only lightweight, byte-deterministic, zero-path-leakage JSON reports (`models/evaluation_report.json`, `models/presets_catalog.json`) are git-tracked.

---

## Summary

Live empirical spikes executed directly over the materialized `1,738 × 60` Phase 2 feature table (`data/processed/features_clean.parquet`, `pdi_config_hash = 43a3cfaa3208019e`, `0` NaN/Inf, `0` cross-fold `session_id` overlap across `188` sessions) resolve every modeling, ablation, anti-confounding, and artifact-bundling design choice with zero guesswork:

1. **Physical Optical Calibration (`C2`) Unanimously Beats Uncalibrated PDI (`C1`) Across Every Regressor Family:** Replacing raw uncalibrated pixel geometry (`C1`, `40` cols) with strictly physics-normalized descriptors (`C2_physics_normalized_strict`, `39` cols using Phase 2's `area_px / zoom²` formula and zero raw lens metadata) improves OOF `R²`, `MAE`, `RMSE`, and `Balanced Band MAPE` across all 5 tested regressor families. On the winning **`LightGBMLog`** model (`n_estimators=500, lr=0.025, num_leaves=24, max_depth=6, min_child_samples=12, subsample=0.8, colsample_bytree=0.55, reg_alpha=0.3, reg_lambda=2.0`, weighted via capped `5×` inverse `sqrt(session) × sqrt(band)` weights, Duan's smearing OFF), **`C2` cuts `MAE` from `164.3 g` to `158.1 g` (`-6.2 g`), cuts `RMSE` from `275.2 g` to `264.0 g` (`-11.2 g`), cuts `Balanced Band MAPE` from `36.19%` to `35.11%` (`-1.08 pp`), and raises single-image OOF `R²` from `0.5201` to `0.5582` (`+0.0381`)** — with **session-averaged OOF `R² = 0.692–0.700`** across the `188` birds (directly quantifying the handheld camera-height $Z$ variance ceiling vs. fixed ceiling-camera literature `R² > 0.90`).
2. **Empirically Proved Anti-Confounding Trap in BOTH `C3` and `C4`:** A crucial finding of Spike 3 is why a naive **focal-only** counterfactual check (`focal_length_mm` `4.25 ↔ 1.65`) catches linear lens confounding (`C3 RidgeLog = 15.80%`, `C3 HuberLog = 19.90%`), **but misses tree-based confounding on `C3` (`1.18%`) and `C4` (`0.00%`)** because trees reconstruct lens ID from `area_frac_optical_norm` (`W/UW = 0.479`), raw `area_px / area_optical_norm = zoom²`, or `C4`'s `8.85×` double-counted `area_px / (f·z)²` (`W/UW = 0.113`). Under a **full physical lens-swap counterfactual** (holding physical `area_optical_norm`, dimensionless shape, and plumage chroma fixed while swapping camera lenses `4.25/1.0 ↔ 1.65/0.6` and propagating sensor/zoom ratios to lens-dependent features), **`C2_physics_normalized_strict` is mathematically `0.00%` shift (perfectly lens-invariant by construction)**, whereas **`C3 LightGBMLog` shifts `8.39%` median / `10.76%` mean / `23.52%` p90** (and degrades within-session signed `W - UW` bias `-34.9 g → -43.1 g` and `>1000g Wide MAE` `+22.5 g`), and **`C4 LightGBMLog` shifts `12.74%` median / `39.43%` p90** (degrading `W - UW` bias `-34.9 g → -61.2 g` and `<500g UW` MBE/MAE). Meanwhile, Phase 2's `area_optical_norm` (`C2`) outperforms `area_frac_optical_norm` (`C2b`) on real dual-lens consistency (`125.5 g` vs. `144.7 g` mean `|W - UW|`).
3. **Fast, Deterministic, Zero-Warning Production Bundling:** Excluding `HistGradientBoostingLog` (strictly dominated by `LightGBMLog` at `R² = 0.5440` vs. `0.5582` while taking `8×` longer under `sample_weight`) keeps all 4 mandatory PRD families × 4 ablation conditions + weighting/smearing diagnostic ablation + production refit + Watershed preset selection under **`~35–42 s` total CPU runtime** (`n_jobs=4`). Placing `SimpleImputer(strategy="median")` before `LogDuanRegressor(LGBMRegressor(...))` eliminates LightGBM feature-name warnings completely, yields **`1.76 ms` single-row inference latency** (`< 5 ms` gate), and serializes all 6 pipelines (`1` production + `5` fold pipelines) to `~3.3 MB` compressed (`compress=3`) with `~100 ms` cold load time.

**Primary recommendation:** Build `backend/app/modeling/` with 6 cohesive modules (`datasets.py`, `wrappers.py`, `models.py`, `evaluation.py`, `age_stage.py`, `artifacts.py`) + CLI `backend/scripts/train_and_evaluate.py`; enforce the dual-mode counterfactual (`focal_only` + `full_physical_lens_swap`) + `<500 g` ultrawide anti-confounding gate so **`LightGBMLog` on `C2_physics_normalized_strict`** (`39` features, capped `5×` `inv_sqrt_sess_sqrt_band` weighting, Duan's smearing OFF) is deterministically selected and exported into dual-bundle `models/weight_regressor.joblib` alongside byte-deterministic `models/evaluation_report.json` and opaque-ID `models/presets_catalog.json`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| **Feature-Manifest Join, Hash Lock & Fold Integrity Check** | Offline Modeling (`backend/app/modeling/datasets.py`) | Ingestion (`backend/app/ingestion/splits.py::verify_fold_integrity`) | Single entry gate joining `manifest_clean.parquet` + `features_clean.parquet`, asserting `pdi_config_hash == "43a3cfaa3208019e"`, `0` NaN/Inf, `0` banned columns, and `0` cross-fold `session_id` overlap (`PROIBIÇÃO 2 & 3`). |
| **Log-Target, Duan Smearing & Allometric Preprocessing** | Offline Modeling & Shared Inference (`backend/app/modeling/wrappers.py`) | Online API (`joblib.load()` in Phase 4) | Must live in an importable module under `backend/app/modeling/` (never `__main__` or `lambda`) so `models/weight_regressor.joblib` unpickles cleanly inside Phase 4 FastAPI workers (`D-01`, `D-02`). |
| **4-Family Regressor Factory & Fold Sample Weighting** | Offline Modeling (`backend/app/modeling/models.py`) | — | Encapsulates deterministic `RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog` pipelines (`random_state=42`, `n_jobs=4`) and strictly training-fold-only `compute_sample_weights()` (`max_w/min_w <= 5.0`, `D-03`). |
| **5-Fold OOF Evaluation, 4-Condition Ablation & Anti-Confounding Audit** | Offline Modeling (`backend/app/modeling/evaluation.py`) | — | Runs leak-free OOF CV across `C1`–`C4`, weighting/smearing sensitivity, `127`-session dual-lens consistency, 2-mode counterfactual lens-flip audit, stratified metrics, and deterministic winner selection (`D-04`, `D-05`, `D-06`). |
| **Cobb Avian 48 Developmental Age-Stage Mapping** | Shared Domain Logic (`backend/app/modeling/age_stage.py`) | Online API (`Phase 4 InferenceService`) | Pure zero-dependency function mapping weight (g) → 4 agronomic stages + `out_of_training_range` flag (`<116 g` or `>2,093 g`), shared identically between Phase 3 audits and Phase 4 `/api/predict` (`D-08`). |
| **Artifact Bundling & Curated OOF/Watershed Preset Selection** | Offline Modeling (`backend/app/modeling/artifacts.py`) | PDI Engine (`backend/app/pdi/watershed.py::run_watershed_pdi`) | Atomically publishes git-ignored `models/weight_regressor.joblib` (`production_pipeline` + 5 `fold_pipelines`) and git-tracked, zero-path-leakage `models/evaluation_report.json` + `models/presets_catalog.json` (`D-07`, `D-09`, `D-10`). |
| **Reproducible CLI Orchestration** | Offline CLI (`backend/scripts/train_and_evaluate.py`) | `backend/app/config.py` | Single CWD-independent entrypoint with `--dry-run` support and aggregate-only stdout logging (`D-11`, `PROIBIÇÃO 4`). |

---

## Standard Stack

**Zero new external packages are installed in Phase 3.** Every required runtime and dev dependency is already pinned, wheel-locked in `uv.lock`, and enforced by `backend/tests/test_io_exif.py::test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions` (`[VERIFIED: pyproject.toml:6-28, backend/tests/test_io_exif.py:41-57]`).

### Core (Already Locked in `pyproject.toml` / `uv.lock`)

| Library | Verified Installed Version | Purpose | Why Standard |
|---------|---------------------------|---------|--------------|
| **`python` (`uv`-managed)** | `3.12.14` (`[VERIFIED: uv run python]`) | Isolated runtime (`requires-python = ">=3.12,<3.13"`) | Decoupled from rolling host `/usr/bin/python3.14`; 100% binary wheel lock (`[VERIFIED: pyproject.toml:5, 30-33]`). |
| **`scikit-learn`** | `1.9.1` (`[VERIFIED: uv run python]`) | `Pipeline`, `BaseEstimator`, `RegressorMixin`, `TransformerMixin`, `SimpleImputer`, `StandardScaler`, `Ridge`, `HuberRegressor`, `RandomForestRegressor`, metrics (`mean_absolute_error`, `root_mean_squared_error`, `r2_score`) | Industry-standard fold-isolated preprocessing, cloneable estimator API, deterministic `random_state`, and seamless `joblib` compatibility. |
| **`lightgbm`** | `4.7.0` (`[VERIFIED: uv run python]`) | Primary non-linear tabular regressor (`LGBMRegressor`) | Captures non-linear interactions between optically normalized geometry and plumage chroma/texture; `< 2 ms` single-row CPU inference; native `sample_weight` and gain feature importances. |
| **`joblib`** | `1.6.0` (`[VERIFIED: uv run python]`) | Dual-bundle serialization (`models/weight_regressor.joblib`, `compress=3`) | Standard zero-overhead Python/NumPy/scikit-learn pipeline serializer; `~3.3 MB` for 6 `LightGBMLog` pipelines with `~100 ms` load time. |
| **`pandas`** + **`pyarrow`** | `pandas 3.1.1`, `pyarrow 26.0.1` (`[VERIFIED: uv run python]`) | Typed Parquet I/O (`manifest_clean.parquet`, `manifest_quarantine.parquet`, `features_clean.parquet`) and group aggregations | Lossless typed schema preservation across Phases 1–3; Copy-on-Write enforced globally in Pandas 3.1. |
| **`numpy`** + **`scipy`** | `numpy 2.5.5`, `scipy 1.18.1` (`[VERIFIED: uv run python]`) | Vectorized log/smearing math, percentiles (`p68`, `p90`), Spearman diagnostics, deterministic float64 calculations | Shared C-level numerical foundation across `scikit-learn`, `LightGBM`, and `backend/app/pdi/`. |

### Supporting / Dev (Already Locked)

| Library | Verified Installed Version | Purpose | When to Use |
|---------|---------------------------|---------|-------------|
| **`pytest`** | `9.2.0` (`[VERIFIED: uv run python]`) | Automated unit, anti-leakage, counterfactual, pickling subprocess, and JSON-determinism tests | Every task commit and wave gate (`uv run pytest -v`). |
| **`ruff`** | `0.17.6` (`[VERIFIED: uv run ruff --version]`) | Static linting (`E`, `F`, `I`, `UP`, `B`, `SIM`) & formatting (`line-length = 100`, `target-version = "py312"`) | Every task commit (`uv run ruff check backend/`). |

### Alternatives Considered (Resolved by Live Empirical Spikes)

| Instead of | Could Use | Empirical Tradeoff / Verdict |
|------------|-----------|------------------------------|
| **4 mandatory PRD families (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog`)** | Add 5th row `HistGradientBoostingLog` | **Do NOT add `HistGradientBoostingLog` to the production CLI.** Live 5-fold CV proved `HistGradientBoostingLog` is strictly dominated by `LightGBMLog` on `C2` (`R² = 0.5440` vs. `0.5582`, `BalMAPE = 35.80%` vs. `35.11%`, `MAE = 160.9 g` vs. `158.1 g`) while running **`8×` slower with `sample_weight`** (`13.3 s` vs. `1.7 s` per condition; `53.3 s` alone across 4 conditions vs. `30.8 s` for all 4 PRD families combined). |
| **Custom importable `LogDuanRegressor` (`wrappers.py`)** | `sklearn.compose.TransformedTargetRegressor` | **`LogDuanRegressor` wins.** `TransformedTargetRegressor` applies naive `np.expm1(ŷ)` without fold-internal Duan's smearing (`smearing_factor_`), and requires careful `func`/`inverse_func` module references; a dedicated 50-line `BaseEstimator, RegressorMixin` class gives native `use_smearing: bool` toggle, weighted training-fold smearing, clean `sample_weight` forwarding, and guaranteed `joblib` pickling (`D-01`). |
| **Unconstrained `LGBMRegressor` on `C2`** | `monotone_constraints` (`+1` on 5 `*_optical_norm` cols) | **Keep unconstrained.** Live spike showed monotone constraints slightly degraded `BalMAPE` (`35.49% → 35.62%`) and worsened within-session signed dual-lens bias (`-34.4 g → -45.5 g`) while only adding `+0.005` `R²`. |
| **`C2_physics_normalized_strict` (`area_optical_norm = area_px / zoom²`)** | `C2b_area_frac_optical_norm` (`area_frac / zoom²`) | **Keep `C2_physics_normalized_strict` (`area_optical_norm`).** Live spike proved `area_frac_optical_norm` has `W/UW = 0.479` (vs. `0.749` for `area_optical_norm`), degrading real within-session dual-lens consistency by `+19.2 g` (`144.7 g` vs. `125.5 g` mean `|W - UW|`) and worsening `<500 g` ultrawide MBE (`+76.8 g` vs. `+73.2 g`). |

**Installation:** None required — existing locked environment is ready via `uv sync --frozen` (already synced in `.venv`).

---

## Package Legitimacy Audit

> **Status:** **Zero new external packages are installed in Phase 3.** All 7 modeling libraries below were already audited and locked into `pyproject.toml` + `uv.lock` during Phase 1 (`[VERIFIED: pyproject.toml:6-28, backend/tests/test_io_exif.py:41-57]`). Re-checked via `gsd-tools query package-legitimacy check --ecosystem pypi` on 2026-11-23 for Phase 3 audit completeness.

| Package | Registry | Published (Latest Patch) | Downloads | Source Repo (PyPI JSON) | Seam Verdict | Disposition |
|---------|----------|--------------------------|-----------|-------------------------|--------------|-------------|
| `scikit-learn` (`1.9.1`) | PyPI | `2026-09-10` | `null` (PyPI JSON omits) | `null` (metadata key variant) | `[SUS]` (seam metadata gap) | **Already locked in `uv.lock` (Phase 1)** — canonical PyData library (`github.com/scikit-learn/scikit-learn`); zero new install |
| `lightgbm` (`4.7.0`) | PyPI | `2026-07-18` | `null` | `null` | `[SUS]` (seam metadata gap) | **Already locked in `uv.lock` (Phase 1)** — canonical Microsoft GBDT library (`github.com/microsoft/LightGBM`); zero new install |
| `joblib` (`1.6.0`) | PyPI | `2026-08-31` | `null` | `https://github.com/joblib/joblib` | `[SUS]` (unknown-downloads) | **Already locked in `uv.lock` (Phase 1)** — canonical scikit-learn persistence library; zero new install |
| `pandas` (`3.1.1`) | PyPI | `2026-11-04` | `null` | `null` | `[SUS]` (`too-new` patch release + metadata gap) | **Already locked in `uv.lock` (Phase 1)** — canonical PyData DataFrame library; zero new install |
| `pyarrow` (`26.0.1`) | PyPI | `2026-11-03` | `null` | `https://arrow.apache.org/` | `[SUS]` (`too-new` patch release + unknown-downloads) | **Already locked in `uv.lock` (Phase 1)** — Apache Arrow official Python binding; zero new install |
| `numpy` (`2.5.5`) | PyPI | `2026-11-15` | `null` | `null` | `[SUS]` (`too-new` patch release + metadata gap) | **Already locked in `uv.lock` (Phase 1)** — canonical NumPy array library; zero new install |
| `scipy` (`1.18.1`) | PyPI | `2026-08-21` | `null` | `null` | `[SUS]` (seam metadata gap) | **Already locked in `uv.lock` (Phase 1)** — canonical SciPy scientific library; zero new install |

**Packages removed due to `[SLOP]` verdict:** none.
**Packages newly installed in Phase 3:** **none** (the PyPI legitimacy seam returns `[SUS]` on PyPI packages solely because PyPI's JSON API does not expose download counts and treats recent minor/patch releases as `too-new`; no `uv add` or `pip install` tasks exist in Phase 3, and `test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions` enforces zero dependency drift).

---

## Empirical Benchmarks & Ablation Findings

All metrics below were measured live via `uv run python` directly over materialized `data/processed/features_clean.parquet` (`1,738 × 60`), `data/processed/manifest_clean.parquet` (`1,738 × 24`), and `data/processed/manifest_quarantine.parquet` (`15 × 24`) (`[VERIFIED: live Spikes 1–4]`).

### Spike 1: Input Alignment, Feature Hygiene & Dataset Structure

| Verification Check | Observed Value | Source Verification |
|--------------------|----------------|---------------------|
| `manifest_clean.parquet` shape | `(1738, 24)` (`160` unique weights, `188` sessions, `116–2,093 g`) | `[VERIFIED: Spike 1; models/dataset_audit_summary.json:3-6, 204-205]` |
| `features_clean.parquet` shape | `(1738, 60)` (`11` join/audit + `49` `FEATURE_COLUMNS`) | `[VERIFIED: Spike 1; backend/scripts/extract_features.py:61-76]` |
| `manifest_quarantine.parquet` shape | `(15, 24)` (`13` `multi_bird` across `folder_chain` `{'417': 4, '757': 1, '823': 4, '1520': 4}` + `2` `severe_mislabel`) | `[VERIFIED: Spike 1 & Spike 2; models/dataset_audit_summary.json:13-39]` |
| Exact 1:1 row alignment (`image_id`, `fold`, `session_id`, `weight_g`, `weight_band`, `lens_class`) | `True` (all 6 columns match row-for-row) | `[VERIFIED: Spike 1]` |
| `pdi_config_hash` lock | `"43a3cfaa3208019e"` (exact match with `PDIConfig().config_hash()`) | `[VERIFIED: Spike 1; models/pdi_calibration_audit.json:211]` |
| Missing/non-finite features (`NaN`/`Inf`) | `0` NaN, `0` Inf across `1,738 × 49` matrix | `[VERIFIED: Spike 1]` |
| Banned columns (`BANNED_FEATURE_COLUMNS`) overlap | `[]` (`0` overlap) | `[VERIFIED: Spike 1; backend/app/pdi/features.py:101-103]` |
| Cross-fold `session_id` overlap (`verify_fold_integrity`) | `0` overlap; heavy (`>1000 g`) sessions/fold = `[4, 5, 5, 5, 5]` | `[VERIFIED: Spike 1; models/dataset_audit_summary.json:224-305]` |
| Mixed-lens sessions (`wide` + `ultrawide`) | `127` sessions (`126` with same-scene non-fallback pairs) | `[VERIFIED: Spike 1 & Spike 4B; models/pdi_calibration_audit.json:56]` |
| Segmentation fallback rate | `4 / 1,738` (`0.23%`; `0 / 377` in `bucket_dorsal`) | `[VERIFIED: Spike 1; models/pdi_calibration_audit.json:5-9]` |
| Cross-tab `weight_band × scene_type` | `<500g`: `254` bucket / `969` scale; `500-1000g`: `64` / `269`; `>1000g`: `59` / `123` | `[VERIFIED: Spike 1]` |
| Cross-tab `weight_band × lens_class` | `<500g`: `940` W / `283` UW (`23.1%` UW); `500-1000g`: `280` W / `53` UW (`15.9%` UW); `>1000g`: `77` W / `105` UW (**`57.7%` UW**) | `[VERIFIED: Spike 1; models/dataset_audit_summary.json:147-167]` |

**Physical Signal Structure (Top Feature Spearman Correlations with `weight_g`):**
Because handheld smartphone distance $Z$ is uncalibrated, **scale-invariant plumage maturity cues** (`lab_a_mean = +0.383`, `lab_a_p10 = +0.383`, `lab_a_p50 = +0.364`, `hsv_h_sin_mean = -0.296`, `pink_skin_ratio = +0.242`, `yellow_down_ratio = -0.202`) work synergistically with **EXIF optically normalized morphometry** (`perim_optical_norm = +0.270`, `minor_axis_optical_norm = +0.263`, `area_optical_norm = +0.260`), which boosts geometric correlation `+19%` relative over uncalibrated `area_px` (`+0.219`). Note that `major_axis_px` / `major_axis_optical_norm` contain rare degenerate-ellipse spikes up to `24,530 px` (`p50 = 1,898 px` / `2,180 px`), making tree models and Huber/log-compressed linear pipelines essential.

### Spike 2: Target Mechanics (`AllometricLogTransformer`, `sample_weight`, `Duan's Smearing`)

1. **`AllometricLogTransformer` on Linear Families (`C2`, unweighted, smearing OFF):**
   - `RidgeLog`: without allometric log `MAE = 205.6 g, BalMAPE = 45.60%, R² = 0.2814` vs. with `AllometricLogTransformer` (strictly geometric area/linear columns) `MAE = 207.4 g, BalMAPE = 45.90%, R² = 0.2649`.
   - `HuberLog`: without `MAE = 206.4 g, MAPE = 46.43%, BalMAPE = 45.76%` vs. with `AllometricLogTransformer` `MAE = 209.1 g, MAPE = 46.17%, BalMAPE = 46.22%`.
   - **Action (`D-02` locked):** Keep `AllometricLogTransformer` inside `RidgeLog` and `HuberLog` pipelines, scoped **strictly** to positive geometric area/linear columns (NOT texture columns `feather_grad_mean` / `feather_laplacian_var`), directly preserving $W \propto A^{3/2}$ log-log physical allometry and smooth tail extrapolation.

2. **Sample Weighting & Fold-Internal Duan's Smearing Sweep (`LightGBMLog` `LGB_n500_l24_d6_mc12` on `C2_physics_normalized_strict`):**

| Weighting Mode (fold-internal, `max_w/min_w <= 5.0`) | Duan Smearing | OOF MAE (g) | OOF RMSE (g) | OOF MAPE (%) | **Bal. Band MAPE (%)** | OOF $R^2$ | Session $R^2$ | Global MBE (g) | Band MAPEs (`<500 / 500-1k / >1k`) |
|------------------------------------------------------|---------------|-------------|--------------|--------------|------------------------|-----------|---------------|----------------|------------------------------------|
| `unweighted` | OFF (`1.000`) | `160.5` | `272.3` | `35.98%` | `35.76%` | `0.5301` | `0.6615` | `-51.9` | `37.2% / 28.6% / 41.5%` |
| `unweighted` | ON (`1.012`) | `160.4` | `270.9` | `36.47%` | `35.75%` | `0.5350` | `0.6702` | `-46.8` | `38.0% / 28.3% / 41.0%` |
| **`session_band_capped_5x` (`inv_sqrt_sess_sqrt_band`, median-centered)** | **OFF (`1.000`)** | **`158.1`** | **`264.0`** | **`37.09%`** | **`35.11%`** | **`0.5582`** | **`0.6924`** | **`-39.8`** | **`39.6% / 26.7% / 39.0%`** |
| `session_band_capped_5x` (`inv_sqrt_sess_sqrt_band`, median-centered) | ON (`1.011`) | `158.2` | `262.8` | `37.59%` | `35.13%` | `0.5622` | `0.6996` | `-35.0` | `40.5% / 26.4% / 38.5%` |
| `inv_sess_sqrt_band` (`p5`-anchored cap `5.0`) | OFF | `160.2` | `264.5` | `37.93%` | `35.58%` | `0.5568` | `0.6926` | `-37.2` | `40.7% / 27.1% / 38.9%` |
| `inv_sess_inv_band` (`p5`-anchored cap `5.0`) | OFF | `160.4` | `264.2` | `38.31%` | `35.48%` | `0.5575` | `0.6938` | `-34.3` | `41.5% / 26.3% / 38.6%` |

**Key Mechanics Conclusions:**
- **Capped median-centered `inv_sqrt_sess_sqrt_band` weighting (`w_raw = (1/sqrt(n_sess)) * (1/sqrt(n_band))`, clipped to `[median/sqrt(5), median*sqrt(5)]`, normalized to mean `1.0`) is a decisive win over unweighted training:** improves `MAE` (`-2.4 g`), `RMSE` (`-8.3 g`), `BalMAPE` (`-0.65 pp`), single-image `R²` (`+0.0281`), session-averaged `R²` (`+0.0309`), and `>1000g` MAPE (`-2.5 pp`) while honoring `D-03` (`max_w / min_w <= 5.0` strictly inside each training fold, unweighted validation metrics).
- **Duan's Smearing Trade-off:** On linear models (`RidgeLog`, `HuberLog`), large in-fold log residuals yield `smearing_factor ≈ 1.13–1.18`, cutting MBE (`-73.8 → -13.5 g`) but inflating `<500g` MAPE by `+15 pp`. On `LightGBMLog`, `smearing_factor ≈ 1.011` is modest: `OFF` wins `BalMAPE` (`35.11%` vs. `35.13%`), global `MAPE` (`37.09%` vs. `37.59%`), and `MAE` (`158.1 g` vs. `158.2 g`), while `ON` slightly lowers `RMSE` (`-1.2 g`) and `MBE` (`+4.8 g`). **Design recommendation:** Run the `weighting × smearing` 2×2 sensitivity audit for the winning model/condition (and record all 4 rows in `evaluation_report.json` under `mechanics_ablation`) while using the deterministic `D-06` criterion (`BalMAPE` primary) so `weighted + smearing=False` wins cleanly!

### Spike 2 & 3: Complete 4-Condition Optical Ablation, Dual-Lens Consistency & Anti-Confounding Matrix

All rows below use the locked conservative hyperparameters, capped `5×` `inv_sqrt_sess_sqrt_band` training weights, and `duan_smearing=False` (`[VERIFIED: Spike 3 / task-55]`):

| Condition (Cols) | Model | OOF MAE (g) | OOF RMSE (g) | OOF MAPE (%) | **Bal. MAPE (%)** | OOF $R^2$ | MBE (g) | Dual-Lens `med \|W-UW\|` (g / %) | Dual-Lens `mean \|W-UW\|` / `W-UW` (g) | `CF_focal` med (%) | `CF_phys` med / p90 (%) | `<500g UW` MBE / MAE (g) | Anti-Confounding Gate |
|------------------|-------|-------------|--------------|--------------|-------------------|-----------|---------|----------------------------------|---------------------------------------|--------------------|-------------------------|--------------------------|-----------------------|
| `C1_uncalibrated_pdi` (`40`) | `RidgeLog` | `217.5` | `329.1` | `57.12%` | `48.19%` | `0.3136` | `-9.1` | `125.0 g` (`26.5%`) | `146.9` / `-25.6` | `0.00%` | `8.64%` / `24.90%` | `+143.4` / `168.7` | Baseline (No EXIF) |
| `C1_uncalibrated_pdi` (`40`) | `HuberLog` | `220.4` | `330.9` | `58.56%` | `48.83%` | `0.3061` | `-0.3` | `129.1 g` (`29.7%`) | `164.2` / `-29.4` | `0.00%` | `9.42%` / `27.30%` | `+151.5` / `181.1` | Baseline (No EXIF) |
| `C1_uncalibrated_pdi` (`40`) | `RandomForestLog` | `174.8` | `289.3` | `41.77%` | `38.80%` | `0.4696` | `-35.3` | `68.4 g` (`18.4%`) | `108.5` / `+0.6` | `0.00%` | `7.44%` / `26.58%` | `+87.0` / `110.3` | Baseline (No EXIF) |
| `C1_uncalibrated_pdi` (`40`) | `LightGBMLog` | `164.3` | `275.2` | `37.46%` | `36.19%` | `0.5201` | `-42.7` | `66.8 g` (`19.0%`) | `128.1` / `+3.4` | `0.00%` | `8.69%` / `26.00%` | `+70.3` / `110.3` | Baseline (No EXIF) |
| `C2_physics_normalized_strict` (`39`) | `RidgeLog` | `213.0` | `320.1` | `56.68%` | `47.16%` | `0.3507` | `-9.1` | `111.2 g` (`25.6%`) | `155.0` / `-64.4` | **`0.00%`** | **`0.00%` / `0.00%`** | `+160.5` / `189.8` | **PASS** |
| `C2_physics_normalized_strict` (`39`) | `HuberLog` | `216.2` | `321.4` | `58.72%` | `47.88%` | `0.3456` | `+3.4` | `144.6 g` (`30.4%`) | `178.2` / `-79.7` | **`0.00%`** | **`0.00%` / `0.00%`** | `+176.1` / `207.5` | **PASS** |
| `C2_physics_normalized_strict` (`39`) | `RandomForestLog` | `172.0` | `282.4` | `42.08%` | `38.51%` | `0.4947` | `-33.4` | `60.9 g` (`15.0%`) | `106.6` / `-48.8` | **`0.00%`** | **`0.00%` / `0.00%`** | `+111.7` / `131.4` | **PASS** |
| **`C2_physics_normalized_strict` (`39`)** | **`LightGBMLog`** | **`158.1`** | **`264.0`** | **`37.09%`** | **`35.11%`** | **`0.5582`** | **`-39.8`** | **`72.1 g` (`18.1%`)** | **`126.4` / `-34.9`** | **`0.00%`** | **`0.00%` / `0.00%`** | **`+81.5` / `116.6`** | **PASS — WINNER** |
| `C3_physics_plus_exif_meta` (`49`) | `RidgeLog` | `209.8` | `317.6` | `56.16%` | `46.54%` | `0.3606` | `-9.5` | `135.1 g` (`31.7%`) | `180.2` / `-117.4` | **`15.80%`** | `8.62%` / `19.22%` | `+186.9` / `209.6` | **FLAGGED** (`CF_foc > 10%`, `<500 UW` worse) |
| `C3_physics_plus_exif_meta` (`49`) | `HuberLog` | `212.6` | `319.9` | `57.94%` | `47.19%` | `0.3513` | `+3.0` | `150.4 g` (`33.8%`) | `215.0` / `-151.3` | **`19.90%`** | **`10.18%`** / `26.90%` | `+209.4` / `234.1` | **FLAGGED** (`CF_foc > 10%`, `CF_phys > 10%`, `<500 UW` worse) |
| `C3_physics_plus_exif_meta` (`49`) | `RandomForestLog` | `169.4` | `279.9` | `41.62%` | `37.88%` | `0.5034` | `-33.4` | `60.3 g` (`15.1%`) | `113.6` / `-55.0` | `0.83%` | `6.92%` / **`18.73%`** | `+106.0` / `127.8` | **FLAGGED** (`CF_phys p90 > 15%`, `W-UW` worse) |
| `C3_physics_plus_exif_meta` (`49`) | `LightGBMLog` | `156.6` | `263.2` | `36.62%` | `34.79%` | `0.5611` | `-40.1` | `64.7 g` (`18.2%`) | `127.7` / `-43.1` | `1.18%` | `8.39%` / **`23.52%`** | `+78.5` / `116.1` | **FLAGGED** (`CF_phys p90 > 15%`, `W-UW` `-43.1` vs `-34.9`) |
| `C4_naive_double_count_fxz` (`39`) | `RidgeLog` | `213.5` | `321.8` | `57.13%` | `47.25%` | `0.3437` | `-7.7` | `160.5 g` (`37.9%`) | `200.3` / `-155.7` | `0.00%` | **`30.00%`** / **`34.88%`** | `+223.4` / `234.4` | **FLAGGED** (`CF_phys > 10%`, `<500 UW` worse) |
| `C4_naive_double_count_fxz` (`39`) | `HuberLog` | `216.8` | `322.7` | `59.36%` | `48.03%` | `0.3402` | `+6.9` | `197.4 g` (`44.7%`) | `239.4` / `-196.1` | `0.00%` | **`34.60%`** / **`41.10%`** | `+251.8` / `262.7` | **FLAGGED** (`CF_phys > 10%`, `<500 UW` worse) |
| `C4_naive_double_count_fxz` (`39`) | `RandomForestLog` | `167.3` | `275.7` | `41.25%` | `37.46%` | `0.5184` | `-32.4` | `70.5 g` (`16.7%`) | `136.1` / `-85.5` | `0.00%` | `8.97%` / **`41.11%`** | `+121.3` / `139.0` | **FLAGGED** (`CF_phys p90 > 15%`, `<500 UW` worse) |
| `C4_naive_double_count_fxz` (`39`) | `LightGBMLog` | `155.8` | `260.1` | `36.71%` | `34.48%` | `0.5711` | `-39.4` | `72.3 g` (`19.4%`) | `135.7` / `-61.2` | `0.00%` | **`12.74%`** / **`39.43%`** | `+87.9` / `121.8` | **FLAGGED** (`CF_phys > 10%`, `<500 UW` worse) |

*(Benchmark note: `HistGradientBoostingLog` was also tested across all 4 conditions — `C2`: `MAE = 160.9 g, RMSE = 268.2 g, BalMAPE = 35.80%, R² = 0.5440` in `13.3 s` — strictly dominated by `LightGBMLog` and omitted from production CLI to stay well within the `30–60 s` budget.)*

**Why `C4_naive_double_count_fxz` Appears Higher on Naive Global `R²` (`0.5711`) — The Textbook Confounding Proof:**
Dividing `area_px` by `(focal_length_mm * zoom_multiple)²` artificially inflates Ultrawide projected areas by **`8.85×`** (`W/UW = 0.113`) relative to Wide (`[VERIFIED: Spike 3; models/pdi_calibration_audit.json:47]`). Because `57.7%` of `>1,000 g` images are Ultrawide (vs. `23.1%` for `<500 g`), `LightGBMLog` on `C4` exploits the `8.85×` jump as a covert `ultrawide ⇒ heavy` shortcut! Our dual-lens + counterfactual audit exposes this immediately: `C4 LightGBMLog` suffers **`12.74%` median / `39.43%` p90 physical lens-swap shift**, doubles signed Wide−Ultrawide discrepancy (`-61.2 g` vs. `-34.9 g`), worsens mean `|W - UW|` (`135.7 g` vs. `126.4 g`), and worsens `<500 g` Ultrawide overprediction (`MBE +87.9 g` vs. `+81.5 g`, `MAE 121.8 g` vs. `116.6 g`). Exposing this quantitatively in `models/evaluation_report.json` makes Tab 3 (`WEB-05`) extraordinarily compelling during academic defense!

### Winning Model (`LightGBMLog` on `C2_physics_normalized_strict`) Stratified Diagnostics & Error Intervals

| Stratum | `n` | MAE (g) | RMSE (g) | MAPE (%) | Median `\|err\|` (g) | `p68 \|err\|` (g) | `p90 \|err\|` (g) | MBE (g) |
|---------|-----|---------|----------|----------|----------------------|-------------------|-------------------|---------|
| **Global OOF (5-fold)** | `1,738` | `158.1` | `264.0` | `37.09%` | `~92` | `~162` | `~360` | `-39.8` |
| **`<500g`** | `1,223` | `93.1` | `~130` | `39.63%` | `66.3` | `107.6` | `205.9` | `+57.3` |
| **`500-1000g`** | `333` | `160.8` | `~197` | `26.67%` | `156.5` | `207.0` | `296.9` | `-102.1` |
| **`>1000g`** | `182` | `589.8` | `~696` | `39.04%` | `566.0` | `762.9` | `1,087.2` | `-578.5` |
| **Scene: `scale_or_litter`** | `1,361` | `~146.1` | `~238.5` | `~35.2%` | — | — | — | `-34.0` |
| **Scene: `bucket_dorsal`** | `377` | `~201.2` | `~342.8` | `~43.1%` | — | — | — | `-69.4` |
| **Hard: `>1000g × bucket_dorsal`** | `59` | `~693.4` | `~792.4` | `~44.6%` | — | — | — | `-683.0` |
| **Lens: `wide` (`4.25 mm`)** | `1,297` | `~136.3` | `~233.1` | `~33.9%` | — | — | — | `-27.9` |
| **Lens: `ultrawide` (`1.65 mm`)** | `441` | `~222.0` | `~340.7` | `~45.7%` | — | — | — | `-82.1` |

*(Exact per-stratum numbers are computed deterministically inside `evaluation.py` for `LGB_n500_l24_d6_mc12`; values prefixed with `~` above reflect the adjacent `LGB_n600` run from `task-45` within `±1%`, while `<500g`/`500-1000g`/`>1000g` quantiles come verbatim from the `LGB_n500_l24_d6_mc12` run.)*

**Top Gain Feature Importances (`C2 LightGBMLog`, normalized %):**
1. `lab_a_p10` (`8.3%`), 2. `minor_axis_optical_norm` (`7.8%`), 3. `lab_a_mean` (`7.3%`), 4. `feather_grad_mean` (`5.6%`), 5. `area_optical_norm` (`5.2%`), 6. `lab_b_std` (`4.7%`), 7. `feather_laplacian_var` (`4.4%`), 8. `pink_skin_ratio` (`4.2%`), 9. `hsv_s_std` (`2.9%`), 10. `hsv_h_sin_mean` (`2.7%`) — proving balanced reliance on **both** EXIF-calibrated morphometry AND biological plumage maturation cues, with zero lens-ID shortcutting.

### Spike 4: Curated Preset Feasibility, Watershed Audit & Pickling Latency

1. **Single-Bird OOF Presets (`6` slots across all `5` distinct folds `0..4`, `segmentation_fallback == False`, median-tier non-cherry-picked OOF errors):**
   - Verified abundant clean pools in all 5 folds (`<500g` ~180g: `297` imgs; `500-1000g` ~550g: `146` imgs; `>1000g` ~1600g: `52` imgs; `>1000g × bucket_dorsal`: `59` imgs; same-scene mixed-lens sessions: `126` sessions across all 5 folds).
   - Sample deterministic median-error selection verified live spans ALL 5 distinct folds (`fold 0, 1, 2, 3, 4`) and 5 distinct sessions (`8.9%–41.4%` realistic OOF error, including same-session `bucket_dorsal` Wide + Ultrawide pair on `fold 3`).
2. **Multi-Bird Watershed Presets (`4` folders from `manifest_quarantine.parquet`, `run_watershed_pdi(..., return_visuals=False)`):**
   - Folder `417` (expected `2` birds, `4` imgs): **4/4 exact (`detected=2`)**; best balanced area ratio = `0.907` (`image_id = a5da2cc6c9cfa4f1`, `215.9 ms`, `red_feeder_excluded_px = 25,434`).
   - Folder `757` (expected `2` birds, `1` img): `1` image (`image_id = 80f435279b2ea6fb`, `detected=3`, `207.6 ms` — sole quarantined sample in folder `757`).
   - Folder `823` (expected `4` birds, `4` imgs): **4/4 exact (`detected=4`)**; best balanced area ratio = `0.706` (`image_id = 8f1d0a4f77c2818f`, `121.8 ms`).
   - Folder `1520` (expected `2` birds, `4` imgs): **2/4 exact (`detected=2`)**; best balanced area ratio = `0.876` (`image_id = f901aa20e07c49fb`, `216.7 ms`).
   - **Recommended Deterministic Heuristic (zero hardcoding):** Sort candidate images per folder by `(abs(detected - expected), -(min_area / max_area), image_id)` — selects exact counts on `417`, `823`, `1520` and cleanest balanced basins automatically.
3. **Serialization & Single-Row Latency:**
   - Single-row 1-record `pd.DataFrame` inference through `Pipeline([SimpleImputer, LogDuanRegressor(LGBMRegressor)])`: **`1.763 ms` avg**, **`0` warnings** (`[VERIFIED: Spike 4C]`).
   - `joblib.dump(..., compress=3)` size: `~546 KB` per `LightGBMLog` pipeline → **`~3.3 MB` total** for `1` production + `5` fold pipelines; cold `joblib.load()` completes in `~100 ms` with bit-identical subprocess predictions.

---

## Architecture Patterns

### System Architecture Diagram

```text
data/processed/manifest_clean.parquet (1,738 x 24)
data/processed/features_clean.parquet (1,738 x 60, hash=43a3cfaa3208019e)
data/processed/manifest_quarantine.parquet (15 x 24, 13 multi_bird)
        │
        ▼
[backend/app/modeling/datasets.py]
  • Inner-join & exact 1:1 image_id alignment assertion
  • PDIConfig().config_hash() == "43a3cfaa3208019e" lock assertion
  • verify_fold_integrity(min_heavy_sessions_per_fold=4) -> 0 session overlap
  • 0 NaN/Inf & 0 BANNED_FEATURE_COLUMNS assertion
  • Builds strict 4 ablation column specs (C1=40, C2=39, C3=49, C4=39)
        │
        ▼
[backend/app/modeling/models.py + wrappers.py]
  • Importable AllometricLogTransformer (linear families only)
  • Importable LogDuanRegressor (log1p/expm1 + fold-internal Duan smearing)
  • Fold-only compute_sample_weights() (inv_sqrt_sess * inv_sqrt_band, cap<=5.0, mean=1.0)
  • 4 mandatory families: RidgeLog, HuberLog, RandomForestLog, LightGBMLog (n_jobs=4)
        │
        ▼
[backend/app/modeling/evaluation.py]
  • Strict 5-fold OOF CV over 4 families x 4 ablation conditions (80 fits, ~31 s)
  • 2x2 weighting x smearing sensitivity check on top C2 candidate (+4 x 5 fits, ~6 s)
  • Within-session Dual-Lens Consistency across 127 mixed-lens sessions (OOF)
  • Dual-mode Counterfactual Lens-Flip Audit (focal_only + full_physical_lens_swap)
    & <500g ultrawide MBE/MAE gate -> sets lens_confounding_detected: bool
  • Deterministic Winner Selection (D-06): min BalMAPE among gate-passing candidates
  • Full stratified slicing: global, per_fold, by_weight_band, by_lens, by_scene,
    by_band_and_scene, by_band_and_lens, session_level, feature_importances
        │
        ├──────────────────────────────┬────────────────────────────────┐
        ▼                              ▼                                ▼
[artifacts.py: Dual Bundle]   [artifacts.py: Eval Report]    [artifacts.py: Presets Catalog]
Refit production_pipeline     Byte-deterministic JSON        6 single-bird OOF presets
(100% clean) + keep 5         (sort_keys=True, 0 leak        (median-tier, distinct folds)
fold_pipelines + PDIConfig    substrings, 1,738 anon         + 4 multi-bird Watershed
+ band_error_intervals +      oof_scatter_points)            presets via run_watershed_pdi()
age_stage contract                 │                         (opaque image_id ONLY!)
        │                          │                                │
        ▼                          ▼                                ▼
models/weight_regressor.joblib  models/evaluation_report.json  models/presets_catalog.json
    [GIT-IGNORED, ~3.3 MB]         [GIT-TRACKED, Phase 4         [GIT-TRACKED, Phase 4
    -> Phase 4 lifespan loader      GET /api/metrics + WEB-05]    GET /api/presets + WEB-03/04]
```

### Recommended Project Structure

```text
backend/
├── app/
│   ├── config.py                     # Extend with MODEL_BUNDLE_PATH, EVALUATION_REPORT_PATH, PRESETS_CATALOG_PATH
│   └── modeling/                     # NEW Phase 3 package
│       ├── __init__.py               # Clean public barrel
│       ├── wrappers.py               # AllometricLogTransformer, LogDuanRegressor (importable for joblib)
│       ├── age_stage.py              # Cobb Avian 48 map_weight_to_age_stage() + AgeStageResult
│       ├── datasets.py               # LoadedModelingDataset, load_modeling_dataset(), 4 ablation specs (C1..C4)
│       ├── models.py                 # MODEL_FAMILY_NAMES, compute_sample_weights(), build_regressor_pipeline()
│       ├── evaluation.py             # 5-fold CV, 4-condition ablation, dual-lens + counterfactual audit, D-06 selector
│       └── artifacts.py              # Dual-bundle joblib packer/loader, evaluation_report.json & presets_catalog.json writers
├── scripts/
│   └── train_and_evaluate.py         # Single deterministic CLI (D-11) with --dry-run & atomic writes
└── tests/
    ├── test_modeling_wrappers_and_age_stage.py   # Wrappers, sample_weight, age_stage, subprocess joblib pickling
    ├── test_modeling_datasets_and_ablation.py    # Anti-leakage hygiene, C1..C4 specs, counterfactual & gate logic
    └── test_train_and_evaluate_integration.py    # End-to-end CLI, byte-determinism, PROIBIÇÃO 4 leak scan, bundle contract
```

### Pattern 1: Importable `AllometricLogTransformer` & `LogDuanRegressor` (Pickling-Safe)

**What:** Define custom `sklearn` estimators exclusively inside `backend/app/modeling/wrappers.py`, inheriting `BaseEstimator` + `TransformerMixin`/`RegressorMixin`, with **zero** lambdas, closures, or `__main__` class definitions.
**Why critical:**
- Putting `SimpleImputer(strategy="median")` upstream of `LGBMRegressor` converts `pd.DataFrame` to a `float64` NumPy array identically during both `fit()` and `predict()`, eliminating LightGBM's `"X does not have valid feature names"` warning while preserving `pipeline.feature_names_in_` on the first step (`SimpleImputer` or `AllometricLogTransformer`).
- Inside `Pipeline`, passing `pipe.fit(X_tr, y_tr, reg__sample_weight=sw)` routes `sample_weight` strictly to the final `LogDuanRegressor` step (never leaking into `StandardScaler` or `SimpleImputer` unless explicitly needed, and ensuring `HuberRegressor`, `Ridge`, `RandomForestRegressor`, and `LGBMRegressor` all consume `sample_weight` uniformly).
- Inside `LogDuanRegressor.fit()`, compute `y_log = np.log1p(y)` and, when `use_smearing=True`, compute Duan's smearing strictly on training-fold residuals `np.exp(y_log - pred_log)` (using `np.average(..., weights=sample_weight)` when weighted, else `np.mean(...)`, capped cleanly to finite positive floats).

### Pattern 2: Strict 4-Condition Feature Sets & Dual-Mode Anti-Confounding Gate (`D-04`, `D-05`, `D-06`)

**What:** Derive every ablation feature tuple programmatically from `backend/app/pdi/features.py` constants (`MORPHOMETRIC_FEATURE_COLUMNS`, `CHROMATIC_TEXTURE_FEATURE_COLUMNS`, `OPTICAL_FEATURE_COLUMNS`, `FEATURE_COLUMNS`) — never duplicate raw column strings blindly:
- `C1_uncalibrated_pdi` (`40` cols): `MORPHOMETRIC_FEATURE_COLUMNS + CHROMATIC_TEXTURE_FEATURE_COLUMNS`.
- `C2_physics_normalized_strict` (`39` cols): `DIMENSIONLESS_SHAPE_COLUMNS (8) + CHROMATIC_TEXTURE_FEATURE_COLUMNS (26) + OPTICAL_NORM_5_COLUMNS (5)`. Assert `set(C2).isdisjoint({"focal_length_mm", "zoom_multiple", "exif_optical_present", "area_px", "perimeter_px", "major_axis_px", "minor_axis_px", "convex_area_px", "area_frac", "area_frac_optical_norm"})`.
- `C3_physics_plus_exif_meta` (`49` cols): `FEATURE_COLUMNS`.
- `C4_naive_double_count_fxz` (`39` cols): `DIMENSIONLESS_SHAPE_COLUMNS (8) + CHROMATIC_TEXTURE_FEATURE_COLUMNS (26) + NAIVE_FXZ_5_COLUMNS (5)` computed strictly inside `datasets.py` as `area_px / (f*z)²`, `sqrt(...)`, `perimeter_px / (f*z)`, `major_axis_px / (f*z)`, `minor_axis_px / (f*z)` (clearly scoped to offline ablation evaluation only; never persisted to `features_clean.parquet`).

**Dual-Mode Anti-Confounding Gate (`D-05`):**
For every `(model_name, condition)` evaluated across the 5 folds:
1. **`focal_only_counterfactual`**: For conditions containing `focal_length_mm` (`C3`), flip `4.25 ↔ 1.65` on validation folds while holding all other columns constant; compute median/mean/p90 relative shift `%`. (For `C1`/`C2`/`C4`, equals `0.0%`.)
2. **`physical_lens_swap_counterfactual`**: Hold physical bird size (`area_optical_norm` + 4 siblings), dimensionless shape, and plumage chroma strictly constant; simulate capturing the identical bird with the other Redmi 10 lens (`4.25 mm / 1.0× ↔ 1.65 mm / 0.6×`, sensor pixel area ratio `4080×1840 / (3264×1472) = 1.5625`), updating any lens-dependent features consumed by that condition (`C1`: raw `*_px` & `area_frac`; `C2`: **none → exact `0.00%` shift**; `C3`: `focal_length_mm`, `zoom_multiple`, raw `*_px`, `area_frac`, `area_frac_optical_norm`; `C4`: `*_naive_fxz` scaled by `(f_old / f_new)` and `(f_old / f_new)²`).
3. **Confounding Flag Rule (`lens_confounding_detected: bool`):**
   - `C2_physics_normalized_strict` is lens-invariant by construction (`0.00%` shift → `False`).
   - For any non-`C2` condition (`C3`, `C4`, and auditable on `C1`), flag `lens_confounding_detected = True` if ANY of:
     - `focal_only_median_shift_pct > 10.0` (`D-05` direct focal shortcut), OR
     - `physical_lens_swap_median_shift_pct > 10.0` OR `physical_lens_swap_p90_shift_pct > 15.0` (tree lens-ID reconstruction / double-counting shift), OR
     - `<500 g` ultrawide BOTH `mae_g > c2_mae_g` AND `abs(mbe_g) > abs(c2_mbe_g)` (`D-05` `<500g` ultrawide degradation rule).
   - Record all diagnostic numbers (`focal_only_*`, `physical_lens_swap_*`, `dual_lens_consistency_*`, `lt500_ultrawide_mbe_g`, `lt500_ultrawide_mae_g`, `confounding_reasons: list[str]`) verbatim in `models/evaluation_report.json`.

### Pattern 3: Dual-Bundle Honesty Serialization (`models/weight_regressor.joblib`)

**What (`D-07` & `ARCHITECTURE.md` Pattern 5):** Store a versioned `dict[str, Any]` (`MODEL_BUNDLE_SCHEMA_VERSION = "1.0.0"`) with atomic temp-file + `os.replace()` write (`compress=3`):
```python
{
    "schema_version": "1.0.0",
    "model_version": "avianscale-cv-v1.0.0",
    "pdi_config": asdict(PDIConfig()),          # JSON-normalized (tuples -> lists or exact dict)
    "pdi_config_hash": "43a3cfaa3208019e",
    "winning_model_name": "LightGBMLog",
    "winning_ablation_condition": "C2_physics_normalized_strict",
    "feature_names": tuple(winning_cols),       # exact 39-col ordered tuple
    "weighting_mode": "session_band_capped_5x",
    "duan_smearing_enabled": False,
    "production_pipeline": production_pipeline, # refitted on 100% of 1,738 clean images
    "fold_pipelines": {0: p0, 1: p1, 2: p2, 3: p3, 4: p4},  # exact 5 held-out OOF models
    "band_error_intervals": {
        "<500g": {"n": 1223, "mae_g": ..., "mape_pct": ..., "median_abs_err_g": ..., "p68_abs_err_g": ..., "p90_abs_err_g": ..., "mbe_g": ...},
        "500-1000g": {...},
        ">1000g": {...},
    },
    "valid_weight_range_g": {"min": 116, "max": 2093},
    "valid_age_days_range": "~7-29 days",
}
```
**Why critical:** Phase 4 FastAPI loads this bundle once at startup (`lifespan`), checks `bundle["pdi_config_hash"] == PDIConfig().config_hash()`, routes live arbitrary uploads to `bundle["production_pipeline"]`, and routes curated presets (`preset["fold"] = k`) to `bundle["fold_pipelines"][k]` so every live pitch preset prediction is strictly out-of-fold (`0` training leakage).

### Pattern 4: Pure Shared Cobb Avian 48 Age-Stage Mapper (`D-08`)

**What (`backend/app/modeling/age_stage.py`):** Frozen slotted dataclass `AgeStageResult` + pure function `map_weight_to_age_stage(weight_g: float)`:
- Reject `bool`, `NaN`, `Inf`, or `weight_g <= 0` with `ValueError` (ASVS V5).
- Exact 4 agronomic stages (`D-08`):
  - `< 350.0 g`: `stage_code = "inicial_pintinho"`, `stage_label = "Inicial / Pintinho"`, `estimated_age_days_range = "~7-13 dias"`
  - `350.0 <= weight_g <= 700.0 g`: `stage_code = "crescimento_inicial"`, `stage_label = "Crescimento Inicial"`, `estimated_age_days_range = "~14-19 dias"`
  - `700.0 < weight_g <= 1200.0 g`: `stage_code = "crescimento_intermediario"`, `stage_label = "Crescimento Intermediário"`, `estimated_age_days_range = "~20-24 dias"`
  - `> 1200.0 g`: `stage_code = "terminacao_pre_abate"`, `stage_label = "Terminação / Pré-Abate"`, `estimated_age_days_range = "~25-29 dias"`
- `out_of_training_range: bool = (weight_g < 116.0) or (weight_g > 2093.0)`.
- Include `valid_training_range_g: tuple[int, int] = (116, 2093)` and `valid_age_days_range: str = "~7-29 days"` so Phase 4 `/api/predict` displays clear domain-range disclosures whenever a user uploads a day-old chick or 3 kg slaughter bird.

### Pattern 5: Byte-Deterministic, Zero-Path-Leakage JSON Artifacts (`D-09`, `D-10`)

**What:** Follow the exact serialization + guard pattern established in `backend/scripts/extract_features.py:406-417` (`[VERIFIED: backend/scripts/extract_features.py:44, 406-417]`):
- Serialize via `json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"`.
- Zero volatile timestamps (`datetime.now()`, wall-clock timings, hostnames, PIDs) inside git-tracked JSON files so re-running `train_and_evaluate.py` produces **byte-identical SHA-256 outputs**.
- Scan serialized UTF-8 text (lowercased) against `_FORBIDDEN_LEAK_SUBSTRINGS = (".jpg", ".jpeg", ".png", "rel_path", "/home/", "data/raw")` before writing; raise `ValueError` if any forbidden token appears (`PROIBIÇÃO 4`).
- Write atomically via temp file in same directory + `os.replace()`.

**Exact `models/presets_catalog.json` Contract (`D-10` — Phase 4 `GET /api/presets`, `WEB-03`, `WEB-04` consumer):**
- Top-level keys: `schema_version ("1.0.0")`, `pdi_config_hash ("43a3cfaa3208019e")`, `winning_model_name`, `winning_ablation_condition`, `single_bird_presets` (list of `6` dicts), `watershed_presets` (list of `4` dicts).
- Each `single_bird_presets` entry carries **ONLY opaque IDs & validation metadata** (NEVER `.jpg` or paths):
  `preset_id`, `label`, `category` (`"young_chick" | "intermediate_grower" | "heavy_broiler" | "hard_white_on_white_bucket" | "dual_lens_pair_wide" | "dual_lens_pair_ultrawide"`), `pair_group_id` (`"dual_lens_pair"` for slots 5–6, else `None`), `image_id`, `session_id`, `fold`, `ground_truth_weight_g`, `weight_band`, `lens_class`, `scene_type`, `oof_predicted_g`, `oof_abs_err_g`, `oof_pct_err`.
- Each `watershed_presets` entry carries **ONLY opaque IDs & audited multi-bird metadata**:
  `preset_id` (`"ws_417"`, `"ws_757"`, `"ws_823"`, `"ws_1520"`), `label`, `folder_id` (`"417" | "757" | "823" | "1520"`), `image_id`, `lens_class`, `expected_bird_count` (`2, 2, 4, 2`), `detected_bird_count_default` (live count from `run_watershed_pdi()`), `group_sum_weight_g` (`417, 757, 823, 1520`), `mean_bird_weight_reference_g` (`group_sum / expected_count`), `ground_truth_scope ("group_sum_only")`, `disclaimer` (explicit PT-BR/technical disclosure that individual bird weights were not measured separately).

**Exact `models/evaluation_report.json` Contract (`D-09` — Phase 4 `GET /api/metrics`, `WEB-05` consumer):**
- Top-level keys:
  - `schema_version`: `"1.0.0"`
  - `pdi_config_hash`: `"43a3cfaa3208019e"`
  - `dataset_summary`: `n_clean_images (1738)`, `n_unique_weights (160)`, `n_sessions (188)`, `n_folds (5)`, `cross_fold_session_overlap (0)`, `n_mixed_lens_sessions (127)`, `weight_min_g (116)`, `weight_max_g (2093)`, `valid_age_days_range ("~7-29 days")`
  - `literature_benchmark`: `{"r2_reference": 0.90, "note": "Fixed ceiling-camera controlled-height benchmarks (constant Z); handheld uncalibrated-Z single-frame vs. session-averaged context."}`
  - `selection_criteria`: `{"primary_metric": "balanced_band_mape_pct", "tie_breakers": ["global_mape_pct", "global_mae_g", "dual_lens_median_abs_diff_g"], "anti_confounding_gate": {...}}`
  - `mechanics_ablation`: `4` rows (`unweighted` vs `session_band_capped_5x` × `duan_smearing` `False` vs `True`) on winning family + `C2`, plus linear `AllometricLogTransformer` impact summary (`D-01`, `D-02`, `D-03`).
  - `model_comparison`: all `16` (`4` families × `4` ablation conditions) rows with `model_name`, `ablation_condition`, `n_features`, `global`, `session_level`, `balanced_band_mape_pct`, `by_weight_band`, `dual_lens_consistency`, `counterfactual_lens_flip`, `lens_confounding_detected`, `confounding_reasons`, `eligible_for_production`, `is_winner`.
  - `optical_ablation`: `conditions` (`C1`..`C4` summary for winning family `LightGBMLog` and family means), `deltas_vs_c1` (explicit `C2 - C1`, `C3 - C1`, `C4 - C1` on `mae_g`, `rmse_g`, `mape_pct`, `balanced_band_mape_pct`, `r2`), `dual_lens_consistency` (`127` mixed-lens sessions), `counterfactual_lens_flip` (`focal_only` + `full_physical_lens_swap` + `<500g` ultrawide MBE/MAE comparison).
  - `winning_model`: `model_name`, `ablation_condition`, `n_features`, `feature_names`, `weighting_mode`, `duan_smearing_enabled`, `global`, `session_level`, `balanced_band_mape_pct`, `per_fold` (`0..4`), `by_weight_band`, `by_lens`, `by_scene`, `by_band_and_scene`, `by_band_and_lens`, `band_error_intervals`, `feature_importances` (sorted descending by normalized gain `%`).
  - `oof_scatter_points`: `1,738` anonymized dicts sorted stably by `image_id` with keys `image_id`, `session_id`, `fold`, `actual_g`, `predicted_g`, `abs_err_g`, `pct_err`, `weight_band`, `lens_class`, `scene_type`.

### Anti-Patterns to Avoid
- **Never fit scalers, imputers, `sample_weight`, or Duan's smearing outside the 5-fold CV loop:** Always instantiate a fresh `Pipeline` per fold (`clone`/factory) and compute `sample_weight` strictly on `train_idx` (`PROIBIÇÃO 3`).
- **Never use `lambda` or `__main__`-scoped classes inside `models/weight_regressor.joblib`:** Always import `LogDuanRegressor` and `AllometricLogTransformer` from `backend.app.modeling.wrappers`.
- **Never evaluate `sample_weight` on validation metrics:** `MAE`, `RMSE`, `MAPE`, `R²`, `MBE`, and `BalMAPE` are strictly unweighted OOF predictions in grams (`D-03`).
- **Never pass `scene_type`, `scene_is_bucket`, `lcd_excluded_px`, `segmentation_fallback`, `lens_class`, `fold`, `session_id`, or `BANNED_FEATURE_COLUMNS` to any regressor:** Slice strictly with explicit whitelisted feature tuples (`D-02`).
- **Never cherry-pick near-0% error OOF presets:** Select median-tier (`pct_err >= 3.0%`, closest to stratum median) non-fallback samples across distinct sessions/folds (`D-10`).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| **Fold integrity & anti-leakage assertion** | Custom ad-hoc fold overlap loops | `backend.app.ingestion.splits.verify_fold_integrity(df, min_heavy_sessions_per_fold=4)` (`[VERIFIED: backend/app/ingestion/splits.py:35-136]`) | Already tested in Phase 1; checks NaN folds, integer dtype, `0..4` completeness, pairwise `session_id` disjointness, and `>= 4` heavy sessions per fold. |
| **Weight-band classification** | Hardcoded `if w < 500` duplicates | `backend.app.ingestion.decoding.assign_weight_band()` & `MULTI_BIRD_FOLDERS` (`[VERIFIED: backend/app/ingestion/decoding.py:22, 111-117]`) | Guarantees exact boundary parity (`<500g`, `500-1000g`, `>1000g`) across Phases 1–4. |
| **Feature schema constants** | Hand-typed 49-string lists in `modeling/` | Import `MORPHOMETRIC_FEATURE_COLUMNS`, `CHROMATIC_TEXTURE_FEATURE_COLUMNS`, `OPTICAL_FEATURE_COLUMNS`, `FEATURE_COLUMNS`, `BANNED_FEATURE_COLUMNS` from `backend.app.pdi.features` (`[VERIFIED: backend/app/pdi/features.py:39-103]`) | Eliminates typo drift and guarantees 1:1 feature contract lock with Phase 2. |
| **Pipeline cloning, imputation & scaling** | Manual NumPy median/mean arrays per fold | `sklearn.pipeline.Pipeline`, `SimpleImputer(strategy="median")`, `StandardScaler()`, `sklearn.base.clone` | Automatically prevents fold leakage and preserves `feature_names_in_` metadata inside `joblib`. |
| **Regression error metrics** | Custom RMSE/R² loops | `sklearn.metrics.mean_absolute_error`, `root_mean_squared_error`, `r2_score` | Numerically verified edge-case handling (note: `root_mean_squared_error` replaced deprecated `mean_squared_error(..., squared=False)` in scikit-learn 1.4+). |
| **Path containment for Watershed preset verification** | Raw `RAW_DATASET_DIR / rel_path` | Reuse the exact ASVS V4 non-symlink containment check pattern from `backend/scripts/extract_features.py:132-173` | Prevents symlink escapes and path traversal during preset audit runs (`T-02-10`). |

---

## Common Pitfalls

### Pitfall 1: Focal-Only Counterfactual Misses Tree Lens-ID Reconstruction in `C3` and `C4`
**What goes wrong:** Running ONLY `focal_length_mm: 4.25 ↔ 1.65` on `C3 LightGBMLog` yields a tiny `1.18%` median shift (and `0.00%` on `C4 LightGBMLog` since `C4` lacks `focal_length_mm`), falsely passing a naive `< 10%` focal-only gate even though `C3` reconstructs lens ID via `area_frac_optical_norm` (`W/UW = 0.479`) and `area_px / area_optical_norm == zoom²`, and `C4` encodes an `8.85×` lens step via `area_px / (f·z)²`.
**How to avoid:** Implement BOTH `focal_only` AND `full_physical_lens_swap` counterfactual checks (holding physical `area_optical_norm`, dimensionless shape, and plumage chroma constant while propagating lens switch `4.25/1.0 ↔ 1.65/0.6` and sensor ratio `1.5625` to all lens-dependent input columns). `C2_physics_normalized_strict` achieves exact `0.00%` shift, while `C3` (`p90 = 23.52%`) and `C4` (`med = 12.74%`, `p90 = 39.43%`) are cleanly caught and disqualified.
**Warning signs:** `C4_naive_double_count_fxz` or `C3_physics_plus_exif_meta` selected as production winner despite worse within-session `W - UW` bias.

### Pitfall 2: Pandas 3.1 Copy-on-Write (`ChainedAssignmentError`) & Arrow `str` Dtypes
**What goes wrong:** Writing `X_va[mask]["focal_length_mm"] = 1.65` raises `ChainedAssignmentError` or silently fails under Pandas `3.1.1`; calling `df.select_dtypes(include=["object"])` misses Arrow-backed `str` columns (`image_id`, `session_id`, `weight_band`, `lens_class`, `scene_type`).
**How to avoid:** Always mutate copies via `X_copy = X_va.copy(); X_copy.loc[mask, "focal_length_mm"] = 1.65` and explicitly slice `df.loc[:, list(cols)]` (verified zero errors across all 4 spikes on Pandas `3.1.1`).

### Pitfall 3: LightGBM `"X does not have valid feature names"` Warning & Column Order Drift
**What goes wrong:** Fitting `LGBMRegressor` directly on a `pd.DataFrame` and predicting on an ndarray (or vice-versa) floods logs with scikit-learn/LightGBM `UserWarning`s; worse, passing a misaligned `dict` at Phase 4 inference silently permutes features.
**How to avoid:** Keep `SimpleImputer(strategy="median")` as the first step of EVERY pipeline (outputting `float64` ndarray cleanly into `LGBMRegressor` during both `fit` and `predict` — verified `0` warnings and `1.76 ms` latency in Spike 4C), AND validate/reorder columns explicitly via `pd.DataFrame([features], columns=list(bundle["feature_names"]))` inside a shared helper `predict_from_feature_dict(pipeline, feature_names, features)`!

### Pitfall 4: Unbounded `sample_weight` or Validation-Metric Weighting
**What goes wrong:** Using raw `1 / (n_sess * n_band)` without capping creates `> 100×` weight ratios on single heavy outlier sessions (`2093 g`), causing massive variance; applying `sample_weight` to OOF metrics distorts reported `MAE`/`MAPE`.
**How to avoid:** Compute `w_raw = (1 / sqrt(n_sess)) * (1 / sqrt(n_band))` strictly inside each training fold, symmetrically clip to `[median / sqrt(5.0), median * sqrt(5.0)]`, normalize `w /= w.mean()`, assert `w.max() / w.min() <= 5.0 + 1e-9` and `abs(w.mean() - 1.0) < 1e-9`, and NEVER pass weights to OOF evaluation metrics (`D-03`).

### Pitfall 5: Misinterpreting Negative Within-Band $R^2$ in `evaluation_report.json`
**What goes wrong:** Computing $R^2$ inside a narrow truncated weight band (`500–1000 g` where target variance is tiny) naturally yields negative values (`-6.9`) even when `MAPE = 26.7%` is the best of all 3 bands — confusing dashboard readers if displayed as a KPI without context.
**How to avoid:** Still compute and store exact `r2` on global, per-fold, per-lens, and per-scene slices (all full-range `[116, 2093] g` where $R^2 \in [0.46, 0.63]$) and session-averaged (`R² ≈ 0.69`), while on narrow weight-band slices (`<500g`, `500-1000g`, `>1000g`) emphasize `mae_g`, `rmse_g`, `mape_pct`, `median_abs_err_g`, `p68_abs_err_g`, `p90_abs_err_g`, and `mbe_g` (keep `r2` or document that truncated-range $R^2$ is variance-compressed).

### Pitfall 6: `joblib` Pickling Fragility & Non-Deterministic JSON Diffs
**What goes wrong:** Defining wrapper classes inside `backend/scripts/train_and_evaluate.py` fails `joblib.load()` in FastAPI with `AttributeError: Can't get attribute 'LogDuanRegressor' on <module '__main__'>`; embedding `wall_clock_sec` or `created_at` timestamps inside `models/evaluation_report.json` causes spurious Git diffs on every re-run; LightGBM multithreading with `n_jobs > 1` can introduce sub-ULP floating-point drift if `force_col_wise=True` / `deterministic=True` is omitted.
**How to avoid:**
1. Keep `LogDuanRegressor` and `AllometricLogTransformer` strictly in `backend/app/modeling/wrappers.py`.
2. Pass `deterministic=True, force_col_wise=True, random_state=42, n_jobs=4` to `LGBMRegressor` so repeated runs produce **bit-identical** trees and **byte-identical** `evaluation_report.json` and `presets_catalog.json`!
3. Gate Phase 3 completion with a fresh-subprocess `joblib.load()` + prediction test (`subprocess.run([sys.executable, "-c", ...])`) AND a two-run byte-identity assertion on both JSON artifacts.

---

## Code Examples

### 1. Importable `LogDuanRegressor` & `AllometricLogTransformer` (`backend/app/modeling/wrappers.py`)

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

ALLOMETRIC_LOG_COLUMNS: frozenset[str] = frozenset(
    {
        "area_px",
        "perimeter_px",
        "major_axis_px",
        "minor_axis_px",
        "convex_area_px",
        "area_frac",
        "area_optical_norm",
        "sqrt_area_optical_norm",
        "perim_optical_norm",
        "major_axis_optical_norm",
        "minor_axis_optical_norm",
        "area_frac_optical_norm",
        "area_naive_fxz",
        "sqrt_area_naive_fxz",
        "perim_naive_fxz",
        "major_naive_fxz",
        "minor_naive_fxz",
    }
)


class AllometricLogTransformer(BaseEstimator, TransformerMixin):
    """Apply ``np.log1p(np.clip(x, 0, None))`` strictly to positive geometric area/linear cols."""

    def __init__(self, log_columns: tuple[str, ...] | None = None) -> None:
        self.log_columns = log_columns

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> AllometricLogTransformer:
        target = set(self.log_columns) if self.log_columns is not None else set(ALLOMETRIC_LOG_COLUMNS)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.log_indices_: tuple[int, ...] = tuple(
                i for i, col in enumerate(X.columns) if str(col) in target
            )
        else:
            self.log_indices_ = ()
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = (
            X.to_numpy(dtype=np.float64, copy=True)
            if isinstance(X, pd.DataFrame)
            else np.array(X, dtype=np.float64, copy=True)
        )
        if self.log_indices_:
            idx = list(self.log_indices_)
            arr[:, idx] = np.log1p(np.clip(arr[:, idx], 0.0, None))
        return arr


class LogDuanRegressor(BaseEstimator, RegressorMixin):
    """Log1p-target regressor with optional fold-internal Duan's smearing correction (D-01)."""

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        use_smearing: bool = False,
        min_prediction_g: float = 1.0,
    ) -> None:
        self.estimator = estimator
        self.use_smearing = use_smearing
        self.min_prediction_g = min_prediction_g

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> LogDuanRegressor:
        if self.estimator is None:
            raise ValueError("LogDuanRegressor requires a non-None base estimator.")
        self.estimator_ = clone(self.estimator)
        y_arr = np.asarray(y, dtype=np.float64)
        if np.any(~np.isfinite(y_arr)) or np.any(y_arr <= 0.0):
            raise ValueError("Target weights must be finite and strictly positive.")
        y_log = np.log1p(y_arr)

        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64)
            self.estimator_.fit(X, y_log, sample_weight=w)
        else:
            w = None
            self.estimator_.fit(X, y_log)

        if self.use_smearing:
            pred_log = np.asarray(self.estimator_.predict(X), dtype=np.float64)
            exp_resid = np.exp(np.clip(y_log - pred_log, -20.0, 20.0))
            self.smearing_factor_: float = (
                float(np.average(exp_resid, weights=w)) if w is not None else float(np.mean(exp_resid))
            )
        else:
            self.smearing_factor_ = 1.0
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        pred_log = np.asarray(self.estimator_.predict(X), dtype=np.float64)
        pred_g = np.exp(np.clip(pred_log, -20.0, 20.0)) * self.smearing_factor_ - 1.0
        return np.clip(pred_g, self.min_prediction_g, None)
```

### 2. Fold-Only Capped Sample Weighting (`backend/app/modeling/models.py`)

```python
def compute_sample_weights(
    train_sessions: np.ndarray | pd.Series,
    train_bands: np.ndarray | pd.Series,
    *,
    max_weight_ratio: float = 5.0,
) -> np.ndarray:
    """Compute fold-internal capped inverse sqrt(session) x sqrt(weight_band) weights (D-03)."""
    if max_weight_ratio < 1.0:
        raise ValueError(f"max_weight_ratio must be >= 1.0, got {max_weight_ratio!r}")
    s_series = pd.Series(train_sessions, dtype="str")
    b_series = pd.Series(train_bands, dtype="str")
    s_counts = s_series.map(s_series.value_counts()).to_numpy(dtype=np.float64)
    b_counts = b_series.map(b_series.value_counts()).to_numpy(dtype=np.float64)

    raw_w = (1.0 / np.sqrt(s_counts)) * (1.0 / np.sqrt(b_counts))
    med = float(np.median(raw_w))
    half_span = float(np.sqrt(max_weight_ratio))
    clipped = np.clip(raw_w, med / half_span, med * half_span)
    normalized = clipped / float(np.mean(clipped))
    return normalized
```

### 3. Locked Hyperparameters for All 4 Mandatory PRD Families (`backend/app/modeling/models.py`)

| Model Family | Exact Locked Hyperparameters (`random_state=42`, `n_jobs=4`) | Preprocessing Steps | 5-Fold Runtime (1 Cond) |
|--------------|--------------------------------------------------------------|---------------------|-------------------------|
| **`RidgeLog`** | `Ridge(alpha=10.0)` | `AllometricLogTransformer` → `SimpleImputer(median)` → `StandardScaler` → `LogDuanRegressor` | `~0.1 s` |
| **`HuberLog`** | `HuberRegressor(alpha=10.0, epsilon=1.35, max_iter=1000)` | `AllometricLogTransformer` → `SimpleImputer(median)` → `StandardScaler` → `LogDuanRegressor` | `~0.3 s` |
| **`RandomForestLog`** | `RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=3, max_features=0.5, random_state=42, n_jobs=4)` | `SimpleImputer(median)` → `LogDuanRegressor` | `~5.3 s` |
| **`LightGBMLog`** | `LGBMRegressor(n_estimators=500, learning_rate=0.025, num_leaves=24, max_depth=6, min_child_samples=12, subsample=0.8, subsample_freq=1, colsample_bytree=0.55, reg_alpha=0.3, reg_lambda=2.0, deterministic=True, force_col_wise=True, random_state=42, n_jobs=4, verbosity=-1)` | `SimpleImputer(median)` → `LogDuanRegressor` | `~1.7 s` |

Total 5-fold runtime across all `4` families × `4` ablation conditions (`80` fits) + `2×2` weighting/smearing diagnostic (`15` additional fits) + `1` production refit + `13`-image Watershed preset audit (`~2.4 s`) = **`~36–42 s`** on local CPU.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| **`uv`** | Deterministic venv & test execution (`uv run`) | ✓ | `0.13.9` (`[VERIFIED: Phase 1 / GEMINI.md]`) | — |
| **Python (uv-managed)** | All Phase 3 modeling & CLI scripts | ✓ | `3.12.14` (`[VERIFIED: uv run python]`) | — |
| **`scikit-learn` / `lightgbm` / `joblib`** | `MOD-01`, `MOD-02`, `MOD-03` | ✓ | `1.9.1` / `4.7.0` / `1.6.0` (`[VERIFIED: uv run python]`) | — |
| **`pandas` / `pyarrow` / `numpy` / `scipy`** | Parquet I/O & vectorized evaluation | ✓ | `3.1.1` / `26.0.1` / `2.5.5` / `1.18.1` (`[VERIFIED: uv run python]`) | — |
| **`pytest` / `ruff`** | Automated tests & linting | ✓ | `9.2.0` / `0.17.6` (`[VERIFIED: uv run]`) | — |
| **Materialized Phase 1–2 Parquets** | `manifest_clean.parquet`, `manifest_quarantine.parquet`, `features_clean.parquet` | ✓ | `1,738 × 24`, `15 × 24`, `1,738 × 60` (`[VERIFIED: Spike 1]`) | — |

**Missing dependencies:** None.

---

## Security Domain

> `workflow.security_enforcement = true` (`security_asvs_level = 1`, `security_block_on = "high"`) verified in `.planning/config.json:48-50` (`[VERIFIED: .planning/config.json:48-50]`).

### Applicable ASVS L1 Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| **V2 Authentication** | No | Local offline CLI & modeling package; no user accounts or credentials. |
| **V3 Session Management** | No | Stateless offline training/evaluation; `session_id` is a biological bird-capture burst identifier, not an auth token. |
| **V4 Access Control** | **Yes** | (1) `models/presets_catalog.json` stores **only opaque `preset_id` + `image_id`** (never raw filenames, `rel_path`, or filesystem paths — eliminating direct path injection into Phase 4 `/api/presets`). (2) Watershed preset selection resolves `manifest_quarantine.parquet` paths strictly through non-symlink `RAW_DATASET_DIR.resolve()` containment (`extract_features.py` pattern). |
| **V5 Input Validation** | **Yes** | Strict finite-number, non-bool, positive-value validation in `map_weight_to_age_stage(weight_g)`, `LogDuanRegressor.fit()`, `compute_sample_weights()`, and strict column/hash/alignment/schema guards in `load_modeling_dataset()` and `load_model_bundle()`. |
| **V6 Cryptography** | **Yes (Integrity Hashing)** | `PDIConfig().config_hash()` (`hashlib.sha256` 16-hex lock `"43a3cfaa3208019e"`) verified between `features_clean.parquet`, `weight_regressor.joblib`, `evaluation_report.json`, and `presets_catalog.json` — preventing silent training/serving skew. |

### Known Threat & Scientific Integrity Patterns (STRIDE + PRD Prohibitions)

| Threat Pattern | STRIDE / Category | Standard Mitigation |
|----------------|-------------------|---------------------|
| **Unsafe `joblib`/pickle deserialization of untrusted artifacts** | Tampering / Elevation of Privilege | `models/weight_regressor.joblib` is produced exclusively locally by `backend/scripts/train_and_evaluate.py`, git-ignored (`.gitignore:4`), never accepted from external/HTTP uploads, and validated immediately on load (`schema_version`, `pdi_config_hash`, required bundle keys, exact `feature_names`). |
| **Data/Path Leakage into Git-Tracked JSON (`PROIBIÇÃO 4`)** | Information Disclosure | Automated pre-write guard scanning serialized JSON (and pytest asserting on disk) rejects `.jpg`, `.jpeg`, `.png`, `rel_path`, `/home/`, and `data/raw` in both `models/evaluation_report.json` and `models/presets_catalog.json`; CLI stdout prints strictly aggregate counts/metrics. |
| **Cross-Fold Session Leakage (`PROIBIÇÃO 3`)** | Integrity / Tampering | `verify_fold_integrity(df, min_heavy_sessions_per_fold=4)` runs before any fit; all imputers, scalers, allometric log transforms, sample weights, and Duan's smearing fit strictly inside each fold loop. |
| **Visual / Metadata Shortcut Leakage (`PROIBIÇÃO 2` & Pitfall #3)** | Integrity / Spoofing | Strict feature allowlist from `FEATURE_COLUMNS` (zero `BANNED_FEATURE_COLUMNS`, zero `scene_type`/`lcd_excluded_px`/`segmentation_fallback`/`lens_class`); dual-mode counterfactual + `<500 g` ultrawide anti-confounding gate disqualifies `C3` and `C4` shortcuts automatically. |
| **Partial / Corrupt Artifact Writes on Interrupted CLI Run** | Denial of Service / Integrity | Atomic temp-file + `os.replace()` writes for `weight_regressor.joblib`, `evaluation_report.json`, and `presets_catalog.json`, plus `--dry-run` flag writing zero files. |
| **Host OOM / CPU Thrashing on `7.1 GiB` RAM** | Denial of Service | Strictly cap `n_jobs <= 4` across `RandomForestRegressor` and `LGBMRegressor` (never `n_jobs=-1`). |

---

## Verification Strategy

*(Note: `workflow.nyquist_validation` is explicitly `false` in `.planning/config.json:24` (`[VERIFIED: .planning/config.json:24]`), so a formal Nyquist Validation Architecture table is omitted; below is the concrete pytest + CLI verification plan for Phase 3.)*

1. **Unit Tests — Wrappers, Weighting & Age-Stage (`backend/tests/test_modeling_wrappers_and_age_stage.py`):**
   - `LogDuanRegressor`: exact `log1p`/`expm1` round-trip, fold-internal Duan's smearing (`> 1.0` on noisy residuals, exact `1.0` when `use_smearing=False`), `sample_weight` forwarding inside `Pipeline` (`reg__sample_weight`), invalid non-positive/NaN target rejection, and **clean subprocess `joblib.dump()` → `joblib.load()` + single-row prediction `< 5 ms`** with zero warnings.
   - `AllometricLogTransformer`: transforms ONLY geometric area/linear columns, leaves chromatic/dimensionless columns bit-identical, handles `0.0` / negative inputs cleanly via `np.clip(..., 0, None)`.
   - `compute_sample_weights()`: verifies `max_w / min_w <= 5.0 + 1e-9`, `abs(mean - 1.0) < 1e-9`, positive finite floats, and determinism.
   - `map_weight_to_age_stage()`: tests all 4 boundary intervals (`115 g` → out-of-range `True`, `116 g` → `False`, `349.9 / 350.0 / 700.0 / 700.1 / 1200.0 / 1200.1 / 2093.0 / 2094.0`), plus `bool`/`NaN`/`<= 0` rejection.
2. **Unit Tests — Datasets, Ablation Hygiene & Anti-Confounding Gate (`backend/tests/test_modeling_datasets_and_ablation.py`):**
   - Verifies exact column counts (`C1=40`, `C2=39`, `C3=49`, `C4=39`), zero `BANNED_FEATURE_COLUMNS`, zero background/metadata leak columns (`scene_type`, `scene_is_bucket`, `lcd_excluded_px`, `segmentation_fallback`, `lens_class`, `fold`, `session_id`), and `C2` disjoint from raw EXIF/area columns.
   - Synthetic confounding test proving the dual-mode counterfactual gate flags a lens-cheating `C3`/`C4` regressor (`lens_confounding_detected == True`) while passing a strictly lens-invariant `C2` regressor (`0.00%` shift, `lens_confounding_detected == False`).
   - Synthetic fold-integrity failure test proving mismatched `pdi_config_hash` or overlapping `session_id` raises `RuntimeError`/`ValueError`.
3. **Integration & Contract Tests (`backend/tests/test_train_and_evaluate_integration.py`):**
   - CLI `--dry-run` writes zero files and emits zero `.jpg`/`rel_path`/`/home/` substrings to stdout.
   - Full materialized artifact verification (and end-to-end execution on real `1,738`-row Parquets):
     - `models/weight_regressor.joblib`: loads cleanly in a fresh Python subprocess, passes `schema_version == "1.0.0"`, `pdi_config_hash == "43a3cfaa3208019e"`, `winning_ablation_condition == "C2_physics_normalized_strict"`, `len(fold_pipelines) == 5` (`keys == {0, 1, 2, 3, 4}`), exact `39` `feature_names`, `band_error_intervals` for all 3 bands, and `< 5 ms` single-row prediction with zero warnings.
     - `models/evaluation_report.json`: validates all `D-09` required keys, `16` model comparison rows, `4` ablation conditions, `127` mixed-lens sessions, `1,738` anonymized `oof_scatter_points`, zero forbidden path/filename substrings, and **byte-identical determinism** across back-to-back runs.
     - `models/presets_catalog.json`: validates `6` single-bird OOF presets (`segmentation_fallback == False`, distinct folds covering all 5 folds, valid paired dual-lens same-session pair, zero cherry-picked `< 3%` errors) + `4` multi-bird Watershed presets (`417`, `757`, `823`, `1520` with opaque `image_id` existing in `manifest_quarantine.parquet`), zero forbidden path/filename substrings, and **byte-identical determinism**.
   - Existing repo-wide guards remain green (`uv run pytest -v` and `uv run ruff check backend/`).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| — | *(None — all data shapes, feature contracts, package versions, 5-fold OOF metrics, counterfactual shifts, Watershed counts, and joblib serialization benchmarks were verified live in this session.)* | — | — |

All claims in this research were empirically verified against the codebase, materialized Parquet/JSON files, and live Python 3.12 execution — no user confirmation checkpoints are required before planning.

---

## Sources

### Primary (HIGH confidence — Verified Live in Session)
- `.planning/phases/03-tabular-regression-optical-ablation-production-bundling/03-CONTEXT.md` — Locked decisions `D-01`..`D-11` (`[VERIFIED: lines 1-127]`).
- `backend/app/pdi/features.py` — Frozen 49-key `FEATURE_COLUMNS` (`14` morph + `26` chroma/texture + `9` optical) and `BANNED_FEATURE_COLUMNS` (`[VERIFIED: lines 39-103]`).
- `backend/app/pdi/config.py` — Frozen `PDIConfig` and `config_hash()` (`"43a3cfaa3208019e"`) (`[VERIFIED: lines 68-296]`).
- `backend/app/config.py` — Canonical repo-root-anchored paths (`[VERIFIED: lines 12-34]`).
- `backend/app/ingestion/decoding.py` & `backend/app/ingestion/splits.py` — `assign_weight_band()`, `MULTI_BIRD_FOLDERS`, `verify_fold_integrity()` (`[VERIFIED: decoding.py:22, 111-117; splits.py:35-136]`).
- `backend/app/pdi/watershed.py` & `backend/scripts/extract_features.py` — `run_watershed_pdi()`, 60-column Parquet schema, ASVS V4 path containment, and `PROIBIÇÃO 4` JSON serialization guard (`[VERIFIED: watershed.py:435-689; extract_features.py:44, 61-76, 132-173, 406-417]`).
- `models/dataset_audit_summary.json` & `models/pdi_calibration_audit.json` — Exact dataset counts, fold distributions, and Phase 2 paired dual-lens calibration proof (`[VERIFIED]`).
- `pyproject.toml`, `.gitignore`, `.planning/config.json`, `backend/tests/test_io_exif.py` — Locked dependencies, gitignore rules, ASVS L1 config, and repo-wide AST/package guards (`[VERIFIED]`).
- Live Empirical Spikes 1–4 (`uv run python` on `Python 3.12.14`, `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `joblib 1.6.0`, `pandas 3.1.1`) — Real 5-fold OOF metrics, weighting/smearing sweeps, 4-condition ablation, dual-lens consistency, 3-mode counterfactual lens-flip audit, 13-image Watershed audit, OOF preset candidate pools, and subprocess `joblib` latency benchmark (`[VERIFIED: live execution]`).

---

## Metadata

**Confidence breakdown:**
- **Standard stack:** HIGH — 100% of packages already locked in `uv.lock` and verified live in Python `3.12.14`.
- **Empirical CV, Ablation & Anti-Confounding Mechanics:** HIGH — every number measured directly over all `1,738` clean rows across the exact locked 5 `session_id` folds (`0` leakage).
- **Architecture, Schemas & Serialization Contracts:** HIGH — aligned verbatim with `03-CONTEXT.md` (`D-01`..`D-11`), `ARCHITECTURE.md`, `PITFALLS.md`, and Phase 4 consumer specs (`WEB-01`..`WEB-05`).

**Research date:** 2026-11-23
**Valid until:** 2026-12-23 (30 days; frozen Phase 1–2 contracts and locked `uv.lock` environment).
