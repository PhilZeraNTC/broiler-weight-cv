# Phase 3: Tabular Regression, Optical Ablation & Production Bundling - Context

**Gathered:** 2026-11-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the reproducible, zero-deep-learning tabular weight regression, optical calibration ablation, stratified out-of-fold validation, and production artifact bundling layer (`backend/app/modeling/` + `backend/scripts/train_and_evaluate.py`) consuming Phase 1's locked 5-fold `session_id` manifest (`data/processed/manifest_clean.parquet`, `manifest_quarantine.parquet`) and Phase 2's frozen 49-feature mask-gated PDI dataset (`data/processed/features_clean.parquet`, `pdi_config_hash = 43a3cfaa3208019e`) (`MOD-01`, `MOD-02`, `MOD-03`):

1. Train and compare interpretable/tabular log-target regressors (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog`) strictly across the locked 5-fold `StratifiedGroupKFold` splits (`0` `session_id` overlap per `PROIBIÇÃO 3`), evaluating capped inverse session×stratum training weights and fold-internal Duan's smearing with zero leakage across folds (`MOD-01`).
2. Execute a mathematically leak-free 4-condition WITH vs. WITHOUT EXIF optical ablation plus a within-session dual-lens consistency (`127` mixed-lens sessions) and counterfactual lens-flip anti-confounding audit, persisting git-tracked, zero-path-leakage `models/evaluation_report.json` with MAE (g), RMSE (g), MAPE (%), $R^2$, and Mean Bias Error globally and stratified by weight band (`<500g`, `500-1000g`, `>1000g`), lens (`wide` vs. `ultrawide`), and visual scenario (`bucket_dorsal` vs. `scale_or_litter`) (`MOD-02`).
3. Publish the dual-bundle production artifact (`models/weight_regressor.joblib`, git-ignored) containing the winning `production_pipeline`, all 5 held-out `fold_pipelines` (for honest live out-of-fold preset scoring in Phase 4), locked `PDIConfig` + SHA-256 hash, exact `feature_names`, version metadata, per-band expected error intervals, and a shared Cobb Avian 48 developmental age-stage mapper, alongside git-tracked, zero-path-leakage `models/presets_catalog.json` (`MOD-03`).

FastAPI REST serving and React/TypeScript/Vite/Tailwind UI (`WEB-01`..`WEB-05`) belong strictly to Phase 4.

</domain>

<decisions>
## Implementation Decisions

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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope, Requirements & Prohibitions
- `PRD.md` — Closed-scope product requirements (`REQ-08`–`REQ-10`, `REQ-13`, `REQ-15`, `PROIBIÇÃO 1–5`).
- `.planning/PROJECT.md` — Core value, constraints, empirical Phase 2 optical calibration proof (`area_px / zoom²`), and `WARNING 1–4` reconciliations.
- `.planning/REQUIREMENTS.md` — Formal v1 requirements `MOD-01`, `MOD-02`, `MOD-03` and downstream consumers `WEB-01`..`WEB-05`.
- `.planning/ROADMAP.md` — Phase 3 goal and 4 observable Success Criteria.
- `.planning/phases/01-forensic-data-curation-anti-leakage-split/01-CONTEXT.md` — Locked Phase 1 `manifest_clean.parquet` / `manifest_quarantine.parquet` schema and 5-fold `StratifiedGroupKFold` (`D-03`, `D-05`).
- `.planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-CONTEXT.md` — Locked Phase 2 `PDIConfig` hash (`43a3cfaa3208019e`) and 49-feature mask-gated contract (`D-08`–`D-10`).

### Forensic Dataset & Phase 2 Empirical Audit Contracts
- `models/dataset_audit_summary.json` — Exact fold balance (`[4, 5, 5, 5, 5]` heavy sessions/fold), `801–1,216 g` sparse gap, and `1,738` clean / `15` quarantined totals.
- `models/pdi_calibration_audit.json` — Paired dual-lens calibration proof across `127` mixed-lens sessions (`175` same-scene pairs) establishing `area_px / (zoom_multiple ** 2)` as the physical winner (`W/UW = 0.9638` on bucket pairs) vs. rejected double-counting `(focal_length_mm * zoom_multiple) ** 2` (`0.145`).
- `docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md` §4, §5, §7, §8 — Lens×weight confounding above `1,000 g` (`56–78%` ultrawide), `47:1` weight imbalance, `801–1,216 g` desert, and `~7–29 days` developmental age curve.
- `docs/proposta_pdi.md.pdf` — Academic methodology context (literature `R² > 0.9` benchmark, interpretable tabular regressors, Watershed illustrative per-bird weight).

### Research & Architecture Contracts
- `.planning/research/STACK.md` — Verified `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `joblib 1.6.0`, `pandas 3.1.1` Copy-on-Write / Arrow `str` dtype notes, and `TransformedTargetRegressor` notes.
- `.planning/research/ARCHITECTURE.md` — `backend/app/modeling/` module layout, Pattern 1 (`PDIConfig` hash lock), Pattern 5 (Dual-Bundle Honesty: `production_pipeline` + 5 `fold_pipelines`), and Flow 1–2 data contracts.
- `.planning/research/PITFALLS.md` — Pitfall #3 (dual-lens `focal × zoom` double-counting & lens×weight confounding shortcut), Pitfall #6 (target imbalance, log-target Jensen's inequality / Duan's smearing, tree non-extrapolation, `joblib` pickling safety), and Pitfall #8 (honest OOF pitch presets & working-dir-independent paths).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/config.py`: Canonical repo-root-anchored paths (`CLEAN_MANIFEST_PATH`, `QUARANTINE_MANIFEST_PATH`, `AUDIT_SUMMARY_PATH`, `FEATURES_CLEAN_PATH`, `PDI_CALIBRATION_AUDIT_PATH`) — extend with `MODEL_BUNDLE_PATH`, `EVALUATION_REPORT_PATH`, and `PRESETS_CATALOG_PATH`.
- `backend/app/pdi/config.py`: Frozen slotted `PDIConfig` and deterministic 16-hex `config_hash()` (`43a3cfaa3208019e`) — embed in `models/weight_regressor.joblib` and verify against `features_clean.parquet`.
- `backend/app/pdi/features.py`: Frozen `MORPHOMETRIC_FEATURE_COLUMNS` (14), `CHROMATIC_TEXTURE_FEATURE_COLUMNS` (26), `OPTICAL_FEATURE_COLUMNS` (9), `FEATURE_COLUMNS` (49), and `BANNED_FEATURE_COLUMNS` — import directly to construct leak-free ablation column lists without string duplication.
- `backend/app/ingestion/decoding.py`: `assign_weight_band()`, `MULTI_BIRD_FOLDERS` — reuse for consistent weight-band classification (`<500g`, `500-1000g`, `>1000g`).
- `backend/app/pdi/watershed.py`: `run_watershed_pdi()` — optionally invoked during preset selection to pick clean representative multi-bird images from `417`/`757`/`823`/`1520` (with exact bird count verification).
- Materialized Parquet inputs verified on disk (`100%` aligned on `image_id`): `data/processed/manifest_clean.parquet` (`1,738 × 24`), `data/processed/manifest_quarantine.parquet` (`15 × 24`, `13` multi-bird), `data/processed/features_clean.parquet` (`1,738 × 60`).

### Established Patterns
- Python `3.12.14` managed by `uv` (`uv run pytest -v`, `uv run ruff check backend/`), root `pyproject.toml`, `pythonpath = ["."]`.
- Byte-deterministic JSON artifacts in `models/` (`json.dumps(..., indent=2, sort_keys=True) + "\n"`, zero volatile timestamps, zero raw `.jpg`/`rel_path`/`/home/` substrings guarded by pytest).
- Pandas 3.1 Copy-on-Write safe indexing (`.loc[...]`), explicit `pd.DataFrame` column ordering preserved into `scikit-learn`/`LightGBM` pipelines (`feature_names`), and `n_jobs <= 4` CPU discipline on this `7.1 GiB` RAM host.
- Repo-wide AST guard in `backend/tests/test_io_exif.py` prohibiting `cv2.imread`/`cv2.imdecode` and DL/OCR dependencies in `pyproject.toml`.

### Integration Points
- `models/weight_regressor.joblib` (git-ignored) + `backend/app/modeling/age_stage.py` will be loaded directly by Phase 4 `backend/app/api/main.py` (`lifespan`) and `backend/app/api/services/inference_service.py` (routing uploads to `production_pipeline` and presets to `fold_pipelines[preset.fold]`).
- `models/evaluation_report.json` (git-tracked) will be served verbatim by Phase 4 `GET /api/metrics` and rendered in Tab 3 (`WEB-05` Metrics & Audit Dashboard).
- `models/presets_catalog.json` (git-tracked) will be loaded at startup by Phase 4 `backend/app/api/services/preset_service.py` (resolving opaque `image_id` → `rel_path` via `manifest_clean.parquet` / `manifest_quarantine.parquet` inside `RAW_DATASET_DIR.resolve()`).

</code_context>

<specifics>
## Specific Ideas

- Run a focused empirical modeling spike during Phase 3 research directly over the materialized `data/processed/features_clean.parquet` (`1,738 × 60`) to measure real 5-fold OOF metrics across all 4 regressor families × 4 ablation conditions × weighting/smearing toggles before locking hyperparameters in the plan — zero synthetic guesswork, zero raw filenames printed (`PROIBIÇÃO 4`).
- Ensure `LightGBMLog` and `RandomForestLog` run with deterministic seeds (`random_state=42`) and bounded threads (`n_jobs=4`) so re-running `backend/scripts/train_and_evaluate.py` produces byte-identical `models/evaluation_report.json` and `models/presets_catalog.json`.
- For single-bird curated presets in `models/presets_catalog.json`, avoid suspicious near-0% cherry-picks; pick realistic, well-segmented (`segmentation_fallback == False`) representative OOF samples across distinct folds so live pitch demonstrations look scientifically credible under academic Q&A.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed strictly within Phase 3 scope. (Leave-One-Date-Out temporal cross-validation `HARD-04` and post-pitch `30–45 d` data expansion `BARN-02` remain deferred per `REQUIREMENTS.md`.)

</deferred>

---

*Phase: 03-tabular-regression-optical-ablation-production-bundling*
*Context gathered: 2026-11-23*
