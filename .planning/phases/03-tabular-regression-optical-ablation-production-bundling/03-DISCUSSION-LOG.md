# Phase 3: Tabular Regression, Optical Ablation & Production Bundling - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-11-23
**Phase:** 03-tabular-regression-optical-ablation-production-bundling
**Areas discussed:** Model Families & Target/Weighting Mechanics, 4-Condition EXIF Optical Ablation & Anti-Confounding Gate, Winner Selection & Dual-Bundle Joblib + Age Mapper, Evaluation Report & Zero-Path-Leakage OOF Presets Catalog (resolved autonomously via `--auto`)

---

## Model Families, Feature Hygiene & Target/Weighting Mechanics (`MOD-01`)

| Option | Description | Selected |
|--------|-------------|----------|
| 4 PRD tabular families (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog`) in importable `LogDuanRegressor` + per-fold `AllometricLogTransformer` + capped (`<=5.0`) session×band sample weights (empirically tested ON/OFF alongside Duan smearing), strictly mask-gated (`FEATURE_COLUMNS` only; `scene_is_bucket` excluded from inputs) | Satisfies `PROIBIÇÃO 1–3`, avoids `joblib` lambda pickling bugs, models $W \propto A^{3/2}$ cleanly in linear baselines, and keeps `scene_is_bucket` background photometry strictly for evaluation slicing | ✓ |
| Pass `scene_is_bucket` and all 49 features directly to raw `TransformedTargetRegressor(func=np.log1p)` without Duan smearing or session weighting | Simpler code, but risks `PROIBIÇÃO 2` border-pixel leakage, Jensen's inequality negative bias on heavy birds, and 47-photo session dominance | |
| Include deep tabular neural networks (MLP/TabNet) or CatBoost/XGBoost | Violates `PROIBIÇÃO 1` (neural nets) or adds unpinned dependencies outside `STACK.md` | |

**User's choice:** Auto-selected recommended option (`--auto`): 4 PRD tabular families + importable `LogDuanRegressor` + `AllometricLogTransformer` + empirically verified capped session×band weighting & Duan smearing + strict mask-gated feature hygiene (`D-01`..`D-03`).
**Notes:** Logged per `modes/auto.md` single-pass autonomous capture.

---

## 4-Condition EXIF Optical Ablation & Anti-Confounding Gate (`MOD-02`)

| Option | Description | Selected |
|--------|-------------|----------|
| 4 mathematically leak-free conditions (`C1_uncalibrated_pdi`, `C2_physics_normalized_strict`, `C3_physics_plus_exif_meta`, `C4_naive_double_count_fxz`) + within-session dual-lens consistency (`127` sessions) + counterfactual lens-flip gate (`>10%` shift flags confounding & disqualifies raw lens shortcut) | Directly resolves Pitfall 3 (preventing trees from reconstructing `zoom` via `area_px / area_optical_norm` inside `C2`), proves Spike 2's physics fix in downstream ML CV, and prevents `ultrawide ↔ >1,000 g` shortcuts | ✓ |
| 2-way ablation (`with EXIF` vs `without EXIF`) keeping all 49 columns together | Fails to separate genuine physical scale normalization from the `ultrawide ↔ >1,000 g` confounding shortcut and misses `focal × zoom` double-counting proof | |

**User's choice:** Auto-selected recommended option (`--auto`): 4-condition leak-free ablation + dual-lens consistency + counterfactual lens-flip anti-confounding gate (`D-04`, `D-05`).
**Notes:** Phase 3 research will also inspect `area_frac_optical_norm` vs. `area_optical_norm` as an internal diagnostic.

---

## Winner Selection, Dual-Bundle `.joblib` & Cobb Avian 48 Age Mapper (`MOD-01`, `MOD-03`)

| Option | Description | Selected |
|--------|-------------|----------|
| Deterministic selection by balanced weight-band OOF MAPE among non-confounded candidates + dual-bundle `models/weight_regressor.joblib` (`production_pipeline` + 5 `fold_pipelines` + `PDIConfig` hash + `feature_names` + per-band error intervals) + shared `age_stage.py` (`~7–29 d` / `116–2,093 g` coverage guard) | Guarantees honest out-of-fold pitch preset inference (Pattern 5), protects `>1,000 g` accuracy against `70.4%` `<500 g` skew, and enforces zero training/serving skew in Phase 4 | ✓ |
| Select solely by unweighted global RMSE and serialize only a single refitted `production_pipeline` | Global RMSE ignores `<500 g` MAPE and single-pipeline serialization forces in-sample predictions on live pitch presets | |

**User's choice:** Auto-selected recommended option (`--auto`): balanced-band OOF MAPE winner + dual-bundle `.joblib` + shared `age_stage.py` (`D-06`..`D-08`).
**Notes:** Verified subprocess `joblib.load()` + `< 5 ms` inference gate included.

---

## Evaluation Report JSON & Zero-Path-Leakage Presets Catalog (`MOD-02`, `MOD-03`)

| Option | Description | Selected |
|--------|-------------|----------|
| Byte-deterministic git-tracked `models/evaluation_report.json` (stratified by band/lens/scene + 4-way ablation + anonymized 1,738-row `oof_scatter_points`) + `models/presets_catalog.json` referencing ONLY opaque `preset_id` + `image_id` (6 single-bird OOF presets incl. dual-lens same-session pair + 4 multi-bird Watershed presets `417`/`757`/`823`/`1520`) | 100% strict `PROIBIÇÃO 4` & ASVS V4 compliance (zero `.jpg` filenames or `rel_path` in git-tracked JSON), resolved safely at runtime in Phase 4 via `manifest_clean.parquet`/`manifest_quarantine.parquet` | ✓ |
| Store raw `rel_path` strings inside git-tracked `models/presets_catalog.json` | Violates the strict `PROIBIÇÃO 4` zero-filename leak test established in Phases 1–2 for `models/*.json` | |

**User's choice:** Auto-selected recommended option (`--auto`): zero-path-leakage opaque `image_id` preset catalog + byte-deterministic stratified `evaluation_report.json` + unified `train_and_evaluate.py` CLI (`D-09`..`D-11`).
**Notes:** Representative median-error non-fallback OOF samples will be selected deterministically (no cherry-picked `0.0%` errors).

---

## the agent's Discretion

- Conservative hyperparameter ranges for `Ridge`, `HuberRegressor`, `RandomForestRegressor`, and `LGBMRegressor` validated empirically via Phase 3 research spike on `features_clean.parquet`.
- Exact internal submodule structure inside `backend/app/modeling/`.
- Optional 5th `HistGradientBoostingLog` benchmark row in `model_comparison`.
- Representative frame selection heuristic per Watershed folder (`417`, `757`, `823`, `1520`).

## Deferred Ideas

- Leave-One-Date-Out temporal diagnostic cross-validation (`HARD-04`) — deferred to v1.x per `REQUIREMENTS.md`.
- Late-cycle `30–45 d` / `2.1–3.2 kg` dataset expansion (`BARN-02`) — deferred to v2+ per `REQUIREMENTS.md`.
