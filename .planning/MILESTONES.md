# Project Milestones: AvianScale CV

## v1.0 MVP (Shipped: 2026-11-24)

**Delivered:** End-to-end, 100% interpretable broiler live-weight estimation MVP (*Cobb Avian 48*, `116–2,093 g`, `~7–29 days`) using strictly classical Digital Image Processing (PDI) and EXIF-calibrated tabular regression, served as a `< 500 ms` local single-origin FastAPI + React/TypeScript/Vite/Tailwind CSS v4 pitch studio — completed 4 days ahead of the `2026-11-28` code freeze (`2026-11-30` pitch).

**Phases completed:** 1–4 (`4` phases, `14` plans, `36` tasks)  
**Closeout type:** `verified_closeout` (`4/4` phases verified `passed`, `15/15` v1 requirements complete, `24/24` cross-phase integration links wired, `6/6` E2E flows verified, `0` open/deferred audit items)

**Key accomplishments:**

- **Forensic Data Curation & Leak-Free 5-Fold Splits (Phase 1):** Pinned binary-only CPython `3.12.14` via `uv`, implemented the deterministic Audit §9 ingestor (nested level-2 traversal, `1340_` sanitization, `367 → 267 g` correction, CP866 mojibake & Eastern-Arabic digit decoding, mandatory Pillow EXIF `Orientation`/`FocalLength`/Xiaomi `39321` extraction) yielding `1,738` clean / `15` quarantined typed Parquet manifests (`160` unique weights, `188` sessions) and zero-overlap 5-fold `StratifiedGroupKFold` partitions (`models/dataset_audit_summary.json`).
- **Single-Source Classical PDI Engine & Empirical Optical Calibration (Phase 2):** Locked frozen `PDIConfig` (`working_long_edge = 768 px`, SHA-256 `43a3cfaa3208019e`) with zero `cv2.imread`/`cv2.imdecode` repo-wide, rotation-invariant SF-400 LCD suppression (`786` activations, `0` bucket false positives, `0.23%` primary fallback), `49` strictly mask-gated features (`0.0`-diff outside-mask invariance), empirically proven `area_px / zoom_multiple²` calibration (`W/UW = 0.9638` on bucket pairs across `127` dual-lens sessions), 4-stage X-Ray visuals (`p95 = 88.3 ms`), and materialized `features_clean.parquet` (`1,738 × 60`, `46.53 s` on 4 workers).
- **Isolated Classical Marker-Based Watershed (Phase 2):** Delivered zero-hardcoding multi-bird separation (chromatic red-feeder pre-suppression, SF-400 LCD suppression, exact EDT, bbox-cropped `h_maxima`, `skimage.segmentation.watershed`) achieving `9/13` exact counts (`12/13` within `±1`, `MAE = 0.38` birds) across all `13` quarantined multi-bird scenes (`417`, `757`, `823`, `1520`) with per-instance 49-feature extraction.
- **Interpretable Tabular Regression, 4-Way EXIF Ablation & Dual-Bundle Publishing (Phase 3):** Benchmarked 4 classical families (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog`) across 4 optical conditions on locked folds; proved unanimous win of `LightGBMLog / C2_physics_normalized_strict` (`39` features, `MAE = 158.11 g`, `RMSE = 264.02 g`, `BalMAPE = 35.11%`, `R² = 0.5582`, `session_averaged_r2 = 0.6905`, `0.0%` dual-mode counterfactual shift while disqualifying `C3`/`C4` lens shortcuts), and published immutable dual-bundle `models/weight_regressor.joblib` (`production_pipeline` + 5 `fold_pipelines`, `2.4 ms` inference), `age_stage.py`, `models/evaluation_report.json`, and opaque-ID `models/presets_catalog.json`.
- **Sub-100 ms FastAPI Backend & 4-Tab Single-Origin Pitch Studio (Phase 4):** Delivered modular FastAPI API (`6` routes, `lifespan` caching, ASVS L1 upload guards, `56.1–79.7 ms` live `/api/predict`, honest OOF `fold_pipelines[fold]` preset scoring, `0.94 g` live Dual-Lens invariance delta, `_NonApiSpaRoute` single-origin serving) paired with a React 19 + TypeScript 7 + Vite 8 + Tailwind CSS v4 + Recharts 3 studio (`Estúdio Raio-X`, `Separação Watershed` with permanent `group_sum_only` banner and reconciled ASVS V5 sliders, `Auditoria & Métricas` with `1,738→188` scatter and Mendeley reconciliation, and 100% static `Visão Futura (v2)`).

**Stats:**

- `183+` files created/modified across `89` commits
- `26,989` lines of application & test code (`18,615` Python + `8,374` TypeScript/TSX/CSS)
- `4` phases, `14` plans, `36` tasks
- `272` backend pytest tests + `26` frontend Vitest tests (`100%` passing; `ruff`, dual-project `tsc`, Biome v2, single-origin E2E clean)
- `4` days from initialization to ship (`2026-11-20` → `2026-11-24`)

**Git range:** `feat(01-01)` (`5c2210e`) → `feat(04-04)` (`2d3eb83`) *(full milestone history: `5582b7f` → `v1.0`)*

**What's next:** Post-pitch `v1.1` Hardening (`HARD-01`–`HARD-05`: offline label audit, bucket-rim `mm/px` calibration, Laplacian motion-blur gate, Leave-One-Date-Out CV, batch folder export + minor helper export cleanup) and `v2.0` Commercial Barn Deployment (`BARN-01`–`BARN-03`: fixed overhead ceiling camera, `30–45 d` late-cycle dataset expansion, CPU-only edge packaging).

---
