---
phase: "04-fastapi-local-backend-interactive-pitch-web-studio"
verified: "2026-11-24T15:39:21Z"
status: passed
score: "5/5 ROADMAP Success Criteria · 5/5 Requirements (WEB-01..WEB-05) · 12/12 Decisions (D-01..D-12) · 52/52 Plan Artifacts · 14/14 Key Links · 11/11 Review Findings Closed"
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-01-PLAN.md"
  - ".planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-01-SUMMARY.md"
  - ".planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-02-PLAN.md"
  - ".planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-02-SUMMARY.md"
  - ".planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-03-PLAN.md"
  - ".planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-03-SUMMARY.md"
  - ".planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-04-PLAN.md"
  - ".planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-04-SUMMARY.md"
  - ".planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-REVIEW.md"
  - ".planning/phases/04-fastapi-local-backend-interactive-pitch-web-studio/04-UI-SPEC.md"
  - "backend/app/api/__init__.py"
  - "backend/app/api/dependencies.py"
  - "backend/app/api/main.py"
  - "backend/app/api/routers/__init__.py"
  - "backend/app/api/routers/health.py"
  - "backend/app/api/routers/metrics.py"
  - "backend/app/api/routers/predict.py"
  - "backend/app/api/routers/presets.py"
  - "backend/app/api/routers/watershed.py"
  - "backend/app/api/schemas.py"
  - "backend/app/api/services/__init__.py"
  - "backend/app/api/services/inference_service.py"
  - "backend/app/api/services/metrics_service.py"
  - "backend/app/api/services/preset_service.py"
  - "backend/app/api/services/upload_guard.py"
  - "backend/app/api/services/watershed_service.py"
  - "backend/app/config.py"
  - "backend/tests/test_api_contract.py"
  - "backend/tests/test_api_latency_and_parity.py"
  - "backend/tests/test_api_security.py"
  - "frontend/biome.json"
  - "frontend/index.html"
  - "frontend/package.json"
  - "frontend/src/App.tsx"
  - "frontend/src/components/layout/AppShell.tsx"
  - "frontend/src/components/layout/InlineErrorBanner.tsx"
  - "frontend/src/components/layout/StageLightbox.tsx"
  - "frontend/src/components/metrics/ActualVsPredictedScatter.tsx"
  - "frontend/src/components/metrics/FeatureImportanceChart.tsx"
  - "frontend/src/components/metrics/ForensicAuditCard.tsx"
  - "frontend/src/components/metrics/MetricsAuditTab.tsx"
  - "frontend/src/components/metrics/ModelComparisonTable.tsx"
  - "frontend/src/components/metrics/OpticalAblationChart.tsx"
  - "frontend/src/components/metrics/StratifiedErrorsPanel.tsx"
  - "frontend/src/components/studio/DualLensComparison.tsx"
  - "frontend/src/components/studio/FlockBatchCard.tsx"
  - "frontend/src/components/studio/OpticalCalibrationCard.tsx"
  - "frontend/src/components/studio/PredictionSummaryCard.tsx"
  - "frontend/src/components/studio/PresetsStrip.tsx"
  - "frontend/src/components/studio/TelemetryPanel.tsx"
  - "frontend/src/components/studio/UploadDropzone.tsx"
  - "frontend/src/components/studio/XRayQuadViewer.tsx"
  - "frontend/src/components/studio/XRayStudioTab.tsx"
  - "frontend/src/components/vision/FutureVisionTab.tsx"
  - "frontend/src/components/watershed/WatershedInstanceTable.tsx"
  - "frontend/src/components/watershed/WatershedPresetStrip.tsx"
  - "frontend/src/components/watershed/WatershedSlidersCard.tsx"
  - "frontend/src/components/watershed/WatershedStageGrid.tsx"
  - "frontend/src/components/watershed/WatershedStudioTab.tsx"
  - "frontend/src/hooks/useStudioHooks.ts"
  - "frontend/src/index.css"
  - "frontend/src/lib/api.ts"
  - "frontend/src/lib/cn.ts"
  - "frontend/src/lib/flock.ts"
  - "frontend/src/lib/format.ts"
  - "frontend/src/lib/scatter.test.ts"
  - "frontend/src/lib/scatter.ts"
  - "frontend/src/lib/studioCore.test.ts"
  - "frontend/src/main.tsx"
  - "frontend/src/types/api.ts"
  - "frontend/tsconfig.app.json"
  - "frontend/tsconfig.json"
  - "frontend/tsconfig.node.json"
  - "frontend/vite.config.ts"
covered_digest: "v1:sha256:9d62ed6d0f1c1124718b9b436d344bac28197d98ca93d2541d788939eca69a88"
behavior_unverified: 0
overrides_applied: 0
overrides: []
gaps: []
---

# Phase 4 Verification Report: FastAPI Local Backend & Interactive Pitch Web Studio

**Verified:** `2026-11-24T15:04:25Z` (fingerprint & full regression re-verified `2026-11-24T15:39:21Z` after micro-polish `eaa9f83`/`16eb74e`)
**Verdict:** `PASSED` (0 blockers, 0 gaps, 0 stub/orphaned artifacts, 0 regressions across Phases 1–4)
**Baseline Commit Audited:** `16eb74e` (incorporating `9519bc1` code-review closure + `eaa9f83` micro-polish: promoted Tab 2 missing-EXIF banner, direct SPA pathname routing, and `04-04-SUMMARY.md` frontmatter parity; `272/272` pytest + `ruff` + `tsc` + `biome` + `26/26` Vitest + single-origin E2E verified green)

---

## Executive Summary

Independent, adversarial goal-backward verification was executed directly against the live codebase, compiled `frontend/dist/` bundle, single-origin FastAPI ASGI application (`backend/app/api/main.py`), `pyproject.toml`, `frontend/package.json`, and Git index.

- **Backend Full Regression & Lint:** `uv run pytest -v` passed **`272/272`** tests in `64.23 s` (`67` Phase 4 API tests across contract, latency/parity, and ASVS L1 security + `205` Phases 1–3 tests); `uv run ruff check backend/` exited `0` (`All checks passed!`).
- **Frontend Full Quality Gate & Production Build:** Dual-project `npm run typecheck` (`tsconfig.app.json` + `tsconfig.node.json`) exited `0`; Biome v2 `npm run lint` (`biome check src`) checked `37` files in `71 ms` with `0` errors/warnings; Vitest `npm test` passed **`26/26`** unit tests in `243 ms`; Vite 8 `npm run build` emitted `frontend/dist/index.html` (`0.43 kB`), CSS (`27.14 kB` / `6.01 kB` gzip), and JS (`865.92 kB` / `239.72 kB` gzip) in `1.26 s`.
- **Live Single-Origin FastAPI E2E Probe (over real `frontend/dist/`):**
  - `GET /` and `GET /metricas` returned `200 text/html` containing `"AvianScale CV"`; both referenced `/assets/*` bundles returned `200`.
  - `GET /api/does-not-exist`, `POST /api/does-not-exist`, and URL-encoded traversal `GET /api/presets/..%2F..%2Fetc%2Fpasswd/thumbnail` returned `404 application/json`; `GET /api/predict` returned `405 application/json`.
  - `GET /api/health` returned `200` with `pdi_config_hash = "43a3cfaa3208019e"`, `winning_model_name = "LightGBMLog"`, `winning_ablation_condition = "C2_physics_normalized_strict"`, `n_features = 39`, `classical_pdi_only = true`.
  - `GET /api/presets` returned `6` single-bird + `4` Watershed presets with zero `rel_path` or raw image filenames; all `10` `/api/presets/{preset_id}/thumbnail` endpoints returned `200 image/jpeg` (`Cache-Control: public, max-age=3600`).
  - `POST /api/predict` scored all `6` curated presets honestly via `fold_pipelines[fold]` (`oof_evaluation = true`, exact `<= 0.01 g` parity against `models/presets_catalog.json`), warm server latency `56.1–79.7 ms` (`< 500 ms` SLA, `< 400 ms` target), and live Dual-Lens delta **`0.9397 g`** (`383.2180 g` vs. `384.1577 g`, `< 2.0 g`). EXIF-stripped JPEG (`64.2 ms`), RGBA PNG (`16.9 ms`), WEBP (`18.7 ms`), and manual Ultrawide override uploads all succeeded with `200`.
  - `POST /api/watershed-demo` scored all `4` curated multi-bird presets in `138.3–152.1 ms` (`< 500 ms`), returning `ws_417` `bird_count = 2`, exact reconciled Phase 2 ASVS V5 bounds, and permanent `ground_truth_scope = "group_sum_only"`.
  - `GET /api/metrics` returned `1,738` `oof_scatter_points` keyed strictly by `actual_g` (zero `weight_g`) and `16` `model_comparison` rows.
- **Code Review Closure (`04-REVIEW.md`):** All **11/11** findings (`CR-01..02`, `WR-01..05`, `IN-01..04`) were independently verified closed in source files and regression tests (`9519bc1`).
- **Governance & Git Hygiene (`PROIBIÇÃO 1–5`):** Zero deep-learning/OCR imports, zero AST `cv2.imread`/`cv2.imdecode` calls repo-wide, zero `train_test_split`, zero tracked files in `data/raw/`, `data/processed/`, `models/*.joblib`, `frontend/dist/`, or `frontend/node_modules/`, and `FutureVisionTab.tsx` is 100% static documentation.

---

## 1. Observable Truths Verification (ROADMAP Success Criteria + Plan Must-Haves)

| # | Observable Truth / ROADMAP Success Criterion | Status | Direct Codebase & Live Evidence |
|---|----------------------------------------------|--------|---------------------------------|
| SC-1 | Local FastAPI backend exposes Pydantic v2-validated `/api/health`, `/api/predict`, `/api/watershed-demo`, `/api/presets`, `/api/presets/{id}/thumbnail`, and `/api/metrics` with `lifespan` startup caching, `async def` bounded upload intake offloading CPU via `run_in_threadpool`, `< 500 ms` end-to-end latency, and graceful handling of EXIF-stripped JPEG & RGBA PNG uploads (`WEB-01`, `D-01`–`D-05`). | ✓ VERIFIED | `backend/app/api/main.py` (`lifespan` sets `cv2.setNumThreads(2)`, validates `pdi_config_hash = 43a3cfaa3208019e`, pre-warms 10 thumbnails), `upload_guard.py` (`15 MiB` cap + `run_in_threadpool(validate_image_bytes)`), routers offload via `run_in_threadpool`. Live probe: `predict` `56.1–79.7 ms`, `watershed` `138.3–152.1 ms`, EXIF-stripped JPEG `64.2 ms`, RGBA PNG `16.9 ms`, WEBP `18.7 ms`. `67/67` API pytest tests pass. |
| SC-2 | User can upload any broiler image in the Inference & "X-Ray" PDI Studio, view auto-extracted EXIF focal/zoom (or missing-EXIF Amber warning pill with live manual lens/zoom override), inspect all 4 classical PDI stages side-by-side + fullscreen lightbox, read estimated weight (g/kg) with `±MAE`/`p68` error band and Cobb Avian 48 age stage, inspect mask-gated morphometric/chromatic attributes + 39-feature table, and accumulate a live simulated flock average and uniformity (`CV%`) tracker (`WEB-02`, `D-07`–`D-09`). | ✓ VERIFIED | `frontend/src/components/studio/{XRayStudioTab,UploadDropzone,XRayQuadViewer,PredictionSummaryCard,OpticalCalibrationCard,TelemetryPanel,FlockBatchCard}.tsx`, `StageLightbox.tsx`, `useFlockBatch` + `flock.ts` (`ddof=1` SD & `CV%`, `18/18` `studioCore.test.ts` pass). Auto-boots `sb_intermediate_grower`, supports `44×44` lightbox controls, missing-EXIF Amber fallback notice, bounded `[0.5, 200] mm` / `[0.1, 20]×` overrides, and inline 2-step `"Confirmar Limpeza do Lote"` confirmation at `>= 2` birds. |
| SC-3 | User can one-click execute curated out-of-fold validation presets (`~180 g`, `~550 g`, `~1,600 g`, hard white-on-white `>1,000 g` dorsal bucket, plus the `541 g` Fold 0 Dual-Lens pair) and view live Ground Truth vs. out-of-fold Predicted weight, absolute error (g), and `%` error computed by `fold_pipelines[preset.fold]` (`WEB-03`, `D-02`, `D-08`). | ✓ VERIFIED | `PresetsStrip.tsx` renders all `6` presets with opaque `/api/presets/{preset_id}/thumbnail` URLs + highlighted `"Par Dual-Lens — Mesmo Frango (541 g · Fold 0)"` tandem & `"Comparar Par Dual-Lens"` (`DualLensComparison.tsx`). `inference_service.py:167-170` routes presets strictly to `bundle["fold_pipelines"][preset.fold]`. Live probe verified exact `<= 0.01 g` parity across all `6` presets and **`0.9397 g`** Dual-Lens delta. |
| SC-4 | User can explore Marker-based Watershed separation on multi-bird presets (`417`, `757`, `823`, `1520`) or custom uploads with 4-stage basin/boundary visuals, separated bird counts, per-bird pixel areas, illustrative per-bird estimated weights + scene mean weight, permanent `group_sum_only` disclaimer, reconciled bounded sliders, and a dedicated conceptual v2 Fixed Ceiling-Camera Barn Roadmap comparison panel (`WEB-04`, `D-04`, `D-10`, `D-12`). | ✓ VERIFIED | `frontend/src/components/watershed/{WatershedStudioTab,WatershedPresetStrip,WatershedStageGrid,WatershedSlidersCard,WatershedInstanceTable}.tsx` auto-loads `ws_417` (`bird_count = 2`), locks slider bounds verbatim to `backend/app/pdi/config.py`, renders permanent Amber `group_sum_only` banner, and maps Stage 4 hex swatches. `FutureVisionTab.tsx` renders the 5-axis MVP v1 vs. v2 comparison matrix and `BARN-01..03` cards with zero API calls or timers (`PROIBIÇÃO 5`). |
| SC-5 | User can inspect the Validation Metrics & Forensic Audit Dashboard displaying 5-fold CV KPIs (with `R² > 0,90` literature benchmark context), filterable Actual vs. Predicted scatter plot (`actual_g`, dashed `y = x` reference line, `1,738` images vs. `188` sessions), WITH vs. WITHOUT EXIF ablation (`C1`–`C4`), stratified error tables (band/lens/scene), Top-12 feature importances, 16-row model comparison, and the `1,753 → 1,738` forensic curation funnel reconciling Mendeley `"45 dias / 1.772 imagens"` and the `801–1.216 g` coverage gap (`WEB-05`, `D-11`). | ✓ VERIFIED | `frontend/src/components/metrics/{MetricsAuditTab,StratifiedErrorsPanel,ActualVsPredictedScatter,OpticalAblationChart,FeatureImportanceChart,ModelComparisonTable,ForensicAuditCard}.tsx` + `frontend/src/lib/scatter.ts` (`8/8` `scatter.test.ts` pass). Recharts v3 verified: zero `<Cell />`, `isAnimationActive={false}` on all series, `actual_g` axis, `min-w-0` + explicit heights. Live `/api/metrics` verified `1,738` points & `16` model rows. |

---

## 2. Requirements (`WEB-01`–`WEB-05`), Context Decisions (`D-01`–`D-12`) & `04-UI-SPEC.md` Matrix

### Phase 4 Requirements Coverage

| Requirement | Mapped Plans | Status | Verification Evidence |
|-------------|--------------|--------|-----------------------|
| **WEB-01** *(PRD REQ-11)* | `04-01` | ✓ SATISFIED | All 6 REST routes live under `backend/app/api/`; `lifespan` caching + hash lock; `async def` bounded intake + `run_in_threadpool`; `< 500 ms` verified live (`predict` max `79.7 ms`, `watershed` max `152.1 ms`); EXIF-stripped JPEG / RGBA PNG / WEBP verified; `67/67` API tests pass. |
| **WEB-02** *(PRD REQ-12)* | `04-02`, `04-03` | ✓ SATISFIED | Vite 8 / React 19 / TS 7 / Tailwind v4 studio with drag-and-drop upload (`15 MiB` guard), automatic EXIF display + missing-EXIF Amber fallback + bounded manual overrides, 2×2 X-Ray + lightbox, `32px` predicted weight + `±MAE`/`p68` + Cobb Avian 48 stage, 39-feature mask-gated telemetry, and simulated flock `CV%` tracker (`18/18` core tests). |
| **WEB-03** *(PRD REQ-13)* | `04-03` | ✓ SATISFIED | 6 curated OOF presets (`sb_young_chick`, `sb_intermediate_grower`, `sb_heavy_broiler`, `sb_hard_white_bucket`, `sb_dual_lens_wide`, `sb_dual_lens_ultrawide`) with opaque thumbnail gallery, 1-click Dual-Lens comparator (`0.94 g` live delta), and Fold K Ground Truth vs. Predicted (`g` and `%`) audit card. |
| **WEB-04** *(PRD REQ-14)* | `04-03`, `04-04` | ✓ SATISFIED | Interactive Tab 2 (`Separação Watershed`) over `ws_417`, `ws_757`, `ws_823`, `ws_1520` + custom upload, reconciled ASVS V5 sliders, color-matched per-bird table, permanent `group_sum_only` Amber disclaimer; plus Tab 4 (`Visão Futura (v2)`) 100% static MVP v1 vs. v2 ceiling-camera matrix & `BARN-01..03` cards. |
| **WEB-05** *(PRD REQ-15)* | `04-04` | ✓ SATISFIED | Tab 3 (`Auditoria & Métricas`) powered by `GET /api/metrics`: 5 Executive KPIs + `R² > 0,90` benchmark context, filter-before-aggregate `1,738`-image / `188`-session Recharts v3 scatter (`y = x`, `actual_g`), `C1`–`C4` ablation chart, stratified error tables, Top-12 feature importances, 16-row comparison table, and `1,753 → 1,738` Mendeley & `801–1.216 g` gap forensic reconciliation. |

*Orphaned requirements check:* `REQUIREMENTS.md` maps exactly 5 requirements (`WEB-01`..`WEB-05`) to Phase 4; all 5 are claimed across Plans `04-01`..`04-04` and satisfied (`0` orphaned).

### Context Decisions (`D-01`–`D-12`) & `04-UI-SPEC.md` Sign-Off

| Decision / Contract | Verification Result | Concrete Evidence |
|---------------------|---------------------|-------------------|
| **D-01** Modular FastAPI + `lifespan` | ✓ PASS | `backend/app/api/{main,schemas,dependencies}.py`, `routers/{health,presets,predict,watershed,metrics}.py`, `services/{upload_guard,preset,inference,watershed,metrics}_service.py`; `cv2.setNumThreads(2)` and `pdi_config_hash` verified at boot. |
| **D-02** Dual-bundle honest OOF & opaque preset security | ✓ PASS | `file` XOR `preset_id` (`422`), allowlisted presets (`404`), `fold_pipelines[preset.fold]` for presets vs. `production_pipeline` for uploads, `15 MiB` (`413`), Pillow magic allowlist (`415`), `40 MP` bomb/corrupt (`400`), zero `rel_path`/`.jpg` leaks. |
| **D-03** Non-blocking CPU offload & `320 px` JPEG thumbnails | ✓ PASS | `run_in_threadpool` verified in `upload_guard.py`, `predict.py`, `watershed.py`; all `10` thumbnails pre-warmed at `<= 320 px` JPEG with `Cache-Control: public, max-age=3600`. |
| **D-04 / D-10** Watershed contract & reconciled bounds | ✓ PASS | Authoritative bounds from `backend/app/pdi/config.py` (`h_rel [0.05, 0.80]` default `0.20`, `min_distance_rel [0.20, 2.50]` default `0.85`, `min_area_ratio [0.01, 0.45]` default `0.10`) mirrored identically in `schemas.py`, `watershed_service.py`, `04-UI-SPEC.md`, and `WatershedSlidersCard.tsx` (reconciling narrower draft ranges in `04-CONTEXT.md`). |
| **D-05** Single-origin serving + Vite `/api` proxy | ✓ PASS | `frontend/vite.config.ts` proxies `/api` to `http://127.0.0.1:8000`; `backend/app/api/main.py` mounts `frontend/dist/assets` and `_NonApiSpaRoute` serving `index.html` while preserving `/api/*` JSON `404`/`405` and localhost-only CORS. |
| **D-06** `pt-BR` 4-tab hash-synced shell | ✓ PASS | `AppShell.tsx` + `App.tsx` hash-whitelist `#estudio`, `#watershed`, `#metricas`, `#visao-futura`, `role="tablist"` with roving focus (`WR-01` fixed), live health & latency pill. |
| **D-07 & `04-UI-SPEC.md`** Clinical Light Theme, Typography, Copy & A11y | ✓ PASS | Automated scan over all `frontend/src/**/*.{ts,tsx,css}` confirmed: strictly **4 font sizes** (`12px`, `14px`, `20px`, `32px`), strictly **2 font weights** (`normal`, `semibold`), `tabular-nums`, dark `#0F172A` frames strictly inside CV viewports/lightbox, all `35/35` checked verbatim `pt-BR` CTAs/empty/loading/disclosure/ARIA strings present, zero `tailwind.config.*`, zero ESLint/PostCSS, zero external CDN links in `frontend/index.html`, zero `dangerouslySetInnerHTML`. |
| **D-08 / D-09** Zero-empty-state boot, Dual-Lens proof, EXIF override & Flock | ✓ PASS | `XRayStudioTab.tsx` boots `sb_intermediate_grower`, renders `6` presets + `DualLensComparison.tsx`, bounded EXIF override + missing-EXIF disclosure, 39-feature drawer, and `FlockBatchCard.tsx`. |
| **D-11** Scientific Metrics Dashboard & Recharts v3 | ✓ PASS | `MetricsAuditTab.tsx` lazy-loads `/api/metrics`; code-only Recharts scan confirmed **0 `<Cell />`**, **100% `isAnimationActive={false}`**, **`actual_g`** axis (0 `weight_g`), `min-w-0` + explicit height wrappers, and filter-before-aggregate `scatter.ts` (`8/8` tests). |
| **D-12** Conceptual Future Vision (`PROIBIÇÃO 5`) | ✓ PASS | `FutureVisionTab.tsx` contains **0** `fetch`, `useEffect`, `useState`, `setInterval`, `setTimeout`, `WebSocket`, `EventSource`, or API/hook imports; displays verbatim `PROIBIÇÃO 5` Amber banner, 5-axis table, and `BARN-01..03` cards. |

---

## 3. Artifact & Key-Link Verification (`gsd-tools.cjs`)

Verified directly via `gsd-tools.cjs query verify.artifacts` and `verify.key-links` across all 4 Phase 4 plans:

| Plan | Artifacts Verified | Key Links Verified | Anti-Pattern / Stub Scan |
|------|--------------------|--------------------|--------------------------|
| `04-01-PLAN.md` (Backend API) | `12 / 12` PASS | `4 / 4` WIRED | `0` debt markers (`TODO`/`FIXME`/`TBD`/`HACK`/`PLACEHOLDER`) |
| `04-02-PLAN.md` (Frontend Core) | `14 / 14` PASS | `3 / 3` WIRED | `0` debt markers |
| `04-03-PLAN.md` (Tabs 1 & 4) | `11 / 11` PASS | `3 / 3` WIRED | `0` debt markers |
| `04-04-PLAN.md` (Tabs 2, 3 & E2E) | `15 / 15` PASS | `4 / 4` WIRED | `0` debt markers |
| **Total** | **`52 / 52`** (`50` unique files) | **`14 / 14`** | **`0` issues** |

---

## 4. `04-REVIEW.md` Closure Audit (`11 / 11` Findings Verified Closed)

Every finding recorded in `04-REVIEW.md` was independently re-inspected in the current working tree (post commit `9519bc1`):

| ID | Severity | Finding Summary | Verification Status | Direct Code / Test Evidence |
|----|----------|-----------------|---------------------|-----------------------------|
| `CR-01` | Critical | `useMetrics` ref-latch stranding `#metricas` blank under React 19 `StrictMode` | ✓ CLOSED | `frontend/src/hooks/useStudioHooks.ts:183-207`: replaced eager `fetchedRef` with `resolvedRef` set strictly inside `.then((result) => { if (result !== null) resolvedRef.current = true; })`. |
| `CR-02` | Critical | `WatershedStudioTab` `hasBootedRef` latch stranding `#watershed` empty under `StrictMode` | ✓ CLOSED | `frontend/src/components/watershed/WatershedStudioTab.tsx:75,121-127,133-145`: `hasCompletedBootRef.current = true` executes strictly inside `if (response)` after resolved API response; `onLatency={setLastLatencyMs}` is referentially stable. |
| `WR-01` | Warning | `AppShell` `role="tablist"` keyboard navigation focus trap | ✓ CLOSED | `frontend/src/components/layout/AppShell.tsx:48-67,96-98`: stores `tabRefs` and calls `tabRefs.current[targetIndex]?.focus()` on `ArrowRight`, `ArrowLeft`, `Home`, and `End`. |
| `WR-02` | Warning | Failed upload in `XRayStudioTab` mutates `activeSource` eagerly & omits `400/413/415` CTA | ✓ CLOSED | `frontend/src/components/studio/XRayStudioTab.tsx:46-79,123-127,162-182` & `UploadDropzone.tsx`: `setActiveSource(source)` commits strictly on resolved result; `fileInputRef` wired to `"Carregar Nova Foto"` on `400`/`413`/`415`. |
| `WR-03` | Warning | No-op `"Enviar Foto de Grupo"` error CTA & eager source mutation in `WatershedStudioTab` | ✓ CLOSED | `frontend/src/components/watershed/WatershedStudioTab.tsx:74,121-127,212-228,240` & `WatershedPresetStrip.tsx`: `setSource` commits strictly on resolved response; `fileInputRef.current?.click()` wired on `400`/`413`/`415`. |
| `WR-04` | Warning | Stale render-closure slider commits on `onPointerUp`/`onKeyUp` in `WatershedSlidersCard` | ✓ CLOSED | `frontend/src/components/watershed/WatershedSlidersCard.tsx:103-109,181-239,292-361`: `commitFieldFromInput(field, event.currentTarget.value)` reads exact DOM value and skips redundant submits via `skipIfUnchanged` against `lastCommittedRef`. |
| `WR-05` | Warning | `validate_image_bytes` blocks event loop & mutates global `warnings.filters` | ✓ CLOSED | `backend/app/api/services/upload_guard.py:14,54,58-78`: offloads via `await run_in_threadpool(validate_image_bytes, raw)` with zero `catch_warnings`; locked by `test_upload_guard_offloads_validation_and_avoids_global_warnings_mutation`. |
| `IN-01` | Info | `DualLensComparison` duplicate abort logic / StrictMode stuck loading | ✓ CLOSED | `frontend/src/components/studio/DualLensComparison.tsx:29-50`: refactored onto shared `useAbortableRequest<DualLensPairResult>()` with post-resolution `resolvedRef`. |
| `IN-02` | Info | JPEG/WEBP decode-stage `ValueError` falling through to HTTP `500` | ✓ CLOSED | `inference_service.py:152-162` & `watershed_service.py:142-152` catch `ValueError` on uploads and return sanitized HTTP `400`; locked by `test_upload_decode_value_error_maps_to_400_on_predict_and_watershed`. |
| `IN-03` | Info | Active SPA catch-all returning HTTP `405` on unknown non-GET `/api/*` | ✓ CLOSED | `backend/app/api/main.py:73-81,139-146`: `_NonApiSpaRoute.matches` returns `Match.NONE, {}` for `/api` and `/api/*`; verified by `test_spa_mount_preserves_404_on_non_get_unknown_api_and_405_on_wrong_method` and live probe (`POST /api/does-not-exist` → `404`, `GET /api/predict` → `405`). |
| `IN-04` | Info | Hardcoded audit figures in `StratifiedErrorsPanel` & `ForensicAuditCard` | ✓ CLOSED | `StratifiedErrorsPanel.tsx:74-85,112-123,169` & `ForensicAuditCard.tsx:40-41`: dynamically interpolate `ds.n_sessions`, `funnel.raw_images`, `funnel.clean_images`, and gate Emerald compliance badges on audited JSON values. |

---

## 5. `PROIBIÇÃO 1–5`, ASVS L1 Security & Git Hygiene Sign-Off

| Prohibition / Policy | Status | Direct Audit Evidence |
|----------------------|--------|-----------------------|
| **PROIBIÇÃO 1** — Zero Deep Learning / Neural / OCR libraries | ✓ PASS | `pyproject.toml` scan returned `0` hits across `{torch, tensorflow, keras, ultralytics, timm, onnxruntime, paddleocr, easyocr, opencv-python, opencv-contrib-python}`; `test_proibicoes_1_2_3_5_ast_and_route_audit` passed. |
| **PROIBIÇÃO 2** — Zero raw `cv2.imread`/`cv2.imdecode` & zero LCD/outside-mask leakage | ✓ PASS | Python AST scan across entire `backend/app/**/*.py` returned `cv2_ast_calls = []` (`0` calls). Outside-mask invariance tests pass (`0.0`-diff). |
| **PROIBIÇÃO 3** — Zero random `train_test_split` / zero cross-fold leakage | ✓ PASS | Scan across `backend/app/**/*.py` returned `train_test_split_hits = []`; `evaluation_report` confirms `cross_fold_session_overlap = 0`. |
| **PROIBIÇÃO 4** — Zero raw image filename / `rel_path` / user home leaks & strict Git hygiene | ✓ PASS | Live responses across `/api/health`, `/api/presets`, `/api/predict`, `/api/watershed-demo`, `/api/metrics` and `frontend/src/` scan returned `0` `rel_path`, `0` raw dataset image filenames, and `0` absolute user-home path leaks. `git ls-files data/raw data/processed "models/*.joblib" frontend/dist frontend/node_modules` returned **`0` tracked files**. |
| **PROIBIÇÃO 5** — Zero v2 barn tracking / heatmap / sanitary backend code or fake telemetry | ✓ PASS | `FutureVisionTab.tsx` static scan returned `0` banned imports/hooks/timers; OpenAPI + router audit confirms exactly the 6 v1 `/api/*` endpoints and zero v2 route terms. |

---

## 6. Live Commands Executed & Observed Outputs

1. **Backend Full Regression & Ruff Lint:**
   - Command: `uv run pytest -v && uv run ruff check backend/`
   - Observed: **`272 passed, 108 warnings in 64.23s (0:01:04)`** (all 108 warnings originate from upstream `scikit-image/morphology/grayreconstruct.py:216` NumPy 2.5 deprecation; `0` warnings from project code) + **`All checks passed!`**.
2. **Frontend Typecheck, Biome Lint, Vitest & Production Build:**
   - Command: `cd frontend && npm run typecheck && npm run lint && npm test && npm run build`
   - Observed:
     - `tsc -p tsconfig.app.json --noEmit && tsc -p tsconfig.node.json --noEmit` → exit `0`
     - `biome check src` → `Checked 37 files in 71ms. No fixes applied.`
     - `vitest run` → `Test Files 2 passed (2)`, **`Tests 26 passed (26)`** (`scatter.test.ts`: `8`, `studioCore.test.ts`: `18`) in `243 ms`
     - `vite build` → `dist/index.html` `0.43 kB` (`0.30 kB` gzip), `dist/assets/index-8a8RUJTw.css` `27.14 kB` (`6.01 kB` gzip), `dist/assets/index-DEUPl2bk.js` `865.92 kB` (`239.72 kB` gzip) in `1.26 s`
3. **Live Single-Origin FastAPI E2E Probe (`TestClient(create_app())` over real `frontend/dist/`):**
   - Observed (`SINGLE_ORIGIN_E2E_PASS`):
     - `GET /` & `GET /metricas`: `200 text/html` (`AvianScale CV`), both `/assets/*` bundles `200`
     - `GET /api/does-not-exist`: `404 application/json` · `POST /api/does-not-exist`: `404 application/json` · `GET /api/presets/..%2F..%2Fetc%2Fpasswd/thumbnail`: `404 application/json` · `GET /api/predict`: `405 application/json`
     - `POST /api/predict` (all 6 OOF presets, warm CPU):
       - `sb_young_chick`: `249.73 g` vs. `180.0 g` (Fold `4`) in `56.1 ms`
       - `sb_intermediate_grower`: `424.05 g` vs. `576.0 g` (Fold `2`) in `76.6 ms`
       - `sb_heavy_broiler`: `998.33 g` vs. `1,555.0 g` (Fold `1`) in `65.7 ms`
       - `sb_hard_white_bucket`: `968.18 g` vs. `1,986.0 g` (Fold `3`) in `59.2 ms`
       - `sb_dual_lens_wide`: `383.22 g` vs. `541.0 g` (Fold `0`) in `79.7 ms`
       - `sb_dual_lens_ultrawide`: `384.16 g` vs. `541.0 g` (Fold `0`) in `58.9 ms`
       - **Dual-Lens delta:** **`0.9397 g`** (`< 2.0 g`)
     - `POST /api/watershed-demo` (all 4 multi-bird presets):
       - `ws_417`: `2` detected / `2` expected (`496.4 g` vs. `417.0 g` group sum) in `143.5 ms`
       - `ws_757`: `3` detected / `2` expected (`1,015.3 g` vs. `757.0 g` group sum) in `144.1 ms`
       - `ws_823`: `4` detected / `4` expected (`869.5 g` vs. `823.0 g` group sum) in `138.3 ms`
       - `ws_1520`: `2` detected / `2` expected (`647.8 g` vs. `1,520.0 g` group sum) in `152.1 ms`
     - Upload format probe: EXIF-stripped JPEG (`200`, `64.2 ms`), RGBA PNG (`200`, `16.9 ms`), WEBP (`200`, `18.7 ms`), manual Ultrawide override (`200`, `override_applied = true`, `lens_class = "ultrawide"`).
     - `GET /api/metrics`: `1,738` `actual_g` scatter points, `16` model comparison rows.
4. **GSD Artifact/Key-Link Queries & Static Compliance Scans:**
   - `gsd-tools.cjs query verify.artifacts` + `verify.key-links`: `52/52` artifacts and `14/14` key links verified across `04-01`..`04-04`.
   - `git ls-files data/raw data/processed "models/*.joblib" frontend/dist frontend/node_modules`: `COUNT=0`.
   - Code-only Recharts v3, `04-UI-SPEC.md` typography/copy/ARIA, Watershed bounds parity, `FutureVisionTab.tsx` `PROIBIÇÃO 5`, and AST `cv2.imread`/`cv2.imdecode` scans: `0` violations.

---

*Verification completed autonomously by `gsd-verifier` on `2026-11-24T15:04:25Z`. Ready for Phase 4 completion and final project sign-off.*
