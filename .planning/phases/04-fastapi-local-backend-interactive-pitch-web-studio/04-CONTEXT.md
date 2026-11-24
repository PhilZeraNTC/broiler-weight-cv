# Phase 4: FastAPI Local Backend & Interactive Pitch Web Studio - Context

**Gathered:** 2026-11-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the complete, pitch-ready local web application (**AvianScale CV**) backed by a `< 500 ms` local FastAPI REST API (`backend/app/api/`) and an interactive React + TypeScript + Vite + Tailwind CSS v4 frontend (`frontend/`), covering all five Phase 4 requirements (`WEB-01` through `WEB-05`):
- **WEB-01:** Local FastAPI backend serving `/api/health`, `/api/presets`, `/api/presets/{preset_id}/thumbnail`, `/api/predict`, `/api/watershed-demo`, and `/api/metrics`, consuming the single-source Phase 2 PDI engine and Phase 3 dual-bundle model artifacts with zero training-serving skew and zero raw filename leakage.
- **WEB-02:** Interactive "Estúdio Raio-X PDI" single-bird weight studio with drag-and-drop upload, 6 curated honest-OOF presets (including the `541__s0` Wide vs. Ultrawide pair), 4-stage X-Ray visual pipeline inspection, optical EXIF inspector + manual lens/zoom override controls, live weight/error-band/age-stage estimation, and interpretable PDI feature readout.
- **WEB-03:** Interactive "Separação Watershed" multi-bird demo tab with 4 curated group presets + custom upload, 4-stage classical Watershed visuals, bounded interactive tuning sliders (`h_rel`, `min_distance_rel`, `min_area_ratio`), per-bird illustrative weight breakdown, and mandatory `group_sum_only` scientific disclaimer.
- **WEB-04:** Visually distinct "Visão Futura — Monitoramento de Aviário (v2)" conceptual roadmap surface documenting multi-frame barn tracking (`BARN-01`), flock density heatmaps (`BARN-02`), and stationary-bird sanitary alerts (`BARN-03`) strictly as future scope with zero tracking/heatmap backend code (`PROIBIÇÃO 5`).
- **WEB-05:** Interactive "Auditoria & Métricas" scientific dashboard powered by `/api/metrics`, displaying dataset audit stats, 5-fold grouped OOF model comparison, 4-condition optical calibration ablation + dual-mode anti-confounding proof, filterable Real vs. Predicted Recharts `ScatterChart` (with `y = x` diagonal), stratified band/lens/scene errors, and top interpretable feature importances.

</domain>

<decisions>
## Implementation Decisions

### API Architecture, Endpoints & Security Guardrails (`WEB-01`)
- **D-01 — Modular FastAPI package & lifespan startup (`backend/app/api/`):** Structure the backend under `backend/app/api/` (`main.py`, `schemas.py`, `dependencies.py`, `routers/{health,presets,predict,watershed,metrics}.py`, `services/{inference,watershed,presets,metrics}.py`) using a FastAPI `lifespan` context manager that sets `cv2.setNumThreads(2)`, loads `models/weight_regressor.joblib` via `load_model_bundle(..., expected_pdi_config_hash=PDIConfig().config_hash())`, loads `models/evaluation_report.json`, `models/dataset_audit_summary.json`, `models/pdi_calibration_audit.json`, and `models/presets_catalog.json`, and builds an in-memory server-side lookup mapping opaque `preset_id -> Path` resolved exclusively from `manifest_clean.parquet` and `manifest_quarantine.parquet` validated under `RAW_DATASET_DIR.resolve()`. — **Reversibility:** `costly` — defines the module layout and typed response schemas consumed by both pytest contract tests and frontend TypeScript API clients.
- **D-02 — Dual-bundle honest OOF routing & opaque preset security:**
  - `POST /api/predict` accepts `multipart/form-data` with either `file: UploadFile` OR `preset_id: str` (mutually exclusive — reject both or neither with HTTP `422`), plus optional bounded manual optical overrides (`focal_length_mm` in `[0.5, 200.0]`, `zoom_multiple` in `[0.1, 20.0]`).
  - When `preset_id` is provided, resolve it strictly against the loaded `single_bird_presets` allowlist (unknown `preset_id` returns HTTP `404`), run `run_single_bird_pdi(..., config=bundle_pdi_config, return_xray=True)`, and score strictly with `bundle["fold_pipelines"][preset.fold]` (honest out-of-fold evaluation, never `production_pipeline`) while returning the preset's `ground_truth_weight_g`, `fold`, `oof_evaluation=True`, and delta against ground truth.
  - When `file` is uploaded, read bounded bytes (`MAX_UPLOAD_BYTES = 15 * 1024 * 1024`, returning HTTP `413` on oversize), validate allowed image magic/Pillow format (`JPEG`, `PNG`, `WEBP` — returning HTTP `400`/`415` on invalid/corrupt or decompression-bomb images without exposing stack traces), run `run_single_bird_pdi(..., return_xray=True)`, and score with `bundle["production_pipeline"]` (`oof_evaluation=False`).
  - Never expose `rel_path`, raw dataset `.jpg` filenames, or absolute filesystem paths in any JSON response, header, or error message (`PROIBIÇÃO 4`).
- **D-03 — Non-blocking CPU offloading & preset thumbnails:**
  - Define `/api/predict`, `/api/watershed-demo`, and `/api/presets/{preset_id}/thumbnail` as `async def` handlers that validate inputs and perform bounded async file reading (`await file.read(MAX_UPLOAD_BYTES + 1)`), then offload all Pillow/OpenCV/scikit-image/LightGBM CPU execution via `starlette.concurrency.run_in_threadpool` so the Uvicorn event loop never blocks on concurrent `/api/metrics` or thumbnail fetches.
  - Expose `GET /api/presets/{preset_id}/thumbnail` returning a cached, EXIF-transposed JPEG thumbnail (`long_edge=320`, `quality=78`, `Cache-Control: public, max-age=3600`) for both single-bird and Watershed presets using only opaque `preset_id`, enabling visual gallery cards without bloating `GET /api/presets` or touching raw filenames.
- **D-04 — Watershed demo API contract (`POST /api/watershed-demo`):**
  - Accepts `multipart/form-data` with mutually exclusive `file: UploadFile` OR `preset_id: str` (allowlisted against `watershed_presets`), plus optional bounded sliders `h_rel` (`[0.05, 0.50]`), `min_distance_rel` (`[0.10, 0.80]`), `min_area_ratio` (`[0.02, 0.30]`), and optional optical overrides (`focal_length_mm`, `zoom_multiple`).
  - Executes `run_watershed_pdi(..., return_visuals=True)` and scores each separated `WatershedInstance.features` using `bundle["production_pipeline"]` via `predict_from_feature_dict` to produce illustrative per-bird weights (`estimated_weight_g`), aggregate `estimated_total_weight_g`, and `estimated_mean_weight_g`.
  - Always returns the explicit scientific `disclaimer` string (`ground_truth_scope = "group_sum_only"`) and, for curated presets, includes `expected_bird_count`, `group_sum_weight_g`, and `mean_bird_weight_reference_g` alongside the detected count and estimated totals.
- **D-05 — Single-origin production serving + Vite dev proxy:**
  - During development, Vite runs on `http://127.0.0.1:5173` and proxies `/api/*` → `http://127.0.0.1:8000` (zero CORS friction). FastAPI also enables strict localhost-only CORS (`http://127.0.0.1:5173`, `http://localhost:5173`, `http://127.0.0.1:8000`, `http://localhost:8000`) for direct API tooling.
  - When `frontend/dist/index.html` exists at startup, FastAPI mounts `frontend/dist/assets` and serves `frontend/dist/index.html` for non-`/api` SPA routes so pitch-day execution requires only a single command (`uv run uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000`).

### Frontend Architecture, Navigation & Visual Direction (`WEB-02`–`WEB-05`)
- **D-06 — Portuguese (`pt-BR`) Pitch Studio with 4-Tab Top Navigation:**
  - All user-facing UI labels, headings, tooltips, banners, and scientific disclaimers are written in crisp Brazilian Portuguese (`pt-BR`) tailored for an academic/product agritech pitch; technical metrics (`MAE`, `RMSE`, `MAPE`, `R²`, `CLAHE`, `EXIF`, `Watershed`, `LightGBM`) retain standard scientific notation.
  - Top navigation bar provides 4 top-level workspaces (state-driven SPA tabs with URL hash sync `#estudio`, `#watershed`, `#metricas`, `#visao-futura` so browser back/forward and deep-linking during pitch work seamlessly):
    1. **Estúdio Raio-X** (`WEB-02` — default landing tab)
    2. **Separação Watershed** (`WEB-03`)
    3. **Auditoria & Métricas** (`WEB-05`)
    4. **Visão Futura (v2)** (`WEB-04` — clearly badged as `Fora do Escopo v1 / Conceitual`)
  - Compact top header displays brand mark (`AvianScale CV`), classical-PDI badge (`PDI Clássico + EXIF · Zero Deep Learning`), and live API health status pill (`model_version` + latency indicator).
- **D-07 — Visual Direction: Precision Agritech Clinical Light Theme:**
  - Clean high-contrast light surface palette (ideal for bright lecture halls/projectors) paired with dark technical viewports (`#0f172a` / slate-950) strictly behind X-Ray and Watershed image stages so silhouettes, contour overlays, and Inferno distance heatmaps pop with maximum contrast.
  - Deep agritech emerald/forest primary accent for live actions and verified states, slate ink typography with strict `tabular-nums` on every weight, gram delta, optical focal length, latency, and metric cell, and warm amber strictly reserved for scientific warnings (EXIF missing, `out_of_training_range`, `group_sum_only` disclaimer, disqualified ablation rows).
  - Exact spacing scale, color tokens, typography hierarchy, and copywriting contract will be formally locked in `04-UI-SPEC.md` via `/gsd-ui-phase 4`.

### Interactive X-Ray Studio & Optical Calibration UX (`WEB-02`)
- **D-08 — Zero-empty-state pitch boot & 6-preset gallery with Dual-Lens Pair highlight:**
  - On initial mount of **Estúdio Raio-X**, automatically execute `POST /api/predict` for the default curated preset (`sb_intermediate_grower`) so the presenter lands immediately on a live, complete 4-stage X-Ray analysis instead of an empty placeholder.
  - Display the 6 curated presets in a compact thumbnail strip grouped clearly (4 weight/difficulty scenarios + a highlighted **"Par Dual-Lens — Mesmo Frango (541 g)"** tandem badge for `sb_dual_lens_wide` and `sb_dual_lens_ultrawide` enabling rapid side-by-side optical invariance demonstration).
  - Provide a drag-and-drop / click file upload dropzone accepting `.jpg`, `.jpeg`, `.png`, `.webp` (client-side 15 MB guard matching server `MAX_UPLOAD_BYTES`) with clear active-source indicator (Curated OOF Preset vs. Upload Externo).
- **D-09 — 4-Stage X-Ray Viewer, Prediction Card & Manual Optical Override:**
  - Prediction summary card prominently shows: estimated weight in grams (`g`) and kilograms (`kg`), band-stratified confidence interval (`±MAE` and `p68` from `band_error_intervals`), age-stage badge (`map_weight_to_age_stage`), `out_of_training_range` amber alert (`116–2.093 g / ~7–29 dias`), latency badge (`ms` with `< 500 ms` SLA indicator), and — when running a curated preset — an **Auditoria OOF (Fold K)** strip comparing Real vs. Estimado (`erro absoluto` & `erro %`).
  - 4-stage X-Ray viewer defaults to a balanced 2×2 grid (`1. Original Orientada EXIF`, `2. Pré-processamento CLAHE-Lab`, `3. Máscara Binária (LCD Excluído)`, `4. Contorno + Elipse + Eixos`) with 1-click expand/focus view for any single stage on projector screens.
  - Optical Calibration & Override drawer/panel displays detected `FocalLength (mm)`, `zoomMultiple`, `lens_class`, `exif_optical_present`, and `override_applied`, with quick presets (`Wide 4,25 mm / 1,0×`, `Ultrawide 1,65 mm / 0,6×`), custom numeric inputs, **"Recalibrar e Reestimar"**, and **"Restaurar EXIF"** buttons. When an uploaded image lacks EXIF optical tags, display a clear amber notice explaining fallback defaults (`4,25 mm / 1,0×`) and guiding manual lens selection.
  - Interpretable PDI telemetry table displays key mask-gated descriptors (`area_optical_norm`, `area_px`, `major_axis_optical_norm`, `minor_axis_optical_norm`, `solidity`, `eccentricity`, `scene_type`, `lcd_excluded_px`, `segmentation_fallback`) plus raw 39-feature expandable drawer for technical jury inspection.

### Watershed Module, Metrics Dashboard & Conceptual Roadmap (`WEB-03`, `WEB-04`, `WEB-05`)
- **D-10 — Interactive Watershed Studio (`WEB-03`):**
  - Auto-loads curated preset `ws_417` on tab open (cached in client state once loaded), displays all 4 curated group presets (`ws_417`, `ws_757`, `ws_823`, `ws_1520`) with visual thumbnails + custom multi-bird upload dropzone.
  - Displays all 4 Watershed stages (`1. Original`, `2. Máscara União Multi-Ave`, `3. Mapa de Distância + Picos`, `4. Bacias Coloridas #ID`) in a 2×2 grid with stage zoom.
  - Provides 3 labeled range sliders matching exact backend ASVS V5 bounds: `h_rel` (`0.05–0.50`, step `0.01`, default `0.15`), `min_distance_rel` (`0.10–0.80`, step `0.02`, default `0.34`), `min_area_ratio` (`0.02–0.30`, step `0.01`, default `0.08`), triggering re-execution on pointer/key commit (or explicit "Aplicar Parâmetros" / "Restaurar Padrões" buttons) using `AbortController` to cancel superseded requests.
  - Shows a permanent amber scientific disclaimer banner (`ground_truth_scope: group_sum_only`), aggregate KPI cards (Detected Birds vs. Expected Birds, Estimated Total vs. Dataset Group Sum, Estimated Mean vs. Group Mean), and a sortable/structured per-bird instance table (`#ID`, Área `px²`, Área Óptica Norm., Peso Ilustrativo `g`).
- **D-11 — Scientific Audit & Metrics Dashboard (`WEB-05`):**
  - Powered by a single `GET /api/metrics` payload loaded on demand and cached in memory.
  - Includes: (1) Executive KPI strip (clean dataset counts, `0` cross-fold overlap, winner `LightGBMLog C2` global & session-level metrics, PDI `p95` latency); (2) **Recharts `ScatterChart`** of Real vs. Predicted (`y = x` `<ReferenceLine segment={[{x: 0, y: 0}, {x: 2200, y: 2200}]} />`, `isAnimationActive={false}`) with mode toggle (`Todas as Imagens (1.738)` vs `Média por Sessão (188)`) and interactive lens/scene/weight-band filter pills; (3) **Optical Ablation `BarChart`** comparing `C1`..`C4` (MAE, BalMAPE, and Dual-Mode Counterfactual Shift % highlighting `C2 = 0.0%` vs `C3/C4` disqualified); (4) **Stratified Error Breakdown** (`<500g`, `500–1000g`, `>1000g`, Wide vs. Ultrawide, Bucket vs. Scale/Litter); (5) **Model Family Comparison Table** (`16` model×condition rows with eligibility & anti-confounding badges); (6) **Top-12 Interpretable Feature Importances horizontal `BarChart`**; and (7) **Data Hygiene & Anti-Leakage Compliance Card** (`dataset_audit_summary` + `pdi_calibration_audit`).
- **D-12 — Conceptual Future Barn Vision Tab (`WEB-04`):**
  - Renders a clean, explicitly badged conceptual architecture & product roadmap view (`Status: Escopo Futuro v2 — Não Implementado no MVP v1`) detailing the 3 post-MVP extensions from `REQUIREMENTS.md` (`BARN-01` multi-frame overhead tracking, `BARN-02` barn density heatmap, `BARN-03` stationary-bird sanitary alert) and how each builds directly on the validated classical PDI + Watershed foundation — with zero mock live cameras, zero fake sensor telemetry, and zero backend endpoints (`PROIBIÇÃO 5`).

### the agent's Discretion
- Exact React component decomposition, custom hooks (`useAbortableRequest`, `useMetrics`, `usePresets`), and TypeScript interface file organization under `frontend/src/`.
- Exact Tailwind CSS v4 `@theme` design tokens and typography pairing (to be finalized in `04-UI-SPEC.md`).
- Exact caching mechanism for `/api/presets/{preset_id}/thumbnail` (bounded in-memory `dict[str, bytes]` pre-warmed or lazily cached for the 10 curated presets — `< 500 KB` total RAM).
- Exact FastAPI exception handler formatting and Pydantic v2 validation helpers.
- Frontend toolchain configuration (`vite.config.ts`, `tsconfig.app.json`, `tsconfig.node.json`, `biome.json`, `package.json`) adhering strictly to `STACK.md` (`React 19.3`, `TypeScript 7.0`, `Vite 8.3`, `Tailwind v4.3`, `Recharts 3.11 + react-is 19.3`, `lucide-react`, `@biomejs/biome` — zero ESLint/typescript-eslint).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Product & Project Specification
- `PRD.md` — Core product requirements (`REQ-01`..`REQ-15`), hard prohibitions (`PROIBIÇÃO 1`..`5`), target `< 500 ms` latency SLA, and pitch scope.
- `.planning/REQUIREMENTS.md` — Formal v1 requirements (`WEB-01`..`WEB-05`), Phase 4 acceptance criteria, and explicit v2/out-of-scope boundary (`BARN-01`..`BARN-03`).
- `.planning/ROADMAP.md` — Phase 4 goal, success criteria, and milestone schedule (MVP freeze `2026-11-28`, pitch `2026-11-30`).
- `.planning/research/STACK.md` — Exact pinned frontend/backend library versions (`React 19.3`, `TS 7.0`, `Vite 8.3`, `Tailwind v4.3`, `Recharts 3.11` + `react-is 19.3`, `lucide-react`, `@biomejs/biome`, `FastAPI 0.165`, `Uvicorn 0.54`, `python-multipart`).
- `.planning/research/ARCHITECTURE.md` — Reference Phase 4 FastAPI router/service layout, `lifespan` model loading, `run_in_threadpool`, dual-bundle OOF preset routing, and Vite `/api` proxy pattern.
- `.planning/research/PITFALLS.md` — Pitfalls #5 (`< 500 ms` budget), #7 (EXIF orientation), and #8 (EXIF-stripped/RGBA uploads, honest OOF presets, path-traversal allowlist, NVM Node path).

### Upstream Phase 1–3 Contracts & Published Artifacts
- `backend/app/config.py` — Canonical path constants (`RAW_DATASET_DIR`, `CLEAN_MANIFEST_PATH`, `QUARANTINE_MANIFEST_PATH`, `MODEL_BUNDLE_PATH`, `EVALUATION_REPORT_PATH`, `PRESETS_CATALOG_PATH`, `AUDIT_SUMMARY_PATH`, `PDI_CALIBRATION_AUDIT_PATH`).
- `backend/app/pdi/pipeline.py` — `run_single_bird_pdi(source, config=..., optical_override=..., return_xray=True)` & `PDIResult`.
- `backend/app/pdi/xray.py` — `XRayArtifacts` (raw RFC 4648 Base64 fields: `stage1_original_b64`, `stage2_clahe_b64`, `stage3_mask_b64`, `stage4_overlay_b64`, `display_w`, `display_h`, `total_b64_bytes`).
- `backend/app/pdi/watershed.py` — `run_watershed_pdi(...)`, `WatershedResult`, `WatershedInstance`, `WatershedVisuals`, and ASVS V5 parameter bounds (`MIN/MAX_H_REL`, `MIN/MAX_DISTANCE_REL`, `MIN/MAX_AREA_RATIO`).
- `backend/app/pdi/io_exif.py` — `OpticalOverride`, `OpticalContext`, `load_working_rgb`, `MIN/MAX_VALID_FOCAL_LENGTH_MM`, `MIN/MAX_VALID_ZOOM_MULTIPLE`.
- `backend/app/modeling/artifacts.py` — `load_model_bundle(path, expected_pdi_config_hash=...)` and `MODEL_BUNDLE_SCHEMA_VERSION`.
- `backend/app/modeling/models.py` — `predict_from_feature_dict(pipeline, feature_names, features)`.
- `backend/app/modeling/age_stage.py` — `map_weight_to_age_stage(weight_g)` & `AgeStageResult`.
- `models/presets_catalog.json` — Exact curated schema for `6` single-bird OOF presets and `4` Watershed group presets (opaque `preset_id` + `image_id`, zero raw filenames).
- `models/evaluation_report.json` — Complete OOF evaluation metrics, 4-condition ablation, `1,738` scatter points, stratified breakdowns, and feature importances.
- `models/dataset_audit_summary.json` — Dataset forensic curation counts & quarantine reasons.
- `models/pdi_calibration_audit.json` — PDI segmentation fallback, outside-mask invariance (`0.0`-diff), and X-Ray latency audit metrics.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/pdi/pipeline.py` (`run_single_bird_pdi`): Accepts either `Path` or `bytes`, returns `PDIResult` with `features`, `optical`, `segmentation`, `pdi_config_hash`, `latency_ms`, and `xray` (`XRayArtifacts`). Measured `p95 = 88.3 ms` on local CPU.
- `backend/app/pdi/watershed.py` (`run_watershed_pdi`): Accepts `Path` or `bytes` + bounded `h_rel`, `min_distance_rel`, `min_area_ratio`, returns `WatershedResult` with `instances` (each carrying a full 49-key `features` dict ready for `predict_from_feature_dict`) and 4-stage Base64 `WatershedVisuals`.
- `backend/app/modeling/artifacts.py` (`load_model_bundle`): Validates schema version, `PDIConfig` reconstruction, and `pdi_config_hash == "43a3cfaa3208019e"` at startup; returns `production_pipeline`, `fold_pipelines[0..4]`, `feature_names` (39 strict physics-normalized features), and `band_error_intervals`.
- `backend/app/modeling/models.py` (`predict_from_feature_dict`): Validated single-sample inference (`~2.4 ms`, zero warnings) accepting `dict[str, float]` directly from `PDIResult.features` or `WatershedInstance.features`.
- `backend/app/modeling/age_stage.py` (`map_weight_to_age_stage`): Deterministic Cobb Avian growth-stage mapping + `out_of_training_range` flag (`116–2,093 g`, `~7–29 days`).

### Established Patterns
- **Single-source EXIF-first Pillow ingestion:** Never decode images with `cv2.imread` / `cv2.imdecode`; always pass `Path` or raw `bytes` into `run_single_bird_pdi` / `run_watershed_pdi` (or `load_oriented_rgb` for thumbnails).
- **Raw Base64 contract from Phase 2:** `XRayArtifacts` and `WatershedVisuals` store raw RFC 4648 Base64 without `data:image/...` prefixes (Stage 1/2/4 are JPEG; Stage 3 X-Ray mask & Stage 2 Watershed union mask are 1-channel PNG). The API should preserve explicit MIME types or emit clean `data:image/jpeg;base64,...` / `data:image/png;base64,...` strings consistently documented in Pydantic & TypeScript schemas.
- **No raw filenames in API/JSON:** `models/presets_catalog.json` intentionally omits `rel_path` and `.jpg` filenames. The Phase 4 preset resolver joins `image_id` against `manifest_clean.parquet` (single-bird) and `manifest_quarantine.parquet` (watershed) strictly in server memory and checks `resolved_path.is_relative_to(RAW_DATASET_DIR.resolve())` + `not resolved_path.is_symlink()`.
- **Tooling environment:** Python `3.12.14` managed by `uv` (`[tool.uv] package = false`, already includes `fastapi`, `uvicorn[standard]`, `pydantic`, `python-multipart`, `pytest`, `httpx`, `ruff`); Node `v24.21.0` located at `/home/philzerantc/.nvm/versions/node/v24.21.0/bin`.

### Integration Points
- `backend/app/api/main.py` — New ASGI entrypoint (`uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000`) connecting FastAPI routers to Phase 1–3 modules and optional static `frontend/dist/`.
- `frontend/` — Currently an empty directory ready to be scaffolded with `package.json`, `vite.config.ts`, `tsconfig*.json`, `biome.json`, `index.html`, and `src/`.

</code_context>

<specifics>
## Specific Ideas

- **Live Dual-Lens Proof Flow (`541__s0`):** `models/presets_catalog.json` includes `sb_dual_lens_wide` (`4.25 mm`, OOF `383.22 g`) and `sb_dual_lens_ultrawide` (`1.65 mm`, OOF `384.16 g`) from the exact same `541 g` bird in fold `0` — a `0.94 g` (`0.24%`) difference despite a ~`6.6×` pixel-area swing. Surfacing a prominent 1-click "Comparar Par Dual-Lens" indicator in the X-Ray Studio makes the pitch demonstration of EXIF optical calibration unforgettable.
- **Honest Scientific Framing:** Even on hard cases like `sb_hard_white_bucket` (`1.986 g` ground truth vs `968 g` OOF due to sparse `>1,000 g` white-on-white tail) or `ws_757` (`expected 2`, default Watershed detects `3`, tunable via slider), presenting honest OOF numbers and tunable controls directly strengthens academic credibility.
- **Latency Visibility:** Since live CPU PDI + LightGBM runs in `~91 ms` (over `5×` faster than the `< 500 ms` PRD SLA), displaying the real server `total_ms` (`pdi_ms + inference_ms`) badge on every prediction gives tangible proof of lightweight edge-ready classical CV performance.

</specifics>

<deferred>
## Deferred Ideas

- Live RTSP/webcam multi-frame bird tracking in commercial barns (`BARN-01`) — documented conceptually in `WEB-04` only, deferred to post-MVP v2.
- Spatial flock density heatmaps over barn floorplans (`BARN-02`) — documented conceptually in `WEB-04` only, deferred to post-MVP v2.
- Temporal stationary-bird sanitary alert engine (`BARN-03`) — documented conceptually in `WEB-04` only, deferred to post-MVP v2.
- User accounts, database persistence, batch CSV export, or cloud deployment — out of scope for the local academic/product pitch MVP (`PRD.md`).

</deferred>

---

*Phase: 04-fastapi-local-backend-interactive-pitch-web-studio*
*Context gathered: 2026-11-24*
