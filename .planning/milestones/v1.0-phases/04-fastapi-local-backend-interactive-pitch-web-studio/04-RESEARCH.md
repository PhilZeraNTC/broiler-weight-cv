# Phase 4: FastAPI Local Backend & Interactive Pitch Web Studio - Research

**Researched:** 2026-11-24
**Domain:** Local FastAPI Serving API (`< 500 ms` CPU) + React 19 / TypeScript 7 / Vite 8 / Tailwind CSS v4 / Recharts 3 Interactive Pitch Web Studio
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### API Architecture, Endpoints & Security Guardrails (`WEB-01`)
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
  - Accepts `multipart/form-data` with mutually exclusive `file: UploadFile` OR `preset_id: str` (allowlisted against `watershed_presets`), plus optional bounded sliders `h_rel` (`[0.05, 0.50]`), `min_distance_rel` (`[0.10, 0.80]`), `min_area_ratio` (`[0.02, 0.30]`), and optional optical overrides (`focal_length_mm`, `zoom_multiple`). *(See reconciliation note in Technical Findings below: Phase 2 `backend/app/pdi/config.py:18-23,104-106` and approved `04-UI-SPEC.md:142-144` lock the authoritative ASVS V5 bounds & defaults as `h_rel` `[0.05, 0.80]` default `0.20`, `min_distance_rel` `[0.20, 2.50]` default `0.85`, `min_area_ratio` `[0.01, 0.45]` default `0.10`.)*
  - Executes `run_watershed_pdi(..., return_visuals=True)` and scores each separated `WatershedInstance.features` using `bundle["production_pipeline"]` via `predict_from_feature_dict` to produce illustrative per-bird weights (`estimated_weight_g`), aggregate `estimated_total_weight_g`, and `estimated_mean_weight_g`.
  - Always returns the explicit scientific `disclaimer` string (`ground_truth_scope = "group_sum_only"`) and, for curated presets, includes `expected_bird_count`, `group_sum_weight_g`, and `mean_bird_weight_reference_g` alongside the detected count and estimated totals.
- **D-05 — Single-origin production serving + Vite dev proxy:**
  - During development, Vite runs on `http://127.0.0.1:5173` and proxies `/api/*` → `http://127.0.0.1:8000` (zero CORS friction). FastAPI also enables strict localhost-only CORS (`http://127.0.0.1:5173`, `http://localhost:5173`, `http://127.0.0.1:8000`, `http://localhost:8000`) for direct API tooling.
  - When `frontend/dist/index.html` exists at startup, FastAPI mounts `frontend/dist/assets` and serves `frontend/dist/index.html` for non-`/api` SPA routes so pitch-day execution requires only a single command (`uv run uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000`).

#### Frontend Architecture, Navigation & Visual Direction (`WEB-02`–`WEB-05`)
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

#### Interactive X-Ray Studio & Optical Calibration UX (`WEB-02`)
- **D-08 — Zero-empty-state pitch boot & 6-preset gallery with Dual-Lens Pair highlight:**
  - On initial mount of **Estúdio Raio-X**, automatically execute `POST /api/predict` for the default curated preset (`sb_intermediate_grower`) so the presenter lands immediately on a live, complete 4-stage X-Ray analysis instead of an empty placeholder.
  - Display the 6 curated presets in a compact thumbnail strip grouped clearly (4 weight/difficulty scenarios + a highlighted **"Par Dual-Lens — Mesmo Frango (541 g)"** tandem badge for `sb_dual_lens_wide` and `sb_dual_lens_ultrawide` enabling rapid side-by-side optical invariance demonstration).
  - Provide a drag-and-drop / click file upload dropzone accepting `.jpg`, `.jpeg`, `.png`, `.webp` (client-side 15 MB guard matching server `MAX_UPLOAD_BYTES`) with clear active-source indicator (Curated OOF Preset vs. Upload Externo).
- **D-09 — 4-Stage X-Ray Viewer, Prediction Card & Manual Optical Override:**
  - Prediction summary card prominently shows: estimated weight in grams (`g`) and kilograms (`kg`), band-stratified confidence interval (`±MAE` and `p68` from `band_error_intervals`), age-stage badge (`map_weight_to_age_stage`), `out_of_training_range` amber alert (`116–2.093 g / ~7–29 dias`), latency badge (`ms` with `< 500 ms` SLA indicator), and — when running a curated preset — an **Auditoria OOF (Fold K)** strip comparing Real vs. Estimado (`erro absoluto` & `erro %`).
  - 4-stage X-Ray viewer defaults to a balanced 2×2 grid (`1. Original Orientada EXIF`, `2. Pré-processamento CLAHE-Lab`, `3. Máscara Binária (LCD Excluído)`, `4. Contorno + Elipse + Eixos`) with 1-click expand/focus view for any single stage on projector screens.
  - Optical Calibration & Override drawer/panel displays detected `FocalLength (mm)`, `zoomMultiple`, `lens_class`, `exif_optical_present`, and `override_applied`, with quick presets (`Wide 4,25 mm / 1,0×`, `Ultrawide 1,65 mm / 0,6×`), custom numeric inputs, **"Recalibrar e Reestimar"**, and **"Restaurar EXIF"** buttons. When an uploaded image lacks EXIF optical tags, display a clear amber notice explaining fallback defaults (`4,25 mm / 1,0×`) and guiding manual lens selection.
  - Interpretable PDI telemetry table displays key mask-gated descriptors (`area_optical_norm`, `area_px`, `major_axis_optical_norm`, `minor_axis_optical_norm`, `solidity`, `eccentricity`, `scene_type`, `lcd_excluded_px`, `segmentation_fallback`) plus raw 39-feature expandable drawer for technical jury inspection.

#### Watershed Module, Metrics Dashboard & Conceptual Roadmap (`WEB-03`, `WEB-04`, `WEB-05`)
- **D-10 — Interactive Watershed Studio (`WEB-03`):**
  - Auto-loads curated preset `ws_417` on tab open (cached in client state once loaded), displays all 4 curated group presets (`ws_417`, `ws_757`, `ws_823`, `ws_1520`) with visual thumbnails + custom multi-bird upload dropzone.
  - Displays all 4 Watershed stages (`1. Original`, `2. Máscara União Multi-Ave`, `3. Mapa de Distância + Picos`, `4. Bacias Coloridas #ID`) in a 2×2 grid with stage zoom.
  - Provides 3 labeled range sliders matching exact backend ASVS V5 bounds: `h_rel` (`0.05–0.50`, step `0.01`, default `0.15`), `min_distance_rel` (`0.10–0.80`, step `0.02`, default `0.34`), `min_area_ratio` (`0.02–0.30`, step `0.01`, default `0.08`), triggering re-execution on pointer/key commit (or explicit "Aplicar Parâmetros" / "Restaurar Padrões" buttons) using `AbortController` to cancel superseded requests. *(Aligned to authoritative `backend/app/pdi/config.py` + approved `04-UI-SPEC.md`: `h_rel` `0.05–0.80` default `0.20`, `min_distance_rel` `0.20–2.50` default `0.85`, `min_area_ratio` `0.01–0.45` default `0.10`.)*
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

### Deferred Ideas (OUT OF SCOPE)
- Live RTSP/webcam multi-frame bird tracking in commercial barns (`BARN-01`) — documented conceptually in `WEB-04` only, deferred to post-MVP v2.
- Spatial flock density heatmaps over barn floorplans (`BARN-02`) — documented conceptually in `WEB-04` only, deferred to post-MVP v2.
- Temporal stationary-bird sanitary alert engine (`BARN-03`) — documented conceptually in `WEB-04` only, deferred to post-MVP v2.
- User accounts, database persistence, batch CSV export, or cloud deployment — out of scope for the local academic/product pitch MVP (`PRD.md`).
</user_constraints>

## Project Constraints (from GEMINI.md)

Extracted from `GEMINI.md:11-18, 108-143` `[VERIFIED: GEMINI.md:11-18]`:
- **Classical PDI Only (`PROIBIÇÃO 1`):** Zero deep learning (CNNs, YOLO, ViTs, SAM, PyTorch, TensorFlow, ONNX Runtime) in segmentation or regression.
- **Anti-Leakage (`PROIBIÇÃO 2` & `PROIBIÇÃO 3`):** Mandatory SF-400 LCD/platter exclusion before feature extraction and strictly `session_id`-grouped 5-fold `StratifiedGroupKFold` validation.
- **Timeline:** MVP code freeze **2026-11-28** for pitch on **2026-11-30**.
- **Performance:** End-to-end `/api/predict` inference `< 500 ms` per image on local CPU (target p95 `< 400 ms` on real `4080×1840` JPEGs).
- **Tech Stack:** Backend Python `3.12` (`uv`-managed, already locked in `pyproject.toml`/`uv.lock` with `fastapi`, `uvicorn[standard]`, `pydantic`, `python-multipart`, `pytest`, `httpx`, `ruff`); Frontend React `19.3` + TypeScript `7.0` + Vite `8.3` + Tailwind CSS `v4.3` + Recharts `3.11` (`react-is@19.3.1`) + `lucide-react` + `@biomejs/biome` (zero ESLint/`typescript-eslint`).
- **Data Hygiene (`PROIBIÇÃO 4`):** `data/raw/`, `data/processed/`, `models/*.joblib`, `*.jpg`/`*.jpeg`/`*.png`, `node_modules/`, and `frontend/dist/` never enter Git (verified in `.gitignore:1-18` `[VERIFIED: .gitignore:1-18]`); never expose `rel_path`, raw `.jpg` filenames, or absolute filesystem paths in API responses, error messages, or UI text.
- **Non-Blocking FastAPI CPU Offload (`GEMINI.md:115` + `04-CONTEXT.md` `D-03`):** Use `async def` routes strictly for bounded async `UploadFile` intake, offloading all Pillow/OpenCV/scikit-image/LightGBM CPU calls via `starlette.concurrency.run_in_threadpool`.
- **Single-Source Pillow EXIF Ingestion:** Never call `cv2.imread()` or `cv2.imdecode()` anywhere in `backend/app/api/`; always pass `Path` or `bytes` to `run_single_bird_pdi`, `run_watershed_pdi`, or `load_working_rgb`.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **WEB-01** | Local FastAPI backend serving `/api/health`, `/api/presets`, `/api/presets/{preset_id}/thumbnail`, `/api/predict`, `/api/watershed-demo`, `/api/metrics` with Pydantic v2 validation, `lifespan` caching, bounded `async def` intake + `run_in_threadpool` CPU offload, `< 500 ms` local CPU latency, graceful EXIF-stripped JPEG & RGBA PNG/WEBP handling, and non-clobbering static `frontend/dist/` SPA serving. | Verified live warm timings (`59.96–80.83 ms` single-bird X-Ray + OOF ML, `145.94–167.88 ms` Watershed + per-bird ML, `71.48 ms` cold bundle load, `160.3 ms` / `103.1 KiB` total for all 10 cached thumbnails); verified `stripped_jpeg`, `rgba_png`, `webp`, and corrupt-bytes Pillow handling; documented exact Pydantic schemas, ASVS L1 upload/containment checks, and `/api` 404-preserving SPA catch-all. |
| **WEB-02** | Interactive Inference & "X-Ray" PDI Studio (`Estúdio Raio-X`): image upload + EXIF detection / missing-EXIF warning / manual lens-zoom override, 4-stage side-by-side PDI X-Ray viewer with lightbox zoom, morphometric/chromatic feature telemetry (including yellow-down vs. white-plumage & expandable 39-feature drawer), weight (g/kg) + band error band (`±MAE`, `p68`) + Cobb Avian 48 age stage, and client-side simulated flock average & uniformity (`CV%`) tracker. | Verified exact 49-key PDI output + 39-key model input subset, exact `band_error_intervals` schema and `<500g`/`500-1000g`/`>1000g` boundaries (`backend/app/ingestion/decoding.py:28-29,111-117`), `AgeStageResult` fields (`backend/app/modeling/age_stage.py:27-38`), portrait (`288×640`) vs. landscape (`640×288`) X-Ray stage framing (`object-contain` on `#0F172A`), and exact Data URI MIME mapping (JPEG stages 1/2/4, PNG stage 3). |
| **WEB-03** | One-click curated out-of-fold validation presets spanning young chick (`~180 g`), intermediate (`~550 g`), heavy (`~1,600 g`), white-on-white dorsal bucket (`>1,000 g`), and Dual-Lens pair (`541 g` Wide vs. Ultrawide), showing live Ground Truth vs. honest held-out Fold `K` Predicted weight, absolute error (g), and `%` error. | Verified all 6 `single_bird_presets` resolve via `manifest_clean.parquet` under `RAW_DATASET_DIR.resolve()` and reproduce `models/presets_catalog.json` `oof_predicted_g` to `<= 0.000048 g` when scored with `bundle["fold_pipelines"][fold]`. Verified `sb_dual_lens_wide` (`383.22 g`) vs. `sb_dual_lens_ultrawide` (`384.16 g`) = `0.94 g` (`0.24%`) live delta. |
| **WEB-04** | Interactive Marker-based Watershed multi-bird demo (`417`, `757`, `823`, `1520` + custom upload) with 4 visual stages, bounded tunable sliders, detected vs. expected counts, per-bird areas & illustrative weights + scene mean + permanent `group_sum_only` disclaimer; plus visually distinct conceptual v2 Fixed Ceiling-Camera Barn Roadmap (`BARN-01..03`, zero tracking/heatmap backend code per `PROIBIÇÃO 5`). | Reconciled authoritative Phase 2 ASVS V5 slider bounds & defaults (`backend/app/pdi/config.py:18-23,104-106` & `04-UI-SPEC.md:142-144`), verified all 4 quarantined presets resolve via `manifest_quarantine.parquet` (`145.94–167.88 ms`, `130–149 KiB` visuals, exact counts `2/3/4/2`), and mapped exact Stage 4 BGR basin palette to CSS swatches for instance table visual parity. |
| **WEB-05** | Validation Metrics & Forensic Audit Dashboard powered by `/api/metrics`: 5-fold CV KPIs + literature `R² > 0.9` context, filterable Recharts Actual vs. Predicted `ScatterChart` (1,738 images vs. 188 sessions, dashed `y=x` diagonal), 4-condition optical ablation `BarChart` (`C1–C4`), stratified error tables (band/lens/scene), 16-row model comparison, Top-12 feature importances, and `1,753 → 1,738` forensic curation funnel reconciling Mendeley `"45 days / 1,772 images"` vs. disk-audited `9 dates / ~7–29 days / 116–2,093 g` and `801–1,216 g` gap. | Inventoried 100% of JSON keys across `evaluation_report.json` (`624 KiB`), `dataset_audit_summary.json`, and `pdi_calibration_audit.json`; verified exact session aggregation semantics (`188` sessions, `multi_lens=127`, `multi_scene=186`, `multi_actual=0`); verified Recharts v3.11.1 breaking changes (deprecated `<Cell />`, numeric `ReferenceLine segment`, explicit height + `min-w-0` parent). |
</phase_requirements>

*(Requirement numbering note: `REQUIREMENTS.md:31-35` maps `WEB-02` = X-Ray Studio + Flock Tracker, `WEB-03` = Curated OOF Presets, `WEB-04` = Watershed + Future Vision Roadmap, and `WEB-05` = Metrics & Audit Dashboard, whereas `04-CONTEXT.md` splits the 4 UI workspaces across Tabs 1–4 (`Estúdio Raio-X`, `Separação Watershed`, `Auditoria & Métricas`, `Visão Futura (v2)`). Every functional requirement in both documents is 100% covered by the 4-tab design locked in `04-UI-SPEC.md`.)*

---

## Summary

Phase 4 wires the already-validated Phase 1–3 Python artifacts directly into a `< 500 ms` local FastAPI serving layer (`backend/app/api/`) and a 4-tab React 19 + TypeScript 7 + Vite 8 + Tailwind CSS v4 + Recharts 3 interactive pitch studio (`frontend/`). Empirical read-only verification spikes confirm massive performance headroom and zero integration blockers: loading the immutable dual-bundle `models/weight_regressor.joblib` takes `71.48 ms` cold (`cv2.setNumThreads(2)`), warm single-bird PDI X-Ray + held-out fold LightGBM scoring runs in **`59.96–80.83 ms`** (`6–8×` below the `< 500 ms` SLA) with exact `<= 0.000048 g` parity against `models/presets_catalog.json`, warm Marker-based Watershed + per-instance LightGBM runs in **`145.94–167.88 ms`** across all 4 multi-bird presets, and pre-warming all 10 EXIF-transposed `320 px` JPEG thumbnails takes `160.3 ms` and consumes only `103.1 KiB` total RAM `[VERIFIED: live empirical spike]`.

Three subtle integration traps were caught during live research and must be reflected directly in Phase 4 plans:
1. **Watershed slider bounds reconciliation:** `04-CONTEXT.md` (`D-04`/`D-10`) drafted narrower bounds (`min_distance_rel` `[0.10, 0.80]` default `0.34`), which contradict Phase 2's locked ASVS V5 constants in `backend/app/pdi/config.py:18-23,104-106` (`h_rel` `[0.05, 0.80]` default `0.20`, `min_distance_rel` `[0.20, 2.50]` default `0.85`, `min_area_ratio` `[0.01, 0.45]` default `0.10`). Both FastAPI Pydantic schemas and React sliders MUST import/mirror `backend/app/pdi/config.py` verbatim (already aligned in approved `04-UI-SPEC.md:142-144`) so default requests and slider edges never trigger `ValueError`/`422`.
2. **Recharts v3 (`3.11.1`) + Biome v2 (`2.6.8`) + Tailwind CSS v4 (`4.3.5`) configuration pitfalls:** Recharts v3 deprecated `<Cell />` (use data-item `fill` attributes or `shape` instead), requires explicit height + `min-w-0` parent containers around `<ResponsiveContainer>`, and enforces strict numeric `type="number"` axes for `<ReferenceLine segment={...} />`. Biome v2 moved `organizeImports` under `assist.actions.source.organizeImports` and requires `"css": { "parser": { "tailwindDirectives": true } }` in `biome.json` so `@theme` blocks in `frontend/src/index.css` parse cleanly.
3. **Single-origin SPA fallback vs. `/api` 404s + Portrait/Landscape X-Ray framing:** FastAPI's optional `frontend/dist/index.html` catch-all route must explicitly guard `full_path == "api" or full_path.startswith("api/")` so unknown `/api/*` routes still return JSON `404`, and `XRayQuadViewer` must use `object-contain` on fixed-aspect `#0F172A` frames because EXIF-transposed dataset photos mix portrait (`288×640`) and landscape (`640×288`) dimensions.

**Primary recommendation:** Build Phase 4 in clean parallelizable layers — (1) Backend FastAPI package (`schemas`, `services`, `routers`, `main` + comprehensive `TestClient` contract/security/latency suite), (2) Frontend scaffold (`Vite 8` + `Tailwind v4 @theme` + `Biome v2` + typed `api/` client & hooks), (3) Tab 1 (`Estúdio Raio-X` + 6 OOF presets + Dual-Lens comparator + Flock simulator) & Tab 2 (`Separação Watershed` + sliders + instance table), and (4) Tab 3 (`Auditoria & Métricas` Recharts dashboard) + Tab 4 (`Visão Futura v2`) + single-origin `frontend/dist/` build verification & pitch runbook.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| **Model bundle & audit JSON lifecycle (`lifespan`)** | API / Backend | — | Heavy `3.18 MiB` joblib bundle + 4 JSON audit files + 10 preset paths/thumbnails must load once at server boot with `PDIConfig` SHA-256 lock verification (`D-01`). |
| **Opaque `preset_id` → `Path` containment & thumbnails** | API / Backend | Browser (HTTP Cache) | Only backend reads `manifest_*.parquet` and validates non-symlink containment under `RAW_DATASET_DIR.resolve()`; browser never sees `rel_path` or `.jpg` names (`PROIBIÇÃO 4`, ASVS V4). |
| **Upload size/format/decompression-bomb validation** | API / Backend | Browser (UX pre-check) | Server enforces authoritative `15 MiB` read cap (`413`), Pillow magic format allowlist `{JPEG, PNG, WEBP}` (`415`), and pixel/corruption checks (`400`); client mirrors MIME/15 MB check for instant UX feedback. |
| **Single-bird PDI + X-Ray + Dual-bundle OOF scoring** | API / Backend (Threadpool) | — | Runs verbatim Phase 2 `run_single_bird_pdi` + Phase 3 `predict_from_feature_dict` (`fold_pipelines[k]` for presets, `production_pipeline` for uploads) offloaded via `run_in_threadpool`. |
| **Marker-based Watershed + per-instance scoring** | API / Backend (Threadpool) | — | Runs isolated Phase 2 `run_watershed_pdi` with ASVS V5 bounded sliders + `production_pipeline` per-instance scoring, always attaching the `group_sum_only` disclaimer. |
| **X-Ray & Watershed 4-stage inspection & lightbox** | Browser / Client | API / Backend (Data URIs) | Server encodes compact `<= 640 px` JPEG/PNG data URIs (`62–225 KiB`); React owns 2×2 grid layout, keyboard-accessible fullscreen lightbox, and `AbortController` request cancellation. |
| **Simulated Flock Average & Uniformity (`CV%`) tracker** | Browser / Client | — | Stateless MVP (`D-08`/`WEB-02`): pure client-side accumulator (`useFlockBatch` + optional `sessionStorage`) computes `n`, mean, sample SD, and `CV%` with zero server database. |
| **Metrics scatter filtering & 188-session aggregation** | Browser / Client | API / Backend (`GET /api/metrics`) | `GET /api/metrics` serves cached `evaluation_report.json` + `dataset_audit_summary.json` + `pdi_calibration_audit.json` once; React filters `1,738` `oof_scatter_points` by band/lens/scene and aggregates to `188` sessions in `< 3 ms` via `useMemo`. |
| **Conceptual v2 Barn Roadmap (`WEB-04` / `PROIBIÇÃO 5`)** | Browser / Client | — | Pure static declarative documentation tab with zero backend endpoints, zero mock cameras, and zero fake telemetry. |
| **Single-origin static SPA delivery (`frontend/dist/`)** | API / Backend | Vite Dev Proxy (Dev) | Vite proxies `/api/*` during development; FastAPI serves `frontend/dist/assets` + guarded SPA fallback for 1-command pitch execution (`D-05`). |

---

## Standard Stack

### Backend (Already Installed & Locked in `pyproject.toml` / `uv.lock`)

Zero new PyPI packages are needed for Phase 4 `[VERIFIED: pyproject.toml:6-28 & live uv run python check]`:

| Library | Installed Version | Purpose | Verification Source |
|---------|-------------------|---------|---------------------|
| `python` (`uv`-managed) | `3.12.14` | Runtime pinned via `.python-version` | `[VERIFIED: uv run python --version]` |
| `fastapi` | `0.165.0` | Async REST framework, `lifespan`, Pydantic v2 integration | `[VERIFIED: importlib.metadata]` |
| `uvicorn[standard]` | `0.54.0` | Local ASGI server (`--host 127.0.0.1 --port 8000`) | `[VERIFIED: importlib.metadata]` |
| `pydantic` | `2.14.4` | Strict request/response schema validation | `[VERIFIED: importlib.metadata]` |
| `python-multipart` | `0.0.34` | `multipart/form-data` parsing for `UploadFile` + `Form` | `[VERIFIED: importlib.metadata]` |
| `opencv-python-headless` | `4.14.0.94` | Classical PDI, JPEG/PNG encoding, `cv2.setNumThreads(2)` | `[VERIFIED: importlib.metadata]` |
| `pillow` | `13.0.0` | EXIF-first image validation, orientation transpose, format magic check | `[VERIFIED: importlib.metadata]` |
| `scikit-image` / `scipy` / `lightgbm` / `scikit-learn` / `joblib` / `pandas` / `pyarrow` | `0.26.0` / `1.18.1` / `4.7.0` / `1.9.1` / `1.6.0` / `3.1.1` / `26.0.1` | Phase 1–3 shared PDI, Watershed, manifest, and dual-bundle model runtime | `[VERIFIED: importlib.metadata]` |
| `pytest` + `httpx` + `ruff` | `9.2.0` / `0.28.1` / `0.17.6` | FastAPI `TestClient` contract/security/latency tests & linting | `[VERIFIED: importlib.metadata]` |

### Frontend Core & Supporting (New `frontend/package.json`)

All versions and `peerDependencies` were verified live against the npm registry on **2026-11-24** under host Node `v24.21.0` / npm `11.19.0`:

| Library | Verified Version | Purpose | Peer Compatibility & Provenance |
|---------|------------------|---------|---------------------------------|
| `react` | `19.3.1` | SPA UI runtime | Official React (`engines.node >=0.10.0`) `[VERIFIED: npm registry]` |
| `react-dom` | `19.3.1` | Browser DOM renderer | `peer: react ^19.3.1` `[VERIFIED: npm registry]` |
| `react-is` | `19.3.1` | Mandatory React 19 peer for Recharts 3 | Matches `react@19.3.1` (`recharts` requires `react-is ^19.0.0`) `[VERIFIED: npm registry]` |
| `recharts` | `3.11.1` *(updated from 3.11.0)* | `ScatterChart`, `BarChart`, `ReferenceLine`, `ResponsiveContainer` | `peer: react/react-dom/react-is ^19.0.0, @types/react *`; `engines.node >=18` `[VERIFIED: npm registry]` *(see Legitimacy Audit note on `too-new` patch release)* |
| `lucide-react` | `1.79.0` *(updated from 1.77.0)* | Vector icon set (`Maximize2`, `X`, `ChevronLeft/Right/Down/Up`, `Trash2`, `Info`, etc.) | `peer: react ^19.0.0, @types/react *` `[VERIFIED: npm registry]` *(see Legitimacy Audit note)* |
| `clsx` | `2.1.1` | Conditional class builder | Zero peers `[VERIFIED: npm registry]` |
| `tailwind-merge` | `3.7.1` | Conflict-free Tailwind v4 utility class merging (`cn()`) | Zero peers `[VERIFIED: npm registry]` *(see Legitimacy Audit note)* |

### Frontend Dev & Build Tooling

| Library | Verified Version | Purpose | Peer Compatibility & Provenance |
|---------|------------------|---------|---------------------------------|
| `vite` | `8.3.8` | Rolldown bundler & dev proxy (`/api` → `127.0.0.1:8000`) | `engines.node ^20.19.0 \|\| >=22.12.0` (host `24.21.0`); `peer: @types/node >=22.12.0` `[VERIFIED: npm registry]` |
| `@vitejs/plugin-react` | `6.3.0` | React Fast Refresh & JSX transform for Vite 8 | `peer: vite ^8.0.0` `[VERIFIED: npm registry]` |
| `tailwindcss` | `4.3.5` | Tailwind CSS v4 utility engine (`@theme` tokens) | Zero peers `[VERIFIED: npm registry]` |
| `@tailwindcss/vite` | `4.3.5` | First-party Tailwind v4 Vite plugin (zero PostCSS/config JS) | `peer: vite ^5.2.0 \|\| ^6 \|\| ^7 \|\| ^8` `[VERIFIED: npm registry]` |
| `typescript` | `7.0.2` | Native Go `tsc` type-checker (`tsc --noEmit`) | `engines.node >=16.20.0`; CLI binary is `tsc` `[VERIFIED: npm registry]` `[CITED: typescript 7.0 docs]` |
| `@biomejs/biome` | `2.6.8` | Linter + formatter (`biome check`) replacing ESLint | Zero peer conflict with TS 7 (`typescript-eslint` breaks on TS 7) `[VERIFIED: npm registry]` |
| `@types/react` | `19.3.0` | React 19 TypeScript definitions | `[VERIFIED: npm registry]` |
| `@types/react-dom` | `19.3.0` | React DOM 19 TypeScript definitions | `peer: @types/react ^19.3.0` `[VERIFIED: npm registry]` |
| `@types/node` | `26.6.6` | Node types for `vite.config.ts` | Satisfies `vite` (`>=22.12.0`) & `vitest` (`>=24.0.0`) `[VERIFIED: npm registry]` |
| `vitest` | `5.1.1` | Lightweight pure-logic unit tests (flock math, session scatter aggregation, formatters, API client) | `peer: vite ^6.4.0 \|\| ^7.0.0 \|\| ^8.0.0`, `@types/node >=24.0.0`; `engines.node ^22.12.0 \|\| ^24.0.0` `[VERIFIED: npm registry]` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `vitest` (pure logic unit tests) | `@testing-library/react` + `jsdom` | Adds ~40+ transitive DOM polyfill packages for a 4-day sprint; testing pure TS modules (`flock.ts`, `scatter.ts`, `format.ts`, `client.ts`) with `vitest` + full `tsc --noEmit` + `biome check` + `vite build` + FastAPI `pytest` keeps install fast and deterministic. |
| Native `fetch` + `AbortController` | Axios / TanStack Query | Locked out by `STACK.md` & `D-06`/`D-10`; 6 local synchronous endpoints need zero query-cache framework. |
| `@biomejs/biome@2.6.8` | ESLint + `typescript-eslint` | Verified incompatibility: `typescript-eslint@8.73.0` caps `typescript < 6.1.0` (`GEMINI.md:116`). |

### Installation (Frontend)

> **Shell Note:** Always prepend `$HOME/.nvm/versions/node/v24.21.0/bin` to `PATH` in non-interactive shells before running `node`/`npm`/`npx`.

```bash
export PATH="$HOME/.nvm/versions/node/v24.21.0/bin:$PATH"
cd frontend
npm init -y
npm pkg set type="module" private=true

npm install \
  react@^19.3.1 \
  react-dom@^19.3.1 \
  react-is@^19.3.1 \
  recharts@^3.11.1 \
  lucide-react@^1.79.0 \
  clsx@^2.1.1 \
  tailwind-merge@^3.7.1

npm install -D \
  vite@^8.3.8 \
  @vitejs/plugin-react@^6.3.0 \
  tailwindcss@^4.3.5 \
  @tailwindcss/vite@^4.3.5 \
  typescript@^7.0.2 \
  @biomejs/biome@^2.6.8 \
  @types/react@^19.3.0 \
  @types/react-dom@^19.3.0 \
  @types/node@^26.6.6 \
  vitest@^5.1.1
```

---

## Package Legitimacy Audit

Executed via `gsd-tools.cjs query package-legitimacy check --ecosystem npm`, cross-checked against live `npm view <pkg> time.created scripts.postinstall` on **2026-11-24**. Zero new PyPI packages are installed in Phase 4 (all Python packages were audited and locked in Phases 1–3).

> **Important Automated-Gate Context:** Every `[SUS]` verdict below was triggered **solely** by `"reasons": ["too-new"]` because the seam checks the latest *patch/minor release* `publishedAt` timestamp (October–November 2026) rather than initial package creation (`time.created`). Every single one of those packages is a 2–11 year old mainstream industry library (`30M–671M` weekly downloads), backed by its official GitHub repository, with `deprecated: false` and **`postinstall: null`** (zero install-time scripts).

| Package | Registry | Package Age (`time.created`) | Latest Publish (`publishedAt`) | Weekly Downloads | Source Repo | `postinstall` | Seam Verdict | Disposition |
|---------|----------|------------------------------|--------------------------------|------------------|-------------|---------------|--------------|-------------|
| `react` | npm | 15.1 yrs (`2011-10-26`) | `2026-10-14` | 275.7M/wk | `github.com/react/react` | `null` | `[OK]` | Approved |
| `react-dom` | npm | 12.6 yrs (`2014-05-06`) | `2026-10-14` | 260.2M/wk | `github.com/react/react` | `null` | `[OK]` | Approved |
| `react-is` | npm | 9.1 yrs (`2017-10-19`) | `2026-10-14` | 500.4M/wk | `github.com/react/react` | `null` | `[OK]` | Approved |
| `clsx` | npm | 7.9 yrs (`2018-12-24`) | `2024-04-23` | 179.1M/wk | `github.com/lukeed/clsx` | `null` | `[OK]` | Approved |
| `typescript` | npm | 14.1 yrs (`2012-10-01`) | `2026-07-08` | 451.0M/wk | `github.com/microsoft/TypeScript` | `null` | `[OK]` | Approved |
| `@types/react` | npm | 10.5 yrs (`2016-05-17`) | `2026-09-09` | 248.4M/wk | `github.com/DefinitelyTyped/DefinitelyTyped` | `null` | `[OK]` | Approved |
| `@types/react-dom` | npm | 10.5 yrs (`2016-05-17`) | `2026-09-09` | 210.8M/wk | `github.com/DefinitelyTyped/DefinitelyTyped` | `null` | `[OK]` | Approved |
| `recharts` `[WARNING: flagged as suspicious — verify before using.]` | npm | 11.3 yrs (`2015-08-07`) | `2026-11-23` (`3.11.1`) | 71.4M/wk | `github.com/recharts/recharts` | `null` | `[SUS]` (`too-new` patch) | Pre-locked in `STACK.md` / `GEMINI.md`; verify version lock in `package-lock.json` |
| `lucide-react` `[WARNING: flagged as suspicious — verify before using.]` | npm | 6.1 yrs (`2020-10-19`) | `2026-11-24` (`1.79.0`) | 143.9M/wk | `github.com/lucide-icons/lucide` | `null` | `[SUS]` (`too-new` patch) | Pre-locked in `STACK.md` / `04-UI-SPEC.md`; verify version lock |
| `tailwind-merge` `[WARNING: flagged as suspicious — verify before using.]` | npm | 5.4 yrs (`2021-07-18`) | `2026-11-19` (`3.7.1`) | 117.7M/wk | `github.com/dcastil/tailwind-merge` | `null` | `[SUS]` (`too-new` patch) | Pre-locked in `STACK.md`; verify version lock |
| `vite` `[WARNING: flagged as suspicious — verify before using.]` | npm | 6.6 yrs (`2020-04-21`) | `2026-11-11` (`8.3.8`) | 287.9M/wk | `github.com/vitejs/vite` | `null` | `[SUS]` (`too-new` patch) | Pre-locked in `STACK.md` / `GEMINI.md`; verify version lock |
| `@vitejs/plugin-react` `[WARNING: flagged as suspicious — verify before using.]` | npm | 5.2 yrs (`2021-09-20`) | `2026-11-09` (`6.3.0`) | 142.0M/wk | `github.com/vitejs/vite-plugin-react` | `null` | `[SUS]` (`too-new` patch) | Pre-locked in `STACK.md`; verify version lock |
| `tailwindcss` `[WARNING: flagged as suspicious — verify before using.]` | npm | 9.1 yrs (`2017-10-06`) | `2026-11-06` (`4.3.5`) | 190.1M/wk | `github.com/tailwindlabs/tailwindcss` | `null` | `[SUS]` (`too-new` patch) | Pre-locked in `STACK.md` / `GEMINI.md`; verify version lock |
| `@tailwindcss/vite` `[WARNING: flagged as suspicious — verify before using.]` | npm | 2.8 yrs (`2024-02-02`) | `2026-11-06` (`4.3.5`) | 70.4M/wk | `github.com/tailwindlabs/tailwindcss` | `null` | `[SUS]` (`too-new` patch) | Pre-locked in `STACK.md`; verify version lock |
| `@biomejs/biome` `[WARNING: flagged as suspicious — verify before using.]` | npm | 3.3 yrs (`2023-08-17`) | `2026-11-15` (`2.6.8`) | 30.3M/wk | `github.com/biomejs/biome` | `null` | `[SUS]` (`too-new` patch) | Pre-locked in `STACK.md`; verify version lock |
| `@types/node` `[WARNING: flagged as suspicious — verify before using.]` | npm | 10.5 yrs (`2016-05-17`) | `2026-10-28` (`26.6.6`) | 671.5M/wk | `github.com/DefinitelyTyped/DefinitelyTyped` | `null` | `[SUS]` (`too-new` patch) | Pre-locked in `STACK.md`; verify version lock |
| `vitest` `[WARNING: flagged as suspicious — verify before using.]` | npm | 5.0 yrs (`2021-12-03`) | `2026-11-18` (`5.1.1`) | 194.0M/wk | `github.com/vitest-dev/vitest` | `null` | `[SUS]` (`too-new` patch) | Optional dev-only unit test runner; verify version lock |

**Packages removed due to `[SLOP]` verdict:** none
**Packages flagged as suspicious `[SUS]`:** `recharts`, `lucide-react`, `tailwind-merge`, `vite`, `@vitejs/plugin-react`, `tailwindcss`, `@tailwindcss/vite`, `@biomejs/biome`, `@types/node`, `vitest` — all flagged strictly due to recent October/November 2026 release timestamps (`too-new` false positive on 2.8–11.3 yr established libraries with 30M–671M weekly downloads, official repos, and `postinstall: null`). Because the first nine are already explicitly locked in `GEMINI.md` / `STACK.md` / `04-CONTEXT.md` / `04-UI-SPEC.md` and `config.json` uses `human_verify_mode: "end-of-phase"` + `mode: "yolo"`, the planner should pin exact verified versions in `frontend/package.json` and run `npm install --ignore-scripts` (or verify zero unexpected lifecycle scripts in `package-lock.json`; note `@biomejs/biome`, `tailwindcss`, and `vite`/Rolldown use optional native binary packages without `postinstall` hooks — verify `npm run build` works with standard `npm install` after inspecting lockfile).

---

## Live Empirical Verification Findings (Phases 1–3 Integration Ground Truth)

### 1. Authoritative Upstream Constants & Schema Reconciliation

Every discrete value below was read directly from source files in this session:

- **Canonical Paths (`backend/app/config.py:12-24` `[VERIFIED: backend/app/config.py:12-24]`):**
  ```python
  REPO_ROOT: Path = Path(__file__).resolve().parents[2]
  DATA_DIR: Path = REPO_ROOT / "data"
  RAW_DATASET_DIR: Path = DATA_DIR / "raw" / "Broiler weights of Cobb Avian"
  PROCESSED_DIR: Path = DATA_DIR / "processed"
  MODELS_DIR: Path = REPO_ROOT / "models"
  CLEAN_MANIFEST_PATH: Path = PROCESSED_DIR / "manifest_clean.parquet"
  QUARANTINE_MANIFEST_PATH: Path = PROCESSED_DIR / "manifest_quarantine.parquet"
  AUDIT_SUMMARY_PATH: Path = MODELS_DIR / "dataset_audit_summary.json"
  FEATURES_CLEAN_PATH: Path = PROCESSED_DIR / "features_clean.parquet"
  PDI_CALIBRATION_AUDIT_PATH: Path = MODELS_DIR / "pdi_calibration_audit.json"
  MODEL_BUNDLE_PATH: Path = MODELS_DIR / "weight_regressor.joblib"
  EVALUATION_REPORT_PATH: Path = MODELS_DIR / "evaluation_report.json"
  PRESETS_CATALOG_PATH: Path = MODELS_DIR / "presets_catalog.json"
  ```
  Add `FRONTEND_DIST_DIR: Path = REPO_ROOT / "frontend" / "dist"` in `backend/app/config.py` (or inside `backend/app/api/main.py` anchored to `REPO_ROOT`).

- **Watershed Bounds & Defaults Reconciliation (`backend/app/pdi/config.py:18-23,104-106` `[VERIFIED: backend/app/pdi/config.py:18-23,104-106]` & `backend/app/pdi/watershed.py:58-63` `[VERIFIED: backend/app/pdi/watershed.py:58-63]`):**
  ```python
  WATERSHED_MIN_H_REL: float = 0.05
  WATERSHED_MAX_H_REL: float = 0.80
  WATERSHED_MIN_DISTANCE_REL: float = 0.20
  WATERSHED_MAX_DISTANCE_REL: float = 2.50
  WATERSHED_MIN_AREA_RATIO: float = 0.01
  WATERSHED_MAX_AREA_RATIO: float = 0.45
  # Defaults inside PDIConfig:
  watershed_h_rel: float = 0.20
  watershed_min_distance_rel: float = 0.85
  watershed_min_area_ratio: float = 0.10
  ```
  **CRITICAL PLANNER ACTION:** `04-CONTEXT.md` `D-04`/`D-10` inadvertently cited older draft bounds (`min_distance_rel` `[0.10, 0.80]` default `0.34`), which would BOTH pass invalid `< 0.20` values to `run_watershed_pdi` (`ValueError`) AND reject the actual Phase 2 default `0.85`! Approved `04-UI-SPEC.md:142-144` already locked the true `backend/app/pdi/config.py` bounds (`h_rel: 0.05..0.80` default `0.20`, `min_distance_rel: 0.20..2.50` default `0.85`, `min_area_ratio: 0.01..0.45` default `0.10`). Import `WATERSHED_MIN_*` / `WATERSHED_MAX_*` directly into FastAPI Pydantic validators so backend and frontend can never diverge.

- **Optical Override Bounds (`backend/app/pdi/io_exif.py:29-35` `[VERIFIED: backend/app/pdi/io_exif.py:29-35]`):**
  ```python
  MIN_VALID_ZOOM_MULTIPLE = 0.1
  MAX_VALID_ZOOM_MULTIPLE = 20.0
  MIN_VALID_FOCAL_LENGTH_MM = 0.5
  MAX_VALID_FOCAL_LENGTH_MM = 200.0
  WIDE_FOCAL_THRESHOLD_MM = 3.0
  DEFAULT_FOCAL_LENGTH_MM = 4.25
  DEFAULT_ZOOM_MULTIPLE = 1.0
  ```

- **Weight Band Classification (`backend/app/ingestion/decoding.py:28-29,111-117` `[VERIFIED: backend/app/ingestion/decoding.py:28-29,111-117]`):**
  ```python
  BAND_LIGHT_MAX_EXCLUSIVE_G: int = 500
  BAND_MEDIUM_MAX_INCLUSIVE_G: int = 1000
  # < 500 -> "<500g", <= 1000 -> "500-1000g", > 1000 -> ">1000g"
  ```
  Map predicted float weight `pred_g` to `bundle["band_error_intervals"][band]` using `< 500.0` → `"<500g"`, `<= 1000.0` → `"500-1000g"`, else `">1000g"`.

- **Cobb Avian 48 Age Stage Contract (`backend/app/modeling/age_stage.py:18-38,48-64` `[VERIFIED: backend/app/modeling/age_stage.py:18-38,48-64]`):**
  - `VALID_TRAINING_WEIGHT_MIN_G = 116`, `VALID_TRAINING_WEIGHT_MAX_G = 2093`, `VALID_AGE_DAYS_RANGE = "~7-29 days"`
  - `stage_code` ∈ `{"inicial_pintinho", "crescimento_inicial", "crescimento_intermediario", "terminacao_pre_abate"}`
  - `stage_label` ∈ `{"Inicial / Pintinho", "Crescimento Inicial", "Crescimento Intermediário", "Terminação / Pré-Abate"}`
  - `estimated_age_days_range` ∈ `{"~7-13 dias", "~14-19 dias", "~20-24 dias", "~25-29 dias"}`
  - `out_of_training_range: bool` (`True` when `< 116` or `> 2093`)

- **Model Bundle Contract (`backend/app/modeling/artifacts.py:61-64,88-105` `[VERIFIED: backend/app/modeling/artifacts.py:61-64,88-105]`):**
  - `MODEL_BUNDLE_SCHEMA_VERSION = "1.0.0"`, `MODEL_VERSION = "avianscale-cv-v1.0.0"`
  - Exact 14 keys verified live: `band_error_intervals`, `duan_smearing_enabled` (`False`), `feature_names` (`39`-tuple), `fold_pipelines` (`{0, 1, 2, 3, 4}`), `model_version`, `pdi_config`, `pdi_config_hash` (`"43a3cfaa3208019e"`), `production_pipeline`, `schema_version`, `valid_age_days_range`, `valid_weight_range_g`, `weighting_mode` (`"session_band_capped_5x"`), `winning_ablation_condition` (`"C2_physics_normalized_strict"`), `winning_model_name` (`"LightGBMLog"`).
  - Exact `band_error_intervals` verified live from `models/weight_regressor.joblib`:
    - `"<500g"`: `n=1223, mae_g=93.1486, mape_pct=39.6349, median_abs_err_g=66.2551, p68_abs_err_g=107.5501, p90_abs_err_g=205.879, mbe_g=57.3291`
    - `"500-1000g"`: `n=333, mae_g=160.7579, mape_pct=26.6674, median_abs_err_g=156.4946, p68_abs_err_g=207.0186, p90_abs_err_g=296.9483, mbe_g=-102.054`
    - `">1000g"`: `n=182, mae_g=589.7691, mape_pct=39.0364, median_abs_err_g=566.0277, p68_abs_err_g=762.9382, p90_abs_err_g=1087.2483, mbe_g=-578.5335`

- **Watershed Stage 4 Basin Palette (`backend/app/pdi/watershed.py:76-85` `[VERIFIED: backend/app/pdi/watershed.py:76-85]`):**
  Because OpenCV stores BGR `(B, G, R)`, converting `(B, G, R) -> #RRGGBB` gives the exact matching CSS swatch hex colors for the per-bird instance table (`(instance_id - 1) % 8`):
  1. `#2DB9D2` (Teal/cyan, BGR `(210, 185, 45)`)
  2. `#F5AF37` (Warm amber, BGR `(55, 175, 245)`)
  3. `#5582EB` (Sky blue, BGR `(235, 130, 85)`)
  4. `#F05F82` (Coral rose, BGR `(130, 95, 240)`)
  5. `#78E169` (Emerald green, BGR `(105, 225, 120)`)
  6. `#C873E1` (Orchid violet, BGR `(225, 115, 200)`)
  7. `#EBDC46` (Golden yellow, BGR `(70, 220, 235)`)
  8. `#FF9BB4` (Soft salmon, BGR `(180, 155, 255)`)

### 2. Live Preset Resolution, OOF Parity & Latency Benchmarks

Measured live on this workstation (`uv run python`, `cv2.setNumThreads(2)`) `[VERIFIED: live empirical spike]`:

- **Manifest schemas:** Both `data/processed/manifest_clean.parquet` (`1,738` rows) and `data/processed/manifest_quarantine.parquet` (`15` rows) share identical 24 columns including `image_id` and `rel_path`.
- **Startup cold load:** `load_model_bundle` = **`71.48 ms`**; pre-warming all 10 `320 px` (`Q=78`) JPEG thumbnails = **`160.3 ms`**, total RAM footprint = **`103.1 KiB`**.
- **Single-bird curated OOF presets (all 6 resolve safely inside `RAW_DATASET_DIR.resolve()` as non-symlinks):**

| `preset_id` | Fold | GT (g) | Live OOF Pred (g) | Catalog Parity Δ | PDI (ms) | ML (ms) | Total (ms) | X-Ray B64 | Display W×H | Stage | LCD Excl px |
|-------------|------|--------|-------------------|------------------|----------|---------|------------|-----------|-------------|-------|-------------|
| `sb_young_chick` | 4 | 180 | 249.7296 | `0.000020 g` | 57.62 | 15.90 | **73.52** | 185.6 KB | `288×640` | `inicial_pintinho` | 0 |
| `sb_intermediate_grower` *(default boot)* | 2 | 576 | 424.0463 | `0.000039 g` | 78.68 | 2.16 | **80.83** | 83.0 KB | `640×288` | `crescimento_inicial` | 0 |
| `sb_heavy_broiler` | 1 | 1555 | 998.3282 | `0.000025 g` | 65.54 | 2.29 | **67.83** | 168.2 KB | `640×288` | `crescimento_intermediario` | 1084 |
| `sb_hard_white_bucket` | 3 | 1986 | 968.1782 | `0.000048 g` | 57.70 | 2.27 | **59.96** | 61.9 KB | `288×640` | `crescimento_intermediario` | 0 |
| `sb_dual_lens_wide` | 0 | 541 | 383.2180 | `0.000020 g` | 75.02 | 2.98 | **78.00** | 221.7 KB | `288×640` | `crescimento_inicial` | 0 |
| `sb_dual_lens_ultrawide` | 0 | 541 | 384.1577 | `0.000048 g` | 58.78 | 2.23 | **61.01** | 224.5 KB | `640×288` | `crescimento_inicial` | 0 |

- **Watershed curated group presets (all 4 resolve safely via `manifest_quarantine.parquet`):**

| `preset_id` | Folder | Expected Birds | Default Detected | Group Sum (g) | Est. Sum (g) | Est. Mean (g) | PDI (ms) | ML (ms) | Total (ms) | Visuals B64 | Red Feeder Excl px |
|-------------|--------|----------------|------------------|---------------|--------------|---------------|----------|---------|------------|-------------|--------------------|
| `ws_417` *(default boot)* | `417` | 2 | 2 | 417 | 496.40 | 248.20 | 163.87 | 4.01 | **167.88** | 148.7 KB | 25,434 |
| `ws_757` | `757` | 2 | 3 | 757 | 1015.29 | 338.43 | 151.89 | 6.30 | **158.20** | 133.5 KB | 89 |
| `ws_823` | `823` | 4 | 4 | 823 | 869.52 | 217.38 | 137.33 | 8.60 | **145.94** | 148.6 KB | 0 |
| `ws_1520` | `1520` | 2 | 2 | 1520 | 647.83 | 323.91 | 153.66 | 3.95 | **157.61** | 130.0 KB | 0 |

- **Synthetic Upload Edge Cases Verified Live (`Pillow==13.0.0`):**
  - EXIF-stripped JPEG: `exif_optical_present=False`, defaults `4.25 mm / 1.0×`, `57.5 ms`.
  - RGBA 4-channel PNG: cleanly converted to RGB via `ImageOps.exif_transpose(im).convert("RGB")`, `15.7 ms`.
  - WEBP: cleanly decoded, `17.3 ms`.
  - Corrupt non-image bytes: raises `PIL.UnidentifiedImageError` cleanly.
  - Default `Image.MAX_IMAGE_PIXELS = 89_478_485` (`~89.5 MP`); add explicit header guard `im.width * im.height > 40_000_000` (40 MP ceiling — >5× Redmi 10's 7.51 MP) to catch decompression bombs early before pixel decode.

### 3. Complete Audit JSON Schema Inventory (`GET /api/metrics` & `GET /api/presets`)

Every key below was inspected directly in `models/*.json` `[VERIFIED: models/evaluation_report.json, models/dataset_audit_summary.json, models/pdi_calibration_audit.json, models/presets_catalog.json]`:

1. **`models/evaluation_report.json` (`624,113 bytes`):**
   - Top-level keys (11): `schema_version`, `pdi_config_hash`, `dataset_summary`, `literature_benchmark`, `selection_criteria`, `mechanics_ablation`, `allometric_log_ablation`, `model_comparison` (`16` rows), `optical_ablation`, `winning_model`, `oof_scatter_points` (`1,738` rows).
   - `optical_ablation` keys (6): `reference_model_name` (`"LightGBMLog"`), `conditions` (`C1_uncalibrated_pdi`, `C2_physics_normalized_strict`, `C3_physics_plus_exif_meta`, `C4_naive_double_count_fxz`), `deltas_vs_c1`, `dual_lens_consistency`, `counterfactual_lens_flip`, `family_means_by_condition`.
   - `winning_model` keys (19): `model_name`, `ablation_condition`, `n_features`, `feature_names`, `weighting_mode`, `duan_smearing_enabled`, `global` (`mae_g`, `mape_pct`, `mbe_g`, `median_abs_err_g`, `n`, `p68_abs_err_g`, `p90_abs_err_g`, `r2`, `rmse_g`, `session_averaged_r2`), `session_level` (`n_sessions=188`, `mae_g=145.296`, `mape_pct=31.2858`, `rmse_g=234.4931`, `session_averaged_r2=0.6905`, ...), `balanced_band_mape_pct` (`35.1129`), `per_fold` (`5` items), `by_weight_band` (`<500g`, `500-1000g`, `>1000g`), `by_lens` (`wide`, `ultrawide`), `by_scene` (`bucket_dorsal`, `scale_or_litter`), `by_band_and_scene` (`6` keys), `by_band_and_lens` (`6` keys), `band_error_intervals`, `dual_lens_consistency`, `counterfactual_lens_flip`, `feature_importances` (`39` items: `{feature, importance_pct, raw_importance}`).
   - `oof_scatter_points[i]` exact 10 keys: **`actual_g`** *(NOTE: NOT `weight_g`!)*, `predicted_g`, `abs_err_g`, `pct_err`, `fold`, `image_id`, `session_id`, `lens_class` (`"wide" | "ultrawide"`), `scene_type` (`"bucket_dorsal" | "scale_or_litter"`), `weight_band` (`"<500g" | "500-1000g" | ">1000g"`).
   - **Session Scatter Aggregation Semantics (verified live across all `1,738` points → `188` sessions):**
     - `actual_g`, `weight_band`, and `fold` are 100% invariant within every `session_id` (`multi_actual=0, multi_band=0, multi_fold=0`).
     - `127/188` sessions contain mixed lenses (`wide` + `ultrawide`), and `186/188` contain mixed scenes (`bucket_dorsal` + `scale_or_litter`).
     - **Prescriptive client rule:** Always apply active `weight_band`, `lens_class`, and `scene_type` filters to `oof_scatter_points` **first**, then when `viewMode === "session"` group the filtered points by `session_id`, averaging `predicted_g` (`mean_predicted_g`), computing `abs_err_g = |mean_predicted_g - actual_g|` and `pct_err = abs_err_g / actual_g * 100`, and tagging each session as `wide` (all filtered frames wide), `ultrawide` (all filtered frames ultrawide), or `mixed` (both lenses present). With no filters active, this reproduces the exact `188` sessions and `R² = 0.6905` reported in `winning_model.session_level`!
2. **`models/dataset_audit_summary.json`:**
   - Top-level keys (6): `schema_version`, `dataset_identity`, `curation_funnel` (`raw_images: 1753`, `quarantined_total: 15`, `quarantined_multi_bird.count: 13`, `quarantined_severe_mislabel.count: 2`, `lcd_corrected_images.count: 6`, `clean_images: 1738`, `clean_unique_weights: 160`, `clean_sessions: 188`), `mendeley_vs_disk_reconciliation` (`published_claims`: `"daily for 45 days"`, `1772`; `disk_audit`: `9` dates, `23` calendar span days, `~7-29 days`, `116–2093 g`, `missing_vs_mendeley_images: 19`), `distributions` (`coverage_gaps.temporal_hiatus`: `2022-04-03` → `2022-04-11` / `8` days; `coverage_gaps.weight_sparse_interval`: `801..1216 g`, `415 g` span, `1` weight `[1009]`, `6` images; `lenses`, `weight_bands`, `timeline_by_date`), `encoding_and_exif_recovery`, `split_audit` (`cross_fold_session_overlap: 0`, `5` fold summaries).
3. **`models/pdi_calibration_audit.json`:**
   - Top-level keys (5): `schema_version`, `pdi_config_hash`, `pdi_config`, `dataset_extraction_summary` (`n_clean_images: 1738`, `n_bucket_dorsal: 377`, `n_scale_or_litter: 1361`, `primary_fallback_count: 4`, `primary_fallback_pct: 0.2301`, `lcd_suppression_triggered_count: 786`, `lcd_suppression_bucket_false_positives: 0`), `paired_dual_lens_calibration` (`winning_formula: "area_px / (zoom_multiple ** 2)"`, `rejected_double_count_formula`, `scopes.{all_mixed_lens_sessions, bucket_only_pairs, same_scene_pairs, scale_or_litter_pairs}`).
4. **`models/presets_catalog.json`:**
   - Top-level keys (6): `schema_version`, `pdi_config_hash`, `winning_model_name`, `winning_ablation_condition`, `single_bird_presets` (`6` entries), `watershed_presets` (`4` entries).
   - **API enhancement for `GET /api/presets`:** Enrich each returned preset object with `thumbnail_url: f"/api/presets/{preset_id}/thumbnail"` (still zero `rel_path` / `.jpg` leaks!).

---

## Architecture Patterns

### System Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────┐
│          Browser SPA (React 19 + TS 7 + Tailwind v4 + Recharts 3)       │
│  Hash-synced 4 tabs: #estudio | #watershed | #metricas | #visao-futura  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────────┐  │
│  │ Estúdio      │ │ Separação    │ │ Auditoria &  │ │ Visão Futura   │  │
│  │ Raio-X +     │ │ Watershed +  │ │ Métricas     │ │ v2 (Static     │  │
│  │ Lote Local   │ │ 3 Sliders    │ │ (Recharts 3) │ │ Conceptual)    │  │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └────────────────┘  │
└─────────┼────────────────┼────────────────┼─────────────────────────────┘
          │ Vite proxy /api (dev) OR same-origin :8000 (pitch single-cmd)
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                FastAPI ASGI App (backend/app/api/main.py)               │
│  lifespan: cv2.setNumThreads(2), load_model_bundle(hash lock),          │
│            load 4 audit JSONs, build opaque preset_id -> safe Path map, │
│            pre-warm 10 JPEG thumbnails (103 KiB RAM)                    │
│  ┌──────────────┐ ┌────────────────────┐ ┌───────────────────────────┐  │
│  │ Sync/Cached  │ │ Async Upload/Preset│ │ Optional Static Mount     │  │
│  │ /api/health  │ │ /api/predict       │ │ /assets -> frontend/dist  │  │
│  │ /api/presets │ │ /api/watershed-demo│ │ /{path} -> index.html     │  │
│  │ /api/metrics │ │ /api/presets/{id}/ │ │   (if api/* -> JSON 404!) │  │
│  │              │ │   thumbnail        │ │                           │  │
│  └──────────────┘ └─────────┬──────────┘ └───────────────────────────┘  │
│                             │ await run_in_threadpool(...)              │
│                             ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Shared Classical PDI Engine + Dual-Bundle Regressor (Phases 2–3) │   │
│  │ • Upload  -> Pillow format/bomb check -> PDI -> production_pipe  │   │
│  │ • Preset  -> Allowlist Path -> PDI -> fold_pipelines[fold] (OOF) │   │
│  │ • Watershed -> PDI Watershed -> production_pipe per instance     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure

```text
backend/app/api/
├── __init__.py
├── main.py                  # create_app(), lifespan, localhost CORS, exception handlers, static SPA mount
├── schemas.py               # Strict Pydantic v2 response/sub-model contracts (zero rel_path)
├── dependencies.py          # AppState dataclass & typed FastAPI Depends accessors
├── services/
│   ├── __init__.py
│   ├── upload_guard.py      # Bounded async UploadFile reader (15 MiB) + Pillow magic/bomb validator
│   ├── preset_service.py    # Manifest join, non-symlink RAW_DATASET_DIR containment, thumbnail RAM cache
│   ├── inference_service.py # Single-bird PDI + dual-bundle OOF vs production routing + age_stage + band CI
│   ├── watershed_service.py # Isolated Watershed PDI + per-instance scoring + mandatory disclaimer
│   └── metrics_service.py   # Aggregated read-only payload from 3 audit JSONs
└── routers/
    ├── __init__.py
    ├── health.py            # GET  /api/health
    ├── presets.py           # GET  /api/presets, GET /api/presets/{preset_id}/thumbnail
    ├── predict.py           # POST /api/predict
    ├── watershed.py         # POST /api/watershed-demo
    └── metrics.py           # GET  /api/metrics

backend/tests/
├── test_api_health_presets_metrics.py   # Lifespan startup, health, presets, thumbnail cache, metrics, PROIBIÇÃO 4 scan
├── test_api_predict.py                  # OOF preset parity, upload JPEG/PNG/WEBP, EXIF fallback, override, 400/413/415/422/404, <500ms
└── test_api_watershed_and_spa.py        # Watershed presets, slider bounds 422, custom upload, disclaimer, SPA fallback vs /api/* 404

frontend/
├── package.json             # Scripts: dev, build, preview, typecheck, lint, format, test
├── vite.config.ts           # @vitejs/plugin-react + @tailwindcss/vite + /api proxy to 127.0.0.1:8000
├── tsconfig.json            # Project references -> tsconfig.app.json + tsconfig.node.json
├── tsconfig.app.json        # Strict ES2022, jsx: react-jsx, moduleResolution: bundler, types: ["vite/client"]
├── tsconfig.node.json       # Strict node config for vite.config.ts, types: ["node"]
├── biome.json               # Biome v2.6.8 schema, css.parser.tailwindDirectives=true, assist.organizeImports
├── index.html               # lang="pt-BR", Inter + JetBrains Mono (system fallback safe offline!)
└── src/
    ├── main.tsx
    ├── index.css            # @import "tailwindcss"; + @theme tokens locked to 04-UI-SPEC.md
    ├── App.tsx              # 4-tab hash-synced shell + global Health header + tab mounting
    ├── api/
    │   ├── types.ts         # Exact TypeScript mirrors of FastAPI Pydantic v2 schemas
    │   └── client.ts        # Native fetch + AbortSignal + structured ApiError (pt-BR messages)
    ├── utils/
    │   ├── cn.ts            # clsx + tailwind-merge
    │   ├── format.ts        # pt-BR number formatting (tabular-nums), weight g/kg, % deltas
    │   ├── flock.ts         # Pure flock accumulator math (mean, sample SD, CV%)
    │   └── scatter.ts       # Pure filter-before-aggregate 1,738 -> 188 session scatter transform
    ├── hooks/
    │   ├── useHealth.ts
    │   ├── usePresets.ts
    │   ├── useMetrics.ts
    │   └── useFlockBatch.ts
    └── components/
        ├── layout/          # AppHeader, StageLightboxModal, InlineErrorBanner, StatusPill
        ├── studio/          # PresetsStrip, DualLensComparisonModal/Strip, UploadDropzone, XRayQuadViewer,
        │                    # PredictionSummaryCard, OpticalCalibrationCard, TelemetryPanel, FlockBatchCard
        ├── watershed/       # WatershedPresetStrip, WatershedStageGrid, WatershedSlidersCard, WatershedInstanceTable
        ├── metrics/         # ExecutiveKpiStrip, ActualVsPredictedScatter, OpticalAblationChart,
        │                    # StratifiedErrorsPanel, FeatureImportanceChart, ModelComparisonTable, ForensicAuditCard
        └── vision/          # FutureVisionTab (static conceptual v2 roadmap, zero backend calls)
```

### Pattern 1: Exact FastAPI REST API Contracts (`WEB-01`)

#### `GET /api/health` → `200 OK`
```json
{
  "status": "ok",
  "model_version": "avianscale-cv-v1.0.0",
  "winning_model_name": "LightGBMLog",
  "winning_ablation_condition": "C2_physics_normalized_strict",
  "pdi_config_hash": "43a3cfaa3208019e",
  "n_features": 39,
  "single_bird_presets_count": 6,
  "watershed_presets_count": 4,
  "classical_pdi_only": true
}
```

#### `GET /api/presets` → `200 OK`
Returns `models/presets_catalog.json` typed via Pydantic v2, augmenting every preset entry with `thumbnail_url: "/api/presets/{preset_id}/thumbnail"` (zero `rel_path` or `.jpg` strings).

#### `GET /api/presets/{preset_id}/thumbnail` → `200 OK` (`image/jpeg`) | `404 Not Found`
Looks up `preset_id` in the pre-warmed server-side allowlist dict (`10` presets total, `103.1 KiB` total RAM) and returns `Response(content=thumb_bytes, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})`. Unknown `preset_id` (including path traversal strings like `../../etc/passwd`) returns clean JSON `404`.

#### `POST /api/predict` (`multipart/form-data`) → `200 OK`
- **Inputs:**
  - `file: UploadFile | None = File(default=None)`
  - `preset_id: str | None = Form(default=None)`
  - `focal_length_mm: float | None = Form(default=None, ge=0.5, le=200.0)`
  - `zoom_multiple: float | None = Form(default=None, ge=0.1, le=20.0)`
  - Mutual exclusivity: normalize blank `preset_id` to `None`; if `(file is None) == (preset_id is None)` → HTTP `422`.
  - Reject non-finite `focal_length_mm` / `zoom_multiple` (`NaN`/`Inf`) with HTTP `422`.
- **Response schema (`PredictResponse`):**
  - `source_type: Literal["preset", "upload"]`
  - `preset_id: str | None`
  - `oof_evaluation: bool` (`True` for curated presets scored by `fold_pipelines[fold]`, `False` for user uploads scored by `production_pipeline`)
  - `model_version: str`, `winning_model_name: str`, `pdi_config_hash: str`
  - `predicted_weight_g: float`, `predicted_weight_kg: float`
  - `weight_band: Literal["<500g", "500-1000g", ">1000g"]`
  - `band_error_interval: BandErrorInterval` (`n`, `mae_g`, `mape_pct`, `median_abs_err_g`, `p68_abs_err_g`, `p90_abs_err_g`, `mbe_g`)
  - `age_stage: AgeStagePayload` (`stage_code`, `stage_label`, `estimated_age_days_range`, `out_of_training_range`, `valid_training_range_g: [116, 2093]`, `valid_age_days_range: "~7-29 days"`)
  - `oof_audit: OofAuditPayload | None` (populated strictly when `preset_id` is used: `fold`, `session_id`, `ground_truth_weight_g`, `abs_err_g`, `pct_err`, `signed_err_g`, `pair_group_id`)
  - `optical: OpticalPayload` (`focal_length_mm`, `zoom_multiple`, `exif_optical_present`, `override_applied`, `lens_class`)
  - `segmentation: SegmentationAuditPayload` (`scene_type`, `scene_is_bucket`, `lcd_excluded_px`, `lcd_core_count`, `segmentation_fallback`, `mask_area_frac`, `ellipse_from_min_area_rect`)
  - `xray: XRayPayload` — explicit data URIs ready for `<img src>` so frontend never mis-tags Stage 3 PNG as JPEG:
    - `stage1_original: str` (`data:image/jpeg;base64,...`)
    - `stage2_clahe: str` (`data:image/jpeg;base64,...`)
    - `stage3_mask: str` (`data:image/png;base64,...`)
    - `stage4_overlay: str` (`data:image/jpeg;base64,...`)
    - `display_w: int`, `display_h: int`, `total_b64_bytes: int`
  - `features_summary: dict[str, float]` (key morphometric/chromatic/optical subset for immediate readout: `area_optical_norm`, `area_px`, `area_frac`, `major_axis_optical_norm`, `minor_axis_optical_norm`, `solidity`, `eccentricity`, `circularity`, `yellow_down_ratio`, `white_plumage_ratio`, `pink_skin_ratio`, `feather_grad_mean`, `lab_a_mean`, `lab_b_mean`)
  - `model_features: dict[str, float]` (exact ordered `39` strictly mask-gated model features for expandable jury table)
  - `timing: TimingPayload` (`pdi_ms`, `inference_ms`, `total_ms`, `sla_target_ms: 500`, `within_sla: bool`)

#### `POST /api/watershed-demo` (`multipart/form-data`) → `200 OK`
- **Inputs:**
  - `file: UploadFile | None = File(default=None)`
  - `preset_id: str | None = Form(default=None)`
  - `h_rel: float | None = Form(default=None, ge=WATERSHED_MIN_H_REL, le=WATERSHED_MAX_H_REL)` (`0.05..0.80`, default `0.20`)
  - `min_distance_rel: float | None = Form(default=None, ge=WATERSHED_MIN_DISTANCE_REL, le=WATERSHED_MAX_DISTANCE_REL)` (`0.20..2.50`, default `0.85`)
  - `min_area_ratio: float | None = Form(default=None, ge=WATERSHED_MIN_AREA_RATIO, le=WATERSHED_MAX_AREA_RATIO)` (`0.01..0.45`, default `0.10`)
  - `focal_length_mm: float | None = Form(default=None, ge=0.5, le=200.0)`
  - `zoom_multiple: float | None = Form(default=None, ge=0.1, le=20.0)`
- **Response schema (`WatershedResponse`):**
  - `source_type: Literal["preset", "upload"]`, `preset_id: str | None`
  - `ground_truth_scope: Literal["group_sum_only"] = "group_sum_only"`
  - `disclaimer: str` (exact pt-BR disclosure matching `04-UI-SPEC.md:247` & `presets_catalog.json`)
  - `bird_count: int`
  - `estimated_total_weight_g: float`, `estimated_mean_weight_g: float | None` (`None` when `bird_count == 0`)
  - `preset_reference: WatershedPresetReference | None` (`folder_id`, `expected_bird_count`, `group_sum_weight_g`, `mean_bird_weight_reference_g`, `sum_diff_g`, `sum_pct_err`)
  - `parameters: WatershedParamsPayload` (`h_rel`, `min_distance_rel`, `min_area_ratio`, plus `defaults` & `bounds` so UI sliders are self-documenting)
  - `optical: OpticalPayload`, `red_feeder_excluded_px: int`, `pdi_config_hash: str`
  - `instances: list[WatershedInstancePayload]` (`instance_id`, `color_hex`, `centroid_xy`, `area_work_px`, `area_px`, `area_frac`, `area_optical_norm`, `estimated_weight_g`, `share_of_total_pct`, `age_stage`)
  - `visuals: WatershedVisualsPayload`:
    - `stage1_original: str` (`data:image/jpeg;base64,...`)
    - `stage2_union_mask: str` (`data:image/png;base64,...`)
    - `stage3_distance_peaks: str` (`data:image/jpeg;base64,...`)
    - `stage4_color_basins: str` (`data:image/jpeg;base64,...`)
    - `total_b64_bytes: int`
  - `timing: TimingPayload` (`pdi_ms`, `inference_ms`, `total_ms`, `sla_target_ms: 500`, `within_sla: bool`)

#### `GET /api/metrics` → `200 OK`
Returns a unified read-only JSON object served straight from `app.state` RAM cache:
```json
{
  "schema_version": "1.0.0",
  "pdi_config_hash": "43a3cfaa3208019e",
  "evaluation_report": { "...models/evaluation_report.json..." },
  "dataset_audit_summary": { "...models/dataset_audit_summary.json..." },
  "pdi_calibration_audit": { "...models/pdi_calibration_audit.json..." }
}
```
*(Bonus: add `Cache-Control: public, max-age=300` header; `~640 KiB` total JSON downloads over localhost in `< 2 ms`.)*

### Pattern 2: Bounded Async Upload Intake + Safe Pillow Validation

```python
# backend/app/api/services/upload_guard.py
from __future__ import annotations

import io
import warnings
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024  # 15 MiB (D-02)
MAX_IMAGE_PIXELS_API: int = 40_000_000    # 40 MP hard cap (>5x Redmi 10's 7.51 MP)
ALLOWED_IMAGE_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP"})


async def read_and_validate_upload(upload: UploadFile) -> bytes:
    """Read bounded upload bytes asynchronously and validate image magic & pixel bounds."""
    raw = await upload.read(MAX_UPLOAD_BYTES + 1)
    await upload.close()
    if len(raw) == 0:
        raise HTTPException(
            status_code=400,
            detail="Arquivo de imagem vazio. Envie uma foto JPG, PNG ou WEBP válida.",
        )
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="O arquivo ultrapassa o limite local de 15 MB.",
        )
    validate_image_bytes(raw)
    return raw


def validate_image_bytes(raw: bytes) -> str:
    """Verify magic format (JPEG/PNG/WEBP) and reject corrupt/decompression-bomb images."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                fmt = (im.format or "").upper()
                if fmt not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail="Formato de imagem não suportado. Envie JPG, PNG ou WEBP.",
                    )
                w, h = im.size
                if w <= 0 or h <= 0 or (w * h) > MAX_IMAGE_PIXELS_API:
                    raise HTTPException(
                        status_code=400,
                        detail="Dimensões da imagem inválidas ou acima do limite seguro (40 MP).",
                    )
                return fmt
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(
            status_code=400,
            detail="Imagem rejeitada por proteção contra descompressão excessiva.",
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível decodificar esta imagem com segurança.",
        ) from exc
```

### Pattern 3: Non-Clobbering Static SPA Serving (`D-05`)

Register all `/api/*` routers **first**, then conditionally mount `frontend/dist/assets` and register the catch-all SPA route **last**, explicitly returning JSON `404` for any unmatched `/api` or `/api/*` request `[CITED: FastAPI / Starlette static SPA pattern]`:

```python
# backend/app/api/main.py (tail of create_app)
dist_dir = FRONTEND_DIST_DIR
index_html = dist_dir / "index.html"
assets_dir = dist_dir / "assets"

if index_html.is_file():
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Endpoint da API não encontrado.")
        return FileResponse(index_html)
```

### Pattern 4: Biome v2 (`2.6.8`) + Tailwind CSS v4 (`4.3.5`) + Vite 8 (`8.3.8`) + TypeScript 7 (`7.0.2`) Config

#### `frontend/biome.json` (`[CITED: biomejs.dev v2 schema]`)
```json
{
  "$schema": "https://biomejs.dev/schemas/2.6.8/schema.json",
  "files": {
    "includes": ["**", "!!**/dist", "!!**/node_modules"]
  },
  "assist": {
    "enabled": true,
    "actions": {
      "source": {
        "organizeImports": "on"
      }
    }
  },
  "css": {
    "parser": {
      "tailwindDirectives": true
    }
  },
  "formatter": {
    "enabled": true,
    "indentStyle": "space",
    "indentWidth": 2,
    "lineWidth": 100
  },
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true
    }
  }
}
```

#### `frontend/vite.config.ts`
```ts
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
```

#### `frontend/tsconfig.app.json` (TypeScript 7.0.2 Native `tsc`)
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "verbatimModuleSyntax": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "types": ["vite/client"]
  },
  "include": ["src"]
}
```

### Pattern 5: Recharts v3 (`3.11.1`) Scatter & Bar Charts Under React 19

- **Never use `<Cell />`** (deprecated in Recharts v3 `[CITED: recharts v3 migration guide]`): put `fill` directly on each data row in the Ablation `BarChart` (`C1`: `#94A3B8`, `C2`: `#047857`, `C3`/`C4`: `#B45309`).
- **Always wrap `<ResponsiveContainer width="100%" height={360}>`** inside a parent with explicit height and `min-w-0` (`<div className="h-[360px] w-full min-w-0">`) so CSS Grid columns never collapse Recharts width to `0`.
- **Disable scatter animation (`isAnimationActive={false}`)** on `<Scatter>` for `1,738` points and control dot size with `<ZAxis range={[22, 22]} />` + `fillOpacity={0.55}` (Image mode) / `fillOpacity={0.85}` (Session mode).
- **Diagonal `y = x` ReferenceLine:** Both `<XAxis type="number" dataKey="actual_g" domain={[0, 2200]} />` and `<YAxis type="number" dataKey="predicted_g" domain={[0, 2200]} />` MUST declare numeric `type="number"` matching `<ReferenceLine segment={[{ x: 0, y: 0 }, { x: 2200, y: 2200 }]} stroke="#64748B" strokeDasharray="4 4" ifOverflow="extendDomain" />`.

### Anti-Patterns to Avoid

- **Never use `async def` for CPU-bound OpenCV/LightGBM work without `run_in_threadpool`:** Blocks Uvicorn's event loop so `/api/health`, `/api/metrics`, and thumbnail fetches stall behind inference.
- **Never decode uploaded bytes with `cv2.imdecode`:** Drops EXIF orientation on 81.86% of dataset photos (`PITFALLS.md` Pitfall 7). Always pass raw `bytes` straight to `run_single_bird_pdi` / `run_watershed_pdi`.
- **Never return raw `Exception` strings or stack traces in HTTP 400/500 responses:** Could accidentally leak absolute paths or dataset filenames (`PROIBIÇÃO 4`). Register a custom FastAPI global exception handler returning generic pt-BR JSON on unexpected 500s.
- **Never hardcode Watershed bounds in multiple places:** Import `WATERSHED_MIN_*`/`WATERSHED_MAX_*` from `backend.app.pdi.config` and expose them in `WatershedResponse.parameters.bounds`.
- **Never use `object-cover` on X-Ray or Watershed image viewports:** EXIF-transposed portraits (`288×640`) and landscapes (`640×288`) would crop out the broiler's head/tail/ellipse overlay. Always use `object-contain` inside fixed-aspect dark `#0F172A` frames.
- **Never block the UI on external CDN font downloads:** Pitch venues frequently have restricted/offline Wi-Fi! Use the system font stack declared in `04-UI-SPEC.md:26` (`"Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif` and `"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace`) with optional non-blocking local/system rendering — zero external font CDNs that hang offline boot.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Path containment for preset manifest resolution | Custom string `replace("..", "")` | Reuse `_resolve_safe_image_path` pattern from `backend/app/modeling/artifacts.py:264-294` (canonical root check + per-segment non-symlink check + `is_relative_to`) | Eliminates symlink escape and partial-prefix bypasses (ASVS V4). |
| Image decoding, EXIF extraction & orientation | Custom `struct` JPEG parser or `cv2.imdecode` | `backend.app.pdi.io_exif.load_working_rgb` + `run_single_bird_pdi` / `run_watershed_pdi` | Single-source EXIF-first path with zero training-serving skew (`pdi_config_hash`). |
| Model bundle validation & inference | Raw `joblib.load()` + manual numpy array ordering | `load_model_bundle(..., expected_pdi_config_hash=...)` + `predict_from_feature_dict` | Validates schema version, `PDIConfig` hash lock, all 5 fold pipelines, and exact 39-column DataFrame ordering with zero LightGBM warnings. |
| Cobb Avian 48 developmental age stage | Hardcoded `if/else` weight buckets in React | `backend.app.modeling.age_stage.map_weight_to_age_stage(weight_g)` | Single pure-stdlib source of truth for stage codes, pt-BR labels, and `116–2,093 g` `out_of_training_range` flag. |
| SVG data visualization charts | Hand-coded `<svg>` scatter/bar coordinate math | `recharts@3.11.1` + `react-is@19.3.1` | Responsive tooltips, numeric scales, dashed `y=x` `ReferenceLine`, and accessible SVG rendering out of the box. |
| Conditional Tailwind class merging | String concatenation `` `${a} ${b}` `` | `clsx@2.1.1` + `tailwind-merge@3.7.1` (`cn()`) | Safely resolves conflicting Tailwind v4 utility overrides. |

---

## Common Pitfalls

### Pitfall 1: Watershed Slider Bounds Mismatch (`04-CONTEXT.md` vs. `backend/app/pdi/config.py`)
- **What goes wrong:** Using `04-CONTEXT.md` draft slider bounds (`min_distance_rel: 0.10..0.80, default 0.34`) causes `run_watershed_pdi` to raise `ValueError` at `min_distance_rel < 0.20` and mismatches the Phase 2 default `0.85` (which produced `ws_417=2, ws_757=3, ws_823=4, ws_1520=2`).
- **How to avoid:** Import `WATERSHED_MIN_H_REL`, `WATERSHED_MAX_H_REL`, `WATERSHED_MIN_DISTANCE_REL`, `WATERSHED_MAX_DISTANCE_REL`, `WATERSHED_MIN_AREA_RATIO`, `WATERSHED_MAX_AREA_RATIO` directly from `backend.app.pdi.config` in FastAPI schemas, and mirror the exact `04-UI-SPEC.md:142-144` ranges (`h_rel: 0.05..0.80` default `0.20`, `min_distance_rel: 0.20..2.50` default `0.85`, `min_area_ratio: 0.01..0.45` default `0.10`) in React.

### Pitfall 2: `oof_scatter_points` Field Name (`actual_g`, NOT `weight_g`)
- **What goes wrong:** Typing `ScatterPoint.weight_g` in TypeScript causes Recharts `<XAxis dataKey="weight_g" />` to render an empty blank chart with zero runtime errors.
- **How to avoid:** Verified live from `models/evaluation_report.json`: the exact field name is `actual_g` (paired with `predicted_g`, `abs_err_g`, `pct_err`, `fold`, `image_id`, `session_id`, `lens_class`, `scene_type`, `weight_band`).

### Pitfall 3: Stage 3 X-Ray Mask & Stage 2 Watershed Mask Are PNG, Not JPEG
- **What goes wrong:** Prefixing all 4 stages with `data:image/jpeg;base64,` breaks rendering of the 1-channel PNG binary mask stages (`stage3_mask_b64` in `xray.py:177` and `stage2_union_mask_b64` in `watershed.py:422`).
- **How to avoid:** Assemble complete RFC 2397 data URIs on the backend (`data:image/png;base64,...` for binary masks, `data:image/jpeg;base64,...` for color/heatmap stages) and verify MIME prefixes in `pytest`.

### Pitfall 4: FastAPI SPA Catch-All Intercepting Unknown `/api/*` Routes
- **What goes wrong:** Mounting `StaticFiles(..., html=True)` at `"/"` or writing an unguarded `/{full_path:path}` route causes typos like `GET /api/unknown` to return `200 OK` with `index.html`, breaking frontend JSON error parsing and API contract tests.
- **How to avoid:** Mount `/assets` explicitly and guard the catch-all with `if full_path == "api" or full_path.startswith("api/"): raise HTTPException(404, ...)`. Test this explicitly in `pytest`!

### Pitfall 5: `multipart/form-data` Empty String vs. `None` Edge Case
- **What goes wrong:** Submitting HTML `FormData` with empty numeric inputs (`focal_length_mm=""`) triggers FastAPI `422` float parsing errors, or sending `preset_id=""` bypasses `preset_id is None` checks.
- **How to avoid:** Frontend omits optional keys from `FormData` unless populated with valid finite numbers; backend normalizes blank `preset_id` strings (`preset_id.strip() or None`) before mutual-exclusivity validation.

### Pitfall 6: Recharts v3 Grid Collapse & Slow Re-renders on 1,738 Scatter Points
- **What goes wrong:** Placing `<ResponsiveContainer>` directly inside a CSS Grid column without `min-w-0` + explicit height collapses charts to `0 px` width/height; leaving default Recharts animations enabled causes lag when toggling filters over `1,738` SVG circles.
- **How to avoid:** Use `<div className="h-[360px] w-full min-w-0">`, set `isAnimationActive={false}` on `<Scatter>`, memoize filtered & session-aggregated point arrays via `useMemo`, and avoid deprecated `<Cell />`.

### Pitfall 7: Rapid Preset / Slider Interactions Causing Out-of-Order UI Overwrites
- **What goes wrong:** Clicking preset A then preset B (or dragging a Watershed slider rapidly) lets a slower earlier response overwrite the newer result.
- **How to avoid:** Cancel superseded requests via `AbortController` on every `POST /api/predict` and `POST /api/watershed-demo` trigger, ignore `AbortError` cleanly, and fire Watershed slider requests on pointer/key commit (`onPointerUp`/`onKeyUp` or explicit `"Reprocessar Watershed"` CTA).

### Pitfall 8: `nvm` Node `PATH` Missing in Subshells / Launch Scripts
- **What goes wrong:** Running `npm run dev` or `npm run build` in a fresh non-interactive shell fails with `npm: command not found` because Node `v24.21.0` lives under `$HOME/.nvm/versions/node/v24.21.0/bin`.
- **How to avoid:** Explicitly prepend `$HOME/.nvm/versions/node/v24.21.0/bin` (or source `$NVM_DIR/nvm.sh`) in any convenience launch script and document the exact command in Phase 4 plans.

---

## Security Domain

`security_enforcement` is `true` at **ASVS Level 1** (`security_asvs_level: 1`, `security_block_on: "high"`) `[VERIFIED: .planning/config.json:48-50]`.

### Applicable ASVS Categories (L1)

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| **V2 Authentication** | No | Local single-user pitch MVP (explicitly out of scope per `REQUIREMENTS.md:67` & `04-CONTEXT.md`); bind Uvicorn strictly to loopback `127.0.0.1` (`--host 127.0.0.1`), never `0.0.0.0`. |
| **V3 Session Management** | No | Stateless read-only API; client-side simulated flock uses ephemeral React state / `sessionStorage` with zero cookies or server sessions. |
| **V4 Access Control** | Yes | Opaque `preset_id` server-side allowlist (`6` single-bird + `4` watershed); manifest `rel_path` resolved strictly under `RAW_DATASET_DIR.resolve()` with per-segment non-symlink verification (`_resolve_safe_image_path` pattern); zero arbitrary filesystem reads. |
| **V5 Input Validation** | Yes | FastAPI + Pydantic v2 strict bounds (`focal_length_mm ∈ [0.5, 200.0]`, `zoom_multiple ∈ [0.1, 20.0]`, Watershed `h_rel ∈ [0.05, 0.80]`, `min_distance_rel ∈ [0.20, 2.50]`, `min_area_ratio ∈ [0.01, 0.45]`, reject `NaN`/`Inf`/`bool`); bounded async upload read (`15 MiB` → `413`); Pillow magic allowlist `{JPEG, PNG, WEBP}` (`415`); decompression-bomb guard (`DecompressionBombWarning` escalated to error + `40 MP` pixel cap → `400`). |
| **V6 Cryptography** | Minimal | `PDIConfig` SHA-256 hash lock (`43a3cfaa3208019e`) verified at startup inside `load_model_bundle` (`hashlib.sha256` stdlib); no secrets, passwords, or JWTs. |

### Known Threat Patterns for FastAPI + Pillow/OpenCV + React SPA

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| **Arbitrary Path Traversal / Symlink Escape** via `preset_id` (`../../etc/passwd`) | Information Disclosure | Server-side in-memory allowlist `dict[str, Path]` populated strictly at startup from `manifest_*.parquet` validated via `_resolve_safe_image_path(..., RAW_DATASET_DIR.resolve())`; reject unknown IDs with HTTP `404`. |
| **Memory Exhaustion / DoS via Oversized Upload or Image Decompression Bomb** | Denial of Service | Bounded async chunk read `await upload.read(15 * 1024 * 1024 + 1)` → HTTP `413`; Pillow `DecompressionBombWarning` escalated to error + `w * h <= 40_000_000` check before pixel decode; `cv2.setNumThreads(2)` to protect `7.1 GiB` RAM host. |
| **Raw Dataset Path / Filename / Stack-Trace Leakage (`PROIBIÇÃO 4`)** | Information Disclosure | Never emit `rel_path`, `.jpg` filenames, or absolute paths in Pydantic schemas or HTTP errors; register global unhandled-exception handler returning sanitized pt-BR JSON `500`; add automated `pytest` assertion scanning all API responses for `.jpg`, `rel_path`, `data/raw`, and home-directory prefixes. |
| **Cross-Origin Unauthorized Local API Access** | Elevation / Spoofing | Bind Uvicorn to `127.0.0.1:8000` and restrict `CORSMiddleware` `allow_origins` strictly to explicit localhost origins (`http://127.0.0.1:5173`, `http://localhost:5173`, `http://127.0.0.1:8000`, `http://localhost:8000`), never `["*"]`. |
| **XSS via User-Uploaded Filename or Base64 Image Injection** | Tampering | React default text escaping (never `dangerouslySetInnerHTML`); truncate uploaded display filenames safely; validate/generate Data URIs strictly server-side from encoded OpenCV buffers (`data:image/jpeg;base64,...` / `data:image/png;base64,...`). |
| **Silent Serving Skew / Tampered Model Bundle** | Tampering | Startup `load_model_bundle(MODEL_BUNDLE_PATH, expected_pdi_config_hash=PDIConfig().config_hash())` fails fast on schema mismatch, missing fold pipelines, or `PDIConfig` drift. |

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `uv` | Backend venv & execution | ✓ | `0.13.9` | — |
| Python (`uv`-managed) | FastAPI, PDI, LightGBM | ✓ | `3.12.14` | — |
| Backend Python stack (`fastapi`, `uvicorn`, `pydantic`, `python-multipart`, `httpx`, `pytest`, `ruff`) | `WEB-01` & API tests | ✓ | Verified in `.venv` (`0.165.0` / `0.54.0` / `2.14.4` / `0.0.34` / `0.28.1` / `9.2.0` / `0.17.6`) | — |
| Node.js (`nvm`) | Frontend Vite 8 / TS 7 / Vitest | ✓ | `v24.21.0` *(prepend `$HOME/.nvm/versions/node/v24.21.0/bin` to `PATH`)* | — |
| npm (`nvm`) | Frontend dependency install | ✓ | `11.19.0` | — |
| Published Phase 1–3 artifacts (`manifest_*.parquet`, `weight_regressor.joblib`, 4 JSON reports) | FastAPI `lifespan` boot | ✓ | All verified on disk (`pdi_config_hash = 43a3cfaa3208019e`) | — |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| *(none)* | All stack versions, peer dependencies, upstream Python signatures, JSON schemas, preset resolutions, and live CPU latencies were empirically verified in this session. | — | — |

---

## Open Questions (RESOLVED)

1. **[RESOLVED] Dual-Lens Side-by-Side Comparison UX Trigger (`sb_dual_lens_wide` vs `sb_dual_lens_ultrawide`):**
   - What we know: `04-UI-SPEC.md:123` specifies a 1-click **"Comparar Par Dual-Lens"** CTA that displays the `541 g` Wide (`383.2 g`) vs. Ultrawide (`384.2 g`) live optical-invariance proof (`0.94 g` / `0.24%` delta despite `~6.6×` pixel-area swing).
   - **Resolution (`04-03-PLAN.md`, Task 1 — `DualLensComparison.tsx`):** Adopted verbatim in `frontend/src/components/studio/DualLensComparison.tsx` (triggered from `PresetsStrip.tsx` via **"Comparar Par Dual-Lens"**). Executes two parallel `POST /api/predict` calls (`preset_id=sb_dual_lens_wide` and `preset_id=sb_dual_lens_ultrawide`, `~140 ms` combined), caches the paired response in React state, and renders a side-by-side dark `#0F172A` Stage 4 overlay panel contrasting the raw `area_px` swing against stabilized `area_optical_norm = area_px / zoom²` and highlighting the live `0,94 g` (`0,24%`) OOF delta in Emerald `#047857`.

---

## Sources

### Primary (HIGH confidence)
- In-repo verified source contracts: `backend/app/config.py:12-24`, `backend/app/pdi/config.py:18-23,68-107`, `backend/app/pdi/io_exif.py:24-36,61-89`, `backend/app/pdi/pipeline.py:28-89`, `backend/app/pdi/xray.py:1-10,34-45,157-190`, `backend/app/pdi/watershed.py:57-128,400-468`, `backend/app/pdi/segmentation.py:33-51`, `backend/app/pdi/features.py:39-103`, `backend/app/ingestion/decoding.py:28-29,111-117`, `backend/app/modeling/artifacts.py:61-149,264-295,297-434,509-521`, `backend/app/modeling/models.py:28-35,201-238`, `backend/app/modeling/age_stage.py:18-77`, `pyproject.toml:1-49`, `.gitignore:1-18`.
- In-repo verified published artifacts: `models/presets_catalog.json`, `models/dataset_audit_summary.json`, `models/pdi_calibration_audit.json`, `models/evaluation_report.json`, `models/weight_regressor.joblib`, `data/processed/manifest_clean.parquet`, `data/processed/manifest_quarantine.parquet`.
- Live empirical spikes executed via `uv run python` (`cv2.setNumThreads(2)`) and live npm registry checks via `npm view` + `gsd-tools.cjs query package-legitimacy check --ecosystem npm`.

### Secondary (MEDIUM confidence)
- Biome v2 configuration docs (`biomejs.dev`) — `assist.actions.source.organizeImports`, `files.includes`, and `css.parser.tailwindDirectives: true` `[CITED: biomejs.dev]`.
- TypeScript 7.0 native compiler (`tsc`) + Vite/React integration docs `[CITED: typescript 7.0 release notes]`.
- Recharts v3 migration guide — `<Cell />` deprecation, numeric `ReferenceLine segment`, and `ResponsiveContainer` parent sizing `[CITED: recharts v3 docs]`.
- FastAPI / Starlette static SPA catch-all guard avoiding `/api/*` interception `[CITED: fastapi / starlette patterns]`.

---

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** — Every Python and npm package version and peer dependency verified live today (`2026-11-24`), with zero peer conflicts.
- Architecture & API contracts: **HIGH** — Every upstream Phase 1–3 function signature, JSON schema key, manifest join, and warm CPU latency (`60–81 ms` single-bird, `146–168 ms` watershed) verified empirically.
- Pitfalls & Security (ASVS L1): **HIGH** — Reconciled Watershed bounds mismatch, verified Pillow format/bomb guards, verified `actual_g` scatter schema and Biome v2/Recharts v3 gotchas.

**Research date:** 2026-11-24
**Valid until:** 2026-12-24 (locked through 2026-11-28 MVP freeze & 2026-11-30 pitch)
