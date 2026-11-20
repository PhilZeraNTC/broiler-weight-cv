<!-- GSD:project-start source:PROJECT.md -->

## Project

**AvianScale CV**

AvianScale CV é um sistema de visão computacional de baixo custo para estimativa não invasiva de peso vivo individual e médio de frangos de corte, combinando técnicas clássicas e interpretáveis de Processamento Digital de Imagens (PDI) com regressão tabular calibrada por parâmetros ópticos EXIF. Voltado a produtores integrados e agroindústrias avícolas, o MVP entrega uma aplicação web interativa local para pitch acadêmico/produto, validada sobre toda a base limpa do dataset público *Broiler weights of Cobb Avian* (1.738 imagens, 160 pesos únicos, 188 sessões), exibindo o passo a passo visual do PDI ("visão raio-X"), estimativa de peso e um módulo demonstrativo de separação de aves encostadas via *Watershed*.

**Core Value:** Estimar o peso vivo individual de frangos de corte de forma precisa, rigorosamente auditada e 100% interpretável usando apenas Processamento Digital de Imagens clássico calibrado por óptica EXIF — sem deep learning caixa-preta e com zero vazamento de dados do visor LCD da balança — exposto visualmente etapa por etapa em uma demonstração web local de alta credibilidade.

### Constraints

- **Classical PDI Only**: Proibido usar redes neurais/deep learning (CNNs/YOLO/ViTs) na segmentação ou regressão principal — exigência central de avaliação técnica em PDI clássico interpretável.
- **Anti-Leakage (Visual & Split)**: Exclusão obrigatória do visor LCD/prato SF-400 antes da extração de atributos e validação estritamente agrupada por `session_id` via 5-fold `StratifiedGroupKFold`.
- **Timeline**: Conclusão do código MVP até **2026-11-28** para apresentação de pitch em **2026-11-30** (8 dias corridos).
- **Performance**: Inferência ponta a ponta na API `/api/predict` abaixo de **500 ms** por imagem em hardware local (CPU).
- **Tech Stack**: Backend Python 3.11+ (`opencv-python-headless`, `scikit-image`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`, `pillow`, `fastapi`, `uvicorn`, `joblib`); Frontend React + TypeScript + Vite + Tailwind CSS.
- **Data Hygiene**: `data/raw/`, `data/processed/`, `models/*.joblib` e imagens avulsas nunca entram no Git (já protegidos pelo `.gitignore`).

<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Python (`uv`-managed)** | `3.12.14` (pin `.python-version = 3.12`) | Backend, PDI engine, ML pipeline runtime | Already pre-installed locally under `uv` and decoupled from the host's rolling `/usr/bin/python3.14` (which can break venv symlinks on OS updates). Battle-tested ABI across OpenCV, scikit-image, LightGBM, and `joblib` with 100% pure binary wheels (`--only-binary :all:`). |
| **`opencv-python-headless`** | `>=4.14.0,<5.0` (verified `4.14.0.94`) | Primary classical PDI engine (CLAHE, bilateral/median blur, Lab/HSV conversion, Otsu/adaptive thresholding, morphology, contours, ellipse fitting, JPEG/PNG encoding) | Headless build omits Qt/GTK/X11/libGL GUI bloat (safe for API workers and containers). Pinning the terminal `4.14.x` line over `5.1.0.95` avoids OpenCV 5.0 generational transitions whose headline gain (rewritten DNN engine) is explicitly banned by PRD `PROIBIÇÃO 1`, while remaining 100% compatible with `numpy 2.5.5`. |
| **`scikit-image`** | `>=0.26.0,<0.27` (verified `0.26.0`) | High-level morphological descriptors & Marker-based Watershed (`regionprops`, `watershed`, `peak_local_max`, `remove_small_objects`) | Complements OpenCV where scikit-image is decisively cleaner: `regionprops` computes all REQ-05 geometric descriptors (`area`, `perimeter`, `axis_major_length`, `axis_minor_length`, `eccentricity`, `area_convex`, `solidity`) in one verified pass, and `skimage.segmentation.watershed` operates directly on float negative distance maps without `cv2.watershed`'s 3-channel uint8/border `-1` quirks (REQ-07). |
| **`pillow`** | `>=13.0.0,<14.0` (verified `13.0.0`) | Forensic JPEG ingestion, mandatory EXIF orientation transpose, and Xiaomi optical tag extraction | OpenCV does **not** expose raw EXIF tags. Pillow's `ImageOps.exif_transpose()` fixes the 81.86% rotated images (`Orientation != 1`), and `im.getexif().get_ifd(0x8769)` reads `FocalLength` (`37386`) and Xiaomi proprietary JSON tag `39321` (`0x9999`, `zoomMultiple`), matching Section 9 of the forensic audit verbatim. |
| **`scikit-learn`** | `>=1.9.0,<2.0` (verified `1.9.1`) | Anti-leakage CV (`StratifiedGroupKFold`), preprocessing pipelines, target log-transform, linear/robust/forest baselines, and metrics | Provides mandatory 5-fold `StratifiedGroupKFold` by `session_id` (`PROIBIÇÃO 3`), `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` to optimize MAPE across 116–2,093 g, `Ridge`, `HuberRegressor`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, and stratified MAE/RMSE/MAPE/$R^2$ metrics. |
| **`lightgbm`** | `>=4.7.0,<5.0` (verified `4.7.0`) | Primary non-linear tabular regressor (`LGBMRegressor`) | Captures non-linear interactions between normalized optical area $A_{px}/(f_{mm}\cdot\text{zoom})^2$, lens class, scene cues, and plumage yellow/white ratios; natively supports Huber/L1 objectives and `sample_weight` for the sparse `>1,000 g` tail; `< 2 ms` single-sample CPU inference. |
| **`fastapi`** + **`uvicorn[standard]`** | `fastapi >=0.165.0,<1.0` (`0.165.0`), `uvicorn[standard] >=0.54.0,<1.0` (`0.54.0`) | Local REST API serving `/api/predict`, `/api/watershed-demo`, `/api/presets`, `/api/metrics` | Pydantic v2 typed request/response contracts, automatic OpenAPI docs for pitch auditing, and native threadpool offloading for sync `def` CPU-bound CV routes to comfortably meet the `< 500 ms` latency budget. |
| **React + TypeScript** | `react 19.3.1`, `react-dom 19.3.1`, `typescript 7.0.2` | Interactive X-Ray PDI Studio SPA | Industry-standard typed component UI for side-by-side 4-stage PDI inspection, interactive preset comparisons, Watershed gallery, and live metrics dashboards. TypeScript 7's native Go compiler gives near-instant type-checking (`tsc --noEmit`). |
| **Vite** | `vite 8.3.8`, `@vitejs/plugin-react 6.3.0` | Frontend build tool & local dev server with `/api` proxy | Rolldown-powered single bundler with sub-second HMR and native `server.proxy` routing `/api/*` → `http://127.0.0.1:8000` (zero CORS friction in dev; matches single-origin static serving for pitch day). |
| **Tailwind CSS v4** | `tailwindcss 4.3.5`, `@tailwindcss/vite 4.3.5` | Utility-first styling for data-dense clinical/agritech UI | First-party Vite plugin (verified Vite 8 peer support) eliminates `tailwind.config.js` and PostCSS boilerplate — just `@import "tailwindcss";` and CSS `@theme` design tokens. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **`numpy`** | `>=2.5.0,<3.0` (verified `2.5.5`) | Core N-dimensional array substrate shared by OpenCV, scikit-image, SciPy, scikit-learn, LightGBM | Always — core numeric layer for masks, Lab/HSV histograms, optical scaling formulas, and vectorized feature extraction. |
| **`scipy`** | `>=1.18.0,<2.0` (verified `1.18.1`) | Exact Euclidean Distance Transform (`ndimage.distance_transform_edt`), Gaussian smoothing (`ndimage.gaussian_filter`), morphological hole filling (`binary_fill_holes`), distribution skewness/kurtosis | Phase 2 segmentation refinement and Watershed multi-bird separation (exact EDT yields smoother, cleaner peaks than OpenCV's chamfer approximation). |
| **`pandas`** | `>=3.1.0,<4.0` (verified `3.1.1`) | Tabular manifest construction, session grouping, feature tables, and stratified evaluation slicing | Phase 1 manifest curation (`manifest_clean.parquet`, `manifest_quarantine.parquet`) and Phase 3 modeling/ablation tables. |
| **`pyarrow`** | `>=26.0.0,<27.0` (verified `26.0.1`) | Typed Parquet I/O engine for Pandas | Phase 1–3 persistence of manifests and extracted feature tables (preserves `datetime64`, `date`, floats, booleans, and UTF-8 Arabic/mojibake filenames losslessly). |
| **`joblib`** | `>=1.6.0,<2.0` (verified `1.6.0`) | Production pipeline serialization (`models/weight_regressor.joblib`) and bounded offline batch parallelism | Phase 2 offline batch extraction (`n_jobs=4` max to respect 7.1 GiB RAM) and Phase 3–4 model export/load with version & error-band metadata. |
| **`pydantic`** | `>=2.14.0,<3.0` (verified `2.14.4`) | Strict API schema validation (input bounds on manual lens/zoom overrides, feature dicts, X-ray base64 images, evaluation report schema) | Phase 4 FastAPI backend (installed alongside FastAPI; pin major `>=2.14,<3.0`). |
| **`python-multipart`** | `>=0.0.34` (verified `0.0.34`) | Multipart form parsing for image uploads + form fields (`UploadFile` + `Form(...)`) | **Mandatory** whenever FastAPI exposes `POST /api/predict` or `POST /api/watershed-demo` file upload endpoints (FastAPI refuses to start without it). |
| **`recharts`** + **`react-is`** | `recharts 3.11.0`, `react-is 19.3.1` | Declarative React SVG charts for Metrics & Audit Dashboard (REQ-15) | Phase 4: Real vs. Predicted `ScatterChart` (with diagonal $y=x$ `<ReferenceLine segment={[...]} />`), Optical Ablation grouped `BarChart`, and per-bin MAE/MAPE charts. **Always install `react-is@19.3.1` alongside `recharts` under React 19.** |
| **`lucide-react`** | `1.77.0` | Clean vector icon set (React 19 native) | Phase 4 UI controls: camera/lens EXIF indicators, X-ray pipeline step badges, Watershed controls, upload dropzones, audit checkmarks. |
| **`clsx`** + **`tailwind-merge`** | `clsx 2.1.1`, `tailwind-merge 3.7.1` | Safe conditional Tailwind class merging (`cn()` helper) | Phase 4 reusable UI primitives (status badges, stage selector tabs, error-rate pills). |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **`uv`** (`0.13.9`) | Python virtualenv, dependency locking (`uv.lock`), and deterministic script execution (`uv run`) | Single workspace `pyproject.toml` at repo root (or `backend/pyproject.toml` — root is cleaner for `scripts/` + `backend/` shared PDI package imports). Use `.python-version` set to `3.12`. |
| **`pytest`** (`9.2.0`) + **`httpx`** (`0.28.1`) | Unit/contract tests for forensic ingestion, anti-leakage fold integrity, PDI mask sanity (zero LCD leakage checks), Watershed counts, and FastAPI `TestClient` | `httpx` is required by `fastapi.testclient.TestClient`. Keep tests fast using synthetic fixtures + a tiny subset of manifest assertions (never dump `data/raw/` paths in logs). |
| **`ruff`** (`0.17.6`) | Ultra-fast Python linter and formatter | Replaces Flake8, isort, and Black in one binary (`ruff check` + `ruff format`). Configure in `pyproject.toml` with `target-version = "py312"`. |
| **`@biomejs/biome`** (`2.6.8`) | Single Rust-based linter + formatter for Frontend TypeScript/TSX/JSON/CSS | **Prescriptive choice over ESLint + `typescript-eslint`:** `typescript-eslint@8.73.0` caps its peer dependency at `typescript < 6.1.0`, conflicting with `typescript@7.0.2`. Biome works seamlessly with TS 7 + Vite 8 with a single `biome.json`. |
| **`@types/react` (`19.3.0`), `@types/react-dom` (`19.3.0`), `@types/node` (`26.6.6`)** | TypeScript ambient declarations for React 19 and `vite.config.ts` | Required devDependencies for strict `tsc --noEmit` checking under `tsconfig.app.json` and `tsconfig.node.json`. |

## Installation

### Backend & PDI/ML Environment (`uv` + Python 3.12)

# Pin Python 3.12 (uses already-installed cpython-3.12.14, isolated from system Python 3.14)

# Core PDI, Data Engineering, ML & API dependencies

# Dev & test dependencies

### Frontend Environment (`nvm` Node 24.21 + Vite 8 + React 19 + Tailwind v4)

# Activate local Node v24.21.0 if not already on PATH

# Scaffold React + TypeScript app in frontend/ (or initialize cleanly inside frontend/)

# Core runtime dependencies

# Build, Tailwind v4, TypeScript 7, and Biome dev dependencies

#### Minimal `frontend/vite.config.ts` (Tailwind v4 + FastAPI Dev Proxy)

#### Minimal `frontend/src/index.css` (Tailwind v4 Zero-Config Entry)

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **Python 3.12 (`uv`)** | **System Python 3.14.10** | Only if benchmarking PEP 703 free-threaded (no-GIL) multi-threaded CV pipelines. All 49 packages have 3.14 wheels today, but tying a project venv to rolling `/usr/bin/python3` risks OS-upgrade breakage mid-sprint. |
| **`opencv-python-headless` `4.14.x`** | **`opencv-python-headless` `5.1.0.95`** | Use 5.1.x on greenfield projects that actively use OpenCV 5's rewritten ONNX/DNN inference engine or true 0D/1D `cv::Mat` APIs. Since DNN is prohibited here, `4.14.0.94` maximizes classical CV stability with zero downside. |
| **`scikit-image` + `scipy` for Watershed** | **Pure `cv2.watershed` + `cv2.distanceTransform`** | Use pure OpenCV only if eliminating `scikit-image` to minimize Docker image size (<50 MB) for an embedded ARM edge device. For our local MVP + REQ-05/REQ-07, `skimage.measure.regionprops` + `skimage.segmentation.watershed` + `scipy.ndimage.distance_transform_edt` cut code volume in half and produce cleaner multi-bird basin boundaries. |
| **`LightGBM` + `scikit-learn` (`Ridge`, `Huber`, `RF`, `HistGB`)** | **CatBoost / XGBoost** | Use CatBoost if the feature table had high-cardinality categorical columns, or XGBoost if deploying to a legacy JVM/Spark inference runtime. With ~25–35 numeric PDI/optical features and 1,738 rows, LightGBM + scikit-learn cover linear, robust, bagging, and GBDT families with minimal binary footprint. |
| **`Tailwind CSS v4` (`@tailwindcss/vite`)** | **Tailwind CSS v3 (`postcss` + `autoprefixer`)** | Use v3 only when integrating legacy third-party UI kits whose plugin API has not migrated to v4 CSS-first `@theme`. Greenfield custom React components are drastically simpler and faster on v4. |
| **`Recharts` `3.11.0`** | **Apache ECharts (`echarts`) / Plotly.js** | Use ECharts or Plotly for 3D plots, WebGL >50k-point scatter clouds, or interactive zoom/brush analytics. For 1,738 validation points (or downsampled/session-grouped points) + 2–3 bar charts in a React 19 pitch dashboard, Recharts keeps bundle size small and integrates idiomatically with Tailwind CSS. |
| **`TypeScript 7` + `Biome`** | **`TypeScript 6.0.3` + `ESLint 10` + `typescript-eslint`** | Use TS 6 + ESLint only if the team requires deep type-aware ESLint plugins (e.g., strict custom enterprise rules). For an 8-day MVP sprint, TS 7 + Biome delivers 10× faster checks with zero peer-dependency conflicts. |
| **Native `fetch` + typed `api.ts`** | **Axios / TanStack Query** | Use TanStack Query for complex multi-user SaaS apps with cache invalidation, optimistic updates, and pagination. AvianScale CV calls 4 local synchronous REST endpoints; native `fetch` with `AbortController` adds zero dependencies. |
| **FastAPI + React/Vite SPA** | **Next.js (static export) or Streamlit/Gradio** | Use Next.js if SEO/SSR/multi-route marketing pages are needed; use Streamlit only for throwaway internal notebooks. A FastAPI + React/Vite SPA directly satisfies REQ-11 through REQ-15 with full pixel-level control over the 4-panel "X-ray" PDI studio and `< 500 ms` responsiveness. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **PyTorch, TensorFlow, Keras, Ultralytics YOLO, SAM, timm, ONNX Runtime, PaddleOCR, EasyOCR** | **Explicitly prohibited by PRD `PROIBIÇÃO 1` and `PROIBIÇÃO 2`**. Violates the core academic/technical requirement (pure classical interpretable PDI + zero LCD OCR/visual leakage) and risks OOM on 7.1 GiB RAM. | `opencv-python-headless` + `scikit-image` + `scipy` + `scikit-learn` + `lightgbm`. |
| **`opencv-python` or `opencv-contrib-python` (GUI builds)** | Bundles Qt/GTK/X11/OpenGL dependencies that bloat the venv, can fail with `libGL.so.1` errors in headless shells, and mask accidental `cv2.imshow()` calls that hang FastAPI workers. | `opencv-python-headless>=4.14.0,<5.0`. |
| **`sklearn.model_selection.train_test_split` or plain `KFold`** | **Explicitly prohibited by PRD `PROIBIÇÃO 3`**. Splits ~9 burst photos of the same bird across train and validation, artificially inflating $R^2$ and destroying scientific credibility. | `sklearn.model_selection.StratifiedGroupKFold(n_splits=5)` grouped by `session_id` and stratified by weight bins (`<500 g`, `500–1000 g`, `>1000 g`). |
| **`cv2.imread()` as sole image ingestion path** | Discards EXIF metadata (`FocalLength`, Xiaomi proprietary tag `39321` `zoomMultiple`) needed to correct the `6.6×` pixel-area optical ambiguity, and creates orientation risks across 81.86% of images. | Open with `PIL.Image.open()`, extract EXIF/IFD `0x8769`, apply `ImageOps.exif_transpose()`, then convert to NumPy for OpenCV. |
| **CSV for intermediate dataset manifests (`manifest_clean.csv`)** | Drops column dtypes (`datetime64`, nullable floats, booleans), invites locale decimal/encoding bugs with Arabic (`٠-٩`) and CP866 mojibake strings, and slows I/O. | Parquet via `pandas` + `pyarrow` (`data/processed/manifest_clean.parquet`). |
| **`async def` FastAPI routes for CPU-bound CV/ML work** | Running OpenCV/scikit-image/LightGBM inside an `async def` coroutine blocks Uvicorn's single event loop, stalling concurrent requests (e.g., `/api/metrics` or preset thumbnails while `/api/predict` runs). | Define CPU-bound endpoints as plain synchronous `def` handlers so FastAPI automatically offloads them to AnyIO's threadpool. |
| **`eslint` + `typescript-eslint@8.73.0` alongside `typescript@7.0.2`** | Verified live peer-dependency conflict: `typescript-eslint@8.73.0` requires `typescript >=4.8.4 <6.1.0`. Installing both triggers npm `ERESOLVE` errors or broken lint runs. | `@biomejs/biome@^2.6.8` with `typescript@^7.0.2` (or pin `typescript@~6.0.3` if ESLint is mandatory). |
| **`recharts` without explicit `react-is@19.3.1`** | Under React 19, missing or version-mismatched `react-is` causes peer warnings or silent blank SVG chart renders in Recharts `ScatterChart` / `ReferenceLine`. | Explicitly install `react-is@^19.3.1` matching `react@^19.3.1`. |
| **Raw `4080×1840` uncompressed PNG base64 payloads in `/api/predict`** | Returning four 7.5 MP PNGs in JSON produces ~25–40 MB payloads and takes >1.5 s just to PNG-compress, violating the `< 500 ms` API latency target. | Downscale X-ray visual inspection artifacts to max dimension `800–960 px` and encode photo stages as JPEG (quality `82–85`) + mask as compact PNG (`< 150 KB` total payload, `< 25 ms` encoding). |
| **Unbounded `n_jobs=-1` across 16 CPUs during offline feature extraction** | Loading 16 full-res `4080×1840` images + intermediate Lab/HSV/distance float64 arrays concurrently on a **7.1 GiB RAM** host triggers Linux OOM killer or heavy swap thrashing. | Cap offline batch extraction to `joblib.Parallel(n_jobs=4)` (and resize to working PDI scale immediately after EXIF transpose). |

## Stack Patterns by Variant

- Run FastAPI via `uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000` (1 worker) and Vite via `npm run dev` on `:5173` with `server.proxy["/api"]` forwarding to `:8000`.
- Because Vite proxies `/api`, frontend code calls relative paths (`/api/predict`, `/api/presets`, `/api/metrics`, `/api/watershed-demo`) with zero hardcoded hosts or CORS workarounds.
- Build static frontend (`npm run build` → `frontend/dist/`) and mount `frontend/dist/` directly in FastAPI (`StaticFiles(..., html=True)`) as an optional single-command launch mode (`uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1`).
- Because a single Python process serves both the UI and API with zero network/HMR failure modes, pitch startup becomes one command and stays well under `400 MB` RAM.
- Write all DataFrame mutations using single-step `.loc[mask, col] = val` (Pandas 3 enforces Copy-on-Write globally; chained assignment raises `ChainedAssignmentError`).
- Select strictly defined numeric feature lists (`FEATURE_COLS`) when passing `X` to `scikit-learn` / `LightGBM`; never rely on `df.select_dtypes(include=["object"])` because Pandas 3 assigns text columns the new Arrow-backed `str` dtype.
- Wrap all regressors in `TransformedTargetRegressor(regressor=..., func=np.log1p, inverse_func=np.expm1)` to optimize relative error (MAPE) cleanly while exposing predictions directly in grams.
- Draw the $y = x$ perfect-prediction diagonal on the Real vs. Predicted `ScatterChart` using `<ReferenceLine segment={[{ x: 0, y: 0 }, { x: 2200, y: 2200 }]} strokeDasharray="4 4" />` with explicit `type="number"` and matching numeric `domain={[0, 2200]}` on both `<XAxis />` and `<YAxis />`.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| **`python@3.12.14`** | All 49 resolved backend/ML/test packages | Verified 100% pure wheel resolution (`uv pip compile --python-version 3.12 --only-binary :all:`). Also verified for Python `3.14`, but `3.12` is pinned via `uv` to isolate from rolling OS updates. |
| **`opencv-python-headless@4.14.0.94`** | `numpy@2.5.5`, `scikit-image@0.26.0`, `scipy@1.18.1` | Verified clean binary resolution under NumPy 2.5.5; keeps stable OpenCV 4.x classical CV bindings (`findContours` 2-tuple return, `createCLAHE`, `bilateralFilter`, `fitEllipse`). |
| **`pandas@3.1.1`** | `pyarrow@26.0.1`, `scikit-learn@1.9.1`, `lightgbm@4.7.0` | Mandatory Copy-on-Write (`ChainedAssignmentError` on chained indexing), Arrow-backed `str` dtype for text columns, microsecond `[us]` default datetimes, read-only `.to_numpy()`. Cast any categorical column explicitly via `.astype("category")` before LightGBM; preserve exact `feature_name_` column ordering at inference. |
| **`fastapi@0.165.0`** | `pydantic@2.14.4`, `starlette@1.10.0`, `uvicorn@0.54.0`, `python-multipart@0.0.34`, `httpx@0.28.1` | `python-multipart` is mandatory for file uploads; `httpx` is mandatory for `TestClient`. |
| **`vite@8.3.8`** | `@vitejs/plugin-react@6.3.0`, `@tailwindcss/vite@4.3.5`, `node@24.21.0` | Verified live peer deps: `@vitejs/plugin-react@6.3.0` requires `vite ^8.0.0`; `@tailwindcss/vite@4.3.5` supports `vite ^5.2.0 \|\| ^6 \|\| ^7 \|\| ^8`; Vite 8 requires Node `^20.19.0 \|\| >=22.12.0` (host has `24.21.0`). |
| **`react@19.3.1`** | `react-dom@19.3.1`, `recharts@3.11.0`, `react-is@19.3.1`, `lucide-react@1.77.0`, `@types/react@19.3.0` | Verified live peer deps: `recharts@3.11.0` explicitly supports `react ^19.0.0` AND requires matching `react-is ^19.0.0`; `lucide-react@1.77.0` natively supports `react ^19.0.0`. |
| **`typescript@7.0.2`** | `vite@8.3.8`, `@biomejs/biome@2.6.8` | **Incompatible with `typescript-eslint@8.73.0`** (which caps `typescript < 6.1.0`). Use `@biomejs/biome` for linting/formatting and `tsconfig.json` with `"moduleResolution": "bundler"`, `"target": "ES2022"`, `"verbatimModuleSyntax": true`. |

## Sources

- **Local Environment & Live Registry Verification (HIGH confidence):**
- **Project Forensic Documentation & Constraints (HIGH confidence):**
- **Official Ecosystem & Compatibility Research (HIGH confidence via web verification):**

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.agents/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
