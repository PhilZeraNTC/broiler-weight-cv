# Project Research Summary

**Project:** AvianScale CV
**Domain:** Classical Digital Image Processing (PDI — No Deep Learning) + EXIF-Calibrated Tabular Regression + Local Interactive Pitch Web Studio
**Researched:** 2026-11-20
**Confidence:** HIGH

## Executive Summary

AvianScale CV is a local, 100% interpretable computer vision MVP for non-invasive broiler live-weight estimation (116–2,093 g; ~7–29 days, Cobb Avian 48) built exclusively with **classical Digital Image Processing (PDI)** and **EXIF-calibrated tabular regression**, targeting a **2026-11-28 code freeze** and **2026-11-30 academic/product pitch**. Across all four research tracks, consensus is strong: the winning architecture is a **two-plane modular monolith** (`backend/app/{ingestion,pdi,modeling,api}` + `frontend/`) anchored by a **single-source classical PDI engine** (`backend/app/pdi/`) shared verbatim between offline batch extraction (1,738 clean images) and online FastAPI inference (`< 500 ms` CPU SLA). Locking an immutable `PDIConfig` hash and feature ordering inside the serialized `models/weight_regressor.joblib` bundle eliminates training-serving skew while keeping the entire stack CPU-native, lightweight, and fully auditable stage-by-stage ("X-Ray PDI Studio").

The recommended technical baseline pins **Python 3.12** via `uv` (decoupled from the host's rolling Python 3.14.10), **`opencv-python-headless 4.14.x`**, **`scikit-image 0.26.x`**, **`scipy 1.18.x`**, **`pillow 13.x`**, **`pandas 3.1.x` + `pyarrow 26.x`**, **`scikit-learn 1.9.x`**, **`lightgbm 4.7.x`**, and **`fastapi 0.165.x` + `uvicorn`** on the backend, paired with **React 19.3 + TypeScript 7.0 + Vite 8.3 + Tailwind CSS v4.3 + Recharts 3.11 (`react-is@19.3.1`) + Biome 2.6** on the frontend (Node `v24.21.0` via `nvm`). Every backend wheel and frontend peer-dependency graph was live-verified against registries on this host (`7.1 GiB` RAM, 16 logical CPUs). To prevent OOM/latency failures and training-serving discrepancies, we reconcile minor cross-researcher variations upfront into hard engineering decisions: canonical aspect-preserving working resolution of **`768–960 px` (default `768 px`, benchmarked for `< 400 ms` p95 on `4080×1840`)**, offline batch extraction capped at **`n_jobs=4`**, unified Pillow EXIF-first loading (banning raw `cv2.imread`/`cv2.imdecode`), and FastAPI `async def` bounded upload intake offloading CPU work via `await run_in_threadpool(...)` with `cv2.setNumThreads(2)` and `uvicorn --workers 1`.

The primary technical risk is **not** web engineering or model complexity—it is **classical segmentation integrity and optical confounding in Phase 2–3**. Specifically: (1) preventing SF-400 scale LCD digits and white plastic platter from leaking into the bird mask (`PROIBIÇÃO 2`); (2) avoiding white-on-white segmentation collapse on heavy `>1,000 g` birds inside the dorsal white bucket (~351 images); and (3) avoiding a newly uncovered **optical physics double-counting trap** where Xiaomi's `zoomMultiple = 0.60` already encodes the Wide→Ultrawide 35mm-equivalent FOV switch, alongside strong `ultrawide ↔ >1,000 g` confounding (`56–78%` ultrawide above `1 kg`). Mitigating these requires explicit LCD/scale suppression before morphological closing, border-ring scene branching with feather micro-texture + contact-shadow cues, mandatory **empirical verification of optical normalization on the 114 paired dual-lens sessions**, a **4-condition ablation + counterfactual lens-flip test**, and a **dual-bundle `joblib` artifact** serving genuine out-of-fold predictions on curated pitch presets.

---

## Key Findings

### Recommended Stack

All backend Python packages resolve 100% from pure binary wheels (`--only-binary :all:`) under `uv`-managed Python `3.12.14` (avoiding OS-upgrade breakage and `cp314` ABI hazards on the host's `/usr/bin/python3.14`), and all frontend packages resolve cleanly under Node `v24.21.0` (`~/.nvm/versions/node/v24.21.0/bin`). See [STACK.md](STACK.md) for full install scripts and compatibility notes.

**Core technologies (reconciled exact pins):**
- **Python `3.12.14` via `uv` (`0.13.9`)**: Isolated `.python-version = 3.12` workspace at repo root; clean reproducible lockfile (`uv.lock`) for both offline CLI scripts and FastAPI runtime.
- **`opencv-python-headless` `>=4.14.0,<5.0` (`4.14.0.94`) + `scikit-image` `>=0.26.0,<0.27` (`0.26.0`) + `scipy` `>=1.18.0,<2.0` (`1.18.1`) + `pillow` `>=13.0.0,<14.0` (`13.0.0`)**: Headless OpenCV 4.14 avoids Qt/libGL bloat and OpenCV 5 DNN churn; Pillow handles mandatory EXIF extraction (Orientation `274`, `FocalLength` `37386`, Xiaomi `39321` `zoomMultiple`) and `ImageOps.exif_transpose()` before OpenCV processing; `scikit-image` + `scipy` power `regionprops`, exact Euclidean Distance Transform (`distance_transform_edt`), `h_maxima`, and Marker-based Watershed (`skimage.segmentation.watershed`).
- **`pandas` `>=3.1.0,<4.0` (`3.1.1`) + `pyarrow` `>=26.0.0,<27.0` (`26.0.1`)**: Typed Parquet manifests and feature tables (note Pandas 3 global Copy-on-Write—always use `.loc[mask, col] = val`—and Arrow-backed `str` dtype).
- **`scikit-learn` `>=1.9.0,<2.0` (`1.9.1`) + `lightgbm` `>=4.7.0,<5.0` (`4.7.0`) + `joblib` `>=1.6.0,<2.0` (`1.6.0`)**: 5-fold `StratifiedGroupKFold`, `Pipeline`, `TransformedTargetRegressor` (log-target), `Ridge`/`HuberRegressor`/`RandomForestRegressor`/`LGBMRegressor`, and versioned bundle serialization.
- **`fastapi` `>=0.165.0,<1.0` (`0.165.0`) + `uvicorn[standard]` `>=0.54.0,<1.0` (`0.54.0`) + `pydantic` `>=2.14.0,<3.0` + `python-multipart` `>=0.0.34`**: Typed local REST API (`< 500 ms` SLA); dev tooling `pytest 9.2`, `httpx 0.28`, `ruff 0.17`.
- **React `19.3.1` + TypeScript `7.0.2` + Vite `8.3.8` + Tailwind CSS `4.3.5` (`@tailwindcss/vite`) + Recharts `3.11.0` + `react-is@19.3.1` + `lucide-react@1.77.0` + `@biomejs/biome@2.6.8`**: CSS-first Tailwind v4 (`@import "tailwindcss";`, no `tailwind.config.ts` boilerplate), Vite `/api` proxy to `:8000` (with optional FastAPI `StaticFiles` mount for one-command pitch launch), mandatory matching `react-is@19.3.1` for Recharts under React 19, and **Biome 2.6 over ESLint** because `typescript-eslint@8.73.0` conflicts with `typescript@7.0.2` (`<6.1.0` cap).

**Strictly excluded (`PROIBIÇÃO 1–3`):** PyTorch, TensorFlow, YOLO, SAM, ViTs, ONNX/OCR libs, GUI `opencv-python`, `train_test_split`, and raw `cv2.imread`/`cv2.imdecode` calls.

### Expected Features

Scope is tightly closed around `REQ-01` through `REQ-15` and `PROIBIÇÃO 1–5` to protect the 8-day delivery schedule. See [FEATURES.md](FEATURES.md) for the full prioritization matrix and competitor benchmark.

**Must have (table stakes — P1 for 2026-11-28 freeze):**
- **Forensic Data Ingestor & Quarantine (`REQ-01`)** — Reproduces Audit §9 verbatim (`1,753` raw → `1,738` clean + `15` quarantined; `367→267 g`; nested `171/173`, `190/191`, `190/193`; `1340_`; CP866 reversal on `547`/`562`; Arabic digits `٠-٩`; reconstructed `capture_dt`; EXIF focal/zoom).
- **Anti-Leakage 5-Fold `session_id` Split (`REQ-02`)** — Groups sequential folder captures (`≤ 10 min`) into `188` clean sessions and assigns 5-fold `StratifiedGroupKFold` across `<500 g`, `500–1000 g`, `>1000 g`.
- **Classical Aspect-Preserving Preprocessing & Anti-LCD Segmentation (`REQ-03`, `REQ-04`)** — EXIF transpose + aspect-preserving downscale + Bilateral/Median + CLAHE Lab + multi-space Lab/HSV & texture/shadow segmentation with hard SF-400 LCD/platter removal.
- **Mask-Gated Morphometric, Chromatic & EXIF Optical Features (`REQ-05`)** — Extracted strictly inside `bird_mask == 255`: geometry (`area`, `perimeter`, ellipse axes, eccentricity, convex area, solidity, circularity), scale-invariant plumage cues (yellow-down vs. white-plumage ratio, Lab/HSV stats), and empirically verified EXIF optical normalization.
- **Interpretable Regression, Optical Ablation & Serialization (`REQ-08`, `REQ-09`, `REQ-10`)** — 5-fold comparison of log-target Ridge/Huber, Random Forest, and LightGBM; `evaluation_report.json` stratified by weight band, lens, and scenario + WITH vs. WITHOUT EXIF ablation; versioned `weight_regressor.joblib` with per-band error intervals.
- **FastAPI `< 500 ms` Backend & 4-Stage "X-Ray" PDI Studio (`REQ-06`, `REQ-11`, `REQ-12`)** — Local REST API + React SPA showing Original, CLAHE, Clean Binary Mask (no LCD), and Fitted Ellipse Overlay alongside features, estimated weight + error band, and Cobb Avian 48 age stage.

**Should have (pitch differentiators — P1/P2 in v1):**
- **Curated Out-of-Fold Pitch Presets (`REQ-13`)** — 4–6 held-out validation samples (`~180 g`, `~550 g`, `~1,600 g`, hard white-on-white bucket `>1,000 g`, optional dual-lens pair) showing live Ground Truth vs. Predicted & `%` error using strictly out-of-fold models.
- **Interactive Marker-Based Watershed Multi-Bird Tab & Ceiling-Camera Roadmap (`REQ-07`, `REQ-14`)** — Turns quarantined folders `417`, `757`, `823`, `1520` (+ custom uploads) into live multi-bird separation with counts and per-bird areas, paired with a conceptual MVP-vs-overhead-barn architectural comparison.
- **Live Simulated Flock Average & Uniformity Tracker (`REQ-12`)** — Client-side accumulator (`sessionStorage`-backed) computing live mean weight, SD, CV%, and bird count as samples are inspected.
- **Forensic Data Audit & Optical Ablation Dashboard (`REQ-15`)** — Recharts Actual vs. Predicted scatter (`<ReferenceLine segment={...} />`), optical ablation bar chart, stratified error tables, and forensic curation funnel (`1,753 → 1,738`).
- **Interactive Optical "What-If" Lens/Zoom Override & Chromatic Feathering Bar (P2 polish inside Phase 4)** — Live manual toggle demonstrating optical calibration physics + missing-EXIF graceful fallback.

**Defer (v1.x / v2+ — explicitly out of v1 scope):**
- Multi-frame video object tracking, barn density heatmaps, immobile/sick-bird alerts (`PROIBIÇÃO 5` — conceptual roadmap visual in Tab 2 only).
- Post-pitch v1.x hardening: offline automated LCD OCR label audit, bucket-rim `mm/pixel` Hough calibration, Laplacian motion-blur quality gate, leave-one-date-out diagnostic CV, batch folder PDF/CSV export, and late-cycle (`30–45 d` / `2.1–3.2 kg`) data expansion.

### Architecture Approach

AvianScale CV uses a **two-plane modular monolith** separating reproducible offline CLI pipelines (Phases 1–3) from an online FastAPI + React/Vite serving plane (Phase 4) through immutable Parquet/JSON/`joblib` contracts and one shared PDI package. See [ARCHITECTURE.md](ARCHITECTURE.md) for complete directory layout, sequence flows, and schemas.

**Major components & reconciled operational standards:**
1. **`backend/app/ingestion/` (Phase 1)** — Deterministic recursive crawl, encoding recovery, quarantine split, `367→267 g` fix, `session_id` clustering, and 5-fold `StratifiedGroupKFold` writing `data/processed/manifest_clean.parquet`, `manifest_quarantine.parquet`, and git-tracked `models/dataset_audit_summary.json`.
2. **`backend/app/pdi/` (Phase 2 — Single Source of Truth)** — Shared `io_exif.py`, `preprocess.py`, `lcd_suppression.py`, `segmentation.py`, `features.py`, `xray.py`, `pipeline.py` (locked by SHA-256 `PDIConfig`), plus isolated `watershed.py` (`scipy.ndimage.distance_transform_edt` → Gaussian smooth → `skimage.morphology.h_maxima` → `peak_local_max` → `skimage.segmentation.watershed`).
   - *Reconciled resolution & concurrency:* Record oriented `(orig_w, orig_h)` **before** `PIL.Image.draft()`, apply `ImageOps.exif_transpose()`, downscale preserving exact aspect ratio via `cv2.INTER_AREA` to **`768–960 px` long edge (default `768 px`, benchmarked `< 400 ms` p95 on `4080×1840`)**, and back-project linear/areal features via `scale_factor` alongside resolution-agnostic `area_frac`. Cap offline batch extraction (`backend/scripts/extract_features.py`) at **`n_jobs=4` (`max_workers=4`) with `cv2.setNumThreads(1)` per worker** to respect the `7.1 GiB` RAM ceiling while finishing 1,738 images in `< 60 s`.
3. **`backend/app/modeling/` (Phase 3)** — Trains log-target tabular pipelines (with importable wrapper classes for optional fold-internal Duan smearing and capped inverse session/stratum `sample_weight`), executes the 4-way optical ablation + lens-flip audit, and packs a **Dual-Bundle `models/weight_regressor.joblib`** containing `production_pipeline` (100% clean data for uploads), `fold_pipelines[0..4]` (for 100% honest out-of-fold live preset scoring), locked `PDIConfig` hash, `feature_names`, and per-band expected error intervals—plus git-tracked `models/evaluation_report.json` and `models/presets_catalog.json` (zero `.jpg` files committed to `frontend/public/`).
4. **`backend/app/api/` + `frontend/` (Phase 4)** — FastAPI loads the bundle, JSON reports, and ~8–12 curated preset raw bytes/thumbnails once in `lifespan` (`app.state`, `< 30 MB` RAM), sets `cv2.setNumThreads(2)`, and runs `uvicorn --workers 1`.
   - *Reconciled FastAPI concurrency:* Upload endpoints (`POST /api/predict`, `POST /api/watershed-demo`) use **`async def`** strictly for bounded non-blocking `await file.read()` (≤ 15 MB), immediately dispatching synchronous CPU-bound OpenCV/joblib work via **`await run_in_threadpool(...)`**. X-ray inspection stages are rendered at `≤ 640–768 px` as JPEG (`Q=82`) + compact 1-ch binary mask PNG (`< 250 KB` total JSON, `< 20 ms` encode). React 19 SPA uses typed native `fetch` (no Redux/TanStack overhead) via Vite `/api` proxy across 3 clean pitch tabs.

### Critical Pitfalls

The dataset's forensic audit and classical CV physics expose 6 high-impact failure modes that must be engineered against from day one. See [PITFALLS.md](PITFALLS.md) for warning signs, checklists, and recovery paths.

1. **Optical Physics Double-Counting (`focal × zoom`) & Lens×Weight Confounding (`Pitfall 3`)** — On the Redmi 10, `zoomMultiple = 0.60` already encodes the 35mm-equivalent FOV switch between Wide (`4.25 mm`, `4080×1840`) and Ultrawide (`1.65 mm`, `3264×1472`), so naive $A_{px} / (f_{mm}\cdot\text{zoom})^2$ can over-correct by `~18×` while raw focal/resolution features let trees cheat via `ultrawide ⇒ >1,000 g` (`56–78%` prevalence). **Avoidance:** Empirically compare `area_frac / zoom_multiple**2` vs. `A_px / (f_mm * zoom)**2` on the **114 paired dual-lens sessions** in Phase 2 (target median Wide/UW ratio `≈ 1.0`), ban `oriented_w`/`oriented_h`/`megapixels` as features, and run a 4-condition ablation + counterfactual lens-flip test in Phase 3.
2. **SF-400 LCD & White Platter Visual Leakage (`Pitfall 1`, `PROIBIÇÃO 2`)** — Naive Otsu thresholding merges white birds with the white SF-400 platter (`1.5×–3×` area inflation) or leaks dark LCD digits/bezel. **Avoidance:** Dedicated `lcd_suppression.py` hard-zeros dark neutral-chroma rectangular LCD bezels and smooth specular plastic *before* morphological closing; `features.py` indexes strictly `img[bird_mask == 255]` (verified by an automated pixel-perturbation invariance test outside the mask).
3. **White-on-White Bucket Segmentation Collapse on `>1,000 g` Broilers (`Pitfall 2`)** — Heavy white broilers inside the white bucket (~351 images) fail luminance-only thresholding and trigger `cv2.fitEllipse` `< 5 points` crashes or wall flood-fill. **Avoidance:** Deterministic outer-ring photometry detector activates a bucket-aware branch combining Lab chroma warmth (gating Hue whenever `S < 25`), CLAHE feather micro-texture/Scharr energy, contact-shadow Black-Hat rim barriers, border-spill rejection, and a degenerate-mask guard/fallback (`area_frac ∈ [0.015, 0.72]`).
4. **Watershed Over-Segmentation & Red-Feeder False Positives (`Pitfall 4`)** — Hardcoded `min_distance` and raw `peak_local_max` shatter 1 bird into 4–6 shards or count the red plastic feeder in `417`. **Avoidance:** Suppress high-chroma red feeder pixels (`Lab a* > 152`), use scale-adaptive `Gaussian` smoothing + `skimage.morphology.h_maxima(h ≈ 0.25 * dist.max())` + relative `min_distance` (never legacy `indices=False`), drop tiny spurious basins, and verify exact audited counts across all 13 images (`417`:2, `757`:2, `823`:4, `1520`:2).
5. **`cv2.imdecode` Orientation Desync, Target Imbalance & Log-Bias (`Pitfalls 6 & 7`)** — `81.86%` of photos have `EXIF Orientation != 1`; `cv2.imdecode` ignores EXIF while `cv2.imread` applies it, causing live uploads to arrive rotated 90°–270°. Meanwhile `70%` of images are `≤500 g` with an `801–1,216 g` desert, where naive `exp(ŷ)` introduces negative Jensen bias and fine bins trigger `StratifiedGroupKFold` empty-stratum warnings. **Avoidance:** Ban `cv2.imread`/`cv2.imdecode` repo-wide in favor of one Pillow EXIF-first loader; stratify 5-fold CV strictly on the 3 coarse PRD weight bands (`≥ 4` sessions `>1,000 g` per fold); evaluate fold-internal Duan smearing, capped inverse session/stratum `sample_weight`, and importable serialization wrappers.
6. **Latency SLA Breach, Host Python 3.14 / 7.1 GiB RAM & Stripped-EXIF Live Demo Failures (`Pitfalls 5 & 8`)** — Native `4080×1840` PDI + PNG base64 takes `1.5–4.5 s`; `n_jobs=-1` risks OOM on `7.1 GiB` RAM; WhatsApp/PNG pitch uploads strip EXIF or include RGBA alpha. **Avoidance:** Pin Python 3.12 via `uv`, cap batch workers at `n_jobs=4`, downscale to `768 px` + JPEG X-ray artifacts (`< 250 KB`, p95 `< 400 ms`), normalize uploads with `.convert("RGB")`, gracefully default missing EXIF (`4.25 mm / 1.0×` + prominent UI warning pill + instant manual lens override), and validate with a pre-pitch smoke script.

---

## Implications for Roadmap

The research directly confirms a strict **4-phase coarse DAG** matching `PRD.md` and the 8-day timeline (**freeze 2026-11-28, pitch 2026-11-30**). No phase requires refactoring an earlier phase's module boundaries.

### Phase 1: Environment Bootstrap, Forensic Data Curation & Anti-Leakage Split
**Rationale:** Every downstream feature table, CV fold, out-of-fold preset, and audit chart depends on an immutable, exact-count manifest and a leak-free `session_id` split. Pinning Python 3.12 via `uv` first eliminates host environment hazards before any CV/ML code runs.
**Delivers:** Root `.python-version` (`3.12`) + `pyproject.toml` + `uv.lock`; `backend/app/ingestion/` + `backend/app/pdi/io_exif.py` (shared Pillow EXIF/Xiaomi `39321` parser); `backend/scripts/build_manifest.py`; `data/processed/manifest_clean.parquet` (1,738 imgs, 160 weights, 188 sessions, 5 folds), `manifest_quarantine.parquet` (15 imgs), `models/dataset_audit_summary.json`; exact-count & zero-session-overlap pytest suite.
**Addresses:** `REQ-01`, `REQ-02`.
**Uses:** `uv` + Python 3.12, `pillow`, `pandas 3.1`, `pyarrow 26`, `scikit-learn 1.9` (`StratifiedGroupKFold`), `pytest`, `ruff`.
**Avoids:** Pitfall 8a (Python 3.14 wheels), Pitfall 7a (nested dirs, `1340_`, CP866 2024 `mtime`, Arabic digits, `367→267 g`), Pitfall 6a (`StratifiedGroupKFold` empty strata & burst split leakage), `PROIBIÇÃO 3 & 4`.

### Phase 2: Classical PDI Engine, Optical Calibration & Multi-Bird Watershed
**Rationale:** This is the **critical-path technical bottleneck**. Segmentation mask quality (especially Scenario A LCD exclusion and Scenario B white-on-white heavy buckets) and empirical optical normalization must be proven visually and quantitatively before modeling begins. Building `backend/app/pdi/` as the single shared library now guarantees zero training-serving skew in Phase 4.
**Delivers:** Complete `backend/app/pdi/` (`config.py`, `io_exif.py`, `preprocess.py`, `lcd_suppression.py`, `segmentation.py`, `features.py`, `xray.py`, `pipeline.py`, isolated `watershed.py`); empirical paired dual-lens optical calibration verification on the 114 mixed-lens folders; `backend/scripts/extract_features.py` (`n_jobs=4`, `cv2.setNumThreads(1)`) generating `data/processed/features_clean.parquet`; anti-leakage pixel-perturbation tests, EXIF `Path`-vs-`bytes` parity tests, mask-sanity gate (`< 2%` fallback), and Watershed verification on all 13 images (`417`, `757`, `823`, `1520`).
**Addresses:** `REQ-03`, `REQ-04`, `REQ-05`, `REQ-06`, `REQ-07`.
**Uses:** `opencv-python-headless 4.14`, `scikit-image 0.26`, `scipy 1.18`, `numpy 2.5`, `pillow 13`, `joblib` (`n_jobs=4`).
**Avoids:** Pitfall 1 (SF-400 LCD/platter leakage), Pitfall 2 (white-on-white bucket collapse & `fitEllipse` crash), Pitfall 3a (`focal × zoom` double-counting), Pitfall 4 (Watershed over-segmentation & red-feeder false positive), Pitfall 5a (native 7.5 MP latency & 7.1 GiB RAM OOM), Pitfall 7b (`cv2.imdecode` desync & BGR/Lab scaling bugs), `PROIBIÇÃO 1 & 2`.

### Phase 3: Interpretable Tabular Regression, Optical Ablation & Production Bundling
**Rationale:** With clean, mask-gated features and locked folds ready, Phase 3 benchmarks linear/allometric and tree regressors rigorously, quantifies true EXIF optical gain without confounding shortcuts, selects honest out-of-fold pitch presets, and locks all serving metadata into immutable artifacts before UI work starts.
**Delivers:** `backend/app/modeling/` (`datasets.py`, `models.py`, `wrappers.py` with importable log/Duan classes, `evaluation.py`, `age_stage.py`, `artifacts.py`); `backend/scripts/train_and_evaluate.py`; `models/weight_regressor.joblib` (Dual-Bundle: `production_pipeline` + 5 `fold_pipelines` + locked `PDIConfig` hash + `feature_names` + per-band error intervals); git-tracked `models/evaluation_report.json` (4-condition optical ablation, within-session Wide/UW consistency, lens-flip test, stratified MAE/RMSE/MAPE/$R^2$/MBE) and `models/presets_catalog.json`; clean-process `joblib.load()` + zero-leakage tests.
**Addresses:** `REQ-08`, `REQ-09`, `REQ-10`.
**Uses:** `scikit-learn 1.9` (`Pipeline`, `TransformedTargetRegressor`, `Ridge`, `HuberRegressor`, `RandomForestRegressor`), `lightgbm 4.7`, `joblib 1.6`.
**Avoids:** Pitfall 3b (`ultrawide ⇒ heavy` shortcut & resolution leakage), Pitfall 6b (target skew, Jensen log-retransformation bias, `__main__`/lambda `PicklingError`), in-sample preset overfitting.

### Phase 4: FastAPI Local Backend & Interactive Pitch Web Studio
**Rationale:** Freezing backend Pydantic schemas and frontend TypeScript mirrors together on day 1 of Phase 4 allows rapid, bug-free construction of the 3 pitch views directly backed by real OOF models and live `< 400 ms` PDI inference, culminating in a pre-pitch end-to-end smoke gate.
**Delivers:** `backend/app/api/` (`main.py` with `lifespan` cache + `cv2.setNumThreads(2)`, `schemas.py`, `routers/{health,predict,watershed,presets,metrics}.py`, `services/`) using `async def` bounded intake + `run_in_threadpool` under `uvicorn --workers 1`; `frontend/` React 19 + TypeScript 7 + Vite 8 + Tailwind v4 + Recharts 3 SPA with 3 pitch tabs (Tab 1: X-Ray PDI Studio + Curated OOF Presets + Optical Override + Simulated Flock Batch Tracker; Tab 2: Interactive Watershed Multi-Bird + Ceiling-Camera Roadmap; Tab 3: Stratified Validation, Optical Ablation & Forensic Audit Dashboard); single-command launch script (sourcing NVM + optional static mount) + automated pre-pitch smoke & latency benchmark (`p95 < 400 ms`, stripped-EXIF JPEG, RGBA PNG).
**Addresses:** `REQ-11`, `REQ-12`, `REQ-13`, `REQ-14`, `REQ-15`.
**Uses:** `fastapi 0.165`, `uvicorn 0.54`, `pydantic 2.14`, `python-multipart`, `react 19.3.1`, `vite 8.3.8`, `tailwindcss 4.3.5`, `recharts 3.11.0`, `react-is 19.3.1`, `lucide-react 1.77`, `typescript 7.0.2`, `@biomejs/biome 2.6.8`.
**Avoids:** Pitfall 5b (`> 500 ms` latency & PNG base64 bloat), Pitfall 8b (stripped EXIF crashes, RGBA PNGs, NVM shell PATH, relative `cwd` paths, CORS, path traversal), Anti-Pattern 6 (committing preset `.jpg`s), `PROIBIÇÃO 4 & 5`.

### Phase Ordering Rationale

- **Strict Data-to-UI Dependency Chain:** `manifest_clean.parquet` (Phase 1) → `features_clean.parquet` & locked `PDIConfig` (Phase 2) → `weight_regressor.joblib` + `evaluation_report.json` + `presets_catalog.json` (Phase 3) → FastAPI + React SPA (Phase 4). No downstream work can be scientifically verified until upstream contracts pass automated assertions.
- **Risk Front-Loading:** Phase 2 isolates the hardest unsolved computer vision work (anti-LCD segmentation, white-on-white heavy bucket masking, dual-lens physics normalization, and Watershed tuning) while leaving ample runway before the 2026-11-28 freeze—and its 4-worker parallel feature runner keeps Phase 2→3 iteration loops under `60 seconds`.
- **Zero-Skew Reuse:** Designing `backend/app/pdi/` as an importable library in Phase 2 (including `io_exif.py` shared with Phase 1) and serializing `PDIConfig.config_hash()` into Phase 3's bundle guarantees Phase 4's live web UI executes the exact audited algorithm bit-for-bit.

### Research Flags

Phases needing deeper research / empirical prototyping during planning:
- **Phase 2 (HIGH — Empirical PDI & Optical Spike Required):** Classical segmentation thresholds across 3 heterogeneous visual domains (SF-400 scale/LCD exclusion, white-on-white `>1,000 g` bucket birds, straw bedding), empirical paired verification of `area_frac / zoom_multiple**2` vs. `A_px / (f_mm * zoom)**2` on the 114 mixed-lens folders, and scale-adaptive `h_maxima` Watershed tuning on `417`/`757`/`823`/`1520` cannot be guessed purely from literature—plan an explicit visual/quantitative inspection checkpoint on sample galleries before finalizing batch extraction.
- **Phase 3 (LOW-MEDIUM — Targeted Experimental Validation):** Standard `scikit-learn`/`LightGBM` pipeline mechanics, but planning should explicitly specify the 4-way optical ablation matrix, counterfactual lens-flip acceptance check, and CV comparison of Duan smearing + capped `sample_weight` vs. unweighted log-target baselines.

Phases with standard, well-documented patterns (skip `/gsd-research-phase`):
- **Phase 1:** Forensic Audit §9 already provides exact deterministic rules, verified file counts, and verified EXIF tag schemas (`274`, `37386`, `39321`); go straight to plan & execute with exact-count unit tests.
- **Phase 4:** FastAPI `lifespan` + `run_in_threadpool` + React 19 / Vite 8 / Tailwind v4 / Recharts 3 patterns and exact compatible versions are already live-verified; go straight to schema contract locking and implementation.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | Every Python wheel (`--only-binary :all:` on Python 3.12) and npm package + `peerDependencies` map was verified live on this exact workstation (`uv 0.13.9`, Node `24.21.0`, `7.1 GiB` RAM), catching the `typescript-eslint` vs. TS 7 conflict and Recharts `react-is` requirement upfront. |
| Features | **HIGH** | Closed-scope PRD (`REQ-01`–`REQ-15`, `PROIBIÇÃO 1–5`) and full forensic audit of all 1,753 raw images leave zero scope ambiguity; cross-checked against classical PLF literature (Mollah 2010, Mortensen 2016, Amraei 2017) and commercial benchmarks. |
| Architecture | **HIGH** | Clean 4-domain modular monolith with verified FastAPI/OpenCV/Pillow patterns (`lifespan` + `run_in_threadpool`, `PDIConfig` hash lock, dual-bundle OOF preset routing, zero committed raw images). Minor cross-doc differences (resolution, worker cap, Watershed library, sync vs. async) cleanly reconciled above. |
| Pitfalls | **HIGH** | Grounded in full dataset forensic counts, local OS probe (`Python 3.14.10`, `7.1 GiB` RAM, NVM), and verified OpenCV/Pillow/scikit-image/scikit-learn edge cases—uncovering the critical `zoomMultiple` double-counting risk and `cv2.imdecode` EXIF desync before a single line of code is written. |

**Overall confidence:** **HIGH**

### Gaps to Address

- **Exact Optical Normalization Constant (`Phase 2` empirical resolution):** Whether `area_frac / zoom_multiple**2` alone, `A_px / (f_mm * zoom)**2`, or an empirically calibrated Wide/Ultrawide ratio best collapses within-session dual-lens variance must be measured directly on the 114 paired dual-lens sessions once Phase 2 segmentation runs—do not hardcode the final formula until that paired ratio check is logged.
- **Achievable Classical Segmentation & Regression Accuracy Floor (`Phase 2–3` empirical resolution):** Because handheld camera-to-bird height $Z$ is unrecorded and slow shutters (`1/20–1/50 s`) induce motion blur, the exact achievable global and `>1,000 g` bucket MAPE/MAE cannot be predicted in advance. Mitigate by building strong scale-invariant plumage/shape features, validating mask health across all 1,738 images, and transparently reporting stratified metrics + the `801–1,216 g` dataset gap in the Audit Dashboard.
- **Exact Working Resolution (`768 px` vs. `960 px`) Latency Trade-off (`Phase 2 & 4` benchmark):** Both preserve exact `20:9` aspect ratio and yield strong morphometric precision; start at `768 px` and benchmark end-to-end `/api/predict` p95 latency on 10 real `4080×1840` JPEGs (`< 400 ms` target), scaling up to `960 px` only if headroom comfortably exceeds `150 ms` and segmentation quality measurably improves.
- **Remaining Unaudited Minor Label Typos Beyond `367→267 g` (`Known Ceiling`):** Full-dataset automated LCD OCR auditing is intentionally deferred to v1.x (and excluded from inference per `PROIBIÇÃO 2`); document the known aleatoric noise floor (`±3 g` scale variance + possible minor typos) explicitly on the Audit tab.

---

## Sources

### Primary (HIGH confidence)
- [PRD.md](../../PRD.md) & [PROJECT.md](../PROJECT.md) — Functional requirements (`REQ-01`–`REQ-15`), hard prohibitions (`PROIBIÇÃO 1–5`), `< 500 ms` SLA, and 2026-11-28 / 2026-11-30 milestones.
- [DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md](../../docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md) — Complete forensic audit of all 1,753 images (Abuhussein et al., Mendeley Data DOI `10.17632/zrs8kk9dvr.1`, CC BY 4.0): exact counts, Xiaomi EXIF tag `39321`, `81.86%` EXIF orientation, CP866 mojibake, quarantine rules, and lens×weight distribution.
- Local workstation & live registry verification ([STACK.md](STACK.md), [PITFALLS.md](PITFALLS.md)) — `uv pip compile --python-version 3.12 --only-binary :all:`, `npm view` peer checks under Node `v24.21.0`, and host probe (`7.1 GiB` RAM, 16 CPUs, system `Python 3.14.10`, `/usr/bin/uv 0.13.9`).
- Detailed domain research documents: [STACK.md](STACK.md), [FEATURES.md](FEATURES.md), [ARCHITECTURE.md](ARCHITECTURE.md), [PITFALLS.md](PITFALLS.md).

### Secondary (MEDIUM confidence)
- Official library documentation & ecosystem release notes — OpenCV 4.14 vs. 5.x, `scikit-image 0.26` (`regionprops`, `h_maxima`, `peak_local_max`, `watershed`), Pillow (`Image.draft`, `ImageOps.exif_transpose`), `pandas 3.1` (Copy-on-Write, Arrow `str` dtype), `scikit-learn 1.9` (`StratifiedGroupKFold`, `TransformedTargetRegressor`), `lightgbm 4.7`, FastAPI (`lifespan`, `run_in_threadpool`), Vite 8, Tailwind CSS v4 (`@tailwindcss/vite`), Recharts 3 (`react-is`, `ReferenceLine segment`), TypeScript 7 vs. Biome 2.6.
- Seminal classical PLF poultry CV literature — Mollah et al. (2010), Mortensen et al. (2016), Amraei et al. (2017/2018), Duan (1983) smearing estimator.

### Tertiary (LOW confidence)
- Commercial PLF poultry benchmark positioning (Big Dutchman *EasyCheck/Swing20*, Fancom *eYeNamic*, Beakbook, PLF Agritech *Weight-Detect™*) — used purely for competitive pitch framing in [FEATURES.md](FEATURES.md).

---
*Research completed: 2026-11-20*
*Ready for roadmap: yes*
