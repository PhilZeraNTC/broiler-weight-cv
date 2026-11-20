# Architecture Research

**Domain:** Classical Digital Image Processing (PDI) + EXIF-Calibrated Tabular Regression + Local FastAPI/React Pitch Web App (AvianScale CV)
**Researched:** 2026-11-20
**Confidence:** HIGH

## Standard Architecture

AvianScale CV is structured as a **two-plane modular monolith** with a **single-source classical PDI core** shared verbatim between offline batch processing (1,738 images) and online single-image inference (`< 500 ms` CPU). The backend (`backend/app/`) cleanly isolates four vertical domains (`ingestion`, `pdi`, `modeling`, `api`), while deterministic serialized artifacts (`data/processed/*.parquet`, `models/*.joblib`, `models/*.json`) act as immutable contracts between offline training (Phases 1–3) and online web serving (Phase 4).

### System Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PRESENTATION PLANE (React + TS + Vite)                    │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐ ┌────────────┐ │
│  │ Inference X-Ray  │ │ Curated Presets  │ │  Watershed & │ │ Metrics &  │ │
│  │ Studio + Batch   │ │ Gallery (GT vs   │ │ Future Vision│ │ Audit      │ │
│  │ Flock Simulator  │ │ Live Pred)       │ │ Multi-Bird   │ │ Dashboard  │ │
│  └────────┬─────────┘ └────────┬─────────┘ └──────┬───────┘ └─────┬──────┘ │
└───────────┼────────────────────┼──────────────────┼───────────────┼────────┘
            │  HTTP/JSON + multipart/form-data (Vite proxy /api -> :8000)
┌───────────┴────────────────────┴──────────────────┴───────────────┴────────┐
│                     ONLINE SERVING PLANE (FastAPI / Uvicorn)                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Lifespan Cache (app.state): weight_regressor.joblib, PDIConfig lock, │  │
│  │ evaluation_report.json, audit_summary.json, cached preset bytes      │  │
│  └───────────────┬───────────────────────────────────┬──────────────────┘  │
│  ┌───────────────┴──────────────┐    ┌───────────────┴──────────────────┐  │
│  │ /api/predict  /api/presets   │    │ /api/watershed-demo /api/metrics │  │
│  │ (async intake -> threadpool) │    │ (isolated multi-bird + reports)  │  │
│  └───────────────┬──────────────┘    └───────────────┬──────────────────┘  │
└──────────────────┼───────────────────────────────────┼─────────────────────┘
                   │ imports verbatim (0 skew)         │ imports verbatim
┌──────────────────┴───────────────────────────────────┴─────────────────────┐
│               SHARED CLASSICAL PDI ENGINE (backend/app/pdi/)               │
│  ┌────────────────────────────────────┐   ┌─────────────────────────────┐  │
│  │ Single-Bird Pipeline               │   │ Dedicated Watershed Module  │  │
│  │ 1. EXIF Transpose + Xiaomi 0x9999  │   │ 1. Shared EXIF + Preprocess │  │
│  │ 2. Canonical Downscale (960px)     │   │ 2. Multi-Bird Union Mask    │  │
│  │ 3. Bilateral + Lab CLAHE           │   │ 3. Distance Transform +     │  │
│  │ 4. SF-400 LCD/Scale Suppression    │   │    Smoothed Local Maxima    │  │
│  │ 5. Multi-Space Lab/HSV + Morph     │   │ 4. Marker-Based Watershed   │  │
│  │ 6. Mask-Gated Features (NO LCD!)   │   │ 5. Basins Overlay + Counts  │  │
│  │ 7. Optional 4-Stage X-Ray Encoder  │   │    & Per-Bird Areas         │  │
│  └──────────────────▲─────────────────┘   └─────────────────────────────┘  │
└─────────────────────┼──────────────────────────────────────────────────────┘
                      │ called in parallel (return_xray=False)
┌─────────────────────┴──────────────────────────────────────────────────────┐
│            OFFLINE DATA & MODELING PLANE (CLI Scripts — Phases 1-3)        │
│  ┌──────────────────────┐   ┌───────────────────────┐   ┌───────────────┐  │
│  │ Forensic Ingestor    │──▶│ Batch Feature Runner  │──▶│ 5-Fold CV &   │  │
│  │ (backend/app/        │   │ (ProcessPoolExecutor, │   │ Optical       │  │
│  │  ingestion/)         │   │  1,738 clean images)  │   │ Ablation      │  │
│  └──────────┬───────────┘   └───────────┬───────────┘   └───────┬───────┘  │
└─────────────┼───────────────────────────┼───────────────────────┼──────────┘
              ▼                           ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LOCAL ARTIFACT & DATA STORE                         │
│  [Git-Ignored Heavy Data]                [Git-Tracked Lightweight Reports]  │
│  • data/raw/Broiler weights... (1,753)   • models/evaluation_report.json    │
│  • data/processed/manifest_clean.parquet • models/dataset_audit_summary.json│
│  • data/processed/manifest_quarantine... • models/presets_catalog.json      │
│  • data/processed/features_clean.parquet                                    │
│  • models/weight_regressor.joblib (bundle: prod + 5 fold pipelines + hash)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **`ingestion`** (`backend/app/ingestion/`) | Deterministic forensic crawl of `data/raw/`, level-2 subfolder capture (`171/173`, `190/191`, `190/193`), folder sanitization (`1340_`), CP866 mojibake reversal (`547/`, `562/`), Arabic-Indic digit normalization (`٠-٩`), `capture_dt` reconstruction (`mtime - 14401 s`), quarantine split (`15` imgs), `367→267 g` LCD fix, `session_id` clustering (>10 min gap), and 5-fold `StratifiedGroupKFold`. | Pure Python (`os.walk`, `re`, `Pillow`, `pandas`, `pyarrow`, `scikit-learn`) invoked via `backend/scripts/build_manifest.py`. |
| **`pdi` (Single-Bird)** (`backend/app/pdi/`) | Single source of truth for EXIF orientation transpose, Xiaomi tag `39321` (`zoomMultiple`) + `FocalLength` extraction, canonical aspect-preserving downscale (`960 px` long edge), Bilateral + CIE Lab CLAHE preprocessing, **SF-400 LCD/scale exclusion**, multi-space Lab/HSV segmentation, strictly mask-gated morphometric/chromatic/optical feature extraction, and optional 4-stage X-ray JPEG/base64 rendering. | `opencv-python-headless`, `scikit-image`, `numpy`, `Pillow`, typed via Pydantic/dataclass `PDIConfig` with SHA-256 hash. |
| **`pdi.watershed`** (`backend/app/pdi/watershed.py`) | Completely isolated multi-bird instance separation using `Distance Transform` + smoothed local maxima (`peak_local_max`) + `cv2.watershed` for the 4 quarantined multi-bird folders (`417`, `757`, `823`, `1520`) or user uploads. Returns count, per-bird areas, and basin/boundary overlays. | Reuses `pdi.io_exif` & `pdi.preprocess`; never feeds into single-bird regression training. |
| **`modeling`** (`backend/app/modeling/`) | Joins `manifest_clean.parquet` + `features_clean.parquet`, verifies zero `session_id` fold leakage, trains & compares interpretable/tabular regressors (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog`), executes optical EXIF ablation (`with` vs `without`), computes stratified metrics (`<500 g`, `500–1000 g`, `>1000 g`; Wide vs Ultrawide; Bucket vs Scale/Litter), and exports the versioned `joblib` bundle + JSON reports. | `scikit-learn` (`Pipeline`, `TransformedTargetRegressor`), `lightgbm`, `joblib`, invoked via `backend/scripts/train_and_evaluate.py`. |
| **`api`** (`backend/app/api/`) | Local FastAPI service exposing `/api/predict`, `/api/watershed-demo`, `/api/presets`, `/api/metrics`, and `/api/health`. Loads model bundle, metrics JSON, and curated preset image bytes into `app.state` once via `lifespan`; validates requests with Pydantic v2; dispatches CPU-bound OpenCV/joblib work via `run_in_threadpool` to maintain `< 500 ms` latency without blocking the event loop. | `fastapi`, `uvicorn`, `pydantic`, `python-multipart`, thin `routers/` delegating to `services/`. |
| **`frontend`** (`frontend/src/`) | Local pitch SPA providing (1) Inference X-Ray Studio + Simulated Flock Batch Accumulator, (2) Curated Real Presets Gallery (Ground Truth vs Live Predicted), (3) Interactive Watershed & Ceiling-Camera Roadmap tab, and (4) Stratified Metrics & Forensic Audit Dashboard. | React 18/19 + TypeScript + Vite + Tailwind CSS + Recharts (for Actual vs Predicted scatter & ablation charts). |

## Recommended Project Structure

```text
broiler-weight-cv/
├── .gitignore                                 # Ignores data/raw/, data/processed/, models/*.joblib, *.jpg/*.png
├── PRD.md
├── docs/
│   └── dataset/
│       └── DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md
├── data/
│   ├── raw/                                   # [GIT-IGNORED] Read-only 1,753 raw JPEGs (3.54 GiB)
│   │   └── Broiler weights of Cobb Avian/
│   └── processed/                             # [GIT-IGNORED] Generated by Phases 1-2 CLI scripts
│       ├── manifest_clean.parquet             # 1,738 rows + session_id + fold (0..4) + scene_hint
│       ├── manifest_quarantine.parquet        # 15 rows (13 multi_bird + 2 severe_mislabel)
│       └── features_clean.parquet             # 1,738 rows x PDI/optical feature vector + QC flags
├── models/
│   ├── weight_regressor.joblib                # [GIT-IGNORED] Versioned bundle (prod + 5 fold pipelines + hash)
│   ├── evaluation_report.json                 # [GIT-TRACKED] 5-fold stratified metrics, ablation, OOF points
│   ├── dataset_audit_summary.json             # [GIT-TRACKED] Forensic funnel counts for Audit Dashboard
│   └── presets_catalog.json                   # [GIT-TRACKED] Curated rel_paths + GT + fold metadata (no JPGs!)
├── backend/
│   ├── pyproject.toml                         # Dependencies, ruff/pytest config, Python >=3.11
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                          # Path settings (DATA_RAW_DIR, MODELS_DIR, API defaults)
│   │   ├── ingestion/                         # PHASE 1: Forensic data curation
│   │   │   ├── __init__.py
│   │   │   ├── decoding.py                    # CP866 mojibake reversal, Arabic digits, timestamp recovery
│   │   │   ├── manifest.py                    # Recursive crawl, quarantine, 367->267g, session_id
│   │   │   └── splits.py                      # 5-fold StratifiedGroupKFold + leakage assertions
│   │   ├── pdi/                               # PHASE 2: Shared single-source Classical PDI Engine
│   │   │   ├── __init__.py
│   │   │   ├── config.py                      # PDIConfig dataclass/Pydantic + deterministic config_hash()
│   │   │   ├── io_exif.py                     # Shared Pillow EXIF + Xiaomi 39321 parser + draft/transpose
│   │   │   ├── preprocess.py                  # Aspect-preserving resize (960px), Bilateral, Lab CLAHE
│   │   │   ├── lcd_suppression.py             # Dedicated SF-400 LCD bezel/digit/scale exclusion mask
│   │   │   ├── segmentation.py                # Scene cue (bucket vs litter), multi-space Lab/HSV, morph, contour
│   │   │   ├── features.py                    # Strictly mask-gated morphometric/chromatic/optical descriptors
│   │   │   ├── xray.py                        # 4-stage inspection rendering + cv2.imencode JPEG base64
│   │   │   ├── pipeline.py                    # Unified run_single_bird_pdi() entrypoint (batch & API)
│   │   │   └── watershed.py                   # Dedicated multi-bird Distance Transform + Marker Watershed
│   │   ├── modeling/                          # PHASE 3: Tabular regression & stratified audit
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py                    # Manifest + feature join, WITH/WITHOUT optical feature lists
│   │   │   ├── models.py                      # RidgeLog, HuberLog, RandomForestLog, LightGBMLog factories
│   │   │   ├── evaluation.py                  # OOF MAE/RMSE/MAPE/R²/MBE by weight band, lens, scene, ablation
│   │   │   ├── age_stage.py                   # Cobb Avian 48 weight-to-age stage mapper (shared with API)
│   │   │   └── artifacts.py                   # Joblib bundle packer, JSON report & presets catalog writers
│   │   └── api/                               # PHASE 4: FastAPI local serving layer
│   │       ├── __init__.py
│   │       ├── main.py                        # FastAPI factory, lifespan loader, CORS, optional static mount
│   │       ├── schemas.py                     # Strict Pydantic v2 request/response contracts
│   │       ├── dependencies.py                # Typed accessors for app.state singletons
│   │       ├── routers/
│   │       │   ├── health.py                  # GET  /api/health
│   │       │   ├── predict.py                 # POST /api/predict
│   │       │   ├── watershed.py               # POST /api/watershed-demo
│   │       │   ├── presets.py                 # GET  /api/presets
│   │       │   └── metrics.py                 # GET  /api/metrics
│   │       └── services/
│   │           ├── inference_service.py       # Orchestrates shared PDI + bundle predict + OOF preset routing
│   │           ├── watershed_service.py       # Orchestrates isolated watershed pipeline + preset bytes
│   │           └── preset_service.py          # Startup preloading of ~8-12 curated preset raw bytes & thumbs
│   ├── scripts/                               # Reproducible offline CLI entrypoints (Phases 1-3)
│   │   ├── build_manifest.py                  # Phase 1 CLI -> manifest_clean + manifest_quarantine + audit json
│   │   ├── extract_features.py                # Phase 2 CLI -> parallel features_clean.parquet
│   │   └── train_and_evaluate.py              # Phase 3 CLI -> weight_regressor.joblib + evaluation_report.json
│   └── tests/                                 # Automated verification & anti-leakage contract tests
│       ├── test_ingestion_rules.py            # Mojibake, Arabic digits, nested folders, 367->267, session split
│       ├── test_pdi_anti_leakage.py           # Proves pixel perturbation outside mask yields identical features
│       ├── test_pdi_latency_and_scale.py      # Asserts < 500 ms latency & scale invariance
│       ├── test_modeling_no_session_leak.py   # Asserts disjoint session_id across all 5 folds
│       └── test_api_endpoints.py              # FastAPI TestClient contract tests
└── frontend/                                  # PHASE 4: React + TypeScript + Vite + Tailwind SPA
    ├── package.json
    ├── vite.config.ts                         # Dev proxy: /api -> http://127.0.0.1:8000
    ├── tailwind.config.ts
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx                            # Top nav shell & 3 primary pitch views
        ├── api/
        │   ├── client.ts                      # Typed fetch wrapper (timeouts, FormData, error handling)
        │   └── types.ts                       # TypeScript mirrors of FastAPI Pydantic schemas
        ├── hooks/
        │   ├── usePresets.ts                  # Fetches & caches GET /api/presets once
        │   ├── useMetrics.ts                  # Fetches & caches GET /api/metrics once
        │   └── useFlockBatch.ts               # Client-side simulated flock accumulator (mean/SD/count)
        ├── components/
        │   ├── layout/
        │   │   └── AppHeader.tsx              # Navigation tabs + live API health/latency badge
        │   ├── studio/
        │   │   ├── UploadAndOpticalControls.tsx # Dropzone, EXIF badge, lens/zoom manual override
        │   │   ├── PresetsStrip.tsx           # Curated real validation presets (GT vs Predicted live)
        │   │   ├── XRayQuadViewer.tsx         # 4-stage side-by-side PDI visual inspection + lightbox zoom
        │   │   ├── FeaturesTable.tsx          # Grouped Morphometric / Chromatic / Optical descriptors
        │   │   ├── PredictionSummaryCard.tsx  # Weight (g), CI band, Cobb age stage, LCD-free badge, ms
        │   │   └── FlockBatchPanel.tsx        # Accumulated simulated batch mean weight & bird history
        │   ├── watershed/
        │   │   ├── MultiBirdPresetBar.tsx     # 417 / 757 / 823 / 1520 scene cards + custom upload
        │   │   ├── WatershedControls.tsx      # Tunable min_distance & peak_threshold sliders
        │   │   ├── WatershedStageGrid.tsx     # Original, Mask, Distance+Peaks, Color Basins + instance table
        │   │   └── CeilingRoadmapPanel.tsx    # MVP vs Conceptual Ceiling-Camera Architecture comparison
        │   └── metrics/
        │       ├── ModelComparisonTable.tsx   # Ridge/Huber/RF/LightGBM 5-fold CV comparison
        │       ├── OpticalAblationCards.tsx   # With vs Without EXIF delta MAE / MAPE / R²
        │       ├── ActualVsPredictedChart.tsx # Interactive Recharts scatter colored by band/lens
        │       ├── StratifiedMetricsGrid.tsx  # Breakdown by weight band, lens, and visual scenario
        │       └── ForensicAuditFunnel.tsx    # 1,753 -> 1,738 curation & session_id anti-leakage audit
        └── pages/
            ├── InferenceStudioPage.tsx        # Tab 1: X-Ray Studio + Presets + Simulated Flock
            ├── WatershedVisionPage.tsx        # Tab 2: Multi-Bird Watershed + Future Vision Roadmap
            └── MetricsAuditPage.tsx           # Tab 3: Stratified Validation + Ablation + Data Audit
```

### Structure Rationale

- **`backend/app/pdi/` as the central shared package:** Both `scripts/extract_features.py` (offline) and `app/api/services/inference_service.py` (online) import `run_single_bird_pdi` from the exact same Python package, eliminating the classic notebook-vs-API skew.
- **`backend/app/pdi/io_exif.py` shared with `ingestion`:** Parsing Xiaomi tag `39321` (`0x9999`), IFD `0x8769` `FocalLength` (`37386`), and `ImageOps.exif_transpose()` lives in one module so Phase 1 manifest creation and Phase 4 live uploads never diverge on EXIF parsing.
- **`backend/app/pdi/lcd_suppression.py` as an explicit standalone module:** Isolating SF-400 scale/LCD removal into its own module (rather than burying it inside generic thresholding) makes `PROIBIÇÃO 2` auditable in code review, unit-testable in isolation, and explainable during the pitch.
- **`backend/app/pdi/watershed.py` separated from `pipeline.py`:** Enforces a hard boundary between single-bird morphometric regression and multi-bird instance separation (`417`, `757`, `823`, `1520`), preventing accidental leakage of group-sum labels into single-bird modeling.
- **`models/presets_catalog.json` instead of copying `.jpg` files into `frontend/public/`:** Respects `.gitignore` (`*.jpg`, `*.png` ignored) and `PROIBIÇÃO 4` (never commit dataset images). FastAPI reads the ~8–12 curated images from `data/raw/` once at startup, generates thumbnails in RAM, and serves everything via `/api/presets` and `/api/predict`.
- **Clean 1-to-1 Phase Alignment:** Phase 1 builds `ingestion/` + `scripts/build_manifest.py`; Phase 2 builds `pdi/` + `scripts/extract_features.py`; Phase 3 builds `modeling/` + `scripts/train_and_evaluate.py`; Phase 4 builds `api/` + `frontend/`. No phase requires refactoring an earlier phase's folder layout.

## Architectural Patterns

### Pattern 1: Single-Source PDI Engine & Config-Hash Lock (Zero Training/Serving Skew)

**What:** Wrap the entire classical PDI sequence (`io_exif → preprocess → lcd_suppression → segmentation → features`) in a single pure Python function parameterized by an immutable `PDIConfig`, hash that config with SHA-256, and store both the serialized `PDIConfig` and its hash inside `models/weight_regressor.joblib`. At FastAPI startup, instantiate the serving engine using the bundle's locked `PDIConfig` and validate feature column order before every `predict()` call.

**When to use:** Always when an offline batch feature extractor feeds an online real-time inference endpoint.

**Trade-offs:**
- *Pro:* Mathematically eliminates parameter drift (e.g., changing `clahe_clip_limit` or `working_long_edge` in code without retraining is immediately caught at startup).
- *Pro:* `return_xray=False` keeps offline batch extraction over 1,738 images fast (~25–45 s on multi-core CPU), while `return_xray=True` adds inspection artifacts on demand for the web UI.
- *Con:* Any modification to `PDIConfig` or segmentation logic requires re-running Phases 2–3 CLI scripts (~1 minute total) before serving. This is desirable discipline, not a burden.

**Example:**
```python
# backend/app/pdi/pipeline.py
from dataclasses import dataclass
import hashlib, json
import numpy as np

@dataclass(frozen=True)
class PDIConfig:
    working_long_edge: int = 960
    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    xray_display_long_edge: int = 640
    xray_jpeg_quality: int = 82

    def config_hash(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

def run_single_bird_pdi(
    source: bytes | Path,
    config: PDIConfig,
    optical_override: OpticalOverride | None = None,
    return_xray: bool = False,
) -> PDIResult:
    loaded = load_and_orient_image(source, working_long_edge=config.working_long_edge)
    optical = resolve_optical_meta(loaded.exif_optical, optical_override)
    prep = preprocess_classical(loaded.bgr_work, config)
    seg = segment_single_bird_no_lcd(prep, config)
    features = extract_mask_gated_features(
        prep=prep,
        bird_mask=seg.bird_mask,
        scale_factor=loaded.scale_factor,
        optical=optical,
    )
    xray = render_xray_stages(loaded, prep, seg, config) if return_xray else None
    return PDIResult(features=features, optical=optical, audit=seg.audit, xray=xray)
```

### Pattern 2: Canonical Working Resolution (`960 px`) with Exact Scale Back-Projection (`< 500 ms` CPU SLA)

**What:** Raw Xiaomi Redmi 10 JPEGs are `4080×1840` (`7.51 MP`, Wide) or `3264×1472` (`4.80 MP`, Ultrawide) with `2.217:1` aspect ratio. Running Bilateral filtering, Lab conversion, CLAHE, morphology, and contour extraction at `7.51 MP` blows past `1.2–2.5 s` on CPU. Instead, immediately record `(orig_w, orig_h)` after EXIF orientation, use Pillow `im.draft("RGB", (working_long_edge * 2, working_long_edge * 2))` for fast libjpeg shrink-on-load, transpose via `ImageOps.exif_transpose()`, and downscale with `cv2.resize(..., interpolation=cv2.INTER_AREA)` to a **fixed canonical working long edge of `960 px`** (`960×433`, `~0.42 MP` — an **18× pixel reduction** preserving exact aspect ratio). Then back-project pixel measurements to canonical native scale using `scale_factor = work_long_edge / orig_long_edge`:
- `area_px = area_work_px / (scale_factor ** 2)`
- `perimeter_px = perimeter_work_px / scale_factor`
- `major_axis_px = major_axis_work_px / scale_factor`
- `area_optical_norm = area_px / ((focal_length_mm * zoom_multiple) ** 2)`
- Plus resolution-invariant `area_frame_frac = area_work_px / (work_w * work_h)` and `area_frac_optical_norm`.

**When to use:** Mandatory for all images entering the single-bird or Watershed PDI engines.

**Trade-offs:**
- *Pro:* Brings full end-to-end CPU inference latency to **~110–220 ms** (more than 2× faster than the `500 ms` hard SLA).
- *Pro:* Zero aspect-ratio distortion (unlike square `224×224` warping, which destroys morphometry on `20:9` images).
- *Pro:* `960×433` still resolves even the smallest `116 g` chick with ~8,000–25,000 foreground pixels—far more than needed for sub-percent contour/ellipse area precision.
- *Critical Caution:* Record `orig_w, orig_h` (adjusted for EXIF `Orientation in (6, 8)`) **BEFORE** calling `im.draft()`, because `im.draft()` mutates `im.size` in place! And apply `im.draft()` identically in offline batch and online inference (safe no-op on non-JPEG user uploads, followed by exact `cv2.INTER_AREA` downscale to `960 px`).

### Pattern 3: Two-Layer Visual Anti-Leakage Architecture (SF-400 LCD Exclusion)

**What:** Structurally enforce `PROIBIÇÃO 2` at two distinct architectural seams:
1. **Explicit Exclusion Layer (`lcd_suppression.py`):** Before selecting the bird connected component, compute `lcd_and_scale_mask` (detecting dark/low-saturation rectangular LCD frames, dark 7-segment ink high-gradient clusters via morphological black-hat, and scale button/bezel geometry) and subtract it (`candidate_mask = raw_fg_mask & ~lcd_and_scale_mask`). Record `lcd_excluded_px` for audit reporting.
2. **Strict Mask-Gated Feature Extraction (`features.py`):** `extract_mask_gated_features()` accepts ONLY `bird_mask` (single binary dominant silhouette) and computes all chromatic/texture statistics strictly over masked pixel arrays (`lab_img[bird_mask_bool]`, `hsv_img[bird_mask_bool]`). It **never** computes whole-image histograms, background means, or scale-region crops.

**When to use:** Non-negotiable across the entire single-bird pipeline.

**Trade-offs:**
- *Pro:* Provides both an engineering guarantee (verified by an automated unit test that paints bright digits outside `bird_mask` and asserts bit-identical feature output) and visual proof in X-Ray Stage 3 (`Máscara binária limpa sem LCD`).
- *Con:* Requires careful handling of white-on-white heavy broilers (>1,000 g) standing on the white SF-400 pan or inside the white bucket so LCD suppression does not accidentally erode the bird's chest/wings (mitigated by limiting suppression to dark/gray low-luminance rectangular bezels + high local black-hat edge density, never pure bright white feather texture).

### Pattern 4: FastAPI Lifespan Startup Caching + Async Intake / Threadpool Offload

**What:** Load heavy read-only state **once** in `@asynccontextmanager async def lifespan(app)` into `app.state` (`model_bundle`, locked `PDIConfig`, `evaluation_report`, `audit_summary`, and the ~8–12 curated preset raw image bytes + thumbnails, totaling `< 20 MB` RAM). Set `cv2.setNumThreads(2)` at startup to prevent thread oversubscription. Define upload endpoints (`/api/predict`, `/api/watershed-demo`) as `async def` solely to perform non-blocking `raw_bytes = await file.read()` with a size cap (e.g. `15 MB`), then immediately offload the synchronous CPU-bound OpenCV + `joblib` function via `await run_in_threadpool(...)` (which leverages OpenCV's C++ GIL release without stalling FastAPI's asyncio event loop).

**When to use:** Always for FastAPI services running OpenCV/scikit-learn/LightGBM inference on uploaded images.

**Trade-offs:**
- *Pro:* Zero disk I/O for user uploads (decodes from `BytesIO` / `np.frombuffer` in RAM) and zero disk I/O for curated pitch presets (served from startup RAM cache).
- *Pro:* Event loop remains responsive to `/api/health`, `/api/metrics`, and concurrent UI tab switches during inference.
- *Con:* Startup takes ~0.5–1.5 s to load joblib + ~12 preset JPEGs into memory—negligible for a local server.

**Example:**
```python
# backend/app/api/routers/predict.py
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

@router.post("/predict", response_model=PredictResponse)
async def predict_weight(
    file: UploadFile | None = File(default=None),
    preset_id: str | None = Form(default=None),
    focal_length_mm: float | None = Form(default=None),
    zoom_multiple: float | None = Form(default=None),
    service: InferenceService = Depends(get_inference_service),
) -> PredictResponse:
    if (file is None) == (preset_id is None):
        raise HTTPException(400, "Provide exactly one of 'file' or 'preset_id'.")

    raw_bytes = await _read_bounded_upload(file, max_bytes=15 * 1024 * 1024) if file else None
    return await run_in_threadpool(
        service.predict,
        raw_bytes=raw_bytes,
        preset_id=preset_id,
        focal_length_mm=focal_length_mm,
        zoom_multiple=zoom_multiple,
    )
```

### Pattern 5: Dual-Bundle Honesty Pattern (Production Pipeline + Out-of-Fold Preset Evaluators)

**What:** Store inside `models/weight_regressor.joblib` not only the `production_pipeline` (refitted on 100% of the 1,738 clean images for arbitrary new user uploads), but also the `5` fold pipelines (`fold_pipelines: dict[int, Pipeline]`, ~200 KB–1 MB each for Ridge/Huber/LightGBM). When a pitch presenter clicks a curated preset (`preset_id` whose `session_id` belonged to `fold k`), `InferenceService` runs live PDI and predicts using `fold_pipelines[k]`—a model that **never saw that bird or session during training**! Arbitrary user uploads use `production_pipeline`.

**When to use:** Academic/technical product pitches where demonstrating Real vs. Predicted on curated validation presets will be scrutinized for in-sample overfitting.

**Trade-offs:**
- *Pro:* Makes every single live preset prediction 100% scientifically honest (strictly out-of-fold, unseen `session_id`) while still executing the real PDI + ML inference live in `< 500 ms`.
- *Pro:* Adds minimal disk/RAM overhead (< 5 MB for tabular models).
- *Con:* Slightly richer bundle dict schema; trivially handled by a 4-line branch in `InferenceService`.

### Pattern 6: Vite API Proxy + Lightweight Feature-Tab React SPA

**What:** Configure Vite's dev proxy (`server.proxy['/api'] = 'http://127.0.0.1:8000'`) alongside FastAPI CORS middleware so frontend code always calls relative `/api/*` paths. Avoid heavy global state libraries (Redux/Zustand/TanStack Query are overkill for 4 local endpoints); use typed native `fetch` (`src/api/client.ts`), three small hooks (`usePresets`, `useMetrics`, `useFlockBatch`), and component-scoped React state.

**When to use:** Single-user local pitch applications with 3–4 views and zero auth/multi-tenant state.

**Trade-offs:**
- *Pro:* Zero CORS/port friction during local dev AND seamless transition if FastAPI mounts `frontend/dist/` via `StaticFiles` for one-command pitch execution (`uvicorn app.api.main:app`).
- *Pro:* Keeps bundle size, setup time, and debugging surface tiny inside the 8-day MVP window.

## Data Flow

### Flow 1: Offline Batch Pipeline (Phases 1 → 2 → 3)

```text
data/raw/Broiler weights of Cobb Avian/ (1,753 JPEGs, 168 leaf dirs, 3.54 GiB)
    │
    ▼  [Phase 1: backend/scripts/build_manifest.py]
    │  • Recursive os.walk -> captures nested 171/173, 190/191, 190/193
    │  • Regex sanitize 1340_ -> 1340; revert CP866 (547, 562); translate ٠-٩
    │  • Reconstruct capture_dt (filename or mtime - 14401 s)
    │  • Read EXIF Orientation (274), FocalLength (37386), Xiaomi 39321 zoomMultiple
    │  • Quarantine 15 imgs (417, 757, 823, 1520 multi-bird; 1351, 1465 mislabel)
    │  • Correct 367 -> 267 g (6 imgs confirmed on SF-400 LCD)
    │  • Group session_id (gap > 600 s within folder_chain) -> 188 clean sessions
    │  • Assign 5-fold StratifiedGroupKFold (bands: <500g, 500-1000g, >1000g)
    │
    ├──▶ data/processed/manifest_quarantine.parquet  (15 rows: 13 multi_bird + 2 mislabel)
    ├──▶ models/dataset_audit_summary.json           (forensic funnel for Dashboard)
    └──▶ data/processed/manifest_clean.parquet       (1,738 rows, 160 weights, 188 sessions, 5 folds)
             │
             ▼  [Phase 2: backend/scripts/extract_features.py]
             │  • ProcessPoolExecutor (cv2.setNumThreads(1) per worker)
             │  • Calls shared run_single_bird_pdi(..., return_xray=False)
             │  • Logs pdi_config_hash + per-image QC/scene/lcd_excluded_px
             │
             └──▶ data/processed/features_clean.parquet (1,738 rows x morph/chroma/optical features)
                      │
                      ▼  [Phase 3: backend/scripts/train_and_evaluate.py]
                      │  • Joins manifest_clean + features_clean; asserts 0 session overlap across folds
                      │  • Trains 4 candidate regressors (RidgeLog, HuberLog, RFLog, LGBMLog) x 5 folds
                      │  • Runs optical ablation: WITH_OPTICAL_EXIF vs WITHOUT_OPTICAL_EXIF
                      │  • Computes OOF MAE, RMSE, MAPE, R², MBE globally + by weight band / lens / scene
                      │  • Selects best model (primary: balanced MAPE + stratified MAE), refits production model
                      │  • Picks real curated presets (180g, 550g, 1600g, white-on-white bucket, lens pair)
                      │
                      ├──▶ models/weight_regressor.joblib  (prod + 5 fold pipelines + feature_order + pdi_config_hash + band intervals)
                      ├──▶ models/evaluation_report.json   (full stratified CV, ablation, OOF scatter)
                      └──▶ models/presets_catalog.json     (curated single-bird + 4 multi-bird preset metadata)
```

### Flow 2: Online Real-Time Inference (`POST /api/predict` — Target `< 500 ms` CPU)

```text
[Browser: Upload Image OR Click Curated Preset (+ Optional Lens/Zoom Override)]
    │  POST /api/predict (multipart/form-data)
    ▼
[FastAPI Router (async def)]
    │  1. Validate input (file XOR preset_id, max 15 MB) -> read bytes in RAM
    │  2. Offload via await run_in_threadpool(inference_service.predict, ...)
    ▼
[InferenceService (Worker Thread)]
    │  3. Resolve bytes: uploaded bytes OR pre-cached preset bytes from app.state
    │  4. Shared PDI Engine: run_single_bird_pdi(..., config=bundle_config, return_xray=True)
    │     ├─ io_exif.py:     EXIF read (274, 37386, Xiaomi 39321) + draft + exif_transpose  [~35-70 ms]
    │     ├─ preprocess.py:  cv2.resize INTER_AREA -> 960px + Bilateral + Lab CLAHE         [~25-45 ms]
    │     ├─ segmentation:   SF-400 LCD exclusion + multi-space Lab/HSV + morph + ellipse   [~20-40 ms]
    │     ├─ features.py:    Mask-gated morph/chroma + optical norm (back-projected scale)   [~5-10 ms]
    │     └─ xray.py:        Render 4 stages @ 640px + cv2.imencode(.jpg, Q=82) + base64    [~25-50 ms]
    │  5. Schema Alignment: build 1-row DataFrame strictly ordered by bundle["feature_names"]
    │  6. Predict: fold_pipelines[preset.fold] (if preset) OR production_pipeline (if upload) [~1-5 ms]
    │  7. Attach weight-band error interval + Cobb Avian 48 age stage + total latency_ms
    ▼
[Pydantic PredictResponse JSON]  ──▶  [React Inference X-Ray Studio + Batch Flock Accumulator]
    Total CPU Latency Budget: ~110–220 ms typical (hard ceiling < 500 ms)
```

### Flow 3: Dedicated Multi-Bird Watershed Demo (`POST /api/watershed-demo`)

```text
[Browser: Select Multi-Bird Scene (417 / 757 / 823 / 1520) OR Upload + Optional Sliders]
    │  POST /api/watershed-demo
    ▼
[FastAPI Router -> run_in_threadpool(watershed_service.run, ...)]
    │  1. Reuse pdi.io_exif + pdi.preprocess (960 px working resolution, EXIF transposed)
    │  2. Multi-bird foreground union mask (handles straw litter AND red feeder in folder 417)
    │  3. Morphological open/close -> Euclidean cv2.distanceTransform (L2)
    │  4. Gaussian smoothing of distance map -> skimage.feature.peak_local_max (min_distance, threshold_rel)
    │  5. Connected-component markers -> cv2.watershed(bgr_work, markers)
    │  6. Filter valid bird basins -> compute per-bird area_px (back-projected), area_optical_norm, centroids
    │  7. Render 4 Watershed X-Ray images: Original | Union Mask | Distance+Peaks Heatmap | Color Basins
    ▼
[Pydantic WatershedResponse JSON]  ──▶  [React Watershed & Future Vision Tab]
```

### Frontend State Management

```text
[App Mount]
   ├── GET /api/health   ──▶ AppHeader status badge
   ├── GET /api/presets  ──▶ usePresets() memoized in memory (shared by Studio + Watershed tabs)
   └── GET /api/metrics  ──▶ useMetrics() memoized in memory (Metrics & Audit Dashboard)

[User Runs Prediction in Studio]
   └── POST /api/predict ──▶ Studio local state (currentResult)
                                  │
                                  └── [Click "Add to Flock Batch" or auto-append toggle]
                                          └── useFlockBatch() local state array
                                              (live mean weight g, SD, min, max, bird count)
```

No database or persistent server state is needed for the MVP; simulated flock accumulation lives cleanly in client state (optionally backed by `sessionStorage` so accidental refreshes during a pitch don't wipe the simulated flock!).

## Scaling Considerations

The MVP is intentionally scoped to a **single-presenter local pitch** running on one laptop/workstation CPU. However, because the business pitch explicitly presents a roadmap toward **fixed ceiling-mounted cameras in commercial broiler houses** (Paraná poultry integrators), the architecture cleanly maps how components evolve without over-engineering v1:

| Scale | Architecture Adjustments |
|-------|--------------------------|
| **v1 MVP (Local Pitch — 1 user, `< 500 ms`)** | Single Uvicorn process (`--workers 1`), `lifespan` RAM cache (`< 20 MB` presets + `< 10 MB` joblib bundle), `run_in_threadpool` + `cv2.setNumThreads(2)`, Vite proxy or FastAPI static mount. Zero database, zero message queue. |
| **Pilot Edge (1–5 Broiler Houses, 1–5 FPS Ceiling Cam)** | Fixed calibrated ceiling height eliminates EXIF zoom/lens ambiguity (`mm/pixel` becomes a fixed homography constant!). Run the exact same `backend/app/pdi/` + `weight_regressor.joblib` on a low-cost edge SBC (e.g. Raspberry Pi 5 / Intel N100 mini-PC) reading RTSP snapshots periodically; log hourly flock weight aggregates to SQLite/PostgreSQL. |
| **Integrator Fleet (50–500 Houses)** | Edge nodes run lightweight classical PDI + local regression on-barn (sending only KB/hour tabular summaries, never GBs of video, over rural internet); central cloud API aggregates flock growth curves vs. Cobb Avian 48 benchmark and triggers feed-conversion alerts. |

### Scaling Priorities (What Breaks First)

1. **First bottleneck in v1 (CPU latency on high-MP inputs):** Decoding and bilateral-filtering raw `4080×1840` (`7.51 MP`) images without downscaling immediately breaks the `< 500 ms` SLA (~1.5–2.5 s). **Fixed upfront** by Pattern 2 (`im.draft()` + `960 px` canonical long edge + `INTER_AREA`).
2. **Second bottleneck in v1 (Payload bloat from 4 X-ray images):** Encoding 4 full-res PNGs as base64 would produce 8–15 MB JSON responses and sluggish UI rendering. **Fixed upfront** by rendering X-ray inspection stages at `640 px` max edge as JPEG (`quality=82`) — ~35–70 KB per stage (~200 KB total JSON payload, renders instantaneously).
3. **Out-of-scope boundary (`PROIBIÇÃO 5`):** Multi-frame video tracking, barn population density estimation, and stationary-bird health alerts are **strictly conceptual roadmap slides** in `CeilingRoadmapPanel.tsx`—never implemented in v1 backend code.

## Anti-Patterns

### Anti-Pattern 1: Duplicating PDI Logic Between Offline Scripts and FastAPI

**What people do:** Prototype OpenCV segmentation and feature formulas inside `scripts/extract_features.py` (or a Jupyter notebook), then copy-paste and "adapt" them inside `api/routers/predict.py` for web uploads.
**Why it's wrong:** Inevitable **training-serving skew** (subtle differences in RGB vs BGR order, `INTER_LINEAR` vs `INTER_AREA`, CLAHE grid size, or optical normalization constants) silently destroys prediction accuracy during the live pitch.
**Do this instead:** Put 100% of image loading, EXIF parsing, preprocessing, LCD exclusion, segmentation, and feature computation inside `backend/app/pdi/pipeline.py` (`run_single_bird_pdi`), lock its `PDIConfig` hash into `weight_regressor.joblib`, and import it from both the batch CLI and FastAPI.

### Anti-Pattern 2: Processing Full `4080×1840` Images or Warping `20:9` to Square

**What people do:** Either run OpenCV filters directly on native `7.51 MP` arrays (exceeding `500 ms`), or resize directly to a square `512×512` (like standard CNN pipelines).
**Why it's wrong:** Full resolution kills latency; square warping distorts the `2.217:1` aspect ratio by `2.2×`, corrupting ellipse eccentricity, circularity, major/minor axes, and projected area differently between portrait (`64.8%`) and landscape (`35.2%`) captures.
**Do this instead:** Downscale preserving exact aspect ratio to `working_long_edge = 960` via `cv2.INTER_AREA` and back-project linear/areal descriptors via `scale_factor = work_long_edge / orig_long_edge`.

### Anti-Pattern 3: Allowing Whole-Image Stats or Unmasked Scale Pixels into Features

**What people do:** Compute global image brightness/color histograms, bounding-box crops that still include the SF-400 scale display, or pass raw `(oriented_w, oriented_h)` pixels into the regressor without optical normalization.
**Why it's wrong:** Violates `PROIBIÇÃO 2` (LCD data leakage) AND exploits the spurious lens×weight confounding documented in the audit (`3264×1472` ultrawide jumps from ~20% to 56–78% above `1,000 g` — a model given raw image dimensions can cheat by guessing "small resolution = heavy bird"!).
**Do this instead:** Explicitly suppress the SF-400 LCD/bezel before component selection, compute all color/shape features **strictly inside `bird_mask == 255`**, never feed raw `(oriented_w, oriented_h)` as regressor features, and verify invariance with `tests/test_pdi_anti_leakage.py`.

### Anti-Pattern 4: Blocking FastAPI's Asyncio Event Loop or Using Bare `joblib` Arrays

**What people do:** Run `cv2.bilateralFilter` + `joblib.predict` directly inside an `async def` route without `run_in_threadpool`, and pass an unvalidated Python list `[f1, f2, ...]` to `model.predict()`.
**Why it's wrong:** Blocks the event loop for every other request; silent feature ordering mismatches produce plausible-looking garbage weights without raising exceptions.
**Do this instead:** Read `UploadFile` bytes asynchronously, dispatch PDI + ML to `run_in_threadpool`, and always reindex a 1-row `DataFrame` against `bundle["feature_names"]` after asserting no missing keys.

### Anti-Pattern 5: Mixing Multi-Bird Folders or Random Image Splits into Single-Bird Training

**What people do:** Run `train_test_split` over all 1,753 images or leave `417`, `757`, `823`, `1520` inside the regressor dataset.
**Why it's wrong:** Violates `PROIBIÇÃO 3` (burst shots of ~9 photos/session of the exact same bird cause massive leakage and fake $R^2$) and poisons single-bird weights with 2–4 bird group sums.
**Do this instead:** Enforce `manifest_quarantine.parquet` separation in Phase 1, restrict `417`/`757`/`823`/`1520` strictly to `pdi/watershed.py`, enforce 5-fold `StratifiedGroupKFold` by `session_id`, and assert disjoint `session_id` sets in `tests/test_modeling_no_session_leak.py`.

### Anti-Pattern 6: Committing Preset `.jpg` Assets into `frontend/public/`

**What people do:** Copy curated dataset photos into `frontend/public/presets/*.jpg` so the React UI can render them directly.
**Why it's wrong:** Violates `.gitignore` (`*.jpg` ignored) and `PROIBIÇÃO 4`, risks bloating Git history, and breaks single-source preset governance.
**Do this instead:** Commit only the lightweight metadata manifest `models/presets_catalog.json` (storing `preset_id`, `rel_path` relative to `data/raw/Broiler weights of Cobb Avian/`, ground truth weight, fold, and optical metadata). FastAPI resolves those paths from local `data/raw/` at startup, caches bytes + thumbnails in memory, and serves them via `/api/presets`.

## Integration Points

### External / Local Runtime Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **Local Filesystem (`data/raw/`)** | Read-only via `pathlib.Path` rooted at `settings.DATA_RAW_DIR` | Never modified in place; mojibake (`547/`, `562/`) and Arabic filenames (`٠-٩`) are decoded in memory only. Startup fails gracefully with a clear diagnostic message if `DATA_RAW_DIR` is missing. |
| **FastAPI ↔ Vite Dev Server** | Vite proxy `/api` → `http://127.0.0.1:8000` + FastAPI CORS allowlist (`localhost:5173`, `127.0.0.1:5173`) | Allows independent hot-reload during development (`uvicorn --reload` + `npm run dev`) and optional single-server pitch mode by mounting `frontend/dist/` in FastAPI. |

### Internal Module & Phase Boundaries

| Boundary | Communication / Artifact Contract | Build Order & Considerations |
|----------|-----------------------------------|------------------------------|
| **`ingestion` → `pdi` (Phase 1 → 2)** | `data/processed/manifest_clean.parquet` (columns: `image_id`, `rel_path`, `folder_chain`, `weight_g`, `raw_weight_g`, `weight_band`, `session_id`, `fold`, `capture_dt`, `focal_length_mm`, `zoom_multiple`, `lens_type`, `exif_orientation`, `oriented_w`, `oriented_h`) + `manifest_quarantine.parquet` | Phase 1 must pass exact row-count & integrity assertions (`1,738` clean, `15` quarantine, `160` unique weights, `188` sessions, `0` session overlap across folds) before Phase 2 runs batch extraction. |
| **`pdi` → `modeling` (Phase 2 → 3)** | `data/processed/features_clean.parquet` keyed by `image_id` + `pdi_config_hash` + `estimated_scene` (`bucket_dorsal` vs `scale_or_litter`) + `lcd_excluded_px` | Feature extraction writes explicit typed float columns; `modeling/datasets.py` defines exact `OPTICAL_FEATURES` vs `CORE_PDI_FEATURES` subsets for reproducible ablation. |
| **`modeling` → `api` (Phase 3 → 4)** | `models/weight_regressor.joblib` (versioned dict bundle) + `models/evaluation_report.json` + `models/presets_catalog.json` + `models/dataset_audit_summary.json` | API validates `bundle_version` and `pdi_config_hash` on startup. `age_stage.py` lives in `modeling/` (or `core/`) and is imported by `api/` so Cobb Avian 48 weight→age rules are never duplicated. |
| **`api` → `frontend` (Phase 4)** | Typed REST JSON contracts:<br>• `GET /api/health` → `{status, model_name, pdi_config_hash, presets_loaded}`<br>• `GET /api/presets` → `{single_bird: PresetCard[], multi_bird: PresetCard[]}`<br>• `GET /api/metrics` → `{evaluation_report, dataset_audit_summary}`<br>• `POST /api/predict` → `PredictResponse` (weight, CI, age stage, optical source, features, audit, 4 base64 X-ray JPEGs, optional GT/error%)<br>• `POST /api/watershed-demo` → `WatershedResponse` (count, per-bird instances, 4 base64 Watershed JPEGs, optional GT group weight) | Define `backend/app/api/schemas.py` and `frontend/src/api/types.ts` together at the start of Phase 4 so frontend components can even develop against the exact schema immediately. |
| **`pdi.pipeline` ↔ `pdi.watershed`** | Shared read-only utilities (`io_exif.py`, `preprocess.py`); completely separate segmentation, feature/instance extraction, and API routers (`/api/predict` vs `/api/watershed-demo`). | Prevents any cross-contamination between single-bird regression and multi-bird instance separation. |

## Sources

- Project Specification & Forensic Audit: `PRD.md`, `.planning/PROJECT.md`, and `docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md` (verified counts, EXIF schema, Xiaomi tag `39321`, quarantine rules, `session_id` logic, and `PROIBIÇÃO 1–5`).
- FastAPI Official Lifespan & Concurrency Documentation: `@asynccontextmanager` lifespan loading into `app.state` and `run_in_threadpool` offloading for CPU-bound OpenCV/ML handlers releasing the GIL (`tiangolo.com` / Starlette concurrency).
- Pillow & OpenCV Performance Practices: `Image.draft()` shrink-on-load prior to `ImageOps.exif_transpose()`, `cv2.resize(..., interpolation=cv2.INTER_AREA)` downscaling, `cv2.setNumThreads()` tuning under ASGI threadpools, and `cv2.imencode('.jpg')` in-memory serialization.
- Scikit-Learn Production Serving Best Practices: `sklearn.pipeline.Pipeline` + `TransformedTargetRegressor` bundled with explicit feature order schema and preprocessing config hash to eliminate training-serving skew (`scikit-learn.org`).

---
*Architecture research for: Classical PDI Broiler Weight Estimation Pipeline + FastAPI + React/Vite Local Pitch Web App (AvianScale CV)*
*Researched: 2026-11-20*
