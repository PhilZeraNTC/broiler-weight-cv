# Roadmap: AvianScale CV

## Overview

AvianScale CV delivers an auditable, 100% interpretable broiler live-weight estimation MVP using strictly classical Digital Image Processing (PDI) and EXIF-calibrated tabular regression on a 4-phase critical path toward the 2026-11-28 code freeze (pitch 2026-11-30). Phase 1 locks a reproducible Python 3.12 environment, executes the deterministic Audit §9 forensic dataset curation (`1,738` clean / `15` quarantined images), and freezes a leak-free 5-fold `session_id` split; Phase 2 builds the shared single-source classical PDI engine with hard SF-400 LCD exclusion, empirically verified dual-lens optical calibration, 4-stage "X-Ray" artifacts, and Marker-based Watershed multi-bird separation; Phase 3 benchmarks interpretable tabular regressors with quantitative EXIF ablation and publishes a dual-bundle production + out-of-fold artifact; and Phase 4 exposes the `< 500 ms` local FastAPI backend and 3-tab React + Tailwind CSS pitch studio.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Forensic Data Curation & Anti-Leakage Split** - Pin Python 3.12 (`uv`), build the deterministic Audit §9 dataset ingestor (`1,738` clean / `15` quarantined), and assign reproducible zero-leakage 5-fold `session_id` splits.
- [ ] **Phase 2: Classical PDI Engine, Optical Calibration & Watershed** - Build the shared EXIF-first OpenCV preprocessing and anti-LCD single-bird segmentation engine, extract strictly mask-gated morphometric/chromatic/optical features with 4-stage "X-Ray" outputs, and implement Marker-based Watershed multi-bird separation.
- [ ] **Phase 3: Tabular Regression, Optical Ablation & Production Bundling** - Train and evaluate interpretable tabular weight regressors on locked folds, prove EXIF optical calibration gain via 4-way ablation and lens-flip audit, and publish dual-bundle production + out-of-fold pitch artifacts.
- [ ] **Phase 4: FastAPI Local Backend & Interactive Pitch Web Studio** - Serve `< 500 ms` local REST endpoints and deliver the 3-tab React + TypeScript + Vite + Tailwind CSS pitch studio for live PDI "X-Ray", curated OOF presets, Watershed demo, and stratified audit metrics.

## Phase Details

### Phase 1: Forensic Data Curation & Anti-Leakage Split

**Goal**: Produce a reproducible Python 3.12 workspace and an immutable, Audit §9-exact forensic dataset manifest (`1,738` clean / `15` quarantined images) partitioned into leak-free 5-fold `session_id` splits.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: [DATA-01, DATA-02]
**Success Criteria** (what must be TRUE):

  1. Running the deterministic ingestor over the raw dataset produces `data/processed/manifest_clean.parquet` with exact audited totals (`1,738` images, `160` unique weights, `188` sessions), `data/processed/manifest_quarantine.parquet` with `15` images (`13` multi-bird across `417`/`757`/`823`/`1520` + `2` mislabels `1351`/`1465`), and git-tracked `models/dataset_audit_summary.json`.
  2. Every known dataset anomaly from Audit §9 is deterministically resolved in `manifest_clean.parquet`: nested level-2 folders (`171/173`, `190/191`, `190/193`) traversed, `1340_` sanitized to `1340 g`, `367` corrected to `267 g`, CP866 mojibake (`547/`, `562/`) and Eastern-Arabic digits (`٠-٩`) decoded into valid `capture_dt` (or `mtime - 14401 s`), and EXIF `Orientation`, `FocalLength`, and Xiaomi tag `39321` `zoomMultiple` populated via a shared Pillow EXIF loader.
  3. Every clean image carries a reproducible `fold` (`0–4`) assigned via 5-fold `StratifiedGroupKFold` grouped strictly by `session_id` (`≤ 10 min` sequential captures within `folder_chain`) and stratified by coarse weight band (`<500 g`, `500–1000 g`, `>1000 g`), verified by automated tests confirming zero `session_id` overlap across folds and balanced `>1000 g` session representation (`≥ 4` sessions per fold).
  4. The backend environment resolves exclusively from binary wheels under `uv`-pinned Python `3.12` (`uv.lock`), with zero raw or processed data artifacts tracked by Git.

**Plans**: TBD

- [x] 01-01-PLAN.md
- [ ] 01-02-PLAN.md
- [ ] 01-03-PLAN.md

### Phase 2: Classical PDI Engine, Optical Calibration & Watershed

**Goal**: Deliver the single-source classical Digital Image Processing engine (zero deep learning) that cleanly isolates bird silhouettes without SF-400 LCD leakage, extracts strictly mask-gated morphometric, chromatic, and empirically verified EXIF optical features, renders 4-stage "X-Ray" visuals, and separates touching birds via Marker-based Watershed.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: [PDI-01, PDI-02, PDI-03, PDI-04, PDI-05]
**Success Criteria** (what must be TRUE):

  1. Passing any image as a `Path` or raw `bytes` through the shared Pillow EXIF-first loader (`ImageOps.exif_transpose()`, with zero `cv2.imread`/`cv2.imdecode` calls repo-wide) yields identical orientation, aspect-preserving canonical downscale (`768–960 px` long edge with tracked `scale_factor`), edge-preserving Bilateral/Median smoothing, and CIE Lab CLAHE preprocessing.
  2. Classical single-bird segmentation isolates the broiler silhouette across all three visual scenarios (SF-400 scale on straw, dorsal white plastic bucket including white-on-white `>1,000 g` broilers, and oblique straw litter) with hard exclusion of the SF-400 LCD display, buttons, and circular platter and `< 2%` degenerate-mask fallback across all `1,738` clean images.
  3. Feature extraction computes geometric morphometrics, scale-invariant plumage chroma descriptors, and EXIF optical features strictly inside `bird_mask == 255` (verified by an automated outside-mask pixel-perturbation invariance test returning bit-identical outputs), logs empirical paired dual-lens calibration verification across the `114` mixed-lens sessions (avoiding `focal × zoom` double-counting and banning raw pixel dimensions), and writes `data/processed/features_clean.parquet` via capped-concurrency (`n_jobs=4`) batch execution in `< 60 s`.
  4. Every single-bird pipeline execution emits all 4 inspection "X-Ray" artifacts (EXIF-oriented original, CLAHE-preprocessed image, clean binary mask without LCD, and refined contour + fitted ellipse/axes overlay) alongside a SHA-256 `PDIConfig` hash, completing on a native `4080×1840` image comfortably within the `< 500 ms` CPU budget.
  5. The dedicated Marker-based Watershed pipeline (red-feeder chromatic pre-suppression, Euclidean Distance Transform, Gaussian smoothing, scale-adaptive `h_maxima`/`peak_local_max`, and `skimage.segmentation.watershed`) separates touching birds across all `13` quarantined multi-bird images (`417`, `757`, `823`, `1520`) and arbitrary custom uploads, returning basin/boundary maps, accurate bird counts, per-bird pixel areas, and per-instance mask-gated features for illustrative individual weight inference.

**Plans**: TBD

### Phase 3: Tabular Regression, Optical Ablation & Production Bundling

**Goal**: Train and rigorously audit interpretable tabular weight regressors over the locked 5-fold `session_id` splits, quantitatively prove the impact of EXIF optical calibration without lens-confounding shortcuts, and publish immutable production and out-of-fold pitch artifacts.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: [MOD-01, MOD-02, MOD-03]
**Success Criteria** (what must be TRUE):

  1. Training fits and compares interpretable tabular regressors (log-target Ridge/Huber, Random Forest, and LightGBM/Gradient Boosting — zero deep learning) across the locked 5-fold `session_id` splits using exclusively Phase 2 mask-gated PDI and optical features, evaluating capped inverse session/stratum weighting and fold-internal Duan smearing with zero leakage across folds.
  2. Out-of-fold evaluation persists git-tracked `models/evaluation_report.json` reporting MAE (g), RMSE (g), MAPE (%), $R^2$, and Mean Bias Error globally and stratified by weight band (`<500 g`, `500–1000 g`, `>1000 g`), lens (`4.25 mm` vs. `1.65 mm`), and scenario (dorsal bucket vs. scale/litter), including a 4-condition WITH vs. WITHOUT EXIF optical ablation and a within-session dual-lens consistency + counterfactual lens-flip audit.
  3. Artifact publishing serializes `models/weight_regressor.joblib` bundling the winning `production_pipeline`, all 5 out-of-fold `fold_pipelines` (for honest held-out preset inference), the locked `PDIConfig` SHA-256 hash, exact `feature_names`, version metadata, and per-band expected error intervals, verified by clean-process `joblib.load()` tests with zero pickling errors.
  4. Curated out-of-fold `models/presets_catalog.json` is generated from real validation fold samples spanning young (`~180 g`), intermediate (`~550 g`), heavy (`~1,600 g`), and hard white-on-white dorsal bucket (`>1,000 g`) broilers, storing only metadata and dataset-relative references with zero `.jpg` files committed to Git.

**Plans**: TBD

### Phase 4: FastAPI Local Backend & Interactive Pitch Web Studio

**Goal**: Deliver the `< 500 ms` local FastAPI serving API and interactive 3-tab React + TypeScript + Vite + Tailwind CSS pitch studio exposing live PDI "X-Ray" inference, out-of-fold curated presets, Watershed multi-bird separation, and the stratified validation and forensic audit dashboard.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: [WEB-01, WEB-02, WEB-03, WEB-04, WEB-05]
**Success Criteria** (what must be TRUE):

  1. Local FastAPI backend exposes Pydantic v2-validated `/api/health`, `/api/predict`, `/api/watershed-demo`, `/api/presets`, and `/api/metrics` using `lifespan` startup caching and `async def` bounded upload intake offloading CPU PDI/ML via `run_in_threadpool`, achieving `< 500 ms` end-to-end latency per image (target p95 `< 400 ms` on real `4080×1840` JPEGs) and gracefully handling EXIF-stripped JPEG and RGBA PNG uploads.
  2. User can upload any broiler image in the Inference & "X-Ray" PDI Studio, observe auto-extracted EXIF focal/zoom (or a missing-EXIF warning pill with live manual lens/zoom override), inspect all 4 classical PDI stages side-by-side alongside morphometric and chromatic plumage attributes, read estimated weight (g) with expected error band and Cobb Avian 48 developmental age stage, and accumulate a live simulated flock average and uniformity (`CV%`) tracker.
  3. User can one-click execute curated out-of-fold validation presets (`~180 g`, `~550 g`, `~1,600 g`, and white-on-white `>1,000 g` dorsal bucket) and view live Ground Truth weight vs. out-of-fold Predicted weight, absolute error (g), and `%` error computed by the matching held-out fold model.
  4. User can explore the Watershed & Future Vision tab to run interactive Marker-based Watershed separation on multi-bird presets (`417`, `757`, `823`, `1520`) or custom uploads—viewing basin/boundary overlays, separated bird counts, per-bird pixel areas, and illustrative per-bird estimated weights + scene mean weight (with explicit group-sum ground-truth disclaimer and predicted-sum sanity comparison)—alongside an architectural comparison panel contrasting the handheld/bucket MVP with the conceptual v2 fixed overhead ceiling-camera barn roadmap.
  5. User can inspect the Validation Metrics & Forensic Audit Dashboard displaying 5-fold cross-validation KPIs (with literature `R² > 0.9` context), an interactive Actual vs. Predicted scatter plot with $y=x$ reference line, WITH vs. WITHOUT EXIF optical ablation charts, stratified error tables by weight band/lens/scenario, and the `1,753 → 1,738` forensic curation funnel explicitly reconciling the Mendeley `"45 days / 1,772 images"` claim against the disk-audited `~7–29 days / 116–2,093 g` coverage and `801–1,216 g` gap.

**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Forensic Data Curation & Anti-Leakage Split | 1/3 | In Progress|  |
| 2. Classical PDI Engine, Optical Calibration & Watershed | 0/0 | Not started | - |
| 3. Tabular Regression, Optical Ablation & Production Bundling | 0/0 | Not started | - |
| 4. FastAPI Local Backend & Interactive Pitch Web Studio | 0/0 | Not started | - |
