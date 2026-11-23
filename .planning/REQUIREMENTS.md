# Requirements: AvianScale CV

**Defined:** 2026-11-20
**Core Value:** Estimar o peso vivo individual de frangos de corte de forma precisa, rigorosamente auditada e 100% interpretável usando apenas Processamento Digital de Imagens clássico calibrado por óptica EXIF — sem deep learning caixa-preta e com zero vazamento de dados do visor LCD da balança — exposto visualmente etapa por etapa em uma demonstração web local de alta credibilidade.

## v1 Requirements

Requirements for the MVP release (code freeze 2026-11-28, pitch 2026-11-30). Each requirement maps directly to a PRD requirement (`REQ-01`–`REQ-15`) and to exactly one roadmap phase.

### Data Curation & Forensic Manifest

- [x] **DATA-01** *(PRD REQ-01)*: User can run a deterministic Python ingestor (pinned to Python 3.12 via `uv`) over `data/raw/Broiler weights of Cobb Avian/` following Audit §9 verbatim — recursive level-2 leaf crawl (`171/173`, `190/191`, `190/193`), folder suffix sanitization (`1340_`), CP866 mojibake reversal (`547/`, `562/`), Eastern-Arabic digit normalization (`٠-٩`), `capture_dt` reconstruction (filename or `mtime - 14401 s`), EXIF metadata extraction (`Orientation` `274`, `FocalLength` `37386`, Xiaomi tag `39321` `zoomMultiple`), quarantine of 13 multi-bird (`417`, `757`, `823`, `1520`) + 2 severe mislabel (`1351`, `1465`) images, and `367` → `267 g` correction — persisting `data/processed/manifest_clean.parquet` (1,738 images, 160 unique weights, 188 sessions), `data/processed/manifest_quarantine.parquet` (15 images), and `models/dataset_audit_summary.json`.
- [x] **DATA-02** *(PRD REQ-02)*: User gets reproducible, leak-free 5-fold `StratifiedGroupKFold` partitions grouped strictly by `session_id` (sequential captures within `folder_chain` separated by `≤ 10 min`) and stratified by weight band (`<500 g`, `500–1000 g`, `>1000 g`), guaranteeing zero session overlap across folds and balanced heavy-bird (`>1000 g`) representation persisted in `manifest_clean.parquet`.

### Classical PDI Engine & Watershed

- [x] **PDI-01** *(PRD REQ-03)*: Pipeline loads any `Path` or uploaded `bytes` through a single shared Pillow EXIF-first loader (extracting EXIF/Xiaomi `39321` before mandatory `ImageOps.exif_transpose()`, banning raw `cv2.imread`/`cv2.imdecode`) and runs configurable classical OpenCV preprocessing: aspect-ratio-preserving downscale to canonical working resolution (`768–960 px` long edge with tracked `scale_factor`), edge-preserving smoothing (Bilateral/Median filter), and local contrast normalization via CLAHE on the CIE Lab `L` channel.
- [x] **PDI-02** *(PRD REQ-04)*: Pipeline isolates the single-bird silhouette using strictly classical PDI (no deep learning) across all three visual scenarios (SF-400 scale on straw, top-down white plastic bucket including low-contrast white-on-white `>1000 g` birds, and free/oblique straw litter), explicitly suppressing the SF-400 LCD display, buttons, and circular platter before morphological open/close, dominant connected-component selection, and contour refinement.
- [x] **PDI-03** *(PRD REQ-05)*: Pipeline extracts features strictly inside the segmented bird mask (`bird_mask == 255`, verified by an automated outside-mask pixel-perturbation invariance test): geometric morphometrics (projected area, perimeter, fitted ellipse major/minor axes, eccentricity, convex area, solidity, circularity), scale-invariant chromatic plumage cues (yellow-down vs. white-plumage ratio, Lab/HSV statistics), and EXIF optical features (`focal_length_mm`, `zoom_multiple`, and optically normalized area validated empirically on the 114 paired dual-lens sessions without raw resolution leakage), plus capped-concurrency (`n_jobs=4`) batch extraction to `data/processed/features_clean.parquet`.
- [x] **PDI-04** *(PRD REQ-06)*: Single-bird PDI pipeline returns 4 encoded visual "X-Ray" inspection artifacts alongside the feature vector: (1) EXIF-oriented original, (2) preprocessed/CLAHE-Lab image, (3) clean binary bird mask (LCD excluded), and (4) final overlay displaying refined contour, fitted ellipse, and major/minor axes.
- [x] **PDI-05** *(PRD REQ-07 + `docs/proposta_pdi.md.pdf`)*: Dedicated classical Marker-based Watershed pipeline (`scipy.ndimage.distance_transform_edt` + Gaussian smoothing + scale-adaptive `h_maxima` / `peak_local_max` + `skimage.segmentation.watershed`, with chromatic red-feeder pre-suppression) separates touching birds across all 4 quarantined multi-bird folders (`417`, `757`, `823`, `1520`) and custom user uploads, returning basin/boundary maps, segmented bird count, per-bird pixel areas, and per-instance mask-gated morphometric/chromatic/optical features to enable illustrative individual weight estimation in `WEB-04`.

### Tabular Regression & Stratified Validation

- [ ] **MOD-01** *(PRD REQ-08)*: Training pipeline fits and compares interpretable tabular regressors (log-target Ridge/Huber, Random Forest, and LightGBM/Gradient Boosting — zero deep learning) across the locked 5-fold `session_id` splits using strictly PDI-extracted features and validated optical descriptors, evaluating capped inverse session/stratum weighting and fold-internal Duan smearing.
- [ ] **MOD-02** *(PRD REQ-09)*: Evaluation pipeline scores all models out-of-fold across the 5 `session_id` folds and writes `models/evaluation_report.json` with MAE (g), RMSE (g), MAPE (%), $R^2$, and Mean Bias Error globally and stratified by weight band (`<500 g`, `500–1000 g`, `>1000 g`), lens (`4.25 mm` vs. `1.65 mm`), and scenario (dorsal bucket vs. scale/litter), including a quantitative WITH vs. WITHOUT EXIF optical ablation and within-session dual-lens consistency/counterfactual check.
- [ ] **MOD-03** *(PRD REQ-10)*: Artifact publisher exports the winning production pipeline bundled with all 5 out-of-fold `fold_pipelines` (enabling honest out-of-fold live preset evaluation), locked `PDIConfig` SHA-256 hash, exact `feature_names`, version metadata, and per-band expected error intervals into `models/weight_regressor.joblib`, plus curated out-of-fold `models/presets_catalog.json`.

### Interactive Pitch Web Studio (Local API & Frontend)

- [ ] **WEB-01** *(PRD REQ-11)*: Local FastAPI backend exposes `/api/predict`, `/api/watershed-demo`, `/api/presets`, `/api/metrics` (and `/api/health`) with Pydantic v2 validation, startup `lifespan` caching, `async def` bounded upload intake offloading CPU PDI/ML via `run_in_threadpool`, end-to-end inference latency `< 500 ms` per image on local CPU, and graceful handling of EXIF-stripped JPEG or RGBA PNG uploads.
- [ ] **WEB-02** *(PRD REQ-12)*: User can upload any broiler image in the React + TypeScript + Vite + Tailwind CSS Inference & "X-Ray" PDI Studio (with automatic EXIF focal/zoom reading, missing-EXIF warning pill, and manual lens/zoom override) to inspect all 4 PDI stages side-by-side, morphometric and chromatic feature values (including yellow-down vs. white-plumage indicator), estimated weight (g) with expected error band, Cobb Avian 48 developmental age stage, and a live client-side simulated flock average/uniformity tracker.
- [ ] **WEB-03** *(PRD REQ-13)*: User can one-click run curated out-of-fold validation presets spanning the growth curve (young chick ~180 g, intermediate ~550 g, heavy broiler ~1,600 g, and hard white-on-white dorsal bucket `>1,000 g`) showing Ground Truth weight vs. Predicted weight, absolute error (g), and `%` error live during the pitch.
- [ ] **WEB-04** *(PRD REQ-14 + `docs/proposta_pdi.md.pdf`)*: User can explore a dedicated Watershed & Future Vision tab demonstrating interactive Marker-based Watershed separation on multi-bird scenes (`417`, `757`, `823`, `1520`) or custom uploads with separated bird counts, per-bird pixel areas, and illustrative per-bird estimated weights (g) + scene mean weight (g) (displaying an explicit disclaimer that folders `417`/`757`/`823`/`1520` carry only group-sum ground truth and transparently comparing predicted sum vs. group-sum label), alongside an architectural comparison panel contrasting the current handheld/bucket MVP with the conceptual v2 fixed ceiling-camera barn roadmap (`BARN-01`).
- [ ] **WEB-05** *(PRD REQ-15 + `docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md` + `docs/proposta_pdi.md.pdf`)*: User can inspect a Validation Metrics & Forensic Audit Dashboard displaying 5-fold cross-validation KPIs (contextualized alongside the literature `R² > 0.9` benchmark), an interactive Actual vs. Predicted scatter plot with $y=x$ reference line, optical calibration ablation charts, stratified error tables by weight band/lens/scenario, and the forensic data curation funnel (`1,753 → 1,738` clean, explicitly reconciling the Mendeley published claim of `"1,772 images / daily for 45 days"` against the disk-audited `9` dates / `~7–29 days` / `116–2,093 g` range and disclosing the `801–1,216 g` coverage gap).

## v2 Requirements

Deferred to post-pitch releases (v1.x / v2+). Tracked for roadmap visibility but excluded from the 2026-11-28 MVP freeze.

### Post-Pitch Hardening (v1.x)

- **HARD-01**: Offline automated SF-400 LCD OCR label audit across all Scenario A images solely to flag any remaining 10 g / 100 g folder-name typos beyond `367 → 267 g`.
- **HARD-02**: Classical geometric `mm/pixel` scale calibration via Hough/contour detection of the white plastic bucket's inner rim on Scenario B dorsal images.
- **HARD-03**: Motion-blur sharpness quality gate (Laplacian variance score) flagging slow-shutter blurry frames with a confidence warning before prediction.
- **HARD-04**: Leave-One-Date-Out diagnostic cross-validation quantifying temporal interpolation error across the 8-day collection hiatus (`2022-04-03 → 2022-04-11`).
- **HARD-05**: Batch folder upload and downloadable CSV/PDF flock weight & uniformity report export.

### Commercial Barn Deployment (v2+)

- **BARN-01**: Fixed overhead ceiling-camera intrinsic calibration replacing handheld EXIF zoom compensation with known constant height ($Z_{const}$) and direct $\text{cm}^2$ surface area estimation.
- **BARN-02**: Late-cycle commercial slaughter weight dataset expansion (`30–45 days`, `2.1–3.2 kg`) beyond the current `2,093 g` upper bound.
- **BARN-03**: Edge hardware packaging (Raspberry Pi / industrial smart camera) exploiting classical PDI's CPU-only footprint.

## Out of Scope

Explicitly excluded from v1 code. Documented to enforce PRD `MUST-NOT` prohibitions and protect the 2026-11-28 deadline.

| Feature | Reason |
|---------|--------|
| Black-box Deep Learning segmentation or end-to-end weight regression (CNNs, YOLO, ViTs, SAM, Mask R-CNN) | Strictly prohibited (`PROIBIÇÃO 1`) — academic evaluation specifically measures mastery and interpretability of classical PDI. |
| SF-400 scale LCD OCR or any pixel/intensity feature extracted outside the segmented bird mask | Strictly prohibited (`PROIBIÇÃO 2`) — textbook visual data leakage that invalidates field generalization. |
| Random image-level `train_test_split` or ungrouped K-Fold cross-validation | Strictly prohibited (`PROIBIÇÃO 3`) — burst captures (~9 photos/session) cause severe split leakage and fake $R^2$. |
| Committing `data/raw/`, `data/processed/`, `models/*.joblib`, or sample `.jpg` files to Git (or listing 1,753 raw image paths in prompts) | Strictly prohibited (`PROIBIÇÃO 4`) — 3.54 GiB data hygiene and LLM context discipline. |
| Multi-frame video object tracking, barn population density heatmaps, and immobile/sick-bird sanitary alerts | Strictly out of v1 code scope (`PROIBIÇÃO 5`) — presented exclusively as conceptual roadmap architecture in `WEB-04`. |
| Aspect-ratio-distorting square resize (`20:9` → `224×224`) or scale-altering random crop augmentation | Destroys true bird morphometry (`2.2×` stretch) and breaks size-to-mass proportionality. |
| User authentication, SQL database, RTSP live camera ingestion, or cloud hosting | Operational overhead unnecessary for a local pitch MVP due 2026-11-28. |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| PDI-01 | Phase 2 | Complete |
| PDI-02 | Phase 2 | Complete |
| PDI-03 | Phase 2 | Complete |
| PDI-04 | Phase 2 | Complete |
| PDI-05 | Phase 2 | Complete |
| MOD-01 | Phase 3 | Pending |
| MOD-02 | Phase 3 | Pending |
| MOD-03 | Phase 3 | Pending |
| WEB-01 | Phase 4 | Pending |
| WEB-02 | Phase 4 | Pending |
| WEB-03 | Phase 4 | Pending |
| WEB-04 | Phase 4 | Pending |
| WEB-05 | Phase 4 | Pending |

**Coverage:**

- v1 requirements: 15 total
- Mapped to phases: 15 mapped
- Unmapped: 0 unmapped

---
*Requirements defined: 2026-11-20*
*Last updated: 2026-11-23 after Phase 2 completion*
