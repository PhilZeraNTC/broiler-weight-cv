# Feature Research

**Domain:** Interpretable Classical Digital Image Processing (PDI) Broiler Live-Weight Estimation & Interactive Academic/Product Pitch Web App (*AvianScale CV*)
**Researched:** 2026-11-20
**Confidence:** HIGH (grounded in closed-scope [PRD.md](../../PRD.md), forensic dataset audit [DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md](../../docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md), seminal PLF literature, and verified commercial poultry vision benchmarks)

---

## Executive Framing

*AvianScale CV* serves **two tightly coupled audiences** during its 2026-11-30 pitch:

1. **Technical / Academic Evaluators (PDI & ML Rigor):** Will actively probe for shortcut learning (SF-400 LCD reading, random burst split leakage, lens-to-weight confounding) and demand proof of authentic classical image processing mastery without black-box neural networks.
2. **Product / Agroindustry Stakeholders (Paraná Poultry Integration Context):** Care about eliminating stressful manual catch-and-weigh sampling, tracking flock average weight and uniformity for Feed Conversion Ratio (FCR) and slaughter scheduling, and seeing a credible low-cost path from single-bird demo to overhead barn deployment.

Every feature below is categorized and prioritized through both lenses, bounded strictly by the **2026-11-28** code-freeze deadline (8 calendar days) and PRD `REQ-01` through `REQ-15` + `PROIBIÇÃO 1–5`.

---

## Feature Landscape

### Table Stakes (Users & Evaluators Expect These)

Non-negotiable foundations. Missing any one of these invalidates either the scientific credibility of the model or the core demonstration promise.

| Feature | Why Expected | Complexity | Pitch Impact | Notes |
|---------|--------------|------------|--------------|-------|
| **Deterministic Forensic Dataset Ingestor (`REQ-01`)** | Raw dataset contains silent traps that crash or corrupt naive loaders (3 nested level-2 folders `171/173`, `190/191`, `190/193`; `1340_` underscore; 1,635 Eastern-Arabic digit filenames; 21 CP866 mojibake files in `547/`, `562/`; 100% corrupted EXIF `DateTime`; 81.86% rotated EXIF `Orientation != 1`; 15 contaminated images; `367→267 g` typo). | MEDIUM | HIGH (Foundation for Audit credibility) | Must follow Forensic Audit §9 verbatim without modifying `data/raw/`. Outputs immutable `data/processed/manifest_clean.parquet` (1,738 imgs, 160 unique weights, 188 sessions) and `data/processed/manifest_quarantine.parquet` (15 imgs). Must extract Xiaomi proprietary tag `39321` (`0x9999`) for `zoomMultiple` alongside `FocalLength`. |
| **Anti-Leakage 5-Fold `session_id` Split (`REQ-02`)** | Average ~9 photos/session (~91.5 s bursts, 24 near-duplicate pairs dHash ≤ 4) mean any image-level split leaks the same bird across train/val. Evaluators immediately check validation hygiene. | LOW | HIGH (Prevents methodological disqualification) | Group sequential captures within `folder_chain` separated by `≤ 10 min` (`600 s`) into `session_id`. Assign deterministic 5-fold `StratifiedGroupKFold` stratified across `<500 g`, `500–1000 g`, `>1000 g` (protecting the sparse `>1,000 g` tail from fold starvation). Persist `fold` column in `manifest_clean.parquet`. |
| **Classical Aspect-Preserving Preprocessing & CLAHE (`REQ-03`)** | Smartphone images are `20:9` (`2.217:1`), shot at slow shutters (`1/20–1/50 s`, ISO `401.5 ± 164.7`) with dark frames (luminance down to `82.2`) and white-on-white bucket glare. Distorting aspect ratio destroys morphometry. | LOW | MEDIUM (Visible in X-Ray Stage 2) | Apply `ImageOps.exif_transpose()` first, resize with preserved aspect ratio (*letterbox* or known uniform downscale factor for `< 500 ms` budget — e.g. long side `640–800 px`), apply edge-preserving smoothing (Bilateral or Median filter), and run **CLAHE** exclusively on CIE Lab `L` channel to normalize local contrast without distorting chroma. |
| **Anti-Leakage Bird Silhouette Segmentation without SF-400 LCD (`REQ-04`)** | Core technical requirement (`PROIBIÇÃO 1 & 2`): isolate bird contour cleanly from straw litter, white bucket, and **completely strip out the SF-400 LCD display, buttons, and circular scale pan** using classical PDI only. | **HIGH** *(Primary Technical Bottleneck)* | **CRITICAL** (Visible proof in X-Ray Stage 3) | Must handle 3 single-bird visual scenarios (A: SF-400 scale on litter; B: top-down white plastic bucket ~351 clean imgs with low-contrast white-on-white birds `>1 kg`; C: free bird on straw/oblique angles). Combine CIE Lab (`a, b` chroma + `L` edges) & HSV (`S, V`) thresholding, Otsu/adaptive masks, structural/geometric suppression of rectangular LCD/white scale base, morphological open/close + hole fill, and dominant connected component + contour refinement. |
| **Morphometric, Chromatic & EXIF Optical Feature Extraction (`REQ-05`)** | Literature standard (`Mollah 2010`, `Amraei 2017`) extended for handheld multi-lens reality: projected pixel area alone shifts by **`6.6×`** between Wide (`4.25 mm`) and Ultrawide (`1.65 mm`) lenses, and `3.8%` use digital zoom up to `2.0×`. | MEDIUM | HIGH (Core interpretability narrative) | Extract strictly **inside the bird mask**: (1) *Geometric:* projected area ($px^2$), perimeter ($px$), fitted ellipse major/minor axes, eccentricity, convex area, solidity, circularity, aspect ratio; (2) *Chromatic (scale-invariant age/feathering proxy):* yellow-down vs. white-plumage pixel ratio, Lab/HSV mean/std/quantiles (tracks developmental transition from 7 d yellow chick to 29 d white broiler); (3) *Optical EXIF:* `focal_length_mm`, `zoom_multiple`, normalized area $A_{norm} = A_{px} / (f_{mm} \cdot \text{zoom})^2$ and normalized linear axes $L / (f_{mm} \cdot \text{zoom})$. |
| **Interpretable Tabular Regression & Optical Ablation Suite (`REQ-08`, `REQ-09`, `REQ-10`)** | Proves classical features accurately predict live weight and quantitatively demonstrates why EXIF optical calibration matters, using honest stratified metrics. | MEDIUM | **CRITICAL** (Numbers on pitch slides & Audit tab) | Train and compare log-target Ridge/Huber (directly optimizes relative error / MAPE and models power-law area-to-mass scaling $W \propto A^\beta$), Random Forest, and LightGBM/Gradient Boosting across all 5 `session_id` folds. Emit `models/evaluation_report.json` with MAE (g), RMSE (g), MAPE (%), $R^2$ globally and stratified by weight band (`<500`, `500–1000`, `>1000 g`), lens (`4.25 mm` vs `1.65 mm`), and scenario (bucket vs. scale/litter), plus **WITH vs. WITHOUT EXIF optical features ablation**. Export winner + expected per-band error intervals to `models/weight_regressor.joblib`. |
| **FastAPI Local Inference Engine `< 500 ms` (`REQ-11`)** | Interactive pitch demo feels broken if each click stalls for seconds. Clean REST contract decouples Python CV/ML from React UI. | LOW | HIGH (Live demo fluidity) | Expose `/api/predict`, `/api/watershed-demo`, `/api/presets`, `/api/metrics` with Pydantic validation. Load `weight_regressor.joblib`, `evaluation_report.json`, and preset metadata into memory at startup. Run full OpenCV PDI + feature extraction + tabular inference + base64 JPEG/PNG encoding of the 4 X-ray stages under **500 ms** on local CPU (achievable at ~100–250 ms by working on aspect-preserved resized frames of ~640–800 px max dimension while recording exact scale factor). |
| **4-Stage "X-Ray" PDI Inference Studio UI (`REQ-06`, `REQ-12`)** | Visual proof that the system is *not* a black box. Lets evaluators inspect every classical PDI transformation step-by-step on any uploaded or preset image. | MEDIUM | **CRITICAL** (Hero screen of the pitch) | React + TS + Vite + Tailwind workspace supporting drag-and-drop image upload (auto-parsing EXIF focal/zoom with manual override controls), displaying the 4 synchronized visual panels: **(1) EXIF-Oriented Original**, **(2) Preprocessed / CLAHE-Lab**, **(3) Clean Binary Bird Mask (LCD Excluded)**, and **(4) Final Overlay (Contour + Fitted Ellipse + Major/Minor Axes)**, paired with extracted feature table, estimated weight (g) + error band, and inferred Cobb Avian 48 age stage (~7–29 days). |

---

### Differentiators (Competitive & Pitch Advantage)

Features that elevate *AvianScale CV* from a solid academic notebook into a memorable, bulletproof product pitch. Every differentiator below either directly fulfills a PRD showcase requirement (`REQ-07`, `REQ-12–15`) or amplifies it at minimal incremental cost.

| Feature | Value Proposition | Complexity | Pitch Impact | Notes |
|---------|-------------------|------------|--------------|-------|
| **Curated Pitch Preset Gallery with Ground Truth vs. Predicted & % Error (`REQ-13`)** | Eliminates live-demo risk (no fumbling through folders, instant `< 500 ms` showcase) and builds immense trust by showcasing held-out validation samples across the full growth curve — including an honest hard edge case. | LOW | **CRITICAL** (Centerpiece live walkthrough) | Curate 4–6 real out-of-fold validation images covering: (1) Young yellow chick (~180 g, ~7 d), (2) Mid-transition broiler (~550 g, ~14–15 d), (3) Heavy white broiler (~1,600 g, ~25–29 d), (4) Hard case: white-on-white dorsal bucket (`>1,000 g`), and optionally (5) Ultrawide (`1.65 mm`) vs. Wide (`4.25 mm`) comparison of the same weight range. Display Ground Truth weight, Predicted weight, absolute delta (g), and % error badges live. |
| **Interactive Watershed Multi-Bird Separation Tab + Overhead Barn Roadmap (`REQ-07`, `REQ-14`)** | Directly answers the #1 agroindustry objection ("In a real barn, chickens huddle and touch each other!"). Turns the 13 quarantined multi-bird images into a showcase asset instead of discarded waste. | MEDIUM | **VERY HIGH** (Connects PDI theory to commercial barn reality) | Dedicated classical pipeline: preprocessing → foreground mask → Euclidean *Distance Transform* → Gaussian/morphological smoothed local maxima markers → *Marker-based Watershed*. Preloaded with the 4 real multi-bird folders (`417` [2 chicks in red feeder], `757` [2 birds], `823` [4 chicks], `1520` [2 birds]) plus custom user upload support. Outputs colorized basin/boundary map, separated bird count, and individual pixel area per bird. Accompanied by a clean side-by-side **Architecture Roadmap Panel** contrasting the current handheld/bucket MVP vs. fixed overhead ceiling-camera deployment (`PROIBIÇÃO 5` stays strictly conceptual here). |
| **Live Simulated Flock Average Weight & Uniformity Tracker (`REQ-12`)** | Translates single-image computer vision predictions into the exact KPIs Paraná poultry integrators actually manage: **Flock Mean Weight (g)**, **Uniformity / CV (%)**, sample count ($n$), and age-stage breakdown. | LOW | HIGH (Business / product bridge) | Client-side accumulator in the Inference Studio: every analyzed bird (via preset click or upload) can be added to the active "Lote Simulado" ledger (with clear/reset button), updating running mean weight, min–max spread, standard deviation, and CV% in real time. Zero backend database complexity required. |
| **Forensic Data Audit & Optical Ablation Dashboard (`REQ-15`)** | Differentiates a rigorous engineering team from "notebook ML" projects. Visually proves the impact of forensic curation, anti-leakage CV, and EXIF optical calibration in one clean tab. | MEDIUM | **VERY HIGH** (Wins academic & technical judges) | Interactive charts (Recharts / lightweight SVG) powered by `GET /api/metrics`: (1) Actual vs. Predicted scatter plot colored by lens (`4.25 mm` vs `1.65 mm`) or weight band; (2) Optical Calibration Ablation bar comparison (MAE/MAPE/$R^2$ WITH vs. WITHOUT EXIF `focal_length`/`zoom`); (3) Stratified metrics table (`<500`, `500–1000`, `>1000 g`; Wide vs. Ultrawide; Bucket vs. Scale/Litter); (4) Forensic Audit Funnel visual (`1,753` raw → `15` quarantined + `367→267 g` LCD fix + `81.9%` EXIF orientation fix + CP866/Arabic decoded → `1,738` clean / `188` leak-free sessions) + explicit disclosure of the `801–1,216 g` dataset gap. |
| **Interactive Optical "What-If" Lens/Zoom Override Control (`REQ-12`)** | Turns the abstract EXIF calibration equation $A_{px}/(f_{mm}\cdot \text{zoom})^2$ into an unforgettable 10-second live demonstration. | LOW | HIGH (Memorable Q&A proof) | Already supported by `REQ-12`'s manual lens/zoom inputs: provide clear segmented toggle buttons (`Auto EXIF` / `Wide 4.25mm 1.0×` / `Ultrawide 1.65mm 0.6×` + numeric zoom override) so the presenter can toggle optical metadata live and show both the raw pixel area staying constant while optically normalized area and predicted weight respond predictably. |
| **Chromatic Feathering Developmental Readout (`REQ-05`, `REQ-12`)** | Showcases clever classical domain engineering: even without a fixed tripod distance, yellow-down vs. white-plumage ratio acts as a **scale-invariant biological age signal** (Cobb Avian 48 transitions predictably from 100% yellow at ~7 d to dense white at ~25 d). | LOW | MEDIUM-HIGH (Domain-specific cleverness) | Render a compact visual bar/pill in the X-Ray feature card showing `% Penugem Amarela` vs. `% Plumagem Branca` alongside the inferred developmental stage (`Inicial ~7–10 d`, `Transição ~11–17 d`, `Crescimento ~24–29 d`). |

---

### Anti-Features (Commonly Requested, Explicitly Prohibited or Counterproductive)

Features that look appealing on the surface but violate PRD prohibitions, destroy scientific validity, or jeopardize the 2026-11-28 delivery deadline.

| Feature | Why Requested | Why Problematic | PRD / Audit Mapping | Alternative |
|---------|---------------|-----------------|---------------------|-------------|
| **Black-Box Deep Learning Segmentation or End-to-End Regression (YOLO-Seg, SAM, Mask R-CNN, ConvNeXt, ViT)** | Higher raw segmentation IoU on messy litter and white-on-white buckets; standard reflex in modern CV projects. | Destroys the primary academic evaluation goal (demonstrating classical PDI mastery), eliminates step-by-step visual interpretability, hides optical/LCD shortcuts, bloats dependencies (PyTorch/CUDA), and threatens the `< 500 ms` local CPU budget. | **`PROIBIÇÃO 1`** (Strict MUST-NOT) | Classical OpenCV/scikit-image pipeline (CLAHE Lab, multi-space Lab/HSV thresholding, morphology, contour/ellipse fitting, Marker-based Watershed) + interpretable tabular regressors (Ridge/Huber, Random Forest, LightGBM). |
| **SF-400 Scale LCD OCR or Full-Frame / Outside-Mask Pixel Features** | Would trivially yield ~99% apparent accuracy on Scenario A images; scale pan or pen offers an easy scale shortcut. | Textbook **visual data leakage**: the model learns to read the scale digits or pan diameter instead of estimating bird mass, failing completely in real barns without SF-400 scales. | **`PROIBIÇÃO 2`** (Strict MUST-NOT) | Strict bird-only silhouette masking (`REQ-04`) verified visually in X-Ray Stage 3, with zero features computed outside the mask. Offline LCD verification is restricted solely to forensic dataset curation (`367 → 267 g`). |
| **Random Image-Level `train_test_split` or Unstratified K-Fold** | Default 1-line boilerplate in scikit-learn tutorials. | Leaked ~9-shot bursts (~91.5 s, 24 near-duplicate pairs dHash ≤ 4) put virtually identical photos of the same bird in both train and test, creating fake $R^2 > 0.98$ metrics and masking collapse on the rare `>1,000 g` tail. | **`PROIBIÇÃO 3`** (Strict MUST-NOT) | Mandatory 5-fold `StratifiedGroupKFold` grouped by `session_id` (`>10 min` gap) and stratified by weight bins (`<500`, `500–1000`, `>1000 g`). |
| **Committing Raw/Processed Data or Listing `data/raw/` Files in Repo/Prompts** | Perceived convenience for "plug-and-play" cloning or agent exploration. | Bloats Git by 3.54 GiB, violates data hygiene, and floods LLM context with 1,753 Arabic/CP866 paths. | **`PROIBIÇÃO 4`** (Strict MUST-NOT) | Keep `data/raw/`, `data/processed/`, and `models/*.joblib` in `.gitignore`; commit deterministic ingestion code (`REQ-01`) + single-source-of-truth forensic doc. |
| **Multi-Frame Video Tracking (*Object Tracking*), Barn Density Heatmaps, & Immobile/Sick Bird Alerts** | Headline features in commercial systems (e.g., Fancom *eYeNamic*); tempting for business pitch flair. | Zero video or multi-frame barn footage exists in the dataset; building simulated/fake tracking wastes scarce engineering hours before 2026-11-28 and dilutes focus from verifiable classical PDI. | **`PROIBIÇÃO 5`** (Strict MUST-NOT for v1 code) | Present exclusively as a clean **Architectural Roadmap Panel** inside the Watershed tab (`REQ-14`), clearly separating working v1 capabilities from v2 fixed-ceiling-camera evolution. |
| **Square Stretch Resizing (`20:9` → `224×224`) & Scale-Altering Augmentations (`RandomResizedCrop`)** | Standard image preprocessing habit imported from ImageNet classification. | Compressing `2.217:1` images into a square distorts bird morphometry by `2.2×`, corrupting area, perimeter, and ellipse eccentricity; random zoom destroys apparent size-to-weight proportionality. | Forensic Audit §8.3 | Always preserve aspect ratio (*letterbox* or uniform scaling with tracked scale factor) and restrict any augmentation strictly to rotation/flip/mild photometric shifts. |
| **Commercial Slaughter-Weight Extrapolation (`2.1–3.2 kg`) Without Domain Guardrails** | Producers slaughter Cobb Avian broilers around 38–45 days (~2.5–3.2 kg), so someone may upload or ask about heavier birds. | Dataset ends at `2,093 g` (~29 days); tree ensembles (RF/LightGBM) plateau flat at training max, while unbounded linear models extrapolate erratically. | Forensic Audit §2, §6.2, §7 | Display an explicit calibrated operational range indicator (`116–2,093 g / ~7–29 dias`) and document late-cycle data collection as a v2 requirement in the Audit Dashboard. |
| **User Accounts, Persistent SQL Database, Live RTSP Camera Streaming, or Cloud Deployment** | Common web-app boilerplate expectations. | Pure operational overhead for a local pitch demo with an 8-day deadline; introduces auth/network failure modes during live presentation. | Scope / Timeline Constraint (2026-11-28) | Run 100% locally (`uvicorn` + Vite dev/static build), keep batch flock tracker in React client state, and serve curated presets directly from local validated assets. |

---

## Feature Dependencies

Build order is strictly governed by a 4-layer data-to-UI DAG matching the PRD's 4 phases. No downstream ML or UI showcase can produce honest, leak-free outputs until upstream artifacts exist.

```
LAYER 1: DATA CURATION & MANIFEST (Phase 1)
  [REQ-01: Forensic Ingestor & Quarantine]
      ├──produces──> manifest_clean.parquet (1,738 imgs, EXIF focal/zoom, clean weights)
      ├──produces──> manifest_quarantine.parquet (15 imgs: 13 multi-bird + 2 mislabels)
      └──requires──> (none — consumes read-only data/raw/ per Audit §9)
             │
             ▼
  [REQ-02: session_id Grouping & 5-Fold StratifiedGroupKFold]
      └──requires──> [REQ-01] (needs reconstructed capture_dt & clean weights)

LAYER 2: CLASSICAL PDI ENGINE (Phase 2)
  [REQ-03: Aspect-Preserving Preprocessing & CLAHE-Lab]
      └──requires──> [REQ-01] (EXIF orientation metadata & image paths)
             │
             ├──────────────────────────────┐
             ▼                              ▼
  [REQ-04: Anti-Leakage Bird           [REQ-07: Marker-Based Watershed
   Segmentation (No SF-400 LCD)]        Multi-Bird Separation Pipeline]
      └──requires──> [REQ-03]              └──requires──> [REQ-03] + [REQ-01 quarantine: 417,757,823,1520]
             │
             ├──────────────────────────────┐
             ▼                              ▼
  [REQ-05: Morphometric, Chromatic     [REQ-06: 4-Stage X-Ray Visual
   & EXIF Optical Feature Vector]       Artifact Encoder (4 PNG/JPEG)]
      └──requires──> [REQ-04] + [REQ-01]   └──requires──> [REQ-03] + [REQ-04] + [REQ-05 (ellipse)]

LAYER 3: REGRESSION & STRATIFIED VALIDATION (Phase 3)
  [REQ-08: Interpretable Tabular Regressors (Ridge/Huber, RF, LightGBM)]
      └──requires──> [REQ-02 (folds)] + [REQ-05 (1,738 feature rows)]
             │
             ├──────────────────────────────┐
             ▼                              ▼
  [REQ-09: Stratified 5-Fold Report    [REQ-10: Production Model Export
   & EXIF Optical Ablation JSON]        (weight_regressor.joblib + Error Bands)]
      └──requires──> [REQ-08] + [REQ-02]   └──requires──> [REQ-08] + [REQ-09]

LAYER 4: INTERACTIVE WEB PITCH STUDIO (Phase 4)
  [REQ-11: FastAPI Local REST Backend (< 500 ms CPU)]
      └──requires──> [REQ-06] + [REQ-07] + [REQ-09] + [REQ-10] + [REQ-02 (out-of-fold presets)]
             │
             ├────────────────────┬────────────────────┬────────────────────┐
             ▼                    ▼                    ▼                    ▼
  [REQ-12: X-Ray PDI      [REQ-13: Curated       [REQ-14: Watershed &    [REQ-15: Metrics &
   Inference Studio]       Pitch Presets]         Barn Roadmap Tab]      Forensic Audit Tab]
      └──requires──> REQ-11   └──requires──> REQ-11  └──requires──> REQ-11  └──requires──> REQ-11
      └──enhances──> [Flock Average Tracker]         └──requires──> REQ-07  └──requires──> REQ-09
      └──enhances──> [Optical "What-If" Toggle]

EXPLICIT CONFLICTS (Guardrails):
  [PROIBIÇÃO 1: Black-box CNN/YOLO/SAM] ──conflicts──> [REQ-04, REQ-06, REQ-08]
  [PROIBIÇÃO 2: LCD OCR / Outside-Mask] ──conflicts──> [REQ-04, REQ-05, REQ-09]
  [PROIBIÇÃO 3: Random Image Split]     ──conflicts──> [REQ-02, REQ-09, REQ-13]
  [PROIBIÇÃO 5: Video Tracking/Alerts]  ──conflicts──> [8-day MVP Schedule] (roadmap visual only in REQ-14)
```

### Dependency Notes

- **Phase 1 (`REQ-01 → REQ-02`) MUST land first and remain immutable:** Every downstream feature table, fold assignment, held-out preset selection, and forensic audit metric references `manifest_clean.parquet` and `manifest_quarantine.parquet`. Any change to `session_id` or `fold` invalidates cached features and out-of-fold validation guarantees.
- **`REQ-04` (Anti-Leakage Segmentation) is the critical-path risk node:** Because `REQ-05` (features), `REQ-06` (X-Ray visuals), and `REQ-08` (model accuracy) all depend directly on mask quality across 3 visually diverse scenarios (SF-400 scale, white-on-white bucket, straw litter), Phase 2 must validate segmentation masks qualitatively across all 3 scenarios and all weight bands before batch-extracting features for all 1,738 images.
- **`REQ-07` (Watershed Multi-Bird Demo) is decoupled from single-bird regression (`REQ-08–10`):** Once `REQ-01` (quarantine list) and `REQ-03` (preprocessing) exist, `REQ-07` can be engineered and tuned independently on folders `417`, `757`, `823`, and `1520` without blocking Phase 3 model training.
- **`REQ-13` (Curated Presets) depends on `REQ-02` + `REQ-09` + `REQ-10`:** Presets must come strictly from **out-of-fold validation predictions** (or verified validation fold images) so Ground Truth vs. Predicted comparisons during the live pitch are scientifically honest — never memorized training examples.
- **UI tabs (`REQ-12`, `REQ-13`, `REQ-14`, `REQ-15`) can be built in parallel once `REQ-11` JSON contracts are frozen:** Defining Pydantic response schemas for `/api/predict`, `/api/watershed-demo`, `/api/presets`, and `/api/metrics` early lets frontend layout and data-visualization components progress cleanly against real or contract-verified payloads.

---

## MVP Definition

### Launch With (v1 — Target Code Freeze: 2026-11-28, Pitch: 2026-11-30)

Closed scope strictly covering `REQ-01` through `REQ-15`. Everything here earns its place in the 8-day sprint:

- [ ] **Forensic Data Pipeline (`REQ-01`, `REQ-02`)** — Reproducible Parquet manifests (`1,738` clean / `15` quarantined) with `367→267 g` correction, EXIF focal/zoom extraction, and 5-fold `StratifiedGroupKFold` by `session_id`.
- [ ] **Classical PDI Engine & 4-Stage X-Ray Visuals (`REQ-03`, `REQ-04`, `REQ-05`, `REQ-06`)** — Aspect-preserving resize + Bilateral/Median + CLAHE Lab + multi-space Lab/HSV anti-LCD bird segmentation + morphometric/chromatic/optical feature extraction + 4 encoded visual stage artifacts.
- [ ] **Watershed Multi-Bird Module (`REQ-07`)** — Distance Transform + smoothed local maxima + Marker-based Watershed working on `417`, `757`, `823`, `1520` and custom user uploads.
- [ ] **Interpretable Regression & Optical Ablation (`REQ-08`, `REQ-09`, `REQ-10`)** — 5-fold comparison of Ridge/Huber (log-target), Random Forest, and LightGBM/Gradient Boosting; `evaluation_report.json` with stratified metrics + EXIF ablation; production `weight_regressor.joblib`.
- [ ] **FastAPI Local Backend (`REQ-11`)** — `< 500 ms` CPU endpoints (`/api/predict`, `/api/watershed-demo`, `/api/presets`, `/api/metrics`) with Pydantic validation and EXIF auto-read + override.
- [ ] **Interactive Pitch Web App (`REQ-12`, `REQ-13`, `REQ-14`, `REQ-15`)** — 3-tab data-driven React/Tailwind studio:
  - *Tab 1 — Inference & X-Ray PDI Studio (`REQ-12` + `REQ-13`):* Upload + Curated Pitch Preset Gallery (Ground Truth vs. Predicted + % error, including white-on-white bucket case) + 4-stage X-Ray strip + feature/chromatic readout + Optical Lens/Zoom override + Live Simulated Flock Average Tracker.
  - *Tab 2 — Watershed Multi-Bird & Overhead Barn Roadmap (`REQ-14`):* Interactive separation of `417/757/823/1520` (and uploads) with count & per-bird areas + conceptual MVP-vs-Barn-Camera architectural comparison panel.
  - *Tab 3 — Validation Metrics & Forensic Audit (`REQ-15`):* 5-fold CV cards, Actual vs. Predicted scatter plot, Optical Calibration Ablation visual, stratified error breakdown, and Forensic Data Curation funnel.

### Add After Validation (v1.x — Post-Pitch Hardening)

Valuable enhancements that do **not** block the 2026-11-30 pitch and must not dilute the 8-day sprint:

- [ ] **Offline Automated SF-400 LCD Label Audit** — Trigger: post-pitch dataset publication or model accuracy push; run offline OCR solely to catch any remaining 10 g / 100 g folder-name typos beyond `367→267 g` (Audit §8.1).
- [ ] **Bucket-Bottom Classical Geometry Calibration (`mm/pixel`)** — Trigger: improving Scenario B (`~351` dorsal bucket images) via Hough/contour detection of the white bucket's known inner rim geometry to eliminate camera-height variance.
- [ ] **Motion-Blur Quality Gate (Laplacian Variance Score)** — Trigger: field reliability testing; flag blurry frames (common at `1/20–1/33 s` shutter speed, e.g. `2022-03-28`) with a low-confidence warning badge before returning a weight estimate.
- [ ] **Leave-One-Date-Out Diagnostic Validation** — Trigger: academic paper extension; quantify temporal interpolation accuracy across the 8-day gap (`2022-04-03 → 2022-04-11`).
- [ ] **Batch Folder Upload & CSV/PDF Flock Report Export** — Trigger: pilot testing with local Paraná poultry technicians.

### Future Consideration (v2+ — Commercial Overhead Barn Deployment)

Explicitly deferred (`PROIBIÇÃO 5` + dataset domain boundaries) and communicated transparently as the product vision in Tab 2 (`REQ-14`):

- [ ] **Fixed Overhead Barn Camera Calibration** — Replaces handheld EXIF zoom estimation with known fixed ceiling height ($Z_{const}$) and pinhole camera intrinsics, converting bird mask pixels directly to physical dorsal surface area ($\text{cm}^2$) without manual handling.
- [ ] **Continuous Multi-Frame Video Tracking & Flock Uniformity Curves** — Multi-bird tracking across frames to sample hundreds of birds per hour, rejecting occluded/bent postures automatically (similar to Mortensen 2016 / commercial PLF systems).
- [ ] **Late-Cycle Data Collection (`30–45 days` / `2.1–3.2 kg`)** — Expands training distribution to commercial slaughter weights absent from the Cobb Avian Mendeley dataset.
- [ ] **Barn Density Heatmaps & Welfare / Immobile Bird Alerts** — Spatial activity and clustering alerts (`PROIBIÇÃO 5`) for thermal stress or health anomalies.
- [ ] **Edge Device Deployment (Raspberry Pi / Industrial Smart Camera)** — Leveraging classical PDI's low CPU footprint (`< 500 ms`, no GPU required) for low-cost rural farm hardware.

---

## Feature Prioritization Matrix

| Feature | PRD Ref | User / Pitch Value | Implementation Cost | Priority | Build Phase |
|---------|---------|--------------------|---------------------|----------|-------------|
| Deterministic Forensic Dataset Ingestor & Quarantine | `REQ-01` | HIGH | MEDIUM | **P1** | Phase 1 |
| Anti-Leakage 5-Fold `session_id` Split | `REQ-02` | HIGH | LOW | **P1** | Phase 1 |
| Classical Aspect-Preserving Preprocessing & CLAHE | `REQ-03` | HIGH | LOW | **P1** | Phase 2 |
| Anti-Leakage Bird Silhouette Segmentation (No LCD) | `REQ-04` | **CRITICAL** | **HIGH** | **P1** | Phase 2 |
| Morphometric, Chromatic & EXIF Optical Features | `REQ-05` | HIGH | MEDIUM | **P1** | Phase 2 |
| 4-Stage "X-Ray" PDI Visual Artifact Generator | `REQ-06` | **CRITICAL** | LOW | **P1** | Phase 2 |
| Marker-Based Watershed Multi-Bird Pipeline | `REQ-07` | HIGH | MEDIUM | **P1** | Phase 2 |
| Interpretable Tabular Regressors (Ridge/Huber, RF, LightGBM) | `REQ-08` | HIGH | MEDIUM | **P1** | Phase 3 |
| Stratified 5-Fold Validation & Optical Ablation Report | `REQ-09` | **CRITICAL** | LOW | **P1** | Phase 3 |
| Serialized Production Model + Per-Band Error Intervals | `REQ-10` | HIGH | LOW | **P1** | Phase 3 |
| FastAPI Local Inference API (`< 500 ms` CPU) | `REQ-11` | HIGH | LOW | **P1** | Phase 4 |
| X-Ray PDI Studio UI + Upload & EXIF Auto-Read | `REQ-12` | **CRITICAL** | MEDIUM | **P1** | Phase 4 |
| Curated Live-Pitch Presets (Ground Truth vs. Pred & % Err) | `REQ-13` | **CRITICAL** | LOW | **P1** | Phase 4 |
| Watershed Multi-Bird Interactive Tab + Barn Roadmap Panel | `REQ-14` | HIGH | MEDIUM | **P1** | Phase 4 |
| Validation Metrics, Optical Ablation & Forensic Audit Dashboard | `REQ-15` | HIGH | MEDIUM | **P1** | Phase 4 |
| Live Simulated Flock Average Weight & Uniformity Tracker | `REQ-12` | HIGH | LOW | **P1** | Phase 4 |
| Interactive Manual Lens/Zoom "What-If" Toggle | `REQ-12` | MEDIUM-HIGH | LOW | **P2** | Phase 4 |
| Chromatic Feathering Stage Visual Bar | `REQ-05/12` | MEDIUM-HIGH | LOW | **P2** | Phase 4 |
| Motion-Blur Sharpness Confidence Warning (Laplacian) | Post-MVP | MEDIUM | LOW | **P3** | v1.x |
| Bucket-Bottom Geometric Reference Calibration (`mm/px`) | Post-MVP | MEDIUM | MEDIUM | **P3** | v1.x |
| Video Tracking, Density Heatmaps & Health Alerts | `PROIBIÇÃO 5` | HIGH (v2) | VERY HIGH | **P3** | v2+ |

**Priority key:**
- **P1:** Must have for 2026-11-28 MVP freeze & 2026-11-30 pitch
- **P2:** High-leverage pitch polish (low effort, add directly inside Phase 4 if P1 core is green)
- **P3:** Explicitly deferred post-pitch (v1.x / v2+)

---

## Competitor Feature Analysis

Positioning *AvianScale CV* against the three real alternatives evaluated by poultry producers and academic juries:

| Capability / Dimension | Traditional Manual & Floor Scales (*SF-400*, *Big Dutchman EasyCheck / Swing20*) | Commercial Overhead Vision / AI Platforms (*Fancom eYeNamic*, *Beakbook*, *Weight-Detect*) | Typical Academic Black-Box DL Baselines (*YOLO-Seg / ConvNeXt / ViT* on raw datasets) | **AvianScale CV (Our Classical PDI Approach)** |
|------------------------|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|------------------------------------------------|
| **Non-Invasive / Zero Bird Stress** | Manual catch-and-weigh stresses birds; voluntary platform scales suffer sampling bias (heavier/lethargic birds avoid stepping on scales). | Fully passive overhead monitoring across the barn. | Non-invasive image capture, though often relies on controlled lab setups. | **Non-invasive visual estimation** validated on real farm images; zero physical handling in target overhead workflow. |
| **Hardware Cost & Edge Deployability** | Low (manual SF-400) to moderate (IoT LoRa platform scales needing litter maintenance). | High CAPEX (proprietary overhead RGB/3D depth sensors, barn cabling, SaaS subscriptions). | High compute (GPU training/inference; heavy weights slow on low-cost farm CPUs). | **Ultra-low cost & CPU-native (`< 500 ms` local CPU)** using standard 2D RGB images and lightweight OpenCV + tabular ML. |
| **Visual Interpretability ("Why This Weight?")** | Direct mechanical reading (high trust, zero diagnostic vision). | Black-box proprietary cloud dashboards; users see curves, not segmentation mechanics. | Opaque neural activations; hard to verify if model looked at bird, scale LCD, or litter. | **100% transparent 4-stage "X-Ray" PDI studio** exposing original, CLAHE, binary bird mask, fitted ellipse, and physical descriptors live. |
| **Anti-Leakage Scientific Audit** | N/A (physical scale ground truth). | Validated on proprietary commercial flocks (closed benchmarks). | Frequently suffers silent split leakage (random burst splits) and LCD/scale shortcuts on public datasets. | **Forensically audited public benchmark:** 15 contaminated images quarantined, `367→267 g` fixed, strict LCD exclusion, 5-fold `session_id` CV. |
| **Multi-Lens / Optical Scale Calibration** | N/A. | Fixed overhead camera height eliminates focal-length ambiguity. | Rarely integrates EXIF focal/zoom; risks learning spurious `ultrawide → heavy bird` shortcuts. | **Explicit EXIF optical normalization** ($A_{px}/(f_{mm}\cdot\text{zoom})^2$) + quantitative ablation study proving calibration gain. |
| **Touching / Clumped Bird Separation** | Floor scales reject simultaneous multi-bird steps or require filtering algorithms. | Mature multi-bird segmentation (often 3D depth/Kinect or YOLO instance segmentation). | Requires dense polygon instance annotations to train YOLO-Seg/Mask R-CNN. | **Annotation-free classical Marker-based Watershed** (`REQ-07/14`) demonstrated interactively on real 2-to-4 bird scenes. |
| **Continuous 24/7 Barn Tracking & Welfare Alerts** | Daily/periodic weight logs; no spatial welfare tracking. | **Market leader:** 24/7 activity indices, migration heatmaps, growth curves, early disease alerts. | Out of scope for single-image papers. | **Honestly scoped to v2 roadmap** (`PROIBIÇÃO 5`); v1 delivers simulated flock average weight & uniformity tracker (`REQ-12`) + clear overhead migration blueprint (`REQ-14`). |

---

## Sources

### Project & Dataset Primary Sources
- [PRD.md](../../PRD.md) — Closed-scope product requirements (`REQ-01`–`REQ-15`), explicit prohibitions (`PROIBIÇÃO 1–5`), and 4-phase delivery plan targeting 2026-11-28 / 2026-11-30.
- [PROJECT.md](../PROJECT.md) — Core value proposition, Paraná poultry integration business context, and constraints.
- [DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md](../../docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md) — Complete forensic audit of *Broiler weights of Cobb Avian* (Abuhussein, Elmessery, Abd El-Wahhab & El-Soaly, Kafrelsheikh University, Mendeley Data DOI `10.17632/zrs8kk9dvr.1`, CC BY 4.0).

### Seminal Classical PDI Poultry Weight Literature
- **Mollah, M. B. R., Hasan, M. A., Salam, M. A., & Ali, M. A. (2010).** *Digital image analysis to estimate the live weight of broiler.* Computers and Electronics in Agriculture — Established 2D projected surface area pixels + log-transformed regression for broiler live-weight estimation.
- **Mortensen, A. K., Lisouski, P., & Ahrendt, P. (2016).** *Weight prediction of broiler chickens using 3D computer vision.* Computers and Electronics in Agriculture — Established range-based Watershed separation for clumped broilers, posture filtering, and continuous overhead weight distribution tracking.
- **Amraei, S., Mehdizadeh, S. A., & Salari, S. (2017 / 2018).** *Broiler weight estimation based on machine vision and artificial neural network / support vector regression* — Validated morphometric descriptors (area, perimeter, convex area, major/minor ellipse axes) extracted from segmented top-down broiler contours.

### Commercial Precision Livestock Farming (PLF) Benchmarks
- **Big Dutchman** (*EasyCheck* wireless floor scale, *Swing20* suspension scale, *BFN Fusion* analytics) — Commercial mechanical/IoT poultry weighing benchmark highlighting sampling bias and floor maintenance constraints.
- **Fancom BV** (*eYeNamic* & *iFarming*) — Commercial overhead ridge-mounted camera system for 24/7 flock distribution, activity indices, and welfare anomaly alerting.
- **Beakbook** — AgriTech poultry vision + IoT growth-curve prediction platform focused on flock average weight, standard deviation/uniformity, and harvest scheduling.
- **PLF Agritech** (*Weight-Detect™*) — Contactless overhead image-analysis weight estimation benchmark for livestock pen averages and daily gain tracking.

---
*Feature research for: AvianScale CV (Classical PDI Broiler Weight Estimation & Interactive Pitch MVP)*
*Researched: 2026-11-20*
