## Conflict Detection Report

### BLOCKERS (0)

None.

### WARNINGS (4)

[WARNING] Competing MVP application input modality (image or video vs. image-only upload)
  Found: docs/proposta_pdi.md.pdf (PRD) defines the MVP deliverable as an application receiving "uma imagem ou vídeo" and returning per-bird weight and flock mean (source: docs/proposta_pdi.md.pdf), whereas existing PRD.md (REQ-11, REQ-12, PROIBIÇÃO 5) and .planning/REQUIREMENTS.md (WEB-01, WEB-02, Out of Scope) scope v1 strictly to single-image uploads and exclude multi-frame video object tracking (source: PRD.md, .planning/REQUIREMENTS.md).
  Impact: Synthesis cannot pick between image+video and image-only without losing intent from either the academic proposal or the locked v1 timeline/anti-scope boundary (code freeze 2026-11-28).
  → Confirm keeping v1 strictly single-image upload (deferring video to conceptual roadmap per PROIBIÇÃO 5) or add a lightweight single-frame video sampler without object tracking before routing.

[WARNING] Competing Watershed multi-bird output contract (per-bird weight + flock mean vs. count + pixel area)
  Found: docs/proposta_pdi.md.pdf (PRD) chains Watershed separation directly into geometric feature extraction and weight regression so the app outputs per-bird estimated weight and flock average (source: docs/proposta_pdi.md.pdf), whereas existing PRD.md (REQ-07, REQ-14) and .planning/REQUIREMENTS.md (PDI-05, WEB-04) scope the Watershed demo to basin/boundary maps, separated bird count, and per-bird pixel area, accumulating simulated flock average separately across single-bird uploads in REQ-12/WEB-02 (source: PRD.md, .planning/REQUIREMENTS.md).
  Impact: Multi-bird dataset folders (417, 757, 823, 1520) only carry group-sum weights with no individual ground truth, so applying the single-bird regressor to separated Watershed masks requires an explicit product decision on whether to display illustrative weights or remain strictly at count + area.
  → Choose between keeping PDI-05/WEB-04 at count + per-bird pixel area only, or adding illustrative per-bird weight estimates + scene mean with a ground-truth disclaimer before routing.

[WARNING] Competing camera geometry assumption (fixed ceiling vertical camera vs. handheld dual-lens EXIF calibration)
  Found: docs/proposta_pdi.md.pdf (PRD) specifies estimating broiler weight from a fixed ceiling-mounted camera pointing vertically downward (source: docs/proposta_pdi.md.pdf), whereas existing PRD.md (§2, REQ-05, REQ-14) and .planning/REQUIREMENTS.md (PDI-03, WEB-04, BARN-01) implement handheld Xiaomi Redmi 10 dual-lens EXIF optical calibration for v1 and defer fixed overhead ceiling-camera deployment to v2 BARN-01 / conceptual comparison panel WEB-04 (source: PRD.md, .planning/REQUIREMENTS.md, docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md).
  Impact: The audited validation dataset consists of handheld dual-lens (4.25 mm vs. 1.65 mm), variable-zoom (0.60x–2.00x), multi-angle captures rather than fixed ceiling camera images; treating fixed ceiling geometry as a v1 requirement conflicts with the actual validation dataset.
  → Confirm retaining handheld EXIF optical calibration for v1 while presenting the fixed ceiling camera as the v2 barn architecture (BARN-01 / WEB-04).

[WARNING] Competing validation dataset age/weight coverage ("45 dias de criação" vs. audited ~7–29 days / 116–2,093 g)
  Found: docs/proposta_pdi.md.pdf (PRD) states MVP validation on "Broiler weights of Cobb Avian" covers "45 dias de criação" (source: docs/proposta_pdi.md.pdf), whereas the forensic disk audit in docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md (§2, §3.1, §6.3) and existing .planning/PROJECT.md / .planning/REQUIREMENTS.md (BARN-02, WEB-05) verify only 9 capture dates over 23 calendar days (2022-03-24 to 2022-04-15, ~7–29 days old, 116–2,093 g), zero samples for days 1–6 or 30–45, and a sparse 801–1,216 g gap (source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md, .planning/PROJECT.md, .planning/REQUIREMENTS.md).
  Impact: Evaluating full 45-day slaughter weights (2.1–3.2 kg) is impossible on the published dataset files and could create misaligned expectations during the 2026-11-30 pitch if not explicitly reconciled.
  → Confirm locking v1 validation coverage to the audited 116–2,093 g (~7–29 days) range, keeping days 30–45 deferred in BARN-02, and disclosing the Mendeley "45 days" discrepancy in the WEB-05 audit dashboard.

### INFO (6)

[INFO] Auto-resolved: PRD > DOC on Deep Learning prohibition (PROIBIÇÃO 1 vs. YOLO-Seg/SAM/CNN suggestions)
  Found: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md (§7, §8.3, type DOC) recommends YOLOv8/v11-Seg, SAM 2, Mask R-CNN, ConvNeXt, EfficientNetV2, EVA-02/DINOv2, and MLP for bird segmentation and end-to-end weight regression (source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md).
  Note: PRD > DOC precedence wins: existing PRD.md (PROIBIÇÃO 1), .planning/PROJECT.md, .planning/REQUIREMENTS.md, and docs/proposta_pdi.md.pdf require strictly classical interpretable PDI + tabular regression and ban neural networks/deep learning in primary segmentation and regression; DOC deep-learning recommendations are excluded from v1 synthesis (source: PRD.md, docs/proposta_pdi.md.pdf, .planning/PROJECT.md, .planning/REQUIREMENTS.md).

[INFO] Auto-resolved: PRD > DOC on SF-400 LCD OCR prohibition (PROIBIÇÃO 2 vs. offline OCR audit suggestion)
  Found: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md (§7, §8.1, type DOC) suggests running PaddleOCR, EasyOCR, or VLM OCR on cropped SF-400 LCD displays across scale-visible folders to catch additional 10 g / 100 g typos (source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md).
  Note: PRD > DOC precedence wins: PRD.md (PROIBIÇÃO 2) and .planning/REQUIREMENTS.md prohibit OCR or outside-mask pixel extraction in the v1 pipeline, while offline LCD label-audit OCR is already tracked as deferred post-pitch item HARD-01 (source: PRD.md, .planning/REQUIREMENTS.md).

[INFO] Auto-resolved: PRD > DOC on dataset root path and Parquet-only manifest filenames
  Found: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md (§8.1, §9, type DOC) references DATASET_ROOT = Path("Broiler weights of Cobb Avian") and metadata_clean.parquet / .csv (source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md).
  Note: PRD > DOC precedence wins: PRD.md (§2, REQ-01) and .planning/REQUIREMENTS.md (DATA-01) lock physical raw path data/raw/Broiler weights of Cobb Avian/ and typed Parquet outputs data/processed/manifest_clean.parquet and data/processed/manifest_quarantine.parquet (source: PRD.md, .planning/REQUIREMENTS.md).

[INFO] Auto-resolved: PRD > DOC on optional CV/ML extensions (CatBoost/XGBoost, bucket-rim mm/px, blur gate, Leave-One-Date-Out)
  Found: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md (§7, §8.2, §8.3, type DOC) suggests CatBoost/XGBoost regressors, white-bucket inner-rim mm/pixel calibration, Laplacian-variance blur filtering, and Leave-One-Date-Out CV (source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md).
  Note: PRD > DOC precedence wins: PRD.md (§5, REQ-08) and .planning/REQUIREMENTS.md lock v1 modeling to scikit-learn (Ridge/Huber, Random Forest) + LightGBM and already track bucket-rim calibration (HARD-02), blur gate (HARD-03), and Leave-One-Date-Out CV (HARD-04) in v1.x (source: PRD.md, .planning/REQUIREMENTS.md).

[INFO] Informational alignment: EXIF optical metadata vs. focal-zoom double-counting guard
  Found: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md (§5.2, type DOC) documents that Ultrawide (1.65 mm) images simultaneously report zoomMultiple = 0.60x relative to Wide (4.25 mm, 1.00x) and recommends passing (focal_length, zoom_multiple) without prescribing A_px / (f_mm * zoom)^2, whereas PRD.md (REQ-05) cites A_px / (f_mm * zoom)^2 (source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md, PRD.md).
  Note: Already reconciled in .planning/REQUIREMENTS.md (PDI-03), .planning/ROADMAP.md (Phase 2), and .planning/STATE.md, which mandate empirical verification on the 114 paired dual-lens sessions to avoid focal x zoom double-counting (source: .planning/REQUIREMENTS.md, .planning/ROADMAP.md, .planning/STATE.md).

[INFO] Informational alignment: Literature R² > 0.9 reference and future density/activity extension
  Found: docs/proposta_pdi.md.pdf (type PRD) notes literature studies achieving R² > 0.9 for projected area vs. weight and mentions flock density and immobile-bird sanitary monitoring as "uma possível extensão" (source: docs/proposta_pdi.md.pdf).
  Note: Aligns without contradiction against existing planning: R² > 0.9 is literature context rather than a hard gate (matching stratified reporting in MOD-02), and density/activity monitoring is framed as an optional extension (matching PRD.md PROIBIÇÃO 5 and .planning/REQUIREMENTS.md Out of Scope / conceptual roadmap in WEB-04) (source: PRD.md, .planning/REQUIREMENTS.md).
