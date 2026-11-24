# Phase 2: Classical PDI Engine, Optical Calibration & Watershed - Context

**Gathered:** 2026-11-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the single-source, strictly classical Digital Image Processing engine (`backend/app/pdi/`, zero deep learning per `PROIBIÇÃO 1`) shared verbatim between offline batch feature extraction and future online FastAPI serving (`PDI-01`..`PDI-05`):

1. Extend the Phase 1 Pillow EXIF-first loader (`backend/app/pdi/io_exif.py`) with aspect-preserving canonical working-resolution preprocessing (Bilateral/Median + CIE Lab CLAHE) locked by an immutable SHA-256 `PDIConfig` (`PDI-01`).
2. Isolate single-bird silhouettes across all three visual scenarios (SF-400 scale on straw, white-on-white dorsal plastic bucket including `>1,000 g` birds, and oblique straw litter) with dedicated rotation-invariant SF-400 LCD/button/platter exclusion (`PROIBIÇÃO 2`) and `< 2%` degenerate-mask fallback across all `1,738` clean images (`PDI-02`).
3. Extract strictly mask-gated (`bird_mask == 255`) morphometric, scale-invariant chromatic plumage, and empirically verified dual-lens EXIF optical features (avoiding `focal × zoom` double-counting and banning raw pixel dimensions), executing capped-concurrency (`n_jobs=4`) batch extraction into `data/processed/features_clean.parquet` plus a lightweight aggregate optical calibration audit (`models/pdi_calibration_audit.json`) (`PDI-03`).
4. Render all 4 classical "X-Ray" inspection artifacts on demand (`return_xray=True`) within the `< 500 ms` CPU budget (target `p95 < 400 ms` on native `4080×1840` JPEGs) (`PDI-04`).
5. Deliver an isolated Marker-based Watershed multi-bird separation module (`backend/app/pdi/watershed.py`) validated across all `13` quarantined multi-bird images (`417`, `757`, `823`, `1520`) and arbitrary uploads, emitting 4 visual stages, counts, per-bird areas, and per-instance mask-gated feature vectors to support illustrative individual weights in Phase 4 (`PDI-05`).

Tabular regression training/ablation (`MOD-01`..`MOD-03`) and FastAPI/React serving (`WEB-01`..`WEB-05`) belong strictly to Phases 3–4.

</domain>

<decisions>
## Implementation Decisions

### Canonical Working Resolution, Preprocessing & X-Ray Contract (`PDI-01`, `PDI-04`)
- **D-01:** Lock all tunable classical PDI parameters inside a frozen `PDIConfig` dataclass (`backend/app/pdi/config.py`) with a deterministic 16-char SHA-256 `config_hash()`, defaulting to aspect-preserving `working_long_edge = 768` (preserving exact `2.217:1` geometry via `cv2.INTER_AREA`, recording pre-draft oriented `(orig_w, orig_h)` strictly before any `PIL.Image.draft()` optimization, and supporting benchmarking `768 px` vs `960 px` during Phase 2 research to keep p95 latency `< 400 ms` on native `4080×1840` JPEGs) — **Reversibility:** costly — changing `working_long_edge` or core preprocessing parameters invalidates `pdi_config_hash` and requires re-running Phase 2 batch extraction and Phase 3 retraining.
- **D-02:** Extend `backend/app/pdi/io_exif.py` as the single repo-wide image ingestion boundary (keeping raw `cv2.imread`/`cv2.imdecode` banned via the Phase 1 AST gate), guaranteeing bit-identical oriented RGB arrays and `ExifMetadata` whether called with a `Path` or raw `bytes` (including graceful `.convert("RGB")` normalization of RGBA/grayscale uploads and `has_exif_optical=False` fallback defaults `4.25 mm / 1.0×` when EXIF is absent).
- **D-03:** Implement classical edge-preserving smoothing (`cv2.bilateralFilter` or `cv2.medianBlur`, parameterized in `PDIConfig`) followed by local contrast normalization via `cv2.createCLAHE` exclusively on the CIE Lab `L` channel (maintaining explicit RGB↔BGR and OpenCV `uint8` `L∈[0,255]`, `a/b∈[0,255]` neutral `128`, `H∈[0,179]` scaling discipline).
- **D-04:** Return 4 encoded "X-Ray" inspection artifacts on demand (`return_xray=True`) at `xray_display_long_edge = 640`: (1) EXIF-oriented original (`JPEG Q=82`), (2) CLAHE-Lab preprocessed (`JPEG Q=82`), (3) clean binary bird mask with LCD excluded (compact 1-channel `PNG`), and (4) refined contour + fitted ellipse + major/minor axes overlay (`JPEG Q=82`), keeping combined 4-image Base64 payload `< 250 KB` and encoding `< 25 ms`, while `return_xray=False` skips encoding during offline batch extraction.

### Anti-LCD Exclusion & 3-Scenario Classical Segmentation (`PDI-02`)
- **D-05:** Isolate SF-400 scale/LCD suppression in a dedicated, auditable module (`backend/app/pdi/lcd_suppression.py`) that suppresses dark neutral-chroma rectangular LCD bezels, high-gradient 7-segment digit/button clusters, and smooth specular white plastic SF-400 platter/housing **strictly before** morphological closing and component selection, using rotation-invariant photometry/shape cues (never hardcoded top/bottom frame strip rules) and reporting `lcd_excluded_px` for audit visibility — **Reversibility:** costly — core enforcement boundary for `PROIBIÇÃO 2` and X-Ray Stage 3 visual credibility.
- **D-06:** Implement deterministic classical scene-type branching (`scene_type ∈ {"bucket_dorsal", "scale_or_litter"}`, boolean `scene_is_bucket`) using outer border-ring photometry (outer `12–15%` frame perimeter high `L*` + low Lab chroma/HSV saturation):
  - **Bucket dorsal branch (`Scenario B`, incl. white-on-white `>1,000 g`):** Multi-cue foreground fusion combining Lab warmth (`a*`, `b*`) + HSV `S` (strictly gating Hue whenever `S < 25`), CLAHE feather micro-texture/Scharr edge energy, morphological Black-Hat contact-shadow rim barriers, and border-connected wall spill rejection with central Gaussian prior.
  - **Scale/litter branch (`Scenarios A & C`):** Anti-LCD/platter suppression + multi-space Lab/HSV bird-vs-straw separation + proportional elliptical morphological opening to sever thin leg/platter bridges + dominant centrally-weighted connected component + hole fill and contour refinement.
- **D-07:** Guard `cv2.fitEllipse(cnt)` with `len(cnt) >= 5` and non-collinear area checks (falling back to `cv2.minAreaRect(cnt)` for ellipse axes when degenerate) and enforce a valid foreground area envelope (`mask_area_frac ∈ [0.015, 0.72]`). If primary segmentation falls outside the envelope, run a deterministic secondary texture+shadow central-contour fallback, set `segmentation_fallback = True`, and gate Phase 2 completion on `< 2%` fallback rate (`≤ 34 / 1,738` clean images) with zero crashes across all `1,738` images.

### Mask-Gated Feature Contract & Empirical Optical Calibration (`PDI-03`)
- **D-08:** Enforce strict mask-gated feature extraction (`backend/app/pdi/features.py`) indexing exclusively `bird_mask == 255` (never whole-frame histograms, background ratios, bounding-box crops, raw `oriented_w`/`oriented_h`/`megapixels`, or axis-aligned `bbox_w`/`bbox_h`), covering three families:
  1. **Rotation-invariant morphometrics:** native back-projected `area_px`, `perimeter_px`, `major_axis_px`, `minor_axis_px`, `convex_area_px`, resolution-agnostic `area_frac`, `aspect_ratio_ellipse`, `eccentricity`, `solidity`, `circularity`, `extent_rotrect`, and scale/rotation-invariant log Hu moments (`log_hu_1..3`).
  2. **Scale-invariant plumage chroma & feather texture:** masked Lab `L*/a*/b*` quantiles (`mean`, `std`, `p10`, `p50`, `p90`), HSV `S/V` summaries (Hue circular cues strictly gated to `S >= 25`), `yellow_down_ratio`, `white_plumage_ratio`, `pink_skin_ratio`, and eroded-interior feather micro-texture energy (Scharr/Laplacian variance).
  3. **EXIF optical & calibrated descriptors:** `focal_length_mm`, `zoom_multiple`, `exif_optical_present`, plus empirically validated optically normalized area and linear descriptors (`area_optical_norm`, `sqrt_area_optical_norm`, `perim_optical_norm`, `major_axis_optical_norm`, `minor_axis_optical_norm`).
  - Verify `PROIBIÇÃO 2` with an automated outside-mask pixel-perturbation invariance test asserting bit-identical (`0.0` max diff) feature output when arbitrary digits/noise are painted onto `bird_mask == 0` — **Reversibility:** one-way — `data/processed/features_clean.parquet` column names become the frozen feature schema for Phase 3 training and Phase 4 serving.
- **D-09:** Empirically validate the dual-lens optical normalization physics across all `114` mixed-lens sessions in `manifest_clean.parquet` before finalizing the primary normalized descriptors: compare median within-session Wide (`4.25 mm`, `1.0×`) / Ultrawide (`1.65 mm`, `0.60×`) ratio for `area_frac / (zoom_multiple ** 2)` (35mm-eq magnification) vs. `area_px / ((focal_length_mm * zoom_multiple) ** 2)` vs. `area_frac / (focal_length_mm ** 2)` (and `area_px / (zoom_multiple ** 2)`), selecting the physical formula whose median Wide/UW ratio collapses closest to `1.0` without double-counting, and persisting aggregate-only verification numbers (zero raw paths) in git-tracked `models/pdi_calibration_audit.json`.
- **D-10:** Execute offline batch feature extraction via `backend/scripts/extract_features.py` reading `data/processed/manifest_clean.parquet` and writing `data/processed/features_clean.parquet` (`1,738` rows, deterministic row order matching `manifest_clean.parquet`, carrying join/audit columns `image_id`, `fold`, `session_id`, `weight_g`, `weight_band`, `lens_class`, `scene_type`, `scene_is_bucket`, `lcd_excluded_px`, `segmentation_fallback`, `pdi_config_hash`), capped at `n_jobs = 4` (`max_workers = 4`) with `cv2.setNumThreads(1)` per worker to stay safely inside the `7.1 GiB` RAM envelope and finish in `< 60 s`.

### Isolated Multi-Bird Marker-Based Watershed (`PDI-05`)
- **D-11:** Build `backend/app/pdi/watershed.py` as an isolated multi-bird instance-separation pipeline (never coupled to single-bird training, reusing `io_exif.py`, `preprocess.py`, and `features.py::extract_mask_gated_features`): chromatic red-feeder pre-suppression (`Lab a* > 152` + saturated red exclusion for folder `417`) → multi-bird union foreground mask + `binary_fill_holes` + gentle morph open/close → `scipy.ndimage.distance_transform_edt` → scale-adaptive Gaussian smoothing (`sigma = max(1.5, 0.06 * dist.max())`) → `skimage.morphology.h_maxima(dist_smooth, h = h_rel * dist_smooth.max())` + `peak_local_max` (modern coordinate API, empty-coords guarded) → `skimage.segmentation.watershed(-dist_smooth, markers, mask=fg_mask)` → spurious tiny-basin pruning.
- **D-12:** Return a structured `WatershedResult` exposing `bird_count`, tunable relative parameters (`h_rel`, `min_distance_rel`, `min_area_ratio`), per-instance records (`instance_id`, `centroid_xy`, `area_work_px`, `area_px`, `area_frac`, `area_optical_norm`, `bbox_work`, AND full mask-gated `features: dict[str, float]` computed per separated instance mask so Phase 4 `WEB-04` can compute illustrative individual weights + scene mean with group-sum disclaimer), and 4 encoded visual stages (`original`, `union_mask`, `distance_peaks`, `color_basins`), validated across all `13` quarantined multi-bird images (`417`: 2, `757`: 2, `823`: 4, `1520`: 2).

### the agent's Discretion
- Exact tuning of Lab/HSV chroma thresholds, Scharr/Black-Hat weights, morphological kernel sizes, and Watershed default relative parameters (`h_rel`, `min_distance_rel`, `min_area_ratio`) based on empirical Phase 2 research prototyping over real dataset galleries (never hardcoded per folder or preset ID).
- Exact helper module structure inside `backend/app/pdi/` (`config.py`, `io_exif.py`, `preprocess.py`, `lcd_suppression.py`, `segmentation.py`, `features.py`, `xray.py`, `pipeline.py`, `watershed.py`).
- Exact visual palette and annotation styling for X-Ray Stage 4 (contour + fitted ellipse + major/minor axes) and Watershed color basins / distance peak heatmap.
- Choice between `ProcessPoolExecutor(max_workers=4)` and `joblib.Parallel(n_jobs=4)` for `extract_features.py` (whichever is cleanest and fastest under `cv2.setNumThreads(1)`).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope, Requirements & Prohibitions
- `PRD.md` — Closed-scope product requirements (`REQ-03`–`REQ-07`, `PROIBIÇÃO 1–5`, `< 500 ms` SLA).
- `.planning/PROJECT.md` — Core value, constraints, and locked ingestion/Watershed reconciliations (`WARNING 1–4`).
- `.planning/REQUIREMENTS.md` — Formal v1 requirements `PDI-01`..`PDI-05` and downstream consumers `MOD-01`..`MOD-03`, `WEB-01`..`WEB-05`.
- `.planning/ROADMAP.md` — Phase 2 goal and 5 observable Success Criteria.
- `.planning/phases/01-forensic-data-curation-anti-leakage-split/01-CONTEXT.md` — Locked Phase 1 workspace layout (`D-01`..`D-08`) and Parquet schema contract.

### Forensic Dataset Audit & Ingestion Synthesis
- `docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md` §4, §5, §6, §7, §8 — Visual scenarios A/B/C/D, EXIF optical dual-lens physics (`4.25 mm` vs `1.65 mm`, Xiaomi tag `39321`), multi-bird counts (`417`:2, `757`:2, `823`:4, `1520`:2), slow-shutter blur, and `801–1,216 g` distribution.
- `docs/proposta_pdi.md.pdf` — Academic PDI methodology context (Watershed per-bird area/weight illustration, morphological descriptors).
- `.planning/intel/SYNTHESIS.md` — Consolidated requirements and resolved reconciliations (`WARNING 2` Watershed per-instance features).

### Research & Architecture Contracts
- `.planning/research/STACK.md` — OpenCV `4.14`, scikit-image `0.26`, SciPy `1.18`, Pillow `13`, Pandas `3.1` CoW notes, and worker concurrency ceiling (`7.1 GiB` RAM).
- `.planning/research/ARCHITECTURE.md` — `backend/app/pdi/` module boundaries, `PDIConfig` hash lock, canonical downscale back-projection, two-layer anti-LCD architecture, and online/offline data flows.
- `.planning/research/PITFALLS.md` — Pitfalls #1 (SF-400 LCD/platter leakage), #2 (white-on-white bucket collapse & `fitEllipse` crash), #3 (`focal × zoom` double-counting & lens confounding), #4 (Watershed over-segmentation & red feeder in `417`), #5 (`< 500 ms` latency & PNG base64 bloat), and #7 (`cv2.imdecode` desync & OpenCV Lab/HSV `uint8` scaling).
- `.planning/research/SUMMARY.md` — Reconciled operational standards across all 4 research tracks.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/config.py`: Repo-root anchored filesystem constants (`REPO_ROOT`, `RAW_DATASET_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, `CLEAN_MANIFEST_PATH`, `QUARANTINE_MANIFEST_PATH`, `AUDIT_SUMMARY_PATH`) — add `FEATURES_CLEAN_PATH` and `PDI_CALIBRATION_AUDIT_PATH` here.
- `backend/app/pdi/io_exif.py`: Already implements `ExifMetadata`, `OrientedImage`, `read_exif_metadata(source: Path | bytes)`, and `load_oriented_rgb(source: Path | bytes)` with 16 KB Xiaomi tag cap, `IFDRational(0,0)` guard, and `ImageOps.exif_transpose()`. Extend cleanly with optional `target_long_edge` / `Image.draft()` fast-decode support while preserving 100% backward compatibility with Phase 1 tests.
- `backend/tests/test_io_exif.py`: Already enforces repo-wide AST ban on `cv2.imread`/`cv2.imdecode` and `pyproject.toml` ban on DL/OCR packages.
- `data/processed/manifest_clean.parquet` (`1,738` × `24` cols) & `data/processed/manifest_quarantine.parquet` (`15` × `24` cols, including the `13` multi-bird rows in `417`/`757`/`823`/`1520`) are materialized and verified on disk.

### Established Patterns
- Python `3.12.14` managed by `uv` (`uv run pytest -v`, `uv run ruff check backend/`), root `pyproject.toml` with `known-first-party = ["backend"]` and `pythonpath = ["."]`.
- Pure frozen dataclasses (`@dataclass(frozen=True, slots=True)`) and typed Pandas 3 Copy-on-Write (`.loc[mask, col] = val`) Parquet outputs with deterministic row ordering.
- Aggregate-only JSON audit reports stored in `models/` (git-tracked, zero raw paths per `PROIBIÇÃO 4`) with automated path-leakage guard tests.

### Integration Points
- `backend/app/pdi/pipeline.py::run_single_bird_pdi()` will be imported verbatim by Phase 2 `backend/scripts/extract_features.py` (`return_xray=False`) and Phase 4 `backend/app/api/services/inference_service.py` (`return_xray=True`).
- `backend/app/pdi/watershed.py::run_watershed_pdi()` will be imported verbatim by Phase 4 `backend/app/api/services/watershed_service.py`.
- `data/processed/features_clean.parquet` and `models/pdi_calibration_audit.json` feed directly into Phase 3 (`backend/scripts/train_and_evaluate.py`) and Phase 4 (`GET /api/metrics`).

</code_context>

<specifics>
## Specific Ideas

- Never hardcode folder names or preset IDs inside segmentation or Watershed logic — all rules must be general, classical, photometric/morphological, and rotation-invariant so arbitrary user uploads work identically to dataset samples.
- Run a focused empirical prototyping/calibration check during Phase 2 research on real samples spanning Scenario A (SF-400 scale: `116`, `233`, `679`, `1009`, `1340_`, `2093`), Scenario B (white-on-white heavy dorsal bucket: `1216–2093 g` plus young yellow bucket chicks), Scenario C (straw litter), paired dual-lens sessions (`114` mixed-lens sessions), and all `13` multi-bird images (`417`, `757`, `823`, `1520`) — never dumping raw filenames to stdout/JSON (`PROIBIÇÃO 4`).
- Keep `backend/app/pdi/features.py` feature names clean, snake_case, self-describing, and 1:1 aligned with Phase 3 modeling and Phase 4 frontend feature tables.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed strictly within Phase 2 scope. (Bucket-rim Hough `mm/pixel` calibration `HARD-02` and Laplacian motion-blur quality gate `HARD-03` remain deferred to v1.x per `REQUIREMENTS.md`.)

</deferred>

---

*Phase: 02-classical-pdi-engine-optical-calibration-watershed*
*Context gathered: 2026-11-22*
