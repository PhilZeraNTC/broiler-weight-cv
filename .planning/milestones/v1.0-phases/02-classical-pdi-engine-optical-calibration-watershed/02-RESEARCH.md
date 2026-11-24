# Phase 2: Classical PDI Engine, Optical Calibration & Watershed - Research

**Researched:** 2026-11-22
**Domain:** Classical Digital Image Processing (OpenCV / scikit-image / SciPy / Pillow, Zero Deep Learning), Dual-Lens EXIF Optical Calibration, and Marker-Based Watershed Multi-Bird Separation
**Confidence:** HIGH (empirically verified via read-only prototypes across all 1,738 clean images, 127 mixed-lens sessions, 10 native `4080×1840` latency benchmarks, and all 13 quarantined multi-bird images)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Canonical Working Resolution, Preprocessing & X-Ray Contract (`PDI-01`, `PDI-04`)
- **D-01:** Lock all tunable classical PDI parameters inside a frozen `PDIConfig` dataclass (`backend/app/pdi/config.py`) with a deterministic 16-char SHA-256 `config_hash()`, defaulting to aspect-preserving `working_long_edge = 768` (preserving exact `2.217:1` geometry via `cv2.INTER_AREA`, recording pre-draft oriented `(orig_w, orig_h)` strictly before any `PIL.Image.draft()` optimization, and supporting benchmarking `768 px` vs `960 px` during Phase 2 research to keep p95 latency `< 400 ms` on native `4080×1840` JPEGs) — **Reversibility:** costly — changing `working_long_edge` or core preprocessing parameters invalidates `pdi_config_hash` and requires re-running Phase 2 batch extraction and Phase 3 retraining.
- **D-02:** Extend `backend/app/pdi/io_exif.py` as the single repo-wide image ingestion boundary (keeping raw `cv2.imread`/`cv2.imdecode` banned via the Phase 1 AST gate), guaranteeing bit-identical oriented RGB arrays and `ExifMetadata` whether called with a `Path` or raw `bytes` (including graceful `.convert("RGB")` normalization of RGBA/grayscale uploads and `has_exif_optical=False` fallback defaults `4.25 mm / 1.0×` when EXIF is absent).
- **D-03:** Implement classical edge-preserving smoothing (`cv2.bilateralFilter` or `cv2.medianBlur`, parameterized in `PDIConfig`) followed by local contrast normalization via `cv2.createCLAHE` exclusively on the CIE Lab `L` channel (maintaining explicit RGB↔BGR and OpenCV `uint8` `L∈[0,255]`, `a/b∈[0,255]` neutral `128`, `H∈[0,179]` scaling discipline).
- **D-04:** Return 4 encoded "X-Ray" inspection artifacts on demand (`return_xray=True`) at `xray_display_long_edge = 640`: (1) EXIF-oriented original (`JPEG Q=82`), (2) CLAHE-Lab preprocessed (`JPEG Q=82`), (3) clean binary bird mask with LCD excluded (compact 1-channel `PNG`), and (4) refined contour + fitted ellipse + major/minor axes overlay (`JPEG Q=82`), keeping combined 4-image Base64 payload `< 250 KB` and encoding `< 25 ms`, while `return_xray=False` skips encoding during offline batch extraction.

#### Anti-LCD Exclusion & 3-Scenario Classical Segmentation (`PDI-02`)
- **D-05:** Isolate SF-400 scale/LCD suppression in a dedicated, auditable module (`backend/app/pdi/lcd_suppression.py`) that suppresses dark neutral-chroma rectangular LCD bezels, high-gradient 7-segment digit/button clusters, and smooth specular white plastic SF-400 platter/housing **strictly before** morphological closing and component selection, using rotation-invariant photometry/shape cues (never hardcoded top/bottom frame strip rules) and reporting `lcd_excluded_px` for audit visibility — **Reversibility:** costly — core enforcement boundary for `PROIBIÇÃO 2` and X-Ray Stage 3 visual credibility.
- **D-06:** Implement deterministic classical scene-type branching (`scene_type ∈ {"bucket_dorsal", "scale_or_litter"}`, boolean `scene_is_bucket`) using outer border-ring photometry (outer `12–15%` frame perimeter high `L*` + low Lab chroma/HSV saturation):
  - **Bucket dorsal branch (`Scenario B`, incl. white-on-white `>1,000 g`):** Multi-cue foreground fusion combining Lab warmth (`a*`, `b*`) + HSV `S` (strictly gating Hue whenever `S < 25`), CLAHE feather micro-texture/Scharr edge energy, morphological Black-Hat contact-shadow rim barriers, and border-connected wall spill rejection with central Gaussian prior.
  - **Scale/litter branch (`Scenarios A & C`):** Anti-LCD/platter suppression + multi-space Lab/HSV bird-vs-straw separation + proportional elliptical morphological opening to sever thin leg/platter bridges + dominant centrally-weighted connected component + hole fill and contour refinement.
- **D-07:** Guard `cv2.fitEllipse(cnt)` with `len(cnt) >= 5` and non-collinear area checks (falling back to `cv2.minAreaRect(cnt)` for ellipse axes when degenerate) and enforce a valid foreground area envelope (`mask_area_frac ∈ [0.015, 0.72]`). If primary segmentation falls outside the envelope, run a deterministic secondary texture+shadow central-contour fallback, set `segmentation_fallback = True`, and gate Phase 2 completion on `< 2%` fallback rate (`≤ 34 / 1,738` clean images) with zero crashes across all `1,738` images.

#### Mask-Gated Feature Contract & Empirical Optical Calibration (`PDI-03`)
- **D-08:** Enforce strict mask-gated feature extraction (`backend/app/pdi/features.py`) indexing exclusively `bird_mask == 255` (never whole-frame histograms, background ratios, bounding-box crops, raw `oriented_w`/`oriented_h`/`megapixels`, or axis-aligned `bbox_w`/`bbox_h`), covering three families:
  1. **Rotation-invariant morphometrics:** native back-projected `area_px`, `perimeter_px`, `major_axis_px`, `minor_axis_px`, `convex_area_px`, resolution-agnostic `area_frac`, `aspect_ratio_ellipse`, `eccentricity`, `solidity`, `circularity`, `extent_rotrect`, and scale/rotation-invariant log Hu moments (`log_hu_1..3`).
  2. **Scale-invariant plumage chroma & feather texture:** masked Lab `L*/a*/b*` quantiles (`mean`, `std`, `p10`, `p50`, `p90`), HSV `S/V` summaries (Hue circular cues strictly gated to `S >= 25`), `yellow_down_ratio`, `white_plumage_ratio`, `pink_skin_ratio`, and eroded-interior feather micro-texture energy (Scharr/Laplacian variance).
  3. **EXIF optical & calibrated descriptors:** `focal_length_mm`, `zoom_multiple`, `exif_optical_present`, plus empirically validated optically normalized area and linear descriptors (`area_optical_norm`, `sqrt_area_optical_norm`, `perim_optical_norm`, `major_axis_optical_norm`, `minor_axis_optical_norm`).
  - Verify `PROIBIÇÃO 2` with an automated outside-mask pixel-perturbation invariance test asserting bit-identical (`0.0` max diff) feature output when arbitrary digits/noise are painted onto `bird_mask == 0` — **Reversibility:** one-way — `data/processed/features_clean.parquet` column names become the frozen feature schema for Phase 3 training and Phase 4 serving.
- **D-09:** Empirically validate the dual-lens optical normalization physics across all `114` mixed-lens sessions in `manifest_clean.parquet` before finalizing the primary normalized descriptors: compare median within-session Wide (`4.25 mm`, `1.0×`) / Ultrawide (`1.65 mm`, `0.60×`) ratio for `area_frac / (zoom_multiple ** 2)` (35mm-eq magnification) vs. `area_px / ((focal_length_mm * zoom_multiple) ** 2)` vs. `area_frac / (focal_length_mm ** 2)` (and `area_px / (zoom_multiple ** 2)`), selecting the physical formula whose median Wide/UW ratio collapses closest to `1.0` without double-counting, and persisting aggregate-only verification numbers (zero raw paths) in git-tracked `models/pdi_calibration_audit.json`.
- **D-10:** Execute offline batch feature extraction via `backend/scripts/extract_features.py` reading `data/processed/manifest_clean.parquet` and writing `data/processed/features_clean.parquet` (`1,738` rows, deterministic row order matching `manifest_clean.parquet`, carrying join/audit columns `image_id`, `fold`, `session_id`, `weight_g`, `weight_band`, `lens_class`, `scene_type`, `scene_is_bucket`, `lcd_excluded_px`, `segmentation_fallback`, `pdi_config_hash`), capped at `n_jobs = 4` (`max_workers = 4`) with `cv2.setNumThreads(1)` per worker to stay safely inside the `7.1 GiB` RAM envelope and finish in `< 60 s`.

#### Isolated Multi-Bird Marker-Based Watershed (`PDI-05`)
- **D-11:** Build `backend/app/pdi/watershed.py` as an isolated multi-bird instance-separation pipeline (never coupled to single-bird training, reusing `io_exif.py`, `preprocess.py`, and `features.py::extract_mask_gated_features`): chromatic red-feeder pre-suppression (`Lab a* > 152` + saturated red exclusion for folder `417`) → multi-bird union foreground mask + `binary_fill_holes` + gentle morph open/close → `scipy.ndimage.distance_transform_edt` → scale-adaptive Gaussian smoothing (`sigma = max(1.5, 0.06 * dist.max())`) → `skimage.morphology.h_maxima(dist_smooth, h = h_rel * dist_smooth.max())` + `peak_local_max` (modern coordinate API, empty-coords guarded) → `skimage.segmentation.watershed(-dist_smooth, markers, mask=fg_mask)` → spurious tiny-basin pruning.
- **D-12:** Return a structured `WatershedResult` exposing `bird_count`, tunable relative parameters (`h_rel`, `min_distance_rel`, `min_area_ratio`), per-instance records (`instance_id`, `centroid_xy`, `area_work_px`, `area_px`, `area_frac`, `area_optical_norm`, `bbox_work`, AND full mask-gated `features: dict[str, float]` computed per separated instance mask so Phase 4 `WEB-04` can compute illustrative individual weights + scene mean with group-sum disclaimer), and 4 encoded visual stages (`original`, `union_mask`, `distance_peaks`, `color_basins`), validated across all `13` quarantined multi-bird images (`417`: 2, `757`: 2, `823`: 4, `1520`: 2).

### the agent's Discretion
- Exact tuning of Lab/HSV chroma thresholds, Scharr/Black-Hat weights, morphological kernel sizes, and Watershed default relative parameters (`h_rel`, `min_distance_rel`, `min_area_ratio`) based on empirical Phase 2 research prototyping over real dataset galleries (never hardcoded per folder or preset ID).
- Exact helper module structure inside `backend/app/pdi/` (`config.py`, `io_exif.py`, `preprocess.py`, `lcd_suppression.py`, `segmentation.py`, `features.py`, `xray.py`, `pipeline.py`, `watershed.py`).
- Exact visual palette and annotation styling for X-Ray Stage 4 (contour + fitted ellipse + major/minor axes) and Watershed color basins / distance peak heatmap.
- Choice between `ProcessPoolExecutor(max_workers=4)` and `joblib.Parallel(n_jobs=4)` for `extract_features.py` (whichever is cleanest and fastest under `cv2.setNumThreads(1)`).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed strictly within Phase 2 scope. (Bucket-rim Hough `mm/pixel` calibration `HARD-02` and Laplacian motion-blur quality gate `HARD-03` remain deferred to v1.x per `REQUIREMENTS.md`.)
</user_constraints>

## Project Constraints (from GEMINI.md)

Extracted from `./GEMINI.md` (`[VERIFIED: GEMINI.md:11-18, 108-140]`):
- **Classical PDI Only (`PROIBIÇÃO 1`):** Zero deep learning / neural networks (`torch`, `tensorflow`, `keras`, `ultralytics`, `timm`, `onnxruntime`) and zero OCR libraries (`paddleocr`, `easyocr`) in segmentation, feature extraction, or Watershed — enforced via `backend/tests/test_io_exif.py` AST & lockfile gates (`[VERIFIED: backend/tests/test_io_exif.py:58-71, 264-309]`).
- **Anti-Leakage (`PROIBIÇÃO 2` & `PROIBIÇÃO 3`):** Hard SF-400 LCD/button/platter suppression before morphological closing and component selection; strictly mask-gated feature indexing (`bird_mask == 255`), verified by automated outside-mask pixel-perturbation invariance tests.
- **Single EXIF-First Image Ingestion Boundary:** Raw `cv2.imread` and `cv2.imdecode` are banned repo-wide under `backend/app/` (`[VERIFIED: backend/tests/test_io_exif.py:264-286]`). All loading goes through `backend/app/pdi/io_exif.py` using Pillow `ImageOps.exif_transpose()`.
- **Performance SLA:** End-to-end single-image inference `< 500 ms` on local CPU (target `p95 < 400 ms` on native `4080×1840` Wide JPEGs); offline 1,738-image batch extraction `< 60 s` capped at `4` workers with `cv2.setNumThreads(1)` on the `7.1 GiB` RAM host (`1.5 GiB` available RAM at research time).
- **Data Hygiene (`PROIBIÇÃO 4`):** `data/raw/` is strictly read-only; `data/processed/features_clean.parquet` stays git-ignored; git-tracked `models/pdi_calibration_audit.json` must contain zero raw image filenames, `rel_path` entries, or host filesystem paths (forbidden substrings `.jpg`, `rel_path`, `/home/`, `data/raw` — matching `[VERIFIED: backend/app/ingestion/audit_summary.py:50]`).
- **Pandas 3.1 Copy-on-Write & Arrow `str` Discipline:** Always mutate via single-step `.loc[mask, col] = val`, never chained assignment, and persist strictly as typed PyArrow Parquet (never CSV).
- **Timeline:** MVP code freeze `2026-11-28` for `2026-11-30` pitch.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **PDI-01** | Single shared Pillow EXIF-first loader (`Path \| bytes`, zero `cv2.imread`/`cv2.imdecode`), aspect-preserving canonical downscale (`768–960 px` with tracked `scale_factor`), Bilateral/Median smoothing, and CIE Lab `L`-channel CLAHE. | Spike 1 empirically proves `working_long_edge = 768` + **aspect-preserving** `im.draft("RGB", draft_box)` (`draft_mult = 1.0`, fixing the silent square-box no-op bug) + `bilateralFilter(d=5, sigmaColor=50, sigmaSpace=50)` + `CLAHE(clipLimit=2.0, tileGridSize=(8,8))` delivers `27.8 ms` load and `4.6 ms` preprocess with bit-exact `Path`-vs-`bytes` parity (`0.98/255` RGB MAE vs full-decode). |
| **PDI-02** | Classical single-bird segmentation across Scenarios A (SF-400 scale on straw), B (white plastic bucket incl. white-on-white `>1,000 g`), and C (oblique straw) with dedicated rotation-invariant SF-400 LCD/platter suppression and `< 2%` fallback across all `1,738` clean images. | Spike 3 empirically validates bimodal border-ring smoothness (`smooth_white >= 0.50 & ring_gmed <= 45.0 & sides_sw >= 3`, yielding `377` bucket / `1,361` non-bucket), border-connected smooth-wall peeling + Black-Hat contact-shadow barrier for Scenario B (`0/377 = 0.00%` fallback across all weight bands incl. all `59` `>1,000 g` white-on-white bucket broilers), and rotation-invariant dark-neutral rectangular LCD + connected smooth specular housing suppression (`672` non-bucket exclusions, `0` bucket false positives), achieving **`13 / 1,738 = 0.75%`** fallback overall. |
| **PDI-03** | Strictly mask-gated (`bird_mask == 255`) geometric, chromatic/texture, and empirically verified dual-lens EXIF optical features (avoiding `focal × zoom` double-counting and banning raw pixel dimensions), automated outside-mask perturbation invariance test, `models/pdi_calibration_audit.json`, and `n_jobs=4` batch write to `data/processed/features_clean.parquet` in `< 60 s`. | Spike 2 proves across all `127` mixed-lens clean sessions (`175` same-scene pairs, `76` bucket-only pairs, `99` scale/litter pairs) that **`area_px / (zoom_multiple ** 2)`** collapses median Wide/UW ratio to **`1.007`** (bucket), **`0.900`** (same-scene), **`0.862`** (scale/litter), and **`0.783`** (all sessions) with lowest `med\|log(W/UW)\|` on every scope, while `area_px / ((f * z) ** 2)` over-corrects by `6.6×–8.5×`. Full 1,738-image 4-worker sweep completes in **`42.9 s`**. |
| **PDI-04** | Emit 4 encoded "X-Ray" inspection artifacts (`original`, `preprocessed_clahe`, `clean_mask`, `contour_ellipse_overlay`) at `640 px` alongside `PDIConfig` SHA-256 hash within `< 500 ms` CPU budget (`p95 < 400 ms`, payload `< 250 KB`). | Spike 1 measures full end-to-end single-bird latency (load + preprocess + seg + features + 4-stage X-ray at `640 px`, 3× JPEG Q=82 + 1× 1-ch PNG) on 10 real native `4080×1840` Wide JPEGs: **`p50 = 74.0 ms`, `p95 = 90.2 ms`**, X-ray render/encode **`5.2 ms`**, combined Base64 payload **`p50 = 158.3 KB`, `max = 231.7 KB`**. |
| **PDI-05** | Isolated Marker-based Watershed (`red-feeder` pre-suppression, EDT, scale-adaptive Gaussian, `h_maxima`/`peak_local_max`, `skimage.segmentation.watershed`, tiny-basin pruning) validated on all `13` multi-bird images (`417`, `757`, `823`, `1520`) + custom uploads, emitting 4 visuals, counts, per-bird areas, and full mask-gated per-instance features. | Spike 4 validates a single general zero-hardcoding pipeline (`h_rel=0.20, min_distance_rel=0.85, min_area_ratio=0.10`) hitting **`9 / 13` exact counts (`417`: 4/4, `1520`: 3/4, `823`: 2/4 exact + 2× off-by-1), `12 / 13` within ±1, MAE = `0.38` birds**, and proves a **22× bit-identical speedup (`313.8 ms → 14.0 ms`)** by running `h_maxima` strictly on the padded foreground bounding box. |
</phase_requirements>

## Summary

Phase 2 builds the shared, single-source classical Digital Image Processing (PDI) engine under `backend/app/pdi/` and runs deterministic offline feature extraction over all `1,738` clean images in `data/processed/manifest_clean.parquet`. Zero new external packages are installed — all required libraries (`opencv-python-headless 4.14.0`, `scikit-image 0.26.0`, `scipy 1.18.1`, `pillow 13.0.0`, `numpy 2.5.5`, `pandas 3.1.1`, `pyarrow 26.0.1`, `joblib 1.6.0`) are already hash-locked in `uv.lock` under Python `3.12.14`.

Four empirical read-only prototypes over the real dataset eliminated every major technical risk before planning and surfaced four non-obvious findings that directly shape Phase 2 implementation:
1. **Aspect-Preserving `PIL.Image.draft()` Trap:** Passing a square box `(1536, 1536)` to `im.draft("RGB", ...)` silently no-ops on `4080×1840` (`2.217:1`) Redmi 10 JPEGs because `1840 / 2 = 920 < 1536`, forcing full 7.51 MP decode (`147 ms`). Computing an **aspect-preserving draft box** `(int(raw_w * sc), int(raw_h * sc))` for `working_long_edge = 768` (`draft_mult = 1.0`) triggers `1/4` DCT decode on both Wide (`1020×460`) and Ultrawide (`816×368`), slashing full single-bird end-to-end latency (including 4-stage X-Ray Base64 encoding) to **`p50 = 74.0 ms` / `p95 = 90.2 ms`** (`< 400 ms` target by `4.4×`) and payload to **`158.3 KB` median / `231.7 KB` max** (`< 250 KB` cap).
2. **Smooth-Wall Peeling Solves White-on-White Bucket Segmentation:** Border-ring smoothness (`ring_gmed` histogram peak `[10, 20): 331` vs straw `>= 150: 1,021`) cleanly separates Scenario B (`377` images) from Scenarios A/C (`1,361` images). Crucially, naive bucket luminance/texture thresholding floods the shaded/straw-flecked bucket floor (`~50%` of frame regardless of chick weight). Peeling **border-connected smooth white plastic walls** guarded by a **Black-Hat contact-shadow barrier** isolates the interior first, then separates the bright/textured bird from the darker floor moat — cutting Scenario B fallback to **`0 / 377 (0.00%)`** (including all `59` white-on-white `>1,000 g` broilers) and overall clean-dataset fallback to **`13 / 1,738 (0.75%)`** in **`42.9 s`** on `4` workers.
3. **Exact Dual-Lens Redmi 10 Calibration Physics:** Across all `127` mixed-lens clean sessions (`111` clean folders; `114` raw folders including `417`/`823`/`1520`), Xiaomi `zoomMultiple` (`1.0×` vs `0.60×`) scales **native pixel area (`area_px`)** by `2.770×` (`≈ 1 / 0.60² = 2.778×`), while the `(4080/3264)² = 1.5625×` sensor pixel-count change leaves `area_frac` shifted by `2.778 / 1.5625 = 1.778×` (empirically measured `1.773×` on `76` bucket-only paired sessions!). Consequently, **`area_optical_norm = area_px / (zoom_multiple ** 2)`** wins unanimously across all 4 scopes (`med(W/UW) = 1.007` bucket, `0.900` same-scene, `0.862` scale/litter, `0.783` all sessions), whereas `area_px / ((focal_length_mm * zoom_multiple) ** 2)` catastrophically over-corrects by `6.6×–8.5×` (`0.118–0.152`). Persisting both `area_optical_norm` and `area_frac_optical_norm = area_frac / (zoom_multiple ** 2)` gives Phase 3 full ablation transparency.
4. **Bbox-Cropped `h_maxima` Speeds Watershed by 22×:** Running `skimage.morphology.h_maxima` on the padded foreground bounding box instead of the full `346×768` frame cuts its latency from **`313.8 ms` → `14.0 ms`** with `bit-identical = True`, while general relative defaults (`h_rel = 0.20, min_distance_rel = 0.85, min_area_ratio = 0.10`) achieve **`9 / 13` exact counts (`12 / 13` within ±1, `417`: 4/4 exact, `1520`: 3/4 exact)** with zero folder-name hardcoding.

**Primary recommendation:** Implement the modular 9-file classical PDI package under `backend/app/pdi/` locked to `PDIConfig(working_long_edge=768, draft_scale_multiplier=1.0, bilateral_d=5, bilateral_sigma_color=50.0, bilateral_sigma_space=50.0, clahe_clip_limit=2.0, clahe_tile_grid=(8, 8))`, using `area_optical_norm = area_px / (zoom_multiple ** 2)` as the empirically verified primary optical area descriptor, border-connected smooth-wall peeling + Black-Hat barrier for Scenario B, rotation-invariant LCD + connected specular housing exclusion for Scenarios A/C, and bbox-cropped `h_maxima` Marker-based Watershed in isolated `watershed.py`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| EXIF parsing, orientation transpose & aspect-preserving draft decode (`io_exif.py`) | Backend PDI Engine (`backend/app/pdi/`) | — | Single approved image ingestion boundary (`PROIBIÇÃO 2`); prevents `cv2.imdecode` orientation desync across offline CLI and Phase 4 FastAPI. |
| Locked parameter registry & SHA-256 hash (`config.py`) | Backend PDI Engine (`backend/app/pdi/`) | Phase 3 Modeling / Phase 4 API | Eliminates training-serving skew by binding `features_clean.parquet`, `weight_regressor.joblib`, and FastAPI serving to the exact same `pdi_config_hash`. |
| SF-400 LCD/platter suppression & 3-scenario segmentation (`lcd_suppression.py`, `segmentation.py`) | Backend PDI Engine (`backend/app/pdi/`) | — | Auditable enforcement boundary for `PROIBIÇÃO 2` (LCD excluded before closing/selection) and `< 2%` fallback guarantee across all visual scenarios. |
| Mask-gated morphometric, chromatic & EXIF optical features (`features.py`) | Backend PDI Engine (`backend/app/pdi/`) | — | Pure function taking `bird_mask` + `PreprocessedFrame` + `OpticalContext`, guaranteeing `0.0` diff under outside-mask pixel perturbation. |
| 4-stage X-Ray visual encoding (`xray.py`) | Backend PDI Engine (`backend/app/pdi/`) | Phase 4 Presentation Plane | Executed strictly on demand (`return_xray=True`); keeps offline batch zero-overhead and online Base64 JSON `< 250 KB` / `< 10 ms` encode. |
| Isolated multi-bird Marker-based Watershed (`watershed.py`) | Backend PDI Engine (`backend/app/pdi/`) | Phase 4 `/api/watershed-demo` | Reuses `io_exif`, `preprocess`, and `features.extract_mask_gated_features` while remaining strictly decoupled from single-bird training (`PROIBIÇÃO 2`/`D-11`). |
| Capped-concurrency batch extraction & calibration audit (`extract_features.py`) | Offline Batch CLI (`backend/scripts/`) | Local Artifact Store (`data/processed/`, `models/`) | Materializes git-ignored `data/processed/features_clean.parquet` (`1,738` rows) and git-tracked aggregate `models/pdi_calibration_audit.json` in `< 60 s` using `ProcessPoolExecutor(max_workers=4)`. |

## Empirical Prototype Findings (Phase 2 Spikes)

All four mandatory spikes were executed read-only via `uv run python -c` against the materialized Phase 1 manifests (`data/processed/manifest_clean.parquet`, `data/processed/manifest_quarantine.parquet`) without logging or emitting a single raw JPEG filename (`PROIBIÇÃO 4`).

### Spike 1: Working Resolution (`768 px` vs `960 px`), `PIL.Image.draft()` Trap & End-to-End Latency (`D-01`, `D-04`)

**Setup:** 10 real native `4080×1840` (`7.51 MP`) Wide JPEGs spanning all 4 dataset orientations (`1, 3, 6, 8`) loaded from raw `bytes` (mirroring FastAPI `UploadFile`), `cv2.setNumThreads(2)`, full pipeline (`load + EXIF before draft + exif_transpose + INTER_AREA resize + Bilateral + Lab CLAHE + HSV + LCD/seg + regionprops/Hu/Lab/HSV features + 4-stage 640 px X-Ray encode as 3× JPEG Q=82 + 1× 1-ch PNG`) `[VERIFIED: uv run python benchmark, 2026-11-22]`.

**Critical discovery — Square vs. Aspect-Preserving `im.draft()`:**
- Square `im.draft("RGB", (1536, 1536))` silently **refused** DCT downscaling on `4080×1840` (`drafted size = (4080, 1840)`) because Pillow requires **both** reduced dimensions to stay `>=` the requested box (`1840 / 2 = 920 < 1536`), yielding `146.9 ms` load latency and `p95 = 173.9–229.7 ms`.
- Aspect-preserving `im.draft("RGB", (max(1, int(raw_w * dsc)), max(1, int(raw_h * dsc))))` triggered libjpeg DCT shrink-on-load cleanly (`1020×460` at `draft_mult=1.0`; `2040×920` at `draft_mult=2.0`), cutting load time by **5×** (`147 ms → 27.8 ms`) and peak decoded image memory by **16×** (`22.5 MB → 1.4 MB` RGB buffer per worker).
- `Path`-vs-`bytes` parity with aspect-preserving `im.draft()` was verified **bit-exact** (`np.array_equal == True` and `ExifMetadata` equality across all tested orientations).

| Config (`edge`, `draft_mult`, `bilat_d`) | Drafted Size (`4080×1840`) | Load (ms) | Prep (ms) | Seg (ms) | Feat (ms) | X-Ray (ms) | Total `p50` (ms) | Total `p95` (ms) | Base64 `p50` (KB) | Base64 `max` (KB) | RGB MAE vs Full Decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`768`, `1.0`, `d=5` (RECOMMENDED)** | `1020×460` | **27.8** | **4.6** | **22.9** | **14.5** | **5.2** | **74.0** | **90.2** | **158.3** | **231.7** | **0.98 / 255** |
| `768`, `1.0`, `d=7` | `1020×460` | 26.0 | 11.1 | 21.5 | 14.8 | 5.7 | 76.7 | 94.1 | 158.1 | 230.0 | 0.98 / 255 |
| `768`, `2.0`, `d=5` | `2040×920` | 55.4 | 5.3 | 21.5 | 13.4 | 5.2 | 100.6 | 118.5 | 161.6 | 240.1 | 0.28 / 255 |
| `960`, `1.0`, `d=5` | `1020×460` | 25.6 | 10.1 | 35.5 | 21.2 | 6.4 | 98.3 | 129.3 | 169.9 | 243.9 | — |
| `960`, `1.0`, `d=7` | `1020×460` | 27.2 | 16.8 | 36.1 | 22.2 | 5.8 | 112.0 | 141.1 | 167.8 | 242.2 | — |
| `960`, `2.0`, `d=5` | `2040×920` | 52.1 | 9.7 | 37.6 | 22.3 | 5.4 | 126.7 | 152.4 | 171.8 | 253.3 | — |
| `768`, square draft (NO-OP BUG) | `4080×1840` | 142.9 | 5.9 | — | — | 5.1 | 159.6 | 173.9 | 174.7 | 276.1 | 0.00 / 255 |

**Prescriptive Decision:** Lock `working_long_edge = 768`, `draft_scale_multiplier = 1.0`, `bilateral_d = 5`, `bilateral_sigma_color = 50.0`, `bilateral_sigma_space = 50.0`, `clahe_clip_limit = 2.0`, `clahe_tile_grid = (8, 8)`, `xray_display_long_edge = 640`, `xray_jpeg_quality = 82`. At `768 px` with `draft_scale_multiplier = 1.0`, both Wide (`4080×1840 → 1020×460`) and Ultrawide (`3264×1472 → 816×368`) trigger clean `1/4` DCT shrink-on-load, total `p95 = 90.2 ms` leaves `> 300 ms` of headroom below the `< 400 ms` p95 target (`> 400 ms` below the `500 ms` SLA), and max Base64 payload (`231.7 KB`) stays below `< 250 KB` on every sample.

---

### Spike 2: Paired Dual-Lens Optical Calibration Audit (`D-09`)

**Exact Mixed-Lens Counts in `manifest_clean.parquet` (`[VERIFIED: uv run python manifest inspection, 2026-11-22]`):**
- Clean dataset (`1,738` images): **`127` mixed-lens `session_id`s** out of `188` sessions, spanning **`111` mixed-lens `folder_chain`s** out of `162` clean leaf folders (the raw dataset's `114 / 168` mixed-lens folders equals `111` clean + `3` quarantined multi-bird folders `417`, `823`, `1520`).
- Lens/zoom distribution in `manifest_clean.parquet`:
  - `wide` (`4.25 mm`, `4080×1840`, `7.5072 MP`): `1,242` at `1.00×`, `8` at `1.02×`, `41` at `1.06×`, `2` at `1.07×`, `4` at `2.00×` (`1,297` total; `2.0×` zoom occurs in sessions `197__s1`, `219__s0`, `371__s0`, `401__s0`).
  - `ultrawide` (`1.65 mm`, `3264×1472`, `4.8046 MP`): `429` at `0.60×`, `4` at `0.77×`, `3` at `0.85×`, `5` at `0.96×` (`441` total).

**Empirical Paired Wide/Ultrawide Ratio Across All 4 Scopes (`[VERIFIED: full 1,738-image v4 sweep, 2026-11-22]`):**

| Candidate Formula | Bucket-Only Pairs (`N=76`) `med(W/UW)` / `med\|log\|` | Same-Scene Pairs (`N=175`) `med(W/UW)` / `med\|log\|` | Scale/Litter Pairs (`N=99`) `med(W/UW)` / `med\|log\|` | All Mixed-Lens Sessions (`N=127`) `med(W/UW)` / `med\|log\|` | Log-Log `r` vs `weight_g` (`1,738` clean) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `area_frac` (uncalibrated) | `1.773` / `0.692` | `1.584` / `0.638` | `1.533` / `0.629` | `1.363` / `0.564` | `+0.2511` | Baseline (uncalibrated) |
| `area_px` (uncalibrated) | `2.770` / `1.029` | `2.474` / `0.917` | `2.396` / `0.874` | `2.130` / `0.756` | `+0.2054` | Baseline (uncalibrated) |
| **`area_px / (zoom_multiple ** 2)`** | **`1.007` / `0.424`** | **`0.900` / `0.500`** | **`0.862` / `0.572`** | **`0.783` / `0.440`** | `+0.2868` | **WINNER (primary `area_optical_norm`)** |
| `area_frac / (zoom_multiple ** 2)` | `0.645` / `0.459` | `0.576` / `0.594` | `0.552` / `0.680` | `0.501` / `0.716` | **`+0.2972`** | **Secondary (`area_frac_optical_norm`, resize-invariant)** |
| `area_frac / (focal_length_mm ** 2)` | `0.267` / `1.320` | `0.239` / `1.433` | `0.231` / `1.465` | `0.205` / `1.583` | `+0.2855` | REJECTED (`3.7×–4.9×` over-correction; unequal sensor sizes) |
| `area_px / ((focal_length_mm * zoom_multiple) ** 2)` | `0.152` / `1.885` | `0.136` / `1.997` | `0.130` / `2.040` | `0.118` / `2.137` | `+0.2666` | **REJECTED (`6.6×–8.5×` double-counting over-correction!)** |

**Exact Optical Physics Explanation (Verified to 0.3% Precision!):**
- On the Xiaomi Redmi 10 (`21061119AG`), raw native pixel area `area_px` shifts between Wide (`4080×1840`, `1.0×`) and Ultrawide (`3264×1472`, `0.60×`) by **`2.770×`** on the 76 controlled-distance bucket pairs — matching `(1.00 / 0.60)² = 2.778×` to **`0.3%`**!
- Simultaneously, the output pixel count drops by `(4080 × 1840) / (3264 × 1472) = (5/4)² = 1.5625×`, leaving `area_frac` shifted by `2.778 / 1.5625 = 1.778×` (empirically measured **`1.773×`** on the 76 bucket pairs!).
- Therefore:
  1. **`area_px / (zoom_multiple ** 2)`** collapses the paired Wide/UW ratio to **`1.007`** (bucket), **`0.900`** (all `175` same-scene pairs), and wins **lowest `med|log(W/UW)|` on all 4 scopes** without double-counting `focal_length_mm`!
  2. **`area_px / ((focal_length_mm * zoom_multiple) ** 2)`** catastrophically double-counts the lens switch (`med(W/UW) = 0.118–0.152`, a `6.6×–8.5×` distortion) and is **permanently disqualified**.
  3. **`area_frac_optical_norm = area_frac / (zoom_multiple ** 2)`** is also retained as an explicit secondary feature column in `features_clean.parquet` (zero raw pixel dimension dependence; highest univariate `r = +0.2972` across all `1,738` images), enabling Phase 3's 4-condition ablation to compare resolution-agnostic vs. native-pixel optical scaling transparently!
  4. **Linear optical descriptors** follow the exact matching square-root law:
     - `area_optical_norm = area_px / (zoom_multiple ** 2)`
     - `sqrt_area_optical_norm = math.sqrt(area_optical_norm)`
     - `perim_optical_norm = perimeter_px / zoom_multiple`
     - `major_axis_optical_norm = major_axis_px / zoom_multiple`
     - `minor_axis_optical_norm = minor_axis_px / zoom_multiple`
     - `area_frac_optical_norm = area_frac / (zoom_multiple ** 2)`

---

### Spike 3: 3-Scenario Segmentation & SF-400 LCD Exclusion Sweep Across All `1,738` Clean Images (`D-05`, `D-06`, `D-07`)

**1. Deterministic Bimodal Bucket Detector (`scene_is_bucket`) at `768 px` (`[VERIFIED: full 1,738-image border-ring scan, 2026-11-22]`):**
Scanning the outer `13%` perimeter ring across all `1,738` clean images revealed a razor-sharp physical phase transition between smooth white plastic bucket walls and high-frequency wood-shaving straw bedding:
- `ring_gmed` (median Scharr gradient magnitude on Lab `L*` in outer `13%` ring): sharp isolated peak **`[10, 20): 331 images`** vs. **`>= 150: 1,021 images`**!
- `smooth_white_frac` (`(L >= 140) & (C <= 22) & (gmag <= 40)` over outer `13%` ring): deep bimodal valley at **`[0.50, 0.70]`** (`[0.5, 0.6): 26`, `[0.6, 0.7): 16`, `[0.7, 0.8): 57`, `[0.8, 0.9): 164`, `[0.9, 1.0): 103`).
- Requiring **`smooth_white_frac >= 0.50 AND ring_gmed <= 45.0 AND sides_sw >= 3`** (where at least 3 of the 4 border strips independently have `>= 35%` smooth white plastic) classifies **`377` images (`21.69%`: `254` `<500g`, `64` `500–1000g`, `59` `>1000g`)** as `bucket_dorsal` (`scene_is_bucket = True`) and **`1,361` images (`78.31%`)** as `scale_or_litter` (`scene_is_bucket = False`) — closely matching Audit §6.4/§8.1's visual estimate (`~351` clean bucket images, `~20.1%`).

**2. Scenario B (White Plastic Bucket — Including White-on-White `>1,000 g` Broilers):**
Empirical diagnosis of v1/v2 prototypes uncovered why naive bucket thresholding failed: the bucket floor contains radial tunnel shading and scattered straw/droppings, so global `dL` or `b*`/`grad` thresholds flooded the entire rounded-square bucket floor (`~40–50%` of frame regardless of bird size, `15%` fallback).
- **Winning Classical Solution (Smooth-Wall Peeling + Contact-Shadow Barrier):**
  1. Compute `bh_rim = morphologyEx(l_eq, MORPH_BLACKHAT, ellipse(17×17))` (contact-shadow moat around the bird) and adaptive smoothness threshold `tex_wall_thr = min(46.0, max(24.0, g_wall * 1.75 + 7.0))` (calibrated from smooth border-ring wall pixels so Ultrawide `g_wall ≈ 14` gets a tight `~32` barrier that cannot bleed into white feathers!).
  2. Mark `shadow_barrier = (bh_rim > 18.0) | (geq_blur > tex_wall_thr)` and identify smooth plastic wall candidates `wall_cand = (~shadow_barrier) & (|b - b_wall| <= 9.5) & (L >= L_wall - 45.0)`.
  3. Keep **only connected components of `wall_cand` touching the image frame perimeter** → `wall_mask`. Define `interior = ~wall_mask`.
  4. Inside `interior` only, separate the brighter, coherently-feathered bird from darker/chaotic floor straw/shadows by fusing: (a) adaptive `Otsu(l_ch[interior])` + `L_p50` upper luminance class (`geq_blur <= 260`, `a <= 133.5`), (b) yellow down (`b >= max(147, b_wall + 11)`, `L >= L_p35 + 10`, `a <= 130`), (c) pink comb/skin (`a >= 134.5`, `L >= 90`), and (d) centrally-weighted feather micro-texture (`geq_blur ∈ [max(42, g_wall*2.2+12), 220]`, `center_prior > 0.25`, `L >= L_p35 + 6`).
  5. Morphological elliptical open (`7×7`), close (`15×15`), `binary_fill_holes`, and centrally-weighted dominant connected component selection.

**3. Scenario A & C (SF-400 Scale on Straw & Oblique Straw Litter) + `lcd_suppression.py`:**
- **Dedicated Rotation-Invariant SF-400 Suppression (`lcd_suppression.py`, executed strictly BEFORE morphological closing/selection):**
  1. Detect dark, near-achromatic (`L < 105`, `C = hypot(a-128, b-128) < 18`) compact rectangular blobs (`area ∈ [0.0005, 0.08] * frame_area`, rotated-box aspect ratio `1.25–4.8`, extent `>= 0.52`, solidity `>= 0.72`) via `cv2.minAreaRect` (100% rotation-invariant — zero top/bottom strip hardcoding!).
  2. Dilate detected LCD polygons (`19×19` ellipse) to cover the dark bezel and `ON/OFF`/`MODE`/`TARE` buttons.
  3. Flood-suppress **smooth cool/neutral white SF-400 plastic housing/platter** (`L >= 155`, `C <= 12`, `b <= 134`, `geq_blur <= 65`) directly connected to a `17×17`-dilated halo around any detected LCD bezel — stripping the scale body without eroding warm/feathered bird tissue.
  4. Re-zero `full_lcd_excl` after morphological open/close and hole-filling so closing can **never** bridge back over the LCD!
- **Bird-vs-Straw Separation:** Estimate outer-ring straw baseline `(L_bg, a_bg, b_bg, g_bg)`, fuse bright low-redness white plumage (`L >= max(otsu - 6, L_bg + 12)`, `a <= max(130.5, a_bg + 1.5)`), bright yellow chick down (`b >= 136`, `a <= min(129, a_bg + 0.5)`, `geq_blur <= max(190, g_bg * 0.92)`), central pink comb/skin (`a >= 138`, `S >= 45`), and coherent brighter plumage (`L >= L_bg + 18`, `geq_blur < 0.75 * g_bg`), subtract `full_lcd_excl`, apply elliptical `MORPH_OPEN(9×9)` to sever thin leg/platter bridges, `MORPH_CLOSE(13×13)`, `binary_fill_holes`, and centrally-weighted dominant component selection.

**Full 1,738-Image Sweep Verification Results (`[VERIFIED: uv run python v4 sweep, 4 workers, 2026-11-22]`):**

| Branch (`scene_is_bucket`) | Weight Band | `N` | Primary Fallback `N` (%) | Median `yellow_down_ratio` | Median `white_plumage_ratio` | Images with `lcd_excluded_px > 0` |
|---|---|---:|---:|---:|---:|---:|
| `False` (`scale_or_litter`) | `<500g` | 969 | 12 (`1.24%`) | `0.073` | `0.779` | 427 |
| `False` (`scale_or_litter`) | `500-1000g` | 269 | 1 (`0.37%`) | `0.044` | `0.851` | 165 |
| `False` (`scale_or_litter`) | `>1000g` | 123 | 0 (`0.00%`) | `0.011` | `0.857` | 80 |
| `True` (`bucket_dorsal`) | `<500g` | 254 | 0 (`0.00%`) | `0.336` | `0.634` | 0 |
| `True` (`bucket_dorsal`) | `500-1000g` | 64 | 0 (`0.00%`) | `0.384` | `0.610` | 0 |
| `True` (`bucket_dorsal`) | `>1000g` | 59 | 0 (`0.00%`) | `0.248` | `0.650` | 0 |
| **TOTAL** | **All** | **1,738** | **13 (`0.75%`)** | — | — | **672 (`0` bucket FP)** |

- **Gate check:** `13 / 1,738 = 0.75%` primary fallback (well under the `D-07` gate of `< 2.0%` / `≤ 34` images!), **zero crashes**, **`42.9 s`** total wall-clock time on `4` workers (`< 60 s` `D-10` target!).
- **Secondary fallback note:** On the `3` images still outside `[0.015, 0.72]` after the single-threshold z-distance fallback, clamp the fallback percentile adaptively (`p70 → p55 → central ellipse fallback`) so 100% of images return a valid, non-empty mask with `area_frac > 0` and zero `fitEllipse` errors.

---

### Spike 4: Multi-Bird Marker-Based Watershed Across All `13` Quarantined Images (`D-11`, `D-12`)

**Pipeline Tested (Zero Folder Hardcoding, General Relative Parameters):**
1. Shared `io_exif.py` (`working_long_edge = 768`) + `preprocess.py` (Bilateral `d=5` + CIE Lab `L` CLAHE).
2. **Chromatic red-feeder pre-suppression:** `(Lab a* > 146 & S > 65) | (((H <= 10) | (H >= 168)) & S > 105 & a* > 140)`, dilated with an `11×11` ellipse (`red_excl`) — isolates the red plastic feeder rim in `417` without touching yellow chicks or straw.
3. **Robust foreground z-score union mask:** Over `valid_px = (red_excl == 0)`, compute robust median/MAD z-scores combining brightness (`+0.55 * z_L`), non-reddish coolness (`+0.30 * (-z_a)`), and plumage smoothness vs. chaotic wood shavings (`+0.25 * (-z_g_eq_blur)`), threshold via Otsu floor (`max(otsu, 78) & L >= L_med + 4`), gentle elliptical open (`5×5`) + close (`9×9`) + `binary_fill_holes`, and reject narrow architectural wall strips flush against frame edges (`span > 55%` edge length & `< 24%` depth) + tiny speckles (`< 0.45%` frame area).
4. **Scale-adaptive Distance Transform + Bbox-Cropped `h_maxima` + `peak_local_max` + `skimage.segmentation.watershed`:**
   - `dist = ndi.distance_transform_edt(fg_bool)`
   - `dist_s = ndi.gaussian_filter(dist, sigma=max(1.5, 0.06 * dist.max()))`
   - Run `skimage.morphology.h_maxima(dist_s[r0:r1, c0:c1], h=max(1.0, h_rel * ds_max))` **strictly on the padded foreground bounding box** (verified **`313.8 ms → 14.0 ms`**, **22× faster**, `bit-identical = True`!).
   - Run `peak_local_max(np.where(hm, dist_s, 0.0), min_distance=max(10, int(round(min_distance_rel * ds_max))), threshold_abs=max(2.0, 0.28 * ds_max), exclude_border=False)` (modern `(N, 2)` API, empty-coords guarded!), plus ensure any disconnected foreground component `>= 0.45%` frame area with peak `>= 0.25 * ds_max` receives at least its own local maximum seed.
   - Run `skimage.segmentation.watershed(-dist_s, markers, mask=fg_bool)` and prune tiny spurious basins `< max(min_area_ratio * total_fg_area, 0.14 * max_basin_area, 0.0045 * frame_area)`.

**Verified Results Across All `13` Quarantined Multi-Bird Images (`h_rel = 0.20, min_distance_rel = 0.85, min_area_ratio = 0.10`):**

| Folder | Ground-Truth Count | Images | Per-Image Detected Counts | Exact Matches | Within ±1 | Notes |
|---|---:|---:|---|---:|---:|---|
| `417` | 2 | 4 | `2, 2, 2, 2` | **4 / 4** | **4 / 4** | Red plastic feeder cleanly suppressed on all 4 wide/ultrawide shots; touching chicks cleanly split! |
| `1520` | 2 | 4 | `2, 2, 2, 3` | **3 / 4** | **4 / 4** | Touching heavy broilers split from single union blob (`n_comp=1 → 2` on 2 shots); architectural wall strip cleanly rejected. |
| `823` | 4 | 4 | `4, 5, 4, 5` | **2 / 4** | **4 / 4** | All 4 small chicks detected on both Wide and Ultrawide (`0.45%` min-area floor prevents losing ultrawide chicks). |
| `757` | 2 | 1 | `4` | 0 / 1 | 0 / 1 | Low-light oblique shot (`L_p50=130`); interactive Phase 4 `min_area_ratio`/`h_rel` sliders tune live cleanly. |
| **Total** | — | **13** | — | **9 / 13 (`69.2%`)** | **12 / 13 (`92.3%`)** | **Count MAE = `0.38` birds** with zero folder hardcoding; latency **`~100–130 ms`** with bbox-cropped `h_maxima`. |

## Standard Stack

Phase 2 installs **zero** new external packages. All libraries are already declared in `pyproject.toml` (`[VERIFIED: pyproject.toml:5-28]`) and resolved as binary wheels in `uv.lock` under `uv`-managed Python `3.12.14` (`[VERIFIED: uv run python version probe, 2026-11-22]`).

### Core
| Library | Verified Installed Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `opencv-python-headless` | `4.14.0` (`>=4.14.0,<5.0` `[VERIFIED: pyproject.toml:7]`) | Aspect-preserving `cv2.resize(..., INTER_AREA)`, `cv2.bilateralFilter`, `cv2.createCLAHE`, Lab/HSV color conversion, Scharr/Black-Hat/morphology, `findContours`, `fitEllipse`, `minAreaRect`, `HuMoments`, JPEG/PNG encoding | Headless build omits GUI/X11 bloat; classical CV 4.14 API is battle-tested and C++-accelerated (`< 30 ms` per `768 px` image). |
| `scikit-image` | `0.26.0` (`>=0.26.0,<0.27` `[VERIFIED: pyproject.toml:8]`) | `skimage.measure.regionprops`, `skimage.morphology.h_maxima`, `skimage.feature.peak_local_max`, `skimage.segmentation.watershed` | Float-native negative-EDT Watershed and single-pass `regionprops` avoid `cv2.watershed`'s uint8/border `-1` quirks. |
| `scipy` | `1.18.1` (`>=1.18.0,<2.0` `[VERIFIED: pyproject.toml:9]`) | `scipy.ndimage.distance_transform_edt`, `scipy.ndimage.gaussian_filter`, `scipy.ndimage.binary_fill_holes`, `scipy.ndimage.label` | Exact Euclidean distance transform yields cleaner, rounder torso peaks than OpenCV's chamfer distance approximation. |
| `pillow` | `13.0.0` (`>=13.0.0,<14.0` `[VERIFIED: pyproject.toml:10]`) | Exclusive EXIF-first image loader (`im.getexif().get_ifd(0x8769)`, aspect-preserving `im.draft("RGB", ...)`, `ImageOps.exif_transpose().convert("RGB")`) | Only library exposing raw Xiaomi tag `39321` + EXIF orientation before transpose (`PROIBIÇÃO 2` / `D-02`). |

### Supporting
| Library | Verified Installed Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `numpy` | `2.5.5` (`>=2.5.0,<3.0` `[VERIFIED: pyproject.toml:11]`) | Mask indexing (`arr[mask == 255]`), quantiles, circular Hue stats, Hu log transforms | Always — vectorized numeric substrate across all PDI modules. |
| `pandas` + `pyarrow` | `3.1.1` / `26.0.1` (`[VERIFIED: pyproject.toml:12-13]`) | Reading `manifest_clean.parquet`, writing typed PyArrow `data/processed/features_clean.parquet` | `backend/scripts/extract_features.py` (CoW-safe `.loc[...]` mutations, explicit `pa.schema`). |
| `concurrent.futures` (stdlib) / `joblib` | stdlib / `1.6.0` (`[VERIFIED: pyproject.toml:16]`) | `ProcessPoolExecutor(max_workers=4)` with `initializer=cv2.setNumThreads, initargs=(1,)` | Offline batch extraction (`extract_features.py`): zero extra pickle/loky overhead, verified `42.9 s` over all `1,738` images with `< 250 MB` total worker RSS. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `ProcessPoolExecutor(max_workers=4)` | `joblib.Parallel(n_jobs=4)` | Both stay inside the 4-worker RAM ceiling; stdlib `ProcessPoolExecutor` supports `initializer=cv2.setNumThreads, initargs=(1,)` cleanly at worker spawn and avoids loky temp-memmap overhead on 7.1 GiB RAM. **Use `ProcessPoolExecutor(max_workers=4)`.** |
| `cv2.bilateralFilter(d=5, 50, 50)` | `cv2.medianBlur(ksize=5)` | Bilateral (`d=5`) runs in just `4.6 ms` (with CLAHE!) at `768 px` while preserving crisp feather edges and contact-shadow rims needed for Scenario B bucket peeling; keep `smoothing_method: Literal["bilateral", "median"] = "bilateral"` in `PDIConfig` per `D-03`. |
| `skimage.segmentation.watershed` | `cv2.watershed` | `cv2.watershed` requires 3-channel uint8 input and writes `-1` boundary borders that complicate per-instance mask extraction; `skimage.segmentation.watershed(-dist_s, markers, mask=fg_bool)` directly yields clean integer instance masks for `D-12`. |

**Installation:** None required — environment is already locked (`uv sync --frozen` / `uv run pytest`).

## Package Legitimacy Audit

> **Phase 2 installs ZERO new external packages.** All 8 packages below were previously installed and SHA-256 hash-locked in `uv.lock` during Phase 1 (`test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions` green, `[VERIFIED: backend/tests/test_io_exif.py:289-309]`). The `gsd-tools.cjs query package-legitimacy check --ecosystem pypi` seam was executed for audit completeness: PyPI's JSON API does not publish weekly download counts (`unknown-downloads`) and several core scientific packages published point releases in Oct–Nov 2026 (`too-new`), causing the heuristic seam to emit `SUS` across canonical packages (`numpy`, `scipy`, `pandas`, `pillow`, `opencv-python-headless`, `scikit-image`, `pyarrow`, `joblib`). All 8 packages are official Tier-1 Python scientific libraries already verified in `uv.lock` with zero new installs required in Phase 2.

| Package | Registry | Installed Version | Locked in `uv.lock` | Repo URL (PyPI Metadata) | Seam Output | Disposition |
|---|---|---|---|---|---|---|
| `opencv-python-headless` | PyPI | `4.14.0` | Yes (Phase 1) | `https://github.com/opencv/opencv-python` | `SUS` (`too-new`, `unknown-downloads`) | Approved — already locked in Phase 1 (`uv.lock`), no install |
| `scikit-image` | PyPI | `0.26.0` | Yes (Phase 1) | Official SciPy ecosystem | `SUS` (`unknown-downloads`, `no-repository`) | Approved — already locked in Phase 1 (`uv.lock`), no install |
| `scipy` | PyPI | `1.18.1` | Yes (Phase 1) | Official SciPy ecosystem | `SUS` (`unknown-downloads`, `no-repository`) | Approved — already locked in Phase 1 (`uv.lock`), no install |
| `pillow` | PyPI | `13.0.0` | Yes (Phase 1) | `https://github.com/python-pillow/Pillow` | `SUS` (`unknown-downloads`) | Approved — already locked in Phase 1 (`uv.lock`), no install |
| `numpy` | PyPI | `2.5.5` | Yes (Phase 1) | Official NumPy ecosystem | `SUS` (`too-new`, `unknown-downloads`, `no-repository`) | Approved — already locked in Phase 1 (`uv.lock`), no install |
| `pandas` | PyPI | `3.1.1` | Yes (Phase 1) | Official PyData ecosystem | `SUS` (`too-new`, `unknown-downloads`, `no-repository`) | Approved — already locked in Phase 1 (`uv.lock`), no install |
| `pyarrow` | PyPI | `26.0.1` | Yes (Phase 1) | `https://arrow.apache.org/` | `SUS` (`too-new`, `unknown-downloads`) | Approved — already locked in Phase 1 (`uv.lock`), no install |
| `joblib` | PyPI | `1.6.0` | Yes (Phase 1) | `https://github.com/joblib/joblib` | `SUS` (`unknown-downloads`) | Approved — already locked in Phase 1 (`uv.lock`), no install |

**Packages removed due to [SLOP] verdict:** none
**New packages installed in Phase 2:** none (zero `uv add` / `pip install` calls in Phase 2 plans)

## Architecture Patterns

### System Architecture Diagram

```text
                      ┌───────────────────────────────────────────────┐
                      │  Input: Path (CLI batch) OR bytes (API upload)│
                      │       + Optional OpticalOverride (f, z)       │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                       ┌─────────────────────────────────────────────┐
                       │  1. EXIF-First Ingestion (io_exif.py)       │
                       │  • Read Orientation/Focal/Xiaomi 39321 FIRST│
                       │  • Aspect-preserving im.draft("RGB", box)   │
                       │  • ImageOps.exif_transpose().convert("RGB") │
                       │  • cv2.resize(INTER_AREA) -> 768px + scale  │
                       └──────────┬───────────────────────┬──────────┘
                                  │                       │
                    [Single-Bird Path]          [Isolated Multi-Bird Path]
                                  ▼                       ▼
         ┌──────────────────────────────────┐  ┌──────────────────────────────────┐
         │ 2. Preprocess (preprocess.py)    │  │ 2W. Shared Preprocess            │
         │ • RGB->BGR, Bilateral d=5        │  │ • Bilateral d=5 + Lab L CLAHE    │
         │ • CIE Lab L-channel CLAHE (2,8x8)│  │ • Chromatic red-feeder suppress  │
         │ • HSV + Scharr gradient maps     │  └──────────────────┬───────────────┘
         └────────────────┬─────────────────┘                     ▼
                          ▼                    ┌──────────────────────────────────┐
         ┌──────────────────────────────────┐  │ 3W. Union Mask + Marker Watershed│
         │ 3. Border-Ring Scene Classifier  │  │ • Robust z-score (L, -a, -grad)  │
         │ smooth_white>=0.50 & gmed<=45    │  │ • Wall-strip reject + fill holes │
         │ & sides_sw>=3                    │  │ • EDT + adaptive Gaussian smooth │
         └───────┬──────────────────┬───────┘  │ • Bbox-cropped h_maxima + peaks  │
                 │ True             │ False    │ • skimage.segmentation.watershed │
                 ▼                  ▼          │ • Prune tiny spurious basins     │
    ┌────────────────────┐ ┌────────────────┐  └──────────────────┬───────────────┘
    │ Scenario B: Bucket │ │ Scenarios A&C  │                     ▼
    │ • Black-Hat shadow │ │ • lcd_suppress │  ┌──────────────────────────────────┐
    │   barrier          │ │   (bezel+smooth│  │ 4W. Per-Instance Features + 4x   │
    │ • Peel border-conn │ │   housing=0!)  │  │     Watershed Visual Stages      │
    │   smooth white wall│ │ • Bird vs straw│  │ • Calls extract_mask_gated_      │
    │ • Interior Otsu +  │ │   Lab/HSV/grad │  │   features() per separated mask  │
    │   yellow/pink/tex  │ │ • Open(9x9) -> │  │ • Returns WatershedResult        │
    └─────────┬──────────┘ │   Close(13x13) │  └──────────────────────────────────┘
              └──────┬─────┴────────┬───────┘
                     ▼              ▼
         ┌──────────────────────────────────┐
         │ 4. Dominant Central Component +  │
         │    Envelope Guard [0.015, 0.72]  │
         │    (Adaptive Fallback if outside)│
         └────────────────┬─────────────────┘
                          ▼
         ┌──────────────────────────────────┐
         │ 5. Strict Mask-Gated Features    │
         │    (features.py, mask == 255)    │
         │ • Morphometrics (regionprops/Hu) │
         │ • Chromatic + eroded texture     │
         │ • Optical norm (area_px / z^2)   │
         └───────┬──────────────────┬───────┘
                 │ return_xray=True │ return_xray=False (CLI)
                 ▼                  ▼
    ┌────────────────────┐ ┌────────────────────────────────────┐
    │ 6. 4-Stage X-Ray   │ │ Batch Runner (extract_features.py) │
    │ @640px (<250 KB)   │ │ ProcessPoolExecutor(max_workers=4) │
    │ -> PDIResult       │ │ -> features_clean.parquet (1,738)  │
    └────────────────────┘ │ -> pdi_calibration_audit.json      │
                           └────────────────────────────────────┘
```

### Recommended Project Structure

Existing Phase 1 files (`[VERIFIED: find backend -maxdepth 3 -type f]`) plus Phase 2 additions:

```text
backend/
├── app/
│   ├── config.py                  # [EXTEND] Add FEATURES_CLEAN_PATH, PDI_CALIBRATION_AUDIT_PATH
│   └── pdi/
│       ├── __init__.py            # [NEW] Clean public exports for pipeline, watershed, config, io_exif
│       ├── config.py              # [NEW] Frozen PDIConfig + deterministic 16-char SHA-256 config_hash()
│       ├── io_exif.py             # [EXTEND] Keep read_exif_metadata/load_oriented_rgb intact; add load_working_rgb() + OpticalContext/OpticalOverride
│       ├── preprocess.py          # [NEW] PreprocessedFrame, Bilateral/Median, Lab CLAHE, HSV, Scharr maps
│       ├── lcd_suppression.py     # [NEW] Rotation-invariant SF-400 LCD bezel + connected smooth housing exclusion
│       ├── segmentation.py        # [NEW] Bimodal scene detector, Scenario B smooth-wall peel, Scenarios A/C bird-vs-straw, fallback guard
│       ├── features.py            # [NEW] Strict mask==255 morphometric, chromatic/texture & optical feature extractor + FEATURE_COLUMNS tuple
│       ├── xray.py                # [NEW] 4-stage X-Ray renderer & Base64 encoder (3x JPEG Q=82 + 1x 1-ch PNG @640px)
│       ├── pipeline.py            # [NEW] Unified run_single_bird_pdi() single-source entrypoint
│       └── watershed.py           # [NEW] Isolated multi-bird Watershed (red-feeder suppress, bbox h_maxima, per-instance features, 4 visuals)
├── scripts/
│   └── extract_features.py        # [NEW] 4-worker CLI -> features_clean.parquet + models/pdi_calibration_audit.json
└── tests/
    ├── test_pdi_preprocess_and_lcd.py   # [NEW] Config hash, draft Path-vs-bytes parity, LCD suppression, fitEllipse guard, bucket peeling
    ├── test_pdi_features_invariance.py  # [NEW] Outside-mask pixel-perturbation invariance (0.0 diff!), rotation invariance, optical formula, banned-column gate
    ├── test_pdi_pipeline_and_xray.py    # [NEW] End-to-end single-bird latency/payload, EXIF-stripped/RGBA uploads, 4-stage X-ray decoding
    ├── test_pdi_watershed.py            # [NEW] Synthetic touching-ellipses split, red-feeder suppression, empty-mask guard, real 13-image multi-bird audit
    └── test_extract_features_integration.py # [NEW] CLI dry-run + real 1,738-row features_clean.parquet schema/fallback/calibration/no-path-leakage verification
```

### Exact Path Constants to Add in `backend/app/config.py`

Verified existing constants (`[VERIFIED: backend/app/config.py:12-19]`):
```python
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATASET_DIR: Path = DATA_DIR / "raw" / "Broiler weights of Cobb Avian"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = REPO_ROOT / "models"
CLEAN_MANIFEST_PATH: Path = PROCESSED_DIR / "manifest_clean.parquet"
QUARANTINE_MANIFEST_PATH: Path = PROCESSED_DIR / "manifest_quarantine.parquet"
AUDIT_SUMMARY_PATH: Path = MODELS_DIR / "dataset_audit_summary.json"
```
Add these two exact constants and include them in `__all__`:
```python
FEATURES_CLEAN_PATH: Path = PROCESSED_DIR / "features_clean.parquet"
PDI_CALIBRATION_AUDIT_PATH: Path = MODELS_DIR / "pdi_calibration_audit.json"
```

### Exact Dataclass Contracts (`@dataclass(frozen=True, slots=True)`)

Following Phase 1's established frozen-dataclass convention (`[VERIFIED: backend/app/pdi/io_exif.py:32-52]`):

1. **`PDIConfig` (`backend/app/pdi/config.py`):**
   ```python
   @dataclass(frozen=True, slots=True)
   class PDIConfig:
       # Canonical working scale & fast JPEG decode (Spike 1)
       working_long_edge: int = 768
       draft_scale_multiplier: float = 1.0
       # Edge-preserving smoothing + Lab CLAHE (D-03)
       smoothing_method: Literal["bilateral", "median"] = "bilateral"
       bilateral_d: int = 5
       bilateral_sigma_color: float = 50.0
       bilateral_sigma_space: float = 50.0
       median_ksize: int = 5
       clahe_clip_limit: float = 2.0
       clahe_tile_grid: tuple[int, int] = (8, 8)
       # Scene classification border ring (Spike 3)
       border_ring_frac: float = 0.13
       bucket_smooth_white_min_frac: float = 0.50
       bucket_ring_grad_max: float = 45.0
       bucket_min_white_sides: int = 3
       # Mask area validity envelope (D-07)
       min_valid_area_frac: float = 0.015
       max_valid_area_frac: float = 0.72
       # Optical EXIF fallback defaults (D-02)
       default_focal_length_mm: float = 4.25
       default_zoom_multiple: float = 1.0
       # X-Ray rendering contract (D-04)
       xray_display_long_edge: int = 640
       xray_jpeg_quality: int = 82
       # Watershed defaults (Spike 4 / D-11)
       watershed_h_rel: float = 0.20
       watershed_min_distance_rel: float = 0.85
       watershed_min_area_ratio: float = 0.10

       def config_hash(self) -> str:
           payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
           return hashlib.sha256(payload).hexdigest()[:16]
   ```
   *Note:* Use `dataclasses.asdict(self)` (not `self.__dict__`!) because `slots=True` dataclasses have no `__dict__` attribute!

2. **`OpticalOverride` & `OpticalContext` (`backend/app/pdi/io_exif.py`):**
   ```python
   @dataclass(frozen=True, slots=True)
   class OpticalOverride:
       focal_length_mm: float | None = None
       zoom_multiple: float | None = None

   @dataclass(frozen=True, slots=True)
   class OpticalContext:
       focal_length_mm: float
       zoom_multiple: float
       exif_optical_present: bool
       override_applied: bool
       lens_class: Literal["wide", "ultrawide", "unknown"]

   @dataclass(frozen=True, slots=True)
   class WorkingImage:
       rgb_work: np.ndarray          # uint8 (H_work, W_work, 3), RGB
       metadata: ExifMetadata        # Pre-draft native raw/oriented dimensions + EXIF
       scale_factor: float           # max(H_work, W_work) / max(oriented_h, oriented_w)
       work_w: int
       work_h: int
   ```
   *Backward compatibility guarantee:* Keep existing `ExifMetadata`, `OrientedImage`, `read_exif_metadata()`, and `load_oriented_rgb()` signatures and return types 100% untouched (`[VERIFIED: backend/app/pdi/io_exif.py:32-52, 144-156]`) so all 39 Phase 1 tests remain green without modification!

3. **`PreprocessedFrame`, `LcdSuppressionResult`, `SingleBirdSegmentation`, `XRayArtifacts`, `PDIResult`:**
   ```python
   @dataclass(frozen=True, slots=True)
   class PreprocessedFrame:
       bgr_work: np.ndarray          # uint8 (H, W, 3) raw working BGR (for X-Ray stage 1 & overlay)
       bgr_smooth: np.ndarray        # uint8 (H, W, 3) edge-preserving smoothed BGR
       bgr_clahe: np.ndarray         # uint8 (H, W, 3) Lab L-CLAHE enhanced BGR (for X-Ray stage 2)
       lab: np.ndarray               # uint8 (H, W, 3) OpenCV Lab of bgr_smooth (L,a,b in [0,255], neutral 128)
       l_clahe: np.ndarray           # uint8 (H, W) CLAHE-enhanced L channel
       hsv: np.ndarray               # uint8 (H, W, 3) OpenCV HSV of bgr_smooth (H in [0,179], S,V in [0,255])
       grad_l: np.ndarray            # float32 (H, W) Scharr magnitude on L
       grad_l_clahe_blur: np.ndarray # float32 (H, W) Gaussian-smoothed (11x11) Scharr magnitude on l_clahe

   @dataclass(frozen=True, slots=True)
   class LcdSuppressionResult:
       exclusion_mask: np.ndarray    # uint8 (H, W) {0, 255} combined LCD bezel + connected smooth housing
       lcd_core_count: int           # Number of dark rectangular LCD bezel blobs detected
       lcd_excluded_px: int          # Count of non-zero pixels in exclusion_mask

   @dataclass(frozen=True, slots=True)
   class SingleBirdSegmentation:
       bird_mask: np.ndarray         # uint8 (H, W) {0, 255}, strictly clean single dominant silhouette
       contour: np.ndarray           # int32 (N, 1, 2) primary external contour in working coords
       ellipse: tuple[tuple[float, float], tuple[float, float], float]  # ((cx, cy), (axis1, axis2), angle_deg)
       ellipse_from_min_area_rect: bool
       scene_type: Literal["bucket_dorsal", "scale_or_litter"]
       scene_is_bucket: bool
       lcd_excluded_px: int
       lcd_core_count: int
       segmentation_fallback: bool
       mask_area_frac: float

   @dataclass(frozen=True, slots=True)
   class XRayArtifacts:
       stage1_original_b64: str      # data-free raw base64 JPEG Q=82 @640px
       stage2_clahe_b64: str         # raw base64 JPEG Q=82 @640px
       stage3_mask_b64: str          # raw base64 1-channel PNG @640px
       stage4_overlay_b64: str       # raw base64 JPEG Q=82 @640px
       display_w: int
       display_h: int
       total_b64_bytes: int

   @dataclass(frozen=True, slots=True)
   class PDIResult:
       features: dict[str, float]    # Exact keys == FEATURE_COLUMNS (49 strictly mask-gated floats)
       optical: OpticalContext
       segmentation: SingleBirdSegmentation
       pdi_config_hash: str
       latency_ms: float
       xray: XRayArtifacts | None = None
   ```

4. **`WatershedInstance` & `WatershedResult` (`backend/app/pdi/watershed.py`):**
   ```python
   @dataclass(frozen=True, slots=True)
   class WatershedInstance:
       instance_id: int              # 1..bird_count, sorted descending by area_work_px
       centroid_xy: tuple[float, float]
       area_work_px: int
       area_px: float
       area_frac: float
       area_optical_norm: float
       bbox_work: tuple[int, int, int, int]  # (x, y, w, h) in working coords for UI annotation only
       features: dict[str, float]    # Full 49-key FEATURE_COLUMNS extracted via extract_mask_gated_features()

   @dataclass(frozen=True, slots=True)
   class WatershedVisuals:
       stage1_original_b64: str      # JPEG Q=82 @640px
       stage2_union_mask_b64: str    # 1-ch PNG @640px
       stage3_distance_peaks_b64: str# JPEG Q=82 @640px (INFERNO/TURBO heatmap + white peak markers)
       stage4_color_basins_b64: str  # JPEG Q=82 @640px (blended distinct basin colors + ID labels)
       total_b64_bytes: int

   @dataclass(frozen=True, slots=True)
   class WatershedResult:
       bird_count: int
       instances: tuple[WatershedInstance, ...]
       optical: OpticalContext
       h_rel: float
       min_distance_rel: float
       min_area_ratio: float
       red_feeder_excluded_px: int
       pdi_config_hash: str
       latency_ms: float
       visuals: WatershedVisuals | None = None
   ```

### Exact Frozen Schema of `data/processed/features_clean.parquet` (`D-08`, `D-10`)

Total **60 columns** = **11 join/audit metadata columns** + **49 strictly mask-gated numeric feature columns (`FEATURE_COLUMNS`)**. Zero raw pixel-dimension columns (`oriented_w`, `oriented_h`, `megapixels`) and zero axis-aligned bounding-box columns (`bbox_w`, `bbox_h`) appear anywhere in `features_clean.parquet` (`D-08`, Pitfall #3).

#### A. Join & Audit Columns (11 columns — exact order & PyArrow dtypes matching `manifest_clean.parquet` `[VERIFIED: backend/app/ingestion/manifest.py:116-143]`)
| Column | PyArrow Type | Purpose |
|---|---|---|
| `image_id` | `pa.string()` | Primary join key (matches `manifest_clean.parquet` 1:1 row order) |
| `fold` | `pa.int8()` | Locked 5-fold `StratifiedGroupKFold` index (`0..4`) |
| `session_id` | `pa.string()` | Grouping key for anti-leakage verification |
| `weight_g` | `pa.int32()` | Cleaned target label (`116..2093`, with `367 → 267`) |
| `weight_band` | `pa.string()` | Coarse stratum (`"<500g"`, `"500-1000g"`, `">1000g"`) |
| `lens_class` | `pa.string()` | `"wide"` / `"ultrawide"` / `"unknown"` (stratified eval only, not a raw tree shortcut) |
| `scene_type` | `pa.string()` | `"bucket_dorsal"` / `"scale_or_litter"` |
| `scene_is_bucket` | `pa.bool_()` | Boolean indicator for Scenario B stratified evaluation |
| `lcd_excluded_px` | `pa.int32()` | Working-res pixels suppressed by `lcd_suppression.py` |
| `segmentation_fallback` | `pa.bool_()` | True iff primary mask fell outside `[0.015, 0.72]` envelope |
| `pdi_config_hash` | `pa.string()` | 16-char SHA-256 hash of `PDIConfig` |

#### B. Strictly Mask-Gated Feature Columns (`FEATURE_COLUMNS`: 49 `pa.float64()` columns)

1. **Rotation-Invariant Morphometrics (14 columns):**
   - `area_px` — native back-projected silhouette area `area_work_px / (scale_factor ** 2)`
   - `perimeter_px` — native back-projected perimeter `perimeter_work_px / scale_factor`
   - `major_axis_px` — native back-projected fitted-ellipse major axis (fallback `minAreaRect` long side)
   - `minor_axis_px` — native back-projected fitted-ellipse minor axis (fallback `minAreaRect` short side)
   - `convex_area_px` — native back-projected convex hull area `area_convex_work / (scale_factor ** 2)`
   - `area_frac` — resolution-agnostic `area_work_px / (work_w * work_h)`
   - `aspect_ratio_ellipse` — `major_axis_px / max(minor_axis_px, 1e-6)` (`>= 1.0`, rotation-invariant)
   - `eccentricity` — `regionprops[0].eccentricity` (`[0, 1]`)
   - `solidity` — `area_work_px / max(area_convex_work, 1e-6)`
   - `circularity` — `4 * pi * area_work_px / max(perimeter_work_px ** 2, 1e-6)` (clamped `[0, 1.5]`)
   - `extent_rotrect` — `area_work_px / max(rotrect_w * rotrect_h, 1e-6)` (rotation-invariant extent!)
   - `log_hu_1`, `log_hu_2`, `log_hu_3` — signed log Hu moments `-sign(h) * log10(|h| + 1e-12)` (rotation/scale-invariant)

2. **Scale-Invariant Plumage Chroma & Eroded Interior Feather Texture (26 columns — strictly indexed at `bird_mask == 255`):**
   - `lab_l_mean`, `lab_l_std`, `lab_l_p10`, `lab_l_p50`, `lab_l_p90` (5)
   - `lab_a_mean`, `lab_a_std`, `lab_a_p10`, `lab_a_p50`, `lab_a_p90` (5)
   - `lab_b_mean`, `lab_b_std`, `lab_b_p10`, `lab_b_p50`, `lab_b_p90` (5)
   - `hsv_s_mean`, `hsv_s_std`, `hsv_v_mean`, `hsv_v_std` (4)
   - `hsv_h_sin_mean`, `hsv_h_cos_mean` (2) — circular Hue stats computed **strictly over masked pixels with `S >= 25`** (returns `0.0` when fewer than 10 saturated pixels exist, eliminating low-saturation Hue noise per `D-08`/Pitfall #2!)
   - `yellow_down_ratio`, `white_plumage_ratio`, `pink_skin_ratio` (3) — `(b* > 142) & (L* > 120) & (a* < 133)`, `(L* > 165) & (hypot(a*-128, b*-128) < 16)`, and `(a* > 136) & (L* > 90) & (S >= 25)`
   - `feather_grad_mean`, `feather_laplacian_var` (2) — Scharr gradient mean & Laplacian variance computed **strictly inside a 3-px eroded interior mask** (`erode(bird_mask, ellipse(7×7))`, returning `0.0` if erosion empties) so background boundary pixels **never** bleed into `3×3` convolution stencils!

3. **EXIF Optical & Empirically Calibrated Descriptors (9 columns):**
   - `focal_length_mm` — resolved EXIF or override or default `4.25`
   - `zoom_multiple` — resolved Xiaomi `39321` or override or default `1.0`
   - `exif_optical_present` — `1.0` if EXIF provided both focal & zoom else `0.0`
   - `area_optical_norm` — **`area_px / (zoom_multiple ** 2)`** (empirical Spike 2 winner: `med(W/UW) = 1.007` bucket, `0.900` same-scene!)
   - `sqrt_area_optical_norm` — `math.sqrt(area_optical_norm)`
   - `perim_optical_norm` — `perimeter_px / zoom_multiple`
   - `major_axis_optical_norm` — `major_axis_px / zoom_multiple`
   - `minor_axis_optical_norm` — `minor_axis_px / zoom_multiple`
   - `area_frac_optical_norm` — `area_frac / (zoom_multiple ** 2)` (resize-agnostic companion for Phase 3 ablation)

### Exact Schema of `models/pdi_calibration_audit.json` (`D-09`, `PROIBIÇÃO 4`)

Aggregate-only UTF-8 deterministic JSON (`indent=2, sort_keys=True`), validated against `_FORBIDDEN_LEAK_SUBSTRINGS = (".jpg", "rel_path", "/home/", "data/raw")` (exact same guard pattern as `[VERIFIED: backend/app/ingestion/audit_summary.py:50, 315-326]`):
```json
{
  "schema_version": "1.0.0",
  "pdi_config_hash": "<16-char sha256>",
  "pdi_config": { "...": "..." },
  "dataset_extraction_summary": {
    "n_clean_images": 1738,
    "n_bucket_dorsal": 377,
    "n_scale_or_litter": 1361,
    "primary_fallback_count": 13,
    "primary_fallback_pct": 0.75,
    "lcd_suppression_triggered_count": 672,
    "lcd_suppression_bucket_false_positives": 0,
    "batch_wall_clock_sec": 42.9,
    "workers": 4
  },
  "paired_dual_lens_calibration": {
    "winning_formula": "area_px / (zoom_multiple ** 2)",
    "rejected_double_count_formula": "area_px / ((focal_length_mm * zoom_multiple) ** 2)",
    "resolution_area_ratio_wide_over_uw": 1.5625,
    "expected_zoom_area_ratio_wide_over_uw": 2.7778,
    "scopes": {
      "all_mixed_lens_sessions": { "n_pairs": 127, "formulas": { "...": { "median_wide_over_uw": 0.0, "iqr_25": 0.0, "iqr_75": 0.0, "median_abs_log_ratio": 0.0 } } },
      "same_scene_pairs": { "n_pairs": 175, "formulas": {} },
      "bucket_only_pairs": { "n_pairs": 76, "formulas": {} },
      "scale_or_litter_pairs": { "n_pairs": 99, "formulas": {} }
    }
  }
}
```

### Critical Pattern: Guaranteeing `0.0` Bit-Identical Outside-Mask Perturbation Invariance (`PROIBIÇÃO 2` / `D-08`)

**Subtle Trap!** Suppose the anti-leakage unit test calls `extract_mask_gated_features(prep_clean, bird_mask, ...)` vs `extract_mask_gated_features(prep_perturbed, bird_mask, ...)` where bright fake LCD digits were painted onto `bird_mask == 0`:
- If `feather_grad_mean` or `feather_laplacian_var` runs `cv2.Scharr(l_clahe)` or `cv2.Laplacian(l_clahe)` on the **unmasked** `l_clahe` image BEFORE indexing `[eroded_mask]`, OR if `prep` itself ran `cv2.bilateralFilter`/`CLAHE` over the perturbed image before calling `extract_mask_gated_features`:
  1. For `extract_mask_gated_features()` given `bgr_work` (or `lab`/`hsv`) and `bird_mask`: if you compute `Scharr`/`Laplacian` on `l_ch` (`3×3` kernel) and index `eroded_mask = cv2.erode(bird_mask, 5×5_ellipse)` (margin `>= 2 px` inside `bird_mask == 255`), **ZERO** `3×3` kernel tap can ever touch a pixel where `bird_mask == 0`!
  2. **Even better:** Compute masked Scharr/Laplacian strictly inside `bird_mask == 255` with `l_masked = np.where(bird_mask > 0, lab[..., 0], 0)` and evaluate ONLY on `interior = cv2.erode(bird_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0` (3-px safety border!). If `interior` is empty (tiny synthetic mask), return `0.0` for texture features rather than reading boundary pixels!
  3. **Test both levels:**
     - **Level 1 (Feature-Extractor Contract Test — strict `0.0` max diff):** Given a fixed `bird_mask`, construct two raw `bgr_work` images identical on `bird_mask == 255` and wildly perturbed (high-contrast fake LCD digits `"9999"` + neon noise) on `bird_mask == 0`. Pass raw masked conversions (`lab_raw = cv2.cvtColor(bgr_work, COLOR_BGR2Lab)`, `hsv_raw = cv2.cvtColor(bgr_work, COLOR_BGR2HSV)`) into `extract_mask_gated_features()` and assert **exact `0.0` max absolute difference across ALL 49 features**!
     - **Level 2 (Full-Pipeline Anti-LCD Test):** Paint a realistic synthetic SF-400 dark LCD rectangle with white digits `"1850"` next to a synthetic bird on straw, run full `run_single_bird_pdi()`, and assert `seg.lcd_excluded_px > 0` AND `np.count_nonzero(seg.bird_mask & lcd_gt_mask) == 0`!
- **Wait — watch out for `lab`/`hsv` from `bgr_smooth` vs `bgr_work`!** Because `cv2.bilateralFilter(d=5)` has spatial radius `2 px` and `CLAHE` computes `8×8` tile histograms across the whole image, `extract_mask_gated_features()` MUST compute its chromatic & texture descriptors from **raw un-CLAHE'd, point-wise `cv2.cvtColor(bgr_work, COLOR_BGR2Lab)` and `COLOR_BGR2HSV`** (100% pixel-local, zero spatial bleed across the mask boundary!) rather than tile-coupled `l_clahe` or 5×5-smoothed `bgr_smooth`! That guarantees:
  - True physical plumage color (un-distorted by CLAHE contrast stretching!)
  - **Mathematical `0.0` bit-identical invariance** whenever ANY pixel at `bird_mask == 0` is modified — even `1 px` outside the mask boundary!

### Anti-Patterns to Avoid
- **Square `im.draft("RGB", (L, L))` on 20:9 images:** Silently fails to trigger libjpeg DCT scaling on `4080×1840` (`1840/2 < 1536`), costing `+120 ms` and `16×` RAM. Always pass aspect-preserving `(int(raw_w * sc), int(raw_h * sc))` AFTER calling `_extract_exif_from_open_image(im)`.
- **Computing features from `bgr_clahe` or `bgr_smooth`:** CLAHE tile histograms couple outside-mask LCD/background pixels into inside-mask `L*`, breaking `PROIBIÇÃO 2` strict outside-mask invariance! Keep `bgr_smooth` and `bgr_clahe` strictly for **segmentation & X-Ray Stage 2**, and compute mask-gated chromatic/texture features from pixel-local `bgr_work` (`bird_mask == 255` + 3-px eroded interior for `3×3` Scharr/Laplacian).
- **Double-counting `focal_length_mm * zoom_multiple`:** Verified `6.6×–8.5×` distortion (`med(W/UW) = 0.118–0.152`). Never use `area_px / ((f * z) ** 2)`.
- **Using `self.__dict__` on `@dataclass(frozen=True, slots=True)`:** Raises `AttributeError`. Use `dataclasses.asdict(self)` in `PDIConfig.config_hash()`.
- **Running `h_maxima` on the full `346×768` distance map:** Costs `314 ms` vs `14 ms` on the padded foreground bounding box (`bit-identical = True`).
- **Passing `indices=False` to `skimage.feature.peak_local_max`:** Removed in modern scikit-image (`0.26.0` verified signature returns `(N, 2)` coordinates only). Always guard `if coords.size > 0:`.
- **Hardcoding folder names (`"417"`, `"2093"`) or top/bottom frame strip coordinates (`y > 0.7 * H`):** Violates rotation invariance (`81.86%` rotated dataset + top-down accelerometer jitter) and fails on user uploads.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| EXIF orientation & Xiaomi vendor tag `39321` parsing | Custom JPEG APP1 byte parser or `cv2.imdecode` | Existing `backend/app/pdi/io_exif.py` (`PIL.Image.open` + `ImageOps.exif_transpose`) | Already handles `IFDRational(0,0)`, 16 KB Xiaomi cap, all 8 EXIF orientations, and AST ban compliance (`[VERIFIED: backend/app/pdi/io_exif.py:54-156]`). |
| Connected-component morphological geometry | Manual moment/perimeter/convexity loops | `skimage.measure.regionprops` + `cv2.fitEllipse` / `cv2.minAreaRect` + `cv2.HuMoments` | Single verified pass computes `area`, `perimeter`, `axis_major_length`, `axis_minor_length`, `eccentricity`, `area_convex`, `solidity` accurately. |
| Exact Euclidean distance transform & morphological hole fill | OpenCV chamfer `distanceTransform` approximations or manual floodfill loops | `scipy.ndimage.distance_transform_edt` + `scipy.ndimage.binary_fill_holes` | Exact L2 EDT gives smoother torso maxima for Watershed; `binary_fill_holes` handles interior holes cleanly in 1 call. |
| Prominence-based peak suppression & Watershed | Custom flooding priority queue or `cv2.watershed` border-`-1` hacks | `skimage.morphology.h_maxima` (bbox-cropped) + `skimage.feature.peak_local_max` + `skimage.segmentation.watershed` | Float-native negative-EDT Watershed directly emits clean integer instance labels `1..K` for per-instance mask-gated feature extraction. |
| Atomic Parquet serialization & zero-path-leakage JSON check | CSV exports or ad-hoc `json.dump` | Typed `pa.schema` Parquet write (`manifest.py` pattern) + `_FORBIDDEN_LEAK_SUBSTRINGS` guard (`audit_summary.py` pattern) | Preserves exact dtypes under Pandas 3.1 CoW and guarantees `PROIBIÇÃO 4` compliance. |

**Key insight:** Every classical operator needed for AvianScale CV exists in highly optimized C/Cython inside `opencv-python-headless 4.14`, `scikit-image 0.26`, and `scipy 1.18`. Engineering value comes from **domain-specific photometric composition** (smooth-wall peeling + contact-shadow barrier, rotation-invariant LCD + connected plastic housing exclusion, empirically calibrated optical scaling, bbox-cropped `h_maxima`), NOT reimplementing low-level CV primitives.

## Common Pitfalls

### Pitfall 1: Square `PIL.Image.draft()` Box Silently No-Ops on `20:9` JPEGs
**What goes wrong:** Calling `im.draft("RGB", (working_long_edge * 2, working_long_edge * 2))` on `4080×1840` images silently decodes all `7.51 MP` (`147 ms`), tripling latency and quadrupling worker RAM.
**Why it happens:** Pillow's libjpeg draft decoder only picks a `1/2`, `1/4`, or `1/8` DCT factor if **both** scaled width and scaled height remain `>=` the requested box dimensions (`1840 / 2 = 920 < 1536`).
**How to avoid:** Extract `ExifMetadata` **first** (before `im.draft` mutates `im.size`!), then compute `draft_scale = min(1.0, (working_long_edge * draft_mult) / max(im.size))` and call `im.draft("RGB", (max(1, int(im.size[0] * draft_scale)), max(1, int(im.size[1] * draft_scale))))`.
**Warning signs:** Single-image load time `> 80 ms` on a `4080×1840` JPEG; `im.size` unchanged after `im.draft()`.

### Pitfall 2: Bucket Floor Flooding on Scenario B (White Plastic Bucket)
**What goes wrong:** Using global `L_wall - L > 14` or naive `b*`/`gradient` thresholds captures the entire shaded/straw-flecked bucket floor (`~40–55%` of frame), making a `150 g` chick look the same area as a `1,800 g` broiler (`r_bucket ≈ 0.01`, `15%` fallback). Conversely, too high a wall smoothness threshold (`48.0`) on Ultrawide (`g_wall ≈ 14`) lets the wall mask bleed into white `>1,000 g` broiler backs.
**Why it happens:** Deep white buckets create radial illumination darkening and hold straw/dropping debris on the bottom floor, creating a 3-zone radial topology (`smooth bright outer wall` → `darker/textured floor moat` → `bright coherent central bird`).
**How to avoid:** Use **border-connected smooth-wall peeling** with an adaptive wall smoothness bound `min(46.0, max(24.0, g_wall * 1.75 + 7.0))` **gated by a Black-Hat contact-shadow barrier** (`bh_rim > 18`), then segment the bird strictly inside `interior = ~wall_mask` via interior Otsu + chroma + central feather texture.
**Warning signs:** `mask_area_frac > 0.35` median on `<500 g` bucket chicks, or `>1,000 g` ultrawide bucket `area_frac` dropping below `<500 g` chicks.

### Pitfall 3: Dual-Lens `focal × zoom` Double-Counting & Raw Resolution Leakage
**What goes wrong:** Dividing `area_px` by `(focal_length_mm * zoom_multiple) ** 2` introduces a verified `6.6×–8.5×` distortion (`med(W/UW) = 0.118–0.152`). Passing raw `oriented_w`/`oriented_h`/`megapixels` lets trees cheat via `3264×1472 ⇒ heavy`.
**Why it happens:** Xiaomi `zoomMultiple` (`1.0×` vs `0.60×`) already accounts for the `2.778×` native pixel-area shift between Wide (`4080×1840`) and Ultrawide (`3264×1472`).
**How to avoid:** Use `area_optical_norm = area_px / (zoom_multiple ** 2)` (verified `med(W/UW) = 1.007` on bucket pairs, `0.900` on all 175 same-scene pairs) plus `area_frac_optical_norm = area_frac / (zoom_multiple ** 2)`, and explicitly assert in unit tests that `oriented_w`, `oriented_h`, `megapixels`, `bbox_w`, `bbox_h` are **absent** from `FEATURE_COLUMNS` and `features_clean.parquet`.
**Warning signs:** Paired Wide/UW median ratio `< 0.5` or `> 1.8` in `models/pdi_calibration_audit.json`.

### Pitfall 4: Feature Leakage Through Pre-Masked Spatial Filters (`PROIBIÇÃO 2`)
**What goes wrong:** Computing mask-gated features from `bgr_clahe` or `bgr_smooth`, or running `Scharr`/`Laplacian` on an un-eroded mask boundary, causes outside-mask pixels (`bird_mask == 0`) to shift inside-mask features by `> 0.0`, failing the outside-mask perturbation invariance test!
**Why it happens:** CLAHE `8×8` tile histograms, `5×5` bilateral neighborhoods, and `3×3` gradient stencils cross the mask border unless color conversions are pixel-local (`bgr_work`) and derivative statistics are restricted to a 3-px eroded interior mask.
**How to avoid:** Extract chromatic features from pixel-local `cv2.cvtColor(bgr_work, ...)` indexed strictly by `bird_mask == 255`, and compute `feather_grad_mean`/`feather_laplacian_var` on `cv2.erode(bird_mask, 7×7 ellipse) > 0` over mask-zeroed `L`.
**Warning signs:** Outside-mask perturbation invariance test reports `max_abs_diff > 0.0`.

### Pitfall 5: OpenCV `uint8` Lab/HSV Scaling, `cv2.fitEllipse` `<5`-Point Crash & Watershed `h_maxima` Latency
**What goes wrong:** (a) Confusing CIE Lab `a*, b* ∈ [-128, 127]` with OpenCV `uint8` `a*, b* ∈ [0, 255]` (neutral `128`) or `360°` Hue with OpenCV `H ∈ [0, 179]`. (b) Calling `cv2.fitEllipse(cnt)` when `len(cnt) < 5` or points are collinear raises `cv2.error (-201)`. (c) Calling `h_maxima(dist_s)` on the entire `346×768` frame consumes `314 ms`.
**How to avoid:** Document OpenCV `uint8` ranges at the top of `preprocess.py` & `features.py`; wrap `fitEllipse` in a guarded helper falling back to `cv2.minAreaRect` whenever `len(cnt) < 5` or `cv2.contourArea(cnt) < 10.0` or `fitEllipse` raises `cv2.error` or yields non-finite/zero axes; run `h_maxima` strictly on the `±2 px`-padded bounding box of `fg_bool` (`14 ms`, bit-identical!).
**Warning signs:** `cv2.error: (-201)` on synthetic or fallback masks; Watershed single-image latency `> 250 ms`.

## Code Examples

All patterns below were verified directly against the installed `opencv-python-headless 4.14.0`, `scikit-image 0.26.0`, `scipy 1.18.1`, and `pillow 13.0.0` runtime (`[VERIFIED: uv run python probes, 2026-11-22]`).

### 1. Aspect-Preserving EXIF-First Working Loader (`backend/app/pdi/io_exif.py`)
```python
def load_working_rgb(
    source: Path | bytes,
    working_long_edge: int = 768,
    draft_scale_multiplier: float = 1.0,
) -> WorkingImage:
    """Read EXIF BEFORE aspect-preserving draft(), exif_transpose(), and INTER_AREA resize."""
    with _open_image_source(source) as im:
        # 1. Extract pre-draft dimensions & optical tags FIRST (draft mutates im.size!)
        metadata = _extract_exif_from_open_image(im)
        if im.format == "JPEG" and working_long_edge > 0:
            raw_long = max(im.size)
            if raw_long > working_long_edge:
                draft_scale = min(1.0, (working_long_edge * draft_scale_multiplier) / float(raw_long))
                draft_box = (
                    max(1, int(im.size[0] * draft_scale)),
                    max(1, int(im.size[1] * draft_scale)),
                )
                im.draft("RGB", draft_box)
        oriented = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.asarray(oriented, dtype=np.uint8)

    h0, w0 = rgb.shape[:2]
    long_cur = max(h0, w0)
    if working_long_edge > 0 and long_cur > working_long_edge:
        resize_scale = working_long_edge / float(long_cur)
        new_w = max(1, int(round(w0 * resize_scale)))
        new_h = max(1, int(round(h0 * resize_scale)))
        rgb_work = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        rgb_work = rgb

    work_h, work_w = rgb_work.shape[:2]
    orig_long = max(metadata.oriented_h, metadata.oriented_w)
    scale_factor = float(max(work_h, work_w)) / float(max(1, orig_long))
    return WorkingImage(
        rgb_work=rgb_work,
        metadata=metadata,
        scale_factor=scale_factor,
        work_w=work_w,
        work_h=work_h,
    )
```

### 2. Guarded Rotation-Invariant Ellipse Fitting (`backend/app/pdi/segmentation.py`)
```python
def fit_ellipse_safe(
    contour: np.ndarray,
) -> tuple[tuple[tuple[float, float], tuple[float, float], float], bool]:
    """Fit ellipse with >=5 point, non-collinear area, and finite-axis guards."""
    rect = cv2.minAreaRect(contour)
    if len(contour) >= 5 and cv2.contourArea(contour) >= 10.0:
        try:
            (cx, cy), (d1, d2), angle = cv2.fitEllipse(contour)
            if (
                math.isfinite(cx)
                and math.isfinite(cy)
                and math.isfinite(d1)
                and math.isfinite(d2)
                and d1 > 0.0
                and d2 > 0.0
            ):
                major, minor = (d1, d2) if d1 >= d2 else (d2, d1)
                return ((float(cx), float(cy)), (float(major), float(minor)), float(angle)), False
        except cv2.error:
            pass
    (rcx, rcy), (rw, rh), rangle = rect
    major, minor = (max(rw, rh), min(rw, rh))
    return ((float(rcx), float(rcy)), (float(major), float(minor)), float(rangle)), True
```

### 3. Bbox-Cropped `h_maxima` + Modern `peak_local_max` + `skimage.segmentation.watershed` (`backend/app/pdi/watershed.py`)
```python
# Source: Verified against scikit-image 0.26.0 & scipy 1.18.1 (22x speedup, bit-identical)
ys, xs = np.nonzero(fg_bool)
r0, r1 = max(0, int(ys.min()) - 2), min(fg_bool.shape[0], int(ys.max()) + 3)
c0, c1 = max(0, int(xs.min()) - 2), min(fg_bool.shape[1], int(xs.max()) + 3)

h_global = max(1.0, h_rel * ds_max)
hm_global = np.zeros_like(fg_bool)
hm_global[r0:r1, c0:c1] = (
    h_maxima(dist_s[r0:r1, c0:c1], h=h_global) & fg_bool[r0:r1, c0:c1]
)

min_dist = max(10, int(round(min_distance_rel * ds_max)))
gated = np.where(hm_global, dist_s, 0.0)
coords = peak_local_max(
    gated,
    min_distance=min_dist,
    threshold_abs=max(2.0, 0.28 * ds_max),
    exclude_border=False,
)
seed_list: list[tuple[int, int]] = [(int(r), int(c)) for r, c in coords] if coords.size > 0 else []
```

### 4. Strict Outside-Mask Perturbation Invariance Test Pattern (`backend/tests/test_pdi_features_invariance.py`)
```python
def test_outside_mask_perturbation_yields_bit_identical_features() -> None:
    rng = np.random.default_rng(42)
    bgr_a = rng.integers(30, 220, size=(346, 768, 3), dtype=np.uint8)
    mask = np.zeros((346, 768), dtype=np.uint8)
    cv2.ellipse(mask, (384, 173), (110, 65), 18.0, 0, 360, 255, -1)

    # Paint extreme fake LCD digits + neon artifacts strictly where mask == 0
    bgr_b = bgr_a.copy()
    bgr_b[mask == 0] = (255, 0, 255)
    cv2.rectangle(bgr_b, (20, 20), (180, 90), (15, 15, 15), -1)
    cv2.putText(bgr_b, "1850", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (250, 250, 250), 3)

    optical = OpticalContext(4.25, 1.0, True, False, "wide")
    f_a = extract_mask_gated_features(bgr_a, mask, scale_factor=768.0 / 4080.0, optical=optical)
    f_b = extract_mask_gated_features(bgr_b, mask, scale_factor=768.0 / 4080.0, optical=optical)

    assert tuple(f_a.keys()) == FEATURE_COLUMNS
    for key in FEATURE_COLUMNS:
        assert f_a[key] == f_b[key], f"PROIBIÇÃO 2 leak detected in feature {key}: {f_a[key]} != {f_b[key]}"
```

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `uv` | Deterministic Python 3.12 virtualenv & CLI execution | ✓ (`[VERIFIED: uv --version]`) | `0.13.9` | — |
| Python (managed by `uv`) | All Phase 2 PDI modules, scripts, pytest | ✓ (`[VERIFIED: uv run python --version]`) | `3.12.14` | — |
| `opencv-python-headless` | `preprocess.py`, `lcd_suppression.py`, `segmentation.py`, `xray.py` | ✓ (`[VERIFIED: uv run python]`) | `4.14.0` | — |
| `scikit-image` | `regionprops`, `h_maxima`, `peak_local_max`, `watershed` | ✓ (`[VERIFIED: uv run python]`) | `0.26.0` | — |
| `scipy` | `distance_transform_edt`, `gaussian_filter`, `binary_fill_holes` | ✓ (`[VERIFIED: uv run python]`) | `1.18.1` | — |
| `pillow` | EXIF-first loader (`io_exif.py`) | ✓ (`[VERIFIED: uv run python]`) | `13.0.0` | — |
| `pandas` + `pyarrow` | Typed Parquet I/O (`features_clean.parquet`) | ✓ (`[VERIFIED: uv run python]`) | `3.1.1` / `26.0.1` | — |
| `pytest` + `ruff` | Automated verification & linting | ✓ (`[VERIFIED: uv run pytest -q -> 39 passed]`) | `9.2.0` / `0.17.6` | — |
| Host RAM / CPU | 4-worker batch extraction (`n_jobs=4`) | ✓ (`16` vCPUs, `7.1 GiB` RAM, `1.5 GiB` available) | Linux x86_64 | Cap workers at `4` + `cv2.setNumThreads(1)` + `1/4` draft decode |
| Materialized Phase 1 Manifests | `extract_features.py` & 13-image Watershed validation | ✓ (`1,738` clean, `15` quarantine rows verified) | Schema `1.0.0` | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none.

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high` (`[VERIFIED: .planning/config.json:48-50]`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Offline Python PDI library & local CLI (no auth boundary in Phase 2). |
| V3 Session Management | No | Stateless pure-function PDI transformations (no user sessions). |
| V4 Access Control | Yes (CLI path containment) | In `extract_features.py`, resolve every `rel_path` against `RAW_DATASET_DIR.resolve()` and enforce `_is_contained_non_symlink()` (`[VERIFIED: backend/app/ingestion/manifest.py:208-216]`) to block path traversal or symlink escapes. |
| V5 Input Validation | **Yes (PRIMARY)** | Keep `PIL.Image.MAX_IMAGE_PIXELS` decompression-bomb guard enabled (`[VERIFIED: backend/tests/test_io_exif.py:258-261]`); bound `OpticalOverride` (`0.5 <= focal_length_mm <= 200.0`, `0.1 <= zoom_multiple <= 20.0`, finite floats); bound Watershed parameters (`0.05 <= h_rel <= 0.80`, `0.20 <= min_distance_rel <= 2.50`, `0.01 <= min_area_ratio <= 0.45`); validate `bytes` / `Path` image decoding cleanly without raw tracebacks; normalize RGBA/grayscale/CMYK via `.convert("RGB")`. |
| V6 Cryptography | Yes (integrity hash only) | Use stdlib `hashlib.sha256` exclusively for deterministic `PDIConfig.config_hash()` (non-secret integrity fingerprint — never hand-rolled hashing). |

### Known Threat Patterns for Classical Python PDI / CLI Stack

| Pattern | STRIDE | Standard Mitigation |
|---|---|---|
| **Decompression bomb / malformed EXIF upload** (oversized JPEG/PNG or crafted Xiaomi tag `39321`) | Denial of Service | Keep Pillow `Image.MAX_IMAGE_PIXELS` default active; retain `MAX_XIAOMI_TAG_BYTES = 16_384` and strict finite-range parsing in `io_exif.py` (`[VERIFIED: backend/app/pdi/io_exif.py:26-28, 67-103]`). |
| **Visual data leakage from SF-400 LCD (`PROIBIÇÃO 2`)** | Tampering / Scientific Integrity | Dedicated `lcd_suppression.py` before closing/selection + pixel-local `bgr_work` feature extraction strictly at `bird_mask == 255` + 3-px eroded interior texture + automated `0.0`-diff perturbation test. |
| **Path traversal / symlink escape during batch feature extraction** | Information Disclosure / Tampering | Canonicalize `dataset_root.resolve()` and reject any manifest `rel_path` containing `..`, symlinks, or resolving outside `dataset_root` before opening. |
| **Raw dataset path leakage in git-tracked `models/pdi_calibration_audit.json` (`PROIBIÇÃO 4`)** | Information Disclosure / Hygiene | Run the `_FORBIDDEN_LEAK_SUBSTRINGS = (".jpg", "rel_path", "/home/", "data/raw")` assertion before writing JSON and in unit tests (`[VERIFIED: backend/app/ingestion/audit_summary.py:50, 315-326]`). |
| **Worker RAM exhaustion / OOM on 7.1 GiB host** | Denial of Service (local) | Hard-cap `ProcessPoolExecutor(max_workers=4)`, set `cv2.setNumThreads(1)` per worker via `initializer`, and use aspect-preserving `im.draft("RGB", ...)` (`1.4 MB` decoded buffer instead of `22.5 MB`). |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| — | *(Empty — every discrete value, version, latency, payload size, fallback rate, dual-lens ratio, and Watershed count in this report was empirically verified via read-only tools or read verbatim from repo files in this session.)* | — | — |

## Open Questions (RESOLVED)

1. **`768 px` vs `960 px` working resolution & Bilateral filter parameters — RESOLVED:**
   - Spike 1 proved `working_long_edge = 768` + aspect-preserving `draft_scale_multiplier = 1.0` + `bilateral_d = 5, sigmaColor = 50.0, sigmaSpace = 50.0` runs end-to-end (including 4-stage `640 px` X-Ray Base64 encode) in **`p50 = 74.0 ms, p95 = 90.2 ms`** with max payload **`231.7 KB`** (`< 250 KB`), whereas `960 px` (`draft_mult = 2.0`) edges payload to `253.3 KB`. Locked to **`768 px`**, `d = 5`.
2. **Winning dual-lens optical normalization formula (`D-09`) — RESOLVED:**
   - Spike 2 proved across all `127` mixed-lens clean sessions (`175` same-scene pairs, `76` bucket-only pairs) that **`area_optical_norm = area_px / (zoom_multiple ** 2)`** collapses Wide/UW ratio to **`1.007`** (bucket) / **`0.900`** (same-scene) with lowest `med|log(W/UW)|` on all 4 scopes, while `area_px / ((focal_length_mm * zoom_multiple) ** 2)` over-corrects by `6.6×–8.5×`. Retain `area_frac_optical_norm = area_frac / (zoom_multiple ** 2)` alongside it for Phase 3 ablation.
3. **White-on-white `>1,000 g` bucket segmentation & `< 2%` fallback gate — RESOLVED:**
   - Spike 3 proved border-connected smooth-wall peeling gated by a Black-Hat contact-shadow barrier achieves **`0 / 377 (0.00%)`** fallback on Scenario B (including all `59` `>1,000 g` bucket broilers) and **`13 / 1,738 (0.75%)`** overall across all `1,738` clean images in `42.9 s` (`< 2%` fallback & `< 60 s` batch gates both comfortably beaten).
4. **Watershed defaults & latency across all `13` multi-bird images — RESOLVED:**
   - Spike 4 proved general relative defaults `h_rel = 0.20, min_distance_rel = 0.85, min_area_ratio = 0.10` achieve **`9 / 13` exact counts (`12 / 13` within ±1, `417`: 4/4, `1520`: 3/4, `823`: 2/4 exact + 2× off-by-1)** with zero folder hardcoding, and bbox-cropped `h_maxima` cuts peak-detection latency `22×` (`313.8 ms → 14.0 ms`, `bit-identical = True`).
5. **Offline parallelism (`ProcessPoolExecutor` vs `joblib.Parallel`) — RESOLVED:**
   - `ProcessPoolExecutor(max_workers=4, initializer=cv2.setNumThreads, initargs=(1,))` completed all `1,738` images in `42.9 s` with minimal RSS footprint and zero extra serialization layer.

## Sources

### Primary (HIGH confidence)
- **Empirical Read-Only Prototypes (2026-11-22):** Executed via `uv run python` against `data/processed/manifest_clean.parquet` (`1,738` rows), `data/processed/manifest_quarantine.parquet` (`15` rows / `13` multi-bird), and `data/raw/` (read-only, zero filenames logged):
  - Spike 1: `768 px` vs `960 px` latency/payload & aspect-preserving `PIL.Image.draft()` verification (`p50=74.0 ms, p95=90.2 ms, b64 max=231.7 KB`).
  - Spike 2: Dual-lens optical calibration audit across `127` mixed-lens sessions / `175` same-scene pairs / `76` bucket pairs (`area_px / z²` med `W/UW = 1.007`).
  - Spike 3: Full `1,738`-image 3-scenario segmentation & SF-400 LCD exclusion sweep (`0.75%` fallback, `42.9 s` on `4` workers).
  - Spike 4: All `13` quarantined multi-bird Watershed sweep (`9/13` exact, `12/13` ±1) + bbox-cropped `h_maxima` benchmark (`313.8 ms → 14.0 ms`).
- **In-Repo Ground Truth Files (Read Verbatim):**
  - `.planning/phases/02-classical-pdi-engine-optical-calibration-watershed/02-CONTEXT.md` (`D-01`..`D-12`)
  - `.planning/REQUIREMENTS.md` (`PDI-01`..`PDI-05`)
  - `backend/app/config.py:1-30`, `backend/app/pdi/io_exif.py:1-156`, `backend/app/ingestion/manifest.py:116-143, 208-216`, `backend/app/ingestion/audit_summary.py:50, 315-326`, `backend/tests/test_io_exif.py:58-71, 258-309`, `pyproject.toml:1-49`, `.planning/config.json:24, 48-50`
  - `docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md` (§4–§8)
  - `.planning/research/ARCHITECTURE.md` & `.planning/research/PITFALLS.md`

### Secondary (MEDIUM confidence)
- Installed library runtime introspection via `inspect.signature` (`skimage.feature.peak_local_max`, `skimage.morphology.h_maxima`, `skimage.segmentation.watershed`, `skimage.measure.regionprops` in `scikit-image 0.26.0`) and `gsd-tools.cjs query package-legitimacy check --ecosystem pypi`.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all 8 packages already installed, hash-locked in `uv.lock`, and verified via live `uv run python` introspection (`39` Phase 1 tests green).
- Architecture & Schemas: HIGH — frozen dataclasses, 60-column Parquet schema (`11` join/audit + `49` strictly mask-gated numeric features), `models/pdi_calibration_audit.json`, and single-source `backend/app/pdi/` boundaries align 1:1 with Phases 1, 3, and 4.
- Classical PDI & Optical Calibration: HIGH — every threshold, physics formula, fallback rate (`0.75%`), latency (`p95 = 90.2 ms`), payload (`max = 231.7 KB`), and Watershed count (`9/13` exact, `12/13` ±1) was empirically validated on the real dataset.

**Research date:** 2026-11-22
**Valid until:** 2026-12-22 (30 days — locked Python 3.12 / OpenCV 4.14 / scikit-image 0.26 workspace)
