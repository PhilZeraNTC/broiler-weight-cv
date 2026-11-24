# Phase 2: Classical PDI Engine, Optical Calibration & Watershed - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-11-22
**Phase:** 2-classical-pdi-engine-optical-calibration-watershed
**Mode:** `--auto` (autonomous recommended-default selection)
**Areas discussed:** Canonical Working Resolution & Preprocessing Contract, Anti-LCD Exclusion & 3-Scenario Segmentation, Mask-Gated Feature Contract & Empirical Optical Calibration, Isolated Multi-Bird Marker-Based Watershed

---

## Canonical Working Resolution & Preprocessing Contract (`PDI-01`, `PDI-04`)

| Option | Description | Selected |
|--------|-------------|----------|
| Frozen `PDIConfig` (`working_long_edge=768`, SHA-256 hash, `Image.draft()` after EXIF read, `INTER_AREA`, 640px JPEG Q=82 + 1-ch PNG mask) | Preserves exact `20:9` aspect ratio, guarantees zero training/serving skew, and comfortably fits `< 400 ms` p95 CPU SLA on native `4080×1840` JPEGs (`< 250 KB` X-ray payload) | ✓ |
| `working_long_edge=960` fixed with 4 lossless PNG X-ray stages | Higher working resolution but risks breaking `< 500 ms` latency and produces multi-MB JSON payloads | |
| Native `4080×1840` processing | Violates `< 500 ms` SLA by 3×–9× and risks OOM on `7.1 GiB` RAM | |

**User's choice:** `[auto]` Selected recommended default: frozen `PDIConfig` with `working_long_edge=768` (benchmarkable vs. `960 px` in Phase 2 research), SHA-256 hash lock, extended Pillow `io_exif.py`, and compact `< 250 KB` on-demand 4-stage X-Ray encoding.
**Notes:** `[auto] Canonical Working Resolution & Preprocessing — Q: "Which working resolution, config lock, and X-Ray encoding contract should Phase 2 adopt?" → Selected: "Frozen PDIConfig (working_long_edge=768, SHA-256 hash, 640px JPEG Q=82 + 1-ch PNG binary mask)" (recommended default).`

---

## Anti-LCD Exclusion & 3-Scenario Classical Segmentation (`PDI-02`)

| Option | Description | Selected |
|--------|-------------|----------|
| Dedicated `lcd_suppression.py` (rotation-invariant pre-closing exclusion) + outer-ring `scene_is_bucket` branching + multi-cue Lab/HSV/texture/Black-Hat fusion + `< 2%` fallback guard | Enforces `PROIBIÇÃO 2` audibly in code, handles white-on-white `>1,000 g` dorsal buckets without `fitEllipse` crashes, and keeps `scene_type` clean for Phase 3 stratified metrics | ✓ |
| Single global Otsu/HSV threshold across all 3 scenarios | Collapses on white-on-white `>1,000 g` bucket birds and merges white SF-400 platter with white broilers | |
| Fixed bottom/top strip crop to hide LCD | Breaks whenever `EXIF Orientation` or handheld camera rotation places the SF-400 scale on a different side | |

**User's choice:** `[auto]` Selected recommended default: dedicated rotation-invariant `lcd_suppression.py`, deterministic outer-ring `bucket_dorsal` vs. `scale_or_litter` branching, `S < 25` Hue gating, feather micro-texture + Black-Hat contact-shadow fusion, and `< 2%` degenerate-mask fallback gate.
**Notes:** `[auto] Anti-LCD & 3-Scenario Segmentation — Q: "How should SF-400 LCD exclusion, white-on-white bucket segmentation, and degenerate-mask fallbacks be architected?" → Selected: "Dedicated rotation-invariant lcd_suppression.py + outer-ring bucket branching + multi-cue fusion + <2% fallback guard" (recommended default).`

---

## Mask-Gated Feature Contract & Empirical Dual-Lens Optical Calibration (`PDI-03`)

| Option | Description | Selected |
|--------|-------------|----------|
| Strict `bird_mask == 255` morphometric/chromatic/texture features + empirical paired check on `114` mixed-lens sessions (`area_frac / zoom**2` vs `A_px / (f*zoom)**2`) + `models/pdi_calibration_audit.json` + outside-mask perturbation invariance test + `n_jobs=4` Parquet export | Eliminates `focal × zoom` double-counting risk, bans raw resolution shortcuts (`oriented_w`/`oriented_h`), proves `PROIBIÇÃO 2` mathematically, and stays inside `7.1 GiB` RAM (`< 60 s`) | ✓ |
| Blindly hardcode `A_px / (f_mm * zoom)**2` without paired session verification | Risks ~`18.4×` over-correction between Redmi 10 Wide (`4.25 mm / 1.0×`) and Ultrawide (`1.65 mm / 0.60×`) | |
| Include raw `oriented_w`/`oriented_h` and whole-image background stats | Leaks lens ID (`ultrawide ⇒ >1,000 g`) and violates `PROIBIÇÃO 2` | |

**User's choice:** `[auto]` Selected recommended default: strict mask-gated features, empirical paired dual-lens calibration check across the `114` mixed-lens sessions logged to `models/pdi_calibration_audit.json`, automated outside-mask pixel-perturbation invariance test, and `n_jobs=4` (`cv2.setNumThreads(1)`) batch extraction to `data/processed/features_clean.parquet`.
**Notes:** `[auto] Mask-Gated Features & Optical Calibration — Q: "How should optical normalization and the mask-gated feature schema be validated and materialized?" → Selected: "Strict bird_mask==255 features + empirical 114-session paired check + outside-mask perturbation test + n_jobs=4 Parquet" (recommended default).`

---

## Isolated Multi-Bird Marker-Based Watershed (`PDI-05`)

| Option | Description | Selected |
|--------|-------------|----------|
| Isolated `watershed.py` (red-feeder suppression + EDT + Gaussian + scale-adaptive `h_maxima`/`peak_local_max` + `skimage.segmentation.watershed`) returning counts, areas, 4 visual stages, AND per-instance mask-gated features | Solves `417` red feeder + over-segmentation across `417`/`757`/`823`/`1520` (`2`/`2`/`4`/`2` birds) and enables illustrative individual bird weights + scene mean in Phase 4 `WEB-04` (`WARNING 2`) | ✓ |
| Return counts and pixel areas only (no per-instance features) | Would require re-implementing feature extraction in Phase 4 to satisfy resolved `WARNING 2` (`WEB-04` illustrative weights) | |
| Fixed pixel `min_distance` + raw `peak_local_max` | Over-segments single birds into 4–6 shards and breaks across Wide vs. Ultrawide and `200 g` vs. `750 g` flocks | |

**User's choice:** `[auto]` Selected recommended default: isolated `backend/app/pdi/watershed.py` with red-feeder chromatic pre-suppression, scale-adaptive `h_maxima`/`peak_local_max` + `skimage.segmentation.watershed`, 4 visual stages, and per-instance mask-gated feature extraction via shared `extract_mask_gated_features()`.
**Notes:** `[auto] Multi-Bird Watershed — Q: "What algorithm contract and per-instance outputs should watershed.py expose?" → Selected: "Isolated scale-adaptive EDT + h_maxima + skimage.watershed with red-feeder suppression and per-instance mask-gated features" (recommended default).`

---

## the agent's Discretion

- Exact Lab/HSV/Scharr/Black-Hat threshold tuning and morphological kernel sizes derived empirically on real dataset samples during Phase 2 research (no folder/preset hardcoding).
- Internal helper decomposition inside `backend/app/pdi/`.
- Visual annotation palette for X-Ray Stage 4 and Watershed overlays.
- `ProcessPoolExecutor(max_workers=4)` vs. `joblib.Parallel(n_jobs=4)` for `extract_features.py`.

## Deferred Ideas

- Bucket-rim Hough `mm/pixel` geometric scale calibration (`HARD-02` — v1.x).
- Motion-blur sharpness quality gate (`HARD-03` — v1.x).
