---
phase: 02-classical-pdi-engine-optical-calibration-watershed
reviewed: 2026-11-23T19:15:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - backend/app/config.py
  - backend/app/pdi/__init__.py
  - backend/app/pdi/config.py
  - backend/app/pdi/io_exif.py
  - backend/app/pdi/preprocess.py
  - backend/app/pdi/lcd_suppression.py
  - backend/app/pdi/segmentation.py
  - backend/app/pdi/features.py
  - backend/app/pdi/xray.py
  - backend/app/pdi/pipeline.py
  - backend/app/pdi/watershed.py
  - backend/scripts/extract_features.py
findings:
  critical: 2
  warning: 7
  info: 3
  total: 12
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-11-23T19:15:00Z
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Reviewed all 12 Phase 2 production source files implementing the classical PDI engine (`backend/app/pdi/`), path registry (`backend/app/config.py`), and offline batch extraction + optical calibration audit CLI (`backend/scripts/extract_features.py`).

Overall architecture, modularity, typing, and Ruff hygiene (`All checks passed!`) are strong. Core classical PDI invariants were independently verified: zero deep learning or OCR imports (`PROIBIÇÃO 1`), zero raw `cv2.imread`/`cv2.imdecode` calls, exact outside-mask pixel-perturbation invariance in `extract_mask_gated_features` (`PROIBIÇÃO 2` on pixel-local `bgr_work`), zero random splits (`PROIBIÇÃO 3`), zero raw dataset filename dumps (`PROIBIÇÃO 4`), zero out-of-scope tracking/density/alerts (`PROIBIÇÃO 5`), exact `_area_convex_exact` parity with `skimage.measure.regionprops`, and accurate `fit_ellipse_safe` + Stage 4 major/minor axis orientation.

However, adversarial inspection and live verification surfaced **2 Critical (Blocker) issues**, **7 Warnings**, and **3 Info items** that should be addressed before locking the Phase 3 feature contract and Phase 4 online serving:

1. **Frozen feature corruption (`CR-01`):** `cv2.moments(crop_u8)` in `features.py:290` omits `binaryImage=True` on `{0, 255}` masks, attenuating Hu moments `1..3` by `255`, `255²`, and `255³` (`16.58M×`) and saturating `55/1,738` (`3.2%`) real dataset silhouettes into the `1e-12` epsilon floor.
2. **ASVS V5 EXIF input-validation bypass (`CR-02`):** `_parse_focal_length()` and `resolve_optical_context()` in `io_exif.py` fail to enforce `[MIN_VALID_FOCAL_LENGTH_MM, MAX_VALID_FOCAL_LENGTH_MM]` (`[0.5, 200.0]`) on untrusted JPEG EXIF `FocalLength` tags (e.g., `999999999/1` or `1/10000000` flows unchecked into `features["focal_length_mm"]`).
3. **Key Warnings (`WR-01`–`WR-07`):** Tautological `lcd_suppression_bucket_false_positives = 0` audit metric (10% real trigger rate when run on bucket scenes); unbounded `1e7` spikes in `extent_rotrect`/`aspect_ratio_ellipse` on thin/tiny/disconnected masks; 8-connectivity vs. 4-connectivity mismatch in `watershed.py` dropping diagonally-connected foreground pixels; `PDIConfig` `int`-vs-`float` hash drift and mismatched Watershed parameter bounds; missing SF-400 LCD suppression in `watershed.py` for arbitrary user uploads; `str(record["rel_path"])` bypassing non-string ASVS V4 checks; and unhandled malformed EXIF parser exceptions.

## Narrative Findings (AI reviewer)

All findings below were independently verified against the source files and reproduced via read-only execution checks (no structural pre-pass was provided).

## Critical Issues

### CR-01: `cv2.moments(crop_u8)` omits `binaryImage=True`, corrupting Hu Moments by `255`–`16.58M×`

**Severity:** BLOCKER / Critical
**File:** `backend/app/pdi/features.py:290` (and `336-339`)
**Issue:** `crop_u8` is a binary silhouette mask encoded with pixel values `{0, 255}` (`uint8`), but line 290 calls `cv2.moments(crop_u8)` with the default `binaryImage=False`. While `eccentricity` cancels the `255` factor in `mu / m00`, `cv2.HuMoments(moments)` (`line 336`) computes scale-normalized central moments `nu_pq = mu_pq / m00^((p+q)/2 + 1)`, which does **not** cancel constant pixel intensity:
- `hu[0]` is scaled down by `255×` (`log_hu_1` shifted by `+2.4065`),
- `hu[1]` is scaled down by `255² = 65,025×` (`log_hu_2` shifted by `+4.8131`),
- `hu[2]` is scaled down by `255³ = 16,581,375×` (`log_hu_3` shifted by `+7.2196`).

Verified on the materialized `data/processed/features_clean.parquet`: `55 / 1,738` (`3.2%`) real broiler silhouettes are compressed into the `+ 1e-12` epsilon floor of `_signed_log_hu` (`|log_hu_3| >= 11.5`, max `11.999242`), destroying shape asymmetry information in a frozen Phase 3 feature schema (`D-08`).

**Fix:** Pass `binaryImage=True` so OpenCV treats non-zero pixels as binary `1`, then re-run `backend/scripts/extract_features.py` to regenerate `data/processed/features_clean.parquet`:
```python
# backend/app/pdi/features.py:290
moments = cv2.moments(crop_u8, binaryImage=True)
```

### CR-02: Untrusted EXIF `FocalLength` bypasses `[0.5, 200.0]` ASVS V5 bounds validation

**Severity:** BLOCKER / Critical
**File:** `backend/app/pdi/io_exif.py:122-133` and `268-281`
**Issue:** While `_parse_xiaomi_zoom()` (`lines 167-170`) bounds untrusted EXIF `zoomMultiple` to `[MIN_VALID_ZOOM_MULTIPLE, MAX_VALID_ZOOM_MULTIPLE]` (`[0.1, 20.0]`) and `resolve_optical_context()` validates both `default_*` and caller `override.*` against `[MIN_VALID_FOCAL_LENGTH_MM, MAX_VALID_FOCAL_LENGTH_MM]` (`[0.5, 200.0]`), `_parse_focal_length()` (`lines 130-132`) checks only `math.isfinite(value) and value > 0.0`. Likewise, `resolve_optical_context()` (`lines 270-278`) trusts `metadata.focal_length_mm` and `metadata.zoom_multiple` directly without range enforcement.

Verified with crafted JPEGs: uploading a JPEG with EXIF tag `37386` (`FocalLength`) set to `999999999/1` (`9.99999999e8 mm`) or `1/10000000` (`1e-7 mm`) passes `read_exif_metadata()`, `load_working_rgb()`, `resolve_optical_context()`, and `run_single_bird_pdi()` untouched, injecting extreme out-of-domain values straight into `features["focal_length_mm"]` (and programmatic `ExifMetadata(..., focal_length_mm=1e200)` passes `resolve_optical_context()` and overflows `focal * focal` to `inf` in calibration metrics). This is an ASVS V5 input-validation bypass on untrusted file metadata.

**Fix:** Enforce the declared physical bounds inside `_parse_focal_length()` and sanitize `metadata` fields inside `resolve_optical_context()`:
```python
# backend/app/pdi/io_exif.py:122-133
def _parse_focal_length(raw: Any) -> float | None:
    """Safely convert an EXIF FocalLength (e.g., IFDRational) to bounded positive mm."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(value) or not (
        MIN_VALID_FOCAL_LENGTH_MM <= value <= MAX_VALID_FOCAL_LENGTH_MM
    ):
        return None
    return value
```
And in `resolve_optical_context()` (`lines 268-281`), treat out-of-range `metadata.focal_length_mm` / `metadata.zoom_multiple` as invalid (`None`) before falling back to `valid_default_*` (and update `exif_optical_present` accordingly).

## Warnings

### WR-01: Tautological `lcd_suppression_bucket_false_positives` audit metric & bypassed LCD suppression on white-surface scale uploads

**Severity:** WARNING
**File:** `backend/app/pdi/segmentation.py:525-529` and `backend/scripts/extract_features.py:361-363`
**Issue:** When `classify_scene_border_ring()` returns `scene_is_bucket = True`, `segment_single_bird()` (`segmentation.py:525-529`) never invokes `suppress_lcd_and_platter()` and hardcodes `lcd_excluded_px = 0, lcd_core_count = 0`. Consequently, `build_calibration_audit()` (`extract_features.py:361-363`) computes:
```python
lcd_suppression_bucket_false_positives = int(np.count_nonzero(lcd_triggered & is_bucket))
```
which is mathematically guaranteed to be `0` for every possible output of `run_single_bird_pdi()`. Empirically running `suppress_lcd_and_platter()` directly on a 60-image evenly spaced sample of the `377` `bucket_dorsal` images actually triggers `lcd_excluded_px > 0` on `6 / 60` (`10.0%`), contradicting both the `"lcd_suppression_bucket_false_positives": 0` claim in `models/pdi_calibration_audit.json` and the comment in `lcd_suppression.py:22`. Furthermore, an arbitrary user upload showing a bird on an SF-400 scale over a bright white table (satisfying the smooth-white border-ring heuristic) will completely bypass `PROIBIÇÃO 2` LCD suppression.

**Fix:** Separate audit measurement from branch gating: either (1) audit `suppress_lcd_and_platter()` honestly on bucket scenes (or rename the metric to `lcd_applied_on_bucket_count` so it is not presented as a detector false-positive measurement), and (2) tighten dark-LCD bezel criteria (e.g., requiring high internal 7-segment digit gradient or sharp rectangular contrast inside the bezel) so dark LCD bezels can be safely suppressed even when the outer border ring is white.

### WR-02: Unbounded `1e7` spikes in `extent_rotrect` and `aspect_ratio_ellipse` on thin, tiny, or disconnected masks

**Severity:** WARNING
**File:** `backend/app/pdi/features.py:307-321` and `330`
**Issue:**
1. `cv2.minAreaRect(active_contour)` computes center-to-center spans `(rw, rh)` rather than inclusive pixel dimensions `(rw + 1, rh + 1)`. For any 1-pixel-thick collinear mask or `1×1` upload, `rw == 0` or `rh == 0`, causing `extent_rotrect = area_work / max(rw * rh, 1e-6)` to explode to `1,000,000.0–20,000,000.0` (reproduced: `1×1` PNG upload produces `extent_rotrect = 1e6`; `1×20` line mask produces `extent_rotrect = 2e7` and `aspect_ratio_ellipse = 1.9e7` via `minor_axis_px = 0.0`). Even a `2×20` bar produces `extent_rotrect = 2.105 > 1.0`.
2. When `bird_mask` contains multiple disconnected contours, `area_work` counts all foreground pixels while `cv2.minAreaRect(active_contour)` bounds only the largest contour, producing `extent_rotrect > 2.0` (reproduced: `2.216` on two `20×20` squares).

**Fix:** Compute `minAreaRect` across all mask contour points (using inclusive `(rw + 1.0) * (rh + 1.0)` and clamping `extent_rotrect` to `[0.0, 1.0]`), and guard degenerate zero-width ellipses cleanly:
```python
# backend/app/pdi/features.py:307-335
contours, _ = cv2.findContours(crop_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
all_pts = np.vstack(contours)
(_, _), (rw, rh), _ = cv2.minAreaRect(all_pts)
rotrect_area = max((float(rw) + 1.0) * (float(rh) + 1.0), 1.0)
extent_rotrect = float(min(1.0, max(0.0, area_work / rotrect_area)))

# Guard 1D degenerate ellipses (minor_work < 1.0 px)
safe_minor_px = max(minor_axis_px, 1.0 / sf)
aspect_ratio_ellipse = major_axis_px / safe_minor_px
```

### WR-03: 8-connectivity vs. 4-connectivity mismatch in `watershed.py` drops diagonally-connected foreground pixels

**Severity:** WARNING
**File:** `backend/app/pdi/watershed.py:545-548` and `573`
**Issue:** Component area filtering (`line 211`), `h_maxima` grouping (`line 260`), and disconnected-component seed rescue (`lines 545-548`) all use **8-connectivity** (`connectivity=8`). However, `skimage_watershed(-dist_s, markers, mask=fg_bool)` (`line 573`) omits `connectivity` and defaults to `connectivity=1` (**4-connectivity** in 2D). Whenever two foreground regions meet only across a diagonal pixel bridge, seed rescue marks the entire 8-connected component as already seeded, while 4-connected Watershed cannot cross the diagonal bridge — silently leaving the unreached sub-region unlabeled (`0`). Verified live: on two `25×25` squares meeting diagonally, `625 / 1,250` foreground pixels are dropped with default `connectivity=1`, compared to `0 / 1,250` with `connectivity=2`.

**Fix:** Align Watershed flooding to 8-connectivity (`connectivity=2` in 2D `skimage.segmentation.watershed`):
```python
# backend/app/pdi/watershed.py:573
ws_labels = skimage_watershed(-dist_s, markers, connectivity=2, mask=fg_bool)
```

### WR-04: `PDIConfig` suffers `int`-vs-`float` hash drift and permits Watershed values rejected by `run_watershed_pdi`

**Severity:** WARNING
**File:** `backend/app/pdi/config.py:101-242` and `backend/app/pdi/watershed.py:49-54, 441-458`
**Issue:**
1. `_validate_finite_float()` accepts `int` or `float` and returns `float(value)`, but `PDIConfig.__post_init__()` discards the return value instead of normalizing float fields. Verified live: `PDIConfig(clahe_clip_limit=2).config_hash()` (`744c37e4752cf453`) != `PDIConfig(clahe_clip_limit=2.0).config_hash()` (`43a3cfaa3208019e`) because `json.dumps` serializes `2` vs `2.0`, breaking SHA-256 fingerprint determinism for identical numeric configs.
2. `PDIConfig` allows `watershed_h_rel` in `(0, 1)`, `watershed_min_distance_rel` in `(0, inf)`, and `watershed_min_area_ratio` in `[0, 1)`, whereas `watershed.py:49-54` restricts them to `[0.05, 0.80]`, `[0.20, 2.50]`, `[0.01, 0.45]`. Verified live: `PDIConfig(watershed_h_rel=0.01, watershed_min_distance_rel=5.0, watershed_min_area_ratio=0.0)` constructs cleanly and then crashes `run_watershed_pdi(config=cfg)` at runtime with `ValueError`.

**Fix:** Normalize every float field via `object.__setattr__(self, name, numeric)` inside `PDIConfig.__post_init__()`, and share the exact Watershed bounds `[0.05, 0.80]`, `[0.20, 2.50]`, `[0.01, 0.45]` as canonical constants imported from `backend/app/pdi/config.py`:
```python
# backend/app/pdi/config.py
WATERSHED_MIN_H_REL = 0.05
WATERSHED_MAX_H_REL = 0.80
WATERSHED_MIN_DISTANCE_REL = 0.20
WATERSHED_MAX_DISTANCE_REL = 2.50
WATERSHED_MIN_AREA_RATIO = 0.01
WATERSHED_MAX_AREA_RATIO = 0.45

# Inside PDIConfig.__post_init__:
object.__setattr__(
    self,
    "clahe_clip_limit",
    _validate_finite_float("clahe_clip_limit", self.clahe_clip_limit, min_value=0.0, min_inclusive=False),
)
# (apply identically to all float fields and use WATERSHED_* bounds)
```

### WR-05: `watershed.py` omits SF-400 LCD/housing suppression for arbitrary user uploads (`PROIBIÇÃO 2` risk)

**Severity:** WARNING
**File:** `backend/app/pdi/watershed.py:477-480` (and `154-237`)
**Issue:** `run_watershed_pdi()` supports arbitrary user uploads (`REQ-07`, `D-11`/`D-12`) and extracts the full 49-key feature vector per separated instance (`lines 617-622`) so Phase 4 can estimate illustrative individual weights. While it suppresses red plastic feeders and thin border walls, it never excludes SF-400 scale LCD bezels or housings. Because `score = 0.55 * z_l + 0.30 * (-z_a) + 0.25 * (-z_g)` scores bright, smooth, achromatic SF-400 plastic housing strongly positive and `ndi.binary_fill_holes` (`line 204`) closes interior holes, uploading a photo containing an SF-400 scale can incorporate scale housing and enclosed LCD pixels into Watershed basins. Verified live on 20 real SF-400 scale images from the clean dataset: `union_mask` overlaps `suppress_lcd_and_platter()`'s exclusion mask in `10 / 20` (`50%`) images and overlaps actual dark LCD bezel/digit pixels in `4 / 20` (`20%`) images.

**Fix:** Suppress verified dark SF-400 LCD bezel cores (and re-zero them after `ndi.binary_fill_holes` alongside `red_excl`) inside `_build_multi_bird_union_mask()`, taking care to gate housing flood suppression carefully and re-verify all 13 quarantined multi-bird images (`417`, `757`, `823`, `1520`) so straw shadows do not clip valid bird silhouettes:
```python
# backend/app/pdi/watershed.py (inside run_watershed_pdi / _build_multi_bird_union_mask)
lcd_res = suppress_lcd_and_platter(frame, cfg)
# At minimum exclude detected dark LCD bezel cores strictly before & after hole fill
combined_excl = cv2.bitwise_or(red_excl, lcd_res.exclusion_mask)
```

### WR-06: `str(record["rel_path"])` bypasses non-string `rel_path` type check in ASVS V4 path containment

**Severity:** WARNING
**File:** `backend/scripts/extract_features.py:500` (and `145`)
**Issue:** `_resolve_safe_image_path()` (`line 145`) explicitly defends against non-string values (`if not isinstance(rel_path, str) or not rel_path.strip(): raise ValueError(...)`). However, `run_extract_features()` (`line 500`) converts `record["rel_path"]` using `str(record["rel_path"])` before passing it in. If a manifest contains `None`, `pd.NA`, or `np.nan` in `rel_path`, `str(...)` silently turns it into `"None"`, `"<NA>"`, or `"nan"`, defeating the `isinstance(..., str)` guard and triggering a filesystem lookup under `canonical_root` (misreporting a schema/containment violation as `FileNotFoundError`, or opening an unintended file with that name).

**Fix:** Pass the raw value directly to `_resolve_safe_image_path()` (or guard `isinstance(raw_rel, str)` prior to calling):
```python
# backend/scripts/extract_features.py:143 & 500
def _resolve_safe_image_path(rel_path: Any, canonical_root: Path) -> Path:
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("ASVS V4 path containment violation: empty or non-string rel_path.")
    ...

# Line 500:
safe_path = _resolve_safe_image_path(record["rel_path"], canonical_root)
```

### WR-07: Missing exception guard around Pillow EXIF/IFD parsing for malformed untrusted uploads

**Severity:** WARNING
**File:** `backend/app/pdi/io_exif.py:174-185`
**Issue:** `_extract_exif_from_open_image()` calls `im.getexif()` and `exif.get_ifd(EXIF_IFD_POINTER)` directly without guarding against malformed EXIF/TIFF structures. While Pillow converts some truncated headers into `UserWarning`, crafted or corrupted EXIF/IFD segments in untrusted web uploads (Phase 4 `/api/predict` and `/api/watershed-demo`) can raise `OSError`, `SyntaxError`, `ValueError`, `TypeError`, `KeyError`, or `ZeroDivisionError` from Pillow's TIFF tag decoder, causing HTTP 500 crashes instead of gracefully falling back to `orientation=1` and `has_exif_optical=False` per `D-02`.

**Fix:** Wrap EXIF/IFD extraction in a narrow exception handler with safe defaults:
```python
# backend/app/pdi/io_exif.py:174-186
orientation = 1
focal_length_mm: float | None = None
zoom_multiple: float | None = None
try:
    exif = im.getexif()
    raw_orient = exif.get(EXIF_TAG_ORIENTATION, 1)
    if isinstance(raw_orient, int) and not isinstance(raw_orient, bool) and 1 <= raw_orient <= 8:
        orientation = int(raw_orient)
    ifd = exif.get_ifd(EXIF_IFD_POINTER)
    focal_length_mm = _parse_focal_length(ifd.get(EXIF_TAG_FOCAL_LENGTH))
    zoom_multiple = _parse_xiaomi_zoom(ifd.get(XIAOMI_TAG_JSON))
except (OSError, ValueError, TypeError, ZeroDivisionError, KeyError, SyntaxError):
    orientation = 1
    focal_length_mm = None
    zoom_multiple = None
```

## Info

### IN-01: Duplicated display resize and Base64 encoding helpers between `xray.py` and `watershed.py`

**Severity:** INFO
**File:** `backend/app/pdi/watershed.py:280-323` and `backend/app/pdi/xray.py:47-68, 137-154, 168-171`
**Issue:** `_resize_for_display`, `_encode_jpeg_b64`, and `_encode_mask_png_b64` are duplicated nearly verbatim across `xray.py` and `watershed.py`. In addition, `xray.py:_resize_for_display` returns a `(resized, scale)` tuple whose `scale` element is discarded (`_`) at all four call sites (`lines 168-171`).
**Fix:** Consolidate display downscaling and JPEG/PNG Base64 encoding into a single shared helper module (or import from `xray.py`) and drop the unused `scale` return value.

### IN-02: Hardcoded pixel dilation kernels and unused `config` parameter in `lcd_suppression.py`

**Severity:** INFO
**File:** `backend/app/pdi/lcd_suppression.py:58, 121, 133`
**Issue:** `suppress_lcd_and_platter(frame, config)` ignores `config` (`_ = config`) and uses fixed pixel kernels `(19, 19)` for `k_bezel` and `(17, 17)` for `k_halo`. Because `load_working_rgb()` does not upscale small user uploads (e.g., `240×180`), fixed pixel dilation radii expand disproportionately on small images and shrink on higher `working_long_edge` settings.
**Fix:** Scale odd morphological kernel sizes proportionally to `max(h, w) / 768.0` (clamped to odd `>= 3`) or bind them to `PDIConfig`.

### IN-03: Practically unreachable second fallback branch in `segment_single_bird` drops `excl_mask`

**Severity:** INFO
**File:** `backend/app/pdi/segmentation.py:544-553`
**Issue:** `_run_adaptive_fallback()` (`lines 490-511`) always draws a non-empty central ellipse as its final guard, making the `if not contours:` branch at `lines 544-553` practically unreachable dead code for any non-empty frame — and if it were ever triggered, passing `np.zeros((h, w), dtype=np.uint8)` at `line 549` would discard `excl_mask`.
**Fix:** Pass `excl_mask` consistently (or replace with a defensive `if not contours: raise RuntimeError(...)` guard) to remove the dead path and preserve `excl_mask` discipline.

---

_Reviewed: 2026-11-23T19:15:00Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_

---

## Post-Review Resolution (`bc2217c`)

All **2 Critical** findings (`CR-01`, `CR-02`) and **6 actionable Warnings** (`WR-02`, `WR-03`, `WR-04`, `WR-05`, `WR-06`, `WR-07`) were fixed immediately in commit `bc2217c`, followed by full corpus re-extraction (`46.53 s`, `log_hu_3 >= 11.5`: `55 → 0`, `extent_rotrect max = 0.958942`, identical `models/pdi_calibration_audit.json` SHA-256), `136/136` passing pytest assertions, and clean `ruff check backend/`. (`WR-01` reflects intentional Spike 3 dorsal-bucket branch gating documented in `02-RESEARCH.md` to preserve `0/377` white-on-white bucket masks; `IN-01`–`IN-03` are non-blocking style/cleanup info items.)
