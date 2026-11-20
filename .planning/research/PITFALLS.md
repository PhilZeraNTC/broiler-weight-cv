# Pitfalls Research

**Domain:** Classical Digital Image Processing (PDI — No Deep Learning) + EXIF-Calibrated Tabular Regression for Non-Invasive Broiler Live Weight Estimation (*AvianScale CV* on *Broiler weights of Cobb Avian*)
**Researched:** 2026-11-20
**Confidence:** HIGH (grounded in full forensic dataset audit of all 1,753 images, local host verification, and official OpenCV / scikit-image / scikit-learn / Pillow documentation)

---

## Critical Pitfalls

### Pitfall 1: SF-400 Scale Platter Merging & LCD Display Visual Leakage (CRITICAL — `PROIBIÇÃO 2`)

**What goes wrong:**
Classical segmentation accidentally merges the bright white **SF-400 digital scale body/platter** (~14.5 cm circular white plastic dish + rectangular housing) or the **dark rectangular LCD bezel** (displaying exact ground-truth weight digits + `ON/OFF`, `MODE`, `TARE` buttons) into the bird foreground mask — OR feature extraction leaks non-masked background/LCD pixels into tabular descriptors (e.g., computing color/texture stats on a bounding-box crop `img[y:y+h, x:x+w]` instead of strictly `img[mask == 255]`). On Scenario A images (`116`, `171/173`, `173`, `190`, `233`, `537`, `679`, `1009`, `1340_`, `2093`), a white-plumaged bird standing directly on the white SF-400 platter fuses into a single giant connected component, inflating apparent projected area by `1.5×–3×` and encoding the constant physical platter diameter as a spurious scale proxy.

**Why it happens:**
1. Naive luminance/Otsu thresholding classifies *both* white broiler plumage and white SF-400 plastic as "bright foreground against dark/warm straw bedding (`R > G > B`)".
2. Aggressive morphological closing (`cv2.MORPH_CLOSE`) or `ndimage.binary_fill_holes()` bridges the narrow contact shadow between the bird's belly/legs and the white platter, or swallows the dark LCD window inside the foreground convex hull.
3. Developers inadvertently compute global frame statistics (mean luminance, dominant background color) or rectangular crop histograms that indirectly expose the scale body or dark LCD screen.

**How to avoid:**
1. **Explicit SF-400 LCD & Housing Suppression BEFORE Morphological Closing:**
   - Detect candidate LCD display regions using classical shape + photometry: dark low-luminance (`L*` low), near-achromatic (`|a* - 128| < 12`, `|b* - 128| < 12` in OpenCV Lab) rectangular blobs with high extent (`area / bbox_area > 0.72`), aspect ratio `1.6–3.8`, and sharp Sobel/Canny rectangular edges, often bordered by ultra-smooth, bright, neutral-white plastic housing. Hard-zero (`mask = 0`) any detected LCD/bezel polygon and its immediate margin before closing or component selection.
2. **Plumage vs. Smooth White Plastic Discrimination:**
   - Distinguish smooth specular SF-400 plastic (low local Laplacian/Sobel micro-texture energy + cool/neutral near-zero Lab chroma) from live bird tissue (feathers exhibit directional gradient micro-texture + warm cream/yellowish `b*` undertone in down/feathers + pinkish `a*` skin/comb/feet).
   - Apply elliptical morphological opening (`cv2.MORPH_OPEN`) sized proportionally to working resolution (e.g., `5×5` to `7×7` at `640–768 px`) to sever thin leg/foot bridges connecting the bird silhouette to the scale platter before picking the dominant centrally weighted connected component.
3. **Strict Masked-Pixel Feature Contract:**
   - Enforce in code that **every** chromatic and texture descriptor indexes exclusively `lab[mask == 255]` and `hsv[mask == 255]`. Never pass bounding-box crops, unmasked background ratios, or raw frame statistics into the regressor.
   - Visually inspect Stage 3 (clean binary mask) and Stage 4 (overlay with contour + ellipse) on a fixed Scenario A audit gallery (`116`, `233`, `679`, `1009`, `1340_`, `2093`).

**Warning signs:**
- Segmented mask area exceeds `55–65%` of the frame on Scenario A (scale) images.
- Mask contours show straight 90° geometric corners or flat edges (scale housing / LCD bezel).
- Cross-validation MAE on Scenario A (oblique angles on SF-400) is suspiciously *better* than Scenario B (controlled top-down bucket), signaling platter/scale-ratio leakage.
- Binary mask visibly contains the rectangular dark cutout or circular rim of the SF-400 in X-ray Stage 3.

**Phase to address:**
Phase 2 (Classical PDI Engine — REQ-04, REQ-05, REQ-06), audited quantitatively in Phase 3 (scenario-stratified metrics — REQ-09) and visibly proved in Phase 4 X-ray view (REQ-12).

---

### Pitfall 2: White-on-White Segmentation Collapse in the Bucket Scene (~351 Clean Images, Especially `>1,000 g`)

**What goes wrong:**
In Scenario B (dorsal top-down shots inside the white plastic bucket, `~20.1%` of the dataset), young yellow chicks (`116–460 g`) segment easily via Lab `b*` yellowness. However, heavier broilers (`1,009–2,093 g`, ~24–29 days old) have dense, mature **white plumage** inside a **white plastic bucket** — creating near-zero luminance (`L*`) contrast between figure and background. A naive global Otsu threshold on grayscale/`L*` either:
- **Collapses to near-empty noise** (capturing only the pink comb, dark eye, beak, or a stray shadow sliver — triggering `cv2.error: (-201) There should be at least 5 points to fit the ellipse` in `cv2.fitEllipse()`), **OR**
- **Spills across the entire white bucket interior**, returning `75–95%` of the frame as "bird" and completely destroying area/perimeter features precisely in the rarest, highest-value weight stratum (`>1,000 g`).

**Why it happens:**
1. Mature Cobb Avian 48 feathers match the white plastic wall reflectance in grayscale/luminance, while the deep bucket creates uneven radial tunnel shading (darker bottom corners, brighter upper rim) plus occasional straw specks, droppings, and handwritten pen marks (`B`) on the bucket wall.
2. Relying on HSV `Hue` fails catastrophically on white-on-white scenes because **Hue is mathematically undefined and hyper-noisy whenever Saturation `S < 20`** — tiny sensor noise makes white feathers and white plastic jump randomly across all 180 hue bins.
3. Applying a single straw-bedding segmentation rule uniformly to both Scenario A/C (brown straw) and Scenario B (white bucket) guarantees failure on at least one of them.

**How to avoid:**
1. **Explicit Classical Scene-Type Branching (Bucket vs. Bedding/Scale):**
   - Detect Scenario B deterministically via border-ring photometry: if the outer perimeter ring (`outer 12–15%` border pixels) is predominantly bright (`L*` high in OpenCV Lab) and achromatic (low chroma $\sqrt{(a^*-128)^2 + (b^*-128)^2}$ and low HSV `S`), activate the **bucket-aware segmentation pathway** (and expose `scene_is_bucket` as a clean interpretable metadata flag for Phase 3 stratified evaluation).
2. **Multi-Cue Foreground Fusion (Never Luminance-Only):**
   - **Chroma & Biological Warmth (`Lab a*, b*` + `HSV S`):** Capture yellow down (`b* > 136`), pinkish comb/wattles/exposed skin (`a* > 134`), and warm cream plumage vs. neutral/cool-white plastic. Gate Hue completely whenever `S < 25`.
   - **Feather Micro-Texture & Edge Energy:** Run `CLAHE` on `L*` (`clipLimit=2.0–3.0, tileGridSize=(8,8)`), then compute local gradient magnitude (Scharr/Sobel) and local standard deviation (`blur(I^2) - blur(I)^2`). Feathered bodies exhibit rich mid-frequency texture; smooth plastic bucket walls stay flat.
   - **Contact-Shadow Rim Detection:** Heavy birds sitting in the bucket cast a soft dark contour shadow where body meets plastic floor/walls. Extract this perimeter cue via morphological Black-Hat or adaptive Gaussian thresholding on CLAHE `L*` to form an edge barrier preventing foreground spill onto bucket walls.
   - **Border-Connected Spill Removal & Centrality Scoring:** Remove or heavily penalize connected components touching `≥ 2` image edges (using `skimage.segmentation.clear_border` on non-central spill regions or scoring components by `area * gaussian_center_prior(centroid) * (1 - border_touch_fraction)`). Because the bird is always roughly centered in the bucket frame, this cleanly rejects bright corner walls.
3. **Degenerate-Mask Guard & Deterministic Fallback:**
   - Guard `cv2.fitEllipse(cnt)` with `len(cnt) >= 5` AND non-collinear area check (`cv2.contourArea(cnt) > min_area`), falling back to `cv2.minAreaRect(cnt)` if needed.
   - If the primary mask area falls outside `[0.015, 0.72]` of frame area, trigger a deterministic secondary fallback (e.g., adaptive texture + central shadow-enclosed contour) and emit `segmentation_warning: true` in metadata so failures are logged and inspectable rather than silent crashes.

**Warning signs:**
- Empty masks or `mask_area_frac < 0.015` on heavy bucket shots (`1216–2093 g`).
- `mask_area_frac > 0.75` (mask flooding into rounded bucket corners).
- `cv2.error: (-201:Incorrect size of input array) There should be at least 5 points to fit the ellipse`.
- Phase 3 stratified metrics show `>1000 g` bucket MAPE `> 2×` worse than `≤500 g` bucket MAPE.

**Phase to address:**
Phase 2 (Classical PDI Engine — REQ-03, REQ-04, REQ-05), verified via Phase 3 scenario × weight-stratum breakdown (REQ-09) and showcased live via the "caso difícil branco-no-balde" pitch preset in Phase 4 (REQ-13).

---

### Pitfall 3: Dual-Lens Scale Shift (`6.6×`), Double-Counting `focal × zoom`, and Lens×Weight Confounding Shortcut

**What goes wrong:**
Three compounding optical traps affect this dataset:
1. **Severe Scale Discontinuity:** The Xiaomi Redmi 10 alternates between Wide (`4.25 mm`, `4080×1840`, 74.5%) and Ultrawide (`1.65 mm`, `3264×1472`, 25.5%) — often within 2 seconds on the same bird (`114 / 168` folders mix both lenses!) — plus 67 digital zoom shots (`0.60×–2.00×` in Xiaomi EXIF tag `39321`). Projected pixel area shifts by up to `6.6×` at equal camera distance, and the 120° Ultrawide introduces barrel distortion.
2. **Physics Double-Counting Trap in Naive Optical Normalization:** On the Redmi 10, Wide has `focal_length_mm = 4.25`, `zoomMultiple = 1.00` (50 MP 1/2.76" sensor ≈ 26 mm eq.), while Ultrawide has `focal_length_mm = 1.65`, `zoomMultiple = 0.60` (8 MP 1/4" sensor ≈ 15.6 mm eq.). Notice that `zoomMultiple = 0.60` **already encodes the 35mm-equivalent optical switch (`15.6 / 26 = 0.60`)**! If you naively compute $A_{px} / (f_{mm} \cdot \text{zoom})^2$, the denominator becomes $(4.25 \times 1.0)^2 = 18.06$ for Wide vs. $(1.65 \times 0.60)^2 = 0.98$ for Ultrawide — an artificial **`18.4×` over-correction** (double-counting the lens switch AND ignoring the different sensor pixel sizes between `4080×1840` and `3264×1472`)!
3. **Lens×Weight Confounding Shortcut:** Above `1,000 g`, Ultrawide usage jumps from `~16–32%` to **`54.6%` (`1001–1500 g`), `56.1%` (`1501–2000 g`), and `77.8%` (`2093 g`)** because operators switched to `0.6×` to fit larger birds in frame. If raw `focal_length_mm`, `is_ultrawide`, or native resolution (`oriented_w`, `oriented_h`) is passed to LightGBM/Random Forest, trees exploit `focal == 1.65 mm` as a statistical shortcut for "heavy bird" rather than learning true physical size!

**Why it happens:**
- Developers treat physical focal length (`mm`) as if both lenses shared the same physical sensor dimensions and pixel pitch.
- Developers pass raw EXIF tags (`focal_length_mm`, `oriented_w`, `oriented_h`) alongside area into GBDTs, letting greedy splits latch onto the strong `ultrawide ↔ weight > 1000 g` correlation in the training distribution.
- Handheld camera-to-bird height $Z$ is unrecorded and varies frame-to-frame, so optical normalization removes lens/zoom scale jump, NOT handheld height noise.

**How to avoid:**
1. **Empirical Optical Calibration on Paired Same-Session Shots (Before Finalizing Formula):**
   - Use **fractional frame area** `area_frac = mask_area_proc / (proc_w * proc_h)` as the resolution-agnostic base (eliminates both `4080×1840` vs. `3264×1472` pixel-count differences AND any downscale factor discrepancy!).
   - Derive and **empirically verify** the optical normalization constant directly on the **114 paired same-session dual-lens folders** (same bird photographed seconds apart on Wide `1.0×` vs. Ultrawide `0.6×`, e.g., `1340_`, `1243`, `173`, `2093`):
     - Compare median ratio $\text{area\_frac}_{\text{wide}} / \text{area\_frac}_{\text{uw}}$ within same-session pairs against:
       - `area_frac / (zoom_multiple ** 2)` (physically consistent with 35mm-equivalent magnification: predicts $1 / 0.60^2 = 2.78\times$), vs.
       - `area_px / (focal_length_mm * zoom_multiple) ** 2` (check whether it over-corrects by `~6×`!).
     - Keep whichever physical formulation collapses within-session Wide/Ultrawide area discrepancy closest to `1.0` — and document this empirical calibration proof in the ablation report!
2. **Block Resolution Leakage & Test for Confounding Explicitly:**
   - **NEVER** include `oriented_w`, `oriented_h`, or raw `megapixels` as model features (they are 1:1 proxies for lens ID).
   - Run a **4-condition ablation** in Phase 3:
     1. *Uncalibrated PDI only* (no EXIF)
     2. *Physics-Normalized PDI* (`area_norm`, `perim_norm`, `axes_norm` + scale-invariant plumage/shape features — **NO raw `focal_length_mm` categorical shortcut**)
     3. *Normalized PDI + raw `focal_length_mm` / `zoom_multiple`*
     4. **Counterfactual Lens-Flip Sanity Check & Within-Session Consistency Metric:** Measure mean `|ŷ_wide - ŷ_uw|` across same-session pairs, AND test flipping `focal_length_mm` on fixed features. If raw `focal_length_mm` adds only illusory gain by memorizing `ultrawide ⇒ heavy` (failing the lens-flip check or hurting `<500 g` ultrawide MAE), drop raw lens ID and keep strictly physics-normalized features + scale-invariant plumage features (yellow/white ratio, Lab stats)!
3. **Lean on Scale-Invariant Biological Proxies:**
   - Because handheld height $Z$ is uncalibrated, scale-invariant morphological/chromatic features — **yellow-down vs. white-plumage ratio**, **Lab `a*` (comb/skin pinkness)**, **Lab `b*` quantiles**, **solidity**, **eccentricity**, and **circularity** — provide essential orthogonal age/weight signal that is 100% immune to camera distance and lens choice.

**Warning signs:**
- `focal_length_mm` or `is_ultrawide` appears as #1 feature importance in LightGBM/RF ahead of normalized area and plumage color.
- Same bird photographed 2 seconds apart (Wide vs. Ultrawide) gets predicted weights differing by `> 25%` (e.g., `850 g` vs `1,450 g`).
- On `<500 g` chicks photographed with Ultrawide (`96` imgs in `100–200 g`, `90` imgs in `201–300 g`), the model systematically overpredicts by `+200–400 g`.
- `area_norm` for Ultrawide shots is `5×–7×` larger than Wide shots of the exact same bird (proving `focal * zoom` double-counting!).

**Phase to address:**
Phase 2 (paired dual-lens empirical normalization check — REQ-05) + Phase 3 (ablation study, lens-stratified MAE/MBE, within-session consistency test — REQ-09).

---

### Pitfall 4: Classical Watershed Over-Segmentation & Heterogeneous-Background Failure on Multi-Bird Scenes (`417`, `757`, `823`, `1520`)

**What goes wrong:**
Applying standard `distance_transform_edt` → `peak_local_max` → `skimage.segmentation.watershed` on the 4 quarantined multi-bird folders (`13` images total) either:
- **Over-segments** a single broiler into 3–6 rainbow shards (splitting head, outstretched wing, tail, and torso because feathered contours are irregular and textured), **OR**
- **Under-segments** huddled chicks (`823`, 4 chicks) into 1–2 merged blobs, **OR**
- **Segments the red plastic feeder rim** in `417` as extra "birds" because `417` has a completely different background (red plastic feeder + granular feed) than `757`, `823`, and `1520` (wood-shaving straw bedding).

**Why it happens:**
1. Non-convex bird contours (extended necks, wings, feather ruffles) create multiple secondary ridges in the Euclidean Distance Transform (EDT); raw `peak_local_max` places a seed marker on every shallow bump.
2. Hardcoded pixel `min_distance` (e.g., `min_distance=40`) breaks across bird sizes (`200 g` chicks in `823`/`417` vs. `750 g` birds in `1520`) AND across lenses (`417` is 75% ultrawide, `823` is 50% ultrawide, `1520` is 75% wide).
3. In `scikit-image`, `peak_local_max` no longer supports the legacy `indices=False` argument — returning an `(N, 2)` coordinate array that crashes marker-mask construction if `coords` is empty.

**How to avoid:**
1. **Multi-Background Foreground Pre-Masking:**
   - Explicitly suppress high-chroma red plastic feeder pixels (`Lab a* > 152` or red HSV wraparound with high `S`) BEFORE segmenting bird pixels — cleanly isolating the 2 yellow chicks in `417`.
   - Suppress straw/litter background via combined luminance/chroma/texture cues; fill internal feather holes (`scipy.ndimage.binary_fill_holes`) and apply gentle morphological closing + opening so ragged feather edges do not inject artificial valleys into the distance transform.
2. **Scale-Adaptive Seed Extraction (`Gaussian` + `h_maxima` + Relative `min_distance`):**
   - Compute `dist = ndi.distance_transform_edt(fg_mask)`.
   - Smooth slightly (`dist_smooth = ndi.gaussian_filter(dist, sigma=max(1.5, 0.06 * dist.max()))`) to suppress feather-boundary jitter.
   - Apply **`skimage.morphology.h_maxima(dist_smooth, h=0.25 * dist_smooth.max())`** (prominence scaled relative to peak distance, NOT a fixed pixel value!) to collapse shallow head/wing bumps while keeping distinct torso peaks.
   - Extract markers with `peak_local_max(dist_smooth, min_distance=max(12, int(0.65 * median_peak_dist)), labels=fg_mask)` (or label connected components of `h_maxima` directly!), guarding `if len(coords) == 0`.
   - Run `skimage.segmentation.watershed(-dist_smooth, markers, mask=fg_mask)` and drop tiny spurious basins (`basin_area < 0.08 * total_fg_area` or `< 0.20 * median_basin_area`).
3. **Ground-Truth Count Verification Across All 13 Multi-Bird Images:**
   - Test deterministically against known audited counts: `417` (2 birds × 4 imgs), `757` (2 birds × 1 img), `823` (4 birds × 4 imgs), `1520` (2 birds × 4 imgs). Tune adaptive relative parameters (never folder-specific hardcoded cheats!) and surface per-bird area + count overlay clearly for the pitch demo.

**Warning signs:**
- Watershed outputs 6–12 colored basins on a 2-bird image (`417`, `757`, `1520`).
- Watershed outputs only 1–2 basins on `823` (4 chicks).
- Red feeder rim in `417` is labeled as foreground bird #3.
- `TypeError: peak_local_max() got an unexpected keyword argument 'indices'` or `IndexError` when `coords` is empty.

**Phase to address:**
Phase 2 (Watershed pipeline — REQ-07) + Phase 4 (interactive Watershed demo tab & visual verification — REQ-14).

---

### Pitfall 5: `< 500 ms` Latency Budget Violation on 7.5 MP (`4080×1840`) Inputs & Slow Offline Iteration

**What goes wrong:**
Running classical PDI operators (`ImageOps.exif_transpose`, `cv2.bilateralFilter`, `CLAHE`, multi-space Lab/HSV morphology, `distance_transform_edt` / `watershed`) at native `4080×1840` (7.51 MP) resolution AND encoding 4 X-ray visualization stages as lossless PNG + Base64 in JSON takes **1.5–4.5 seconds per request** on CPU — violating the hard `< 500 ms` latency budget (REQ-11) by 3×–9× and making the live pitch UI feel sluggish. Offline, running single-threaded full-res feature extraction across `1,738` images burns `30–60+ minutes` per experiment, threatening the 8-day MVP deadline.

**Why it happens:**
1. `cv2.bilateralFilter` scales as $O(W \cdot H \cdot d^2)$: at 7.51 MP with `d=9`, it alone consumes `400–900 ms`.
2. `cv2.imencode(".png", ...)` uses CPU-bound zlib compression: encoding 4 full-res PNGs takes `> 1,200 ms` and generates `~15–25 MB` of Base64 JSON (`+33%` Base64 bloat + massive GC pressure in FastAPI & browser).
3. Loading `models/weight_regressor.joblib` inside the route handler instead of FastAPI `lifespan` adds `50–200 ms` per call.
4. Resizing *without* preserving exact aspect ratio (e.g., squashing `4080×1840` `20:9` into `512×512`) speeds up inference but **destroys bird morphometry** (squashes major/minor axis ratio and circularity by `2.2×`!).

**How to avoid:**
1. **Standardized Early Aspect-Preserving Downscale (`max_side = 640` or `768 px`):**
   - Immediately after reading EXIF and calling `ImageOps.exif_transpose()`, downscale preserving exact aspect ratio via `cv2.resize(..., interpolation=cv2.INTER_AREA)` so `max(H, W) == 640` (or `768`, yielding `~640×288` = `0.18 MP`, a **40× pixel reduction**!).
   - **NEVER** distort aspect ratio! Record `scale_factor = proc_w / oriented_w` and verify bit-exact mathematical consistency between offline feature extraction (Phase 2) and online `/api/predict` (Phase 4) by sharing the exact same Python function.
2. **CPU-Lean Classical Operators:**
   - At `640–768 px`, `cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)` (or `cv2.medianBlur(img, 5)`) executes in `~4–12 ms`, and single-channel `L` CLAHE (`tileGridSize=(8, 8)`) executes in `~2–4 ms`.
   - Avoid `cv2.grabCut` (multi-iteration GMM min-cut takes `200–1,000+ ms`) on the synchronous request path.
3. **JPEG-Encoded X-Ray Inspection Artifacts:**
   - Encode X-ray images (Original, CLAHE, Overlay) at working resolution (`≤ 768 px`) as **JPEG (`IMWRITE_JPEG_QUALITY=82`)** — ~3–5 ms each, ~30–60 KB each! Only the flat binary mask (Stage 3) may optionally use 1-channel PNG (compresses flat 0/255 masks to ~5–12 KB in `< 3 ms`) or JPEG. Total 4-image Base64 JSON stays `< 250 KB` and encodes in `< 15 ms`.
4. **FastAPI `lifespan` Preload + Multi-Core Offline Extraction:**
   - Load `weight_regressor.joblib` once at server startup in FastAPI `lifespan`.
   - Use `concurrent.futures.ProcessPoolExecutor` across the host's **16 CPU cores** for Phase 2 offline extraction over all 1,738 images (full dataset extracts in `~15–25 seconds`, enabling rapid PDI iteration!).

**Warning signs:**
- Single `/api/predict` request exceeds `300 ms` locally on a `4080×1840` JPEG.
- JSON response size from `/api/predict` exceeds `500 KB`–`1 MB`.
- Full-dataset Phase 2 feature extraction script takes `> 2 minutes`.

**Phase to address:**
Phase 2 (early aspect-preserving resize + fast classical ops + multiprocessing — REQ-03, REQ-06) + Phase 4 (JPEG X-ray encoding, `lifespan` preload, p95 `< 400 ms` benchmark gate — REQ-11).

---

### Pitfall 6: Extreme Target Imbalance, Log-Target Retransformation Bias & `StratifiedGroupKFold` Empty-Strata Failure

**What goes wrong:**
Five coupled statistical traps lurk in the weight distribution (`116–2,093 g`):
1. **Severe Skew & Mid-Range Desert:** `70.0%` of images are `≤ 500 g`, `47:1` imbalance across weights (`1` to `47` images per weight), a **415 g desert between `801 g` and `1,216 g`** (only 1 clean bird: `1009 g`, 6 imgs), and only **1 bird `> 2,000 g`** (`2093 g`, 9 imgs).
2. **MSE vs. MAPE Conflict:** Unweighted MSE is dominated by heavy-bird squared errors (wrecking MAPE on `116–250 g` chicks where `40 g` = `25%` error), while unweighted models trained on 70% small chicks pull predictions downward on rare `> 1,000 g` birds.
3. **Log-Target Retransformation Bias (Jensen's Inequality):** Training on `log(y)` naturally aligns with MAPE, but naive back-transformation `np.exp(ŷ_log)` (the default in `TransformedTargetRegressor`) predicts the *geometric mean/median* rather than the *arithmetic mean* ($E[e^Z] > e^{E[Z]}$), introducing systematic negative bias (`Mean Bias Error < 0`), especially on noisy heavy birds.
4. **`StratifiedGroupKFold` Empty-Strata & Fold Imbalance:** Creating too many fine weight bins (e.g., 10 equal-width bins) isolates `801–1,000 g` (`1` clean session: `801 g`) and `2,001–2,500 g` (`1` clean session: `2093 g`), triggering `UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5` and producing folds with uneven or missing `> 1,000 g` validation sessions.
5. **Tree Non-Extrapolation & `joblib` Serialization Failure:** Tree ensembles (RF, LightGBM) cannot predict above the max training label (`1,992 g` in the fold holding out `2093 g`), AND passing `lambda` or `__main__`-defined functions to `TransformedTargetRegressor` crashes `joblib.dump()` / FastAPI `joblib.load()` with `PicklingError` or `AttributeError`.

**How to avoid:**
1. **Session-Level Coarse Stratification Verified Programmatically:**
   - Stratify 5-fold `StratifiedGroupKFold` by `session_id` using the **3 PRD weight strata** (`<500 g`, `500–1000 g`, `>1000 g`) — or 5 quantile bins computed on the **188 unique sessions** so every stratum contains `≥ 25` distinct sessions (the `>1000 g` stratum has `~26` clean sessions → `~5` sessions per fold!).
   - Assert in Phase 1 tests: `0` shared `session_id`s across folds, `0` `StratifiedGroupKFold` warnings, and every fold has `≥ 4` distinct sessions with `weight_g > 1000`.
2. **Principled Log-Target Modeling + Duan's Smearing + Bounded Sample Weighting:**
   - Compare (a) **Allometric Log-Log Linear/Ridge/Huber** ($\log W \approx \beta_0 + \beta_1 \log A_{\text{norm}} + \dots$, physically grounded in $W \propto A^{3/2}$ and capable of smooth tail extrapolation) against (b) **Random Forest** and (c) **LightGBM** in log-target space.
   - Compute training-fold-only `sample_weight` combining inverse session count (`1 / n_imgs_in_session`, preventing 47-photo weight `191` from overpowering 6-photo birds) × moderate inverse-stratum weight (capped at `max_w / min_w ≤ 5.0` to avoid memorizing single outliers like `2093 g`). **NEVER** apply sample weights to validation metrics!
   - Implement **Duan's smearing factor** ($\hat{\sigma} = \frac{1}{n}\sum \exp(e_i)$ fit strictly inside each training fold!) inside an importable module class (e.g., `backend/app/ml/wrappers.py`, NEVER a `lambda` or `__main__` class), and verify empirically via CV whether smearing improves `Mean Bias Error` and `RMSE` without degrading `MAPE`.
3. **All Preprocessing Strictly Inside `sklearn.pipeline.Pipeline`:**
   - Put `SimpleImputer`, `StandardScaler`, feature selection, and log/smearing wrappers inside a single `Pipeline` fit *inside* each CV fold so zero validation statistics leak.
   - Explicitly display the valid model range (`116–2,093 g`, ~7–29 days) and per-stratum expected MAE band in `models/evaluation_report.json` and the UI.

**Warning signs:**
- `UserWarning: The least populated class in y has only X members, which is less than n_splits=5`.
- High fold-to-fold variance on `>1000 g` MAE (e.g., fold 2 has only 1 heavy session).
- Systematic negative `Mean Bias Error` (`-80 g` to `-200 g`) on `>1000 g` validation predictions.
- `PicklingError` on `joblib.dump` or `AttributeError: Can't get attribute ... on <module '__main__'>` when FastAPI loads `models/weight_regressor.joblib`.

**Phase to address:**
Phase 1 (fold assignment & stratum assertions — REQ-02) + Phase 3 (log-target, sample weighting, Duan smearing, importable wrapper serialization — REQ-08, REQ-09, REQ-10).

---

### Pitfall 7: Forensic Ingestion Edge Cases & `cv2.imdecode` vs. `ImageOps.exif_transpose()` Orientation Desync

**What goes wrong:**
Two fatal data-integrity failure modes:
1. **Six Simultaneous Forensic Dataset Quirks:**
   - Depth-2 nested folders (`171/173`, `190/191`, `190/193`): naive 1-level scan drops weight `193` completely (22 imgs) and mislabels 17 images as `171 g`/`190 g`.
   - Non-numeric folder suffix (`1340_`): `int(dir_name)` raises `ValueError`.
   - Filename encoding chaos: `1,635` Arabic-Indic digit filenames (`٠-٩`), `7` `_1` burst-second suffixes, and `21` UTF-8→CP866 mojibake filenames (`547/`, `562/`) whose filesystem `mtime` is **March 2024** (repackaging date) instead of 2022!
   - Broken EXIF `DateTime*` (`????:??:?? ??:??:??` in `100%` of files) + `mtime` offset `+14,401 s`.
   - Uncleaned anomalies: `13` multi-bird images (`417`, `757`, `823`, `1520`), `2` extreme mislabels (`1351`, `1465`), and `367` (6 imgs whose SF-400 LCD reads **`267 g`**).
2. **The Silent `cv2.imdecode` vs. `ImageOps.exif_transpose()` Production Bug:**
   - **`81.86%` (`1,435 / 1,753`)** of dataset images have `EXIF Orientation != 1` (`6`: 58.0%, `3`: 17.1%, `8`: 6.8%).
   - In OpenCV, `cv2.imread(path)` auto-applies EXIF rotation, **but `cv2.imdecode(bytes, cv2.IMREAD_COLOR)` IGNORES EXIF orientation completely!** If offline Phase 2 reads files via Pillow `exif_transpose()` (or `cv2.imread`), while Phase 4 FastAPI `/api/predict` decodes uploaded `UploadFile` bytes via `cv2.imdecode()`, **82% of user-uploaded dataset photos arrive rotated 90°/180°/270° sideways/upside-down** — breaking X-ray display and any non-rotation-invariant heuristic!
   - Even *after* `exif_transpose()`, smartphone accelerometer jitter on top-down shots (`679`, `1340_`, `1599`, `2093`) leaves birds at arbitrary 0°–360° orientations.

**Why it happens:**
OpenCV's EXIF rotation code is implemented inside its file-path decoder (`imread`), not its memory-buffer decoder (`imdecode`). Meanwhile, ingesting raw folders without exact forensic count assertions lets encoding/nesting bugs fail silently (e.g., 1,714 rows instead of 1,738).

**How to avoid:**
1. **One Unified Image + EXIF Loader Across All 4 Phases (`load_oriented_image_and_exif(source: Path | bytes)`):**
   - Always open via `PIL.Image.open(...)` (wrapping `bytes` in `io.BytesIO`).
   - **Extract EXIF BEFORE calling `ImageOps.exif_transpose()`** (because `exif_transpose()` mutates/strips orientation tags and can drop MakerNote/Exif IFDs): read `Orientation` (tag `274`), `FocalLength` (Exif IFD `0x8769` tag `37386`), and Xiaomi `39321` (`0x9999`) JSON `zoomMultiple` with defensive parsing.
   - Apply `ImageOps.exif_transpose(im).convert("RGB")` and convert `RGB → BGR` for OpenCV. **Ban direct calls to `cv2.imread` and `cv2.imdecode` across the entire repo.**
2. **Rotation-Invariant Descriptors Only:**
   - Extract strictly rotation-invariant geometry (`area`, `perimeter`, `major_axis`, `minor_axis`, `eccentricity`, `convex_area`, `solidity`, `circularity`, Hu moments) — **NEVER** raw axis-aligned `bbox_width` / `bbox_height` or hardcoded "top/bottom strip" LCD rules.
3. **Hard Exact-Count Forensic Assertions in Phase 1:**
   - Implement Section 9 of `DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md` verbatim and assert on every run:
     - Raw scan: `1,753` images, `168` leaf folders, `166` raw weights, `1,306` Wide (`4.25 mm`), `447` Ultrawide (`1.65 mm`).
     - Quarantine parquet: exactly `15` images (`417`:4, `757`:1, `823`:4, `1520`:4, `1351`:1, `1465`:1).
     - Clean parquet: exactly `1,738` images, `160` unique weights, `188` sessions, `367` folder mapped to `weight_g == 267` (6 rows), `193` present (22 rows), `1340` present (6 rows), `0` null `capture_dt`, `0` timestamps in year `2024` (proving `547/` and `562/` CP866 decode worked!).

**Warning signs:**
- Clean manifest row count != `1,738`, unique weights != `160`, or sessions != `188`.
- Any `capture_date` falls in `2024` (mojibake fallback hit `mtime` instead of decoded filename).
- Uploading a raw dataset JPEG to `/api/predict` renders rotated 90°/180° in the React X-ray viewer compared to the OS photo viewer.
- Features change when an identical test image is rotated 90° and re-scored.

**Phase to address:**
Phase 1 (forensic ingestor & exact-count test suite — REQ-01) + Phase 2 (unified Pillow loader & rotation-invariant features — REQ-03, REQ-05) + Phase 4 (FastAPI upload parity test — REQ-11).

---

### Pitfall 8: Host Environment Trap (Python 3.14.10 Wheels / NVM Node) & Live Pitch Demo Fragility (Stripped EXIF Uploads)

**What goes wrong:**
1. **Verified Local Environment Trap:** On this development machine (Arch Linux, kernel `7.2.6`), the system `python3` is **`Python 3.14.10`** with no `python3.11`/`3.12`/`3.13` in `/usr/bin`, AND `node`/`npm` are not in default PATH (managed inside `~/.nvm`). Running `pip install opencv-python-headless lightgbm pyarrow scikit-image` on Python 3.14 risks missing `cp314` binary wheels, failing C++/CMake/OpenMP source builds, or hitting NumPy 2.x C-API incompatibilities — wasting hours of an 8-day timeline.
2. **Pitch Demo Fragility — Missing EXIF on Real-World Uploads:** Judges and users during live demos routinely upload photos sent via WhatsApp/Telegram, browser screenshots, or edited PNGs — **all of which strip EXIF completely** (`FocalLength = None`, `zoomMultiple = None`). If `/api/predict` fails on `None` EXIF (HTTP 500 or `NaN` in Ridge/Scaler) OR silently feeds wrong lens assumptions without notifying the user, the pitch breaks live on stage.
3. **Additional Demo Breakers:** RGBA 4-channel PNG screenshots crashing 3-channel OpenCV conversions; relative `cwd` paths breaking preset loading when `uvicorn` starts from repo root vs. `backend/`; CORS mismatch (`localhost:5173` → `localhost:8000`); or showing **in-sample training predictions** on pitch presets instead of honest **out-of-fold validation predictions**.

**Why it happens:**
Developers test exclusively with clean `data/raw/` JPEGs (which always carry Xiaomi EXIF tags) inside their personal interactive shell (where NVM is sourced), overlooking stripped-EXIF uploads, PNG alpha channels, and fresh launch shells.

**How to avoid:**
1. **Pin Python 3.12 via `uv` Immediately (Already Installed at `/usr/bin/uv 0.13.9`):**
   - Bootstrap the backend with `uv venv --python 3.12` and lock `.python-version` (`3.12`) + `pyproject.toml`. Python 3.12 guarantees battle-tested binary wheels (`cp312`) for `opencv-python-headless`, `scikit-image`, `pyarrow`, `lightgbm`, `scikit-learn`, and `pillow` with zero C++ source compiles.
   - Document/script NVM activation (`source ~/.nvm/nvm.sh`) in a single root startup script (`scripts/dev.sh` or `Makefile`) so frontend + backend boot deterministically in one command.
2. **Graceful EXIF Fallback + Interactive Manual Lens/Zoom Override UI:**
   - When EXIF `focal_length_mm` or `zoom_multiple` is missing, NEVER crash: fall back cleanly to default Wide (`focal_length_mm = 4.25`, `zoom_multiple = 1.0`, `exif_detected = false`), return structured metadata indicating fallback mode, display a clear UI warning pill (`"EXIF não detectado — usando padrão Wide 4,25 mm / 1,0×"`), and expose one-click manual Lens/Zoom controls (`Wide 1.0×` / `Ultrawide 0.6×` / custom zoom) that re-run inference immediately.
   - Always normalize uploads via `ImageOps.exif_transpose(im).convert("RGB")` (handles RGBA PNGs, grayscale, WebP cleanly), validate file size (`≤ 20 MB`) and magic bytes, and return clean JSON HTTP 400/422 errors (never uncaught HTTP 500 tracebacks).
3. **Honest Pitch Presets & Working-Directory-Independent Paths:**
   - Resolve all paths relative to `Path(__file__).resolve().parents[...]` (or explicit env vars), never `Path.cwd()`.
   - Curate the Phase 4 pitch presets (`~180 g`, `~550 g`, `~1,600 g`, white-on-white bucket case) using **strictly out-of-fold validation predictions** (or record the exact CV fold model that never saw that `session_id` during training) so live Ground Truth vs. Predicted comparisons are 100% scientifically defensible under academic Q&A.
   - Add a pre-pitch automated smoke test (`pytest` / `curl` script) testing: raw Xiaomi JPEG upload, **EXIF-stripped JPEG upload**, **RGBA PNG upload**, all presets, all 4 Watershed folders (`417`, `757`, `823`, `1520`), and `/api/metrics` latency `< 500 ms`.

**Warning signs:**
- Package install begins compiling `.tar.gz` source tarballs for `pyarrow`, `scipy`, `opencv`, or `lightgbm`.
- `node: command not found` when launching Vite from scripts or subshells.
- Uploading an EXIF-stripped JPEG or RGBA screenshot PNG triggers HTTP 500.
- Pitch presets show suspiciously near-0% error (training-set memorization instead of OOF CV).

**Phase to address:**
Phase 1 (`uv` + Python 3.12 pinned environment & repo hygiene — REQ-01) + Phase 4 (EXIF fallback UI, RGBA safety, OOF presets, CORS, single-command launch + smoke test — REQ-11, REQ-12, REQ-13).

---

## Technical Debt Patterns

Shortcuts that seem tempting under an 8-day sprint deadline, and whether they are acceptable.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| **Hardcoding per-folder or per-preset segmentation thresholds** | Instant pretty masks on 4 pitch presets | Destroys CV honesty, fails on user uploads, fails academic audit | **NEVER** |
| **Using `cv2.imread` offline and `cv2.imdecode` in FastAPI** | 2 fewer lines of Pillow glue code | 81.86% of uploaded dataset photos arrive rotated 90°–270° in live demo | **NEVER** |
| **Fitting `StandardScaler` / `SimpleImputer` / Duan smearing before CV split** | Simpler training script | Preprocessing leakage invalidates 5-fold CV metrics | **NEVER** |
| **Passing raw `oriented_w` / `oriented_h` into GBDT** | Slight artificial drop in CV RMSE | Exploits `3264×1472 → Ultrawide → >1 kg` shortcut; breaks on resized/non-Redmi uploads | **NEVER** |
| **Encoding 4 X-ray artifacts as Base64 JPEG inside `/api/predict` JSON (at `≤ 768 px`)** | Atomic 1-request React UI update, zero temp-file cleanup | `+33%` payload overhead (~180–250 KB JSON vs ~150 KB binary) | **Acceptable in MVP** (adds `< 15 ms` at `640–768 px`; **never** acceptable with full-res PNGs) |
| **Serializing winning model with `joblib` (importable module classes) instead of ONNX** | Zero export friction for `sklearn` + `LightGBM` pipelines | Tied to Python 3.12 + exact library versions in `pyproject.toml` | **Acceptable in MVP** (local single-user pitch app with locked `uv` venv) |
| **Accepting audited `367 → 267 g` fix without running full-base LCD OCR audit** | Saves 1–2 days vs. building an OCR verifier (and avoids violating `PROIBIÇÃO 2` confusion) | Undetected minor typos beyond `367`/`1351`/`1465` remain in aleatoric noise floor (`±3 g`+) | **Acceptable in MVP** (document explicitly in Metrics/Audit panel as known data ceiling) |

---

## Integration Gotchas

Common mistakes when wiring OpenCV, scikit-image, Pillow, scikit-learn/LightGBM, FastAPI, and React together.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| **OpenCV Lab/HSV vs. scikit-image Lab/HSV** | Applying standard CIE Lab ranges (`L ∈ [0,100]`, `a,b ∈ [-128,127]`) or 360° Hue directly to OpenCV `uint8` images | Remember OpenCV `uint8` scaling: **`L ∈ [0, 255]`**, **`a, b ∈ [0, 255]` with neutral = `128`**, and **`H ∈ [0, 179]`** (half-degrees!). Centralize all color-space conversions and document units explicitly in `pdi/color.py`. |
| **Pillow → OpenCV Channel Order** | Passing `np.array(pil_rgb)` straight to `cv2.cvtColor(..., cv2.COLOR_BGR2Lab)` | Yellow chicks turn cyan/blue and `a*`/`b*` invert! Always call `cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)` at the Pillow→OpenCV boundary (or use `COLOR_RGB2Lab` consistently and convert back to BGR only for `cv2.imencode`). |
| **Pillow `ImageOps.exif_transpose()` vs. EXIF Reading** | Calling `exif_transpose()` *before* reading `im.getexif().get_ifd(0x8769)` | `exif_transpose()` resets orientation and can drop vendor tags. Always extract `Orientation` (`274`), `FocalLength` (`37386`), and Xiaomi `39321` (`0x9999`) **first**, then transpose. |
| **Xiaomi Tag `39321` (`0x9999`) Parsing** | Assuming `ifd.get(39321)` is always a valid JSON string | On non-Xiaomi uploads or stripped images, tag `39321` is `None`, `bytes`, or absent. Guard with `isinstance(x_raw, (str, bytes))` + `try/except json.JSONDecodeError` + fallback `zoom_multiple = 1.0`. |
| **scikit-image `peak_local_max` API** | Using removed `indices=False` kwarg or indexing empty `coords` | Current `skimage.feature.peak_local_max` returns `(N, 2)` coordinate array only. Guard `if coords.size > 0: mask[tuple(coords.T)] = True`. |
| **LightGBM / scikit-learn Feature Ordering** | Passing raw positional `np.array([...])` in FastAPI while training used a `pd.DataFrame` | Triggers LightGBM feature-name warnings and risks silent column permutation bugs. Store locked `feature_columns: list[str]` inside the `joblib` bundle dict and build a 1-row `pd.DataFrame([features])[feature_columns]` at inference. |
| **FastAPI + Vite Localhost & NVM Shell** | Launching `npm run dev` in non-interactive shell without NVM, or forgetting FastAPI `CORSMiddleware` for `http://localhost:5173` | Source `~/.nvm/nvm.sh` in launch scripts; configure FastAPI CORS explicitly for `localhost:5173` / `127.0.0.1:5173` (or Vite dev proxy `/api` → `http://127.0.0.1:8000`). |

---

## Performance Traps

Patterns that pass quick unit tests on thumbnails but fail on real `4080×1840` Redmi 10 JPEGs or full-dataset training.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| **Running `cv2.bilateralFilter` & morphology at full `4080×1840` (7.51 MP)** | `/api/predict` takes `1,500–4,000 ms`; CPU spikes to 100% | Downscale early (aspect-preserving, `max_side = 640–768 px`, `INTER_AREA`) right after EXIF transpose | Immediately on **1st real `4080×1840` image** (`> 3×` over 500 ms SLA) |
| **Encoding 4 X-ray stages as PNG Base64** | API response takes `> 1.2 s`; JSON payload `10–25 MB`; React tab freezes on image decode | Encode working-res (`≤ 768 px`) X-ray stages as JPEG (`quality=82`) + compact 1-ch binary mask | Immediately on **1st `/api/predict` call** in browser |
| **Using `cv2.grabCut` synchronously in `/api/predict`** | Inference latency spikes unpredictably to `600–2,500 ms` | Stick to deterministic O(N) Lab/HSV + gradient + morphology + connected components (`< 50 ms` total) | Every complex-background Scenario A/C frame |
| **Single-threaded Phase 2 batch extraction over 1,738 images** | Every segmentation tweak requires `15–45 min` wait before seeing CV impact | Use `ProcessPoolExecutor` across local 16 cores + save features to `data/processed/features.parquet` | After **2–3 PDI tuning iterations** (destroys 8-day schedule) |
| **Reloading `weight_regressor.joblib` per request** | Extra `80–250 ms` latency jitter on every call | Load model artifact once in FastAPI `lifespan` context manager | Every API request + rapid preset switching in pitch |
| **Reading 7.5 MP preset JPEGs & running live Watershed on every tab visit** | Watershed tab hangs for `1–2 s` switching between `417`, `757`, `823`, `1520` | Keep Watershed pipeline `< 150 ms` at `max_side = 768 px` (and optionally cache deterministic preset responses in memory) | During live pitch tab-switching Q&A |

---

## Security Mistakes

Domain-specific scientific integrity, data hygiene, and local web-API safety issues beyond generic OWASP checklists.

| Mistake | Risk | Prevention |
|---------|------|------------|
| **Visual Data Leakage from SF-400 LCD (`PROIBIÇÃO 2`)** | **CRITICAL (Scientific Invalidity):** Model exploits LCD digits or fixed 14.5 cm platter scale; fails academic grading and real-world deployment | Hard-suppress LCD/bezel & platter before closing; index features strictly via `img[mask == 255]`; prove zero LCD inclusion visually in X-ray Stage 3/4 |
| **Session Split Leakage (`PROIBIÇÃO 3`)** | **CRITICAL (Fake Metrics):** Burst photos 1–2 s apart (`24` near-duplicate pairs dHash ≤ 4; ~9 imgs/session) appear in both train & val, inflating $R^2$ artificially | Group strictly by `session_id` (`gap > 600 s`) via 5-fold `StratifiedGroupKFold`; assert `set(train_sessions).isdisjoint(val_sessions)` in automated tests |
| **Arbitrary Path Traversal in `/api/presets/{id}` or `/api/watershed-demo/{id}`** | **MEDIUM:** Passing `../../etc/passwd` or arbitrary local file path if endpoint joins raw query params to filesystem | Resolve presets and watershed samples strictly through a **server-side allowlist dict** mapping opaque IDs (e.g., `"preset_chick_180"`, `"ws_417_0"`) to validated paths |
| **Unbounded Upload / Decompression Bomb (`UploadFile`)** | **MEDIUM:** Uploading a corrupted 50 MB file or decompression bomb hangs FastAPI RAM during live demo | Enforce `MAX_UPLOAD_BYTES = 20 * 1024 * 1024`, keep Pillow `Image.MAX_IMAGE_PIXELS` default guard active, validate decoded image dimensions and format cleanly |
| **Committing `data/raw/`, `data/processed/`, or `models/*.joblib` (`PROIBIÇÃO 4`)** | **HIGH:** Pushing `3.54 GiB` dataset or large binary artifacts corrupts Git history and violates project rules | Verify `.gitignore` excludes `data/raw/`, `data/processed/`, `models/*.joblib`, and `*.jpg`/`*.parquet` before Phase 1 commit |

---

## UX Pitfalls

Common user-experience mistakes that undercut credibility during an academic/product pitch demo.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| **Silent failure or wild prediction on EXIF-stripped uploads (WhatsApp / screenshots)** | Judge uploads a phone screenshot; app either throws 500 or predicts `400 g` on a `1,600 g` bird without warning | Detect missing EXIF gracefully, show a prominent **"EXIF ausente — modo padrão Wide 4,25 mm / 1,0×"** badge, and provide instant **Lens (`Wide` / `Ultrawide`) + Zoom override controls** |
| **Displaying training-set predictions on Pitch Presets** | Judge asks *"Did the model train on this bird?"* — admitting yes undermines the entire validation rigor | Compute preset predictions using the **strict Out-Of-Fold (OOF) model** whose training folds excluded that `session_id`, and badge it `"Validação Out-of-Fold (ave nunca vista no treino)"` |
| **Showing a single bare gram number with no uncertainty or domain range** | Overstates precision on the sparse `801–1,216 g` gap or `> 2,000 g` boundary | Display predicted weight alongside **stratum-specific expected error band (`±MAE`)**, estimated age stage (Cobb Avian 48 curve), and explicit valid operating range (`116–2,093 g`) |
| **X-ray mask stage doesn't visibly highlight LCD exclusion** | Audience can't immediately see *how* `PROIBIÇÃO 2` (anti-leakage) was enforced | Show clean 4-panel progression: (1) EXIF-oriented Original, (2) CLAHE L-channel, (3) Clean Binary Mask (visually black over SF-400 LCD), (4) Final Overlay with bird contour + fitted ellipse + major/minor axes |
| **Simulated Flock Batch Average lacks reset/remove or double-counts repeated clicks** | Clicking a preset twice skews the live simulated flock mean during presentation | Give each prediction an explicit **"Adicionar ao Lote"** action (or deduplicate by image hash) plus clear **"Remover / Limpar Lote"** controls |
| **Hiding failure modes in the Metrics Dashboard** | Experienced reviewers distrust cherry-picked global $R^2$ without subgroup error | Prominently display stratified tables (`<500 g`, `500–1000 g`, `>1000 g`; Wide vs. Ultrawide; Bucket vs. Scale/Litter) AND the **with vs. without EXIF optical ablation** chart |

---

## "Looks Done But Isn't" Checklist

Verify every box before claiming a phase or the pitch MVP is complete:

- [ ] **Forensic Manifest (Phase 1):** Often misses depth-2 folders (`190/193`), `1340_`, or 2024 `mtime` on CP866 files (`547`, `562`) — verify exact counts: `1,753` raw → `1,738` clean + `15` quarantine, `160` clean weights, `188` sessions, `0` dates in 2024, `367 → 267 g`.
- [ ] **Session Fold Integrity (Phase 1):** Often passes `weight_g` bins that trigger `StratifiedGroupKFold` empty-stratum warnings — verify `0` warnings, `0` overlapping `session_id`s across folds, and `≥ 4` sessions with `>1,000 g` in every single fold.
- [ ] **EXIF Parity Offline vs. Upload (Phase 2 & 4):** Often works on disk (`cv2.imread`) and fails on HTTP bytes (`cv2.imdecode`) — verify bit-identical oriented pixels and identical predicted weight (to `0.01 g`) between loading from `Path` and POSTing raw bytes of an `Orientation == 6` image to `/api/predict`.
- [ ] **LCD & Platter Exclusion (Phase 2):** Often looks fine on bucket photos while silently swallowing the SF-400 platter/LCD on scale photos — verify X-ray Stages 3 & 4 visually on `116`, `233`, `679`, `1340_`, `2093`, and confirm zero bounding-box/unmasked features exist in the feature extractor.
- [ ] **White-on-White Heavy Bucket Segmentation (Phase 2):** Often tuned on yellow chicks (`< 500 g`) and collapses on white `> 1,200 g` birds in buckets — verify non-degenerate masks (`0.05 ≤ area_frac ≤ 0.65`) on heavy bucket images (`1216`, `1480`, `1689`, `2093`).
- [ ] **Optical Calibration Anti-Confounding (Phase 2 & 3):** Often double-counts `focal_length_mm * zoom_multiple` or lets LightGBM memorize `ultrawide ⇒ heavy` — verify within-session Wide vs. Ultrawide ratio ≈ `1.0` after normalization AND pass the lens-flip sanity test.
- [ ] **Watershed Multi-Bird Demo (Phase 2 & 4):** Often splits 1 bird into 5 pieces or counts the red feeder in `417` — verify clean basin separation on all 4 audited multi-bird folders (`417`, `757`, `823`, `1520`) without folder-specific hardcoded cheats.
- [ ] **Model Serialization & Cold Boot (Phase 3 & 4):** Often trains in a script (`__main__`) and fails to deserialize inside `uvicorn` — verify `joblib.load("models/weight_regressor.joblib")` succeeds in a clean Python process imported from `backend/` and predicts on a single-row DataFrame.
- [ ] **Latency `< 500 ms` & EXIF-Stripped Upload Safety (Phase 4):** Often benchmarked only on tiny test images with full EXIF — verify p95 end-to-end `/api/predict` `< 400 ms` on real `4080×1840` JPEGs AND HTTP 200 graceful fallback on both EXIF-stripped JPEG and RGBA PNG uploads.

---

## Recovery Strategies

When a pitfall triggers mid-sprint despite prevention, use these targeted recovery paths (calibrated to the 8-day timeline).

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| **White-on-White Bucket Segmentation Fails on Heavy Birds (`> 1,000 g`)** | **MEDIUM (3–5 hrs)** | Isolate bucket frames via outer-ring white detector; switch bucket branch to **local texture energy (Scharr + local std on CLAHE `L*`) + contact-shadow Black-Hat barrier + central flood-fill suppression**; visually validate on 15 heavy bucket frames before re-extracting features (takes `~20 s` on 16 cores). |
| **SF-400 Platter Bridges into Bird Mask on Scenario A** | **MEDIUM (2–4 hrs)** | Add a morphological erosion-then-dominant-central-blob step combined with **minimum texture + warm-chroma (`b*`/`a*`) mask gate** so smooth neutral-white plastic cannot fuse with feathered pixels; hard-mask detected dark rectangular LCD contours before morphological closing. |
| **Optical Normalization (`focal × zoom`) Hurts CV Instead of Helping** | **LOW (1–2 hrs)** | Inspect paired same-session Wide/Ultrawide ratios across the 114 dual-lens folders; replace `area_px / (focal * zoom)^2` with sensor-consistent **`area_frac / (zoom_multiple ** 2)`** (and separately test `area_frac` + calibrated lens scale ratio `k_empirical`); re-run Phase 3 ablation matrix (`~30 s`). |
| **Watershed Over-Segments Multi-Bird Demo Scenes** | **LOW (1–2 hrs)** | Increase `h_maxima` relative prominence (`h = 0.25 → 0.35 * dist_max`), increase Gaussian pre-smoothing `sigma` on distance transform, and enforce post-watershed small-basin merge (`area < 0.15 * median_basin_area` merged into adjacent basin). |
| **FastAPI `/api/predict` Exceeds `500 ms`** | **LOW (1 hr)** | Profile stage timings (`decode`, `pdi`, `model`, `encode`); lower working `max_side` from `768` to `640 px`, replace `bilateralFilter(d=9)` with `medianBlur(5)` or `bilateralFilter(d=5)`, switch X-ray encoding from PNG to JPEG (`quality=80`), and ensure model loads in FastAPI `lifespan`. |
| **Python 3.14 Wheel / Compilation Failure** | **LOW (15 min)** | Delete broken venv immediately; run `uv venv --python 3.12 && uv pip install ...` (leverages already-installed `/usr/bin/uv 0.13.9` and guaranteed `cp312` binary wheels). |
| **GBDT Underpredicts Heavy Birds (`> 1,000 g`) Due to Skew** | **LOW (1–2 hrs)** | Switch target transform to `log(y)` + fold-internal Duan smearing + capped inverse-stratum `sample_weight`, AND include Allometric Log-Log Ridge/Huber in comparison (often beats GBDTs on extrapolation/sparse tails!). |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification Gate |
|---------|------------------|-------------------|
| **#8a. Host Python 3.14.10 Wheel Trap & NVM Node** | **Phase 1** | `uv venv --python 3.12` resolves all backend deps from binary wheels with 0 source builds; `.python-version` committed; `pytest` runs cleanly. |
| **#7a. Forensic Ingestion Edge Cases (nested folders, `1340_`, CP866, Arabic digits, `367→267 g`, quarantine)** | **Phase 1** | Automated assertion suite verifies exact `1,753` raw → `1,738` clean + `15` quarantine, `160` weights, `188` sessions, `0` 2024 timestamps, `193` & `1340` present. |
| **#6a. `StratifiedGroupKFold` Empty-Strata & Session Split Leakage** | **Phase 1** | Automated test verifies `0` sklearn warnings, `0` shared `session_id`s across folds, and `≥ 4` sessions with `weight_g > 1,000` in every fold. |
| **#7b. `cv2.imdecode` vs. `ImageOps.exif_transpose()` Desync & Color-Space Scaling** | **Phase 2** | Shared Pillow loader used everywhere (grep asserts 0 direct `cv2.imread`/`cv2.imdecode`); unit test confirms identical pixels & features from `Path` vs. raw `bytes` on `Orientation=6` JPEG. |
| **#1. SF-400 LCD & Scale Platter Visual Leakage** | **Phase 2** | Visual audit of X-ray Stage 3/4 on Scenario A test suite (`116`, `233`, `679`, `1340_`, `2093`) + code audit confirming all features index strictly `img[mask == 255]`. |
| **#2. White-on-White Bucket Segmentation Collapse (`> 1,000 g`)** | **Phase 2** | Automated mask-sanity gate across all 1,738 clean images: `< 2%` degenerate-mask fallback rate, `0` `cv2.fitEllipse` crashes, clean masks on heavy bucket samples (`1216–2093 g`). |
| **#3a. Dual-Lens Optical Scale Shift & `focal × zoom` Double-Counting** | **Phase 2** | Empirical paired check on 114 dual-lens folders verifies median within-session `area_norm_wide / area_norm_uw ≈ 1.0 ± 0.2`. |
| **#4. Watershed Over-Segmentation on Multi-Bird Scenes (`417`, `757`, `823`, `1520`)** | **Phase 2** | Visual + count verification across all 13 quarantined multi-bird images (`417`:2, `757`:2, `823`:4, `1520`:2) with zero red-feeder false positives. |
| **#5a. Offline & Online PDI Latency Trap** | **Phase 2** | Aspect-preserving `max_side ≤ 768 px` pipeline processes 1 full `4080×1840` image in `< 150 ms` CPU; 16-core batch extraction over 1,738 images completes in `< 60 s`. |
| **#3b. Lens×Weight Confounding Shortcut** | **Phase 3** | 4-way optical ablation table + within-session Wide/UW consistency error + counterfactual lens-flip sanity test documented in `models/evaluation_report.json`. |
| **#6b. Target Imbalance, Log-Target Bias & `joblib` Serialization** | **Phase 3** | Stratified metrics (`<500 g`, `500–1000 g`, `>1000 g`; Wide vs. UW; Bucket vs. Scale) show controlled `Mean Bias Error` & `MAPE`; clean-process `joblib.load()` smoke test passes. |
| **#5b. End-to-End `< 500 ms` HTTP Latency & Base64 Payload Bloat** | **Phase 4** | Benchmark script on 10 real `4080×1840` JPEGs via live FastAPI `/api/predict` achieves `p95 < 400 ms` and JSON payload `< 400 KB`. |
| **#8b. Live Pitch Fragility (Stripped EXIF, RGBA PNGs, OOF Presets, Path/CORS)** | **Phase 4** | Pre-pitch smoke test passes on raw Xiaomi JPEG, EXIF-stripped JPEG, RGBA PNG, manual lens-override toggle, OOF presets, all 4 Watershed scenes, and fresh repo-root launch. |

---

## Sources

- **Forensic Dataset Audit (Primary Ground Truth):** `docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md` (Sections 2–10: exact counts of 1,753 raw / 1,738 clean images, 168 leaf folders, 194 raw / 188 clean sessions, Wide `4.25 mm` vs. Ultrawide `1.65 mm` lens confounding, Xiaomi tag `39321`, CP866 mojibake in `547`/`562`, `367 → 267 g` LCD error, `81.86%` EXIF orientation rate, 4 visual scenarios, slow shutter motion blur).
- **Project Specification & Constraints:** `PRD.md` & `.planning/PROJECT.md` (REQ-01 through REQ-15, `PROIBIÇÃO 1–5`, `< 500 ms` SLA, 2026-11-28 / 2026-11-30 milestone deadlines).
- **Local Host Verification:** Direct non-invasive probe confirming Arch Linux (`7.2.6-arch2-1`), 16 CPU cores, system `Python 3.14.10`, `/usr/bin/uv 0.13.9`, and Node managed under `~/.nvm`.
- **OpenCV & Pillow EXIF Behavior:** Verified OpenCV `cv2.imread` vs. `cv2.imdecode` EXIF orientation discrepancy (GitHub issues & Pillow `ImageOps.exif_transpose` documentation) and `cv2.fitEllipse` `< 5 points` `cv2.error (-201)` edge case.
- **scikit-image Marker-Controlled Watershed:** Official `skimage.morphology.h_maxima`, `skimage.feature.peak_local_max`, and `skimage.segmentation.watershed` over-segmentation mitigation patterns.
- **scikit-learn Validation & Retransformation:** Official `StratifiedGroupKFold` behavior on binned continuous targets, `TransformedTargetRegressor` retransformation bias (Jensen's inequality / Duan's 1983 smearing estimator), and `joblib`/pickle serialization constraints for custom callables.
- **Classical Poultry Computer Vision Literature:** Known failure modes of HSV hue instability at low saturation, wood-shaving/feather `b*` overlap, feather-texture watershed over-segmentation, and 2D posture-induced area variance in broiler weight estimation.
- **Python 3.14 Ecosystem Wheel Status:** PyPI `cp314` binary wheel & NumPy 2.x ABI hazards for `opencv-python-headless`, `lightgbm`, `pyarrow`, and `scikit-image` vs. mature `cp312` wheels via `uv`.

---
*Pitfalls research for: Classical PDI Broiler Weight Estimation (AvianScale CV)*
*Researched: 2026-11-20*
