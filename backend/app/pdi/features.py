"""Frozen 49-feature strictly mask-gated PDI descriptor extractor (D-08, D-09, PROIBIÇÃO 2).

Enforces exact mathematical outside-mask perturbation invariance (``0.0`` max
absolute difference across all 49 features whenever pixels at ``bird_mask == 0``
are modified):
- Chromatic Lab and HSV descriptors are computed strictly from pixel-local
  ``cv2.cvtColor(bgr_work, ...)`` sliced at ``bird_mask == 255`` (NEVER from
  tile-coupled ``bgr_clahe`` or spatial ``bgr_smooth``).
- OpenCV ``uint8`` color scaling: CIE Lab ``L, a, b in [0, 255]`` with neutral
  origin ``a = b = 128``; HSV ``H in [0, 179]`` (2 deg per step), ``S, V in
  [0, 255]``.
- Circular Hue statistics (``hsv_h_sin_mean``, ``hsv_h_cos_mean``) are strictly
  gated to saturated pixels (``S >= 25``, returning ``0.0`` when ``< 10``
  pixels qualify).
- Interior feather texture (``feather_grad_mean``, ``feather_laplacian_var``)
  is evaluated strictly inside a 3-px eroded interior mask
  (``cv2.erode(bird_mask, 7x7 ellipse, borderValue=0) > 0``) over mask-zeroed
  ``L``, guaranteeing zero ``3x3`` derivative stencil bleed across the silhouette
  border.
- Optical calibration applies the empirically verified Spike 2 physical winner
  ``area_optical_norm = area_px / (zoom_multiple ** 2)`` alongside
  resize-agnostic ``area_frac_optical_norm = area_frac / (zoom_multiple ** 2)``,
  permanently avoiding ``focal_length_mm * zoom_multiple`` double-counting.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from skimage.measure import perimeter as skimage_perimeter

from backend.app.pdi.io_exif import OpticalContext
from backend.app.pdi.segmentation import EllipseTuple, fit_ellipse_safe

# Immutable 49-column frozen feature schema (D-08) — one-way contract for Phases 3 & 4.
MORPHOMETRIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "area_px",
    "perimeter_px",
    "major_axis_px",
    "minor_axis_px",
    "convex_area_px",
    "area_frac",
    "aspect_ratio_ellipse",
    "eccentricity",
    "solidity",
    "circularity",
    "extent_rotrect",
    "log_hu_1",
    "log_hu_2",
    "log_hu_3",
)

CHROMATIC_TEXTURE_FEATURE_COLUMNS: tuple[str, ...] = (
    "lab_l_mean",
    "lab_l_std",
    "lab_l_p10",
    "lab_l_p50",
    "lab_l_p90",
    "lab_a_mean",
    "lab_a_std",
    "lab_a_p10",
    "lab_a_p50",
    "lab_a_p90",
    "lab_b_mean",
    "lab_b_std",
    "lab_b_p10",
    "lab_b_p50",
    "lab_b_p90",
    "hsv_s_mean",
    "hsv_s_std",
    "hsv_v_mean",
    "hsv_v_std",
    "hsv_h_sin_mean",
    "hsv_h_cos_mean",
    "yellow_down_ratio",
    "white_plumage_ratio",
    "pink_skin_ratio",
    "feather_grad_mean",
    "feather_laplacian_var",
)

OPTICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "focal_length_mm",
    "zoom_multiple",
    "exif_optical_present",
    "area_optical_norm",
    "sqrt_area_optical_norm",
    "perim_optical_norm",
    "major_axis_optical_norm",
    "minor_axis_optical_norm",
    "area_frac_optical_norm",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    MORPHOMETRIC_FEATURE_COLUMNS + CHROMATIC_TEXTURE_FEATURE_COLUMNS + OPTICAL_FEATURE_COLUMNS
)

BANNED_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {"oriented_w", "oriented_h", "megapixels", "bbox_w", "bbox_h", "raw_w", "raw_h"}
)

_MIN_SATURATED_HUE_PIXELS: int = 10
_HUE_SATURATION_THRESHOLD: float = 25.0


def _signed_log_hu(moment: float) -> float:
    """Compute signed log10 Hu moment ``-sign(h) * log10(|h| + 1e-12)``."""
    if not math.isfinite(moment) or moment == 0.0:
        return 0.0
    return float(-math.copysign(1.0, moment) * math.log10(abs(moment) + 1e-12))


_DIAMOND_OFFSETS_2D: np.ndarray = np.array(
    [[-0.5, 0.0], [0.5, 0.0], [0.0, -0.5], [0.0, 0.5]],
    dtype=np.float64,
)
_BINS_256_F64: np.ndarray = np.arange(256, dtype=np.float64)


def _channel_summary_u8(values_u8: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return exact ``(mean, std, p10, p50, p90)`` for 1D ``uint8`` pixels via histogram."""
    n = values_u8.size
    counts = np.bincount(values_u8, minlength=256)
    n_f = float(n)
    mean_v = float(np.dot(counts, _BINS_256_F64)) / n_f
    var_v = float(np.dot(counts, (_BINS_256_F64 - mean_v) ** 2)) / n_f
    std_v = float(math.sqrt(max(0.0, var_v)))

    cdf = np.cumsum(counts)
    quantiles: list[float] = []
    for q in (0.10, 0.50, 0.90):
        pos = q * float(n - 1)
        lo_i = int(math.floor(pos))
        hi_i = int(math.ceil(pos))
        frac = pos - lo_i
        v_lo = float(np.searchsorted(cdf, lo_i + 1))
        v_hi = float(np.searchsorted(cdf, hi_i + 1))
        quantiles.append(v_lo + frac * (v_hi - v_lo))
    return mean_v, std_v, quantiles[0], quantiles[1], quantiles[2]


def _mean_std_u8(values_u8: np.ndarray) -> tuple[float, float]:
    """Return exact ``(mean, std)`` for 1D ``uint8`` pixels via histogram."""
    n = values_u8.size
    counts = np.bincount(values_u8, minlength=256)
    n_f = float(n)
    mean_v = float(np.dot(counts, _BINS_256_F64)) / n_f
    var_v = float(np.dot(counts, (_BINS_256_F64 - mean_v) ** 2)) / n_f
    return mean_v, float(math.sqrt(max(0.0, var_v)))


def _area_convex_exact(crop_u8: np.ndarray, area_work: float) -> float:
    """Compute exact 2D convex hull pixel count matching ``skimage.measure.regionprops``."""
    bh, bw = crop_u8.shape
    if bh <= 1 or bw <= 1:
        return float(bh * bw)

    contours, _ = cv2.findContours(crop_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return area_work

    pts_xy = np.vstack([c.reshape(-1, 2) for c in contours])
    pts_rc = pts_xy[:, ::-1].astype(np.float64)
    if len(pts_rc) >= 3:
        try:
            hull0 = ConvexHull(pts_rc)
            pts_rc = hull0.points[hull0.vertices]
        except QhullError:
            pass

    shifted = (pts_rc[:, None, :] + _DIAMOND_OFFSETS_2D[None, :, :]).reshape(-1, 2)
    try:
        hull = ConvexHull(shifted)
    except QhullError:
        return area_work

    eq = hull.equations  # a*r + b*c + d <= 0
    rows = np.arange(bh, dtype=np.float64)[:, None]
    a = eq[None, :, 0]
    b = eq[None, :, 1]
    d = eq[None, :, 2]
    tol = 1e-10
    rhs = -(a * rows + d) + tol
    pos = b > 1e-12
    neg = b < -1e-12
    zero = (~pos) & (~neg)

    safe_b = np.where(zero, 1.0, b)
    bound = rhs / safe_b
    c_max = np.min(np.where(pos, bound, float(bw - 1)), axis=1)
    c_min = np.max(np.where(neg, bound, 0.0), axis=1)
    row_ok = np.all(np.where(zero, a * rows + d <= tol, True), axis=1)

    lo = np.maximum(0, np.ceil(c_min).astype(np.int64))
    hi = np.minimum(bw - 1, np.floor(c_max).astype(np.int64))
    counts = np.where(row_ok & (hi >= lo), hi - lo + 1, 0)
    return float(np.sum(counts))


def _validate_inputs(
    bgr_work: np.ndarray,
    bird_mask: np.ndarray,
    scale_factor: float,
    optical: OpticalContext,
) -> float:
    if not isinstance(bgr_work, np.ndarray):
        raise ValueError(f"bgr_work must be a numpy ndarray, got {type(bgr_work).__name__}")
    if bgr_work.ndim != 3 or bgr_work.shape[2] != 3:
        raise ValueError(f"bgr_work must have shape (H, W, 3), got {bgr_work.shape!r}")
    if bgr_work.shape[0] == 0 or bgr_work.shape[1] == 0:
        raise ValueError(f"bgr_work must be non-empty, got shape {bgr_work.shape!r}")
    if bgr_work.dtype != np.uint8:
        raise ValueError(f"bgr_work must have dtype uint8, got {bgr_work.dtype!r}")

    if not isinstance(bird_mask, np.ndarray):
        raise ValueError(f"bird_mask must be a numpy ndarray, got {type(bird_mask).__name__}")
    if bird_mask.ndim != 2 or bird_mask.shape != bgr_work.shape[:2]:
        raise ValueError(
            f"bird_mask must have shape {bgr_work.shape[:2]!r}, got {bird_mask.shape!r}"
        )

    if isinstance(scale_factor, bool) or not isinstance(scale_factor, (int, float)):
        raise ValueError(f"scale_factor must be a finite positive number, got {scale_factor!r}")
    sf = float(scale_factor)
    if not math.isfinite(sf) or sf <= 0.0:
        raise ValueError(f"scale_factor must be finite and > 0, got {scale_factor!r}")

    if not math.isfinite(optical.focal_length_mm) or optical.focal_length_mm <= 0.0:
        raise ValueError(
            f"optical.focal_length_mm must be finite and > 0, got {optical.focal_length_mm!r}"
        )
    if not math.isfinite(optical.zoom_multiple) or optical.zoom_multiple <= 0.0:
        raise ValueError(
            f"optical.zoom_multiple must be finite and > 0, got {optical.zoom_multiple!r}"
        )
    return sf


def _finalize_ordered_features(raw: dict[str, float]) -> dict[str, float]:
    ordered: dict[str, float] = {}
    for key in FEATURE_COLUMNS:
        val = float(raw[key])
        if not math.isfinite(val):
            raise ValueError(f"Non-finite feature value computed for {key!r}: {val!r}")
        ordered[key] = val
    return ordered


def extract_mask_gated_features(
    bgr_work: np.ndarray,
    bird_mask: np.ndarray,
    scale_factor: float,
    optical: OpticalContext,
    *,
    contour: np.ndarray | None = None,
    ellipse: EllipseTuple | None = None,
) -> dict[str, float]:
    """Extract all 49 strictly mask-gated morphometric, chromatic, texture & optical features."""
    sf = _validate_inputs(bgr_work, bird_mask, scale_factor, optical)

    focal_mm = float(optical.focal_length_mm)
    zoom = float(optical.zoom_multiple)
    exif_present = 1.0 if optical.exif_optical_present else 0.0

    if bird_mask.dtype == np.uint8:
        _, mask_u8 = cv2.threshold(bird_mask, 0, 255, cv2.THRESH_BINARY)
    else:
        mask_u8 = np.where(bird_mask > 0, 255, 0).astype(np.uint8)

    bx, by, bw, bh = cv2.boundingRect(mask_u8)
    if bw == 0 or bh == 0:
        zeroed = dict.fromkeys(FEATURE_COLUMNS, 0.0)
        zeroed["focal_length_mm"] = focal_mm
        zeroed["zoom_multiple"] = zoom
        zeroed["exif_optical_present"] = exif_present
        return _finalize_ordered_features(zeroed)

    work_h, work_w = bgr_work.shape[:2]
    crop_u8 = mask_u8[by : by + bh, bx : bx + bw]
    crop_bool = crop_u8 == 255

    # 1. Rotation-invariant morphometrics (14 columns).
    area_work = float(np.count_nonzero(crop_bool))
    perimeter_work = float(skimage_perimeter(crop_bool))
    area_convex_work = _area_convex_exact(crop_u8, area_work)

    moments = cv2.moments(crop_u8)
    m00 = float(moments["m00"])
    if m00 > 0.0:
        u20 = float(moments["mu20"]) / m00
        u02 = float(moments["mu02"]) / m00
        u11 = float(moments["mu11"]) / m00
        common = math.sqrt(4.0 * u11 * u11 + (u20 - u02) * (u20 - u02))
        lam1 = 0.5 * (u20 + u02 + common)
        lam2 = max(0.0, 0.5 * (u20 + u02 - common))
        eccentricity_raw = math.sqrt(max(0.0, 1.0 - lam2 / lam1)) if lam1 > 0.0 else 0.0
    else:
        eccentricity_raw = 0.0
    eccentricity = (
        min(1.0, max(0.0, eccentricity_raw)) if math.isfinite(eccentricity_raw) else 0.0
    )
    solidity = float(area_work / max(area_convex_work, 1e-6))

    active_contour = contour
    if active_contour is None or len(active_contour) == 0:
        contours, _ = cv2.findContours(crop_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        active_contour = max(contours, key=lambda c: (float(cv2.contourArea(c)), len(c)))

    active_ellipse = ellipse
    if active_ellipse is None:
        active_ellipse, _ = fit_ellipse_safe(active_contour)

    (_, _), (d1, d2), _ = active_ellipse
    major_work = float(max(d1, d2))
    minor_work = float(min(d1, d2))

    (_, _), (rw, rh), _ = cv2.minAreaRect(active_contour)
    extent_rotrect = float(area_work / max(float(rw) * float(rh), 1e-6))

    sf_sq = sf * sf
    area_px = area_work / sf_sq
    perimeter_px = perimeter_work / sf
    major_axis_px = major_work / sf
    minor_axis_px = minor_work / sf
    convex_area_px = area_convex_work / sf_sq
    area_frac = area_work / float(work_h * work_w)
    aspect_ratio_ellipse = major_axis_px / max(minor_axis_px, 1e-6)
    circularity = min(
        1.5,
        max(0.0, 4.0 * math.pi * area_work / max(perimeter_work * perimeter_work, 1e-6)),
    )

    hu_vals = cv2.HuMoments(moments).ravel()
    log_hu_1 = _signed_log_hu(float(hu_vals[0]))
    log_hu_2 = _signed_log_hu(float(hu_vals[1]))
    log_hu_3 = _signed_log_hu(float(hu_vals[2]))

    # 2. Strictly pixel-local chromatic plumage & eroded interior texture (26 columns).
    # CRITICAL (PROIBIÇÃO 2 / D-08): Convert strictly bgr_work (NEVER tile-coupled bgr_clahe
    # or spatial bgr_smooth) on the tight mask bounding box so outside-mask pixels have zero
    # influence across the mask border.
    bgr_crop = bgr_work[by : by + bh, bx : bx + bw]
    lab_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2Lab)
    hsv_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)

    l_u8 = lab_crop[:, :, 0][crop_bool]
    a_u8 = lab_crop[:, :, 1][crop_bool]
    b_u8 = lab_crop[:, :, 2][crop_bool]
    h_u8 = hsv_crop[:, :, 0][crop_bool]
    s_u8 = hsv_crop[:, :, 1][crop_bool]
    v_u8 = hsv_crop[:, :, 2][crop_bool]

    lab_l_mean, lab_l_std, lab_l_p10, lab_l_p50, lab_l_p90 = _channel_summary_u8(l_u8)
    lab_a_mean, lab_a_std, lab_a_p10, lab_a_p50, lab_a_p90 = _channel_summary_u8(a_u8)
    lab_b_mean, lab_b_std, lab_b_p10, lab_b_p50, lab_b_p90 = _channel_summary_u8(b_u8)

    hsv_s_mean, hsv_s_std = _mean_std_u8(s_u8)
    hsv_v_mean, hsv_v_std = _mean_std_u8(v_u8)

    sat_mask = s_u8 >= int(_HUE_SATURATION_THRESHOLD)
    if int(np.count_nonzero(sat_mask)) >= _MIN_SATURATED_HUE_PIXELS:
        angles = h_u8[sat_mask].astype(np.float64) * (2.0 * math.pi / 180.0)
        hsv_h_sin_mean = float(np.mean(np.sin(angles), dtype=np.float64))
        hsv_h_cos_mean = float(np.mean(np.cos(angles), dtype=np.float64))
    else:
        hsv_h_sin_mean = 0.0
        hsv_h_cos_mean = 0.0

    n_px_f = float(l_u8.size)
    yellow_down_ratio = (
        float(np.count_nonzero((b_u8 > 142) & (l_u8 > 120) & (a_u8 < 133))) / n_px_f
    )
    da_i32 = a_u8.astype(np.int32) - 128
    db_i32 = b_u8.astype(np.int32) - 128
    white_plumage_ratio = (
        float(np.count_nonzero((l_u8 > 165) & ((da_i32 * da_i32 + db_i32 * db_i32) < 256)))
        / n_px_f
    )
    pink_skin_ratio = (
        float(np.count_nonzero((a_u8 > 136) & (l_u8 > 90) & sat_mask)) / n_px_f
    )

    # Eroded interior feather micro-texture (3-px safety border for 3x3 Scharr/Laplacian stencils).
    # Pad tight crop by 4 zero pixels so borderType=BORDER_CONSTANT(0) stays bit-identical
    # even when bird_mask touches the image perimeter.
    pad_mask = np.pad(crop_u8, 4, mode="constant", constant_values=0)
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded = cv2.erode(pad_mask, k_erode, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    interior = eroded > 0
    if np.any(interior):
        l_crop_masked = np.where(crop_bool, lab_crop[:, :, 0], 0).astype(np.uint8)
        l_pad = np.pad(l_crop_masked, 4, mode="constant", constant_values=0)
        gx = cv2.Scharr(l_pad, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(l_pad, cv2.CV_32F, 0, 1)
        grad_mag_interior = np.hypot(
            gx[interior].astype(np.float64),
            gy[interior].astype(np.float64),
        )
        feather_grad_mean = float(np.mean(grad_mag_interior, dtype=np.float64))
        lap = cv2.Laplacian(l_pad, cv2.CV_32F)
        feather_laplacian_var = float(np.var(lap[interior].astype(np.float64), dtype=np.float64))
    else:
        feather_grad_mean = 0.0
        feather_laplacian_var = 0.0

    # 3. EXIF optical & empirically verified calibrated descriptors (9 columns, D-09).
    zoom_sq = zoom * zoom
    area_optical_norm = area_px / zoom_sq
    sqrt_area_optical_norm = math.sqrt(max(0.0, area_optical_norm))
    perim_optical_norm = perimeter_px / zoom
    major_axis_optical_norm = major_axis_px / zoom
    minor_axis_optical_norm = minor_axis_px / zoom
    area_frac_optical_norm = area_frac / zoom_sq

    raw_features: dict[str, float] = {
        # Morphometrics (14)
        "area_px": area_px,
        "perimeter_px": perimeter_px,
        "major_axis_px": major_axis_px,
        "minor_axis_px": minor_axis_px,
        "convex_area_px": convex_area_px,
        "area_frac": area_frac,
        "aspect_ratio_ellipse": aspect_ratio_ellipse,
        "eccentricity": eccentricity,
        "solidity": solidity,
        "circularity": circularity,
        "extent_rotrect": extent_rotrect,
        "log_hu_1": log_hu_1,
        "log_hu_2": log_hu_2,
        "log_hu_3": log_hu_3,
        # Chromatic & Eroded Interior Texture (26)
        "lab_l_mean": lab_l_mean,
        "lab_l_std": lab_l_std,
        "lab_l_p10": lab_l_p10,
        "lab_l_p50": lab_l_p50,
        "lab_l_p90": lab_l_p90,
        "lab_a_mean": lab_a_mean,
        "lab_a_std": lab_a_std,
        "lab_a_p10": lab_a_p10,
        "lab_a_p50": lab_a_p50,
        "lab_a_p90": lab_a_p90,
        "lab_b_mean": lab_b_mean,
        "lab_b_std": lab_b_std,
        "lab_b_p10": lab_b_p10,
        "lab_b_p50": lab_b_p50,
        "lab_b_p90": lab_b_p90,
        "hsv_s_mean": hsv_s_mean,
        "hsv_s_std": hsv_s_std,
        "hsv_v_mean": hsv_v_mean,
        "hsv_v_std": hsv_v_std,
        "hsv_h_sin_mean": hsv_h_sin_mean,
        "hsv_h_cos_mean": hsv_h_cos_mean,
        "yellow_down_ratio": yellow_down_ratio,
        "white_plumage_ratio": white_plumage_ratio,
        "pink_skin_ratio": pink_skin_ratio,
        "feather_grad_mean": feather_grad_mean,
        "feather_laplacian_var": feather_laplacian_var,
        # Optical & Calibrated (9)
        "focal_length_mm": focal_mm,
        "zoom_multiple": zoom,
        "exif_optical_present": exif_present,
        "area_optical_norm": area_optical_norm,
        "sqrt_area_optical_norm": sqrt_area_optical_norm,
        "perim_optical_norm": perim_optical_norm,
        "major_axis_optical_norm": major_axis_optical_norm,
        "minor_axis_optical_norm": minor_axis_optical_norm,
        "area_frac_optical_norm": area_frac_optical_norm,
    }
    return _finalize_ordered_features(raw_features)


__all__ = [
    "MORPHOMETRIC_FEATURE_COLUMNS",
    "CHROMATIC_TEXTURE_FEATURE_COLUMNS",
    "OPTICAL_FEATURE_COLUMNS",
    "FEATURE_COLUMNS",
    "BANNED_FEATURE_COLUMNS",
    "extract_mask_gated_features",
]
