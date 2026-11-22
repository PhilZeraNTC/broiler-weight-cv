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
from skimage.measure import regionprops

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


def _channel_summary(values: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return ``(mean, std, p10, p50, p90)`` as Python floats."""
    mean_v = float(np.mean(values, dtype=np.float64))
    std_v = float(np.std(values, dtype=np.float64))
    p10, p50, p90 = np.percentile(values, (10.0, 50.0, 90.0))
    return mean_v, std_v, float(p10), float(p50), float(p90)


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

    mask_u8 = np.where(bird_mask > 0, 255, 0).astype(np.uint8)
    mask_bool = mask_u8 == 255

    if not np.any(mask_bool):
        zeroed = dict.fromkeys(FEATURE_COLUMNS, 0.0)
        zeroed["focal_length_mm"] = focal_mm
        zeroed["zoom_multiple"] = zoom
        zeroed["exif_optical_present"] = exif_present
        return _finalize_ordered_features(zeroed)

    work_h, work_w = bgr_work.shape[:2]

    # 1. Rotation-invariant morphometrics (14 columns).
    props = regionprops(mask_bool.astype(np.uint8))[0]
    area_work = float(props.area)
    perimeter_work = float(props.perimeter)
    area_convex_work = float(props.area_convex)
    eccentricity_raw = float(props.eccentricity)
    eccentricity = (
        min(1.0, max(0.0, eccentricity_raw)) if math.isfinite(eccentricity_raw) else 0.0
    )
    solidity = float(area_work / max(area_convex_work, 1e-6))

    active_contour = contour
    if active_contour is None or len(active_contour) == 0:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    hu_vals = cv2.HuMoments(cv2.moments(mask_u8)).ravel()
    log_hu_1 = _signed_log_hu(float(hu_vals[0]))
    log_hu_2 = _signed_log_hu(float(hu_vals[1]))
    log_hu_3 = _signed_log_hu(float(hu_vals[2]))

    # 2. Strictly pixel-local chromatic plumage & eroded interior texture (26 columns).
    # CRITICAL (PROIBIÇÃO 2 / D-08): Convert strictly bgr_work (NEVER tile-coupled bgr_clahe
    # or spatial bgr_smooth) so outside-mask pixels have zero influence across the mask border.
    lab_raw = cv2.cvtColor(bgr_work, cv2.COLOR_BGR2Lab)
    hsv_raw = cv2.cvtColor(bgr_work, cv2.COLOR_BGR2HSV)

    l_vals = lab_raw[:, :, 0][mask_bool].astype(np.float64)
    a_vals = lab_raw[:, :, 1][mask_bool].astype(np.float64)
    b_vals = lab_raw[:, :, 2][mask_bool].astype(np.float64)
    h_vals = hsv_raw[:, :, 0][mask_bool].astype(np.float64)
    s_vals = hsv_raw[:, :, 1][mask_bool].astype(np.float64)
    v_vals = hsv_raw[:, :, 2][mask_bool].astype(np.float64)

    lab_l_mean, lab_l_std, lab_l_p10, lab_l_p50, lab_l_p90 = _channel_summary(l_vals)
    lab_a_mean, lab_a_std, lab_a_p10, lab_a_p50, lab_a_p90 = _channel_summary(a_vals)
    lab_b_mean, lab_b_std, lab_b_p10, lab_b_p50, lab_b_p90 = _channel_summary(b_vals)

    hsv_s_mean = float(np.mean(s_vals, dtype=np.float64))
    hsv_s_std = float(np.std(s_vals, dtype=np.float64))
    hsv_v_mean = float(np.mean(v_vals, dtype=np.float64))
    hsv_v_std = float(np.std(v_vals, dtype=np.float64))

    sat_mask = s_vals >= _HUE_SATURATION_THRESHOLD
    if int(np.count_nonzero(sat_mask)) >= _MIN_SATURATED_HUE_PIXELS:
        angles = h_vals[sat_mask] * (2.0 * math.pi / 180.0)
        hsv_h_sin_mean = float(np.mean(np.sin(angles), dtype=np.float64))
        hsv_h_cos_mean = float(np.mean(np.cos(angles), dtype=np.float64))
    else:
        hsv_h_sin_mean = 0.0
        hsv_h_cos_mean = 0.0

    yellow_down_ratio = float(
        np.mean((b_vals > 142.0) & (l_vals > 120.0) & (a_vals < 133.0), dtype=np.float64)
    )
    white_plumage_ratio = float(
        np.mean(
            (l_vals > 165.0) & (np.hypot(a_vals - 128.0, b_vals - 128.0) < 16.0),
            dtype=np.float64,
        )
    )
    pink_skin_ratio = float(
        np.mean(
            (a_vals > 136.0) & (l_vals > 90.0) & (s_vals >= _HUE_SATURATION_THRESHOLD),
            dtype=np.float64,
        )
    )

    # Eroded interior feather micro-texture (3-px safety border for 3x3 Scharr/Laplacian stencils).
    l_masked = np.where(mask_bool, lab_raw[:, :, 0], 0).astype(np.uint8)
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded = cv2.erode(mask_u8, k_erode, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    interior = eroded > 0
    if np.any(interior):
        gx = cv2.Scharr(l_masked, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(l_masked, cv2.CV_32F, 0, 1)
        grad_mag_interior = np.hypot(
            gx[interior].astype(np.float64),
            gy[interior].astype(np.float64),
        )
        feather_grad_mean = float(np.mean(grad_mag_interior, dtype=np.float64))
        lap = cv2.Laplacian(l_masked, cv2.CV_32F)
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
