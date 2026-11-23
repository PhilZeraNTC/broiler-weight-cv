"""Classical 3-scenario single-bird segmentation and guarded ellipse fitting (D-06, D-07).

Implements strictly classical, zero-deep-learning, zero-folder-hardcoding
segmentation across all three single-bird visual regimes:
1. Bimodal outer border-ring scene classification (``bucket_dorsal`` vs
   ``scale_or_litter``).
2. Scenario B (white plastic dorsal bucket, including white-on-white ``>1,000 g``
   broilers): border-connected smooth-wall peeling gated by a morphological
   Black-Hat contact-shadow barrier, followed by interior multi-cue plumage
   fusion.
3. Scenarios A & C (SF-400 scale on straw and oblique straw litter): dedicated
   rotation-invariant ``suppress_lcd_and_platter()`` strictly BEFORE
   morphological opening/closing/selection (and re-zeroed after hole fill),
   paired with Lab/HSV/gradient bird-vs-straw fusion and proportional elliptical
   ``MORPH_OPEN(9x9)`` bridge severing.
4. Orientation-preserving guarded ``fit_ellipse_safe()`` and deterministic
   ``[0.015, 0.72]`` adaptive fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from backend.app.pdi.config import PDIConfig
from backend.app.pdi.lcd_suppression import LcdSuppressionResult, suppress_lcd_and_platter
from backend.app.pdi.preprocess import PreprocessedFrame

SceneType = Literal["bucket_dorsal", "scale_or_litter"]
EllipseTuple = tuple[tuple[float, float], tuple[float, float], float]


@dataclass(frozen=True, slots=True)
class SingleBirdSegmentation:
    """Clean single-bird binary silhouette, contour, ellipse, and audit metadata."""

    bird_mask: np.ndarray
    contour: np.ndarray
    ellipse: EllipseTuple
    ellipse_from_min_area_rect: bool
    scene_type: SceneType
    scene_is_bucket: bool
    lcd_excluded_px: int
    lcd_core_count: int
    segmentation_fallback: bool
    mask_area_frac: float


def _border_ring_and_strips(
    h: int,
    w: int,
    frac: float,
) -> tuple[np.ndarray, tuple[tuple[slice, slice], ...]]:
    """Build the outer perimeter ring mask and 4 independent border strip slices."""
    bh = max(1, int(round(h * frac)))
    bw = max(1, int(round(w * frac)))
    ring = np.zeros((h, w), dtype=bool)
    ring[:bh, :] = True
    ring[h - bh :, :] = True
    ring[:, :bw] = True
    ring[:, w - bw :] = True
    strips = (
        (slice(0, bh), slice(None)),
        (slice(h - bh, h), slice(None)),
        (slice(None), slice(0, bw)),
        (slice(None), slice(w - bw, w)),
    )
    return ring, strips


def _build_center_prior(h: int, w: int) -> np.ndarray:
    """Build a symmetric 2D Gaussian centrality prior peaking at ``1.0`` in frame center."""
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    sy = max(1.0, 0.35 * h)
    sx = max(1.0, 0.35 * w)
    return np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2)).astype(np.float32)


def _fill_holes_u8(binary_u8: np.ndarray) -> np.ndarray:
    """Fill all enclosed binary holes via external contour flood fill (bit-identical to SciPy)."""
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(binary_u8)
    if contours:
        cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def _percentile_u8(values_u8: np.ndarray, percentiles: tuple[float, ...]) -> tuple[float, ...]:
    """Compute exact linear-interpolated percentiles on 1D ``uint8`` values via histogram."""
    n = values_u8.size
    if n == 0:
        return tuple(0.0 for _ in percentiles)
    cdf = np.cumsum(np.bincount(values_u8, minlength=256))
    results: list[float] = []
    for p in percentiles:
        pos = (p / 100.0) * float(n - 1)
        lo_i = int(math.floor(pos))
        hi_i = int(math.ceil(pos))
        frac = pos - lo_i
        v_lo = float(np.searchsorted(cdf, lo_i + 1))
        v_hi = float(np.searchsorted(cdf, hi_i + 1))
        results.append(v_lo + frac * (v_hi - v_lo))
    return tuple(results)


def _median_u8(values_u8: np.ndarray) -> float:
    """Compute exact median on 1D ``uint8`` values via histogram."""
    return _percentile_u8(values_u8, (50.0,))[0]


def _otsu_threshold_masked(values: np.ndarray) -> float:
    """Compute OpenCV Otsu scalar threshold strictly over a 1D uint8 pixel subset."""
    if values.size == 0:
        return 128.0
    if values.size < 16:
        return float(np.median(values))
    thr, _ = cv2.threshold(
        values.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return float(thr)


def _normalize_angle_deg(angle_deg: float) -> float:
    """Wrap angle strictly into ``[0.0, 180.0)`` guarding IEEE-754 ``-0.0`` edge rounding."""
    wrapped = float(angle_deg) % 180.0
    if wrapped >= 180.0 or wrapped == 0.0:
        return 0.0
    return wrapped


def _normalize_ellipse_axes(
    cx: float,
    cy: float,
    d1: float,
    d2: float,
    angle: float,
) -> EllipseTuple:
    """Ensure ``major >= minor`` and ``major`` stays strictly aligned with ``angle_deg``."""
    if d1 >= d2:
        major, minor = float(d1), float(d2)
        angle_deg = _normalize_angle_deg(angle)
    else:
        major, minor = float(d2), float(d1)
        angle_deg = _normalize_angle_deg(float(angle) + 90.0)
    return ((float(cx), float(cy)), (major, minor), angle_deg)


def fit_ellipse_safe(contour: np.ndarray) -> tuple[EllipseTuple, bool]:
    """Fit an orientation-normalized ellipse with ``>=5``-point and non-collinear guards.

    Returns ``(((cx, cy), (major, minor), angle_deg), used_min_area_rect_fallback)`` where
    ``major >= minor``, ``0.0 <= angle_deg < 180.0``, and ``major`` is strictly aligned
    along ``angle_deg`` (consistent with ``cv2.ellipse(img, ellipse, ...)``).
    """
    if contour is None or len(contour) == 0:
        raise ValueError("contour must be non-empty")

    (rcx, rcy), (rw, rh), rangle = cv2.minAreaRect(contour)
    if len(contour) >= 5 and float(cv2.contourArea(contour)) >= 10.0:
        try:
            (cx, cy), (d1, d2), angle = cv2.fitEllipse(contour)
            max_allowed = max(100.0, 10.0 * max(float(rw), float(rh)))
            if (
                math.isfinite(cx)
                and math.isfinite(cy)
                and math.isfinite(d1)
                and math.isfinite(d2)
                and math.isfinite(angle)
                and d1 > 0.0
                and d2 > 0.0
                and max(d1, d2) <= max_allowed
            ):
                return _normalize_ellipse_axes(cx, cy, d1, d2, angle), False
        except cv2.error:
            pass

    return _normalize_ellipse_axes(rcx, rcy, rw, rh, rangle), True


def _smooth_white_patch(
    lab_patch: np.ndarray,
    grad_patch: np.ndarray,
) -> np.ndarray:
    """Compute exact smooth-white mask on a border patch without full-frame float32 work."""
    l_ch = lab_patch[:, :, 0]
    a_d = lab_patch[:, :, 1].astype(np.float32) - 128.0
    b_d = lab_patch[:, :, 2].astype(np.float32) - 128.0
    return (l_ch >= 140) & (np.hypot(a_d, b_d) <= 22.0) & (grad_patch <= 40.0)


def classify_scene_border_ring(
    frame: PreprocessedFrame,
    config: PDIConfig,
) -> tuple[SceneType, bool]:
    """Classify Scenario B (``bucket_dorsal``) vs A/C (``scale_or_litter``) via border ring."""
    h, w = frame.lab.shape[:2]
    ring, strips = _border_ring_and_strips(h, w, config.border_ring_frac)

    l_ring = frame.lab[:, :, 0][ring]
    a_ring = frame.lab[:, :, 1][ring].astype(np.float32) - 128.0
    b_ring = frame.lab[:, :, 2][ring].astype(np.float32) - 128.0
    g_ring = frame.grad_l[ring]

    ring_sw_frac = float(
        ((l_ring >= 140) & (np.hypot(a_ring, b_ring) <= 22.0) & (g_ring <= 40.0)).mean()
    )
    if ring_sw_frac < config.bucket_smooth_white_min_frac:
        return "scale_or_litter", False

    ring_gmed = float(np.median(g_ring))
    if ring_gmed > config.bucket_ring_grad_max:
        return "scale_or_litter", False

    sides_sw = sum(
        1
        for row_sl, col_sl in strips
        if float(
            _smooth_white_patch(
                frame.lab[row_sl, col_sl],
                frame.grad_l[row_sl, col_sl],
            ).mean()
        )
        >= 0.35
    )
    if sides_sw >= config.bucket_min_white_sides:
        return "bucket_dorsal", True
    return "scale_or_litter", False


def _select_dominant_component(
    binary_mask: np.ndarray,
    center_prior: np.ndarray,
) -> np.ndarray:
    """Select dominant centrally-weighted component with deterministic tie-breaking."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 1:
        return np.zeros(binary_mask.shape[:2], dtype=np.uint8)

    best_label = 0
    best_key: tuple[float, int, float, float] | None = None
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        sub_labels = labels[y : y + bh, x : x + bw]
        sub_prior = center_prior[y : y + bh, x : x + bw]
        mean_prior = float(np.mean(sub_prior[sub_labels == lbl], dtype=np.float64))
        score = float(area) * (0.25 + 0.75 * mean_prior)
        cx = float(centroids[lbl, 0])
        cy = float(centroids[lbl, 1])
        key = (-round(score, 6), -area, round(cy, 4), round(cx, 4))
        if best_key is None or key < best_key:
            best_key = key
            best_label = lbl

    if best_label == 0:
        return np.zeros(binary_mask.shape[:2], dtype=np.uint8)
    out = np.zeros(binary_mask.shape[:2], dtype=np.uint8)
    bx = int(stats[best_label, cv2.CC_STAT_LEFT])
    by = int(stats[best_label, cv2.CC_STAT_TOP])
    bw = int(stats[best_label, cv2.CC_STAT_WIDTH])
    bh = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    out[by : by + bh, bx : bx + bw][labels[by : by + bh, bx : bx + bw] == best_label] = 255
    return out


def _segment_bucket_dorsal(
    frame: PreprocessedFrame,
    config: PDIConfig,
    center_prior: np.ndarray,
) -> np.ndarray:
    """Scenario B: peel border-connected smooth white bucket walls + Black-Hat barrier."""
    h, w = frame.lab.shape[:2]
    ring, _ = _border_ring_and_strips(h, w, config.border_ring_frac)

    l_ch = frame.lab[:, :, 0]
    a_ch = frame.lab[:, :, 1]
    b_ch = frame.lab[:, :, 2]

    l_ring = l_ch[ring]
    a_ring = a_ch[ring].astype(np.float32) - 128.0
    b_ring = b_ch[ring]
    g_ring = frame.grad_l[ring]
    smooth_ring_sel = (
        (l_ring >= 140)
        & (np.hypot(a_ring, b_ring.astype(np.float32) - 128.0) <= 22.0)
        & (g_ring <= 40.0)
    )
    if int(np.count_nonzero(smooth_ring_sel)) >= 32:
        l_wall = _median_u8(l_ring[smooth_ring_sel])
        b_wall = _median_u8(b_ring[smooth_ring_sel])
        g_wall = float(np.median(frame.grad_l_clahe_blur[ring][smooth_ring_sel]))
    else:
        l_wall = _median_u8(l_ring)
        b_wall = _median_u8(b_ring)
        g_wall = float(np.median(frame.grad_l_clahe_blur[ring]))

    k_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    bh_rim = cv2.morphologyEx(frame.l_clahe, cv2.MORPH_BLACKHAT, k_bh)
    tex_wall_thr = min(46.0, max(24.0, g_wall * 1.75 + 7.0))

    b_lo = int(math.ceil(b_wall - 9.5))
    b_hi = int(math.floor(b_wall + 9.5))
    l_min = int(math.ceil(l_wall - 45.0))
    shadow_barrier = (bh_rim > 18) | (frame.grad_l_clahe_blur > tex_wall_thr)
    wall_cand = (~shadow_barrier) & (b_ch >= b_lo) & (b_ch <= b_hi) & (l_ch >= l_min)

    _, wall_labels = cv2.connectedComponents(wall_cand.astype(np.uint8), connectivity=8)
    border_vals = np.concatenate(
        (
            wall_labels[0, :],
            wall_labels[-1, :],
            wall_labels[:, 0],
            wall_labels[:, -1],
        )
    )
    border_labels = np.unique(border_vals)
    border_labels = border_labels[border_labels > 0]
    if border_labels.size > 0:
        wall_mask = np.isin(wall_labels, border_labels)
    else:
        wall_mask = np.zeros((h, w), dtype=bool)

    interior = ~wall_mask
    if int(np.count_nonzero(interior)) < 64:
        return np.zeros((h, w), dtype=np.uint8)

    l_int = l_ch[interior]
    otsu_int = _otsu_threshold_masked(l_int)
    l_p35, l_p50 = _percentile_u8(l_int, (35.0, 50.0))

    l_white_min = max(int(math.floor(otsu_int)) + 1, int(math.ceil(l_p50)))
    cue_white = (l_ch >= l_white_min) & (frame.grad_l_clahe_blur <= 260.0) & (a_ch <= 133)
    cue_yellow = (
        (b_ch >= int(math.ceil(max(147.0, b_wall + 11.0))))
        & (l_ch >= int(math.ceil(l_p35 + 10.0)))
        & (a_ch <= 130)
    )
    cue_pink = (a_ch >= 135) & (l_ch >= 90)
    cue_tex = (
        (frame.grad_l_clahe_blur >= max(42.0, g_wall * 2.2 + 12.0))
        & (frame.grad_l_clahe_blur <= 220.0)
        & (center_prior > 0.25)
        & (l_ch >= int(math.ceil(l_p35 + 6.0)))
    )

    raw_fg = np.where(
        interior & (cue_white | cue_yellow | cue_pink | cue_tex),
        np.uint8(255),
        np.uint8(0),
    )
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(raw_fg, cv2.MORPH_OPEN, k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
    filled = _fill_holes_u8(cleaned)
    return _select_dominant_component(filled, center_prior)


def _segment_scale_or_litter(
    frame: PreprocessedFrame,
    config: PDIConfig,
    center_prior: np.ndarray,
) -> tuple[np.ndarray, LcdSuppressionResult]:
    """Scenarios A & C: suppress SF-400 LCD/platter FIRST, then isolate bird vs straw."""
    h, w = frame.lab.shape[:2]
    ring, _ = _border_ring_and_strips(h, w, config.border_ring_frac)

    # D-05 / PROIBIÇÃO 2: suppress SF-400 LCD and connected smooth housing strictly FIRST.
    lcd_res = suppress_lcd_and_platter(frame, config)
    excl = lcd_res.exclusion_mask > 0
    valid = ~excl

    ring_valid = ring & valid
    if int(np.count_nonzero(ring_valid)) >= 32:
        bg_ref = ring_valid
    elif int(np.count_nonzero(valid)) >= 32:
        bg_ref = valid
    else:
        bg_ref = np.ones((h, w), dtype=bool)

    l_ch = frame.lab[:, :, 0]
    a_ch = frame.lab[:, :, 1]
    b_ch = frame.lab[:, :, 2]
    s_ch = frame.hsv[:, :, 1]

    l_bg = _median_u8(l_ch[bg_ref])
    a_bg = _median_u8(a_ch[bg_ref])
    g_bg = float(np.median(frame.grad_l_clahe_blur[bg_ref]))

    otsu_l = _otsu_threshold_masked(l_ch[valid] if np.any(valid) else l_ch.ravel())

    cue_white = (l_ch >= int(math.ceil(max(otsu_l - 6.0, l_bg + 12.0)))) & (
        a_ch <= int(math.floor(max(130.5, a_bg + 1.5)))
    )
    cue_yellow = (
        (b_ch >= 136)
        & (a_ch <= int(math.floor(min(129.0, a_bg + 0.5))))
        & (frame.grad_l_clahe_blur <= max(190.0, g_bg * 0.92))
    )
    cue_pink = (a_ch >= 138) & (s_ch >= 45) & (center_prior > 0.20)
    cue_coherent = (l_ch >= int(math.ceil(l_bg + 18.0))) & (frame.grad_l_clahe_blur < 0.75 * g_bg)

    # Subtract LCD/platter exclusion strictly BEFORE morphological opening/closing.
    raw_fg = np.where(
        (cue_white | cue_yellow | cue_pink | cue_coherent) & valid,
        np.uint8(255),
        np.uint8(0),
    )

    # Elliptical MORPH_OPEN(9x9) severs thin leg/platter bridges before closing.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    opened = cv2.morphologyEx(raw_fg, cv2.MORPH_OPEN, k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    filled = _fill_holes_u8(closed)

    # Re-zero LCD exclusion after closing/hole-filling so bridging can never re-enter LCD.
    filled[excl] = 0

    bird_mask = _select_dominant_component(filled, center_prior)
    return bird_mask, lcd_res


def _run_adaptive_fallback(
    frame: PreprocessedFrame,
    center_prior: np.ndarray,
    excl_mask: np.ndarray,
    config: PDIConfig,
) -> np.ndarray:
    """Deterministic secondary Lab z-distance + central-prior fallback (D-07)."""
    h, w = frame.lab.shape[:2]
    frame_area = float(max(1, h * w))
    excl = excl_mask > 0
    valid = ~excl
    ring, _ = _border_ring_and_strips(h, w, config.border_ring_frac)
    bg_ref = (ring & valid) if int(np.count_nonzero(ring & valid)) >= 32 else valid

    l_f = frame.lab[:, :, 0].astype(np.float32)
    a_f = frame.lab[:, :, 1].astype(np.float32)
    b_f = frame.lab[:, :, 2].astype(np.float32)

    if np.any(bg_ref):
        l_r = float(np.median(l_f[bg_ref]))
        a_r = float(np.median(a_f[bg_ref]))
        b_r = float(np.median(b_f[bg_ref]))
        s_l = max(5.0, float(np.median(np.abs(l_f[bg_ref] - l_r))) * 1.4826)
        s_a = max(5.0, float(np.median(np.abs(a_f[bg_ref] - a_r))) * 1.4826)
        s_b = max(5.0, float(np.median(np.abs(b_f[bg_ref] - b_r))) * 1.4826)

        z_dist = np.hypot(
            (l_f - l_r) / s_l,
            np.hypot((a_f - a_r) / s_a, (b_f - b_r) / s_b),
        )
        score = z_dist * (0.35 + 0.65 * center_prior)
        valid_scores = score[valid] if np.any(valid) else score.ravel()

        if valid_scores.size > 0 and float(np.ptp(valid_scores)) > 1e-4:
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            for pct in (70.0, 55.0):
                thr = float(np.percentile(valid_scores, pct))
                cand = np.where((score >= thr) & valid, np.uint8(255), np.uint8(0))
                cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k_open)
                cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k_close)
                cand = _fill_holes_u8(cand)
                cand[excl] = 0
                cand = _select_dominant_component(cand, center_prior)
                frac = float(np.count_nonzero(cand)) / frame_area
                if config.min_valid_area_frac <= frac <= config.max_valid_area_frac:
                    return cand

    # Guaranteed deterministic central fallback ellipse (~0.12 frame area, minus excl).
    fallback = np.zeros((h, w), dtype=np.uint8)
    axis_scale = math.sqrt(0.12 / math.pi)
    rx = max(3, min((w // 2) - 1, int(round(w * axis_scale))))
    ry = max(3, min((h // 2) - 1, int(round(h * axis_scale))))
    cv2.ellipse(fallback, (w // 2, h // 2), (rx, ry), 0.0, 0.0, 360.0, 255, -1)
    fallback[excl] = 0
    if int(np.count_nonzero(fallback)) > 0:
        return _select_dominant_component(fallback, center_prior)

    # Extreme guard if excl covers the entire central ellipse: seed at best valid pixel.
    if np.any(valid):
        masked_prior = np.where(valid, center_prior, -1.0)
        flat_idx = int(np.argmax(masked_prior))
        py, px = divmod(flat_idx, w)
        cv2.circle(fallback, (px, py), max(3, min(h, w) // 12), 255, -1)
        fallback[excl] = 0
        if int(np.count_nonzero(fallback)) > 0:
            return _select_dominant_component(fallback, center_prior)

    # Last-resort guard only if 100% of pixels were excluded.
    cv2.ellipse(fallback, (w // 2, h // 2), (rx, ry), 0.0, 0.0, 360.0, 255, -1)
    return fallback


def segment_single_bird(
    frame: PreprocessedFrame,
    config: PDIConfig,
) -> SingleBirdSegmentation:
    """Segment a single broiler silhouette across bucket, SF-400 scale, and straw scenes."""
    h, w = frame.lab.shape[:2]
    frame_area = float(max(1, h * w))
    center_prior = _build_center_prior(h, w)

    scene_type, scene_is_bucket = classify_scene_border_ring(frame, config)

    if scene_is_bucket:
        bird_mask = _segment_bucket_dorsal(frame, config, center_prior)
        excl_mask = np.zeros((h, w), dtype=np.uint8)
        lcd_excluded_px = 0
        lcd_core_count = 0
    else:
        bird_mask, lcd_res = _segment_scale_or_litter(frame, config, center_prior)
        excl_mask = lcd_res.exclusion_mask
        lcd_excluded_px = lcd_res.lcd_excluded_px
        lcd_core_count = lcd_res.lcd_core_count

    mask_area_frac = float(np.count_nonzero(bird_mask)) / frame_area
    segmentation_fallback = False
    if not (config.min_valid_area_frac <= mask_area_frac <= config.max_valid_area_frac):
        segmentation_fallback = True
        bird_mask = _run_adaptive_fallback(frame, center_prior, excl_mask, config)
        mask_area_frac = float(np.count_nonzero(bird_mask)) / frame_area

    contours, _ = cv2.findContours(bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        segmentation_fallback = True
        bird_mask = _run_adaptive_fallback(
            frame,
            center_prior,
            np.zeros((h, w), dtype=np.uint8),
            config,
        )
        mask_area_frac = float(np.count_nonzero(bird_mask)) / frame_area
        contours, _ = cv2.findContours(bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = max(contours, key=lambda c: (float(cv2.contourArea(c)), len(c)))
    ellipse, ellipse_from_min_area_rect = fit_ellipse_safe(contour)

    return SingleBirdSegmentation(
        bird_mask=bird_mask,
        contour=contour,
        ellipse=ellipse,
        ellipse_from_min_area_rect=ellipse_from_min_area_rect,
        scene_type=scene_type,
        scene_is_bucket=scene_is_bucket,
        lcd_excluded_px=lcd_excluded_px,
        lcd_core_count=lcd_core_count,
        segmentation_fallback=segmentation_fallback,
        mask_area_frac=mask_area_frac,
    )


__all__ = [
    "SingleBirdSegmentation",
    "classify_scene_border_ring",
    "fit_ellipse_safe",
    "segment_single_bird",
]
