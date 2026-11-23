"""Dedicated rotation-invariant SF-400 LCD bezel and plastic housing suppression (D-05).

Core anti-leakage enforcement boundary for PROIBIÇÃO 2: detects dark,
near-achromatic rectangular SF-400 LCD displays via rotation-invariant
``cv2.minAreaRect`` geometry (zero hardcoded top/bottom frame-strip rules),
dilates each LCD core to cover adjacent ``ON/OFF``/``MODE``/``TARE`` buttons,
and flood-suppresses directly connected smooth cool/neutral plastic SF-400
housing/platter strictly before morphological closing and component selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.pdi.config import PDIConfig
from backend.app.pdi.preprocess import PreprocessedFrame

# Empirically verified SF-400 LCD & smooth cool-white housing photometry/geometry bounds
# (02-RESEARCH.md Spike 3: 672 non-bucket exclusions, 0 bucket false positives).
_LCD_MAX_L = 105
_LCD_MAX_CHROMA = 18.0
_LCD_MAX_CHROMA_SQ = int(_LCD_MAX_CHROMA * _LCD_MAX_CHROMA)
_LCD_MIN_AREA_FRAC = 0.0005
_LCD_MAX_AREA_FRAC = 0.08
_LCD_MIN_ASPECT = 1.25
_LCD_MAX_ASPECT = 4.8
_LCD_MIN_EXTENT = 0.52
_LCD_MIN_SOLIDITY = 0.72

_HOUSING_MIN_L = 155
_HOUSING_MAX_CHROMA = 12.0
_HOUSING_MAX_CHROMA_SQ = int(_HOUSING_MAX_CHROMA * _HOUSING_MAX_CHROMA)
_HOUSING_MAX_B = 134
_HOUSING_MAX_GRAD = 65.0


@dataclass(frozen=True, slots=True)
class LcdSuppressionResult:
    """Output mask and audit counters from rotation-invariant SF-400 LCD suppression."""

    exclusion_mask: np.ndarray
    lcd_core_count: int
    lcd_excluded_px: int


def suppress_lcd_and_platter(
    frame: PreprocessedFrame,
    config: PDIConfig,
) -> LcdSuppressionResult:
    """Suppress SF-400 dark rectangular LCD bezels and connected smooth plastic housing.

    All shape and color rules are strictly rotation-invariant (no frame coordinate
    strips) so arbitrary EXIF orientations and handheld camera angles behave identically.
    """
    _ = config  # Reserved for future config-coupled LCD tuning while preserving uniform API.

    h, w = frame.lab.shape[:2]
    frame_area = float(max(1, h * w))

    l_ch = frame.lab[:, :, 0]
    a_ch = frame.lab[:, :, 1]
    b_ch = frame.lab[:, :, 2]
    da = a_ch.astype(np.int32) - 128
    db = b_ch.astype(np.int32) - 128
    chroma_sq = da * da + db * db

    dark_neutral = np.where(
        (l_ch < _LCD_MAX_L) & (chroma_sq < _LCD_MAX_CHROMA_SQ),
        np.uint8(255),
        np.uint8(0),
    )
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_clean = cv2.morphologyEx(dark_neutral, cv2.MORPH_OPEN, k_open)
    dark_clean = cv2.morphologyEx(dark_clean, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(dark_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lcd_cores = np.zeros((h, w), dtype=np.uint8)
    lcd_core_count = 0
    min_area = _LCD_MIN_AREA_FRAC * frame_area
    max_area = _LCD_MAX_AREA_FRAC * frame_area

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if not (min_area <= area <= max_area):
            continue

        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        short_side = min(rw, rh)
        long_side = max(rw, rh)
        if short_side <= 1e-6:
            continue

        aspect = long_side / short_side
        if not (_LCD_MIN_ASPECT <= aspect <= _LCD_MAX_ASPECT):
            continue

        extent = area / max(rw * rh, 1e-6)
        if extent < _LCD_MIN_EXTENT:
            continue

        hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
        solidity = area / max(hull_area, 1e-6)
        if solidity < _LCD_MIN_SOLIDITY:
            continue

        cv2.drawContours(lcd_cores, [cnt], -1, 255, thickness=cv2.FILLED)
        lcd_core_count += 1

    if lcd_core_count == 0:
        return LcdSuppressionResult(
            exclusion_mask=np.zeros((h, w), dtype=np.uint8),
            lcd_core_count=0,
            lcd_excluded_px=0,
        )

    k_bezel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    bezel_mask = cv2.dilate(lcd_cores, k_bezel)

    housing_cand = np.where(
        (l_ch >= _HOUSING_MIN_L)
        & (chroma_sq <= _HOUSING_MAX_CHROMA_SQ)
        & (b_ch <= _HOUSING_MAX_B)
        & (frame.grad_l_clahe_blur <= _HOUSING_MAX_GRAD),
        np.uint8(1),
        np.uint8(0),
    )

    k_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    halo = cv2.dilate(bezel_mask, k_halo) > 0

    _, labels = cv2.connectedComponents(housing_cand, connectivity=8)
    touching_labels = np.unique(labels[halo])
    touching_labels = touching_labels[touching_labels > 0]
    if touching_labels.size > 0:
        housing_mask = np.isin(labels, touching_labels)
    else:
        housing_mask = np.zeros((h, w), dtype=bool)

    exclusion_mask = np.where((bezel_mask > 0) | housing_mask, np.uint8(255), np.uint8(0))
    return LcdSuppressionResult(
        exclusion_mask=exclusion_mask,
        lcd_core_count=lcd_core_count,
        lcd_excluded_px=int(np.count_nonzero(exclusion_mask)),
    )


__all__ = ["LcdSuppressionResult", "suppress_lcd_and_platter"]
