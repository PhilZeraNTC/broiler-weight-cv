"""Classical OpenCV edge-preserving smoothing, CIE Lab L-CLAHE, HSV, and Scharr maps (D-03).

OpenCV ``uint8`` color-space conventions enforced throughout AvianScale CV:
- Input ``rgb_work`` is strictly RGB ``uint8`` ``(H, W, 3)`` and converted explicitly to
  ``bgr_work`` via ``cv2.COLOR_RGB2BGR``.
- ``lab`` uses OpenCV ``uint8`` CIE Lab scaling: ``L``, ``a``, ``b`` in ``[0, 255]`` with
  achromatic neutral origin at ``a = 128``, ``b = 128``.
- ``hsv`` uses OpenCV ``uint8`` HSV scaling: ``H`` in ``[0, 179]`` (2 deg per step),
  ``S`` and ``V`` in ``[0, 255]``.
- CLAHE is applied exclusively to the CIE Lab ``L`` luminance channel so hue and
  chromaticity stay undistorted.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.pdi.config import PDIConfig


@dataclass(frozen=True, slots=True)
class PreprocessedFrame:
    """Classical OpenCV multi-space representations derived from canonical working RGB."""

    bgr_work: np.ndarray
    bgr_smooth: np.ndarray
    bgr_clahe: np.ndarray
    lab: np.ndarray
    l_clahe: np.ndarray
    hsv: np.ndarray
    grad_l: np.ndarray
    grad_l_clahe_blur: np.ndarray


def _scharr_magnitude(channel: np.ndarray) -> np.ndarray:
    """Compute float32 Scharr gradient magnitude on a single-channel uint8 image."""
    gx = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy).astype(np.float32, copy=False)


def preprocess_working_rgb(rgb_work: np.ndarray, config: PDIConfig) -> PreprocessedFrame:
    """Run edge-preserving smoothing, Lab L-channel CLAHE, HSV, and Scharr gradient maps."""
    if not isinstance(rgb_work, np.ndarray):
        raise ValueError(f"rgb_work must be a numpy ndarray, got {type(rgb_work).__name__}")
    if rgb_work.ndim != 3 or rgb_work.shape[2] != 3:
        raise ValueError(f"rgb_work must have shape (H, W, 3), got {rgb_work.shape!r}")
    if rgb_work.shape[0] == 0 or rgb_work.shape[1] == 0:
        raise ValueError(f"rgb_work must be non-empty, got shape {rgb_work.shape!r}")
    if rgb_work.dtype != np.uint8:
        raise ValueError(f"rgb_work must have dtype uint8, got {rgb_work.dtype!r}")

    bgr_work = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2BGR)

    if config.smoothing_method == "bilateral":
        bgr_smooth = cv2.bilateralFilter(
            bgr_work,
            d=config.bilateral_d,
            sigmaColor=config.bilateral_sigma_color,
            sigmaSpace=config.bilateral_sigma_space,
        )
    else:
        bgr_smooth = cv2.medianBlur(bgr_work, config.median_ksize)

    lab = cv2.cvtColor(bgr_smooth, cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(bgr_smooth, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid,
    )
    l_clahe = clahe.apply(lab[:, :, 0])
    lab_clahe = cv2.merge((l_clahe, lab[:, :, 1], lab[:, :, 2]))
    bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_Lab2BGR)

    grad_l = _scharr_magnitude(lab[:, :, 0])
    grad_l_clahe = _scharr_magnitude(l_clahe)
    grad_l_clahe_blur = cv2.GaussianBlur(grad_l_clahe, (11, 11), 0).astype(np.float32, copy=False)

    return PreprocessedFrame(
        bgr_work=bgr_work,
        bgr_smooth=bgr_smooth,
        bgr_clahe=bgr_clahe,
        lab=lab,
        l_clahe=l_clahe,
        hsv=hsv,
        grad_l=grad_l,
        grad_l_clahe_blur=grad_l_clahe_blur,
    )


__all__ = ["PreprocessedFrame", "preprocess_working_rgb"]
