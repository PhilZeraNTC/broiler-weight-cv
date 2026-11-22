"""4-stage classical PDI "X-Ray" inspection renderer and Base64 encoder (D-04).

Produces four aspect-preserving inspection stages downscaled to
``config.xray_display_long_edge`` (default ``640 px``) as raw RFC 4648 ASCII
Base64 payloads (no ``data:image/...`` URI prefixes) kept below ``250 KB``:
1. EXIF-oriented working original (``JPEG Q=82``)
2. CLAHE-Lab preprocessed image (``JPEG Q=82``)
3. Clean single-bird binary mask with SF-400 LCD excluded (1-channel ``PNG``)
4. Refined contour + fitted ellipse + orthogonal major/minor axes overlay (``JPEG Q=82``)
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.pdi.config import PDIConfig
from backend.app.pdi.preprocess import PreprocessedFrame
from backend.app.pdi.segmentation import SingleBirdSegmentation

_SILHOUETTE_TINT_BGR: tuple[float, float, float] = (160.0, 220.0, 60.0)
_SILHOUETTE_ALPHA: float = 0.18
_CONTOUR_COLOR_BGR: tuple[int, int, int] = (90, 235, 90)
_ELLIPSE_COLOR_BGR: tuple[int, int, int] = (0, 180, 255)
_MAJOR_AXIS_COLOR_BGR: tuple[int, int, int] = (0, 165, 255)  # Warm amber/orange
_MINOR_AXIS_COLOR_BGR: tuple[int, int, int] = (255, 215, 0)  # Sky cyan
_PNG_COMPRESSION_LEVEL: int = 6


@dataclass(frozen=True, slots=True)
class XRayArtifacts:
    """Encoded 4-stage visual X-Ray artifacts and display/payload metadata."""

    stage1_original_b64: str
    stage2_clahe_b64: str
    stage3_mask_b64: str
    stage4_overlay_b64: str
    display_w: int
    display_h: int
    total_b64_bytes: int


def _resize_for_display(
    img: np.ndarray,
    long_edge: int,
    *,
    is_mask: bool = False,
) -> tuple[np.ndarray, float]:
    """Aspect-preserving downscale to ``long_edge`` without upscaling smaller frames."""
    h, w = img.shape[:2]
    cur_long = max(h, w)
    if long_edge > 0 and cur_long > long_edge:
        scale = float(long_edge) / float(cur_long)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        if is_mask:
            resized = np.where(resized > 0, 255, 0).astype(np.uint8)
        return resized, scale

    if is_mask:
        return np.where(img > 0, 255, 0).astype(np.uint8), 1.0
    return img, 1.0


def _draw_stage4_overlay(
    bgr_work: np.ndarray,
    segmentation: SingleBirdSegmentation,
) -> np.ndarray:
    """Annotate ``bgr_work`` with translucent mask tint, contour, ellipse, and major/minor axes."""
    canvas = bgr_work.copy()

    mask_bool = segmentation.bird_mask > 0
    if np.any(mask_bool):
        tint = np.array(_SILHOUETTE_TINT_BGR, dtype=np.float32)
        blended = (
            canvas[mask_bool].astype(np.float32) * (1.0 - _SILHOUETTE_ALPHA)
            + tint * _SILHOUETTE_ALPHA
        )
        canvas[mask_bool] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    if segmentation.contour is not None and len(segmentation.contour) > 0:
        cv2.drawContours(
            canvas,
            [segmentation.contour],
            -1,
            _CONTOUR_COLOR_BGR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    ((cx, cy), (major, minor), angle_deg) = segmentation.ellipse
    if (
        math.isfinite(cx)
        and math.isfinite(cy)
        and math.isfinite(major)
        and math.isfinite(minor)
        and math.isfinite(angle_deg)
        and major > 0.0
        and minor > 0.0
    ):
        cv2.ellipse(
            canvas,
            ((float(cx), float(cy)), (float(major), float(minor)), float(angle_deg)),
            color=_ELLIPSE_COLOR_BGR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # fit_ellipse_safe() guarantees major >= minor aligned along angle_deg clockwise from +x.
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        half_maj = major / 2.0
        half_min = minor / 2.0

        maj_p1 = (int(round(cx - half_maj * cos_a)), int(round(cy - half_maj * sin_a)))
        maj_p2 = (int(round(cx + half_maj * cos_a)), int(round(cy + half_maj * sin_a)))
        cv2.line(canvas, maj_p1, maj_p2, _MAJOR_AXIS_COLOR_BGR, thickness=2, lineType=cv2.LINE_AA)

        min_p1 = (int(round(cx + half_min * sin_a)), int(round(cy - half_min * cos_a)))
        min_p2 = (int(round(cx - half_min * sin_a)), int(round(cy + half_min * cos_a)))
        cv2.line(canvas, min_p1, min_p2, _MINOR_AXIS_COLOR_BGR, thickness=2, lineType=cv2.LINE_AA)

        center_pt = (int(round(cx)), int(round(cy)))
        cv2.circle(canvas, center_pt, 4, (20, 20, 20), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center_pt, 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    return canvas


def _encode_jpeg_b64(bgr: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode X-Ray stage as JPEG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_mask_png_b64(mask_u8: np.ndarray) -> str:
    if mask_u8.ndim != 2:
        raise ValueError(f"Stage 3 mask must be 2D 1-channel uint8, got shape {mask_u8.shape!r}")
    ok, buf = cv2.imencode(
        ".png",
        mask_u8,
        [int(cv2.IMWRITE_PNG_COMPRESSION), _PNG_COMPRESSION_LEVEL],
    )
    if not ok:
        raise RuntimeError("Failed to encode X-Ray stage 3 mask as PNG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_xray_artifacts(
    frame: PreprocessedFrame,
    segmentation: SingleBirdSegmentation,
    config: PDIConfig,
) -> XRayArtifacts:
    """Render and Base64-encode all 4 classical PDI X-Ray inspection stages."""
    long_edge = config.xray_display_long_edge
    jpeg_q = config.xray_jpeg_quality

    overlay_work = _draw_stage4_overlay(frame.bgr_work, segmentation)

    stage1_disp, _ = _resize_for_display(frame.bgr_work, long_edge, is_mask=False)
    stage2_disp, _ = _resize_for_display(frame.bgr_clahe, long_edge, is_mask=False)
    stage3_disp, _ = _resize_for_display(segmentation.bird_mask, long_edge, is_mask=True)
    stage4_disp, _ = _resize_for_display(overlay_work, long_edge, is_mask=False)

    display_h, display_w = stage1_disp.shape[:2]

    s1_b64 = _encode_jpeg_b64(stage1_disp, jpeg_q)
    s2_b64 = _encode_jpeg_b64(stage2_disp, jpeg_q)
    s3_b64 = _encode_mask_png_b64(stage3_disp)
    s4_b64 = _encode_jpeg_b64(stage4_disp, jpeg_q)

    total_b64_bytes = len(s1_b64) + len(s2_b64) + len(s3_b64) + len(s4_b64)
    return XRayArtifacts(
        stage1_original_b64=s1_b64,
        stage2_clahe_b64=s2_b64,
        stage3_mask_b64=s3_b64,
        stage4_overlay_b64=s4_b64,
        display_w=int(display_w),
        display_h=int(display_h),
        total_b64_bytes=int(total_b64_bytes),
    )


__all__ = ["XRayArtifacts", "render_xray_artifacts"]
