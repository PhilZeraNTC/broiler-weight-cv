"""Shared Pillow EXIF-first image metadata reader and orientation transposer.

This module is the single approved image ingestion path for AvianScale CV
(D-07 / PROIBIÇÃO 2): it reads EXIF orientation, focal length, and Xiaomi
vendor optical zoom metadata strictly before calling ``ImageOps.exif_transpose``.
Raw OpenCV image decoding (``cv2.imread`` / ``cv2.imdecode``) is banned
repository-wide.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

EXIF_TAG_ORIENTATION = 274
EXIF_IFD_POINTER = 0x8769
EXIF_TAG_FOCAL_LENGTH = 37386
XIAOMI_TAG_JSON = 39321
MAX_XIAOMI_TAG_BYTES = 16_384
MIN_VALID_ZOOM_MULTIPLE = 0.1
MAX_VALID_ZOOM_MULTIPLE = 20.0
TRANSPOSE_SWAP_ORIENTATIONS = frozenset({5, 6, 7, 8})


@dataclass(frozen=True, slots=True)
class ExifMetadata:
    """Header-level EXIF orientation and optical calibration metadata."""

    raw_w: int
    raw_h: int
    oriented_w: int
    oriented_h: int
    exif_orientation: int
    focal_length_mm: float | None
    zoom_multiple: float | None
    has_exif_optical: bool


@dataclass(frozen=True, slots=True)
class OrientedImage:
    """EXIF-transposed RGB uint8 image paired with pre-transpose EXIF metadata."""

    rgb: np.ndarray
    metadata: ExifMetadata


def _parse_focal_length(raw: Any) -> float | None:
    """Safely convert an EXIF FocalLength (e.g., IFDRational) to positive finite mm."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _parse_xiaomi_zoom(raw: Any) -> float | None:
    """Safely extract bounded ``zoomMultiple`` from Xiaomi EXIF tag 39321 JSON."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        if len(raw) > MAX_XIAOMI_TAG_BYTES:
            return None
        try:
            text = raw.decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            return None
    elif isinstance(raw, str):
        if len(raw.encode("utf-8", errors="ignore")) > MAX_XIAOMI_TAG_BYTES:
            return None
        text = raw.rstrip("\x00")
    else:
        return None

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    raw_zoom = payload.get("zoomMultiple")
    if raw_zoom is None or isinstance(raw_zoom, bool):
        return None
    try:
        zoom = float(raw_zoom)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(zoom) or not (
        MIN_VALID_ZOOM_MULTIPLE <= zoom <= MAX_VALID_ZOOM_MULTIPLE
    ):
        return None
    return zoom


def _extract_exif_from_open_image(im: Image.Image) -> ExifMetadata:
    """Extract orientation, optical EXIF tags, and pre/post-transpose dimensions."""
    exif = im.getexif()
    raw_orient = exif.get(EXIF_TAG_ORIENTATION, 1)
    if isinstance(raw_orient, int) and not isinstance(raw_orient, bool) and 1 <= raw_orient <= 8:
        orientation = int(raw_orient)
    else:
        orientation = 1

    ifd = exif.get_ifd(EXIF_IFD_POINTER)
    focal_length_mm = _parse_focal_length(ifd.get(EXIF_TAG_FOCAL_LENGTH))
    zoom_multiple = _parse_xiaomi_zoom(ifd.get(XIAOMI_TAG_JSON))

    raw_w, raw_h = im.size
    if orientation in TRANSPOSE_SWAP_ORIENTATIONS:
        oriented_w, oriented_h = raw_h, raw_w
    else:
        oriented_w, oriented_h = raw_w, raw_h

    return ExifMetadata(
        raw_w=raw_w,
        raw_h=raw_h,
        oriented_w=oriented_w,
        oriented_h=oriented_h,
        exif_orientation=orientation,
        focal_length_mm=focal_length_mm,
        zoom_multiple=zoom_multiple,
        has_exif_optical=(focal_length_mm is not None and zoom_multiple is not None),
    )


def _open_image_source(source: Path | bytes) -> Image.Image:
    """Open a Path or raw image bytes using Pillow without altering MAX_IMAGE_PIXELS."""
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    return Image.open(source)


def read_exif_metadata(source: Path | bytes) -> ExifMetadata:
    """Header-only EXIF & post-transpose dimension reader (does not decode pixel data)."""
    with _open_image_source(source) as im:
        return _extract_exif_from_open_image(im)


def load_oriented_rgb(source: Path | bytes) -> OrientedImage:
    """Extract EXIF metadata BEFORE applying ``ImageOps.exif_transpose().convert('RGB')``."""
    with _open_image_source(source) as im:
        metadata = _extract_exif_from_open_image(im)
        oriented = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.asarray(oriented, dtype=np.uint8)
    return OrientedImage(rgb=rgb, metadata=metadata)
