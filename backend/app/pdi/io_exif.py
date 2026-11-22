"""Shared Pillow EXIF-first image metadata reader, orientation transposer, and working loader.

This module is the single approved image ingestion path for AvianScale CV
(D-01 / D-02 / PROIBIÇÃO 2): it reads EXIF orientation, focal length, and
Xiaomi vendor optical zoom metadata strictly before calling ``im.draft()`` or
``ImageOps.exif_transpose()``. Raw OpenCV image decoding is banned
repository-wide, and this module stays strictly acyclic with zero imports from
``backend.app.ingestion``.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

EXIF_TAG_ORIENTATION = 274
EXIF_IFD_POINTER = 0x8769
EXIF_TAG_FOCAL_LENGTH = 37386
XIAOMI_TAG_JSON = 39321
MAX_XIAOMI_TAG_BYTES = 16_384
MIN_VALID_ZOOM_MULTIPLE = 0.1
MAX_VALID_ZOOM_MULTIPLE = 20.0
MIN_VALID_FOCAL_LENGTH_MM = 0.5
MAX_VALID_FOCAL_LENGTH_MM = 200.0
WIDE_FOCAL_THRESHOLD_MM = 3.0
DEFAULT_FOCAL_LENGTH_MM = 4.25
DEFAULT_ZOOM_MULTIPLE = 1.0
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


@dataclass(frozen=True, slots=True)
class OpticalOverride:
    """Optional caller-supplied manual optical parameters overriding EXIF tags."""

    focal_length_mm: float | None = None
    zoom_multiple: float | None = None


@dataclass(frozen=True, slots=True)
class OpticalContext:
    """Resolved optical calibration context after override -> EXIF -> default precedence."""

    focal_length_mm: float
    zoom_multiple: float
    exif_optical_present: bool
    override_applied: bool
    lens_class: Literal["wide", "ultrawide", "unknown"]


@dataclass(frozen=True, slots=True)
class WorkingImage:
    """Aspect-preserving canonical working-resolution RGB uint8 image + pre-draft EXIF."""

    rgb_work: np.ndarray
    metadata: ExifMetadata
    scale_factor: float
    work_w: int
    work_h: int


def classify_lens(focal_length_mm: float | None) -> Literal["wide", "ultrawide", "unknown"]:
    """Classify optical lens family from EXIF focal length (``>= 3.0 mm`` wide vs ultrawide)."""
    if (
        focal_length_mm is None
        or isinstance(focal_length_mm, bool)
        or not math.isfinite(focal_length_mm)
    ):
        return "unknown"
    if focal_length_mm >= WIDE_FOCAL_THRESHOLD_MM:
        return "wide"
    return "ultrawide"


def _validate_bounded_float(
    name: str,
    value: Any,
    *,
    min_value: float,
    max_value: float,
) -> float:
    """Validate an ASVS V5 finite numeric bound, rejecting bool/NaN/Inf/out-of-range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not (min_value <= numeric <= max_value):
        raise ValueError(
            f"{name} must be finite within [{min_value}, {max_value}], got {value!r}"
        )
    return numeric


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


def resolve_optical_context(
    metadata: ExifMetadata,
    override: OpticalOverride | None = None,
    *,
    default_focal_length_mm: float = DEFAULT_FOCAL_LENGTH_MM,
    default_zoom_multiple: float = DEFAULT_ZOOM_MULTIPLE,
) -> OpticalContext:
    """Resolve optical context with ``override -> EXIF -> default`` precedence (ASVS V5)."""
    valid_default_focal = _validate_bounded_float(
        "default_focal_length_mm",
        default_focal_length_mm,
        min_value=MIN_VALID_FOCAL_LENGTH_MM,
        max_value=MAX_VALID_FOCAL_LENGTH_MM,
    )
    valid_default_zoom = _validate_bounded_float(
        "default_zoom_multiple",
        default_zoom_multiple,
        min_value=MIN_VALID_ZOOM_MULTIPLE,
        max_value=MAX_VALID_ZOOM_MULTIPLE,
    )

    override_focal: float | None = None
    override_zoom: float | None = None
    if override is not None:
        if override.focal_length_mm is not None:
            override_focal = _validate_bounded_float(
                "focal_length_mm",
                override.focal_length_mm,
                min_value=MIN_VALID_FOCAL_LENGTH_MM,
                max_value=MAX_VALID_FOCAL_LENGTH_MM,
            )
        if override.zoom_multiple is not None:
            override_zoom = _validate_bounded_float(
                "zoom_multiple",
                override.zoom_multiple,
                min_value=MIN_VALID_ZOOM_MULTIPLE,
                max_value=MAX_VALID_ZOOM_MULTIPLE,
            )

    if override_focal is not None:
        focal_length_mm = override_focal
    elif metadata.focal_length_mm is not None:
        focal_length_mm = metadata.focal_length_mm
    else:
        focal_length_mm = valid_default_focal

    if override_zoom is not None:
        zoom_multiple = override_zoom
    elif metadata.zoom_multiple is not None:
        zoom_multiple = metadata.zoom_multiple
    else:
        zoom_multiple = valid_default_zoom

    override_applied = bool(override_focal is not None or override_zoom is not None)
    return OpticalContext(
        focal_length_mm=focal_length_mm,
        zoom_multiple=zoom_multiple,
        exif_optical_present=bool(metadata.has_exif_optical),
        override_applied=override_applied,
        lens_class=classify_lens(focal_length_mm),
    )


def load_working_rgb(
    source: Path | bytes,
    working_long_edge: int = 768,
    draft_scale_multiplier: float = 1.0,
) -> WorkingImage:
    """Read EXIF BEFORE aspect-preserving ``draft()``, ``exif_transpose()``, and ``INTER_AREA``."""
    if isinstance(working_long_edge, bool) or not isinstance(working_long_edge, int):
        raise ValueError(f"working_long_edge must be an integer, got {working_long_edge!r}")
    if working_long_edge < 0:
        raise ValueError(f"working_long_edge must be >= 0, got {working_long_edge!r}")
    if isinstance(draft_scale_multiplier, bool) or not isinstance(
        draft_scale_multiplier, (int, float)
    ):
        raise ValueError(
            "draft_scale_multiplier must be a finite positive number, "
            f"got {draft_scale_multiplier!r}"
        )
    draft_mult = float(draft_scale_multiplier)
    if not math.isfinite(draft_mult) or draft_mult <= 0.0:
        raise ValueError(
            f"draft_scale_multiplier must be finite and > 0, got {draft_scale_multiplier!r}"
        )

    with _open_image_source(source) as im:
        # 1. Extract pre-draft dimensions & optical tags FIRST (im.draft mutates im.size!).
        metadata = _extract_exif_from_open_image(im)
        if im.format == "JPEG" and working_long_edge > 0:
            raw_long = max(im.size)
            if raw_long > working_long_edge:
                draft_scale = min(1.0, (working_long_edge * draft_mult) / float(raw_long))
                draft_box = (
                    max(1, int(im.size[0] * draft_scale)),
                    max(1, int(im.size[1] * draft_scale)),
                )
                im.draft("RGB", draft_box)
        oriented = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.asarray(oriented, dtype=np.uint8)

    h0, w0 = rgb.shape[:2]
    long_cur = max(h0, w0)
    if working_long_edge > 0 and long_cur > working_long_edge:
        resize_scale = working_long_edge / float(long_cur)
        new_w = max(1, int(round(w0 * resize_scale)))
        new_h = max(1, int(round(h0 * resize_scale)))
        rgb_work = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        rgb_work = rgb

    work_h, work_w = rgb_work.shape[:2]
    orig_long = max(metadata.oriented_h, metadata.oriented_w)
    scale_factor = float(max(work_h, work_w)) / float(max(1, orig_long))
    return WorkingImage(
        rgb_work=rgb_work,
        metadata=metadata,
        scale_factor=scale_factor,
        work_w=work_w,
        work_h=work_h,
    )


__all__ = [
    "EXIF_TAG_ORIENTATION",
    "EXIF_IFD_POINTER",
    "EXIF_TAG_FOCAL_LENGTH",
    "XIAOMI_TAG_JSON",
    "MAX_XIAOMI_TAG_BYTES",
    "MIN_VALID_ZOOM_MULTIPLE",
    "MAX_VALID_ZOOM_MULTIPLE",
    "MIN_VALID_FOCAL_LENGTH_MM",
    "MAX_VALID_FOCAL_LENGTH_MM",
    "WIDE_FOCAL_THRESHOLD_MM",
    "DEFAULT_FOCAL_LENGTH_MM",
    "DEFAULT_ZOOM_MULTIPLE",
    "TRANSPOSE_SWAP_ORIENTATIONS",
    "ExifMetadata",
    "OrientedImage",
    "OpticalOverride",
    "OpticalContext",
    "WorkingImage",
    "classify_lens",
    "read_exif_metadata",
    "load_oriented_rgb",
    "load_working_rgb",
    "resolve_optical_context",
]
