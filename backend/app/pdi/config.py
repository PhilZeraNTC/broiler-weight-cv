"""Immutable classical PDI configuration registry and SHA-256 fingerprint (D-01).

Locks every tunable working-scale, smoothing, CLAHE, scene-classification,
mask-envelope, optical-default, X-Ray, and Watershed parameter inside a frozen
slotted dataclass so offline batch extraction (Phase 2), model training
(Phase 3), and online FastAPI inference (Phase 4) stay bound to an identical
16-hex ``pdi_config_hash``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal


def _validate_int(
    name: str,
    value: object,
    *,
    min_value: int,
    max_value: int | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value!r}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value!r}")


def _validate_finite_float(
    name: str,
    value: object,
    *,
    min_value: float | None = None,
    min_inclusive: bool = True,
    max_value: float | None = None,
    max_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None:
        if min_inclusive and numeric < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {value!r}")
        if not min_inclusive and numeric <= min_value:
            raise ValueError(f"{name} must be > {min_value}, got {value!r}")
    if max_value is not None:
        if max_inclusive and numeric > max_value:
            raise ValueError(f"{name} must be <= {max_value}, got {value!r}")
        if not max_inclusive and numeric >= max_value:
            raise ValueError(f"{name} must be < {max_value}, got {value!r}")
    return numeric


@dataclass(frozen=True, slots=True)
class PDIConfig:
    """Frozen parameter contract for the AvianScale CV classical PDI engine."""

    # Canonical working scale & aspect-preserving JPEG draft decode (Spike 1 / D-01)
    working_long_edge: int = 768
    draft_scale_multiplier: float = 1.0

    # Edge-preserving smoothing + CIE Lab L-channel CLAHE (D-03)
    smoothing_method: Literal["bilateral", "median"] = "bilateral"
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    median_ksize: int = 5
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # Bimodal border-ring scene classifier (Spike 3 / D-06)
    border_ring_frac: float = 0.13
    bucket_smooth_white_min_frac: float = 0.50
    bucket_ring_grad_max: float = 45.0
    bucket_min_white_sides: int = 3

    # Foreground silhouette validity envelope (D-07)
    min_valid_area_frac: float = 0.015
    max_valid_area_frac: float = 0.72

    # Optical EXIF fallback defaults (D-02)
    default_focal_length_mm: float = 4.25
    default_zoom_multiple: float = 1.0

    # 4-stage X-Ray visual encoding contract (D-04)
    xray_display_long_edge: int = 640
    xray_jpeg_quality: int = 82

    # Isolated Marker-based Watershed relative defaults (Spike 4 / D-11)
    watershed_h_rel: float = 0.20
    watershed_min_distance_rel: float = 0.85
    watershed_min_area_ratio: float = 0.10

    def __post_init__(self) -> None:
        _validate_int("working_long_edge", self.working_long_edge, min_value=64)
        _validate_finite_float(
            "draft_scale_multiplier",
            self.draft_scale_multiplier,
            min_value=0.0,
            min_inclusive=False,
        )
        if self.smoothing_method not in ("bilateral", "median"):
            raise ValueError(
                "smoothing_method must be 'bilateral' or 'median', "
                f"got {self.smoothing_method!r}"
            )
        _validate_int("bilateral_d", self.bilateral_d, min_value=1)
        _validate_finite_float(
            "bilateral_sigma_color",
            self.bilateral_sigma_color,
            min_value=0.0,
            min_inclusive=False,
        )
        _validate_finite_float(
            "bilateral_sigma_space",
            self.bilateral_sigma_space,
            min_value=0.0,
            min_inclusive=False,
        )
        _validate_int("median_ksize", self.median_ksize, min_value=3)
        if self.median_ksize % 2 != 1:
            raise ValueError(f"median_ksize must be odd, got {self.median_ksize!r}")
        _validate_finite_float(
            "clahe_clip_limit",
            self.clahe_clip_limit,
            min_value=0.0,
            min_inclusive=False,
        )
        if not isinstance(self.clahe_tile_grid, tuple) or len(self.clahe_tile_grid) != 2:
            raise ValueError(
                f"clahe_tile_grid must be a 2-tuple of positive ints, got {self.clahe_tile_grid!r}"
            )
        _validate_int("clahe_tile_grid[0]", self.clahe_tile_grid[0], min_value=1)
        _validate_int("clahe_tile_grid[1]", self.clahe_tile_grid[1], min_value=1)

        _validate_finite_float(
            "border_ring_frac",
            self.border_ring_frac,
            min_value=0.0,
            min_inclusive=False,
            max_value=0.5,
            max_inclusive=False,
        )
        _validate_finite_float(
            "bucket_smooth_white_min_frac",
            self.bucket_smooth_white_min_frac,
            min_value=0.0,
            max_value=1.0,
        )
        _validate_finite_float(
            "bucket_ring_grad_max",
            self.bucket_ring_grad_max,
            min_value=0.0,
            min_inclusive=False,
        )
        _validate_int(
            "bucket_min_white_sides",
            self.bucket_min_white_sides,
            min_value=1,
            max_value=4,
        )

        min_frac = _validate_finite_float(
            "min_valid_area_frac",
            self.min_valid_area_frac,
            min_value=0.0,
            min_inclusive=False,
            max_value=1.0,
            max_inclusive=False,
        )
        max_frac = _validate_finite_float(
            "max_valid_area_frac",
            self.max_valid_area_frac,
            min_value=0.0,
            min_inclusive=False,
            max_value=1.0,
            max_inclusive=False,
        )
        if min_frac >= max_frac:
            raise ValueError(
                "min_valid_area_frac must be strictly less than max_valid_area_frac, "
                f"got ({min_frac!r}, {max_frac!r})"
            )

        _validate_finite_float(
            "default_focal_length_mm",
            self.default_focal_length_mm,
            min_value=0.5,
            max_value=200.0,
        )
        _validate_finite_float(
            "default_zoom_multiple",
            self.default_zoom_multiple,
            min_value=0.1,
            max_value=20.0,
        )

        _validate_int("xray_display_long_edge", self.xray_display_long_edge, min_value=64)
        _validate_int(
            "xray_jpeg_quality",
            self.xray_jpeg_quality,
            min_value=1,
            max_value=100,
        )

        _validate_finite_float(
            "watershed_h_rel",
            self.watershed_h_rel,
            min_value=0.0,
            min_inclusive=False,
            max_value=1.0,
            max_inclusive=False,
        )
        _validate_finite_float(
            "watershed_min_distance_rel",
            self.watershed_min_distance_rel,
            min_value=0.0,
            min_inclusive=False,
        )
        _validate_finite_float(
            "watershed_min_area_ratio",
            self.watershed_min_area_ratio,
            min_value=0.0,
            max_value=1.0,
            max_inclusive=False,
        )

    def config_hash(self) -> str:
        """Return a deterministic 16-char lowercase SHA-256 hex digest of all fields."""
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


__all__ = ["PDIConfig"]
