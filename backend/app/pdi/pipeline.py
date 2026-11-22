"""Single-source classical single-bird PDI orchestrator (D-01..D-08).

Shared verbatim by offline batch feature extraction (``return_xray=False``,
Phase 2 ``extract_features.py``) and online FastAPI inference
(``return_xray=True``, Phase 4 ``/api/predict``) to guarantee zero
training-serving skew bound by ``PDIConfig.config_hash()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from backend.app.pdi.config import PDIConfig
from backend.app.pdi.features import extract_mask_gated_features
from backend.app.pdi.io_exif import (
    OpticalContext,
    OpticalOverride,
    load_working_rgb,
    resolve_optical_context,
)
from backend.app.pdi.preprocess import preprocess_working_rgb
from backend.app.pdi.segmentation import SingleBirdSegmentation, segment_single_bird
from backend.app.pdi.xray import XRayArtifacts, render_xray_artifacts


@dataclass(frozen=True, slots=True)
class PDIResult:
    """Complete single-bird classical PDI output contract."""

    features: dict[str, float]
    optical: OpticalContext
    segmentation: SingleBirdSegmentation
    pdi_config_hash: str
    latency_ms: float
    xray: XRayArtifacts | None = None


def run_single_bird_pdi(
    source: Path | bytes,
    *,
    config: PDIConfig | None = None,
    optical_override: OpticalOverride | None = None,
    return_xray: bool = False,
) -> PDIResult:
    """Run the unified single-bird classical PDI pipeline identically on ``Path`` or ``bytes``."""
    t0 = time.perf_counter()
    cfg = config if config is not None else PDIConfig()

    working = load_working_rgb(
        source,
        working_long_edge=cfg.working_long_edge,
        draft_scale_multiplier=cfg.draft_scale_multiplier,
    )
    optical = resolve_optical_context(
        working.metadata,
        override=optical_override,
        default_focal_length_mm=cfg.default_focal_length_mm,
        default_zoom_multiple=cfg.default_zoom_multiple,
    )
    frame = preprocess_working_rgb(working.rgb_work, cfg)
    seg = segment_single_bird(frame, cfg)

    # CRITICAL (PROIBIÇÃO 2 / D-08): Always pass pixel-local frame.bgr_work
    # (NEVER tile-coupled frame.bgr_clahe or spatial frame.bgr_smooth) into feature extraction.
    features = extract_mask_gated_features(
        frame.bgr_work,
        seg.bird_mask,
        working.scale_factor,
        optical,
        contour=seg.contour,
        ellipse=seg.ellipse,
    )

    xray = render_xray_artifacts(frame, seg, cfg) if return_xray else None
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PDIResult(
        features=features,
        optical=optical,
        segmentation=seg,
        pdi_config_hash=cfg.config_hash(),
        latency_ms=latency_ms,
        xray=xray,
    )


__all__ = ["PDIResult", "run_single_bird_pdi"]
