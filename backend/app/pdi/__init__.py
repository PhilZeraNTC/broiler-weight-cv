"""Public package barrel for AvianScale CV classical PDI and Watershed engines."""

from __future__ import annotations

from backend.app.pdi.config import PDIConfig
from backend.app.pdi.features import (
    BANNED_FEATURE_COLUMNS,
    CHROMATIC_TEXTURE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MORPHOMETRIC_FEATURE_COLUMNS,
    OPTICAL_FEATURE_COLUMNS,
    extract_mask_gated_features,
)
from backend.app.pdi.io_exif import (
    WIDE_FOCAL_THRESHOLD_MM,
    ExifMetadata,
    OpticalContext,
    OpticalOverride,
    OrientedImage,
    WorkingImage,
    classify_lens,
    load_oriented_rgb,
    load_working_rgb,
    read_exif_metadata,
    resolve_optical_context,
)
from backend.app.pdi.lcd_suppression import LcdSuppressionResult, suppress_lcd_and_platter
from backend.app.pdi.pipeline import PDIResult, run_single_bird_pdi
from backend.app.pdi.preprocess import PreprocessedFrame, preprocess_working_rgb
from backend.app.pdi.segmentation import (
    SingleBirdSegmentation,
    classify_scene_border_ring,
    fit_ellipse_safe,
    segment_single_bird,
)
from backend.app.pdi.watershed import (
    WatershedInstance,
    WatershedResult,
    WatershedVisuals,
    run_watershed_pdi,
)
from backend.app.pdi.xray import XRayArtifacts, render_xray_artifacts

__all__ = [
    "BANNED_FEATURE_COLUMNS",
    "CHROMATIC_TEXTURE_FEATURE_COLUMNS",
    "ExifMetadata",
    "FEATURE_COLUMNS",
    "LcdSuppressionResult",
    "MORPHOMETRIC_FEATURE_COLUMNS",
    "OPTICAL_FEATURE_COLUMNS",
    "OpticalContext",
    "OpticalOverride",
    "OrientedImage",
    "PDIConfig",
    "PDIResult",
    "PreprocessedFrame",
    "SingleBirdSegmentation",
    "WIDE_FOCAL_THRESHOLD_MM",
    "WatershedInstance",
    "WatershedResult",
    "WatershedVisuals",
    "WorkingImage",
    "XRayArtifacts",
    "classify_lens",
    "classify_scene_border_ring",
    "extract_mask_gated_features",
    "fit_ellipse_safe",
    "load_oriented_rgb",
    "load_working_rgb",
    "preprocess_working_rgb",
    "read_exif_metadata",
    "render_xray_artifacts",
    "resolve_optical_context",
    "run_single_bird_pdi",
    "run_watershed_pdi",
    "segment_single_bird",
    "suppress_lcd_and_platter",
]
