"""Unit & PROIBIÇÃO 2 invariance tests for the frozen 49-feature PDI extractor (Plan 02-02)."""

from __future__ import annotations

import math
import re

import cv2
import numpy as np
import pytest

from backend.app.pdi.features import (
    BANNED_FEATURE_COLUMNS,
    CHROMATIC_TEXTURE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MORPHOMETRIC_FEATURE_COLUMNS,
    OPTICAL_FEATURE_COLUMNS,
    extract_mask_gated_features,
)
from backend.app.pdi.io_exif import OpticalContext


def _make_textured_bird_frame(
    height: int = 346,
    width: int = 768,
    *,
    center: tuple[int, int] = (384, 173),
    axes: tuple[int, int] = (110, 65),
    angle_deg: float = 18.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic BGR working image with textured multi-tone plumage and ellipse mask."""
    rng = np.random.default_rng(seed)
    bgr = rng.integers(30, 210, size=(height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle_deg, 0, 360, 255, -1)

    # Paint realistic multi-tone plumage strictly inside the mask:
    # warm white feathers + subtle micro-texture + yellow down patch + pink comb patch.
    inside = mask == 255
    base_plumage = np.zeros_like(bgr)
    base_plumage[:, :] = (210, 218, 224)  # BGR warm white
    noise = rng.integers(-18, 19, size=(height, width, 3), dtype=np.int16)
    feathered = np.clip(base_plumage.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    bgr[inside] = feathered[inside]

    # Yellow down patch (low B, high G, high R in BGR -> Lab b* > 142, L* > 120, a* < 133).
    yellow_patch = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(yellow_patch, (center[0] - 25, center[1]), 22, 255, -1)
    y_inside = (yellow_patch == 255) & inside
    bgr[y_inside] = (75, 215, 225)

    # Pink comb patch (moderate B/G, high R in BGR -> Lab a* > 136, L* > 90, S >= 25).
    pink_patch = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(pink_patch, (center[0] + 45, center[1] - 10), 14, 255, -1)
    p_inside = (pink_patch == 255) & inside
    bgr[p_inside] = (120, 115, 220)

    return bgr, mask


def test_feature_schema_has_exact_49_unique_columns_and_excludes_banned_columns() -> None:
    assert len(MORPHOMETRIC_FEATURE_COLUMNS) == 14
    assert len(CHROMATIC_TEXTURE_FEATURE_COLUMNS) == 26
    assert len(OPTICAL_FEATURE_COLUMNS) == 9
    assert len(FEATURE_COLUMNS) == 49
    assert len(set(FEATURE_COLUMNS)) == 49
    assert FEATURE_COLUMNS == (
        MORPHOMETRIC_FEATURE_COLUMNS
        + CHROMATIC_TEXTURE_FEATURE_COLUMNS
        + OPTICAL_FEATURE_COLUMNS
    )

    snake_case_re = re.compile(r"^[a-z][a-z0-9_]*$")
    for col in FEATURE_COLUMNS:
        assert isinstance(col, str)
        assert snake_case_re.match(col), f"Feature name {col!r} is not snake_case"

    expected_banned = frozenset(
        {"oriented_w", "oriented_h", "megapixels", "bbox_w", "bbox_h", "raw_w", "raw_h"}
    )
    assert expected_banned == BANNED_FEATURE_COLUMNS
    assert BANNED_FEATURE_COLUMNS.isdisjoint(FEATURE_COLUMNS)


@pytest.mark.parametrize(
    "center",
    [
        (384, 173),  # Centered silhouette
        (384, 45),   # Silhouette touching top image border
    ],
)
def test_outside_mask_perturbation_yields_bit_identical_features(
    center: tuple[int, int],
) -> None:
    """PROIBIÇÃO 2 / D-08: arbitrary outside-mask LCD digits & noise give exact 0.0 diff."""
    bgr_a, mask = _make_textured_bird_frame(center=center, axes=(110, 65), angle_deg=18.0)

    # Paint extreme high-contrast fake SF-400 LCD digits ("1850") + neon magenta +
    # checkerboard noise strictly on EVERY pixel where bird_mask == 0.
    bgr_b = bgr_a.copy()
    bgr_b[mask == 0] = (255, 0, 255)
    # Add alternating high-frequency stripes right up to the 1-px mask boundary.
    yy, xx = np.indices(mask.shape)
    checker = ((yy + xx) % 2 == 0) & (mask == 0)
    bgr_b[checker] = (0, 255, 0)
    # Paint dark SF-400 LCD box with bright white "1850" digits outside mask.
    lcd_box = np.zeros_like(mask)
    cv2.rectangle(lcd_box, (20, 200), (210, 290), 255, -1)
    bgr_b[(lcd_box == 255) & (mask == 0)] = (15, 15, 15)
    digit_layer = np.zeros_like(bgr_b)
    cv2.putText(
        digit_layer,
        "1850",
        (32, 268),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.7,
        (250, 250, 250),
        4,
    )
    digits_outside = (digit_layer[:, :, 0] > 0) & (mask == 0)
    bgr_b[digits_outside] = (250, 250, 250)

    optical = OpticalContext(
        focal_length_mm=4.25,
        zoom_multiple=1.0,
        exif_optical_present=True,
        override_applied=False,
        lens_class="wide",
    )
    scale_factor = 768.0 / 4080.0
    f_a = extract_mask_gated_features(bgr_a, mask, scale_factor=scale_factor, optical=optical)
    f_b = extract_mask_gated_features(bgr_b, mask, scale_factor=scale_factor, optical=optical)

    assert tuple(f_a.keys()) == FEATURE_COLUMNS
    assert tuple(f_b.keys()) == FEATURE_COLUMNS
    assert f_a["feather_grad_mean"] > 0.0
    assert f_a["feather_laplacian_var"] > 0.0

    for key in FEATURE_COLUMNS:
        assert isinstance(f_a[key], float)
        assert math.isfinite(f_a[key])
        assert f_a[key] == f_b[key], (
            f"PROIBIÇÃO 2 outside-mask leak in {key!r}: {f_a[key]!r} != {f_b[key]!r}"
        )


@pytest.mark.parametrize("rot_k", [1, 2])
def test_rotation_invariance_90_and_180_degrees(rot_k: int) -> None:
    """Rotating both bgr_work and bird_mask by 90°/180° preserves all 49 descriptors."""
    bgr_0, mask_0 = _make_textured_bird_frame(center=(384, 173), axes=(105, 58), angle_deg=23.0)
    bgr_rot = np.ascontiguousarray(np.rot90(bgr_0, k=rot_k))
    mask_rot = np.ascontiguousarray(np.rot90(mask_0, k=rot_k))

    optical = OpticalContext(
        focal_length_mm=4.25,
        zoom_multiple=1.0,
        exif_optical_present=True,
        override_applied=False,
        lens_class="wide",
    )
    scale_factor = 768.0 / 4080.0
    f_0 = extract_mask_gated_features(bgr_0, mask_0, scale_factor=scale_factor, optical=optical)
    f_r = extract_mask_gated_features(bgr_rot, mask_rot, scale_factor=scale_factor, optical=optical)

    assert tuple(f_0.keys()) == FEATURE_COLUMNS
    assert tuple(f_r.keys()) == FEATURE_COLUMNS

    for key in FEATURE_COLUMNS:
        assert f_r[key] == pytest.approx(f_0[key], rel=1e-3, abs=1e-4), (
            f"Rotation invariance broken for {key!r} (rot_k={rot_k}): "
            f"{f_r[key]} vs {f_0[key]}"
        )


def test_achromatic_hue_gating_tiny_mask_and_empty_mask_guards() -> None:
    """Verify Hue S>=25 gating (<10 px -> 0.0), empty 3-px erosion guard, and empty mask."""
    optical = OpticalContext(
        focal_length_mm=4.25,
        zoom_multiple=1.0,
        exif_optical_present=False,
        override_applied=False,
        lens_class="wide",
    )
    scale_factor = 768.0 / 4080.0

    # 1. Pure neutral achromatic white silhouette (S == 0 < 25 everywhere inside mask).
    bgr_white = np.full((120, 160, 3), 210, dtype=np.uint8)
    mask_white = np.zeros((120, 160), dtype=np.uint8)
    cv2.ellipse(mask_white, (80, 60), (35, 22), 0.0, 0, 360, 255, -1)
    f_white = extract_mask_gated_features(bgr_white, mask_white, scale_factor, optical)

    assert f_white["hsv_h_sin_mean"] == 0.0
    assert f_white["hsv_h_cos_mean"] == 0.0
    assert f_white["white_plumage_ratio"] == pytest.approx(1.0)
    assert f_white["yellow_down_ratio"] == 0.0
    assert f_white["pink_skin_ratio"] == 0.0
    assert f_white["exif_optical_present"] == 0.0

    # 2. Fewer than 10 saturated pixels inside mask (< 10 threshold -> 0.0 circular Hue).
    bgr_few_sat = bgr_white.copy()
    bgr_few_sat[58:61, 78:81] = (20, 200, 240)  # 9 saturated pixels (< 10)
    f_few_sat = extract_mask_gated_features(bgr_few_sat, mask_white, scale_factor, optical)
    assert f_few_sat["hsv_h_sin_mean"] == 0.0
    assert f_few_sat["hsv_h_cos_mean"] == 0.0

    # 3. >= 10 saturated pixels inside mask -> non-zero unit-circle Hue vector.
    bgr_many_sat = bgr_white.copy()
    bgr_many_sat[56:62, 76:82] = (20, 200, 240)  # 36 saturated pixels (>= 10)
    f_many_sat = extract_mask_gated_features(bgr_many_sat, mask_white, scale_factor, optical)
    hue_norm = math.hypot(f_many_sat["hsv_h_sin_mean"], f_many_sat["hsv_h_cos_mean"])
    assert hue_norm > 0.9

    # 4. Tiny 3x3 mask whose 3-px erosion (7x7 ellipse) is completely empty.
    bgr_tiny = np.full((64, 64, 3), 180, dtype=np.uint8)
    mask_tiny = np.zeros((64, 64), dtype=np.uint8)
    mask_tiny[30:33, 30:33] = 255
    f_tiny = extract_mask_gated_features(bgr_tiny, mask_tiny, scale_factor, optical)
    assert f_tiny["feather_grad_mean"] == 0.0
    assert f_tiny["feather_laplacian_var"] == 0.0
    assert all(math.isfinite(v) for v in f_tiny.values())

    # 5. Completely empty mask returns finite zeros while preserving optical context metadata.
    mask_empty = np.zeros((64, 64), dtype=np.uint8)
    f_empty = extract_mask_gated_features(bgr_tiny, mask_empty, scale_factor, optical)
    assert tuple(f_empty.keys()) == FEATURE_COLUMNS
    assert all(math.isfinite(v) for v in f_empty.values())
    assert f_empty["focal_length_mm"] == pytest.approx(4.25)
    assert f_empty["zoom_multiple"] == pytest.approx(1.0)
    assert f_empty["exif_optical_present"] == 0.0
    for key in FEATURE_COLUMNS:
        if key not in {"focal_length_mm", "zoom_multiple", "exif_optical_present"}:
            assert f_empty[key] == 0.0


def test_optical_calibration_formulas_and_no_focal_zoom_double_counting() -> None:
    """D-09: verify area_px / (zoom_multiple ** 2) and zero focal*zoom double-counting."""
    bgr, mask = _make_textured_bird_frame(center=(384, 173), axes=(100, 60), angle_deg=0.0)
    scale_wide = 768.0 / 4080.0
    scale_uw = 768.0 / 3264.0

    opt_wide = OpticalContext(4.25, 1.0, True, False, "wide")
    opt_uw = OpticalContext(1.65, 0.60, True, False, "ultrawide")
    opt_focal_only_changed = OpticalContext(1.65, 1.0, True, True, "ultrawide")

    f_wide = extract_mask_gated_features(bgr, mask, scale_wide, opt_wide)
    f_uw = extract_mask_gated_features(bgr, mask, scale_uw, opt_uw)
    f_same_zoom_low_focal = extract_mask_gated_features(
        bgr, mask, scale_wide, opt_focal_only_changed
    )

    for feats, z in ((f_wide, 1.0), (f_uw, 0.60)):
        assert feats["zoom_multiple"] == pytest.approx(z)
        assert feats["exif_optical_present"] == 1.0
        assert feats["area_optical_norm"] == pytest.approx(feats["area_px"] / (z ** 2))
        assert feats["sqrt_area_optical_norm"] == pytest.approx(
            math.sqrt(feats["area_optical_norm"])
        )
        assert feats["perim_optical_norm"] == pytest.approx(feats["perimeter_px"] / z)
        assert feats["major_axis_optical_norm"] == pytest.approx(feats["major_axis_px"] / z)
        assert feats["minor_axis_optical_norm"] == pytest.approx(feats["minor_axis_px"] / z)
        assert feats["area_frac_optical_norm"] == pytest.approx(feats["area_frac"] / (z ** 2))

    # Changing focal_length_mm alone (4.25 -> 1.65 at zoom=1.0) MUST NOT alter optical norms,
    # proving zero focal_length_mm * zoom_multiple double-counting.
    assert f_same_zoom_low_focal["focal_length_mm"] == pytest.approx(1.65)
    assert f_same_zoom_low_focal["area_optical_norm"] == f_wide["area_optical_norm"]
    assert f_same_zoom_low_focal["sqrt_area_optical_norm"] == f_wide["sqrt_area_optical_norm"]
    assert f_same_zoom_low_focal["perim_optical_norm"] == f_wide["perim_optical_norm"]
    assert f_same_zoom_low_focal["major_axis_optical_norm"] == f_wide["major_axis_optical_norm"]
    assert f_same_zoom_low_focal["minor_axis_optical_norm"] == f_wide["minor_axis_optical_norm"]
    assert f_same_zoom_low_focal["area_frac_optical_norm"] == f_wide["area_frac_optical_norm"]

    # Physical Wide vs Ultrawide check: when the same bird's linear size in native pixels
    # scales by 0.60 (so native pixel area scales by 0.36) on Ultrawide (zoom=0.60),
    # area_optical_norm = area_px / z^2 collapses Wide / UW to ~1.0!
    bgr_uw_phys, mask_uw_phys = _make_textured_bird_frame(
        center=(384, 173),
        axes=(75, 45),  # 0.75 * work axes = 0.60 native axes after (3264/4080 = 0.80) sensor ratio
        angle_deg=0.0,
    )
    f_uw_phys = extract_mask_gated_features(bgr_uw_phys, mask_uw_phys, scale_uw, opt_uw)
    ratio_wide_over_uw = f_wide["area_optical_norm"] / f_uw_phys["area_optical_norm"]
    assert 0.98 <= ratio_wide_over_uw <= 1.02


def test_extract_mask_gated_features_rejects_invalid_inputs() -> None:
    bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    optical = OpticalContext(4.25, 1.0, True, False, "wide")

    with pytest.raises(ValueError, match="bgr_work"):
        extract_mask_gated_features(bgr.astype(np.float32), mask, 0.5, optical)
    with pytest.raises(ValueError, match="bgr_work"):
        extract_mask_gated_features(np.zeros((64, 64), dtype=np.uint8), mask, 0.5, optical)
    with pytest.raises(ValueError, match="bird_mask"):
        extract_mask_gated_features(bgr, np.zeros((32, 32), dtype=np.uint8), 0.5, optical)
    with pytest.raises(ValueError, match="scale_factor"):
        extract_mask_gated_features(bgr, mask, 0.0, optical)
    with pytest.raises(ValueError, match="scale_factor"):
        extract_mask_gated_features(bgr, mask, math.nan, optical)
    with pytest.raises(ValueError, match="scale_factor"):
        extract_mask_gated_features(bgr, mask, True, optical)


def test_hu_moments_use_binary_image_and_avoid_255_power_attenuation() -> None:
    """CR-01 regression: {0, 255} mask must yield identical Hu moments to {0, 1} binary mask."""
    bgr, mask_255 = _make_textured_bird_frame(center=(320, 173), axes=(95, 48), angle_deg=22.0)
    # Add an asymmetric beak protrusion so Hu moment 3 is non-zero and sensitive to 255^3 scaling.
    cv2.ellipse(mask_255, (425, 190), (22, 12), 10.0, 0, 360, 255, -1)
    optical = OpticalContext(4.25, 1.0, True, False, "wide")

    feats = extract_mask_gated_features(bgr, mask_255, 768.0 / 4080.0, optical)

    bx, by, bw, bh = cv2.boundingRect(mask_255)
    crop_01 = (mask_255[by : by + bh, bx : bx + bw] > 0).astype(np.uint8)
    expected_hu = cv2.HuMoments(cv2.moments(crop_01, binaryImage=True)).ravel()

    for idx, key in enumerate(("log_hu_1", "log_hu_2", "log_hu_3")):
        hu_val = float(expected_hu[idx])
        expected_signed = (
            -math.copysign(1.0, hu_val) * math.log10(abs(hu_val) + 1e-12)
            if hu_val != 0.0
            else 0.0
        )
        assert feats[key] == pytest.approx(expected_signed, abs=1e-9)
    # Without binaryImage=True, log_hu_1 shifted by +2.4065 (~3.1 instead of ~0.7).
    assert abs(feats["log_hu_1"]) < 1.5
    assert abs(feats["log_hu_3"]) < 10.0


def test_degenerate_thin_tiny_and_disconnected_masks_stay_bounded() -> None:
    """WR-02 regression: 1-px collinear, 1x1, and disconnected masks stay bounded."""
    bgr = np.full((80, 80, 3), 180, dtype=np.uint8)
    optical = OpticalContext(4.25, 1.0, True, False, "wide")

    # 1x1 single pixel mask
    m_1x1 = np.zeros((80, 80), dtype=np.uint8)
    m_1x1[40, 40] = 255
    f_1x1 = extract_mask_gated_features(bgr, m_1x1, 0.5, optical)
    assert 0.0 <= f_1x1["extent_rotrect"] <= 1.0
    assert 0.0 <= f_1x1["aspect_ratio_ellipse"] <= 10.0

    # 1x20 collinear strip mask
    m_line = np.zeros((80, 80), dtype=np.uint8)
    m_line[40, 20:40] = 255
    f_line = extract_mask_gated_features(bgr, m_line, 0.5, optical)
    assert 0.0 <= f_line["extent_rotrect"] <= 1.0
    assert f_line["aspect_ratio_ellipse"] <= 25.0

    # Two disconnected 20x20 squares
    m_two = np.zeros((80, 80), dtype=np.uint8)
    m_two[10:30, 10:30] = 255
    m_two[50:70, 50:70] = 255
    f_two = extract_mask_gated_features(bgr, m_two, 0.5, optical)
    assert 0.0 <= f_two["extent_rotrect"] <= 1.0


def test_exif_focal_bounds_and_pdi_config_float_normalization() -> None:
    """CR-02 & WR-04 regressions: EXIF focal bounds [0.5, 200] & PDIConfig hash normalization."""
    from backend.app.pdi.config import PDIConfig
    from backend.app.pdi.io_exif import ExifMetadata, _parse_focal_length, resolve_optical_context

    assert _parse_focal_length(999_999_999.0) is None
    assert _parse_focal_length(1e-7) is None
    assert _parse_focal_length(0.49) is None
    assert _parse_focal_length(200.1) is None
    assert _parse_focal_length(4.25) == pytest.approx(4.25)

    bad_meta = ExifMetadata(
        raw_w=100,
        raw_h=100,
        oriented_w=100,
        oriented_h=100,
        exif_orientation=1,
        focal_length_mm=1e200,
        zoom_multiple=100.0,
        has_exif_optical=True,
    )
    resolved = resolve_optical_context(bad_meta)
    assert resolved.focal_length_mm == pytest.approx(4.25)
    assert resolved.zoom_multiple == pytest.approx(1.0)
    assert resolved.exif_optical_present is False

    assert PDIConfig(clahe_clip_limit=2).config_hash() == PDIConfig(
        clahe_clip_limit=2.0
    ).config_hash()
    with pytest.raises(ValueError, match="watershed_h_rel"):
        PDIConfig(watershed_h_rel=0.01)
    with pytest.raises(ValueError, match="watershed_min_distance_rel"):
        PDIConfig(watershed_min_distance_rel=5.0)
    with pytest.raises(ValueError, match="watershed_min_area_ratio"):
        PDIConfig(watershed_min_area_ratio=0.0)
