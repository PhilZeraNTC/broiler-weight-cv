"""Unit and edge-case tests for Phase 2 Plan 02-01 (PDIConfig, io_exif, preprocess, LCD, seg)."""

from __future__ import annotations

import ast
import io
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from backend.app import config as app_config
from backend.app.ingestion.decoding import classify_lens as ingestion_classify_lens
from backend.app.pdi import io_exif
from backend.app.pdi.config import PDIConfig
from backend.app.pdi.io_exif import (
    DEFAULT_FOCAL_LENGTH_MM,
    DEFAULT_ZOOM_MULTIPLE,
    ExifMetadata,
    OpticalOverride,
    classify_lens,
    load_working_rgb,
    resolve_optical_context,
)
from backend.app.pdi.lcd_suppression import suppress_lcd_and_platter
from backend.app.pdi.preprocess import preprocess_working_rgb
from backend.tests.conftest import make_synthetic_jpeg_bytes, make_synthetic_png_bytes


def _make_synthetic_scale_scene(
    *,
    width: int = 768,
    height: int = 346,
    lcd_center: tuple[int, int] = (170, 173),
    lcd_size: tuple[int, int] = (72, 34),
    lcd_angle_deg: float = 0.0,
    housing_offset: tuple[int, int] = (0, 0),
    housing_size: tuple[int, int] = (180, 150),
    bird_center: tuple[int, int] = (540, 173),
    bird_axes: tuple[int, int] = (90, 62),
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize a textured straw scene with a rotated SF-400 LCD/housing and warm yellow bird.

    Returns ``(rgb_work, lcd_gt_mask, bird_gt_mask)`` where masks are uint8 ``{0, 255}``.
    """
    rng = np.random.default_rng(seed)
    # Textured brownish-tan straw litter background (high Scharr gradient, warm a/b).
    base = np.zeros((height, width, 3), dtype=np.int16)
    base[:, :, 0] = 135  # R
    base[:, :, 1] = 102  # G
    base[:, :, 2] = 66   # B
    noise = rng.integers(-32, 33, size=(height, width, 1), dtype=np.int16)
    rgb = np.clip(base + noise, 0, 255).astype(np.uint8)
    # Add fine straw streaks to mirror real high-frequency bedding.
    for _ in range(350):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        dx = int(rng.integers(-22, 23))
        dy = int(rng.integers(-22, 23))
        col = (
            int(rng.integers(110, 175)),
            int(rng.integers(85, 135)),
            int(rng.integers(45, 95)),
        )
        cv2.line(rgb, (x1, y1), (x1 + dx, y1 + dy), col, 1)

    # Smooth cool-neutral white SF-400 plastic housing/platter rotated by lcd_angle_deg.
    hx = lcd_center[0] + housing_offset[0]
    hy = lcd_center[1] + housing_offset[1]
    housing_box = cv2.boxPoints(
        (
            (float(hx), float(hy)),
            (float(housing_size[0]), float(housing_size[1])),
            float(lcd_angle_deg),
        )
    )
    cv2.fillConvexPoly(rgb, np.int32(np.round(housing_box)), (226, 230, 234))

    # Dark neutral-chroma rectangular LCD bezel with bright simulated digits inside.
    lcd_gt = np.zeros((height, width), dtype=np.uint8)
    lcd_box = cv2.boxPoints(
        (
            (float(lcd_center[0]), float(lcd_center[1])),
            (float(lcd_size[0]), float(lcd_size[1])),
            float(lcd_angle_deg),
        )
    )
    lcd_poly = np.int32(np.round(lcd_box))
    cv2.fillConvexPoly(rgb, lcd_poly, (48, 52, 50))
    cv2.fillConvexPoly(lcd_gt, lcd_poly, 255)
    # Paint dark 7-segment-style strokes inside the LCD display.
    cv2.putText(
        rgb,
        "1280",
        (max(2, lcd_center[0] - 24), min(height - 4, lcd_center[1] + 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (22, 24, 22),
        1,
        cv2.LINE_AA,
    )

    # Warm yellow feathered broiler chick separated from the scale housing.
    bird_gt = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(rgb, bird_center, bird_axes, 0, 0, 360, (235, 210, 120), -1)
    cv2.ellipse(bird_gt, bird_center, bird_axes, 0, 0, 360, 255, -1)
    # Add mild feather texture inside the bird only.
    feather_noise = rng.integers(-12, 13, size=(height, width, 3), dtype=np.int16)
    rgb = np.where(
        bird_gt[:, :, None] > 0,
        np.clip(rgb.astype(np.int16) + feather_noise, 0, 255).astype(np.uint8),
        rgb,
    )
    return rgb, lcd_gt, bird_gt


def test_pdi_config_hash_determinism_immutability_and_validation() -> None:
    cfg_a = PDIConfig()
    cfg_b = PDIConfig()

    # Frozen + slots=True contract (must not depend on __dict__).
    assert not hasattr(cfg_a, "__dict__")
    with pytest.raises(FrozenInstanceError):
        cfg_a.working_long_edge = 960  # type: ignore[misc]

    h_a = cfg_a.config_hash()
    assert len(h_a) == 16
    assert h_a == h_a.lower()
    assert all(ch in "0123456789abcdef" for ch in h_a)
    assert h_a == cfg_b.config_hash()

    # Any tunable change must alter config_hash().
    assert h_a != PDIConfig(working_long_edge=960).config_hash()
    assert h_a != PDIConfig(bilateral_d=7).config_hash()
    assert h_a != PDIConfig(smoothing_method="median").config_hash()
    assert h_a != PDIConfig(clahe_clip_limit=3.0).config_hash()
    assert h_a != PDIConfig(watershed_h_rel=0.25).config_hash()

    # Invalid parameter rejection (ASVS V5 / T-02-02).
    invalid_kwargs: list[dict[str, object]] = [
        {"working_long_edge": 0},
        {"working_long_edge": 32},
        {"working_long_edge": True},
        {"draft_scale_multiplier": 0.0},
        {"draft_scale_multiplier": -1.0},
        {"draft_scale_multiplier": math.nan},
        {"draft_scale_multiplier": math.inf},
        {"draft_scale_multiplier": True},
        {"smoothing_method": "gaussian"},
        {"bilateral_d": 0},
        {"bilateral_d": False},
        {"median_ksize": 4},
        {"median_ksize": 1},
        {"clahe_clip_limit": 0.0},
        {"clahe_tile_grid": (0, 8)},
        {"clahe_tile_grid": (8,)},
        {"min_valid_area_frac": 0.80, "max_valid_area_frac": 0.72},
        {"min_valid_area_frac": 0.0},
        {"xray_jpeg_quality": 0},
        {"xray_jpeg_quality": 101},
        {"default_focal_length_mm": 0.1},
        {"default_zoom_multiple": 25.0},
    ]
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            PDIConfig(**kwargs)  # type: ignore[arg-type]


def test_io_exif_is_acyclic_and_matches_ingestion_classify_lens() -> None:
    io_exif_path = app_config.REPO_ROOT / "backend" / "app" / "pdi" / "io_exif.py"
    tree = ast.parse(io_exif_path.read_text(encoding="utf-8"), filename=str(io_exif_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("backend.app.ingestion"), (
                    f"Circular import risk: io_exif imports {alias.name!r}"
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("backend.app.ingestion"), (
                f"Circular import risk: io_exif imports from {node.module!r}"
            )

    for probe in (None, math.nan, math.inf, -math.inf, 1.65, 2.99, 3.0, 4.25):
        assert classify_lens(probe) == ingestion_classify_lens(probe)
    assert classify_lens(True) == "unknown"  # type: ignore[arg-type]


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_load_working_rgb_preserves_pre_draft_dims_and_path_bytes_parity(
    tmp_path: Path,
    orientation: int,
) -> None:
    raw_bytes = make_synthetic_jpeg_bytes(
        size=(1600, 720),
        orientation=orientation,
        focal_length_mm=4.25,
        zoom_multiple=1.0,
        draw_corner_marker=True,
    )
    img_path = tmp_path / f"work_orient_{orientation}.jpg"
    img_path.write_bytes(raw_bytes)

    wp = load_working_rgb(img_path, working_long_edge=768, draft_scale_multiplier=1.0)
    wb = load_working_rgb(raw_bytes, working_long_edge=768, draft_scale_multiplier=1.0)

    assert np.array_equal(wp.rgb_work, wb.rgb_work)
    assert wp.metadata == wb.metadata
    assert wp.scale_factor == pytest.approx(wb.scale_factor)
    assert wp.work_w == wb.work_w and wp.work_h == wb.work_h

    expected_w, expected_h = (720, 1600) if orientation in {6, 8} else (1600, 720)
    assert wp.metadata.raw_w == 1600
    assert wp.metadata.raw_h == 720
    assert wp.metadata.oriented_w == expected_w
    assert wp.metadata.oriented_h == expected_h
    assert max(wp.work_w, wp.work_h) == 768
    assert wp.scale_factor == pytest.approx(768.0 / 1600.0)
    assert wp.rgb_work.dtype == np.uint8
    assert wp.rgb_work.shape == (wp.work_h, wp.work_w, 3)


def test_load_working_rgb_and_optical_context_graceful_degradation_and_bounds() -> None:
    # EXIF-stripped JPEG
    stripped_jpeg = make_synthetic_jpeg_bytes(
        size=(120, 60),
        orientation=None,
        focal_length_mm=None,
        zoom_multiple=None,
        draw_corner_marker=False,
    )
    w_jpeg = load_working_rgb(stripped_jpeg, working_long_edge=768)
    assert w_jpeg.rgb_work.shape == (60, 120, 3)
    assert w_jpeg.metadata.has_exif_optical is False
    opt_default = resolve_optical_context(w_jpeg.metadata)
    assert opt_default.focal_length_mm == pytest.approx(DEFAULT_FOCAL_LENGTH_MM)
    assert opt_default.zoom_multiple == pytest.approx(DEFAULT_ZOOM_MULTIPLE)
    assert opt_default.exif_optical_present is False
    assert opt_default.override_applied is False
    assert opt_default.lens_class == "wide"

    # RGBA PNG & grayscale PNG normalization
    rgba_png = make_synthetic_png_bytes(size=(80, 50), mode="RGBA")
    w_rgba = load_working_rgb(rgba_png, working_long_edge=768)
    assert w_rgba.rgb_work.shape == (50, 80, 3)
    assert w_rgba.rgb_work.dtype == np.uint8

    gray_buf = io.BytesIO()
    Image.new("L", (64, 48), color=140).save(gray_buf, format="PNG")
    w_gray = load_working_rgb(gray_buf.getvalue(), working_long_edge=768)
    assert w_gray.rgb_work.shape == (48, 64, 3)
    assert w_gray.rgb_work.dtype == np.uint8

    # Valid override precedence (override -> EXIF -> default)
    meta_wide = ExifMetadata(
        raw_w=4080,
        raw_h=1840,
        oriented_w=1840,
        oriented_h=4080,
        exif_orientation=6,
        focal_length_mm=4.25,
        zoom_multiple=1.0,
        has_exif_optical=True,
    )
    opt_over = resolve_optical_context(
        meta_wide,
        OpticalOverride(focal_length_mm=1.65, zoom_multiple=0.6),
    )
    assert opt_over.focal_length_mm == pytest.approx(1.65)
    assert opt_over.zoom_multiple == pytest.approx(0.6)
    assert opt_over.exif_optical_present is True
    assert opt_over.override_applied is True
    assert opt_over.lens_class == "ultrawide"

    # Boundary values accepted
    opt_bounds_low = resolve_optical_context(
        meta_wide,
        OpticalOverride(focal_length_mm=0.5, zoom_multiple=0.1),
    )
    assert opt_bounds_low.focal_length_mm == pytest.approx(0.5)
    assert opt_bounds_low.zoom_multiple == pytest.approx(0.1)
    opt_bounds_high = resolve_optical_context(
        meta_wide,
        OpticalOverride(focal_length_mm=200.0, zoom_multiple=20.0),
    )
    assert opt_bounds_high.focal_length_mm == pytest.approx(200.0)
    assert opt_bounds_high.zoom_multiple == pytest.approx(20.0)

    # ASVS V5 invalid override rejection (T-02-02)
    invalid_overrides = [
        OpticalOverride(focal_length_mm=True),  # type: ignore[arg-type]
        OpticalOverride(zoom_multiple=False),  # type: ignore[arg-type]
        OpticalOverride(focal_length_mm=math.nan),
        OpticalOverride(focal_length_mm=math.inf),
        OpticalOverride(zoom_multiple=math.nan),
        OpticalOverride(zoom_multiple=-math.inf),
        OpticalOverride(focal_length_mm=0.2),
        OpticalOverride(focal_length_mm=250.0),
        OpticalOverride(zoom_multiple=0.05),
        OpticalOverride(zoom_multiple=25.0),
    ]
    for bad in invalid_overrides:
        with pytest.raises(ValueError):
            resolve_optical_context(meta_wide, bad)

    # Verify io_exif did not weaken Pillow's decompression bomb guard
    assert io_exif is not None
    assert Image.MAX_IMAGE_PIXELS is not None and Image.MAX_IMAGE_PIXELS > 0


@pytest.mark.parametrize("method", ["bilateral", "median"])
def test_preprocess_working_rgb_bilateral_and_median_contracts(method: str) -> None:
    rng = np.random.default_rng(7)
    rgb = rng.integers(20, 235, size=(120, 200, 3), dtype=np.uint8)
    cfg = PDIConfig(smoothing_method=method)  # type: ignore[arg-type]
    frame = preprocess_working_rgb(rgb, cfg)

    assert frame.bgr_work.shape == (120, 200, 3)
    assert frame.bgr_smooth.shape == (120, 200, 3)
    assert frame.bgr_clahe.shape == (120, 200, 3)
    assert frame.lab.shape == (120, 200, 3)
    assert frame.l_clahe.shape == (120, 200)
    assert frame.hsv.shape == (120, 200, 3)
    assert frame.grad_l.shape == (120, 200)
    assert frame.grad_l_clahe_blur.shape == (120, 200)

    assert frame.bgr_clahe.dtype == np.uint8
    assert frame.lab.dtype == np.uint8
    assert frame.hsv.dtype == np.uint8
    assert int(frame.hsv[:, :, 0].max()) <= 179
    assert frame.grad_l.dtype == np.float32
    assert frame.grad_l_clahe_blur.dtype == np.float32

    # RGB->BGR channel order sanity
    assert np.array_equal(frame.bgr_work[:, :, 0], rgb[:, :, 2])
    assert np.array_equal(frame.bgr_work[:, :, 2], rgb[:, :, 0])


def test_preprocess_working_rgb_rejects_invalid_inputs() -> None:
    cfg = PDIConfig()
    with pytest.raises(ValueError):
        preprocess_working_rgb(np.zeros((64, 64), dtype=np.uint8), cfg)
    with pytest.raises(ValueError):
        preprocess_working_rgb(np.zeros((64, 64, 4), dtype=np.uint8), cfg)
    with pytest.raises(ValueError):
        preprocess_working_rgb(np.zeros((64, 64, 3), dtype=np.float32), cfg)


@pytest.mark.parametrize(
    ("angle_deg", "rot_k"),
    [
        (0.0, 0),
        (90.0, 1),
        (180.0, 2),
        (270.0, 3),
        (35.0, 0),
    ],
)
def test_suppress_lcd_and_platter_is_rotation_invariant_and_spares_bird(
    angle_deg: float,
    rot_k: int,
) -> None:
    cfg = PDIConfig()
    rgb, lcd_gt, bird_gt = _make_synthetic_scale_scene(
        lcd_angle_deg=angle_deg if rot_k == 0 else 0.0,
    )
    if rot_k > 0:
        rgb = np.ascontiguousarray(np.rot90(rgb, k=rot_k))
        lcd_gt = np.ascontiguousarray(np.rot90(lcd_gt, k=rot_k))
        bird_gt = np.ascontiguousarray(np.rot90(bird_gt, k=rot_k))

    frame = preprocess_working_rgb(rgb, cfg)
    res = suppress_lcd_and_platter(frame, cfg)

    assert res.exclusion_mask.shape == rgb.shape[:2]
    assert res.exclusion_mask.dtype == np.uint8
    assert set(np.unique(res.exclusion_mask).tolist()).issubset({0, 255})
    assert res.lcd_core_count >= 1
    assert res.lcd_excluded_px == int(np.count_nonzero(res.exclusion_mask))

    # Entire dark LCD rectangle must be covered by exclusion_mask (bezel + connected housing).
    lcd_pixels = int(np.count_nonzero(lcd_gt))
    lcd_covered = int(np.count_nonzero((res.exclusion_mask > 0) & (lcd_gt > 0)))
    assert lcd_covered / lcd_pixels >= 0.98
    # Connected smooth plastic housing must also expand exclusion well beyond LCD core alone.
    assert res.lcd_excluded_px > int(lcd_pixels * 2.5)
    # Separated warm yellow bird must remain untouched (0 px false suppression).
    assert int(np.count_nonzero((res.exclusion_mask > 0) & (bird_gt > 0))) == 0


def test_suppress_lcd_and_platter_has_zero_false_positives_on_clean_white_bucket() -> None:
    cfg = PDIConfig()
    h, w = 346, 768
    # Smooth white bucket background + warm yellow chick with zero dark LCD rectangles.
    rgb = np.full((h, w, 3), (236, 236, 238), dtype=np.uint8)
    cv2.ellipse(rgb, (w // 2, h // 2), (95, 70), 0, 0, 360, (238, 212, 118), -1)

    frame = preprocess_working_rgb(rgb, cfg)
    res = suppress_lcd_and_platter(frame, cfg)

    assert res.lcd_core_count == 0
    assert res.lcd_excluded_px == 0
    assert int(np.count_nonzero(res.exclusion_mask)) == 0
