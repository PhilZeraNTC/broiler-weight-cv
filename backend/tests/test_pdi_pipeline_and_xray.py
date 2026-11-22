"""End-to-end single-bird PDI pipeline, 4-stage X-Ray, anti-LCD & latency tests (Plan 02-02)."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from backend.app.config import CLEAN_MANIFEST_PATH, RAW_DATASET_DIR
from backend.app.pdi.config import PDIConfig
from backend.app.pdi.features import FEATURE_COLUMNS
from backend.app.pdi.io_exif import OpticalOverride
from backend.app.pdi.pipeline import PDIResult, run_single_bird_pdi
from backend.app.pdi.preprocess import preprocess_working_rgb
from backend.app.pdi.segmentation import segment_single_bird
from backend.app.pdi.xray import XRayArtifacts, _draw_stage4_overlay, render_xray_artifacts
from backend.tests.conftest import make_synthetic_png_bytes


def _encode_rgb_array_to_jpeg(
    rgb: np.ndarray,
    *,
    orientation: int | None = None,
    focal_length_mm: float | None = None,
    zoom_multiple: float | None = None,
    quality: int = 92,
) -> bytes:
    """Encode an RGB uint8 NumPy array to JPEG bytes with optional EXIF/Xiaomi tags."""
    im = Image.fromarray(rgb, mode="RGB")
    exif = Image.Exif()
    has_exif = False
    if orientation is not None:
        exif[274] = orientation
        has_exif = True
    if focal_length_mm is not None or zoom_multiple is not None:
        ifd = exif.get_ifd(0x8769)
        if focal_length_mm is not None:
            ifd[37386] = focal_length_mm
        if zoom_multiple is not None:
            ifd[39321] = json.dumps({"zoomMultiple": zoom_multiple})
        has_exif = True

    buf = io.BytesIO()
    if has_exif:
        im.save(buf, format="JPEG", quality=quality, exif=exif.tobytes())
    else:
        im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_straw_bird_scene_rgb(
    height: int = 720,
    width: int = 1600,
    *,
    include_sf400_lcd: bool = False,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Synthesize a straw-litter scene with a bright broiler bird and optional SF-400 LCD."""
    rng = np.random.default_rng(seed)
    # High-frequency warm brown wood-shaving straw litter (RGB).
    base = np.zeros((height, width, 3), dtype=np.int16)
    base[:, :, 0] = 148  # R
    base[:, :, 1] = 108  # G
    base[:, :, 2] = 66   # B
    straw_noise = rng.integers(-42, 43, size=(height, width, 3), dtype=np.int16)
    rgb = np.clip(base + straw_noise, 0, 255).astype(np.uint8)
    # Add coarse high-frequency wood shaving streaks.
    for _ in range(320):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        x2 = int(np.clip(x1 + rng.integers(-45, 46), 0, width - 1))
        y2 = int(np.clip(y1 + rng.integers(-22, 23), 0, height - 1))
        shade = int(rng.integers(75, 165))
        cv2.line(rgb, (x1, y1), (x2, y2), (shade + 25, shade, max(20, shade - 35)), 2)

    lcd_gt_mask: np.ndarray | None = None
    if include_sf400_lcd:
        lcd_gt_mask = np.zeros((height, width), dtype=np.uint8)
        # Smooth cool-white plastic SF-400 scale housing near lower-left of bird.
        hx0, hy0, hx1, hy1 = int(width * 0.08), int(height * 0.56), int(width * 0.40), int(
            height * 0.90
        )
        rgb[hy0:hy1, hx0:hx1] = (224, 226, 232)
        # Dark neutral rectangular LCD bezel inside the SF-400 housing.
        lx0, ly0, lx1, ly1 = int(width * 0.12), int(height * 0.65), int(width * 0.30), int(
            height * 0.80
        )
        rgb[ly0:ly1, lx0:lx1] = (28, 30, 32)
        lcd_gt_mask[ly0:ly1, lx0:lx1] = 255
        # High-contrast white 7-segment digits "1850" inside the LCD rectangle.
        cv2.putText(
            rgb,
            "1850",
            (lx0 + 18, ly1 - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (238, 240, 242),
            5,
            cv2.LINE_AA,
        )

    # Bright warm-white broiler silhouette in upper-right/center with subtle feather texture.
    bird_mask = np.zeros((height, width), dtype=np.uint8)
    bird_center = (int(width * 0.58), int(height * 0.45)) if include_sf400_lcd else (
        int(width * 0.50),
        int(height * 0.50),
    )
    bird_axes = (int(width * 0.15), int(height * 0.22))
    cv2.ellipse(bird_mask, bird_center, bird_axes, 20.0, 0, 360, 255, -1)

    feather_noise = rng.integers(-12, 13, size=(height, width, 3), dtype=np.int16)
    bird_fill = np.clip(
        np.array((236, 228, 210), dtype=np.int16) + feather_noise,
        0,
        255,
    ).astype(np.uint8)
    rgb[bird_mask == 255] = bird_fill[bird_mask == 255]
    return rgb, lcd_gt_mask


def _decode_b64_image(b64_payload: str) -> Image.Image:
    assert not b64_payload.startswith("data:"), "X-Ray payloads must be raw RFC 4648 Base64"
    raw = base64.b64decode(b64_payload, validate=True)
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


def test_render_xray_artifacts_decodes_all_four_stages_within_budget_and_axis_alignment() -> None:
    cfg = PDIConfig()
    rgb_big, _ = _make_straw_bird_scene_rgb(height=346, width=768, include_sf400_lcd=False)
    frame = preprocess_working_rgb(rgb_big, cfg)
    seg = segment_single_bird(frame, cfg)
    xray = render_xray_artifacts(frame, seg, cfg)

    assert isinstance(xray, XRayArtifacts)
    assert max(xray.display_w, xray.display_h) == cfg.xray_display_long_edge
    assert xray.display_w == 640
    assert abs((xray.display_w / xray.display_h) - (768.0 / 346.0)) < 0.05

    expected_total = (
        len(xray.stage1_original_b64)
        + len(xray.stage2_clahe_b64)
        + len(xray.stage3_mask_b64)
        + len(xray.stage4_overlay_b64)
    )
    assert xray.total_b64_bytes == expected_total
    assert 0 < xray.total_b64_bytes < 250_000

    im1 = _decode_b64_image(xray.stage1_original_b64)
    im2 = _decode_b64_image(xray.stage2_clahe_b64)
    im3 = _decode_b64_image(xray.stage3_mask_b64)
    im4 = _decode_b64_image(xray.stage4_overlay_b64)

    for im in (im1, im2, im4):
        assert im.format == "JPEG"
        assert im.mode == "RGB"
        assert im.size == (xray.display_w, xray.display_h)

    assert im3.format == "PNG"
    assert im3.mode == "L"
    assert im3.size == (xray.display_w, xray.display_h)
    mask_arr = np.asarray(im3)
    assert set(np.unique(mask_arr).tolist()).issubset({0, 255})
    assert np.count_nonzero(mask_arr == 255) > 0

    # Verify Stage 4 draws orthogonal major (amber) and minor (cyan) axes aligned along angle_deg.
    canvas_bg = np.full((240, 320, 3), 40, dtype=np.uint8)
    syn_mask = np.zeros((240, 320), dtype=np.uint8)
    cv2.ellipse(syn_mask, (160, 120), (80, 34), 32.0, 0, 360, 255, -1)
    cnts, _ = cv2.findContours(syn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    from backend.app.pdi.segmentation import SingleBirdSegmentation, fit_ellipse_safe

    ell, from_rect = fit_ellipse_safe(cnts[0])
    syn_seg = SingleBirdSegmentation(
        bird_mask=syn_mask,
        contour=cnts[0],
        ellipse=ell,
        ellipse_from_min_area_rect=from_rect,
        scene_type="scale_or_litter",
        scene_is_bucket=False,
        lcd_excluded_px=0,
        lcd_core_count=0,
        segmentation_fallback=False,
        mask_area_frac=float(np.count_nonzero(syn_mask)) / float(240 * 320),
    )
    overlay = _draw_stage4_overlay(canvas_bg, syn_seg)
    (cx, cy), (major, minor), angle_deg = ell
    rad = math.radians(angle_deg)
    # Probe 60% along major axis -> warm amber BGR=(0, 165, 255): R >> B.
    mp_x = int(round(cx + 0.60 * (major / 2.0) * math.cos(rad)))
    mp_y = int(round(cy + 0.60 * (major / 2.0) * math.sin(rad)))
    patch_maj = overlay[mp_y - 2 : mp_y + 3, mp_x - 2 : mp_x + 3]
    assert int(patch_maj[:, :, 2].max()) > 200  # High R (amber)
    assert int(patch_maj[:, :, 0].min()) < 60   # Low B (amber)

    # Probe 60% along minor axis -> sky cyan BGR=(255, 215, 0): B >> R.
    np_x = int(round(cx + 0.60 * (minor / 2.0) * math.sin(rad)))
    np_y = int(round(cy - 0.60 * (minor / 2.0) * math.cos(rad)))
    patch_min = overlay[np_y - 2 : np_y + 3, np_x - 2 : np_x + 3]
    assert int(patch_min[:, :, 0].max()) > 200  # High B (cyan)
    assert int(patch_min[:, :, 2].min()) < 60   # Low R (cyan)

    # Small frame (< 640 px) must not be upscaled.
    small_rgb = np.full((120, 200, 3), 120, dtype=np.uint8)
    small_frame = preprocess_working_rgb(small_rgb, cfg)
    small_seg = segment_single_bird(small_frame, cfg)
    small_xray = render_xray_artifacts(small_frame, small_seg, cfg)
    assert (small_xray.display_w, small_xray.display_h) == (200, 120)


def test_run_single_bird_pdi_path_vs_bytes_parity_and_xray_toggle(tmp_path: Path) -> None:
    rgb_scene, _ = _make_straw_bird_scene_rgb(height=920, width=2040, include_sf400_lcd=False)
    # Rotate raw sensor pixels so EXIF orientation=6 transposes cleanly back to landscape.
    raw_bytes = _encode_rgb_array_to_jpeg(
        np.rot90(rgb_scene, k=1),
        orientation=6,
        focal_length_mm=4.25,
        zoom_multiple=1.0,
    )
    img_path = tmp_path / "sample_bird.jpg"
    img_path.write_bytes(raw_bytes)

    cfg = PDIConfig()
    res_path = run_single_bird_pdi(img_path, config=cfg, return_xray=False)
    res_bytes = run_single_bird_pdi(raw_bytes, config=cfg, return_xray=False)
    res_xray = run_single_bird_pdi(raw_bytes, config=cfg, return_xray=True)

    assert isinstance(res_path, PDIResult)
    assert res_path.xray is None
    assert res_bytes.xray is None
    assert res_xray.xray is not None
    assert res_xray.xray.total_b64_bytes < 250_000

    assert res_path.pdi_config_hash == cfg.config_hash()
    assert res_bytes.pdi_config_hash == cfg.config_hash()
    assert res_path.latency_ms > 0.0
    assert res_bytes.latency_ms > 0.0
    assert res_xray.latency_ms > 0.0

    assert res_path.optical == res_bytes.optical == res_xray.optical
    assert res_path.optical.exif_optical_present is True
    assert res_path.optical.override_applied is False
    assert res_path.optical.lens_class == "wide"

    assert tuple(res_path.features.keys()) == FEATURE_COLUMNS
    assert res_path.features == res_bytes.features
    assert res_path.features == res_xray.features
    assert all(math.isfinite(v) for v in res_path.features.values())


def test_run_single_bird_pdi_handles_stripped_jpeg_rgba_grayscale_png_and_override() -> None:
    rgb_scene, _ = _make_straw_bird_scene_rgb(height=280, width=520, include_sf400_lcd=False)
    stripped_jpeg = _encode_rgb_array_to_jpeg(
        rgb_scene,
        orientation=None,
        focal_length_mm=None,
        zoom_multiple=None,
    )
    rgba_png = make_synthetic_png_bytes(size=(240, 160), mode="RGBA")
    gray_buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(rgb_scene, cv2.COLOR_RGB2GRAY), mode="L").save(
        gray_buf, format="PNG"
    )
    gray_png = gray_buf.getvalue()

    for payload in (stripped_jpeg, rgba_png, gray_png):
        res = run_single_bird_pdi(payload, return_xray=True)
        assert res.optical.exif_optical_present is False
        assert res.optical.override_applied is False
        assert res.optical.focal_length_mm == pytest.approx(4.25)
        assert res.optical.zoom_multiple == pytest.approx(1.0)
        assert res.optical.lens_class == "wide"
        assert res.features["exif_optical_present"] == 0.0
        assert tuple(res.features.keys()) == FEATURE_COLUMNS
        assert all(math.isfinite(v) for v in res.features.values())
        assert res.xray is not None

    # Valid OpticalOverride takes precedence over fallback defaults and EXIF.
    override = OpticalOverride(focal_length_mm=1.65, zoom_multiple=0.60)
    res_ovr = run_single_bird_pdi(stripped_jpeg, optical_override=override, return_xray=False)
    assert res_ovr.optical.override_applied is True
    assert res_ovr.optical.exif_optical_present is False
    assert res_ovr.optical.focal_length_mm == pytest.approx(1.65)
    assert res_ovr.optical.zoom_multiple == pytest.approx(0.60)
    assert res_ovr.optical.lens_class == "ultrawide"
    assert res_ovr.features["area_optical_norm"] == pytest.approx(
        res_ovr.features["area_px"] / (0.60 ** 2)
    )

    with pytest.raises(ValueError, match="zoom_multiple"):
        run_single_bird_pdi(stripped_jpeg, optical_override=OpticalOverride(zoom_multiple=0.0))


def test_full_pipeline_sf400_anti_lcd_excludes_digits_completely() -> None:
    """PROIBIÇÃO 2 full-pipeline test: synthetic bird on straw beside SF-400 LCD digits."""
    rgb_scene, lcd_gt_full = _make_straw_bird_scene_rgb(
        height=720,
        width=1600,
        include_sf400_lcd=True,
    )
    assert lcd_gt_full is not None
    raw_bytes = _encode_rgb_array_to_jpeg(
        rgb_scene,
        orientation=1,
        focal_length_mm=4.25,
        zoom_multiple=1.0,
    )

    res = run_single_bird_pdi(raw_bytes, return_xray=True)
    seg = res.segmentation
    work_h, work_w = seg.bird_mask.shape[:2]
    lcd_gt_work = cv2.resize(lcd_gt_full, (work_w, work_h), interpolation=cv2.INTER_NEAREST)

    assert seg.scene_type == "scale_or_litter"
    assert seg.scene_is_bucket is False
    assert seg.lcd_core_count >= 1
    assert seg.lcd_excluded_px > 0
    assert np.count_nonzero(seg.bird_mask & lcd_gt_work) == 0
    assert seg.mask_area_frac >= 0.02
    assert res.xray is not None and res.xray.total_b64_bytes < 250_000


def test_real_native_4080x1840_wide_jpeg_meets_latency_and_payload_sla() -> None:
    """Read-only SLA check on one real 4080x1840 Wide JPEG (PROIBIÇÃO 4: zero filename logs)."""
    if not CLEAN_MANIFEST_PATH.is_file() or not RAW_DATASET_DIR.is_dir():
        pytest.skip("Materialized clean manifest or raw dataset directory unavailable.")

    manifest = pd.read_parquet(CLEAN_MANIFEST_PATH)
    wide_native = manifest.loc[
        (manifest["lens_class"] == "wide")
        & (manifest[["oriented_w", "oriented_h"]].max(axis=1) == 4080)
    ]
    assert not wide_native.empty, "Expected at least one native 4080x1840 Wide row in manifest"

    # Read strictly read-only without printing or asserting raw filename strings (PROIBIÇÃO 4).
    rel_path = str(wide_native.iloc[0]["rel_path"])
    sample_path = (RAW_DATASET_DIR / rel_path).resolve()
    assert sample_path.is_relative_to(RAW_DATASET_DIR.resolve())
    assert sample_path.is_file()
    raw_bytes = sample_path.read_bytes()

    res = run_single_bird_pdi(raw_bytes, return_xray=True)

    assert res.latency_ms < 400.0, f"Latency SLA exceeded: {res.latency_ms:.1f} ms >= 400 ms"
    assert res.xray is not None
    assert res.xray.total_b64_bytes < 250_000, (
        f"X-Ray payload exceeded 250 KB: {res.xray.total_b64_bytes} bytes"
    )
    assert max(res.xray.display_w, res.xray.display_h) <= PDIConfig().xray_display_long_edge
    assert res.optical.exif_optical_present is True
    assert res.optical.lens_class == "wide"
    assert tuple(res.features.keys()) == FEATURE_COLUMNS
    assert all(math.isfinite(v) for v in res.features.values())
    assert res.features["area_px"] > 0.0
    assert res.features["area_optical_norm"] > 0.0
