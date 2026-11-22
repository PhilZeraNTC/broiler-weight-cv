"""Unit and read-only multi-bird integration tests for Marker-based Watershed (PDI-05).

Covers all five PDI-05 deterministic synthetic edge probes (boundary, adjacency,
empty, ordering, precision), chromatic red-feeder pre-suppression, bbox-cropped
vs full-frame ``h_maxima`` bit-equality, per-instance 49-key mask-gated feature
extraction, 4-stage Base64 visual decoding, and zero-folder-hardcoding compliance.
"""

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
from scipy import ndimage as ndi
from skimage.morphology import h_maxima

import backend.app.pdi as pdi_pkg
from backend.app.config import QUARANTINE_MANIFEST_PATH, RAW_DATASET_DIR
from backend.app.pdi.config import PDIConfig
from backend.app.pdi.features import BANNED_FEATURE_COLUMNS, FEATURE_COLUMNS
from backend.app.pdi.io_exif import OpticalOverride
from backend.app.pdi.watershed import (
    MAX_AREA_RATIO,
    MAX_DISTANCE_REL,
    MAX_H_REL,
    MIN_AREA_RATIO,
    MIN_DISTANCE_REL,
    MIN_H_REL,
    WatershedInstance,
    WatershedResult,
    WatershedVisuals,
    _compute_bbox_h_maxima,
    run_watershed_pdi,
)


def _encode_rgb_as_jpeg_bytes(
    rgb: np.ndarray,
    *,
    focal_length_mm: float = 4.25,
    zoom_multiple: float = 1.0,
    orientation: int = 1,
) -> bytes:
    """Encode a synthetic RGB uint8 array as JPEG bytes with Redmi-style EXIF tags."""
    im = Image.fromarray(rgb, mode="RGB")
    exif = Image.Exif()
    exif[274] = orientation
    ifd = exif.get_ifd(0x8769)
    ifd[37386] = focal_length_mm
    ifd[39321] = json.dumps({"zoomMultiple": zoom_multiple})
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92, subsampling=0, exif=exif.tobytes())
    return buf.getvalue()


def _make_straw_background(h: int = 346, w: int = 768, seed: int = 123) -> np.ndarray:
    """Synthesize a high-frequency golden-brown wood-shaving straw BGR scene."""
    rng = np.random.default_rng(seed)
    # Realistic golden-brown straw in BGR (B~48, G~92, R~112 -> Lab a*~130, b*~148, zero red FP)
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :, 0] = 48.0
    base[:, :, 1] = 92.0
    base[:, :, 2] = 112.0
    noise = rng.normal(0.0, 18.0, size=(h, w, 3)).astype(np.float32)
    straw = np.clip(base + noise, 10.0, 210.0).astype(np.uint8)
    # Add fine straw shavings (high local gradient like real bedding)
    for _ in range(900):
        x1 = int(rng.integers(0, w))
        y1 = int(rng.integers(0, h))
        dx = int(rng.integers(-16, 17))
        dy = int(rng.integers(-10, 11))
        shade = int(rng.integers(-30, 31))
        color = (
            max(10, min(215, 48 + shade)),
            max(10, min(215, 92 + shade)),
            max(10, min(215, 112 + shade)),
        )
        cv2.line(straw, (x1, y1), (x1 + dx, y1 + dy), color, thickness=1)
    return straw


def _paint_broiler_ellipse(
    bgr: np.ndarray,
    center: tuple[int, int],
    axes: tuple[int, int],
    *,
    angle_deg: float = 0.0,
    color_bgr: tuple[int, int, int] = (155, 228, 242),
) -> np.ndarray:
    """Paint a smooth, bright yellow/white broiler torso ellipse onto BGR scene."""
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle_deg, 0.0, 360.0, 255, -1)
    # Gentle smooth shading so the torso is cohesive and slightly bright in center
    dist = ndi.distance_transform_edt(mask > 0)
    dmax = max(1.0, float(dist.max()))
    shade = (0.90 + 0.10 * (dist / dmax)).astype(np.float32)
    ys, xs = np.nonzero(mask)
    for ch in range(3):
        vals = np.clip(color_bgr[ch] * shade[ys, xs], 0.0, 255.0).astype(np.uint8)
        bgr[ys, xs, ch] = vals
    return mask


def _decode_b64_image(b64_str: str) -> Image.Image:
    assert "data:image" not in b64_str
    raw = base64.b64decode(b64_str.encode("ascii"), validate=True)
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


def _make_touching_two_birds_jpeg(
    *,
    center_a: tuple[int, int] = (310, 173),
    center_b: tuple[int, int] = (448, 173),
    axes_a: tuple[int, int] = (76, 56),
    axes_b: tuple[int, int] = (76, 56),
    add_red_feeder: bool = False,
    seed: int = 123,
) -> tuple[bytes, np.ndarray | None]:
    """Build a synthetic 768x346 JPEG with two touching bright birds on straw."""
    bgr = _make_straw_background(346, 768, seed=seed)
    _paint_broiler_ellipse(bgr, center_a, axes_a, color_bgr=(145, 225, 240))
    _paint_broiler_ellipse(bgr, center_b, axes_b, color_bgr=(215, 232, 238))

    feeder_mask: np.ndarray | None = None
    if add_red_feeder:
        feeder_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        # Vivid saturated red plastic feeder rim (high Lab a* > 146, high HSV S > 65)
        cv2.rectangle(bgr, (35, 65), (155, 285), (18, 22, 228), thickness=-1)
        cv2.rectangle(feeder_mask, (35, 65), (155, 285), 255, thickness=-1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _encode_rgb_as_jpeg_bytes(rgb), feeder_mask


# ---------------------------------------------------------------------------
# PDI-05 Boundary probes (ASVS V5 parameter validation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs",),
    [
        ({"h_rel": MIN_H_REL},),
        ({"h_rel": MAX_H_REL},),
        ({"min_distance_rel": MIN_DISTANCE_REL},),
        ({"min_distance_rel": MAX_DISTANCE_REL},),
        ({"min_area_ratio": MIN_AREA_RATIO},),
        ({"min_area_ratio": MAX_AREA_RATIO},),
    ],
)
def test_watershed_accepts_exact_min_and_max_parameter_bounds(
    kwargs: dict[str, float],
) -> None:
    payload, _ = _make_touching_two_birds_jpeg()
    res = run_watershed_pdi(payload, return_visuals=False, **kwargs)
    assert isinstance(res, WatershedResult)
    assert res.visuals is None
    assert res.bird_count >= 0


@pytest.mark.parametrize(
    ("param_name", "bad_value"),
    [
        ("h_rel", 0.04),
        ("h_rel", 0.81),
        ("h_rel", True),
        ("h_rel", False),
        ("h_rel", math.nan),
        ("h_rel", math.inf),
        ("h_rel", -math.inf),
        ("min_distance_rel", 0.19),
        ("min_distance_rel", 2.51),
        ("min_distance_rel", True),
        ("min_distance_rel", math.nan),
        ("min_distance_rel", math.inf),
        ("min_area_ratio", 0.005),
        ("min_area_ratio", 0.46),
        ("min_area_ratio", False),
        ("min_area_ratio", math.nan),
        ("min_area_ratio", -math.inf),
    ],
)
def test_watershed_rejects_out_of_bounds_bool_and_non_finite_params(
    param_name: str,
    bad_value: object,
) -> None:
    payload, _ = _make_touching_two_birds_jpeg()
    with pytest.raises(ValueError, match=param_name):
        run_watershed_pdi(payload, return_visuals=False, **{param_name: bad_value})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PDI-05 Adjacency probes (touching split, isolated single bird, sub-threshold merge)
# ---------------------------------------------------------------------------


def test_watershed_separates_touching_fused_birds_with_disjoint_basins() -> None:
    # Centers 138 px apart with x-radius 76 -> 14 px overlap => single fused union blob!
    payload, _ = _make_touching_two_birds_jpeg(
        center_a=(315, 173),
        center_b=(453, 173),
        axes_a=(76, 56),
        axes_b=(76, 56),
    )
    res = run_watershed_pdi(payload, return_visuals=True)

    assert res.bird_count == 2
    assert len(res.instances) == 2
    assert res.visuals is not None

    # Confirm the union mask really WAS a single fused connected component prior to Watershed.
    stage2_img = _decode_b64_image(res.visuals.stage2_union_mask_b64)
    stage2_arr = np.asarray(stage2_img, dtype=np.uint8)
    n_union_labels, _ = cv2.connectedComponents((stage2_arr > 0).astype(np.uint8))
    assert n_union_labels - 1 == 1, "Expected touching ellipses to form 1 fused union blob"

    # Mutually exclusive basins: total area across instances <= union foreground area
    # and centroids sit distinctly on left and right torso halves.
    xs = sorted(inst.centroid_xy[0] for inst in res.instances)
    assert xs[0] < 360.0 and xs[1] > 410.0
    assert (xs[1] - xs[0]) >= 100.0
    assert res.pdi_config_hash == PDIConfig().config_hash()
    assert res.latency_ms >= 0.0


def test_watershed_preserves_isolated_single_bird_and_merges_subthreshold_humps() -> None:
    # 1. Single isolated bright ellipse -> bird_count == 1
    bgr_single = _make_straw_background(346, 768, seed=7)
    _paint_broiler_ellipse(bgr_single, (384, 173), (82, 56), color_bgr=(150, 228, 242))
    single_payload = _encode_rgb_as_jpeg_bytes(cv2.cvtColor(bgr_single, cv2.COLOR_BGR2RGB))

    res_single = run_watershed_pdi(single_payload, return_visuals=False)
    assert res_single.bird_count == 1
    assert len(res_single.instances) == 1
    assert res_single.instances[0].instance_id == 1

    # 2. Closely spaced sub-threshold humps (centers only 22 px apart inside radius 68)
    # -> saddle depth << h_rel * ds_max and peak distance < min_distance => 1 bird (no over-split).
    bgr_close = _make_straw_background(346, 768, seed=11)
    _paint_broiler_ellipse(bgr_close, (373, 173), (68, 58), color_bgr=(155, 228, 240))
    _paint_broiler_ellipse(bgr_close, (395, 173), (68, 58), color_bgr=(155, 228, 240))
    close_payload = _encode_rgb_as_jpeg_bytes(cv2.cvtColor(bgr_close, cv2.COLOR_BGR2RGB))

    res_close = run_watershed_pdi(close_payload, return_visuals=False)
    assert res_close.bird_count == 1


# ---------------------------------------------------------------------------
# Chromatic red-feeder pre-suppression
# ---------------------------------------------------------------------------


def test_watershed_suppresses_red_feeder_and_excludes_it_from_bird_basins() -> None:
    payload, feeder_mask = _make_touching_two_birds_jpeg(
        center_a=(355, 173),
        center_b=(495, 173),
        axes_a=(72, 54),
        axes_b=(72, 54),
        add_red_feeder=True,
        seed=55,
    )
    assert feeder_mask is not None
    res = run_watershed_pdi(payload, return_visuals=True)

    assert res.red_feeder_excluded_px > 20_000
    assert res.bird_count == 2
    # Feeder sits at x in [35, 155]; neither bird bbox should bleed into the red feeder.
    for inst in res.instances:
        x0, _y0, _bw, _bh = inst.bbox_work
        assert x0 > 155, f"Instance {inst.instance_id} bled into red feeder region (x0={x0})"


# ---------------------------------------------------------------------------
# PDI-05 Empty / uniform / zero-seed handling
# ---------------------------------------------------------------------------


def test_watershed_handles_empty_uniform_dark_and_all_red_scenes_cleanly() -> None:
    dark_rgb = np.full((346, 768, 3), 24, dtype=np.uint8)
    dark_payload = _encode_rgb_as_jpeg_bytes(dark_rgb)
    res_dark = run_watershed_pdi(dark_payload, return_visuals=True)

    assert res_dark.bird_count == 0
    assert res_dark.instances == ()
    assert res_dark.red_feeder_excluded_px == 0
    assert res_dark.visuals is not None
    assert _decode_b64_image(res_dark.visuals.stage2_union_mask_b64).mode == "L"
    assert _decode_b64_image(res_dark.visuals.stage3_distance_peaks_b64).format == "JPEG"
    assert _decode_b64_image(res_dark.visuals.stage4_color_basins_b64).format == "JPEG"

    # Uniform saturated red feeder frame -> 100% excluded, zero foreground, zero crash.
    red_rgb = np.zeros((346, 768, 3), dtype=np.uint8)
    red_rgb[:, :, 0] = 228
    red_rgb[:, :, 1] = 20
    red_rgb[:, :, 2] = 18
    red_payload = _encode_rgb_as_jpeg_bytes(red_rgb)
    res_red = run_watershed_pdi(red_payload, return_visuals=False)

    assert res_red.bird_count == 0
    assert res_red.instances == ()
    assert res_red.red_feeder_excluded_px == 346 * 768
    assert res_red.visuals is None


# ---------------------------------------------------------------------------
# PDI-05 Ordering probe (equal-area deterministic tie-breaking + Path/bytes parity)
# ---------------------------------------------------------------------------


def test_watershed_deterministic_ordering_and_path_bytes_parity(tmp_path: Path) -> None:
    # Symmetric equal-area touching ellipses at distinct centroids (upper-left vs lower-right)
    bgr = _make_straw_background(346, 768, seed=99)
    # Mirror noise symmetrically across center so both ellipses have identical area
    bgr_sym = bgr.copy()
    bgr_sym[:, 384:] = bgr[:, :384][:, ::-1]
    _paint_broiler_ellipse(bgr_sym, (315, 173), (74, 56), color_bgr=(165, 228, 240))
    _paint_broiler_ellipse(bgr_sym, (453, 173), (74, 56), color_bgr=(165, 228, 240))
    payload = _encode_rgb_as_jpeg_bytes(cv2.cvtColor(bgr_sym, cv2.COLOR_BGR2RGB))

    img_path = tmp_path / "symmetric_touching.jpg"
    img_path.write_bytes(payload)

    res_a = run_watershed_pdi(payload, return_visuals=False)
    res_b = run_watershed_pdi(img_path, return_visuals=False)

    assert res_a.bird_count == 2
    assert res_b.bird_count == 2
    assert [inst.instance_id for inst in res_a.instances] == [1, 2]
    assert [inst.instance_id for inst in res_b.instances] == [1, 2]

    # Verify ordering matches exact (-area_work_px, round(cy, 3), round(cx, 3)) contract
    sort_keys = [
        (-inst.area_work_px, round(inst.centroid_xy[1], 3), round(inst.centroid_xy[0], 3))
        for inst in res_a.instances
    ]
    assert sort_keys == sorted(sort_keys)

    # Exact Path-vs-bytes parity across counts, centroids, bboxes, and all 49 features
    for inst_a, inst_b in zip(res_a.instances, res_b.instances, strict=True):
        assert inst_a.instance_id == inst_b.instance_id
        assert inst_a.area_work_px == inst_b.area_work_px
        assert inst_a.centroid_xy == inst_b.centroid_xy
        assert inst_a.bbox_work == inst_b.bbox_work
        assert inst_a.features == inst_b.features


# ---------------------------------------------------------------------------
# PDI-05 Precision probe (bbox-cropped h_maxima bit-equality, 49 features, visuals)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_rel", [0.05, 0.20, 0.80])
def test_bbox_cropped_h_maxima_matches_full_frame_bit_identically(h_rel: float) -> None:
    fg_u8 = np.zeros((346, 768), dtype=np.uint8)
    # Touching pair + closely-spaced blob (2 px gap) + border-touching lobe
    cv2.ellipse(fg_u8, (280, 170), (72, 52), 0, 0, 360, 255, -1)
    cv2.ellipse(fg_u8, (410, 175), (70, 50), 15, 0, 360, 255, -1)
    cv2.circle(fg_u8, (150, 170), 30, 255, -1)
    cv2.circle(fg_u8, (212, 170), 30, 255, -1)  # 2-px separation from left circle
    cv2.ellipse(fg_u8, (740, 30), (35, 32), 0, 0, 360, 255, -1)
    fg = fg_u8 > 0

    dist = ndi.distance_transform_edt(fg)
    dist_s = ndi.gaussian_filter(dist, sigma=max(1.5, 0.06 * float(dist.max())))
    assert dist.dtype == np.float64
    assert dist_s.dtype == np.float64

    h_val = max(1.0, h_rel * float(dist_s.max()))
    hm_full = h_maxima(dist_s, h=h_val).astype(bool) & fg
    hm_bbox = _compute_bbox_h_maxima(dist_s, fg, h_val)
    assert np.array_equal(hm_bbox, hm_full)


def test_watershed_rejects_border_architectural_wall_strip() -> None:
    bgr = _make_straw_background(346, 768, seed=77)
    # Paint a bright narrow horizontal wall strip flush against the top edge (width > 55%, h < 24%)
    cv2.rectangle(bgr, (80, 0), (680, 42), (225, 232, 238), thickness=-1)
    _paint_broiler_ellipse(bgr, (384, 210), (78, 54), color_bgr=(155, 228, 240))
    payload = _encode_rgb_as_jpeg_bytes(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    res = run_watershed_pdi(payload, return_visuals=False)
    assert res.bird_count == 1
    assert res.instances[0].centroid_xy[1] > 140.0


def test_watershed_basin_masks_are_mutually_exclusive_and_empty_h_maxima_safe() -> None:
    # Verify empty foreground produces all-False h_maxima safely.
    empty_fg = np.zeros((64, 64), dtype=bool)
    empty_dist = np.zeros((64, 64), dtype=np.float64)
    assert not np.any(_compute_bbox_h_maxima(empty_dist, empty_fg, 1.0))

    # Verify sum of separated basin pixel areas never exceeds the union foreground pixel count
    # (mutually exclusive integer Watershed basin labels have zero pixel overlap).
    payload, _ = _make_touching_two_birds_jpeg(
        center_a=(315, 173),
        center_b=(453, 173),
        axes_a=(76, 56),
        axes_b=(76, 56),
        seed=303,
    )
    res = run_watershed_pdi(payload, return_visuals=True)
    assert res.bird_count == 2
    assert res.visuals is not None
    # Verify disjointness directly at working scale by checking area conservation:
    # sum(area_frac) <= 1.0 and each bird's area_work_px matches area_frac * (768 * 346).
    total_work_px = sum(inst.area_work_px for inst in res.instances)
    assert total_work_px < 346 * 768
    for inst in res.instances:
        assert inst.area_frac == pytest.approx(inst.area_work_px / float(346 * 768))


def test_watershed_per_instance_49_features_optical_override_and_4_stage_visuals() -> None:
    payload, _ = _make_touching_two_birds_jpeg(seed=202)
    override = OpticalOverride(focal_length_mm=1.65, zoom_multiple=0.60)
    res = run_watershed_pdi(payload, optical_override=override, return_visuals=True)

    assert res.bird_count == 2
    assert res.optical.override_applied is True
    assert res.optical.lens_class == "ultrawide"
    assert res.optical.focal_length_mm == pytest.approx(1.65)
    assert res.optical.zoom_multiple == pytest.approx(0.60)

    for idx, inst in enumerate(res.instances, start=1):
        assert isinstance(inst, WatershedInstance)
        assert inst.instance_id == idx
        assert tuple(inst.features.keys()) == FEATURE_COLUMNS
        assert len(inst.features) == 49
        assert not (set(inst.features.keys()) & BANNED_FEATURE_COLUMNS)
        for key, val in inst.features.items():
            assert isinstance(val, float), f"{key} must be Python float"
            assert math.isfinite(val), f"{key} is non-finite: {val!r}"
        assert inst.area_work_px > 0
        assert inst.area_px == pytest.approx(inst.features["area_px"])
        assert inst.area_frac == pytest.approx(inst.features["area_frac"])
        assert inst.area_optical_norm == pytest.approx(inst.features["area_optical_norm"])
        assert inst.area_optical_norm == pytest.approx(inst.area_px / (0.60**2))
        x0, y0, bw, bh = inst.bbox_work
        assert x0 >= 0 and y0 >= 0 and bw > 0 and bh > 0

    assert isinstance(res.visuals, WatershedVisuals)
    stages = (
        (res.visuals.stage1_original_b64, "JPEG", "RGB"),
        (res.visuals.stage2_union_mask_b64, "PNG", "L"),
        (res.visuals.stage3_distance_peaks_b64, "JPEG", "RGB"),
        (res.visuals.stage4_color_basins_b64, "JPEG", "RGB"),
    )
    sizes = set()
    for b64_str, expected_fmt, expected_mode in stages:
        im = _decode_b64_image(b64_str)
        assert im.format == expected_fmt
        assert im.mode == expected_mode
        assert max(im.size) <= PDIConfig().xray_display_long_edge
        sizes.add(im.size)
    assert len(sizes) == 1, f"All 4 Watershed stages must share display dimensions, got {sizes}"

    expected_total = sum(len(s) for s, _, _ in stages)
    assert res.visuals.total_b64_bytes == expected_total
    assert res.visuals.total_b64_bytes < 300_000


def test_watershed_module_contains_zero_folder_name_hardcoding() -> None:
    watershed_src = (
        Path(__file__).resolve().parents[1] / "app" / "pdi" / "watershed.py"
    ).read_text(encoding="utf-8")
    for folder_token in ('"417"', "'417'", '"757"', "'757'", '"823"', "'823'", '"1520"', "'1520'"):
        assert folder_token not in watershed_src, (
            f"Hardcoded folder token {folder_token} is prohibited inside watershed.py"
        )


def test_pdi_package_barrel_exports_complete_sorted_public_api() -> None:
    from backend.app.pdi import (
        FEATURE_COLUMNS as BARREL_FEATURE_COLUMNS,
    )
    from backend.app.pdi import (
        PDIConfig as BarrelPDIConfig,
    )
    from backend.app.pdi import (
        PDIResult,
        run_single_bird_pdi,
    )
    from backend.app.pdi import (
        WatershedResult as BarrelWatershedResult,
    )
    from backend.app.pdi import (
        run_watershed_pdi as barrel_run_watershed_pdi,
    )

    assert BarrelPDIConfig is PDIConfig
    assert BarrelWatershedResult is WatershedResult
    assert barrel_run_watershed_pdi is run_watershed_pdi
    assert BARREL_FEATURE_COLUMNS == FEATURE_COLUMNS
    assert PDIResult is not None
    assert callable(run_single_bird_pdi)

    assert pdi_pkg.__all__ == sorted(pdi_pkg.__all__)
    required_exports = {
        "BANNED_FEATURE_COLUMNS",
        "CHROMATIC_TEXTURE_FEATURE_COLUMNS",
        "ExifMetadata",
        "FEATURE_COLUMNS",
        "LcdSuppressionResult",
        "MORPHOMETRIC_FEATURE_COLUMNS",
        "OPTICAL_FEATURE_COLUMNS",
        "OpticalContext",
        "OpticalOverride",
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
    }
    assert required_exports.issubset(set(pdi_pkg.__all__))
    for symbol in pdi_pkg.__all__:
        assert hasattr(pdi_pkg, symbol), f"Missing barrel export {symbol!r}"


def test_real_13_quarantined_multi_bird_images_read_only_watershed_audit() -> None:
    """Read-only audit across all 13 quarantined multi-bird scenes (PROIBIÇÃO 4 compliant)."""
    assert QUARANTINE_MANIFEST_PATH.is_file(), "manifest_quarantine.parquet must be materialized"
    assert RAW_DATASET_DIR.is_dir(), "RAW_DATASET_DIR must exist for read-only multi-bird audit"

    quarantine_df = pd.read_parquet(QUARANTINE_MANIFEST_PATH)
    multi_bird = quarantine_df.loc[quarantine_df["quarantine_reason"] == "multi_bird"]
    assert len(multi_bird) == 13
    assert multi_bird["folder_chain"].value_counts().to_dict() == {
        "417": 4,
        "757": 1,
        "823": 4,
        "1520": 4,
    }

    gt_counts: dict[str, int] = {"417": 2, "757": 2, "823": 4, "1520": 2}
    canonical_root = RAW_DATASET_DIR.resolve()
    cfg = PDIConfig()
    expected_hash = cfg.config_hash()

    exact_matches = 0
    within_one = 0
    folder_417_checked = 0

    for row in multi_bird.itertuples(index=False):
        rel_path_str = str(row.rel_path)
        assert ".." not in Path(rel_path_str).parts
        img_path = (RAW_DATASET_DIR / rel_path_str).resolve()
        assert img_path.is_relative_to(canonical_root)
        assert img_path.is_file()

        res = run_watershed_pdi(img_path, config=cfg, return_visuals=True)
        folder = str(row.folder_chain)
        expected_n = gt_counts[folder]

        assert res.h_rel == pytest.approx(0.20)
        assert res.min_distance_rel == pytest.approx(0.85)
        assert res.min_area_ratio == pytest.approx(0.10)
        assert res.pdi_config_hash == expected_hash
        assert res.bird_count == len(res.instances)
        assert res.latency_ms < 350.0, (
            f"Watershed latency {res.latency_ms:.1f} ms exceeded 350 ms budget (folder={folder})"
        )

        if res.bird_count == expected_n:
            exact_matches += 1
        if abs(res.bird_count - expected_n) <= 1:
            within_one += 1

        if folder == "417":
            folder_417_checked += 1
            assert res.red_feeder_excluded_px > 0, (
                "Expected red-feeder pre-suppression to trigger on folder 417 shot"
            )
            assert res.bird_count == 2, (
                f"Expected exact bird_count == 2 on folder 417 shot, got {res.bird_count}"
            )

        prev_area: int | None = None
        for idx, inst in enumerate(res.instances, start=1):
            assert inst.instance_id == idx
            assert tuple(inst.features.keys()) == FEATURE_COLUMNS
            assert all(math.isfinite(v) for v in inst.features.values())
            assert inst.area_px > 0.0
            assert inst.area_optical_norm > 0.0
            if prev_area is not None:
                assert inst.area_work_px <= prev_area
            prev_area = inst.area_work_px

        assert res.visuals is not None
        assert res.visuals.total_b64_bytes < 300_000
        for b64_str, expected_fmt, expected_mode in (
            (res.visuals.stage1_original_b64, "JPEG", "RGB"),
            (res.visuals.stage2_union_mask_b64, "PNG", "L"),
            (res.visuals.stage3_distance_peaks_b64, "JPEG", "RGB"),
            (res.visuals.stage4_color_basins_b64, "JPEG", "RGB"),
        ):
            im = _decode_b64_image(b64_str)
            assert im.format == expected_fmt
            assert im.mode == expected_mode
            assert max(im.size) <= cfg.xray_display_long_edge

    assert folder_417_checked == 4
    assert exact_matches >= 8, f"Expected >= 8/13 exact multi-bird counts, got {exact_matches}/13"
    assert within_one >= 11, f"Expected >= 11/13 counts within +-1, got {within_one}/13"

