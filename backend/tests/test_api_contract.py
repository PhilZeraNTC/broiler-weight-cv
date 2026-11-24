"""FastAPI REST endpoint contract tests for AvianScale CV (WEB-01, D-01..D-05).

All API-only fixtures isolate themselves via
``create_app(frontend_dist_dir=tmp_path_factory.mktemp("no_dist"))`` so building
``frontend/dist/index.html`` in later waves never alters API-only routing.
Dedicated SPA fallback tests materialize their own temporary ``dist/`` tree.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from backend.app.api.main import create_app
from backend.app.api.services.inference_service import FEATURES_SUMMARY_KEYS
from backend.app.api.services.preset_service import THUMBNAIL_CACHE_CONTROL, THUMBNAIL_LONG_EDGE
from backend.app.api.services.watershed_service import (
    BASIN_PALETTE_HEX,
    WATERSHED_GROUP_SUM_DISCLAIMER,
)
from backend.app.pdi.config import (
    WATERSHED_MAX_AREA_RATIO,
    WATERSHED_MAX_DISTANCE_REL,
    WATERSHED_MAX_H_REL,
    WATERSHED_MIN_AREA_RATIO,
    WATERSHED_MIN_DISTANCE_REL,
    WATERSHED_MIN_H_REL,
    PDIConfig,
)
from backend.tests.conftest import make_synthetic_jpeg_bytes, make_synthetic_png_bytes


def _open_test_client(app: Any, **kwargs: Any) -> Any:
    """Instantiate FastAPI TestClient while filtering Starlette's httpx2 deprecation notice."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*httpx.*starlette\.testclient.*",
            category=UserWarning,
        )
        from fastapi.testclient import TestClient

    return TestClient(app, **kwargs)


@pytest.fixture(scope="module")
def api_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """Module-scoped TestClient strictly isolated from repository ``frontend/dist/``."""
    no_dist = tmp_path_factory.mktemp("no_dist")
    with _open_test_client(create_app(frontend_dist_dir=no_dist)) as client:
        yield client


def _make_synthetic_webp_bytes(size: tuple[int, int] = (200, 150)) -> bytes:
    """Create a synthetic RGB WEBP image in memory via Pillow."""
    im = Image.new("RGB", size, color=(70, 95, 65))
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=85)
    return buf.getvalue()


def _make_synthetic_two_bird_jpeg_bytes() -> bytes:
    """Synthesize a dark straw-like scene with two bright touching bird blobs for Watershed."""
    rng = np.random.default_rng(7)
    canvas = rng.integers(32, 48, size=(360, 480, 3), dtype=np.uint8)
    cv2.ellipse(canvas, (175, 180), (68, 52), 0.0, 0.0, 360.0, (235, 232, 222), -1)
    cv2.ellipse(canvas, (305, 185), (68, 52), 0.0, 0.0, 360.0, (232, 228, 218), -1)
    im = Image.fromarray(canvas, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _make_uniform_dark_jpeg_bytes(size: tuple[int, int] = (160, 120)) -> bytes:
    """Synthesize a uniform dark image guaranteed to yield 0 Watershed bird basins."""
    im = Image.new("RGB", size, color=(12, 12, 12))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_health_endpoint_contract(api_client: Any) -> None:
    resp = api_client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "avianscale-cv-v1.0.0"
    assert body["winning_model_name"] == "LightGBMLog"
    assert body["winning_ablation_condition"] == "C2_physics_normalized_strict"
    assert body["pdi_config_hash"] == PDIConfig().config_hash()
    assert body["n_features"] == 39
    assert body["single_bird_presets_count"] == 6
    assert body["watershed_presets_count"] == 4
    assert body["classical_pdi_only"] is True


def test_presets_endpoint_contract_and_zero_rel_path(api_client: Any) -> None:
    resp = api_client.get("/api/presets")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pdi_config_hash"] == PDIConfig().config_hash()
    assert body["winning_model_name"] == "LightGBMLog"
    assert body["winning_ablation_condition"] == "C2_physics_normalized_strict"

    sb_presets = body["single_bird_presets"]
    ws_presets = body["watershed_presets"]
    assert len(sb_presets) == 6
    assert len(ws_presets) == 4

    for item in sb_presets:
        assert "rel_path" not in item
        assert item["thumbnail_url"] == f"/api/presets/{item['preset_id']}/thumbnail"
        assert 0 <= item["fold"] <= 4
        assert item["scene_type"] in ("bucket_dorsal", "scale_or_litter")

    for item in ws_presets:
        assert "rel_path" not in item
        assert item["thumbnail_url"] == f"/api/presets/{item['preset_id']}/thumbnail"
        assert item["ground_truth_scope"] == "group_sum_only"
        assert item["disclaimer"] == WATERSHED_GROUP_SUM_DISCLAIMER


def test_all_ten_preset_thumbnails_contract(api_client: Any) -> None:
    catalog = api_client.get("/api/presets").json()
    all_ids = [item["preset_id"] for item in catalog["single_bird_presets"]] + [
        item["preset_id"] for item in catalog["watershed_presets"]
    ]
    assert len(all_ids) == 10

    for preset_id in all_ids:
        resp = api_client.get(f"/api/presets/{preset_id}/thumbnail")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/jpeg")
        assert resp.headers["cache-control"] == THUMBNAIL_CACHE_CONTROL
        with Image.open(io.BytesIO(resp.content)) as im:
            assert im.format == "JPEG"
            assert max(im.size) <= THUMBNAIL_LONG_EDGE


def test_metrics_endpoint_contract(api_client: Any) -> None:
    resp = api_client.get("/api/metrics")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "public, max-age=300"

    body = resp.json()
    expected_hash = PDIConfig().config_hash()
    assert body["pdi_config_hash"] == expected_hash

    eval_report = body["evaluation_report"]
    assert eval_report["pdi_config_hash"] == expected_hash
    assert len(eval_report["model_comparison"]) == 16
    assert len(eval_report["oof_scatter_points"]) == 1738
    first_pt = eval_report["oof_scatter_points"][0]
    assert "actual_g" in first_pt
    assert "predicted_g" in first_pt
    assert "weight_g" not in first_pt

    audit_summary = body["dataset_audit_summary"]
    assert audit_summary["curation_funnel"]["clean_images"] == 1738
    assert audit_summary["split_audit"]["cross_fold_session_overlap"] == 0

    pdi_audit = body["pdi_calibration_audit"]
    assert pdi_audit["pdi_config_hash"] == expected_hash
    assert pdi_audit["dataset_extraction_summary"]["n_clean_images"] == 1738


@pytest.mark.parametrize(
    ("case_name", "filename", "content_type", "expected_exif", "expected_lens"),
    [
        ("exif_jpeg", "bird_exif.jpg", "image/jpeg", True, "ultrawide"),
        ("stripped_jpeg", "bird_no_exif.jpg", "image/jpeg", False, "wide"),
        ("rgba_png", "bird_rgba.png", "image/png", False, "wide"),
        ("webp", "bird.webp", "image/webp", False, "wide"),
    ],
)
def test_predict_upload_formats_exif_stripped_rgba_webp(
    api_client: Any,
    case_name: str,
    filename: str,
    content_type: str,
    expected_exif: bool,
    expected_lens: str,
) -> None:
    if case_name == "exif_jpeg":
        payload = make_synthetic_jpeg_bytes(
            size=(240, 180),
            orientation=6,
            focal_length_mm=1.65,
            zoom_multiple=0.6,
        )
    elif case_name == "stripped_jpeg":
        payload = make_synthetic_jpeg_bytes(
            size=(240, 180),
            orientation=None,
            focal_length_mm=None,
            zoom_multiple=None,
        )
    elif case_name == "rgba_png":
        payload = make_synthetic_png_bytes(size=(200, 150), mode="RGBA")
    else:
        payload = _make_synthetic_webp_bytes(size=(200, 150))

    resp = api_client.post(
        "/api/predict",
        files={"file": (filename, payload, content_type)},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["source_type"] == "upload"
    assert body["preset_id"] is None
    assert body["oof_evaluation"] is False
    assert body["oof_audit"] is None
    assert body["pdi_config_hash"] == PDIConfig().config_hash()
    assert body["predicted_weight_g"] > 0.0
    assert abs(body["predicted_weight_kg"] - (body["predicted_weight_g"] / 1000.0)) < 1e-4
    assert body["weight_band"] in ("<500g", "500-1000g", ">1000g")
    assert body["band_error_interval"]["n"] >= 1
    assert body["age_stage"]["valid_training_range_g"] == [116, 2093]

    optical = body["optical"]
    assert optical["exif_optical_present"] is expected_exif
    assert optical["override_applied"] is False
    assert optical["lens_class"] == expected_lens
    if not expected_exif:
        assert optical["focal_length_mm"] == 4.25
        assert optical["zoom_multiple"] == 1.0

    xray = body["xray"]
    assert xray["stage1_original"].startswith("data:image/jpeg;base64,")
    assert xray["stage2_clahe"].startswith("data:image/jpeg;base64,")
    assert xray["stage3_mask"].startswith("data:image/png;base64,")
    assert xray["stage4_overlay"].startswith("data:image/jpeg;base64,")
    assert xray["display_w"] >= 1 and xray["display_h"] >= 1

    assert list(body["features_summary"].keys()) == list(FEATURES_SUMMARY_KEYS)
    assert len(body["model_features"]) == 39
    assert body["timing"]["sla_target_ms"] == 500
    assert body["timing"]["total_ms"] >= 0.0


def test_predict_manual_optical_override(api_client: Any) -> None:
    stripped = make_synthetic_jpeg_bytes(
        size=(240, 180),
        orientation=None,
        focal_length_mm=None,
        zoom_multiple=None,
    )
    resp = api_client.post(
        "/api/predict",
        files={"file": ("stripped.jpg", stripped, "image/jpeg")},
        data={"focal_length_mm": "1.65", "zoom_multiple": "0.6"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["optical"]["override_applied"] is True
    assert body["optical"]["exif_optical_present"] is False
    assert body["optical"]["focal_length_mm"] == 1.65
    assert body["optical"]["zoom_multiple"] == 0.6
    assert body["optical"]["lens_class"] == "ultrawide"

    base_preset = api_client.post(
        "/api/predict",
        data={"preset_id": "sb_dual_lens_wide"},
    ).json()
    override_preset = api_client.post(
        "/api/predict",
        data={
            "preset_id": "sb_dual_lens_wide",
            "focal_length_mm": "1.65",
            "zoom_multiple": "0.6",
        },
    ).json()
    assert override_preset["optical"]["override_applied"] is True
    assert override_preset["optical"]["zoom_multiple"] == 0.6
    assert override_preset["oof_evaluation"] is True
    assert (
        override_preset["features_summary"]["area_optical_norm"]
        > base_preset["features_summary"]["area_optical_norm"]
    )


def test_watershed_presets_contract_and_hex_swatches(api_client: Any) -> None:
    cfg = PDIConfig()
    resp = api_client.post("/api/watershed-demo", data={"preset_id": "ws_417"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["source_type"] == "preset"
    assert body["preset_id"] == "ws_417"
    assert body["ground_truth_scope"] == "group_sum_only"
    assert body["disclaimer"] == WATERSHED_GROUP_SUM_DISCLAIMER
    assert body["bird_count"] == 2
    assert body["estimated_total_weight_g"] > 0.0
    assert body["estimated_mean_weight_g"] is not None

    ref = body["preset_reference"]
    assert ref is not None
    assert ref["folder_id"] == "417"
    assert ref["expected_bird_count"] == 2
    assert ref["group_sum_weight_g"] == 417.0
    assert ref["mean_bird_weight_reference_g"] == 208.5

    params = body["parameters"]
    assert params["h_rel"] == cfg.watershed_h_rel
    assert params["min_distance_rel"] == cfg.watershed_min_distance_rel
    assert params["min_area_ratio"] == cfg.watershed_min_area_ratio
    assert params["defaults"] == {
        "h_rel": cfg.watershed_h_rel,
        "min_distance_rel": cfg.watershed_min_distance_rel,
        "min_area_ratio": cfg.watershed_min_area_ratio,
    }
    assert params["bounds"] == {
        "h_rel": [WATERSHED_MIN_H_REL, WATERSHED_MAX_H_REL],
        "min_distance_rel": [WATERSHED_MIN_DISTANCE_REL, WATERSHED_MAX_DISTANCE_REL],
        "min_area_ratio": [WATERSHED_MIN_AREA_RATIO, WATERSHED_MAX_AREA_RATIO],
    }

    visuals = body["visuals"]
    assert visuals["stage1_original"].startswith("data:image/jpeg;base64,")
    assert visuals["stage2_union_mask"].startswith("data:image/png;base64,")
    assert visuals["stage3_distance_peaks"].startswith("data:image/jpeg;base64,")
    assert visuals["stage4_color_basins"].startswith("data:image/jpeg;base64,")

    assert len(body["instances"]) == 2
    for inst in body["instances"]:
        expected_hex = BASIN_PALETTE_HEX[(inst["instance_id"] - 1) % len(BASIN_PALETTE_HEX)]
        assert inst["color_hex"] == expected_hex
        assert inst["estimated_weight_g"] > 0.0
        assert 0.0 < inst["share_of_total_pct"] <= 100.0
        assert inst["age_stage"]["valid_training_range_g"] == [116, 2093]


def test_watershed_custom_upload_and_slider_overrides(api_client: Any) -> None:
    two_bird_jpg = _make_synthetic_two_bird_jpeg_bytes()
    resp = api_client.post(
        "/api/watershed-demo",
        files={"file": ("group.jpg", two_bird_jpg, "image/jpeg")},
        data={
            "h_rel": "0.15",
            "min_distance_rel": "0.60",
            "min_area_ratio": "0.05",
            "focal_length_mm": "4.25",
            "zoom_multiple": "1.0",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_type"] == "upload"
    assert body["preset_id"] is None
    assert body["preset_reference"] is None
    assert body["ground_truth_scope"] == "group_sum_only"
    assert body["disclaimer"] == WATERSHED_GROUP_SUM_DISCLAIMER
    assert body["parameters"]["h_rel"] == 0.15
    assert body["parameters"]["min_distance_rel"] == 0.60
    assert body["parameters"]["min_area_ratio"] == 0.05
    assert body["optical"]["override_applied"] is True
    assert body["bird_count"] >= 1
    assert body["estimated_mean_weight_g"] is not None


def test_watershed_zero_bird_uniform_dark_image(api_client: Any) -> None:
    dark_jpg = _make_uniform_dark_jpeg_bytes()
    resp = api_client.post(
        "/api/watershed-demo",
        files={"file": ("dark.jpg", dark_jpg, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["bird_count"] == 0
    assert body["instances"] == []
    assert body["estimated_total_weight_g"] == 0.0
    assert body["estimated_mean_weight_g"] is None
    assert body["ground_truth_scope"] == "group_sum_only"
    assert body["disclaimer"] == WATERSHED_GROUP_SUM_DISCLAIMER
    assert body["visuals"]["stage2_union_mask"].startswith("data:image/png;base64,")


def test_spa_fallback_serves_html_and_preserves_api_404(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text(
        "<!doctype html><html lang='pt-BR'><title>AvianScale CV</title></html>",
        encoding="utf-8",
    )
    (assets_dir / "app.js").write_text("console.log('avianscale');", encoding="utf-8")

    with _open_test_client(create_app(frontend_dist_dir=dist_dir)) as spa_client:
        root_resp = spa_client.get("/")
        assert root_resp.status_code == 200
        assert "text/html" in root_resp.headers["content-type"]
        assert "AvianScale CV" in root_resp.text

        sub_resp = spa_client.get("/estudio")
        assert sub_resp.status_code == 200
        assert "text/html" in sub_resp.headers["content-type"]

        asset_resp = spa_client.get("/assets/app.js")
        assert asset_resp.status_code == 200
        assert "avianscale" in asset_resp.text

        health_resp = spa_client.get("/api/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        missing_api = spa_client.get("/api/nonexistent")
        assert missing_api.status_code == 404
        assert "application/json" in missing_api.headers["content-type"]
        assert "detail" in missing_api.json()

        bare_api = spa_client.get("/api")
        assert bare_api.status_code == 404
        assert "application/json" in bare_api.headers["content-type"]
