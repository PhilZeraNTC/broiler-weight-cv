"""ASVS L1 security, upload guard, path containment & PROIBIÇÃO 1/2/3/4/5 API tests.

Verifies bounded upload DoS/MIME/decompression-bomb rejection, ASVS V5 parameter
validation, ASVS V4 non-symlink path containment + URL-encoded & form-body traversal
404 JSON responses under BOTH ``no_dist`` and active ``dist/index.html`` fixtures,
strict localhost-only CORS, sanitized HTTP 500 error responses, zero raw-filename
leakage, and AST/route compliance with PROIBIÇÕES 1, 2, 3, and 5.
"""

from __future__ import annotations

import ast
import io
import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from backend.app.api.main import LOCALHOST_CORS_ORIGINS, create_app
from backend.app.api.services.preset_service import build_preset_store
from backend.app.api.services.upload_guard import MAX_UPLOAD_BYTES
from backend.app.config import PRESETS_CATALOG_PATH, REPO_ROOT
from backend.app.modeling.artifacts import _FORBIDDEN_LEAK_SUBSTRINGS
from backend.tests.conftest import make_synthetic_jpeg_bytes


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
    """Module-scoped API-only TestClient isolated from ``frontend/dist/``."""
    no_dist = tmp_path_factory.mktemp("no_dist")
    with _open_test_client(create_app(frontend_dist_dir=no_dist)) as client:
        yield client


@pytest.fixture()
def spa_enabled_client(tmp_path: Path) -> Iterator[Any]:
    """Dedicated TestClient with an active ``dist/index.html`` + ``dist/assets/app.js``."""
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<!doctype html><title>SPA</title>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log(1);", encoding="utf-8")
    with _open_test_client(create_app(frontend_dist_dir=dist_dir)) as client:
        yield client


def _make_disallowed_format_bytes(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(100, 120, 140)).save(buf, format=fmt)
    return buf.getvalue()


def _make_over_40mp_png_bytes() -> bytes:
    """Create a tiny compressed 1-bit PNG whose header exceeds the 40 MP API ceiling (45.5 MP)."""
    buf = io.BytesIO()
    Image.new("1", (7000, 6500), 0).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("endpoint", ["/api/predict", "/api/watershed-demo"])
@pytest.mark.parametrize(
    ("case_name", "expected_status"),
    [
        ("oversize_15mb_plus_1", 413),
        ("empty_zero_bytes", 400),
        ("disallowed_bmp_spoofed_mime", 415),
        ("disallowed_gif", 415),
        ("corrupt_random_bytes", 400),
        ("truncated_jpeg_header", 400),
        ("decompression_bomb_over_40mp", 400),
    ],
)
def test_upload_guard_rejections(
    api_client: Any,
    endpoint: str,
    case_name: str,
    expected_status: int,
) -> None:
    if case_name == "oversize_15mb_plus_1":
        raw = b"\xff\xd8\xff\xe0" + (b"\x00" * (MAX_UPLOAD_BYTES - 3))
        file_tuple = ("huge.jpg", raw, "image/jpeg")
    elif case_name == "empty_zero_bytes":
        file_tuple = ("empty.jpg", b"", "image/jpeg")
    elif case_name == "disallowed_bmp_spoofed_mime":
        file_tuple = ("spoofed.jpg", _make_disallowed_format_bytes("BMP"), "image/jpeg")
    elif case_name == "disallowed_gif":
        file_tuple = ("anim.gif", _make_disallowed_format_bytes("GIF"), "image/gif")
    elif case_name == "corrupt_random_bytes":
        file_tuple = ("corrupt.jpg", b"not-an-image-payload-1234567890", "image/jpeg")
    elif case_name == "truncated_jpeg_header":
        valid = make_synthetic_jpeg_bytes(size=(64, 48))
        file_tuple = ("trunc.jpg", valid[:40], "image/jpeg")
    else:
        file_tuple = ("bomb.png", _make_over_40mp_png_bytes(), "image/png")

    resp = api_client.post(endpoint, files={"file": file_tuple})
    assert resp.status_code == expected_status
    assert "application/json" in resp.headers["content-type"]
    detail = resp.json()["detail"]
    assert isinstance(detail, str) and len(detail) > 0
    assert "Traceback" not in detail


@pytest.mark.parametrize("endpoint", ["/api/predict", "/api/watershed-demo"])
def test_mutual_exclusivity_file_xor_preset_id(api_client: Any, endpoint: str) -> None:
    valid_jpg = make_synthetic_jpeg_bytes(size=(64, 48))
    preset_id = "sb_intermediate_grower" if endpoint == "/api/predict" else "ws_417"

    both = api_client.post(
        endpoint,
        files={"file": ("bird.jpg", valid_jpg, "image/jpeg")},
        data={"preset_id": preset_id},
    )
    assert both.status_code == 422

    neither = api_client.post(endpoint)
    assert neither.status_code == 422

    blank_only = api_client.post(endpoint, data={"preset_id": "   "})
    assert blank_only.status_code == 422

    blank_with_file = api_client.post(
        endpoint,
        files={"file": ("bird.jpg", valid_jpg, "image/jpeg")},
        data={"preset_id": "   "},
    )
    assert blank_with_file.status_code == 422


@pytest.mark.parametrize(
    "form_data",
    [
        {"preset_id": "sb_intermediate_grower", "focal_length_mm": "0.49"},
        {"preset_id": "sb_intermediate_grower", "focal_length_mm": "200.1"},
        {"preset_id": "sb_intermediate_grower", "focal_length_mm": "nan"},
        {"preset_id": "sb_intermediate_grower", "focal_length_mm": "inf"},
        {"preset_id": "sb_intermediate_grower", "focal_length_mm": "-inf"},
        {"preset_id": "sb_intermediate_grower", "zoom_multiple": "0.09"},
        {"preset_id": "sb_intermediate_grower", "zoom_multiple": "20.1"},
        {"preset_id": "sb_intermediate_grower", "zoom_multiple": "nan"},
        {"preset_id": "sb_intermediate_grower", "zoom_multiple": "inf"},
    ],
)
def test_predict_optical_parameter_bounds_422(
    api_client: Any,
    form_data: dict[str, str],
) -> None:
    resp = api_client.post("/api/predict", data=form_data)
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "form_data",
    [
        {"preset_id": "ws_417", "h_rel": "0.04"},
        {"preset_id": "ws_417", "h_rel": "0.81"},
        {"preset_id": "ws_417", "h_rel": "nan"},
        {"preset_id": "ws_417", "h_rel": "inf"},
        {"preset_id": "ws_417", "min_distance_rel": "0.19"},
        {"preset_id": "ws_417", "min_distance_rel": "2.51"},
        {"preset_id": "ws_417", "min_distance_rel": "nan"},
        {"preset_id": "ws_417", "min_distance_rel": "-inf"},
        {"preset_id": "ws_417", "min_area_ratio": "0.005"},
        {"preset_id": "ws_417", "min_area_ratio": "0.46"},
        {"preset_id": "ws_417", "min_area_ratio": "nan"},
        {"preset_id": "ws_417", "min_area_ratio": "inf"},
        {"preset_id": "ws_417", "focal_length_mm": "0.2"},
        {"preset_id": "ws_417", "zoom_multiple": "99.0"},
    ],
)
def test_watershed_slider_and_optical_bounds_422(
    api_client: Any,
    form_data: dict[str, str],
) -> None:
    resp = api_client.post("/api/watershed-demo", data=form_data)
    assert resp.status_code == 422


def _assert_traversal_and_unknown_404_json(client: Any) -> None:
    get_cases = (
        "/api/presets/..%2F..%2Fetc%2Fpasswd/thumbnail",
        "/api/presets/..%2F..%2F..%2Fetc%2Fpasswd/thumbnail",
        "/api/presets/sb_unknown/thumbnail",
        "/api/nonexistent",
    )
    for url in get_cases:
        resp = client.get(url)
        assert resp.status_code == 404, f"Expected 404 for {url}, got {resp.status_code}"
        assert "application/json" in resp.headers["content-type"]
        body = resp.json()
        assert "detail" in body
        assert "etc/passwd" not in json.dumps(body)

    post_cases = (
        ("/api/predict", "../../etc/passwd"),
        ("/api/predict", "ws_417"),
        ("/api/watershed-demo", "../../etc/passwd"),
        ("/api/watershed-demo", "sb_young_chick"),
    )
    for endpoint, bad_id in post_cases:
        resp = client.post(endpoint, data={"preset_id": bad_id})
        assert resp.status_code == 404, f"Expected 404 for {endpoint} preset_id={bad_id!r}"
        assert "application/json" in resp.headers["content-type"]
        body = resp.json()
        assert "detail" in body
        assert "etc/passwd" not in json.dumps(body)


def test_traversal_and_cross_category_404_without_spa(api_client: Any) -> None:
    _assert_traversal_and_unknown_404_json(api_client)


def test_traversal_and_nonexistent_api_404_with_active_spa_dist(
    spa_enabled_client: Any,
) -> None:
    _assert_traversal_and_unknown_404_json(spa_enabled_client)


@pytest.mark.parametrize(
    "attack_mode",
    ["parent_traversal", "external_symlink", "internal_symlink"],
)
def test_preset_store_rejects_unsafe_manifest_paths(
    tmp_path: Path,
    attack_mode: str,
) -> None:
    dataset_root = tmp_path / "dataset_root"
    (dataset_root / "180").mkdir(parents=True)
    valid_bytes = make_synthetic_jpeg_bytes(size=(64, 48))
    inside_img = dataset_root / "180" / "inside.jpg"
    inside_img.write_bytes(valid_bytes)

    outside_img = tmp_path / "outside_secret.jpg"
    outside_img.write_bytes(valid_bytes)

    if attack_mode == "parent_traversal":
        bad_rel = "../outside_secret.jpg"
    elif attack_mode == "external_symlink":
        (dataset_root / "180" / "escape_link.jpg").symlink_to(outside_img)
        bad_rel = "180/escape_link.jpg"
    else:
        (dataset_root / "180" / "alias_link.jpg").symlink_to(inside_img)
        bad_rel = "180/alias_link.jpg"

    raw_catalog = json.loads(PRESETS_CATALOG_PATH.read_text(encoding="utf-8"))
    mini_catalog = {
        "schema_version": raw_catalog["schema_version"],
        "pdi_config_hash": raw_catalog["pdi_config_hash"],
        "winning_model_name": raw_catalog["winning_model_name"],
        "winning_ablation_condition": raw_catalog["winning_ablation_condition"],
        "single_bird_presets": [raw_catalog["single_bird_presets"][0]],
        "watershed_presets": [],
    }
    cat_path = tmp_path / "presets_catalog.json"
    cat_path.write_text(json.dumps(mini_catalog), encoding="utf-8")

    clean_manifest = tmp_path / "manifest_clean.parquet"
    pd.DataFrame(
        [{"image_id": mini_catalog["single_bird_presets"][0]["image_id"], "rel_path": bad_rel}]
    ).to_parquet(clean_manifest, index=False)

    quarantine_manifest = tmp_path / "manifest_quarantine.parquet"
    pd.DataFrame(columns=["image_id", "rel_path"]).to_parquet(quarantine_manifest, index=False)

    with pytest.raises(ValueError, match="ASVS V4"):
        build_preset_store(
            catalog_path=cat_path,
            clean_manifest_path=clean_manifest,
            quarantine_manifest_path=quarantine_manifest,
            dataset_root=dataset_root,
            expected_pdi_config_hash=raw_catalog["pdi_config_hash"],
        )


def test_cors_allows_only_localhost_origins(api_client: Any) -> None:
    for origin in LOCALHOST_CORS_ORIGINS:
        resp = api_client.get("/api/health", headers={"Origin": origin})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == origin

    for bad_origin in ("https://evil.example.com", "http://192.168.1.99:5173"):
        resp = api_client.get("/api/health", headers={"Origin": bad_origin})
        assert resp.status_code == 200
        allow_origin = resp.headers.get("access-control-allow-origin")
        assert allow_origin is None
        assert allow_origin != "*"


def test_sanitized_global_500_handler_strips_tracebacks_and_paths(tmp_path: Path) -> None:
    app = create_app(frontend_dist_dir=tmp_path / "no_dist")

    @app.get("/api/_test_internal_crash", include_in_schema=False)
    def _crash() -> dict[str, str]:
        raise RuntimeError("Leak /home/user/data/raw/180/secret.jpg rel_path traceback")

    with _open_test_client(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/_test_internal_crash")
        assert resp.status_code == 500
        assert "application/json" in resp.headers["content-type"]
        raw_text = resp.text.lower()
        assert "traceback" not in raw_text
        assert "runtimeerror" not in raw_text
        for token in _FORBIDDEN_LEAK_SUBSTRINGS:
            assert token not in raw_text


def test_proibicao4_zero_leak_scan_across_all_responses(api_client: Any) -> None:
    valid_jpg = make_synthetic_jpeg_bytes(size=(180, 120))
    responses = [
        api_client.get("/api/health"),
        api_client.get("/api/presets"),
        api_client.get("/api/metrics"),
        api_client.post("/api/predict", data={"preset_id": "sb_intermediate_grower"}),
        api_client.post(
            "/api/predict",
            files={"file": ("upload.jpg", valid_jpg, "image/jpeg")},
        ),
        api_client.post("/api/watershed-demo", data={"preset_id": "ws_417"}),
        api_client.post(
            "/api/watershed-demo",
            files={"file": ("group.jpg", valid_jpg, "image/jpeg")},
        ),
        api_client.get("/api/presets/sb_missing/thumbnail"),
        api_client.post("/api/predict", data={"preset_id": "../../etc/passwd"}),
        api_client.post(
            "/api/predict",
            data={"preset_id": "sb_intermediate_grower", "zoom_multiple": "999"},
        ),
        api_client.post("/api/predict", files={"file": ("bad.bin", b"not-a-jpg", "image/jpeg")}),
    ]

    for resp in responses:
        body_lower = resp.text.lower()
        headers_lower = json.dumps(dict(resp.headers)).lower()
        for token in _FORBIDDEN_LEAK_SUBSTRINGS:
            assert token not in body_lower, (
                f"PROIBIÇÃO 4 violation: forbidden token {token!r} in response body"
            )
            assert token not in headers_lower, (
                f"PROIBIÇÃO 4 violation: forbidden token {token!r} in response headers"
            )


def test_proibicoes_1_2_3_5_ast_and_route_audit(tmp_path: Path) -> None:
    api_dir = REPO_ROOT / "backend" / "app" / "api"
    py_files = sorted(api_dir.rglob("*.py"))
    assert len(py_files) >= 14

    banned_dl_ocr = {
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "keras",
        "ultralytics",
        "timm",
        "onnxruntime",
        "paddleocr",
        "easyocr",
    }
    banned_splitters = {
        "train_test_split",
        "KFold",
        "StratifiedKFold",
        "ShuffleSplit",
        "StratifiedShuffleSplit",
        "GroupShuffleSplit",
    }
    banned_cv_loaders = {"imread", "imdecode"}

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned_dl_ocr
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split(".")[0] not in banned_dl_ocr
                for alias in node.names:
                    assert alias.name not in banned_splitters
                    if node.module == "cv2":
                        assert alias.name not in banned_cv_loaders
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    assert func.id not in banned_splitters
                    assert func.id not in banned_cv_loaders
                elif isinstance(func, ast.Attribute):
                    assert func.attr not in banned_splitters
                    assert func.attr not in banned_cv_loaders

    app = create_app(frontend_dist_dir=tmp_path / "no_dist")
    openapi_paths = set(app.openapi()["paths"].keys())
    discovered_paths: set[str] = set(openapi_paths)
    stack: list[Any] = list(app.routes)
    while stack:
        route_obj = stack.pop()
        if hasattr(route_obj, "routes"):
            stack.extend(route_obj.routes)
        elif hasattr(route_obj, "router") and hasattr(route_obj.router, "routes"):
            stack.extend(route_obj.router.routes)
        path_val = getattr(route_obj, "path", None)
        if isinstance(path_val, str) and path_val.startswith("/api"):
            discovered_paths.add(path_val)

    expected_api_paths = {
        "/api/health",
        "/api/presets",
        "/api/presets/{preset_id}/thumbnail",
        "/api/predict",
        "/api/watershed-demo",
        "/api/metrics",
    }
    assert openapi_paths == expected_api_paths
    assert {p for p in discovered_paths if p.startswith("/api")} == expected_api_paths
    banned_v2_terms = (
        "tracking",
        "rtsp",
        "heatmap",
        "density",
        "sanitary",
        "barn",
        "stream",
        "video",
    )
    for route_path in discovered_paths:
        lower_path = route_path.lower()
        for term in banned_v2_terms:
            assert term not in lower_path, f"PROIBIÇÃO 5 violation: {route_path}"
