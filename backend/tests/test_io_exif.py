"""Unit, security, prohibition-compliance, and CLI tracer tests for Phase 1 Plan 01."""

from __future__ import annotations

import ast
import importlib
import math
import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from backend.app import config
from backend.app.pdi import io_exif
from backend.app.pdi.io_exif import (
    MAX_XIAOMI_TAG_BYTES,
    _parse_focal_length,
    _parse_xiaomi_zoom,
    load_oriented_rgb,
    read_exif_metadata,
)
from backend.scripts import build_manifest
from backend.tests.conftest import make_synthetic_jpeg_bytes, make_synthetic_png_bytes

# Expected red (raw top-left) and green (raw top-right) block probe coordinates (row, col)
# after ImageOps.exif_transpose() on a raw (W=40, H=20) image.
_EXPECTED_MARKER_PROBES: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {
    1: ((2, 2), (2, 37)),
    3: ((17, 37), (17, 2)),
    5: ((2, 2), (37, 2)),
    6: ((2, 17), (37, 17)),
    7: ((37, 17), (2, 17)),
    8: ((37, 2), (2, 2)),
}

_LOCKED_D02_RUNTIME_PACKAGES = {
    "opencv-python-headless",
    "scikit-image",
    "scipy",
    "pillow",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "joblib",
    "fastapi",
    "uvicorn",
    "pydantic",
    "python-multipart",
}
_LOCKED_D02_DEV_PACKAGES = {"pytest", "httpx", "ruff"}
_BANNED_PACKAGES = {
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
    "opencv-python",
    "opencv-contrib-python",
}


def _extract_pkg_name(req: str) -> str:
    match = re.match(r"^[A-Za-z0-9_.-]+", req.strip())
    assert match is not None, f"Could not parse package name from {req!r}"
    return match.group(0).lower().replace("_", "-")


def test_config_paths_are_repo_root_anchored_and_cwd_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_root = Path(__file__).resolve().parents[2]
    expected_raw = expected_root / "data" / "raw" / "Broiler weights of Cobb Avian"
    expected_processed = expected_root / "data" / "processed"
    expected_models = expected_root / "models"

    assert expected_root == config.REPO_ROOT
    assert (config.REPO_ROOT / "pyproject.toml").is_file()
    assert expected_root / "data" == config.DATA_DIR
    assert expected_raw == config.RAW_DATASET_DIR
    assert expected_processed == config.PROCESSED_DIR
    assert expected_models == config.MODELS_DIR
    assert expected_processed / "manifest_clean.parquet" == config.CLEAN_MANIFEST_PATH
    assert expected_processed / "manifest_quarantine.parquet" == config.QUARANTINE_MANIFEST_PATH
    assert expected_models / "dataset_audit_summary.json" == config.AUDIT_SUMMARY_PATH

    monkeypatch.chdir(tmp_path)
    reloaded = importlib.reload(config)
    assert expected_root == reloaded.REPO_ROOT
    assert expected_raw == reloaded.RAW_DATASET_DIR

    args = build_manifest.parse_args([])
    assert args.dataset_root == config.RAW_DATASET_DIR
    assert args.output_dir == config.PROCESSED_DIR
    assert args.summary_path == config.AUDIT_SUMMARY_PATH


@pytest.mark.parametrize("orientation", [1, 3, 5, 6, 7, 8])
def test_exif_metadata_and_oriented_rgb_path_bytes_parity(
    tmp_path: Path,
    orientation: int,
) -> None:
    raw_bytes = make_synthetic_jpeg_bytes(
        size=(40, 20),
        orientation=orientation,
        focal_length_mm=4.25,
        zoom_multiple=1.0,
        draw_corner_marker=True,
    )
    img_path = tmp_path / f"orient_{orientation}.jpg"
    img_path.write_bytes(raw_bytes)

    meta_path = read_exif_metadata(img_path)
    meta_bytes = read_exif_metadata(raw_bytes)
    loaded_path = load_oriented_rgb(img_path)
    loaded_bytes = load_oriented_rgb(raw_bytes)

    assert meta_path == meta_bytes
    assert loaded_path.metadata == meta_path
    assert loaded_bytes.metadata == meta_bytes
    assert np.array_equal(loaded_path.rgb, loaded_bytes.rgb)
    assert loaded_path.rgb.dtype == np.uint8

    expected_w, expected_h = (20, 40) if orientation in {5, 6, 7, 8} else (40, 20)
    assert meta_path.exif_orientation == orientation
    assert meta_path.raw_w == 40
    assert meta_path.raw_h == 20
    assert meta_path.oriented_w == expected_w
    assert meta_path.oriented_h == expected_h
    assert loaded_path.rgb.shape == (expected_h, expected_w, 3)
    assert meta_path.focal_length_mm == pytest.approx(4.25)
    assert meta_path.zoom_multiple == pytest.approx(1.0)
    assert meta_path.has_exif_optical is True

    # Confirm pixels were physically transposed (not just header dimensions swapped).
    (red_r, red_c), (green_r, green_c) = _EXPECTED_MARKER_PROBES[orientation]
    red_px = loaded_path.rgb[red_r, red_c]
    green_px = loaded_path.rgb[green_r, green_c]
    assert int(red_px[0]) > 180 and int(red_px[1]) < 60 and int(red_px[2]) < 60
    assert int(green_px[1]) > 180 and int(green_px[0]) < 60 and int(green_px[2]) < 60


def test_orientation_6_extracts_exif_before_transpose() -> None:
    raw_bytes = make_synthetic_jpeg_bytes(
        size=(40, 20),
        orientation=6,
        focal_length_mm=1.65,
        zoom_multiple=0.6,
        draw_corner_marker=True,
    )
    result = load_oriented_rgb(raw_bytes)

    assert result.metadata.exif_orientation == 6
    assert result.metadata.raw_w == 40
    assert result.metadata.raw_h == 20
    assert result.metadata.oriented_w == 20
    assert result.metadata.oriented_h == 40
    assert result.rgb.shape == (40, 20, 3)
    assert result.metadata.focal_length_mm == pytest.approx(1.65)
    assert result.metadata.zoom_multiple == pytest.approx(0.6)
    assert result.metadata.has_exif_optical is True


def test_stripped_exif_jpeg_and_rgba_png_degrade_gracefully(tmp_path: Path) -> None:
    stripped_jpeg = make_synthetic_jpeg_bytes(
        size=(32, 24),
        orientation=None,
        focal_length_mm=None,
        zoom_multiple=None,
        draw_corner_marker=False,
    )
    jpeg_path = tmp_path / "stripped.jpg"
    jpeg_path.write_bytes(stripped_jpeg)

    jpeg_loaded = load_oriented_rgb(jpeg_path)
    assert jpeg_loaded.metadata.exif_orientation == 1
    assert jpeg_loaded.metadata.focal_length_mm is None
    assert jpeg_loaded.metadata.zoom_multiple is None
    assert jpeg_loaded.metadata.has_exif_optical is False
    assert jpeg_loaded.rgb.shape == (24, 32, 3)
    assert jpeg_loaded.rgb.dtype == np.uint8

    rgba_png = make_synthetic_png_bytes(size=(24, 16), mode="RGBA")
    png_loaded = load_oriented_rgb(rgba_png)
    assert png_loaded.metadata.exif_orientation == 1
    assert png_loaded.metadata.focal_length_mm is None
    assert png_loaded.metadata.zoom_multiple is None
    assert png_loaded.metadata.has_exif_optical is False
    assert png_loaded.rgb.shape == (16, 24, 3)
    assert png_loaded.rgb.dtype == np.uint8


def test_parse_focal_length_handles_zero_denominator_and_invalid_inputs() -> None:
    assert _parse_focal_length(IFDRational(425, 100)) == pytest.approx(4.25)
    assert _parse_focal_length(IFDRational(0, 0)) is None
    assert _parse_focal_length(IFDRational(10, 0)) is None
    assert _parse_focal_length(-4.25) is None
    assert _parse_focal_length(0.0) is None
    assert _parse_focal_length(math.nan) is None
    assert _parse_focal_length(math.inf) is None
    assert _parse_focal_length(True) is None
    assert _parse_focal_length("invalid") is None
    assert _parse_focal_length(None) is None


def test_parse_xiaomi_zoom_enforces_security_bounds_and_null_termination() -> None:
    assert _parse_xiaomi_zoom(b'{"zoomMultiple": 1.65}\x00\x00') == pytest.approx(1.65)
    assert _parse_xiaomi_zoom('{"zoomMultiple": 0.1}\x00') == pytest.approx(0.1)
    assert _parse_xiaomi_zoom('{"zoomMultiple": 20.0}') == pytest.approx(20.0)

    # Invalid / malicious payloads must degrade cleanly to None (T-01-01).
    assert _parse_xiaomi_zoom(None) is None
    assert _parse_xiaomi_zoom(1234) is None
    assert _parse_xiaomi_zoom("{malformed json") is None
    assert _parse_xiaomi_zoom("[1, 2, 3]") is None
    assert _parse_xiaomi_zoom("1.0") is None
    assert _parse_xiaomi_zoom("{}") is None
    assert _parse_xiaomi_zoom('{"zoomMultiple": true}') is None
    assert _parse_xiaomi_zoom('{"zoomMultiple": false}') is None
    assert _parse_xiaomi_zoom('{"zoomMultiple": 0.05}') is None
    assert _parse_xiaomi_zoom('{"zoomMultiple": 25.0}') is None
    assert _parse_xiaomi_zoom('{"zoomMultiple": "nan"}') is None
    assert _parse_xiaomi_zoom('{"zoomMultiple": "inf"}') is None
    assert _parse_xiaomi_zoom(b'{"zoomMultiple": \x00}') is None
    assert _parse_xiaomi_zoom(b"\xff\xfe\x00") is None

    oversized_str = '{"zoomMultiple": 1.0, "pad": "' + ("a" * (MAX_XIAOMI_TAG_BYTES + 10)) + '"}'
    assert _parse_xiaomi_zoom(oversized_str) is None
    assert _parse_xiaomi_zoom(oversized_str.encode("utf-8")) is None


def test_malformed_xiaomi_tag_inside_jpeg_degrades_safely() -> None:
    bad_jpeg = make_synthetic_jpeg_bytes(
        size=(30, 20),
        orientation=1,
        focal_length_mm=4.25,
        zoom_multiple=None,
        raw_xiaomi_tag="{broken-json",
    )
    meta = read_exif_metadata(bad_jpeg)
    assert meta.focal_length_mm == pytest.approx(4.25)
    assert meta.zoom_multiple is None
    assert meta.has_exif_optical is False


def test_decompression_bomb_guard_remains_enabled() -> None:
    assert io_exif is not None
    assert Image.MAX_IMAGE_PIXELS is not None
    assert Image.MAX_IMAGE_PIXELS > 0


def test_ast_scan_confirms_zero_raw_opencv_image_loading_under_backend_app() -> None:
    backend_app_dir = config.REPO_ROOT / "backend" / "app"
    py_files = sorted(backend_app_dir.rglob("*.py"))
    assert py_files, "Expected at least one Python file under backend/app/"

    banned_calls = {"imread", "imdecode"}
    violations: list[str] = []
    for py_file in py_files:
        rel_file = py_file.relative_to(config.REPO_ROOT)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in banned_calls:
                    violations.append(f"{rel_file}:{node.lineno}:{func.attr}")
                elif isinstance(func, ast.Name) and func.id in banned_calls:
                    violations.append(f"{rel_file}:{node.lineno}:{func.id}")
            elif isinstance(node, ast.ImportFrom) and node.module == "cv2":
                for alias in node.names:
                    if alias.name in banned_calls:
                        violations.append(f"{rel_file}:{node.lineno}:import {alias.name}")

    assert not violations, f"PROIBIÇÃO 2 / D-07 violation — banned OpenCV loaders: {violations}"


def test_pyproject_and_uv_lock_comply_with_d02_and_prohibitions() -> None:
    pyproject_path = config.REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    uv_cfg = data["tool"]["uv"]
    assert uv_cfg["package"] is False
    assert uv_cfg["no-build"] is True
    assert uv_cfg["python-preference"] == "only-managed"

    runtime_pkgs = {_extract_pkg_name(dep) for dep in data["project"]["dependencies"]}
    dev_pkgs = {_extract_pkg_name(dep) for dep in data["dependency-groups"]["dev"]}
    assert runtime_pkgs == _LOCKED_D02_RUNTIME_PACKAGES
    assert dev_pkgs == _LOCKED_D02_DEV_PACKAGES
    assert not ((runtime_pkgs | dev_pkgs) & _BANNED_PACKAGES)

    lock_path = config.REPO_ROOT / "uv.lock"
    lock_data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    locked_names = {pkg["name"].lower().replace("_", "-") for pkg in lock_data.get("package", [])}
    banned_in_lock = locked_names & _BANNED_PACKAGES
    assert not banned_in_lock, f"Banned packages resolved in uv.lock: {banned_in_lock}"


def test_build_manifest_cli_dry_run_writes_zero_files(
    synthetic_dataset_tree: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "out_dry"
    summary_path = tmp_path / "summary_dry.json"

    rc = build_manifest.main(
        [
            "--dataset-root",
            str(synthetic_dataset_tree),
            "--output-dir",
            str(out_dir),
            "--summary-path",
            str(summary_path),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert not out_dir.exists()
    assert not summary_path.exists()
    # PROIBIÇÃO 4: stdout must never dump individual JPEG filenames.
    assert ".jpg" not in captured.out.lower()


def test_build_manifest_cli_writes_parquet_and_rejects_symlink_escapes(
    synthetic_dataset_tree: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "out_run"
    summary_path = tmp_path / "summary_run.json"

    rc = build_manifest.main(
        [
            "--dataset-root",
            str(synthetic_dataset_tree),
            "--output-dir",
            str(out_dir),
            "--summary-path",
            str(summary_path),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert ".jpg" not in captured.out.lower()

    clean_parquet = out_dir / "manifest_clean.parquet"
    assert clean_parquet.is_file()
    df = pd.read_parquet(clean_parquet)
    rel_paths = set(df["rel_path"].astype(str))

    # Symlink escape outside dataset_root must never be ingested (T-01-03).
    assert not any("escaped_link" in p or "outside_secret" in p for p in rel_paths)

    # Clean rows from representative clean folders must be present
    # (compatible with both Plan 01-01 tracer and Plan 01-03 full quarantine pipeline).
    assert any(p.startswith("116/") for p in rel_paths)
    assert any(p.startswith("250/") for p in rel_paths)
    assert any(p.startswith("1400/") for p in rel_paths)
