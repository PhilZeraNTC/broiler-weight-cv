"""CLI, security, schema, and real-corpus integration tests for ``extract_features.py``.

Verifies D-07, D-08, D-09, D-10, ASVS V4 path/symlink containment (T-02-10),
4-worker concurrency capping (T-02-11), and PROIBIÇÃO 4 zero-path-leakage /
git-hygiene enforcement (T-02-12). Never mutates ``data/raw/`` and never prints
raw JPEG filenames or ``rel_path`` values.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from backend.app.config import (
    CLEAN_MANIFEST_PATH,
    FEATURES_CLEAN_PATH,
    PDI_CALIBRATION_AUDIT_PATH,
    RAW_DATASET_DIR,
)
from backend.app.pdi.config import PDIConfig
from backend.app.pdi.features import BANNED_FEATURE_COLUMNS, FEATURE_COLUMNS
from backend.scripts import extract_features

_FORBIDDEN_LEAK_MARKERS: tuple[str, ...] = (".jpg", "rel_path", "/home/", "data/raw")

_EXPECTED_CANDIDATE_FORMULAS: tuple[str, ...] = (
    "area_frac",
    "area_px",
    "area_px / (zoom_multiple ** 2)",
    "area_frac / (zoom_multiple ** 2)",
    "area_frac / (focal_length_mm ** 2)",
    "area_px / ((focal_length_mm * zoom_multiple) ** 2)",
)

_EXPECTED_SCOPES: tuple[str, ...] = (
    "all_mixed_lens_sessions",
    "same_scene_pairs",
    "bucket_only_pairs",
    "scale_or_litter_pairs",
)


def _expected_arrow_types() -> dict[str, pa.DataType]:
    mapping: dict[str, pa.DataType] = {
        "image_id": pa.string(),
        "fold": pa.int8(),
        "session_id": pa.string(),
        "weight_g": pa.int32(),
        "weight_band": pa.string(),
        "lens_class": pa.string(),
        "scene_type": pa.string(),
        "scene_is_bucket": pa.bool_(),
        "lcd_excluded_px": pa.int32(),
        "segmentation_fallback": pa.bool_(),
        "pdi_config_hash": pa.string(),
    }
    for col in FEATURE_COLUMNS:
        mapping[col] = pa.float64()
    return mapping


def _make_synthetic_bird_jpeg(
    *,
    width: int,
    height: int,
    bird_axes: tuple[int, int],
    bucket_scene: bool,
    focal_length_mm: float,
    zoom_multiple: float,
) -> bytes:
    """Synthesize a clean bucket or straw bird scene JPEG with EXIF focal + Xiaomi zoom."""
    rng = np.random.default_rng(1234 if bucket_scene else 5678)
    if bucket_scene:
        # Smooth bright white plastic bucket wall with slight central shading and yellow chick.
        img_bgr = np.full((height, width, 3), (232, 234, 236), dtype=np.uint8)
        cx, cy = width // 2, height // 2
        # Slight contact-shadow rim and golden-yellow chick body inside bucket.
        cv2.ellipse(
            img_bgr,
            (cx, cy),
            (bird_axes[0] + 6, bird_axes[1] + 6),
            0.0,
            0,
            360,
            (130, 138, 145),
            -1,
        )
        cv2.ellipse(img_bgr, (cx, cy), bird_axes, 0.0, 0, 360, (90, 200, 232), -1)
    else:
        # Textured golden-brown straw bedding with bright white/cream broiler silhouette.
        base = np.full((height, width, 3), (52, 92, 114), dtype=np.int16)
        noise = rng.integers(-28, 29, size=(height, width, 3), dtype=np.int16)
        img_bgr = np.clip(base + noise, 0, 255).astype(np.uint8)
        cx, cy = width // 2, height // 2
        cv2.ellipse(img_bgr, (cx, cy), bird_axes, 12.0, 0, 360, (228, 232, 236), -1)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb, mode="RGB")

    exif = Image.Exif()
    exif[274] = 1
    ifd = exif.get_ifd(0x8769)
    ifd[37386] = focal_length_mm
    ifd[39321] = json.dumps({"zoomMultiple": zoom_multiple})

    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92, subsampling=0, exif=exif.tobytes())
    return buf.getvalue()


def _create_synthetic_mixed_lens_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create a 4-image synthetic mixed-lens mini-dataset (bucket + straw) and manifest."""
    dataset_root = tmp_path / "synthetic_corpus"
    (dataset_root / "350").mkdir(parents=True)
    (dataset_root / "1450").mkdir(parents=True)

    specs: list[dict[str, Any]] = [
        # Intentionally non-alphabetical image_id order to verify 1:1 row order preservation.
        {
            "image_id": "ffff000000000004",
            "rel_path": "350/wide_bucket.jpg",
            "fold": 2,
            "session_id": "350__s0",
            "weight_g": 350,
            "weight_band": "<500g",
            "lens_class": "wide",
            "payload": _make_synthetic_bird_jpeg(
                width=640,
                height=288,
                bird_axes=(100, 62),
                bucket_scene=True,
                focal_length_mm=4.25,
                zoom_multiple=1.0,
            ),
        },
        {
            "image_id": "aaaa000000000001",
            "rel_path": "350/uw_bucket.jpg",
            "fold": 2,
            "session_id": "350__s0",
            "weight_g": 350,
            "weight_band": "<500g",
            "lens_class": "ultrawide",
            "payload": _make_synthetic_bird_jpeg(
                width=512,
                height=230,
                bird_axes=(60, 37),
                bucket_scene=True,
                focal_length_mm=1.65,
                zoom_multiple=0.60,
            ),
        },
        {
            "image_id": "cccc000000000003",
            "rel_path": "1450/wide_straw.jpg",
            "fold": 4,
            "session_id": "1450__s0",
            "weight_g": 1450,
            "weight_band": ">1000g",
            "lens_class": "wide",
            "payload": _make_synthetic_bird_jpeg(
                width=640,
                height=288,
                bird_axes=(120, 74),
                bucket_scene=False,
                focal_length_mm=4.25,
                zoom_multiple=1.0,
            ),
        },
        {
            "image_id": "bbbb000000000002",
            "rel_path": "1450/uw_straw.jpg",
            "fold": 4,
            "session_id": "1450__s0",
            "weight_g": 1450,
            "weight_band": ">1000g",
            "lens_class": "ultrawide",
            "payload": _make_synthetic_bird_jpeg(
                width=512,
                height=230,
                bird_axes=(72, 44),
                bucket_scene=False,
                focal_length_mm=1.65,
                zoom_multiple=0.60,
            ),
        },
    ]

    manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        target = dataset_root / spec["rel_path"]
        target.write_bytes(spec["payload"])
        manifest_rows.append(
            {
                "image_id": spec["image_id"],
                "rel_path": spec["rel_path"],
                "fold": np.int8(spec["fold"]),
                "session_id": spec["session_id"],
                "weight_g": np.int32(spec["weight_g"]),
                "weight_band": spec["weight_band"],
                "lens_class": spec["lens_class"],
            }
        )

    manifest_path = tmp_path / "mini_manifest.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, engine="pyarrow", index=False)
    return manifest_path, dataset_root


def test_parse_args_defaults_worker_clamping_and_invalid_workers_rejection() -> None:
    defaults = extract_features.parse_args([])
    assert defaults.manifest_path == CLEAN_MANIFEST_PATH
    assert defaults.dataset_root == RAW_DATASET_DIR
    assert defaults.output_path == FEATURES_CLEAN_PATH
    assert defaults.audit_path == PDI_CALIBRATION_AUDIT_PATH
    assert defaults.workers == extract_features.MAX_WORKERS_CAP == 4
    assert defaults.dry_run is False

    assert extract_features.parse_args(["--workers", "1"]).workers == 1
    assert extract_features.parse_args(["--workers", "3"]).workers == 3
    assert extract_features.parse_args(["--workers", "16"]).workers == 4

    for invalid in ("0", "-1", "-4"):
        with pytest.raises(ValueError, match="workers"):
            extract_features.parse_args(["--workers", invalid])


def test_schema_constants_have_exact_60_columns_and_exclude_banned_columns() -> None:
    assert len(extract_features.JOIN_AUDIT_COLUMNS) == 11
    assert len(FEATURE_COLUMNS) == 49
    assert len(extract_features.FEATURES_PARQUET_COLUMNS) == 60
    assert (
        extract_features.FEATURES_PARQUET_COLUMNS
        == extract_features.JOIN_AUDIT_COLUMNS + FEATURE_COLUMNS
    )
    assert len(set(extract_features.FEATURES_PARQUET_COLUMNS)) == 60
    assert BANNED_FEATURE_COLUMNS.isdisjoint(extract_features.FEATURES_PARQUET_COLUMNS)


@pytest.mark.parametrize(
    "case_name",
    ["parent_traversal", "absolute_path", "external_symlink", "internal_symlink", "symlink_dir"],
)
def test_asvs_v4_path_containment_rejects_traversal_and_symlinks(
    tmp_path: Path,
    case_name: str,
) -> None:
    dataset_root = tmp_path / "root"
    (dataset_root / "200").mkdir(parents=True)
    valid_bytes = _make_synthetic_bird_jpeg(
        width=320,
        height=160,
        bird_axes=(50, 30),
        bucket_scene=False,
        focal_length_mm=4.25,
        zoom_multiple=1.0,
    )
    inside_file = dataset_root / "200" / "real.jpg"
    inside_file.write_bytes(valid_bytes)

    outside_file = tmp_path / "secret_outside.jpg"
    outside_file.write_bytes(valid_bytes)

    if case_name == "parent_traversal":
        bad_rel_path = "../secret_outside.jpg"
    elif case_name == "absolute_path":
        bad_rel_path = str(outside_file)
    elif case_name == "external_symlink":
        (dataset_root / "200" / "escape.jpg").symlink_to(outside_file)
        bad_rel_path = "200/escape.jpg"
    elif case_name == "internal_symlink":
        (dataset_root / "200" / "alias.jpg").symlink_to(inside_file)
        bad_rel_path = "200/alias.jpg"
    else:
        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()
        (outside_dir / "nested.jpg").write_bytes(valid_bytes)
        (dataset_root / "linked_dir").symlink_to(outside_dir)
        bad_rel_path = "linked_dir/nested.jpg"

    manifest_path = tmp_path / "bad_manifest.parquet"
    pd.DataFrame(
        [
            {
                "image_id": "0123456789abcdef",
                "rel_path": bad_rel_path,
                "fold": np.int8(0),
                "session_id": "200__s0",
                "weight_g": np.int32(200),
                "weight_band": "<500g",
                "lens_class": "wide",
            }
        ]
    ).to_parquet(manifest_path, engine="pyarrow", index=False)

    out_parquet = tmp_path / "out" / "features.parquet"
    out_audit = tmp_path / "out" / "audit.json"

    with pytest.raises(ValueError, match="ASVS V4|containment|traversal|symlink"):
        extract_features.run_extract_features(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            output_path=out_parquet,
            audit_path=out_audit,
            workers=1,
            dry_run=False,
        )

    assert not out_parquet.exists()
    assert not out_audit.exists()


def test_cli_dry_run_writes_zero_files_and_emits_zero_jpg_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, dataset_root = _create_synthetic_mixed_lens_fixture(tmp_path)
    out_parquet = tmp_path / "dry_out" / "features_clean.parquet"
    out_audit = tmp_path / "dry_models" / "pdi_calibration_audit.json"

    rc = extract_features.main(
        [
            "--manifest-path",
            str(manifest_path),
            "--dataset-root",
            str(dataset_root),
            "--output-path",
            str(out_parquet),
            "--audit-path",
            str(out_audit),
            "--workers",
            "1",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert not out_parquet.exists()
    assert not out_parquet.parent.exists()
    assert not out_audit.exists()
    assert not out_audit.parent.exists()

    # PROIBIÇÃO 4: stdout logs strictly aggregate metrics, zero filenames/paths.
    lowered_out = captured.out.lower()
    for forbidden in _FORBIDDEN_LEAK_MARKERS:
        assert forbidden not in lowered_out
    assert "n_clean_images=4" in captured.out
    assert "dry_run=True" in captured.out


def test_cli_full_run_mixed_lens_schema_order_clamped_workers_and_byte_idempotency(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, dataset_root = _create_synthetic_mixed_lens_fixture(tmp_path)
    manifest_df = pd.read_parquet(manifest_path)

    out_parquet_1 = tmp_path / "run1" / "features_clean.parquet"
    out_audit_1 = tmp_path / "run1" / "pdi_calibration_audit.json"

    # Request 16 workers to verify clamping to MAX_WORKERS_CAP == 4 and real ProcessPoolExecutor.
    rc1 = extract_features.main(
        [
            "--manifest-path",
            str(manifest_path),
            "--dataset-root",
            str(dataset_root),
            "--output-path",
            str(out_parquet_1),
            "--audit-path",
            str(out_audit_1),
            "--workers",
            "16",
        ]
    )
    captured1 = capsys.readouterr()
    assert rc1 == 0
    for forbidden in _FORBIDDEN_LEAK_MARKERS:
        assert forbidden not in captured1.out.lower()
    assert "workers=4" in captured1.out

    assert out_parquet_1.is_file()
    assert out_audit_1.is_file()

    # Exact 60-column typed PyArrow schema check.
    schema = pq.read_schema(out_parquet_1)
    assert schema.names == list(extract_features.FEATURES_PARQUET_COLUMNS)
    assert len(schema.names) == 60
    for col_name, expected_type in _expected_arrow_types().items():
        assert schema.field(col_name).type == expected_type, (
            f"Arrow type mismatch on {col_name!r}: {schema.field(col_name).type} != {expected_type}"
        )

    feat_df = pd.read_parquet(out_parquet_1)
    assert len(feat_df) == 4
    # Exact 1:1 row order preserved against manifest_clean.parquet.
    assert feat_df["image_id"].tolist() == manifest_df["image_id"].tolist()
    assert feat_df["session_id"].tolist() == manifest_df["session_id"].tolist()
    assert set(feat_df["pdi_config_hash"].unique().tolist()) == {PDIConfig().config_hash()}
    assert BANNED_FEATURE_COLUMNS.isdisjoint(feat_df.columns)

    feat_matrix = feat_df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    assert np.all(np.isfinite(feat_matrix))

    # Audit JSON contract & PROIBIÇÃO 4 zero-path-leakage verification.
    raw_audit_1 = out_audit_1.read_text(encoding="utf-8")
    for forbidden in _FORBIDDEN_LEAK_MARKERS:
        assert forbidden not in raw_audit_1.lower()

    payload = json.loads(raw_audit_1)
    assert payload["schema_version"] == extract_features.CALIBRATION_AUDIT_SCHEMA_VERSION
    assert payload["pdi_config_hash"] == PDIConfig().config_hash()
    summary = payload["dataset_extraction_summary"]
    assert summary["n_clean_images"] == 4
    assert summary["workers"] == 4
    assert "batch_wall_clock_sec" not in summary

    cal = payload["paired_dual_lens_calibration"]
    assert cal["winning_formula"] == "area_px / (zoom_multiple ** 2)"
    assert (
        cal["rejected_double_count_formula"]
        == "area_px / ((focal_length_mm * zoom_multiple) ** 2)"
    )
    assert set(cal["scopes"].keys()) == set(_EXPECTED_SCOPES)
    assert cal["scopes"]["all_mixed_lens_sessions"]["n_pairs"] == 2
    expected_formulas = set(_EXPECTED_CANDIDATE_FORMULAS)
    for scope_name in _EXPECTED_SCOPES:
        assert set(cal["scopes"][scope_name]["formulas"].keys()) == expected_formulas

    # Byte-idempotency across consecutive runs (SHA-256 identical).
    out_parquet_2 = tmp_path / "run2" / "features_clean.parquet"
    out_audit_2 = tmp_path / "run2" / "pdi_calibration_audit.json"
    rc2 = extract_features.main(
        [
            "--manifest-path",
            str(manifest_path),
            "--dataset-root",
            str(dataset_root),
            "--output-path",
            str(out_parquet_2),
            "--audit-path",
            str(out_audit_2),
            "--workers",
            "4",
        ]
    )
    capsys.readouterr()
    assert rc2 == 0
    digest_1 = hashlib.sha256(out_audit_1.read_bytes()).hexdigest()
    digest_2 = hashlib.sha256(out_audit_2.read_bytes()).hexdigest()
    assert digest_1 == digest_2


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"bad": "116/116 (1).jpg"},
        {"bad": "116/IMAGE.JPG"},
        {"rel_path": "116/foo"},
        {"host": "/home/user/dataset"},
        {"dir": "data/raw/broiler"},
    ],
)
def test_write_calibration_audit_rejects_proibicao_4_leak_markers(
    tmp_path: Path,
    bad_payload: dict[str, Any],
) -> None:
    target = tmp_path / "rejected_audit.json"
    with pytest.raises(ValueError, match="PROIBIÇÃO 4"):
        extract_features.write_calibration_audit(bad_payload, target)
    assert not target.exists()
